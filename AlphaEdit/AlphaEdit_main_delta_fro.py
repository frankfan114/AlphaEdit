"""
AlphaEdit with Delta-level Frobenius-norm clipping.

Before applying each layer update:
    ΔW' = ΔW · min(1, τ_F / ‖ΔW‖_F)
    W'  = W + ΔW'

τ_F is controlled by --delta_fro_tau (absolute) or --delta_fro_ratio (relative to ‖W₀‖_F).
If both are set, the absolute tau takes priority.
"""

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .AlphaEdit_hparams import AlphaEditHyperParams
from .delta_clipping import clip_delta_frobenius
from .AlphaEdit_main import get_cov, get_context_templates, upd_matrix_match_shape

CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def apply_AlphaEdit_delta_fro_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    cache_template: Optional[str] = None,
    cache_c=None,
    P=None,
    weight_reference=None,
) -> Dict[str, Tuple[torch.Tensor]]:

    edit_diagnostics = {"layers": {}, "aggregate": {}}

    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"MEMIT request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if cache_fname is not None and cache_fname.exists():
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        if not data_loaded:
            cur_z = compute_z(model, tok, request, hparams, z_layer, context_templates)
            z_list.append(cur_z)
            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(cache_fname, **{"v_star": cur_z.detach().cpu().numpy()})
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)

    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        cur_zs = get_module_input_output_at_words(
            model, tok, z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = layer_ks.size(1) // targets.size(1)
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(hparams.layers) - i)
        upd_matrix = torch.linalg.solve(
            P[i, :, :].cuda() @ (layer_ks @ layer_ks.T + cache_c[i, :, :].cuda())
            + hparams.L2 * torch.eye(layer_ks.shape[0], dtype=torch.float, device="cuda"),
            P[i, :, :].cuda() @ layer_ks @ resid.T,
        )
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        orig_norm = torch.linalg.norm(weights[weight_name]).item()
        upd_norm_pre = torch.linalg.norm(upd_matrix).item()
        print("orig norm", orig_norm)
        print("upd norm (pre-clip)", upd_norm_pre)

        # ── Delta-level Frobenius clipping ──────────────────────────────────
        tau_f = hparams.delta_fro_tau
        if tau_f <= 0.0 and hparams.delta_fro_ratio > 0.0 and weight_reference is not None:
            tau_f = hparams.delta_fro_ratio * torch.linalg.norm(
                weight_reference[weight_name].float(), ord="fro"
            ).item()

        clip_info = {"clipped": False, "scale": 1.0,
                     "pre_clip_norm": upd_norm_pre, "post_clip_norm": upd_norm_pre}
        if tau_f > 0.0:
            clip_info = clip_delta_frobenius(upd_matrix, tau_f)
            upd_matrix = clip_info["clipped_delta"].to(
                upd_matrix.device, dtype=upd_matrix.dtype
            )
            if clip_info["clipped"]:
                print(f"  Frobenius clip applied: scale={clip_info['scale']:.4f}, "
                      f"pre={clip_info['pre_clip_norm']:.4f}, post={clip_info['post_clip_norm']:.4f}")
        # ────────────────────────────────────────────────────────────────────

        upd_norm_post = torch.linalg.norm(upd_matrix).item()
        delta_spectral = torch.linalg.norm(upd_matrix.float(), ord=2).item()
        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix

        edit_diagnostics["layers"][str(layer)] = {
            "weight_name": weight_name,
            "orig_norm": orig_norm,
            "update_norm": upd_norm_post,
            "update_to_orig_norm_ratio": float(upd_norm_post / (orig_norm + 1e-12)),
            "pre_clip_delta_norm": clip_info["pre_clip_norm"],
            "post_clip_delta_norm": clip_info["post_clip_norm"],
            "delta_spectral": delta_spectral,
            "clip_scale": clip_info["scale"],
            "clip_applied": clip_info["clipped"],
            # keep same keys as baseline for compatibility with plot script
            "pre_projection_delta_norm": clip_info["pre_clip_norm"],
            "post_projection_delta_norm": clip_info["post_clip_norm"],
            "projection_applied": clip_info["clipped"],
            "spectral_upper_bound": None,
            "condition_lower_bound": None,
        }

        for x in [layer_ks, cur_zs, targets, upd_matrix]:
            x.cpu()
            del x
        torch.cuda.empty_cache()

    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cache_c[i, :, :] += layer_ks.cpu() @ layer_ks.cpu().T

    layer_metrics = list(edit_diagnostics["layers"].values())
    if len(layer_metrics) > 0:
        edit_diagnostics["aggregate"] = {
            "mean_update_norm": float(
                sum(m["update_norm"] for m in layer_metrics) / len(layer_metrics)
            ),
            "mean_update_to_orig_norm_ratio": float(
                sum(m["update_to_orig_norm_ratio"] for m in layer_metrics) / len(layer_metrics)
            ),
            "mean_pre_projection_delta_norm": float(
                sum(m["pre_projection_delta_norm"] for m in layer_metrics
                    if m["pre_projection_delta_norm"] is not None)
                / max(sum(1 for m in layer_metrics if m["pre_projection_delta_norm"] is not None), 1)
            ),
            "mean_post_projection_delta_norm": float(
                sum(m["post_projection_delta_norm"] for m in layer_metrics
                    if m["post_projection_delta_norm"] is not None)
                / max(sum(1 for m in layer_metrics if m["post_projection_delta_norm"] is not None), 1)
            ),
            "projection_applied_layers": int(
                sum(1 for m in layer_metrics if m["projection_applied"])
            ),
            "mean_delta_spectral": float(
                sum(m["delta_spectral"] for m in layer_metrics) / len(layer_metrics)
            ),
        }

    print(f"Deltas successfully computed for {list(weights.keys())}")
    return model, cache_c, edit_diagnostics
