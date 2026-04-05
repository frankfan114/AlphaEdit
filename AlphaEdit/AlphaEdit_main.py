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
from .analysis_tools import apply_stability_projection_
from .delta_clipping import clip_delta_frobenius, clip_delta_spectral
from .diagnostics import (
    compute_conflict_sub,
    compute_block_ratio,
    protected_rank_from_projector,
)
# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

def apply_AlphaEdit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    cache_template: Optional[str] = None,
    cache_c = None,
    P = None,
    weight_reference = None,
    enable_diagnostics: bool = False,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    edit_diagnostics = {"layers": {}, "aggregate": {}}

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"MEMIT request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        # Retrieve k/v pair if already stored in cache
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
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )

            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)

    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
        upd_matrix = torch.linalg.solve(
                P[i,:,:].cuda() @ (layer_ks @ layer_ks.T + cache_c[i,:,:].cuda()) + hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device="cuda"), P[i,:,:].cuda() @ layer_ks @ resid.T
        )
        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        orig_norm = torch.linalg.norm(weights[weight_name]).item()

        # ── Delta-level clipping (optional, controlled by hparams) ──────────
        clip_info = {"clipped": False, "scale": 1.0}
        if getattr(hparams, "delta_spectral_tau", 0.0) > 0.0:
            clip_info = clip_delta_spectral(upd_matrix, hparams.delta_spectral_tau)
            upd_matrix = clip_info["clipped_delta"].to(upd_matrix.device, dtype=upd_matrix.dtype)
            if clip_info["clipped"]:
                print(f"  Spectral clip: scale={clip_info['scale']:.4f}, "
                      f"pre={clip_info['pre_clip_spectral']:.4f}, post={clip_info['post_clip_spectral']:.4f}")
        else:
            tau_f = getattr(hparams, "delta_fro_tau", 0.0)
            if tau_f <= 0.0 and getattr(hparams, "delta_fro_ratio", 0.0) > 0.0 and weight_reference is not None:
                tau_f = hparams.delta_fro_ratio * torch.linalg.norm(
                    weight_reference[weight_name].float(), ord="fro"
                ).item()
            if tau_f > 0.0:
                clip_info = clip_delta_frobenius(upd_matrix, tau_f)
                upd_matrix = clip_info["clipped_delta"].to(upd_matrix.device, dtype=upd_matrix.dtype)
                if clip_info["clipped"]:
                    print(f"  Frobenius clip: scale={clip_info['scale']:.4f}, "
                          f"pre={clip_info['pre_clip_norm']:.4f}, post={clip_info['post_clip_norm']:.4f}")
        # ────────────────────────────────────────────────────────────────────

        # ── Exact ΔW diagnostics (computed after clipping, before applying) ─
        upd_float = upd_matrix.detach().float().cpu()
        delta_fro      = torch.linalg.norm(upd_float, ord="fro").item()
        delta_spectral = torch.linalg.norm(upd_float, ord=2).item()
        delta_stable_rank = (delta_fro ** 2) / (delta_spectral ** 2 + 1e-12)
        upd_norm = delta_fro
        print("orig norm", orig_norm)
        print(f"upd norm  ||ΔW||_F={delta_fro:.4f}  ||ΔW||_2={delta_spectral:.4f}  stable_rank(ΔW)={delta_stable_rank:.2f}")
        # ────────────────────────────────────────────────────────────────────

        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix
        projection_info = {
            "applied": False,
            "pre_projection_delta_norm": None,
            "post_projection_delta_norm": None,
            "spectral_upper_bound": None,
            "condition_lower_bound": None,
        }
        if weight_reference is not None:
            projection_info = apply_stability_projection_(
                weights[weight_name], weight_reference[weight_name], hparams
            )
            if projection_info.get("applied"):
                print(
                    f"Applied numerical stability projection to {weight_name}: {projection_info}"
                )
        layer_diag = {
            "weight_name": weight_name,
            "orig_norm": orig_norm,
            "update_norm": upd_norm,
            "update_to_orig_norm_ratio": float(upd_norm / (orig_norm + 1e-12)),
            "delta_fro": delta_fro,
            "delta_spectral": delta_spectral,
            "delta_stable_rank": delta_stable_rank,
            "pre_projection_delta_norm": projection_info.get("pre_projection_delta_norm"),
            "post_projection_delta_norm": projection_info.get("post_projection_delta_norm"),
            "projection_applied": projection_info.get("applied", False),
            "spectral_upper_bound": projection_info.get("spectral_upper_bound"),
            "condition_lower_bound": projection_info.get("condition_lower_bound"),
        }

        # ── Mechanism-analysis diagnostics (enabled via enable_diagnostics) ──
        # These quantities are computed using the SAME layer_ks / resid / cache_c
        # that drove the actual update.  They add one extra linear solve per layer
        # but do NOT change any model weights or accumulators.
        if enable_diagnostics and P is not None:
            P_i = P[i, :, :].cuda()

            # conflict_sub(t) = ‖P k_t‖² / ‖k_t‖²
            # Measures how much of the edit key lives in the protected null-space.
            # Uses the FIXED initial projector P (AlphaEdit never updates P).
            conflict_info = compute_conflict_sub(layer_ks, P_i)

            # block_ratio(t) = ‖Δ_block‖_F / ‖Δ_raw‖_F
            # Δ_raw is the MEMIT-style update (no null-space constraint).
            # Δ_block = (I − P) @ Δ_raw  (LEFT mult on key dimension).
            block_info = compute_block_ratio(
                layer_ks,
                resid,
                cache_c[i, :, :],
                P_i,
                hparams.L2,
            )

            p_rank = protected_rank_from_projector(P_i)
            layer_diag.update({
                # conflict_sub(t)
                "conflict_sub":    conflict_info["conflict_sub"],
                "k_t_norm":        conflict_info["k_t_norm"],
                # block_ratio(t)
                "block_ratio":     block_info["block_ratio"],
                "delta_raw_fro":   block_info["delta_raw_fro"],
                "delta_allow_fro": block_info["delta_allow_fro"],
                "delta_block_fro": block_info["delta_block_fro"],
                "diag_solve_failed": block_info["solve_failed"],
                # subspace structure
                "protected_rank":  p_rank,
                "total_dim":       P_i.shape[0],
                "free_rank_proxy": P_i.shape[0] - p_rank,
            })

        edit_diagnostics["layers"][str(layer)] = layer_diag
        # ─────────────────────────────────────────────────────────────────────

        # Clear GPU memory
        #del U,S,cov
        for x in [layer_ks, cur_zs, targets, upd_matrix]:
            x.cpu()
            del x
        torch.cuda.empty_cache()
    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T

    layer_metrics = list(edit_diagnostics["layers"].values())
    if len(layer_metrics) > 0:
        agg = {
            "mean_update_norm": float(
                sum(metric["update_norm"] for metric in layer_metrics) / len(layer_metrics)
            ),
            "mean_update_to_orig_norm_ratio": float(
                sum(metric["update_to_orig_norm_ratio"] for metric in layer_metrics)
                / len(layer_metrics)
            ),
            "mean_pre_projection_delta_norm": float(
                sum(
                    metric["pre_projection_delta_norm"]
                    for metric in layer_metrics
                    if metric["pre_projection_delta_norm"] is not None
                )
                / max(
                    sum(
                        1
                        for metric in layer_metrics
                        if metric["pre_projection_delta_norm"] is not None
                    ),
                    1,
                )
            ),
            "mean_post_projection_delta_norm": float(
                sum(
                    metric["post_projection_delta_norm"]
                    for metric in layer_metrics
                    if metric["post_projection_delta_norm"] is not None
                )
                / max(
                    sum(
                        1
                        for metric in layer_metrics
                        if metric["post_projection_delta_norm"] is not None
                    ),
                    1,
                )
            ),
            "projection_applied_layers": int(
                sum(1 for metric in layer_metrics if metric["projection_applied"])
            ),
            "mean_delta_spectral": float(
                sum(metric["delta_spectral"] for metric in layer_metrics) / len(layer_metrics)
            ),
        }

        # Aggregate diagnostics fields (only present when enable_diagnostics=True)
        if enable_diagnostics and all("conflict_sub" in m for m in layer_metrics):
            valid_cs = [m["conflict_sub"] for m in layer_metrics if m.get("conflict_sub") is not None]
            valid_br = [m["block_ratio"] for m in layer_metrics
                        if m.get("block_ratio") is not None and not (m["block_ratio"] != m["block_ratio"])]
            valid_raw = [m["delta_raw_fro"] for m in layer_metrics
                         if m.get("delta_raw_fro") is not None and not (m["delta_raw_fro"] != m["delta_raw_fro"])]
            valid_blk = [m["delta_block_fro"] for m in layer_metrics
                         if m.get("delta_block_fro") is not None and not (m["delta_block_fro"] != m["delta_block_fro"])]
            agg.update({
                "mean_conflict_sub":    float(sum(valid_cs) / len(valid_cs))  if valid_cs  else None,
                "mean_block_ratio":     float(sum(valid_br) / len(valid_br))  if valid_br  else None,
                "mean_delta_raw_fro":   float(sum(valid_raw) / len(valid_raw)) if valid_raw else None,
                "mean_delta_block_fro": float(sum(valid_blk) / len(valid_blk)) if valid_blk else None,
            })

        edit_diagnostics["aggregate"] = agg

    print(f"Deltas successfully computed for {list(weights.keys())}")
    return model, cache_c, edit_diagnostics


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
