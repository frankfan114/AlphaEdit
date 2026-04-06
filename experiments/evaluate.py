import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import csv
import json
import shutil
from datetime import datetime
from itertools import islice
from time import time
from typing import Tuple, Union
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    MQUAKEDataset,
    get_tfidf_vectorizer,
    KnownsDataset,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from experiments.py.eval_utils_mquake import compute_rewrite_quality_mquake
from memit import MEMITHyperParams
from memit.compute_z import get_module_input_output_at_words, compute_z
from memit.memit_main import apply_memit_to_model, get_context_templates
from memit.memit_seq_main import apply_memit_seq_to_model
from memit.memit_rect_main import apply_memit_rect_to_model
from AlphaEdit import AlphaEditHyperParams
from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model, get_cov
from AlphaEdit.analysis_tools import (
    aggregate_probe_metrics,
    clone_rewrite_weights,
    compute_conflict_metrics_for_record,
    select_history_probe,
    select_history_probe_oldest,
    select_history_probe_recent,
    select_history_probe_stratified,
    summarize_rewrite_weight_metrics,
)
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util.globals import *
from nse import NSEHyperParams
from nse.nse_main import apply_nse_to_model
from glue_eval.glue_eval import GLUEEval
ALG_DICT = {
    "AlphaEdit": (AlphaEditHyperParams, apply_AlphaEdit_to_model),
    "MEMIT_seq": (MEMITHyperParams, apply_memit_seq_to_model),
    "MEMIT_prune": (MEMITHyperParams, apply_memit_to_model),
    "MEMIT_rect": (MEMITHyperParams, apply_memit_rect_to_model),
    "NSE": (NSEHyperParams, apply_nse_to_model),
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
    "mquake": (MQUAKEDataset, compute_rewrite_quality_mquake),
}


def evaluate_record_list(model, tok, records, ds_eval_method):
    case_metrics = []
    for record in records:
        post_metrics = ds_eval_method(model, tok, record, None, None)
        case_metrics.append(
            {
                "case_id": record["case_id"],
                "post": post_metrics,
                "conflict_metrics": compute_conflict_metrics_for_record(model, tok, record),
            }
        )

    return {
        "case_ids": [record["case_id"] for record in records],
        "summary": aggregate_probe_metrics(case_metrics),
        "cases": case_metrics,
    }


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
    args = None,
    enable_diagnostics: bool = False,
    diagnostics_save_path: str = "",
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    run_dir = None
    if continue_from_run is not None:
        run_dir = RESULTS_DIR / dir_name / continue_from_run
    if continue_from_run is None or not run_dir.exists():
        continue_from_run = None
    if continue_from_run is None:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")
    if "MEMIT" in alg_name:
    # Get run hyperparameters
        params_path = (
            run_dir / "0_params.json"
            if continue_from_run is not None
            else HPARAMS_DIR / "MEMIT" / hparams_fname
        )
    else:
        params_path = (
            run_dir / "0_params.json"
            if continue_from_run is not None
            else HPARAMS_DIR / alg_name / hparams_fname
        )
    hparams = params_class.from_json(params_path)
    if alg_name == "AlphaEdit" and args is not None:
        hparams.numerical_stability = args.numerical_stability
        hparams.stability_spectral_multiplier = args.stability_spectral_multiplier
        hparams.stability_condition_number = args.stability_condition_number
        hparams.stability_fro_drift_ratio = args.stability_fro_drift_ratio
        hparams.knowledge_conflict = args.knowledge_conflict
        hparams.conflict_loss_weight = args.conflict_loss_weight
        hparams.conflict_margin = args.conflict_margin
        hparams.analysis_top_singular_values = args.analysis_top_singular_values
        hparams.delta_fro_tau = args.delta_fro_tau
        hparams.delta_fro_ratio = args.delta_fro_ratio
        hparams.delta_spectral_tau = args.delta_spectral_tau
    # Override diagnostics flags from args (works for all alg_names, no-op if not AlphaEdit)
    if args is not None:
        if hasattr(args, "enable_diagnostics"):
            enable_diagnostics = args.enable_diagnostics
        if hasattr(args, "diagnostics_save_path") and args.diagnostics_save_path:
            diagnostics_save_path = args.diagnostics_save_path
    
    
    # if not (run_dir / "0_params.json").exists():
    #     shutil.copyfile(params_path, run_dir / "0_params.json")
    if not (run_dir / "0_params.json").exists():
        # 1. 读取原始 hparams json
        with open(params_path, "r") as f:
            hparams_dict = json.load(f)

        # 2. 组织运行参数（bench / dataset / edits 等）
        run_args = {
            "alg_name": alg_name,
            "model_name": model_name,
            "ds_name": ds_name,
            "dataset_size_limit": dataset_size_limit,
            "num_edits": num_edits,
            "use_cache": use_cache,
            "skip_generation_tests": skip_generation_tests,
            "generation_test_interval": generation_test_interval,
            "conserve_memory": conserve_memory,
            "continue_from_run": continue_from_run,
            "downstream_eval_steps": args.downstream_eval_steps if args is not None else None,
            "analysis_interval": args.analysis_interval if args is not None else None,
            "history_probe_size": args.history_probe_size if args is not None else None,
            "numerical_stability": args.numerical_stability if args is not None else None,
            "stability_spectral_multiplier": args.stability_spectral_multiplier if args is not None else None,
            "stability_condition_number": args.stability_condition_number if args is not None else None,
            "stability_fro_drift_ratio": args.stability_fro_drift_ratio if args is not None else None,
            "knowledge_conflict": args.knowledge_conflict if args is not None else None,
            "conflict_loss_weight": args.conflict_loss_weight if args is not None else None,
            "conflict_margin": args.conflict_margin if args is not None else None,
            "analysis_top_singular_values": args.analysis_top_singular_values if args is not None else None,
            "enable_diagnostics": args.enable_diagnostics if args is not None and hasattr(args, "enable_diagnostics") else False,
            "diagnostics_save_path": args.diagnostics_save_path if args is not None and hasattr(args, "diagnostics_save_path") else "",
        }

        # 3. 合并成一个实验配置快照
        full_params = {
            "hparams": hparams_dict,
            "run_args": run_args,
        }

        # 4. 写入 run_dir/0_params.json
        with open(run_dir / "0_params.json", "w") as f:
            json.dump(full_params, f, indent=2)

    
    print(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
    if type(model_name) is str:
        print("Instantiating model")
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    if num_edits > 1:
        assert ds_name != "cf", f"{ds_name} does not support multiple edits"

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)
    # Get cache templates
    cache_template = None
    if use_cache:
        if any(alg in alg_name for alg in ["MEMIT","AlphaEdit", "MEMIT_seq", "MEMIT_prune", "MEMIT_rect"]):
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_MEMIT"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
            )
        else:
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_{alg_name}"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
            )
        print(f"Will load cache from {cache_template}")
    if alg_name == "NSE":
        cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_{alg_name}"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        for record in ds:
            # Retrieve k/v pair if already stored in cache
            cache_fname = (
                Path(
                    str(cache_template).format(
                        hparams.layers[-1], hparams.clamp_norm_factor, record["case_id"]
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
                continue
            # Compute k/v pair if not loaded from cache
            if not data_loaded:
                context_templates = get_context_templates(model, tok)
                cur_z = compute_z(
                    model,
                    tok,
                    {"case_id": record["case_id"], **record["requested_rewrite"]},
                    hparams,
                    hparams.layers[-1],
                    context_templates,
                )
                if cache_fname is not None:
                    cache_fname.parent.mkdir(exist_ok=True, parents=True)
                    np.savez(
                        cache_fname,
                        **{
                            "v_star": cur_z.detach().cpu().numpy(),
                        },
                    )
                    print(f"Cached k/v pair at {cache_fname}")
    if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "MEMIT_prune", "NSE"]):
        # Iterate through dataset
        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        if hparams.model_name == "gpt2-xl":
            cache_c = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
            if alg_name == "AlphaEdit":
                P = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
        elif hparams.model_name in ["EleutherAI_gpt-j-6B","Llama3-8B","phi-1.5"]:
            cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
            if alg_name == "AlphaEdit":
                P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
        del W_out
    if alg_name == "AlphaEdit":
        for i, layer in enumerate(hparams.layers):
            P[i,:,:] = get_project(model,tok,layer,hparams)
        torch.save(P, run_dir / "null_space_project.pt")
        weight_reference = clone_rewrite_weights(model, hparams)
    else:
        weight_reference = None

    # ── Diagnostics JSONL setup ──────────────────────────────────────────────
    # When enable_diagnostics=True (and alg_name=="AlphaEdit"), we write one
    # JSONL record per sequential edit step containing:
    #   • conflict_sub(t) and block_ratio(t) per layer
    #   • update norms
    #   • eval metrics (ES/locality) at analysis_interval steps
    # When enable_diagnostics=False the file handle stays None and no overhead
    # is added to the baseline run.
    _diag_fh = None  # JSONL file handle
    if enable_diagnostics and alg_name == "AlphaEdit":
        if not diagnostics_save_path:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            _model_tag = model_name.replace("/", "_") if isinstance(model_name, str) else "model"
            diagnostics_save_path = str(
                run_dir / f"diagnostics_{alg_name}_{_model_tag}_{ds_name}_n{num_edits}_{ts}.jsonl"
            )
        import pathlib
        pathlib.Path(diagnostics_save_path).parent.mkdir(parents=True, exist_ok=True)
        _diag_fh = open(diagnostics_save_path, "a")
        print(f"[diagnostics] Writing step-level diagnostics to: {diagnostics_save_path}")
    # ─────────────────────────────────────────────────────────────────────────
    # hs = get_module_input_output_at_words(
    #         model,
    #         tok,
    #         hparams.layers[-1],
    #         context_templates=[request["template"] for request in eval_ds],
    #         words=[request["subject"] for request in eval_ds],
    #         module_template=hparams.layer_module_tmp,
    #         fact_token_strategy=hparams.fact_token,
    #     )[1].T
    # torch.save(hs, "pre_edit_hs.pt")
    # del hs
    glue_save_location = str(run_dir) + '/' + 'glue_eval/'
    os.makedirs(glue_save_location, exist_ok=True)
    round_save_dir = run_dir / "sequential_rounds"
    round_save_dir.mkdir(exist_ok=True)
    edited_history = []
    if alg_name == "AlphaEdit":
        with open(round_save_dir / "round_0000.json", "w") as f:
            json.dump(
                {
                    "round_idx": 0,
                    "edits_applied": 0,
                    "weight_metrics": summarize_rewrite_weight_metrics(
                        model, hparams, weight_reference
                    ),
                },
                f,
                indent=2,
            )
    cnt = 0
    for record_chunks in chunks(ds, num_edits):
        case_result_template = str(run_dir / "{}_edits-case_{}.json")
        print(f"=================================================================={cnt+1}_edit==================================================================")
        # Is the chunk already done?
        already_finished = True
        for record in record_chunks:
            if not Path(
                case_result_template.format(num_edits, record["case_id"])
            ).exists():
                already_finished = False
                break
        if already_finished:
            continue
        
        # Compute weight changes + record weights that changed
        case_ids = [record["case_id"] for record in record_chunks]
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT","AlphaEdit", "MEMIT_seq", "MEMIT_prune", "NSE"]) else dict()
        seq_args = dict(cache_c=cache_c) if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "NSE"]) else dict()
        nc_args = dict(P = P) if any(alg in alg_name for alg in ["AlphaEdit"]) else dict()
        stability_args = dict(weight_reference=weight_reference) if alg_name == "AlphaEdit" else dict()
        diag_args = dict(enable_diagnostics=enable_diagnostics) if alg_name == "AlphaEdit" else dict()
        if cnt == 0 and args.downstream_eval_steps > 0:#do initial GLUE EVAL WITH ORIGINAL MODEL
            glue_results = {'edit_num': -1}

            out_file = glue_save_location + "base.json"
            
            glue_eval = GLUEEval(model, tok, number_of_tests = 100)
            glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)

            #store the individual overall result file
            output_filename = out_file.replace('.json', '_glue.json')
            with open(output_filename, "w") as f:
                json.dump(glue_results, f, indent=4)
        start = time()
        round_edit_diagnostics = None
        if alg_name == "AlphaEdit":
            edited_model, cache_c, round_edit_diagnostics = apply_algo(
                model,
                tok,
                [
                    {"case_id": record["case_id"], **rewrite_dict}
                    for record in record_chunks
                    for rewrite_dict in (
                        record["requested_rewrite"]
                        if isinstance(record["requested_rewrite"], list)
                        else [record["requested_rewrite"]]
                    )
                ],
                hparams,
                **args_conserve_memory,
                **etc_args,
                **seq_args,
                **nc_args,
                **stability_args,
                **diag_args,
            )
        elif any(alg in alg_name for alg in ["MEMIT_seq", "NSE"]):
            edited_model, cache_c = apply_algo(
                model,
                tok,
                [
                    {"case_id": record["case_id"], **rewrite_dict}
                    for record in record_chunks
                    for rewrite_dict in (
                        record["requested_rewrite"]
                        if isinstance(record["requested_rewrite"], list)
                        else [record["requested_rewrite"]]
                    )
                ],
                hparams,
                **args_conserve_memory,
                **etc_args,
                **seq_args,
                **nc_args,
            )
        elif alg_name == "MEMIT_prune":
            if cnt == 0:
                edited_model, weights_copy = apply_algo(
                    model,
                    tok,
                    [
                        {"case_id": record["case_id"], **rewrite_dict}
                        for record in record_chunks
                        for rewrite_dict in (
                            record["requested_rewrite"]
                            if isinstance(record["requested_rewrite"], list)
                            else [record["requested_rewrite"]]
                        )
                    ],
                    hparams,
                    return_orig_weights=True,
                    **args_conserve_memory,
                    **etc_args,
                )
                # Initialize the upd_matrix dictionary
                upd_matrix = {}
            else:
                edited_model, _ = apply_algo(
                    model,
                    tok,
                    [
                        {"case_id": record["case_id"], **rewrite_dict}
                        for record in record_chunks
                        for rewrite_dict in (
                            record["requested_rewrite"]
                            if isinstance(record["requested_rewrite"], list)
                            else [record["requested_rewrite"]]
                        )
                    ],
                    hparams,
                    return_orig_weights=False,
                    **args_conserve_memory,
                    **etc_args,
                )
            if cnt == (dataset_size_limit/num_edits) - 1:
            # Calculate the weight update matrix
                with torch.no_grad():
                    for k, v in weights_copy.items():
                        current_weight = nethook.get_parameter(model, k)
                        upd_matrix[k] = current_weight - v.to("cuda")
                        # Calculate max singular value of the original weight
                        _, S_orig, _ = torch.svd(v)
                        max_sigma = S_orig.max().item()

                        # Adjust the upd_matrix singular values
                        U_upd, S_upd, V_upd = torch.svd(upd_matrix[k])
                        adjusted_S = torch.where(
                            S_upd > max_sigma,
                            torch.log(S_upd) - torch.log(torch.tensor(max_sigma, device='cuda')) + max_sigma,
                            S_upd
                        )
                        upd_matrix[k] = torch.matmul(U_upd, torch.matmul(torch.diag(adjusted_S), V_upd.t()))

                # Apply the adjusted updates to the model
                with torch.no_grad():
                    for k in upd_matrix:
                        original_weight = nethook.get_parameter(model, k)
                        adjusted_weight = original_weight + upd_matrix[k]
                        original_weight.copy_(adjusted_weight)
        else:
            edited_model, _ = apply_algo(
                model,
                tok,
                [
                    {"case_id": record["case_id"], **rewrite_dict}
                    for record in record_chunks
                    for rewrite_dict in (
                        record["requested_rewrite"]
                        if isinstance(record["requested_rewrite"], list)
                        else [record["requested_rewrite"]]
                    )
                ],
                hparams,
                return_orig_weights=False,
                **args_conserve_memory,
                **etc_args,
            )
        exec_time = time() - start
        cnt+=1
        edited_history.extend(record_chunks)
        print("Execution took", exec_time)
        # Evaluate new model

        round_glue_results = None
        if args.downstream_eval_steps > 0 and cnt % args.downstream_eval_steps == 0:
            glue_results = {
                        'edit_num': cnt*num_edits,
                        'case_id': case_ids
                        }

            out_file = glue_save_location + "case_{}.json".format(record["case_id"])#stores the last case ID of the batch

            glue_eval = GLUEEval(model, tok, number_of_tests = 100)
            glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)
                    
            #store the individual overall result file
            output_filename = out_file.replace('.json', '_glue.json')
            with open(output_filename, "w") as f:
                json.dump(glue_results, f, indent=4)
            round_glue_results = glue_results

        if alg_name == "AlphaEdit" and args.analysis_interval > 0 and cnt % args.analysis_interval == 0:
            # --- multi-strategy history probing ---
            probe_random   = select_history_probe(edited_history, args.history_probe_size)
            probe_oldest   = select_history_probe_oldest(edited_history, args.history_probe_size)
            probe_recent   = select_history_probe_recent(edited_history, args.history_probe_size)
            probe_strata   = select_history_probe_stratified(edited_history, args.history_probe_size)

            stratified_eval = {
                stratum: evaluate_record_list(model, tok, records, ds_eval_method)
                for stratum, records in probe_strata.items()
                if records
            }

            round_payload = {
                "round_idx": cnt,
                "edits_applied": cnt * num_edits,
                "chunk_case_ids": [record["case_id"] for record in record_chunks],
                "edit_time": exec_time,
                "update_diagnostics": round_edit_diagnostics,
                "weight_metrics": summarize_rewrite_weight_metrics(
                    model, hparams, weight_reference
                ),
                "current_batch_eval": evaluate_record_list(
                    model, tok, record_chunks, ds_eval_method
                ),
                "history_probe_eval": evaluate_record_list(
                    model, tok, probe_random, ds_eval_method
                ),
                "history_probe_oldest_eval": evaluate_record_list(
                    model, tok, probe_oldest, ds_eval_method
                ),
                "history_probe_recent_eval": evaluate_record_list(
                    model, tok, probe_recent, ds_eval_method
                ),
                "history_probe_stratified_eval": stratified_eval,
            }
            round_payload["immediate_current_batch_edit_success"] = (
                round_payload["current_batch_eval"]["summary"]["edit_success"]
            )
            if round_glue_results is not None:
                round_payload["downstream_eval"] = round_glue_results
            with open(round_save_dir / f"round_{cnt:04d}.json", "w") as f:
                json.dump(round_payload, f, indent=2)

        # ── Diagnostics JSONL: write one record per step ─────────────────────
        # Update diagnostics (conflict_sub, block_ratio, …) are always written.
        # Eval metrics (ES, locality) come from a lightweight current-batch eval
        # that runs only when diagnostics are enabled — independent of
        # analysis_interval so the diagnostics run does not need sequential_rounds.
        if _diag_fh is not None and alg_name == "AlphaEdit" and round_edit_diagnostics is not None:
            # Collect request metadata from the current chunk
            _requests_meta = []
            for _rec in record_chunks:
                _rw = _rec.get("requested_rewrite", {})
                _rw_list = _rw if isinstance(_rw, list) else [_rw]
                for _r in _rw_list:
                    _t_new  = _r.get("target_new")
                    _t_true = _r.get("target_true")
                    _requests_meta.append({
                        "case_id":    _rec["case_id"],
                        "subject":    _r.get("subject"),
                        "relation":   _r.get("relation_id"),
                        "target_new":  _t_new.get("str")  if isinstance(_t_new,  dict) else _t_new,
                        "target_true": _t_true.get("str") if isinstance(_t_true, dict) else _t_true,
                    })

            # Current-batch eval: reuse from analysis_interval block if available,
            # otherwise run a fresh lightweight eval on the current batch only.
            _eval_summary = None
            try:
                _eval_summary = round_payload["current_batch_eval"]["summary"]
            except (NameError, KeyError, TypeError):
                pass
            if _eval_summary is None:
                try:
                    _cb = evaluate_record_list(model, tok, record_chunks, ds_eval_method)
                    _eval_summary = _cb.get("summary")
                except Exception as _ee:
                    print(f"[diagnostics] Warning: current-batch eval failed – {_ee}")

            # History probe eval: only available when analysis_interval fires.
            # Random probe: mixed age sample.
            # Oldest probe: EARLIEST edited records — most at risk of forgetting.
            # Stratified probe: 4 quartile buckets (q1_oldest … q4_recent).
            _hp_summary = None
            _hp_oldest_summary = None
            _hp_strat_summaries = {}   # keyed by stratum label
            try:
                _hp_summary        = round_payload["history_probe_eval"]["summary"]
                _hp_oldest_summary = round_payload["history_probe_oldest_eval"]["summary"]
            except (NameError, KeyError, TypeError):
                pass
            try:
                for _stratum, _strat_eval in round_payload["history_probe_stratified_eval"].items():
                    if isinstance(_strat_eval, dict) and "summary" in _strat_eval:
                        _hp_strat_summaries[_stratum] = _strat_eval["summary"]
            except (NameError, KeyError, TypeError):
                pass

            _agg = round_edit_diagnostics.get("aggregate", {})

            _diag_record = {
                "step_id":          cnt,
                "edits_applied":    cnt * num_edits,
                "edit_time_sec":    exec_time,
                "case_ids":         [_rec["case_id"] for _rec in record_chunks],
                "requests":         _requests_meta,
                # ── per-layer diagnostics (dict keyed by layer index as string)
                "layers":           round_edit_diagnostics.get("layers", {}),
                # ── step-level aggregates ────────────────────────────────────
                "mean_conflict_sub":              _agg.get("mean_conflict_sub"),
                "mean_block_ratio":               _agg.get("mean_block_ratio"),
                "mean_rhs_block_ratio":           _agg.get("mean_rhs_block_ratio"),
                "mean_delta_raw_fro":             _agg.get("mean_delta_raw_fro"),
                "mean_delta_block_fro":           _agg.get("mean_delta_block_fro"),
                "mean_update_norm":               _agg.get("mean_update_norm"),
                "mean_update_to_orig_norm_ratio": _agg.get("mean_update_to_orig_norm_ratio"),
                "mean_delta_spectral":            _agg.get("mean_delta_spectral"),
                # ── eval metrics: current batch (immediate write success) ────
                "edit_success":              _eval_summary.get("edit_success")             if _eval_summary else None,
                "locality_preservation":     _eval_summary.get("locality_preservation")    if _eval_summary else None,
                "paraphrase_success":        _eval_summary.get("paraphrase_success")       if _eval_summary else None,
                "new_first_token_logit":     _eval_summary.get("new_first_token_logit")    if _eval_summary else None,
                "first_token_logit_margin":  _eval_summary.get("first_token_logit_margin") if _eval_summary else None,
                # ── history probe eval (retention / forgetting curve) ─────────
                # Only populated at analysis_interval steps; None otherwise.
                # random: mixed-age sample
                "hp_edit_success":       _hp_summary.get("edit_success")          if _hp_summary else None,
                "hp_locality":           _hp_summary.get("locality_preservation") if _hp_summary else None,
                "hp_paraphrase_success": _hp_summary.get("paraphrase_success")    if _hp_summary else None,
                # oldest: earliest edits only — strongest forgetting signal
                "hp_oldest_edit_success":       _hp_oldest_summary.get("edit_success")          if _hp_oldest_summary else None,
                "hp_oldest_locality":           _hp_oldest_summary.get("locality_preservation") if _hp_oldest_summary else None,
                "hp_oldest_paraphrase_success": _hp_oldest_summary.get("paraphrase_success")    if _hp_oldest_summary else None,
                # stratified quartile probes: q1_oldest / q2 / q3 / q4_recent
                # Each stratum samples probe_size//4 records from that age bucket.
                **{
                    f"strat_{_s}_edit_success":       _hp_strat_summaries[_s].get("edit_success")          if _s in _hp_strat_summaries else None
                    for _s in ["q1_oldest", "q2", "q3", "q4_recent"]
                },
                **{
                    f"strat_{_s}_locality":           _hp_strat_summaries[_s].get("locality_preservation") if _s in _hp_strat_summaries else None
                    for _s in ["q1_oldest", "q2", "q3", "q4_recent"]
                },
                **{
                    f"strat_{_s}_paraphrase_success": _hp_strat_summaries[_s].get("paraphrase_success")    if _s in _hp_strat_summaries else None
                    for _s in ["q1_oldest", "q2", "q3", "q4_recent"]
                },
            }

            _diag_fh.write(json.dumps(_diag_record) + "\n")
            _diag_fh.flush()
        # ─────────────────────────────────────────────────────────────────────

    # hs = get_module_input_output_at_words(
    #         edited_model,
    #         tok,
    #         hparams.layers[-1],
    #         context_templates=[request["template"] for request in eval_ds],
    #         words=[request["subject"] for request in eval_ds],
    #         module_template=hparams.layer_module_tmp,
    #         fact_token_strategy=hparams.fact_token,
    #     )[1].T
    # torch.save(hs, "post_edit_hs_memit.pt")
    # Skip final per-record eval and model saving when --skip_final_eval is set.
    # Use this for diagnostics runs where the JSONL already captures what's needed.
    _skip_final = getattr(args, "skip_final_eval", False) if args is not None else False
    if not _skip_final:
        start = time()
        gen_test_vars = [snips, vec]
        for record in ds:
            out_file = Path(case_result_template.format(num_edits, record["case_id"]))
            if out_file.exists():
                print(f"Skipping {out_file}; already exists")
                continue
            metrics = {
                "case_id": record["case_id"],
                "grouped_case_ids": case_ids,
                "num_edits": num_edits,
                "requested_rewrite": record["requested_rewrite"],
                "time": exec_time,
                "conflict_metrics": compute_conflict_metrics_for_record(
                    edited_model, tok, record
                ),
                "post": ds_eval_method(
                    edited_model,
                    tok,
                    record,
                    *(
                        gen_test_vars
                        if record["case_id"] % generation_test_interval == 0
                        else [None, None]
                    ),  # Only test generation every generation_test_interval cases
                ),
            }

            # Dump metrics in .json
            with open(out_file, "w") as f:
                json.dump(metrics, f, indent=1)

            print("Evaluation took", time() - start)
    else:
        print("[skip_final_eval] Skipping final per-record evaluation and model save.")

    # ── Diagnostics finalisation ─────────────────────────────────────────────
    if _diag_fh is not None:
        _diag_fh.close()
        print(f"[diagnostics] Closed JSONL: {diagnostics_save_path}")

        # Best-effort CSV summary (one row per step, flat numeric fields only)
        try:
            _csv_path = diagnostics_save_path.replace(".jsonl", "_summary.csv")
            import csv as _csv_mod
            _rows = []
            with open(diagnostics_save_path, "r") as _jf:
                for _line in _jf:
                    _line = _line.strip()
                    if not _line:
                        continue
                    _rec = json.loads(_line)
                    _flat = {
                        "step_id":          _rec.get("step_id"),
                        "edits_applied":    _rec.get("edits_applied"),
                        "edit_time_sec":    _rec.get("edit_time_sec"),
                        "mean_conflict_sub":              _rec.get("mean_conflict_sub"),
                        "mean_block_ratio":               _rec.get("mean_block_ratio"),
                        "mean_rhs_block_ratio":           _rec.get("mean_rhs_block_ratio"),
                        "mean_delta_raw_fro":             _rec.get("mean_delta_raw_fro"),
                        "mean_delta_block_fro":           _rec.get("mean_delta_block_fro"),
                        "mean_update_norm":               _rec.get("mean_update_norm"),
                        "mean_update_to_orig_norm_ratio": _rec.get("mean_update_to_orig_norm_ratio"),
                        "mean_delta_spectral":            _rec.get("mean_delta_spectral"),
                        "edit_success":          _rec.get("edit_success"),
                        "locality_preservation": _rec.get("locality_preservation"),
                        "paraphrase_success":    _rec.get("paraphrase_success"),
                        "first_token_logit_margin": _rec.get("first_token_logit_margin"),
                        "hp_edit_success":       _rec.get("hp_edit_success"),
                        "hp_locality":           _rec.get("hp_locality"),
                        "hp_paraphrase_success": _rec.get("hp_paraphrase_success"),
                    }
                    # Flatten per-layer scalars (e.g. layer "13" → conflict_sub_L13)
                    for _lname, _ldata in (_rec.get("layers") or {}).items():
                        if not isinstance(_ldata, dict):
                            continue
                        for _k, _v in _ldata.items():
                            if isinstance(_v, (int, float, type(None))):
                                _flat[f"{_k}_L{_lname}"] = _v
                    _rows.append(_flat)

            if _rows:
                _all_keys = list(dict.fromkeys(k for row in _rows for k in row))
                with open(_csv_path, "w", newline="") as _cf:
                    writer = _csv_mod.DictWriter(_cf, fieldnames=_all_keys, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(_rows)
                print(f"[diagnostics] CSV summary saved to: {_csv_path}")
        except Exception as _e:
            print(f"[diagnostics] Warning: could not write CSV summary – {_e}")
    # ─────────────────────────────────────────────────────────────────────────

    if not _skip_final:
        save_dir = run_dir / "final_model"
        edited_model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)

    
    
    
    
    
def get_project(model, tok, layer, hparams):
    force_recompute = False
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples
        if not force_recompute
        else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
    ).cpu()
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    print(len(small_singular_indices))
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T
def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["AlphaEdit","MEMIT_rect", "MEMIT_seq","MEMIT_prune", "MEMIT", "ROME", "FT", "MEND","NSE"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre", "mquake"],
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--downstream_eval_steps",
        type=int,
        default=0,
        help="If we want to do sequential editing or not",
    )
    parser.add_argument(
        "--analysis_interval",
        type=int,
        default=1,
        help="Save round-level sequential diagnostics every N edit rounds.",
    )
    parser.add_argument(
        "--history_probe_size",
        type=int,
        default=32,
        help="How many edited cases to reevaluate for retention/locality diagnostics.",
    )
    parser.add_argument(
        "--analysis_top_singular_values",
        type=int,
        default=8,
        help="How many leading/trailing singular values to store per edited layer.",
    )
    parser.add_argument(
        "--numerical_stability",
        choices=["none", "svd_clip"],
        default="none",
        help="Post-edit numerical stability intervention for AlphaEdit.",
    )
    parser.add_argument(
        "--stability_spectral_multiplier",
        type=float,
        default=0.0,
        help="Cap edited-layer spectral norm to this multiple of the pre-edit reference.",
    )
    parser.add_argument(
        "--stability_condition_number",
        type=float,
        default=0.0,
        help="Cap edited-layer condition number by lifting small singular values after each edit.",
    )
    parser.add_argument(
        "--stability_fro_drift_ratio",
        type=float,
        default=0.0,
        help="Cap Frobenius drift relative to the original edited weight matrix.",
    )
    parser.add_argument(
        "--delta_fro_tau",
        type=float,
        default=0.0,
        help="AlphaEditDeltaFro: absolute Frobenius threshold τ_F for ΔW clipping. 0 = use --delta_fro_ratio.",
    )
    parser.add_argument(
        "--delta_fro_ratio",
        type=float,
        default=0.0,
        help="AlphaEditDeltaFro: τ_F = ratio * ||W0||_F. Used when --delta_fro_tau is 0. 0 = disabled.",
    )
    parser.add_argument(
        "--delta_spectral_tau",
        type=float,
        default=0.0,
        help="AlphaEditDeltaSpectral: spectral-norm threshold τ_2 for ΔW clipping. 0 = disabled.",
    )
    parser.add_argument(
        "--skip_final_eval",
        action="store_true",
        default=False,
        help=(
            "Skip the final per-record evaluation loop and model save. "
            "Use for diagnostics runs where the JSONL captures all needed data."
        ),
    )
    parser.add_argument(
        "--enable_diagnostics",
        action="store_true",
        default=False,
        help=(
            "AlphaEdit only: compute and log per-step mechanism-analysis diagnostics "
            "(conflict_sub, block_ratio) to a JSONL file.  Off by default; adds one "
            "extra linear solve per layer per step.  Does NOT change any model weights "
            "or evaluation results."
        ),
    )
    parser.add_argument(
        "--diagnostics_save_path",
        type=str,
        default="",
        help=(
            "Path for the diagnostics JSONL file.  Auto-generated from run_dir and "
            "timestamp when empty (default).  Only used when --enable_diagnostics is set."
        ),
    )
    parser.add_argument(
        "--knowledge_conflict",
        choices=["none", "old_target_margin"],
        default="none",
        help="Conflict intervention used inside compute_z for AlphaEdit.",
    )
    parser.add_argument(
        "--conflict_loss_weight",
        type=float,
        default=0.0,
        help="Weight on the old-vs-new target margin loss.",
    )
    parser.add_argument(
        "--conflict_margin",
        type=float,
        default=0.0,
        help="Desired margin between new-target and old-target sequence losses.",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        args =args,
    )
