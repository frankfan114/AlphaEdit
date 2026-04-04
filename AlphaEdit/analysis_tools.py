import random
from typing import Dict, List, Optional, Sequence

import torch

from util import nethook


EPS = 1e-12


def get_rewrite_weight_names(hparams) -> List[str]:
    return [
        f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        for layer in hparams.layers
    ]


def clone_rewrite_weights(model, hparams) -> Dict[str, torch.Tensor]:
    return {
        weight_name: nethook.get_parameter(model, weight_name)
        .detach()
        .float()
        .cpu()
        .clone()
        for weight_name in get_rewrite_weight_names(hparams)
    }


def _safe_float(value: float) -> Optional[float]:
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.item()
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def _target_text(record: Dict, key: str, fallback_key: Optional[str] = None) -> Optional[str]:
    requested = record.get("requested_rewrite", {})
    if isinstance(requested, dict) and key in requested:
        target = requested[key]
        if isinstance(target, dict):
            return target.get("str")
        if isinstance(target, str):
            return target
    if fallback_key is not None:
        return record.get(fallback_key)
    return None


def _tokenize_target(tok, target_text: str, device: torch.device) -> torch.Tensor:
    if target_text is None:
        return None
    if len(target_text) > 0 and target_text[0] != " ":
        target_text = " " + target_text
    target_ids = tok(target_text, return_tensors="pt").to(device)["input_ids"][0]
    if len(target_ids) > 0 and (
        target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id
    ):
        target_ids = target_ids[1:]
    return target_ids


def compute_target_score_on_prompt(
    model,
    tok,
    prompt_text: str,
    target_text: str,
) -> Dict[str, Optional[float]]:
    device = next(model.parameters()).device
    prompt_ids = tok(prompt_text, return_tensors="pt").to(device)["input_ids"][0]
    target_ids = _tokenize_target(tok, target_text, device)
    if target_ids is None or target_ids.numel() == 0:
        return {
            "first_token_logit": None,
            "mean_token_logit": None,
            "avg_logprob": None,
            "target_token_count": 0,
        }

    full_ids = torch.cat([prompt_ids, target_ids], dim=0).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids=full_ids).logits[0]

    prompt_len = prompt_ids.shape[0]
    positions = torch.arange(
        prompt_len - 1,
        prompt_len - 1 + target_ids.shape[0],
        device=device,
    )
    target_logits = logits[positions]
    gathered_logits = target_logits.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    gathered_logprobs = torch.log_softmax(target_logits, dim=-1).gather(
        1, target_ids.unsqueeze(1)
    ).squeeze(1)

    return {
        "first_token_logit": _safe_float(gathered_logits[0]),
        "mean_token_logit": _safe_float(gathered_logits.mean()),
        "avg_logprob": _safe_float(gathered_logprobs.mean()),
        "target_token_count": int(target_ids.shape[0]),
    }


def compute_conflict_metrics_for_record(model, tok, record: Dict) -> Dict[str, Optional[float]]:
    subject = record["requested_rewrite"]["subject"]
    prompt_list = [record["requested_rewrite"]["prompt"].format(subject)] + record.get(
        "paraphrase_prompts", []
    )
    new_text = _target_text(record, "target_new", fallback_key="new_answer")
    old_text = _target_text(record, "target_true", fallback_key="answer")

    new_scores = [
        compute_target_score_on_prompt(model, tok, prompt_text, new_text)
        for prompt_text in prompt_list
    ]
    old_scores = (
        [
            compute_target_score_on_prompt(model, tok, prompt_text, old_text)
            for prompt_text in prompt_list
        ]
        if old_text is not None
        else []
    )

    def mean_key(items: Sequence[Dict], key: str) -> Optional[float]:
        values = [item[key] for item in items if item.get(key) is not None]
        if not values:
            return None
        return _safe_float(sum(values) / len(values))

    new_first = mean_key(new_scores, "first_token_logit")
    old_first = mean_key(old_scores, "first_token_logit")
    new_avg_logprob = mean_key(new_scores, "avg_logprob")
    old_avg_logprob = mean_key(old_scores, "avg_logprob")

    return {
        "prompt_count": len(prompt_list),
        "new_first_token_logit": new_first,
        "old_first_token_logit": old_first,
        "first_token_logit_margin": (
            _safe_float(new_first - old_first)
            if new_first is not None and old_first is not None
            else None
        ),
        "new_avg_logprob": new_avg_logprob,
        "old_avg_logprob": old_avg_logprob,
        "avg_logprob_margin": (
            _safe_float(new_avg_logprob - old_avg_logprob)
            if new_avg_logprob is not None and old_avg_logprob is not None
            else None
        ),
    }


def compute_weight_svd_metrics(
    weight: torch.Tensor,
    reference_weight: Optional[torch.Tensor] = None,
    topk: int = 8,
) -> Dict[str, Optional[float]]:
    weight_cpu = weight.detach().float().cpu()
    singular_values = torch.linalg.svdvals(weight_cpu)
    nonzero_singular_values = singular_values[singular_values > EPS]

    spectral_norm = _safe_float(singular_values[0]) if singular_values.numel() > 0 else 0.0
    min_nonzero = (
        _safe_float(nonzero_singular_values[-1])
        if nonzero_singular_values.numel() > 0
        else None
    )
    fro_norm = _safe_float(torch.linalg.norm(weight_cpu, ord="fro"))
    stable_rank = (
        _safe_float((fro_norm**2) / (spectral_norm**2 + EPS))
        if fro_norm is not None and spectral_norm not in (None, 0.0)
        else None
    )
    condition_number = (
        _safe_float(spectral_norm / min_nonzero)
        if spectral_norm is not None and min_nonzero not in (None, 0.0)
        else None
    )

    metrics = {
        "shape": list(weight_cpu.shape),
        "frobenius_norm": fro_norm,
        "spectral_norm": spectral_norm,
        "min_nonzero_singular_value": min_nonzero,
        "condition_number": condition_number,
        "stable_rank": stable_rank,
        "singular_values_topk": [
            _safe_float(x) for x in singular_values[:topk]
        ],
        "singular_values_bottomk": [
            _safe_float(x) for x in singular_values[-topk:]
        ],
        "singular_value_quantiles": {
            "p25": _safe_float(torch.quantile(singular_values, 0.25))
            if singular_values.numel() > 0
            else None,
            "p50": _safe_float(torch.quantile(singular_values, 0.50))
            if singular_values.numel() > 0
            else None,
            "p75": _safe_float(torch.quantile(singular_values, 0.75))
            if singular_values.numel() > 0
            else None,
            "p90": _safe_float(torch.quantile(singular_values, 0.90))
            if singular_values.numel() > 0
            else None,
            "p99": _safe_float(torch.quantile(singular_values, 0.99))
            if singular_values.numel() > 0
            else None,
        },
    }

    if reference_weight is not None:
        reference_cpu = reference_weight.detach().float().cpu()
        ref_singular_values = torch.linalg.svdvals(reference_cpu)
        ref_nonzero = ref_singular_values[ref_singular_values > EPS]
        ref_spectral_norm = (
            _safe_float(ref_singular_values[0]) if ref_singular_values.numel() > 0 else 0.0
        )
        ref_min_nonzero = (
            _safe_float(ref_nonzero[-1]) if ref_nonzero.numel() > 0 else None
        )
        ref_condition = (
            _safe_float(ref_spectral_norm / ref_min_nonzero)
            if ref_spectral_norm is not None and ref_min_nonzero not in (None, 0.0)
            else None
        )
        drift = weight_cpu - reference_cpu
        drift_norm = _safe_float(torch.linalg.norm(drift, ord="fro"))
        reference_fro = _safe_float(torch.linalg.norm(reference_cpu, ord="fro"))
        metrics.update(
            {
                "frobenius_norm_drift": drift_norm,
                "relative_frobenius_drift": (
                    _safe_float(drift_norm / (reference_fro + EPS))
                    if drift_norm is not None and reference_fro is not None
                    else None
                ),
                "spectral_norm_ratio_to_reference": (
                    _safe_float(spectral_norm / (ref_spectral_norm + EPS))
                    if spectral_norm is not None and ref_spectral_norm is not None
                    else None
                ),
                "condition_number_ratio_to_reference": (
                    _safe_float(condition_number / (ref_condition + EPS))
                    if condition_number is not None and ref_condition is not None
                    else None
                ),
            }
        )

    return metrics


def summarize_rewrite_weight_metrics(
    model,
    hparams,
    reference_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, Dict]:
    per_layer = {}
    aggregate = {
        "mean_frobenius_norm_drift": [],
        "max_spectral_norm": [],
        "max_condition_number": [],
        "mean_stable_rank": [],
    }

    for layer in hparams.layers:
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        weight = nethook.get_parameter(model, weight_name)
        metrics = compute_weight_svd_metrics(
            weight,
            None if reference_weights is None else reference_weights.get(weight_name),
            topk=hparams.analysis_top_singular_values,
        )
        per_layer[str(layer)] = metrics
        if metrics.get("frobenius_norm_drift") is not None:
            aggregate["mean_frobenius_norm_drift"].append(
                metrics["frobenius_norm_drift"]
            )
        if metrics.get("spectral_norm") is not None:
            aggregate["max_spectral_norm"].append(metrics["spectral_norm"])
        if metrics.get("condition_number") is not None:
            aggregate["max_condition_number"].append(metrics["condition_number"])
        if metrics.get("stable_rank") is not None:
            aggregate["mean_stable_rank"].append(metrics["stable_rank"])

    summary = {
        "layers": per_layer,
        "aggregate": {
            "mean_frobenius_norm_drift": _safe_float(
                sum(aggregate["mean_frobenius_norm_drift"])
                / max(len(aggregate["mean_frobenius_norm_drift"]), 1)
            )
            if aggregate["mean_frobenius_norm_drift"]
            else None,
            "max_spectral_norm": _safe_float(max(aggregate["max_spectral_norm"]))
            if aggregate["max_spectral_norm"]
            else None,
            "max_condition_number": _safe_float(max(aggregate["max_condition_number"]))
            if aggregate["max_condition_number"]
            else None,
            "mean_stable_rank": _safe_float(
                sum(aggregate["mean_stable_rank"])
                / max(len(aggregate["mean_stable_rank"]), 1)
            )
            if aggregate["mean_stable_rank"]
            else None,
        },
    }
    return summary


def aggregate_probe_metrics(case_metrics: Sequence[Dict]) -> Dict[str, Optional[float]]:
    def collect_nested_bool_means(key: str) -> List[float]:
        values = []
        for metric in case_metrics:
            post = metric.get("post", {})
            if key in post and len(post[key]) > 0:
                values.append(float(sum(post[key])) / len(post[key]))
        return values

    def collect_scalar(key: str) -> List[float]:
        values = []
        for metric in case_metrics:
            value = metric.get("conflict_metrics", {}).get(key)
            if value is not None:
                values.append(value)
        return values

    summary = {
        "num_cases": len(case_metrics),
        "edit_success": None,
        "paraphrase_success": None,
        "locality_preservation": None,
        "new_first_token_logit": None,
        "old_first_token_logit": None,
        "first_token_logit_margin": None,
        "new_avg_logprob": None,
        "old_avg_logprob": None,
        "avg_logprob_margin": None,
    }

    rewrite = collect_nested_bool_means("rewrite_prompts_correct")
    paraphrase = collect_nested_bool_means("paraphrase_prompts_correct")
    locality = collect_nested_bool_means("neighborhood_prompts_correct")

    if rewrite:
        summary["edit_success"] = _safe_float(sum(rewrite) / len(rewrite))
    if paraphrase:
        summary["paraphrase_success"] = _safe_float(sum(paraphrase) / len(paraphrase))
    if locality:
        summary["locality_preservation"] = _safe_float(sum(locality) / len(locality))

    for key in [
        "new_first_token_logit",
        "old_first_token_logit",
        "first_token_logit_margin",
        "new_avg_logprob",
        "old_avg_logprob",
        "avg_logprob_margin",
    ]:
        values = collect_scalar(key)
        if values:
            summary[key] = _safe_float(sum(values) / len(values))

    return summary


def select_history_probe(
    edited_history: Sequence[Dict],
    probe_size: int,
) -> List[Dict]:
    """Uniform random sample from all history (unbiased baseline)."""
    if probe_size <= 0 or len(edited_history) <= probe_size:
        return list(edited_history)
    return random.sample(list(edited_history), probe_size)


def select_history_probe_oldest(
    edited_history: Sequence[Dict],
    probe_size: int,
) -> List[Dict]:
    """Take the oldest N edits — worst-case forgetting."""
    if probe_size <= 0 or len(edited_history) <= probe_size:
        return list(edited_history)
    return list(edited_history[:probe_size])


def select_history_probe_recent(
    edited_history: Sequence[Dict],
    probe_size: int,
) -> List[Dict]:
    """Take the most recent N edits — best-case forgetting."""
    if probe_size <= 0 or len(edited_history) <= probe_size:
        return list(edited_history)
    return list(edited_history[-probe_size:])


def select_history_probe_stratified(
    edited_history: Sequence[Dict],
    probe_size: int,
    n_strata: int = 4,
) -> Dict[str, List[Dict]]:
    """
    Split history into n_strata equal-size buckets and sample probe_size//n_strata
    from each. Returns a dict keyed by stratum label (q1_oldest … q4_recent).
    """
    history = list(edited_history)
    n = len(history)
    per_stratum = max(1, probe_size // n_strata)
    labels = [f"q{i+1}_oldest" if i == 0 else (f"q{n_strata}_recent" if i == n_strata - 1 else f"q{i+1}") for i in range(n_strata)]
    result = {}
    for i, label in enumerate(labels):
        lo = int(i * n / n_strata)
        hi = int((i + 1) * n / n_strata)
        bucket = history[lo:hi]
        if len(bucket) == 0:
            result[label] = []
        elif len(bucket) <= per_stratum:
            result[label] = bucket
        else:
            result[label] = random.sample(bucket, per_stratum)
    return result


def apply_stability_projection_(
    parameter: torch.Tensor,
    reference_weight: torch.Tensor,
    hparams,
) -> Dict[str, Optional[float]]:
    current_cpu = parameter.detach().float().cpu()
    reference_cpu = reference_weight.detach().float().cpu()
    pre_projection_delta_norm = _safe_float(
        torch.linalg.norm(current_cpu - reference_cpu, ord="fro")
    )

    if hparams.numerical_stability == "none":
        return {
            "applied": False,
            "pre_projection_delta_norm": pre_projection_delta_norm,
            "post_projection_delta_norm": pre_projection_delta_norm,
            "spectral_upper_bound": None,
            "condition_lower_bound": None,
        }

    u_mat, singular_values, vh_mat = torch.linalg.svd(current_cpu, full_matrices=False)
    reference_singular_values = torch.linalg.svdvals(reference_cpu)

    clipped = singular_values.clone()
    applied = False
    upper = None
    lower = None

    if (
        hparams.stability_spectral_multiplier is not None
        and hparams.stability_spectral_multiplier > 0
        and reference_singular_values.numel() > 0
    ):
        upper = reference_singular_values[0] * hparams.stability_spectral_multiplier
        clipped = torch.clamp(clipped, max=upper)
        applied = True

    if hparams.stability_condition_number is not None and hparams.stability_condition_number > 0:
        if upper is None:
            upper = clipped[0]
        lower = upper / hparams.stability_condition_number
        clipped = torch.clamp(clipped, min=lower)
        applied = True

    projected = (u_mat * clipped.unsqueeze(0)) @ vh_mat

    if (
        hparams.stability_fro_drift_ratio is not None
        and hparams.stability_fro_drift_ratio > 0
    ):
        delta = projected - reference_cpu
        delta_norm = torch.linalg.norm(delta, ord="fro")
        max_delta_norm = (
            hparams.stability_fro_drift_ratio
            * torch.linalg.norm(reference_cpu, ord="fro")
        )
        if delta_norm > max_delta_norm and delta_norm > 0:
            projected = reference_cpu + delta * (max_delta_norm / delta_norm)
            applied = True

    if applied:
        with torch.no_grad():
            parameter.copy_(projected.to(parameter.device, dtype=parameter.dtype))

    return {
        "applied": applied,
        "pre_projection_delta_norm": pre_projection_delta_norm,
        "post_projection_delta_norm": _safe_float(
            torch.linalg.norm(projected - reference_cpu, ord="fro")
        ),
        "spectral_upper_bound": _safe_float(upper) if upper is not None else None,
        "condition_lower_bound": _safe_float(lower) if lower is not None else None,
    }
