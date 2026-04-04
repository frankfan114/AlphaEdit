#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1)


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = flatten(a).astype(np.float64)
    b = flatten(b).astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def topk_indices(mat: np.ndarray, k: int) -> np.ndarray:
    """
    Return top-k indices (flat indices) of mat, descending by value.
    """
    x = flatten(mat)
    k = max(1, min(k, x.size))
    # argpartition for speed
    idx = np.argpartition(-x, k - 1)[:k]
    # sort those k
    idx = idx[np.argsort(-x[idx])]
    return idx


def frac_topk_in_subject(mat: np.ndarray, subj_range: Tuple[int, int], k: int) -> float:
    """
    mat: [T, L]
    subj_range: (b, e) over token axis
    """
    T, L = mat.shape
    b, e = subj_range
    idx = topk_indices(mat, k)
    # convert flat -> (t, l)
    t = idx // L
    in_subj = (t >= b) & (t < e)
    return float(in_subj.mean())


def layer_mass(mat: np.ndarray) -> np.ndarray:
    """
    Average over tokens -> [L]
    """
    return mat.mean(axis=0)


def peak_layer(mat: np.ndarray) -> int:
    lm = layer_mass(mat)
    return int(np.argmax(lm))


def recoverable_ratio(mat: np.ndarray, low: float, high: float, alpha: float = 0.5) -> float:
    """
    mat: scores [T,L]
    Threshold: low + alpha*(high-low)
    """
    if not np.isfinite(low) or not np.isfinite(high):
        return np.nan
    thr = low + alpha * (high - low)
    return float((mat >= thr).mean())


# -----------------------------
# NPZ parsing
# -----------------------------
FNAME_RE = re.compile(
    r"^(base|edited)_knowledge_(\d+)(?:_(mlp|attn))?\.npz$"
)


@dataclass(frozen=True)
class NPZKey:
    known_id: int
    kind: str  # "none" | "mlp" | "attn"


def index_cases(run_dir: Path) -> Dict[NPZKey, Dict[str, Path]]:
    """
    Build index:
      key -> {"base": path, "edited": path}
    Expect structure:
      run_dir/cases/base_knowledge_{id}{_mlp/_attn}.npz
      run_dir/cases/edited_knowledge_{id}{_mlp/_attn}.npz
    """
    cases_dir = run_dir / "cases"
    if not cases_dir.is_dir():
        raise FileNotFoundError(f"cases/ not found under: {run_dir}")

    out: Dict[NPZKey, Dict[str, Path]] = {}
    for p in cases_dir.glob("*.npz"):
        m = FNAME_RE.match(p.name)
        if not m:
            continue
        label, kid, kind = m.group(1), int(m.group(2)), m.group(3)
        kind = kind if kind is not None else "none"
        key = NPZKey(known_id=kid, kind=kind)
        out.setdefault(key, {})
        out[key][label] = p

    # keep only those with both base + edited
    out = {k: v for k, v in out.items() if "base" in v and "edited" in v}
    return out


def load_npz(path: Path) -> Dict[str, object]:
    d = dict(np.load(path, allow_pickle=True))
    # normalize some fields
    if "scores" in d:
        d["scores"] = np.array(d["scores"])
    if "subject_range" in d:
        sr = np.array(d["subject_range"]).astype(int).tolist()
        if len(sr) == 2:
            d["subject_range"] = (int(sr[0]), int(sr[1]))
    return d


def get_scalar(d: Dict[str, object], key: str) -> float:
    if key not in d:
        return np.nan
    v = d[key]
    if isinstance(v, np.ndarray) and v.shape == ():
        return safe_float(v.item())
    return safe_float(v)


# -----------------------------
# Core metrics per run
# -----------------------------
def compute_pair_metrics(
    base_npz: Dict[str, object],
    edit_npz: Dict[str, object],
    topk: int,
    alpha: float,
) -> Dict[str, object]:
    Sb = base_npz["scores"].astype(np.float64)
    Se = edit_npz["scores"].astype(np.float64)

    # sanity: shapes
    if Sb.shape != Se.shape:
        # allow but warn in output
        shape_ok = False
    else:
        shape_ok = True

    low_b = get_scalar(base_npz, "low_score")
    high_b = get_scalar(base_npz, "high_score")
    if not np.isfinite(high_b):  # your script also stores p_expect
        high_b = get_scalar(base_npz, "p_expect")

    low_e = get_scalar(edit_npz, "low_score")
    high_e = get_scalar(edit_npz, "high_score")
    if not np.isfinite(high_e):
        high_e = get_scalar(edit_npz, "p_expect")

    sr_b = base_npz.get("subject_range", None)
    sr_e = edit_npz.get("subject_range", None)
    sr = sr_b if sr_b is not None else sr_e
    if sr is None:
        sr = (0, 0)

    # choose k: support "topk=0 => 1% of cells"
    T, L = Sb.shape
    if topk <= 0:
        k = max(1, int(0.01 * T * L))
    else:
        k = min(topk, T * L)

    # recoverability
    Rb = recoverable_ratio(Sb, low_b, high_b, alpha=alpha)
    Re = recoverable_ratio(Se, low_e, high_e, alpha=alpha)

    # localization in subject span
    frac_subj_b = frac_topk_in_subject(Sb, sr, k)
    frac_subj_e = frac_topk_in_subject(Se, sr, k)

    # layer peak
    peak_b = peak_layer(Sb)
    peak_e = peak_layer(Se)

    # base<->edited similarity + deltas
    sim_be = cosine_sim(Sb, Se)
    delta = Se - Sb
    return {
        "shape_match": shape_ok,
        "T": T,
        "L": L,
        "subject_b": int(sr[0]),
        "subject_e": int(sr[1]),
        "p_expect_base": high_b,
        "p_expect_edited": high_e,
        "low_base": low_b,
        "low_edited": low_e,
        "recoverable_ratio_base": Rb,
        "recoverable_ratio_edited": Re,
        "topk": k,
        "frac_subj_topk_base": frac_subj_b,
        "frac_subj_topk_edited": frac_subj_e,
        "peak_layer_base": peak_b,
        "peak_layer_edited": peak_e,
        "cosine_S_base_vs_S_edited": sim_be,
        "deltaS_mean": float(np.mean(delta)),
        "deltaS_absmean": float(np.mean(np.abs(delta))),
        "deltaS_l2": float(np.linalg.norm(flatten(delta))),
        "deltaS_max": float(np.max(delta)),
        "deltaS_min": float(np.min(delta)),
    }


def compute_pref_metrics(
    base_true: Dict[str, object],
    edit_true: Dict[str, object],
    base_new: Dict[str, object],
    edit_new: Dict[str, object],
    topk: int,
) -> Dict[str, object]:
    """
    Preference maps:
      Pref = S_true - S_new
      ShiftPref = Pref_edit - Pref_base
    """
    Sb_t = base_true["scores"].astype(np.float64)
    Se_t = edit_true["scores"].astype(np.float64)
    Sb_n = base_new["scores"].astype(np.float64)
    Se_n = edit_new["scores"].astype(np.float64)

    # use subject_range from true run if present
    sr = base_true.get("subject_range", None) or edit_true.get("subject_range", None) or (0, 0)

    Pref_b = Sb_t - Sb_n
    Pref_e = Se_t - Se_n
    Shift = Pref_e - Pref_b

    T, L = Pref_b.shape
    if topk <= 0:
        k = max(1, int(0.01 * T * L))
    else:
        k = min(topk, T * L)

    b, e = sr
    # region masks
    subj_mask = np.zeros((T, L), dtype=bool)
    subj_mask[b:e, :] = True
    non_mask = ~subj_mask

    # summary stats
    shift_mean = float(np.mean(Shift))
    shift_pos_mass = float(np.mean(np.maximum(Shift, 0.0)))
    shift_neg_mass = float(np.mean(np.maximum(-Shift, 0.0)))

    shift_subj_mean = float(np.mean(Shift[subj_mask])) if (e > b) else np.nan
    shift_non_mean = float(np.mean(Shift[non_mask])) if (e > b) else float(np.mean(Shift))

    # topk localization on Shift (where did preference change most)
    frac_shift_topk_in_subj = frac_topk_in_subject(Shift, sr, k)

    # did preference flip sign from base->edited on the strongest base-pref cells?
    # (optional but useful)
    idx = topk_indices(np.abs(Pref_b), k)  # cells where base preference magnitude is large
    base_sign = np.sign(flatten(Pref_b)[idx])
    edit_sign = np.sign(flatten(Pref_e)[idx])
    # count flips excluding zeros
    valid = (base_sign != 0)
    flip_rate = float(np.mean((base_sign[valid] != edit_sign[valid]))) if np.any(valid) else np.nan

    return {
        "pref_shift_mean": shift_mean,
        "pref_shift_pos_mass": shift_pos_mass,
        "pref_shift_neg_mass": shift_neg_mass,
        "pref_shift_subj_mean": shift_subj_mean,
        "pref_shift_non_subj_mean": shift_non_mean,
        "pref_shift_topk": k,
        "pref_shift_topk_frac_in_subj": float(frac_shift_topk_in_subj),
        "pref_flip_rate_on_topk_basepref": flip_rate,
        "cosine_Pref_base_vs_Pref_edited": cosine_sim(Pref_b, Pref_e),
        "cosine_ShiftPref_vs_0": np.nan,  # placeholder if you want
    }


def p_expect_from_run(npz_d: Dict[str, object]) -> float:
    v = get_scalar(npz_d, "p_expect")
    if np.isfinite(v):
        return v
    return get_scalar(npz_d, "high_score")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Analyze causal_trace_compare .npz outputs and compute metrics (base vs edited; optional true-vs-new preference)."
    )
    ap.add_argument("--run_dir_true", type=str, required=True,
                    help="Output dir of compare script for expect=target_true (contains cases/).")
    ap.add_argument("--run_dir_new", type=str, default=None,
                    help="Optional: output dir for expect=target_new, to compute Pref/ShiftPref metrics.")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Where to write CSV/JSON. Default: run_dir_true/metrics")
    ap.add_argument("--topk", type=int, default=50,
                    help="Top-K cells used for localization metrics. Use 0 for 1%% of cells.")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Recoverability threshold: low + alpha*(high-low).")
    ap.add_argument("--kinds", type=str, default="none,mlp,attn",
                    help="Comma-separated kinds to analyze: none,mlp,attn")
    ap.add_argument("--max_cases", type=int, default=500,
                    help="Optional: only analyze known_id <= max_cases.")
    args = ap.parse_args()

    run_true = Path(args.run_dir_true)
    run_new = Path(args.run_dir_new) if args.run_dir_new else None

    out_dir = Path(args.out_dir) if args.out_dir else (run_true / "metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    assert all(k in ["none", "mlp", "attn"] for k in kinds), f"Invalid kinds: {kinds}"

    idx_true = index_cases(run_true)
    idx_new = index_cases(run_new) if run_new else {}

    rows: List[Dict[str, object]] = []

    # iterate keys in true run; optionally require key exists in new run for pref metrics
    for key, paths in sorted(idx_true.items(), key=lambda kv: (kv[0].kind, kv[0].known_id)):
        if key.kind not in kinds:
            continue
        if args.max_cases is not None and key.known_id > args.max_cases:
            continue

        base_path = paths["base"]
        edit_path = paths["edited"]
        base_npz = load_npz(base_path)
        edit_npz = load_npz(edit_path)

        base_expect = str(base_npz.get("expect", ""))
        edit_expect = str(edit_npz.get("expect", ""))
        # (Should be same expect inside a run.)
        expect_str = base_expect if base_expect else edit_expect

        r = {
            "known_id": key.known_id,
            "kind": key.kind,
            "expect": expect_str,
            "run_dir_true": str(run_true),
            "base_npz": str(base_path),
            "edited_npz": str(edit_path),
        }
        r.update(compute_pair_metrics(base_npz, edit_npz, topk=args.topk, alpha=args.alpha))

        # Optional: compute margin-like behavior if new run provided (p_true - p_new)
        if run_new is not None:
            # need matching key in new run too
            if key in idx_new:
                base_new_npz = load_npz(idx_new[key]["base"])
                edit_new_npz = load_npz(idx_new[key]["edited"])

                # store p_true/p_new and margins (log space is better; we do both)
                p_true_b = p_expect_from_run(base_npz)
                p_true_e = p_expect_from_run(edit_npz)
                p_new_b = p_expect_from_run(base_new_npz)
                p_new_e = p_expect_from_run(edit_new_npz)

                # avoid log(0)
                eps = 1e-12
                logm_b = math.log(max(p_true_b, eps)) - math.log(max(p_new_b, eps))
                logm_e = math.log(max(p_true_e, eps)) - math.log(max(p_new_e, eps))

                r.update({
                    "p_true_base": p_true_b,
                    "p_true_edited": p_true_e,
                    "p_new_base": p_new_b,
                    "p_new_edited": p_new_e,
                    "log_margin_true_minus_new_base": logm_b,
                    "log_margin_true_minus_new_edited": logm_e,
                    "delta_log_margin": float(logm_e - logm_b),
                    "run_dir_new": str(run_new),
                })

                # preference / shiftpref on causal maps
                r.update(compute_pref_metrics(
                    base_true=base_npz,
                    edit_true=edit_npz,
                    base_new=base_new_npz,
                    edit_new=edit_new_npz,
                    topk=args.topk,
                ))
            else:
                r["run_dir_new"] = str(run_new)
                r["pref_metrics_missing"] = True

        rows.append(r)

    df = pd.DataFrame(rows)

    # write one CSV per kind + overall
    all_csv = out_dir / "metrics_all.csv"
    df.to_csv(all_csv, index=False)

    for kind in kinds:
        dfi = df[df["kind"] == kind].copy()
        if len(dfi) == 0:
            continue
        dfi.to_csv(out_dir / f"metrics_{kind}.csv", index=False)

    # simple JSON summary
    summary = {}
    for kind in kinds:
        dfi = df[df["kind"] == kind]
        if len(dfi) == 0:
            continue
        summary[kind] = {
            "n": int(len(dfi)),
            "p_expect_base_mean": safe_float(dfi["p_expect_base"].mean()),
            "p_expect_edited_mean": safe_float(dfi["p_expect_edited"].mean()),
            "recoverable_ratio_base_mean": safe_float(dfi["recoverable_ratio_base"].mean()),
            "recoverable_ratio_edited_mean": safe_float(dfi["recoverable_ratio_edited"].mean()),
            "deltaS_absmean_mean": safe_float(dfi["deltaS_absmean"].mean()),
            "cosine_S_base_vs_S_edited_mean": safe_float(dfi["cosine_S_base_vs_S_edited"].mean()),
        }
        if "delta_log_margin" in dfi.columns:
            summary[kind].update({
                "delta_log_margin_mean": safe_float(dfi["delta_log_margin"].mean()),
                "pref_shift_pos_mass_mean": safe_float(dfi.get("pref_shift_pos_mass", pd.Series(dtype=float)).mean()),
                "pref_shift_subj_mean_mean": safe_float(dfi.get("pref_shift_subj_mean", pd.Series(dtype=float)).mean()),
                "pref_shift_non_subj_mean_mean": safe_float(dfi.get("pref_shift_non_subj_mean", pd.Series(dtype=float)).mean()),
            })

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Wrote:\n  {all_csv}\n  {out_dir / 'summary.json'}")
    print(f"[INFO] Rows: {len(df)}  Kinds: {kinds}")
    if run_new is None:
        print("[INFO] run_dir_new not provided -> Pref/ShiftPref and margin metrics are skipped.")


if __name__ == "__main__":
    main()
