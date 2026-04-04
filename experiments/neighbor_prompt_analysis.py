#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A. Base vs Edited comparison on NEIGHBOR prompts.
Inputs:
  - run_dir_true: folder like base_gpt2-xl__edited_...__expect_target_true
  - run_dir_new : folder like base_gpt2-xl__edited_...__expect_target_new
Each folder contains:
  cases/
    base_knowledge_{known_id}.npz
    base_knowledge_{known_id}_mlp.npz
    base_knowledge_{known_id}_attn.npz
    edited_knowledge_{known_id}.npz
    edited_knowledge_{known_id}_mlp.npz
    edited_knowledge_{known_id}_attn.npz

known_id encoding:
  - neighbor index = ones digit
  - case id = remaining higher digits

This script:
  - loads base+edited pairs for each known_id and kind
  - computes normalized recovery map:
        R = (scores - low) / (high - low)
  - projects to layer vector R_layer via token_strategy
  - computes structural comparison metrics between base vs edited:
      cosine similarity, peak layer shift, concentration (entropy/spread), etc.
  - writes per-pair CSV and aggregated JSON summary
"""

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


KIND_SUFFIX = {
    "none": "",
    "mlp": "_mlp",
    "attn": "_attn",
}

NPZ_PREFIX = ("base_", "edited_")


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def parse_known_id_from_filename(fname: str) -> Optional[int]:
    # matches base_knowledge_347_attn.npz or edited_knowledge_568.npz
    m = re.search(r"knowledge_(\d+)(?:_(?:mlp|attn))?\.npz$", fname)
    if not m:
        return None
    return int(m.group(1))


def split_case_neighbor(known_id: int) -> Tuple[int, int]:
    # user-defined encoding: ones digit = neighbor index
    return known_id // 10, known_id % 10


def load_npz(path: str) -> Dict:
    d = dict(np.load(path, allow_pickle=True))
    # unwrap 0-d arrays into python scalars where appropriate
    for k, v in list(d.items()):
        if isinstance(v, np.ndarray) and v.shape == ():
            d[k] = v.item()
    return d


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def entropy_from_nonneg(x: np.ndarray, eps: float = 1e-12) -> float:
    """
    x is nonnegative vector (or will be clipped).
    Normalize into a probability distribution and compute Shannon entropy.
    """
    x = np.clip(x.astype(np.float64), 0.0, None)
    s = float(x.sum())
    if s <= eps:
        return float("nan")
    p = x / s
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))


def spread_count(x: np.ndarray, alpha: float = 0.5) -> int:
    """
    Count layers whose value >= alpha * max(value).
    """
    x = x.astype(np.float64).reshape(-1)
    mx = float(np.nanmax(x))
    if not np.isfinite(mx) or mx <= 0:
        return 0
    thr = alpha * mx
    return int(np.sum(x >= thr))


def project_to_layer_vector(
    R: np.ndarray,
    subject_range: Tuple[int, int],
    strategy: str = "last",
) -> np.ndarray:
    """
    R: [ntoks, nlayers] recovery map
    subject_range: (s, e)
    strategy:
      - last: use token e-1
      - first: use token s
      - maxspan: per-layer max over tokens in [s,e)
      - meanspan: per-layer mean over tokens in [s,e)
    """
    s, e = int(subject_range[0]), int(subject_range[1])
    s = max(0, s)
    e = min(R.shape[0], e)
    if e <= s:
        # fallback: use token 0
        return R[0].copy()

    if strategy == "last":
        return R[e - 1].copy()
    if strategy == "first":
        return R[s].copy()
    if strategy == "maxspan":
        return np.max(R[s:e, :], axis=0)
    if strategy == "meanspan":
        return np.mean(R[s:e, :], axis=0)

    raise ValueError(f"Unknown token_strategy: {strategy}")


@dataclass
class PairMetrics:
    expect_field: str         # "target_true" or "target_new"
    kind: str                 # "none" | "mlp" | "attn"
    known_id: int
    case_id: int
    neighbor_id: int

    # baselines
    high_base: float
    low_base: float
    high_edited: float
    low_edited: float

    # recovery summaries
    peakR_base: float
    peakR_edited: float
    delta_peakR: float

    # structural similarity
    cosine_R: float
    peak_layer_base: int
    peak_layer_edited: int
    peak_layer_shift: int

    # concentration
    entropy_base: float
    entropy_edited: float
    spread_base: int
    spread_edited: int
    delta_entropy: float
    delta_spread: int


def compute_metrics_for_pair(
    base_npz: Dict,
    edited_npz: Dict,
    expect_field: str,
    kind: str,
    known_id: int,
    token_strategy: str,
    spread_alpha: float,
) -> Optional[PairMetrics]:
    # Required fields in your saved npz:
    # scores, low_score, high_score, subject_range
    for key in ("scores", "low_score", "high_score", "subject_range"):
        if key not in base_npz or key not in edited_npz:
            return None

    scores_b = np.array(base_npz["scores"], dtype=np.float64)
    scores_e = np.array(edited_npz["scores"], dtype=np.float64)

    low_b = safe_float(base_npz["low_score"])
    high_b = safe_float(base_npz["high_score"])
    low_e = safe_float(edited_npz["low_score"])
    high_e = safe_float(edited_npz["high_score"])

    denom_b = high_b - low_b
    denom_e = high_e - low_e
    if not np.isfinite(denom_b) or not np.isfinite(denom_e) or denom_b <= 0 or denom_e <= 0:
        # Can't normalize reliably
        return None

    R_b = (scores_b - low_b) / denom_b
    R_e = (scores_e - low_e) / denom_e

    subj_b = tuple(base_npz["subject_range"])
    subj_e = tuple(edited_npz["subject_range"])

    # In A, prompt is the same neighbor prompt, so subject_range should normally match.
    # But keep it robust: project using each one's own subject_range.
    Rl_b = project_to_layer_vector(R_b, subj_b, token_strategy)
    Rl_e = project_to_layer_vector(R_e, subj_e, token_strategy)

    # Some models can produce small negatives / >1 due to noise; clip for concentration stats
    Rl_b_clip = np.clip(Rl_b, 0.0, None)
    Rl_e_clip = np.clip(Rl_e, 0.0, None)

    peakR_b = float(np.nanmax(Rl_b))
    peakR_e = float(np.nanmax(Rl_e))
    peak_layer_b = int(np.nanargmax(Rl_b)) if np.isfinite(peakR_b) else -1
    peak_layer_e = int(np.nanargmax(Rl_e)) if np.isfinite(peakR_e) else -1
    shift = abs(peak_layer_b - peak_layer_e) if (peak_layer_b >= 0 and peak_layer_e >= 0) else -1

    cos = cosine_sim(Rl_b, Rl_e)

    ent_b = entropy_from_nonneg(Rl_b_clip)
    ent_e = entropy_from_nonneg(Rl_e_clip)
    spr_b = spread_count(Rl_b_clip, alpha=spread_alpha)
    spr_e = spread_count(Rl_e_clip, alpha=spread_alpha)

    case_id, neighbor_id = split_case_neighbor(known_id)

    return PairMetrics(
        expect_field=expect_field,
        kind=kind,
        known_id=known_id,
        case_id=case_id,
        neighbor_id=neighbor_id,
        high_base=high_b,
        low_base=low_b,
        high_edited=high_e,
        low_edited=low_e,
        peakR_base=peakR_b,
        peakR_edited=peakR_e,
        delta_peakR=float(peakR_e - peakR_b),
        cosine_R=cos,
        peak_layer_base=peak_layer_b,
        peak_layer_edited=peak_layer_e,
        peak_layer_shift=shift,
        entropy_base=ent_b,
        entropy_edited=ent_e,
        spread_base=spr_b,
        spread_edited=spr_e,
        delta_entropy=float(ent_e - ent_b) if (np.isfinite(ent_b) and np.isfinite(ent_e)) else float("nan"),
        delta_spread=int(spr_e - spr_b),
    )


def iter_pairs_in_cases_dir(cases_dir: str) -> Dict[Tuple[int, str], Tuple[str, str]]:
    """
    Return mapping:
      (known_id, kind) -> (base_path, edited_path)
    Only include pairs where both base and edited exist.
    """
    files = os.listdir(cases_dir)
    # index by (prefix, known_id, kind)
    idx: Dict[Tuple[str, int, str], str] = {}
    for fn in files:
        if not fn.endswith(".npz"):
            continue
        if not fn.startswith(NPZ_PREFIX):
            continue
        known_id = parse_known_id_from_filename(fn)
        if known_id is None:
            continue
        # kind inference
        kind = "none"
        if fn.endswith("_mlp.npz"):
            kind = "mlp"
        elif fn.endswith("_attn.npz"):
            kind = "attn"

        prefix = "base" if fn.startswith("base_") else "edited"
        idx[(prefix, known_id, kind)] = os.path.join(cases_dir, fn)

    pairs: Dict[Tuple[int, str], Tuple[str, str]] = {}
    for (prefix, known_id, kind), path in idx.items():
        if prefix != "base":
            continue
        p_edit = idx.get(("edited", known_id, kind))
        if p_edit:
            pairs[(known_id, kind)] = (path, p_edit)
    return pairs


def summarize_metrics(rows: List[PairMetrics]) -> Dict:
    """
    Aggregate by (expect_field, kind).
    """
    def stats(arr: List[float]) -> Dict:
        a = np.array([x for x in arr if np.isfinite(x)], dtype=np.float64)
        if a.size == 0:
            return {"n": 0}
        return {
            "n": int(a.size),
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "p10": float(np.quantile(a, 0.10)),
            "p90": float(np.quantile(a, 0.90)),
        }

    out: Dict[str, Dict[str, Dict]] = {}
    groups: Dict[Tuple[str, str], List[PairMetrics]] = {}
    for r in rows:
        groups.setdefault((r.expect_field, r.kind), []).append(r)

    for (expect_field, kind), rs in groups.items():
        key = f"{expect_field}:{kind}"
        out[key] = {
            "count": len(rs),
            "cosine_R": stats([x.cosine_R for x in rs]),
            "peak_layer_shift": stats([float(x.peak_layer_shift) for x in rs if x.peak_layer_shift >= 0]),
            "delta_peakR": stats([x.delta_peakR for x in rs]),
            "delta_entropy": stats([x.delta_entropy for x in rs]),
            "delta_spread": stats([float(x.delta_spread) for x in rs]),
            "peakR_base": stats([x.peakR_base for x in rs]),
            "peakR_edited": stats([x.peakR_edited for x in rs]),
        }
    return out


def write_csv(path: str, rows: List[PairMetrics]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "expect_field", "kind", "known_id", "case_id", "neighbor_id",
            "high_base", "low_base", "high_edited", "low_edited",
            "peakR_base", "peakR_edited", "delta_peakR",
            "cosine_R",
            "peak_layer_base", "peak_layer_edited", "peak_layer_shift",
            "entropy_base", "entropy_edited", "delta_entropy",
            "spread_base", "spread_edited", "delta_spread",
        ])
        for r in rows:
            w.writerow([
                r.expect_field, r.kind, r.known_id, r.case_id, r.neighbor_id,
                r.high_base, r.low_base, r.high_edited, r.low_edited,
                r.peakR_base, r.peakR_edited, r.delta_peakR,
                r.cosine_R,
                r.peak_layer_base, r.peak_layer_edited, r.peak_layer_shift,
                r.entropy_base, r.entropy_edited, r.delta_entropy,
                r.spread_base, r.spread_edited, r.delta_spread,
            ])


def main():
    ap = argparse.ArgumentParser(description="A: base vs edited on neighbor prompts (target_true / target_new)")
    ap.add_argument("--run_dir_true", required=True, help="Folder ...__expect_target_true (contains cases/)")
    ap.add_argument("--run_dir_new", required=True, help="Folder ...__expect_target_new (contains cases/)")
    ap.add_argument("--out_dir", required=True, help="Output directory for CSV/JSON")
    ap.add_argument("--token_strategy", default="last", choices=["last", "first", "maxspan", "meanspan"])
    ap.add_argument("--spread_alpha", type=float, default=0.5, help="spread threshold as alpha*max")
    args = ap.parse_args()

    inputs = [
        ("target_true", os.path.join(args.run_dir_true, "cases")),
        ("target_new", os.path.join(args.run_dir_new, "cases")),
    ]

    all_rows: List[PairMetrics] = []

    for expect_field, cases_dir in inputs:
        if not os.path.isdir(cases_dir):
            raise FileNotFoundError(f"cases/ not found: {cases_dir}")

        pairs = iter_pairs_in_cases_dir(cases_dir)

        for (known_id, kind), (p_base, p_edit) in sorted(pairs.items()):
            
            if known_id//10 > 50:
                # only process cases 0-50 (as in paper experiments)
                continue
            
            base_npz = load_npz(p_base)
            edit_npz = load_npz(p_edit)

            m = compute_metrics_for_pair(
                base_npz=base_npz,
                edited_npz=edit_npz,
                expect_field=expect_field,
                kind=kind,
                known_id=known_id,
                token_strategy=args.token_strategy,
                spread_alpha=args.spread_alpha,
            )
            if m is not None:
                all_rows.append(m)

    # write outputs
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "per_pair.csv")
    write_csv(csv_path, all_rows)

    summary = summarize_metrics(all_rows)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "token_strategy": args.token_strategy,
                "spread_alpha": args.spread_alpha,
                "n_pairs": len(all_rows),
                "groups": summary,
            },
            f,
            indent=2,
        )

    print(f"[OK] wrote: {csv_path}")
    print(f"[OK] wrote: {os.path.join(args.out_dir, 'summary.json')}")
    print(f"[OK] total pairs: {len(all_rows)}")


if __name__ == "__main__":
    main()
