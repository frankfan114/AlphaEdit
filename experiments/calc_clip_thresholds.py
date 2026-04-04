"""
Calculate recommended delta-clipping thresholds from an ongoing / completed run.

Uses sequential_rounds/*.json to compute:
  - tau_F candidates  (Frobenius-norm clip threshold for ΔW)
  - tau_2 candidates  (Spectral-norm clip threshold for ΔW)

Exact values are used when available (runs using the updated AlphaEdit_main.py
which logs delta_fro, delta_spectral, delta_stable_rank directly).

For older runs that only have update_norm, spectral norm is approximated as:
    ||ΔW||_2 ≈ ||ΔW||_F / sqrt(stable_rank(W'))
This is flagged clearly in the output.

Thresholds are reported both pooled (all layers) and per-layer.

Usage:
    python3 -m experiments.calc_clip_thresholds --run run_032
    python3 -m experiments.calc_clip_thresholds --run run_032 --percentile 75
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

DEFAULT_RESULTS_ROOT = Path("results") / "AlphaEdit"


def load_rounds(run_path: Path):
    round_dir = run_path / "sequential_rounds"
    rows = []
    for f in sorted(round_dir.glob("round_*.json")):
        try:
            with open(f) as fp:
                d = json.load(fp)
        except Exception:
            continue
        if d.get("round_idx", 0) == 0:
            continue
        rows.append(d)
    return rows


def collect_stats(rows):
    """
    Returns per-layer-per-round stats.
    Uses exact delta_fro / delta_spectral when available (new runs),
    falls back to approximation for old runs.
    """
    records = []
    has_exact = False

    for row in rows:
        ud = row.get("update_diagnostics", {}).get("layers", {})
        wm = row.get("weight_metrics", {}).get("layers", {})
        edits = row.get("edits_applied")

        for layer in ud:
            orig_norm   = ud[layer].get("orig_norm")
            update_norm = ud[layer].get("update_norm")
            if orig_norm is None or update_norm is None:
                continue

            # Prefer exact values logged by updated AlphaEdit_main.py
            delta_fro      = ud[layer].get("delta_fro")
            delta_spectral = ud[layer].get("delta_spectral")
            delta_sr       = ud[layer].get("delta_stable_rank")
            exact = delta_fro is not None and delta_spectral is not None

            if exact:
                has_exact = True
            else:
                # Fallback approximation using W' stable_rank
                delta_fro = update_norm
                stable_rank_w = wm.get(layer, {}).get("stable_rank")
                if stable_rank_w and stable_rank_w > 0:
                    delta_spectral = delta_fro / math.sqrt(stable_rank_w)
                    delta_sr = stable_rank_w  # approximation
                else:
                    delta_spectral = None
                    delta_sr = None

            records.append({
                "edits_applied": edits,
                "layer": str(layer),
                "orig_norm": orig_norm,
                "delta_fro": delta_fro,
                "delta_spectral": delta_spectral,
                "delta_stable_rank": delta_sr,
                "fro_ratio": delta_fro / orig_norm if orig_norm > 0 else None,
                "exact": exact,
            })

    return records, has_exact


def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    idx = (len(s) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def print_dist(label, values, p):
    if not values:
        print(f"  {label}: no data")
        return
    print(f"  {label}")
    print(f"    min={min(values):.4f}  mean={sum(values)/len(values):.4f}  "
          f"p50={percentile(values,50):.4f}  p{p:.0f}={percentile(values,p):.4f}  "
          f"p90={percentile(values,90):.4f}  max={max(values):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True,
                        help="Run name (e.g. run_032) or full path.")
    parser.add_argument("--results_root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--percentile", type=float, default=75,
                        help="Recommend thresholds at this percentile (default 75).")
    args = parser.parse_args()

    run_path = Path(args.run)
    if not run_path.exists():
        run_path = Path(args.results_root) / args.run
    if not run_path.exists():
        raise FileNotFoundError(f"Run not found: {run_path}")

    rows = load_rounds(run_path)
    if not rows:
        print("No round files found (index > 0).")
        return

    records, has_exact = collect_stats(rows)
    p = args.percentile

    exact_label = "EXACT (logged by AlphaEdit_main)" if has_exact else "APPROXIMATED (||ΔW||_2 ≈ ||ΔW||_F / sqrt(stable_rank(W')))"
    print(f"\nLoaded {len(rows)} rounds, {len(records)} layer-round records from {run_path.name}")
    print(f"Spectral norm source: {exact_label}\n")

    # ── Pooled distributions ────────────────────────────────────────────────
    all_fro      = [r["delta_fro"]      for r in records]
    all_spectral = [r["delta_spectral"] for r in records if r["delta_spectral"] is not None]
    all_sr       = [r["delta_stable_rank"] for r in records if r["delta_stable_rank"] is not None]
    all_ratio    = [r["fro_ratio"]      for r in records if r["fro_ratio"] is not None]
    orig_norms   = [r["orig_norm"]      for r in records]

    print("=" * 65)
    print("  POOLED  (all layers combined)")
    print("=" * 65)
    print_dist("||ΔW||_F", all_fro, p)
    print_dist("||ΔW||_2", all_spectral, p)
    print_dist("stable_rank(ΔW)", all_sr, p)
    print_dist("||ΔW||_F / ||W₀||_F", all_ratio, p)

    mean_orig = sum(orig_norms) / len(orig_norms)
    tau_f_abs   = percentile(all_fro, p)
    tau_f_ratio = tau_f_abs / mean_orig
    tau_2_pool  = percentile(all_spectral, p)

    print()
    print(f"  Pooled recommendations (p{p:.0f}):")
    print(f"    --delta_fro_ratio   = {tau_f_ratio:.3f}  (tau_F = {tau_f_abs:.2f})")
    print(f"    --delta_spectral_tau = {tau_2_pool:.3f}")

    # ── Per-layer distributions ─────────────────────────────────────────────
    by_layer = defaultdict(list)
    for r in records:
        by_layer[r["layer"]].append(r)

    print()
    print("=" * 65)
    print("  PER-LAYER")
    print("=" * 65)
    layer_tau2 = {}
    for layer in sorted(by_layer.keys(), key=lambda x: int(x)):
        recs = by_layer[layer]
        fro_vals      = [r["delta_fro"]      for r in recs]
        spec_vals     = [r["delta_spectral"] for r in recs if r["delta_spectral"] is not None]
        orig_vals     = [r["orig_norm"]      for r in recs]
        mean_orig_l   = sum(orig_vals) / len(orig_vals)

        tau_f_l   = percentile(fro_vals, p)
        tau_2_l   = percentile(spec_vals, p)
        ratio_l   = tau_f_l / mean_orig_l if mean_orig_l > 0 else None
        layer_tau2[layer] = tau_2_l

        print(f"\n  Layer {layer}  (||W₀||_F mean = {mean_orig_l:.2f})")
        print_dist("  ||ΔW||_F", fro_vals, p)
        print_dist("  ||ΔW||_2", spec_vals, p)
        print(f"    → delta_fro_ratio={ratio_l:.3f}  delta_spectral_tau={tau_2_l:.3f}")

    # ── Final recommendation ────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  FINAL RECOMMENDED THRESHOLDS")
    print(f"  (p{p:.0f} of {len(rows)} rounds, source: {exact_label})")
    print("=" * 65)
    print()
    print(f"  Frobenius clip  --delta_fro_ratio    = {tau_f_ratio:.3f}")
    print(f"  Spectral  clip  --delta_spectral_tau = {tau_2_pool:.3f}  (pooled)")
    print()
    print("  Per-layer spectral tau:")
    for layer in sorted(layer_tau2.keys(), key=lambda x: int(x)):
        print(f"    Layer {layer}: {layer_tau2[layer]:.3f}")
    print()
    if not has_exact:
        print("  ⚠  Spectral values are approximated. Re-run after submitting a job")
        print("     with the updated AlphaEdit_main.py to get exact per-layer tau_2.")
    print("=" * 65)


if __name__ == "__main__":
    main()
