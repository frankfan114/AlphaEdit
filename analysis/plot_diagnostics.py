"""
plot_diagnostics.py
-------------------
Read the step-level diagnostics JSONL produced by AlphaEdit with
--enable_diagnostics and generate a set of PNG plots for the
late-stage decay analysis.

Usage
-----
    python analysis/plot_diagnostics.py \
        --input  results/AlphaEdit/run_000/diagnostics_*.jsonl \
        --outdir plots/

Plots produced (all PNG)
------------------------
1.  conflict_sub_over_steps.png      – mean conflict_sub(t) across layers
2.  block_ratio_over_steps.png       – mean block_ratio(t) across layers
3.  edit_success_over_steps.png      – ES curve (only at eval steps)
4.  overlay_cs_br_es.png             – conflict_sub + block_ratio + ES on one figure
5.  delta_norms_over_steps.png       – raw / block / actual update Frobenius norms
6.  per_layer_conflict_sub.png       – conflict_sub per layer over steps
7.  per_layer_block_ratio.png        – block_ratio per layer over steps
8.  scatter_conflict_vs_es.png       – conflict_sub vs ES (eval steps only)
9.  scatter_block_vs_es.png          – block_ratio  vs ES (eval steps only)
10. heatmap_conflict_sub.png         – step × layer heatmap for conflict_sub
11. heatmap_block_ratio.png          – step × layer heatmap for block_ratio
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless / no display needed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> pd.DataFrame:
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_flat_df(records: list) -> pd.DataFrame:
    """
    Flatten JSONL records into a DataFrame.
    Per-layer fields become columns like conflict_sub_L13, block_ratio_L14, …
    """
    rows = []
    for rec in records:
        flat = {
            "step_id":        rec.get("step_id"),
            "edits_applied":  rec.get("edits_applied"),
            "edit_time_sec":  rec.get("edit_time_sec"),
            "mean_conflict_sub":              rec.get("mean_conflict_sub"),
            "mean_block_ratio":               rec.get("mean_block_ratio"),
            "mean_delta_raw_fro":             rec.get("mean_delta_raw_fro"),
            "mean_delta_block_fro":           rec.get("mean_delta_block_fro"),
            "mean_update_norm":               rec.get("mean_update_norm"),
            "mean_update_to_orig_norm_ratio": rec.get("mean_update_to_orig_norm_ratio"),
            "edit_success":          rec.get("edit_success"),
            "locality_preservation": rec.get("locality_preservation"),
            "paraphrase_success":    rec.get("paraphrase_success"),
            "first_token_logit_margin": rec.get("first_token_logit_margin"),
        }
        for lname, ldata in (rec.get("layers") or {}).items():
            if not isinstance(ldata, dict):
                continue
            for k, v in ldata.items():
                if isinstance(v, (int, float, type(None))):
                    flat[f"{k}_L{lname}"] = v
        rows.append(flat)
    return pd.DataFrame(rows)


def _savefig(fig, path: Path, title: str):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path.name}")


def _layer_cols(df: pd.DataFrame, prefix: str):
    """Return sorted column names matching prefix_L<num>."""
    cols = [c for c in df.columns if c.startswith(f"{prefix}_L")]
    cols.sort(key=lambda c: int(c.split("_L")[-1]))
    return cols


def _layer_labels(cols):
    return [c.split("_L")[-1] for c in cols]


def rolling(s: pd.Series, w: int = 5) -> pd.Series:
    return s.rolling(w, min_periods=1, center=True).mean()


# ── individual plot functions ─────────────────────────────────────────────────

def plot_single_metric(df, col, ylabel, title, outpath, color="steelblue", window=5):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = df["edits_applied"].values
    y = df[col].values
    ax.plot(x, y, alpha=0.35, color=color, lw=0.8)
    ax.plot(x, rolling(df[col], window).values, color=color, lw=2.0, label=f"rolling mean (w={window})")
    ax.set_xlabel("Edits applied")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(fig, outpath, title)


def plot_edit_success(df, outpath, window=5):
    es = df[df["edit_success"].notna()].copy()
    if es.empty:
        print("  [skip] no edit_success data available")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    x = es["edits_applied"].values
    y = es["edit_success"].values
    ax.plot(x, y, "o-", ms=4, alpha=0.6, color="darkorange", label="edit success (ES)")
    if len(y) >= window:
        ax.plot(x, rolling(pd.Series(y), window).values, lw=2.5,
                color="darkorange", label=f"rolling mean (w={window})")
    ax.set_xlabel("Edits applied"); ax.set_ylabel("Edit Success (ES)")
    ax.set_title("Edit Success over sequential edits")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, alpha=0.3)
    _savefig(fig, outpath, "ES")


def plot_overlay(df, outpath, window=5):
    """conflict_sub + block_ratio (left y) and ES (right y) on one figure."""
    es_df = df[df["edit_success"].notna()].copy()
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    x_all = df["edits_applied"].values
    colors = {"cs": "royalblue", "br": "crimson", "es": "darkorange"}

    def _plot_col(ax, col, label, color):
        vals = df[col].values
        ax.plot(x_all, vals, alpha=0.2, color=color, lw=0.8)
        ax.plot(x_all, rolling(df[col], window).values, color=color, lw=2.0, label=label)

    if df["mean_conflict_sub"].notna().any():
        _plot_col(ax1, "mean_conflict_sub", "conflict_sub (mean layers)", colors["cs"])
    if df["mean_block_ratio"].notna().any():
        _plot_col(ax1, "mean_block_ratio",  "block_ratio  (mean layers)", colors["br"])

    if not es_df.empty:
        ax2.plot(es_df["edits_applied"], es_df["edit_success"], "o",
                 ms=5, alpha=0.5, color=colors["es"])
        if len(es_df) >= window:
            ax2.plot(es_df["edits_applied"],
                     rolling(es_df["edit_success"], window).values,
                     lw=2.5, color=colors["es"], label="Edit Success (ES)")

    ax1.set_xlabel("Edits applied")
    ax1.set_ylabel("conflict_sub / block_ratio", color="black")
    ax2.set_ylabel("Edit Success (ES)", color=colors["es"])
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor=colors["es"])

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper right")
    ax1.set_title("conflict_sub  &  block_ratio  vs  Edit Success")
    ax1.grid(True, alpha=0.3)
    _savefig(fig, outpath, "overlay")


def plot_delta_norms(df, outpath, window=5):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = df["edits_applied"].values
    pairs = [
        ("mean_delta_raw_fro",   "Δ_raw ||·||_F",    "steelblue"),
        ("mean_delta_block_fro", "Δ_block ||·||_F",  "crimson"),
        ("mean_update_norm",     "Δ_actual ||·||_F", "seagreen"),
    ]
    for col, label, color in pairs:
        if col in df.columns and df[col].notna().any():
            ax.plot(x, df[col].values, alpha=0.2, color=color, lw=0.8)
            ax.plot(x, rolling(df[col], window).values, color=color, lw=2.0, label=label)
    ax.set_xlabel("Edits applied"); ax.set_ylabel("Frobenius norm")
    ax.set_title("Update Frobenius norms over sequential edits")
    ax.legend(); ax.grid(True, alpha=0.3)
    _savefig(fig, outpath, "delta norms")


def plot_per_layer(df, prefix, ylabel, title, outpath, window=5):
    cols = _layer_cols(df, prefix)
    if not cols:
        print(f"  [skip] no per-layer columns for {prefix}")
        return
    palette = cm.tab10(np.linspace(0, 1, len(cols)))
    fig, ax = plt.subplots(figsize=(12, 5))
    x = df["edits_applied"].values
    for col, color in zip(cols, palette):
        lbl = f"L{col.split('_L')[-1]}"
        vals = df[col].values
        ax.plot(x, vals, alpha=0.15, color=color, lw=0.6)
        ax.plot(x, rolling(df[col], window).values, color=color, lw=1.8, label=lbl)
    ax.set_xlabel("Edits applied"); ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=3, fontsize=8); ax.grid(True, alpha=0.3)
    _savefig(fig, outpath, title)


def plot_scatter(df, xcol, ycol, xlabel, ylabel, title, outpath):
    sub = df[[xcol, ycol, "edits_applied"]].dropna()
    if sub.empty:
        print(f"  [skip] {title}: no data")
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(sub[xcol], sub[ycol],
                    c=sub["edits_applied"], cmap="viridis",
                    s=40, alpha=0.7, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Edits applied")
    # regression line
    m, b = np.polyfit(sub[xcol].values, sub[ycol].values, 1)
    xs = np.linspace(sub[xcol].min(), sub[xcol].max(), 100)
    ax.plot(xs, m * xs + b, "r--", lw=1.5, label=f"slope={m:.3f}")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    _savefig(fig, outpath, title)


def plot_heatmap(df, prefix, title, outpath):
    cols = _layer_cols(df, prefix)
    if not cols:
        print(f"  [skip] heatmap for {prefix}: no columns")
        return
    mat = df[cols].values.T          # [layers, steps]
    x_ticks = df["edits_applied"].values
    fig, ax = plt.subplots(figsize=(14, max(3, len(cols) * 0.6)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                   extent=[x_ticks[0], x_ticks[-1], len(cols) - 0.5, -0.5],
                   vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(_layer_labels(cols))
    ax.set_xlabel("Edits applied")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    _savefig(fig, outpath, title)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot AlphaEdit diagnostics JSONL → PNG")
    parser.add_argument("--input",  required=True,
                        help="Path to diagnostics JSONL file")
    parser.add_argument("--outdir", default="plots",
                        help="Output directory for PNG files (default: plots/)")
    parser.add_argument("--rolling_window", type=int, default=5,
                        help="Window size for rolling mean smoothing (default: 5)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.input}")
    records = load_jsonl(args.input)
    print(f"  {len(records)} records loaded")

    df = build_flat_df(records)
    df = df.sort_values("step_id").reset_index(drop=True)
    print(f"  columns: {list(df.columns[:10])} …")

    w = args.rolling_window
    print(f"\nGenerating plots → {outdir}/")

    # 1. conflict_sub
    if df["mean_conflict_sub"].notna().any():
        plot_single_metric(df, "mean_conflict_sub",
                           "conflict_sub(t)", "Mean conflict_sub over sequential edits",
                           outdir / "conflict_sub_over_steps.png", color="royalblue", window=w)

    # 2. block_ratio
    if df["mean_block_ratio"].notna().any():
        plot_single_metric(df, "mean_block_ratio",
                           "block_ratio(t)", "Mean block_ratio over sequential edits",
                           outdir / "block_ratio_over_steps.png", color="crimson", window=w)

    # 3. edit success
    plot_edit_success(df, outdir / "edit_success_over_steps.png", window=w)

    # 4. overlay
    plot_overlay(df, outdir / "overlay_cs_br_es.png", window=w)

    # 5. delta norms
    plot_delta_norms(df, outdir / "delta_norms_over_steps.png", window=w)

    # 6. per-layer conflict_sub
    plot_per_layer(df, "conflict_sub",
                   "conflict_sub(t)", "Per-layer conflict_sub over steps",
                   outdir / "per_layer_conflict_sub.png", window=w)

    # 7. per-layer block_ratio
    plot_per_layer(df, "block_ratio",
                   "block_ratio(t)", "Per-layer block_ratio over steps",
                   outdir / "per_layer_block_ratio.png", window=w)

    # 8. scatter: conflict_sub vs ES
    plot_scatter(df, "mean_conflict_sub", "edit_success",
                 "conflict_sub", "Edit Success (ES)",
                 "conflict_sub vs Edit Success",
                 outdir / "scatter_conflict_vs_es.png")

    # 9. scatter: block_ratio vs ES
    plot_scatter(df, "mean_block_ratio", "edit_success",
                 "block_ratio", "Edit Success (ES)",
                 "block_ratio vs Edit Success",
                 outdir / "scatter_block_vs_es.png")

    # 10. heatmap conflict_sub
    plot_heatmap(df, "conflict_sub",
                 "conflict_sub(t) – step × layer",
                 outdir / "heatmap_conflict_sub.png")

    # 11. heatmap block_ratio
    plot_heatmap(df, "block_ratio",
                 "block_ratio(t) – step × layer",
                 outdir / "heatmap_block_ratio.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
