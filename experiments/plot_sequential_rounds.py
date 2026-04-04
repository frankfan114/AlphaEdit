import argparse
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    plt = None


DEFAULT_RESULTS_ROOT = Path("results") / "AlphaEdit"

COLORS = ["#0b7285", "#d9480f", "#2b8a3e", "#7b2cbf", "#c92a2a", "#1c7ed6"]

# ─── metric definitions ────────────────────────────────────────────────────────
# Each entry: (dotted_key, panel_title, y_label, section)
# dotted_key supports "layers.AVG.<field>" to average across all layers

METRICS_EDIT_QUALITY = [
    ("current_batch_eval.summary.edit_success",           "Current Batch Edit Success",       "Rate",  "Edit Quality"),
    ("current_batch_eval.summary.paraphrase_success",     "Current Batch Paraphrase Success", "Rate",  "Edit Quality"),
    ("current_batch_eval.summary.locality_preservation",  "Current Batch Locality",           "Rate",  "Edit Quality"),
    ("current_batch_eval.summary.first_token_logit_margin","Current Batch Logit Margin",      "Logit", "Edit Quality"),
    ("current_batch_eval.summary.new_first_token_logit",  "Current Batch New Token Logit",    "Logit", "Edit Quality"),
    ("current_batch_eval.summary.old_first_token_logit",  "Current Batch Old Token Logit",    "Logit", "Edit Quality"),
    ("current_batch_eval.summary.avg_logprob_margin",     "Current Batch LogProb Margin",     "LogP",  "Edit Quality"),
    ("current_batch_eval.summary.new_avg_logprob",        "Current Batch New LogProb",        "LogP",  "Edit Quality"),
    ("current_batch_eval.summary.old_avg_logprob",        "Current Batch Old LogProb",        "LogP",  "Edit Quality"),
    ("history_probe_eval.summary.edit_success",           "History Retention (Edit Success)", "Rate",  "Edit Quality"),
    ("history_probe_eval.summary.paraphrase_success",     "History Retention (Paraphrase)",   "Rate",  "Edit Quality"),
    ("history_probe_eval.summary.locality_preservation",  "History Locality Preservation",    "Rate",  "Edit Quality"),
    ("history_probe_eval.summary.first_token_logit_margin","History Logit Margin",            "Logit", "Edit Quality"),
    ("history_probe_eval.summary.new_first_token_logit",  "History New Token Logit",          "Logit", "Edit Quality"),
    ("history_probe_eval.summary.old_first_token_logit",  "History Old Token Logit",          "Logit", "Edit Quality"),
    ("history_probe_eval.summary.avg_logprob_margin",     "History LogProb Margin",           "LogP",  "Edit Quality"),
    ("history_probe_eval.summary.new_avg_logprob",        "History New LogProb",              "LogP",  "Edit Quality"),
    ("history_probe_eval.summary.old_avg_logprob",        "History Old LogProb",              "LogP",  "Edit Quality"),
]

METRICS_CONFLICT = [
    ("current_batch_eval.cases.CASES_AVG.conflict_metrics.first_token_logit_margin", "Current Batch Conflict Logit Margin",  "Logit", "Conflict"),
    ("current_batch_eval.cases.CASES_AVG.conflict_metrics.avg_logprob_margin",       "Current Batch Conflict LogProb Margin","LogP",  "Conflict"),
    ("current_batch_eval.cases.CASES_AVG.conflict_metrics.new_first_token_logit",    "Current Batch Conflict New Logit",     "Logit", "Conflict"),
    ("current_batch_eval.cases.CASES_AVG.conflict_metrics.old_first_token_logit",    "Current Batch Conflict Old Logit",     "Logit", "Conflict"),
    ("history_probe_eval.cases.CASES_AVG.conflict_metrics.first_token_logit_margin", "History Conflict Logit Margin",        "Logit", "Conflict"),
    ("history_probe_eval.cases.CASES_AVG.conflict_metrics.avg_logprob_margin",       "History Conflict LogProb Margin",      "LogP",  "Conflict"),
    ("history_probe_eval.cases.CASES_AVG.conflict_metrics.new_first_token_logit",    "History Conflict New Logit",           "Logit", "Conflict"),
    ("history_probe_eval.cases.CASES_AVG.conflict_metrics.old_first_token_logit",    "History Conflict Old Logit",           "Logit", "Conflict"),
]

METRICS_WEIGHT_AGGREGATE = [
    ("weight_metrics.aggregate.mean_frobenius_norm_drift", "Mean Frobenius Drift",        "Value", "Weight (Agg)"),
    ("weight_metrics.aggregate.max_spectral_norm",         "Max Spectral Norm",           "Value", "Weight (Agg)"),
    ("weight_metrics.aggregate.max_condition_number",      "Max Condition Number",        "Value", "Weight (Agg)"),
    ("weight_metrics.aggregate.mean_stable_rank",          "Mean Stable Rank",            "Value", "Weight (Agg)"),
]

METRICS_WEIGHT_PER_LAYER = [
    ("weight_metrics.layers.AVG.frobenius_norm",              "Avg Frobenius Norm (Absolute)", "Value", "Weight (Layer)"),
    ("weight_metrics.layers.AVG.frobenius_norm_drift",        "Avg Frobenius Drift",           "Value", "Weight (Layer)"),
    ("weight_metrics.layers.AVG.relative_frobenius_drift",    "Avg Relative Frobenius Drift",  "Ratio", "Weight (Layer)"),
    ("weight_metrics.layers.AVG.spectral_norm",               "Avg Spectral Norm",             "Value", "Weight (Layer)"),
    ("weight_metrics.layers.AVG.spectral_norm_ratio_to_reference","Avg Spectral Norm Ratio to Init","Ratio","Weight (Layer)"),
    ("weight_metrics.layers.AVG.min_nonzero_singular_value",  "Avg Min Singular Value",        "Value", "Weight (Layer)"),
    ("weight_metrics.layers.AVG.condition_number",            "Avg Condition Number",          "Value", "Weight (Layer)"),
    ("weight_metrics.layers.AVG.condition_number_ratio_to_reference","Avg Cond# Ratio to Init","Ratio","Weight (Layer)"),
    ("weight_metrics.layers.AVG.stable_rank",                 "Avg Stable Rank",               "Value", "Weight (Layer)"),
    ("weight_metrics.layers.AVG.singular_values_topk.0",      "Avg Top-1 Singular Value",      "Value", "Weight (Layer)"),
    ("weight_metrics.layers.AVG.singular_values_topk.1",      "Avg Top-2 Singular Value",      "Value", "Weight (Layer)"),
    ("weight_metrics.layers.AVG.singular_values_bottomk.0",   "Avg Bottom-1 Singular Value",   "Value", "Weight (Layer)"),
    ("weight_metrics.layers.AVG.singular_value_quantiles.p25","Avg Singular Value p25",        "Value", "Weight (Layer)"),
    ("weight_metrics.layers.AVG.singular_value_quantiles.p50","Avg Singular Value p50",        "Value", "Weight (Layer)"),
    ("weight_metrics.layers.AVG.singular_value_quantiles.p75","Avg Singular Value p75",        "Value", "Weight (Layer)"),
    ("weight_metrics.layers.AVG.singular_value_quantiles.p90","Avg Singular Value p90",        "Value", "Weight (Layer)"),
    ("weight_metrics.layers.AVG.singular_value_quantiles.p99","Avg Singular Value p99",        "Value", "Weight (Layer)"),
]

METRICS_UPDATE = [
    ("update_diagnostics.layers.AVG.orig_norm",                "Avg Original Weight Norm",      "Value", "Update"),
    ("update_diagnostics.layers.AVG.update_norm",              "Avg Update Norm",               "Value", "Update"),
    ("update_diagnostics.layers.AVG.update_to_orig_norm_ratio","Avg Update/Orig Norm Ratio",    "Ratio", "Update"),
    ("update_diagnostics.layers.AVG.pre_projection_delta_norm","Avg Pre-Projection Delta Norm", "Value", "Update"),
    ("update_diagnostics.layers.AVG.post_projection_delta_norm","Avg Post-Projection Delta Norm","Value","Update"),
]

METRICS_TIMING = [
    ("edit_time", "Edit Time per Round", "Seconds", "Timing"),
]

ALL_METRICS = (
    METRICS_EDIT_QUALITY
    + METRICS_CONFLICT
    + METRICS_WEIGHT_AGGREGATE
    + METRICS_WEIGHT_PER_LAYER
    + METRICS_UPDATE
    + METRICS_TIMING
)

SECTIONS = [
    ("Edit Quality",   METRICS_EDIT_QUALITY),
    ("Conflict",       METRICS_CONFLICT),
    ("Weight (Agg)",   METRICS_WEIGHT_AGGREGATE),
    ("Weight (Layer)", METRICS_WEIGHT_PER_LAYER),
    ("Update",         METRICS_UPDATE),
    ("Timing",         METRICS_TIMING),
]

# ─── focused single-topic plots ────────────────────────────────────────────────
FOCUS_GROUPS = [
    ("focus_history_locality", "History Retention & Locality", [
        ("current_batch_eval.summary.edit_success",                    "Current Batch Edit Success",        "Rate"),
        ("history_probe_eval.summary.edit_success",                    "History Retention (Random)",        "Rate"),
        ("history_probe_oldest_eval.summary.edit_success",             "History Retention (Oldest)",        "Rate"),
        ("history_probe_recent_eval.summary.edit_success",             "History Retention (Recent)",        "Rate"),
        ("current_batch_eval.summary.locality_preservation",           "Current Batch Locality",            "Rate"),
        ("history_probe_eval.summary.locality_preservation",           "History Locality (Random)",         "Rate"),
        ("history_probe_oldest_eval.summary.locality_preservation",    "History Locality (Oldest)",         "Rate"),
        ("history_probe_recent_eval.summary.locality_preservation",    "History Locality (Recent)",         "Rate"),
    ]),
    ("focus_forgetting_gradient", "Forgetting by Edit Age (Stratified)", [
        ("history_probe_stratified_eval.q1_oldest.summary.edit_success",  "Retention Q1 (Oldest 25%)",     "Rate"),
        ("history_probe_stratified_eval.q2.summary.edit_success",         "Retention Q2",                  "Rate"),
        ("history_probe_stratified_eval.q3.summary.edit_success",         "Retention Q3",                  "Rate"),
        ("history_probe_stratified_eval.q4_recent.summary.edit_success",  "Retention Q4 (Recent 25%)",     "Rate"),
        ("history_probe_oldest_eval.summary.edit_success",                "Retention Oldest N",            "Rate"),
        ("history_probe_recent_eval.summary.edit_success",                "Retention Recent N",            "Rate"),
    ]),
    ("focus_retention_four_probes", "History Retention — All Four Sampling Methods", [
        ("history_probe_eval.summary.edit_success",                        "Random Uniform",                "Rate"),
        ("history_probe_oldest_eval.summary.edit_success",                 "Oldest N",                     "Rate"),
        ("history_probe_recent_eval.summary.edit_success",                 "Recent N",                     "Rate"),
        ("history_probe_stratified_eval.q1_oldest.summary.edit_success",   "Stratified Q1 (Oldest 25%)",   "Rate"),
        ("history_probe_stratified_eval.q2.summary.edit_success",          "Stratified Q2",                "Rate"),
        ("history_probe_stratified_eval.q3.summary.edit_success",          "Stratified Q3",                "Rate"),
        ("history_probe_stratified_eval.q4_recent.summary.edit_success",   "Stratified Q4 (Recent 25%)",   "Rate"),
        ("current_batch_eval.summary.edit_success",                        "Current Batch (reference)",    "Rate"),
    ]),
    ("focus_frobenius_drift", "Frobenius Drift", [
        ("weight_metrics.aggregate.mean_frobenius_norm_drift",       "Mean Frobenius Drift (Agg)",        "Value"),
        ("weight_metrics.layers.AVG.frobenius_norm_drift",           "Avg Frobenius Drift (Per Layer)",   "Value"),
        ("weight_metrics.layers.AVG.relative_frobenius_drift",       "Avg Relative Frobenius Drift",      "Ratio"),
    ]),
    ("focus_condition_number", "Condition Number", [
        ("weight_metrics.aggregate.max_condition_number",            "Max Condition Number (Agg)",        "Value"),
        ("weight_metrics.layers.AVG.condition_number",               "Avg Condition Number (Per Layer)",  "Value"),
        ("weight_metrics.layers.AVG.condition_number_ratio_to_reference", "Avg Cond# Ratio to Init",     "Ratio"),
    ]),
    ("focus_stable_rank_singular", "Stable Rank & Singular Values", [
        ("weight_metrics.aggregate.mean_stable_rank",                "Mean Stable Rank (Agg)",            "Value"),
        ("weight_metrics.layers.AVG.stable_rank",                    "Avg Stable Rank (Per Layer)",       "Value"),
        ("weight_metrics.layers.AVG.singular_values_topk.0",         "Avg σ_max (Top-1)",                 "Value"),
        ("weight_metrics.layers.AVG.singular_values_topk.1",         "Avg σ Top-2",                      "Value"),
        ("weight_metrics.layers.AVG.singular_values_bottomk.0",      "Avg σ_min (Bottom-1)",              "Value"),
        ("weight_metrics.layers.AVG.singular_value_quantiles.p50",   "Avg σ Median (p50)",               "Value"),
    ]),
]


# ─── helpers ───────────────────────────────────────────────────────────────────

def resolve_run_path(run_arg: str, results_root: Path) -> Path:
    run_path = Path(run_arg)
    if run_path.exists():
        return run_path.resolve()
    return (results_root / run_arg).resolve()


def _get_nested(obj, parts):
    cur = obj
    for part in parts:
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        else:
            return None
    return cur


def resolve_metric(payload, dotted_key):
    """
    Resolve a dotted key from a round payload.
    Supports 'weight_metrics.layers.AVG.<field>' to average across layers.
    Supports 'update_diagnostics.layers.AVG.<field>' similarly.
    Supports '<section>.cases.CASES_AVG.<rest>' to average across per-case entries.
    Supports list indexing via integer parts (e.g. singular_values_topk.0).
    """
    parts = dotted_key.split(".")

    # detect CASES_AVG pattern: <prefix>.cases.CASES_AVG.<rest>
    try:
        cases_avg_idx = parts.index("CASES_AVG")
    except ValueError:
        cases_avg_idx = -1

    if cases_avg_idx >= 0:
        prefix_parts = parts[:cases_avg_idx]   # e.g. ['current_batch_eval', 'cases']
        rest_parts   = parts[cases_avg_idx + 1:]
        container = _get_nested(payload, prefix_parts)
        if not isinstance(container, list):
            return None
        values = []
        for case_obj in container:
            v = _get_nested(case_obj, rest_parts)
            if v is not None:
                try:
                    values.append(float(v))
                except (TypeError, ValueError):
                    pass
        if not values:
            return None
        return sum(values) / len(values)

    # detect AVG pattern: <prefix>.layers.AVG.<rest>
    try:
        avg_idx = parts.index("AVG")
    except ValueError:
        avg_idx = -1

    if avg_idx >= 0:
        prefix_parts = parts[:avg_idx]      # e.g. ['weight_metrics', 'layers']
        rest_parts   = parts[avg_idx + 1:]  # e.g. ['condition_number']
        container = _get_nested(payload, prefix_parts)
        if not isinstance(container, dict):
            return None
        values = []
        for layer_obj in container.values():
            v = _get_nested(layer_obj, rest_parts)
            if v is not None:
                try:
                    values.append(float(v))
                except (TypeError, ValueError):
                    pass
        if not values:
            return None
        return sum(values) / len(values)

    # top-level simple key
    if len(parts) == 1:
        v = payload.get(parts[0])
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return _get_nested(payload, parts)


def load_rounds(run_path: Path):
    round_dir = run_path / "sequential_rounds"
    if not round_dir.exists():
        return []

    rows = []
    for json_file in sorted(round_dir.glob("round_*.json")):
        try:
            with open(json_file, "r") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        edits_applied = payload.get("edits_applied")
        if edits_applied is None:
            continue

        row = {"edits_applied": float(edits_applied), "_payload": payload}
        rows.append(row)

    return rows


def metric_series(rows, dotted_key):
    series = []
    for row in rows:
        v = resolve_metric(row["_payload"], dotted_key)
        if v is None:
            continue
        try:
            y = float(v)
            x = row["edits_applied"]
        except (TypeError, ValueError):
            continue
        if math.isnan(y) or math.isinf(y):
            continue
        series.append((x, y))
    return series


def format_tick(value):
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


# ─── matplotlib plotting ───────────────────────────────────────────────────────

def plot_section_matplotlib(run_paths, labels, metrics, section_title, output_path: Path):
    n = len(metrics)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), squeeze=False)

    for i, (metric_key, metric_title, y_label, _) in enumerate(metrics):
        ax = axes[i // ncols][i % ncols]
        for run_path, label, color in zip(run_paths, labels, COLORS):
            rows = load_rounds(run_path)
            series = metric_series(rows, metric_key)
            if not series:
                continue
            x_values = [x for x, _ in series]
            y_values = [y for _, y in series]
            ax.plot(x_values, y_values, marker="o", linewidth=2, label=label, color=color)
        ax.set_title(metric_title, fontsize=10)
        ax.set_xlabel("Edits Applied", fontsize=8)
        ax.set_ylabel(y_label, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # hide unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="upper center",
                   ncol=max(1, len(labels)), fontsize=10)

    fig.suptitle(f"AlphaEdit — {section_title}", fontsize=14, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_focused_matplotlib(run_paths, labels, slug, title, metrics, output_dir: Path):
    """One large figure per focus group; metrics is list of (key, panel_title, y_label)."""
    n = len(metrics)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5), squeeze=False)

    for i, (metric_key, metric_title, y_label) in enumerate(metrics):
        ax = axes[i // ncols][i % ncols]
        for run_path, label, color in zip(run_paths, labels, COLORS):
            rows = load_rounds(run_path)
            series = metric_series(rows, metric_key)
            if not series:
                continue
            x_values = [x for x, _ in series]
            y_values = [y for _, y in series]
            ax.plot(x_values, y_values, marker="o", linewidth=2.5, markersize=5,
                    label=label, color=color)
        ax.set_title(metric_title, fontsize=12)
        ax.set_xlabel("Edits Applied", fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="upper center",
                   ncol=max(1, len(labels)), fontsize=11)

    fig.suptitle(f"AlphaEdit — {title}", fontsize=16, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = output_dir / f"{slug}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_all_matplotlib(run_paths, labels, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for section_name, metrics in SECTIONS:
        slug = section_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        out = output_dir / f"plot_{slug}.png"
        plot_section_matplotlib(run_paths, labels, metrics, section_name, out)

    # combined overview: one panel per section using aggregate metric
    overview_metrics = [
        ("current_batch_eval.summary.edit_success",           "Current Batch Edit Success",    "Rate",  ""),
        ("history_probe_eval.summary.edit_success",           "History Retention",             "Rate",  ""),
        ("history_probe_eval.summary.locality_preservation",  "History Locality",              "Rate",  ""),
        ("weight_metrics.aggregate.mean_frobenius_norm_drift","Mean Frobenius Drift",          "Value", ""),
        ("weight_metrics.aggregate.max_condition_number",     "Max Condition Number",          "Value", ""),
        ("weight_metrics.layers.AVG.spectral_norm_ratio_to_reference","Avg Spectral Norm Ratio","Ratio",""),
        ("update_diagnostics.layers.AVG.update_to_orig_norm_ratio","Avg Update/Orig Ratio",    "Ratio", ""),
        ("weight_metrics.aggregate.mean_stable_rank",         "Mean Stable Rank",              "Value", ""),
    ]
    plot_section_matplotlib(run_paths, labels, overview_metrics, "Overview", output_dir / "plot_overview.png")

    for slug, title, metrics in FOCUS_GROUPS:
        plot_focused_matplotlib(run_paths, labels, slug, title, metrics, output_dir)


# ─── SVG fallback ──────────────────────────────────────────────────────────────

def _svg_panel(lines, left, top, plot_w, plot_h, series_list, labels,
               metric_title, y_label, text_color, axis_color, grid_color):
    right  = left + plot_w
    bottom = top  + plot_h

    all_x = [x for s in series_list for x, _ in s]
    all_y = [y for s in series_list for _, y in s]
    x_min, x_max = (min(all_x), max(all_x)) if all_x else (0.0, 1.0)
    if x_min == x_max:
        x_min -= 1.0; x_max += 1.0
    y_min, y_max = (min(all_y), max(all_y)) if all_y else (0.0, 1.0)
    if y_min == y_max:
        pad = 0.1 if y_min == 0 else abs(y_min) * 0.1
        y_min -= pad; y_max += pad
    else:
        pad = (y_max - y_min) * 0.08
        y_min -= pad; y_max += pad

    lines.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#ced4da" stroke-width="1" rx="6" />')
    lines.append(f'<text x="{left + plot_w/2}" y="{top-8}" text-anchor="middle" font-size="11" font-family="Arial,sans-serif" fill="{text_color}">{escape(metric_title)}</text>')

    def mx(v): return left + (v - x_min) / (x_max - x_min) * plot_w
    def my(v): return bottom - (v - y_min) / (y_max - y_min) * plot_h

    for ti in range(5):
        frac = ti / 4.0
        gy = bottom - frac * plot_h
        tv = y_min + frac * (y_max - y_min)
        lines.append(f'<line x1="{left}" y1="{gy}" x2="{right}" y2="{gy}" stroke="{grid_color}" stroke-width="1" />')
        lines.append(f'<text x="{left-6}" y="{gy+4}" text-anchor="end" font-size="9" font-family="Arial,sans-serif" fill="{axis_color}">{format_tick(tv)}</text>')
    for ti in range(5):
        frac = ti / 4.0
        gx = left + frac * plot_w
        tv = x_min + frac * (x_max - x_min)
        lines.append(f'<line x1="{gx}" y1="{top}" x2="{gx}" y2="{bottom}" stroke="{grid_color}" stroke-width="1" />')
        lines.append(f'<text x="{gx}" y="{bottom+14}" text-anchor="middle" font-size="9" font-family="Arial,sans-serif" fill="{axis_color}">{format_tick(tv)}</text>')

    lines.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{axis_color}" stroke-width="1.5" />')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="{axis_color}" stroke-width="1.5" />')

    for si, series in enumerate(series_list):
        if not series:
            continue
        color = COLORS[si % len(COLORS)]
        pts = " ".join(f"{mx(x):.1f},{my(y):.1f}" for x, y in series)
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}" />')
        for xv, yv in series:
            lines.append(f'<circle cx="{mx(xv):.1f}" cy="{my(yv):.1f}" r="3" fill="{color}" />')


def make_svg_section(run_paths, labels, metrics, section_title, output_path: Path):
    ncols = 4
    nrows = math.ceil(len(metrics) / ncols)
    pw, ph = 280, 160
    margin_l, margin_t = 60, 30
    gap_x, gap_y = 80, 60
    leg_h = 40
    title_h = 50

    total_w = margin_l + ncols * pw + (ncols - 1) * gap_x + 40
    total_h = title_h + leg_h + nrows * (ph + gap_y) + 20

    bg = "#f8f9fa"; axis_color = "#495057"; grid_color = "#dee2e6"; text_color = "#212529"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" viewBox="0 0 {total_w} {total_h}">',
        f'<rect x="0" y="0" width="{total_w}" height="{total_h}" fill="{bg}" />',
        f'<text x="{total_w/2}" y="32" text-anchor="middle" font-size="18" font-family="Arial,sans-serif" fill="{text_color}">AlphaEdit — {escape(section_title)}</text>',
    ]

    # legend
    lx = 60
    for idx, label in enumerate(labels):
        color = COLORS[idx % len(COLORS)]
        x0 = lx + idx * 220
        lines.append(f'<line x1="{x0}" y1="{title_h+16}" x2="{x0+24}" y2="{title_h+16}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<circle cx="{x0+12}" cy="{title_h+16}" r="4" fill="{color}"/>')
        lines.append(f'<text x="{x0+32}" y="{title_h+21}" font-size="12" font-family="Arial,sans-serif" fill="{text_color}">{escape(label)}</text>')

    for i, (metric_key, metric_title, y_label, _) in enumerate(metrics):
        col = i % ncols
        row = i // ncols
        left = margin_l + col * (pw + gap_x)
        top  = title_h + leg_h + row * (ph + gap_y)

        series_list = []
        for run_path in run_paths:
            rows = load_rounds(run_path)
            series_list.append(metric_series(rows, metric_key))

        _svg_panel(lines, left, top, pw, ph, series_list, labels,
                   metric_title, y_label, text_color, axis_color, grid_color)

    lines.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {output_path}")


def make_all_svg(run_paths, labels, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for section_name, metrics in SECTIONS:
        slug = section_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        out = output_dir / f"plot_{slug}.svg"
        make_svg_section(run_paths, labels, metrics, section_name, out)


# ─── entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run name like run_029 or a direct path. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--label",
        action="append",
        help="Optional labels for each run, in the same order as --run.",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=str(DEFAULT_RESULTS_ROOT),
        help="Base directory used when --run is not an existing path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save all plots. Defaults to results/AlphaEdit/plots/.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="(Legacy) single output file path — kept for backwards compatibility.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    run_paths = [resolve_run_path(r, results_root) for r in args.run]

    if args.label is not None and len(args.label) != len(run_paths):
        raise ValueError("Number of --label values must match number of --run values.")
    labels = args.label if args.label is not None else [p.name for p in run_paths]

    # determine output directory — auto-name from run names if not specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.output:
        output_dir = Path(args.output).parent / "plots"
    else:
        run_names = [p.name for p in run_paths]  # e.g. ['run_029', 'run_031']
        folder_name = "plots_" + "_vs_".join(run_names)  # plots_run029 or plots_run029_vs_run031
        output_dir = results_root / folder_name

    if plt is not None:
        plot_all_matplotlib(run_paths, labels, output_dir)
    else:
        make_all_svg(run_paths, labels, output_dir)


if __name__ == "__main__":
    main()
