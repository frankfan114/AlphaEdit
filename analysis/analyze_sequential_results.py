import argparse
import json
from pathlib import Path


DEFAULT_RESULTS_ROOT = Path("results") / "AlphaEdit"

METRICS = [
    "immediate_current_batch_edit_success",
    "current_batch_eval.summary.edit_success",
    "history_probe_eval.summary.edit_success",
    "update_diagnostics.aggregate.mean_update_norm",
    "update_diagnostics.aggregate.mean_update_to_orig_norm_ratio",
    "update_diagnostics.aggregate.mean_pre_projection_delta_norm",
    "update_diagnostics.aggregate.mean_post_projection_delta_norm",
    "update_diagnostics.aggregate.projection_applied_layers",
    "weight_metrics.aggregate.max_condition_number",
    "weight_metrics.aggregate.mean_frobenius_norm_drift",
]


def resolve_run_path(run_arg, results_root):
    run_path = Path(run_arg)
    if run_path.exists():
        return run_path.resolve()
    return (results_root / run_arg).resolve()


def nested_get(obj, dotted_key):
    cur = obj
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def format_value(value):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == 0:
            return "0"
        abs_value = abs(value)
        if abs_value >= 10000 or abs_value < 1e-3:
            return "{:.3e}".format(value)
        return "{:.4f}".format(value)
    return str(value)


def try_float(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def load_rounds(run_path):
    round_dir = run_path / "sequential_rounds"
    rows = {}
    if not round_dir.exists():
        return rows

    for json_file in sorted(round_dir.glob("round_*.json")):
        try:
            with open(json_file, "r") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        round_idx = payload.get("round_idx")
        if round_idx is None:
            continue
        row = {
            "round_idx": round_idx,
            "edits_applied": payload.get("edits_applied"),
        }
        for metric in METRICS:
            row[metric] = nested_get(payload, metric)
        rows[round_idx] = row

    return rows


def build_comparison_rows(base_rows, exp_rows):
    round_indices = sorted(set(base_rows.keys()) | set(exp_rows.keys()))
    rows = []
    for round_idx in round_indices:
        base = base_rows.get(round_idx, {})
        exp = exp_rows.get(round_idx, {})
        row = {
            "round_idx": round_idx,
            "edits_applied": exp.get("edits_applied", base.get("edits_applied")),
        }
        for metric in METRICS:
            base_value = base.get(metric)
            exp_value = exp.get(metric)
            row["base:" + metric] = base_value
            row["exp:" + metric] = exp_value
            base_float = try_float(base_value)
            exp_float = try_float(exp_value)
            row["delta:" + metric] = (
                None
                if base_float is None or exp_float is None
                else exp_float - base_float
            )
        rows.append(row)
    return rows


def make_table(rows):
    headers = [
        "round_idx",
        "edits_applied",
    ]
    for metric in METRICS:
        headers.extend(
            [
                "base:" + metric,
                "exp:" + metric,
                "delta:" + metric,
            ]
        )

    formatted_rows = []
    for row in rows:
        formatted_rows.append([format_value(row.get(header)) for header in headers])

    widths = [len(header) for header in headers]
    for row in formatted_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    lines = []
    lines.append("  ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))))
    lines.append("  ".join("-" * widths[idx] for idx in range(len(headers))))
    for row in formatted_rows:
        lines.append("  ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))
    return "\n".join(lines)


def summarize_metric(rows, metric):
    first = rows[0]
    last = rows[-1]
    return {
        "base_first": first.get("base:" + metric),
        "exp_first": first.get("exp:" + metric),
        "delta_first": first.get("delta:" + metric),
        "base_last": last.get("base:" + metric),
        "exp_last": last.get("exp:" + metric),
        "delta_last": last.get("delta:" + metric),
    }


def make_summary(rows, baseline_path, experiment_path):
    lines = []
    lines.append("baseline: {}".format(baseline_path))
    lines.append("experiment: {}".format(experiment_path))
    if not rows:
        lines.append("no comparable round logs found")
        return "\n".join(lines)

    lines.append("rounds_compared: {}".format(len(rows)))
    lines.append("first_round: {}".format(format_value(rows[0].get("round_idx"))))
    lines.append("last_round: {}".format(format_value(rows[-1].get("round_idx"))))
    lines.append("")
    for metric in METRICS:
        summary = summarize_metric(rows, metric)
        lines.append(metric)
        lines.append(
            "  first: base={} exp={} delta={}".format(
                format_value(summary["base_first"]),
                format_value(summary["exp_first"]),
                format_value(summary["delta_first"]),
            )
        )
        lines.append(
            "  last : base={} exp={} delta={}".format(
                format_value(summary["base_last"]),
                format_value(summary["exp_last"]),
                format_value(summary["delta_last"]),
            )
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Baseline run name or path.")
    parser.add_argument("--experiment", required=True, help="Experimental run name or path.")
    parser.add_argument(
        "--results_root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Base directory used when run arguments are not existing paths.",
    )
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help="Only print concise summary without per-round comparison table.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    baseline_path = resolve_run_path(args.baseline, results_root)
    experiment_path = resolve_run_path(args.experiment, results_root)

    baseline_rows = load_rounds(baseline_path)
    experiment_rows = load_rounds(experiment_path)
    comparison_rows = build_comparison_rows(baseline_rows, experiment_rows)

    if not args.summary_only:
        print(make_table(comparison_rows))
        print()
    print(make_summary(comparison_rows, baseline_path, experiment_path))


if __name__ == "__main__":
    main()
