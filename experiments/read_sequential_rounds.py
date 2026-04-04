import argparse
import json
from pathlib import Path


DEFAULT_RESULTS_ROOT = Path("results") / "AlphaEdit"

FIELDS = [
    ("immediate_current_batch_edit_success", "immediate_current_batch_edit_success"),
    ("current_batch_eval.summary.edit_success", "current_batch_eval.summary.edit_success"),
    ("history_probe_eval.summary.edit_success", "history_probe_eval.summary.edit_success"),
    ("update_diagnostics.aggregate.mean_update_norm", "update_diagnostics.aggregate.mean_update_norm"),
    (
        "update_diagnostics.aggregate.mean_update_to_orig_norm_ratio",
        "update_diagnostics.aggregate.mean_update_to_orig_norm_ratio",
    ),
    (
        "update_diagnostics.aggregate.mean_pre_projection_delta_norm",
        "update_diagnostics.aggregate.mean_pre_projection_delta_norm",
    ),
    (
        "update_diagnostics.aggregate.mean_post_projection_delta_norm",
        "update_diagnostics.aggregate.mean_post_projection_delta_norm",
    ),
    (
        "update_diagnostics.aggregate.projection_applied_layers",
        "update_diagnostics.aggregate.projection_applied_layers",
    ),
    ("weight_metrics.aggregate.max_condition_number", "weight_metrics.aggregate.max_condition_number"),
    (
        "weight_metrics.aggregate.mean_frobenius_norm_drift",
        "weight_metrics.aggregate.mean_frobenius_norm_drift",
    ),
]


def resolve_run_path(run_arg: str, results_root: Path) -> Path:
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


def load_round_rows(run_path: Path):
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

        row = {
            "round_idx": payload.get("round_idx"),
            "edits_applied": payload.get("edits_applied"),
        }
        for label, dotted_key in FIELDS:
            row[label] = nested_get(payload, dotted_key)
        rows.append(row)

    return rows


def make_table(rows):
    headers = ["round_idx", "edits_applied"] + [label for label, _ in FIELDS]
    formatted_rows = []
    for row in rows:
        formatted_rows.append([format_value(row.get(header)) for header in headers])

    widths = [len(header) for header in headers]
    for row in formatted_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    header_line = "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
    separator_line = "  ".join("-" * widths[i] for i in range(len(headers)))
    body_lines = [
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        for row in formatted_rows
    ]
    return "\n".join([header_line, separator_line] + body_lines)


def summarize_rows(rows):
    if not rows:
        return "no round logs found"

    first_row = rows[0]
    last_row = rows[-1]
    summary_keys = [
        "immediate_current_batch_edit_success",
        "history_probe_eval.summary.edit_success",
        "update_diagnostics.aggregate.mean_update_norm",
        "update_diagnostics.aggregate.mean_post_projection_delta_norm",
        "update_diagnostics.aggregate.projection_applied_layers",
        "weight_metrics.aggregate.max_condition_number",
        "weight_metrics.aggregate.mean_frobenius_norm_drift",
    ]
    lines = [
        "rounds: {}".format(len(rows)),
        "first_round_idx: {}".format(format_value(first_row.get("round_idx"))),
        "last_round_idx: {}".format(format_value(last_row.get("round_idx"))),
    ]
    for key in summary_keys:
        lines.append(
            "{}: first={} last={}".format(
                key,
                format_value(first_row.get(key)),
                format_value(last_row.get(key)),
            )
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run name like run_030 or a direct path to a run directory. Repeat twice for baseline and numerical_stability.",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=str(DEFAULT_RESULTS_ROOT),
        help="Base directory used when --run is not an existing path.",
    )
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help="Only print compact summary instead of full per-round table.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)

    for idx, run_arg in enumerate(args.run):
        run_path = resolve_run_path(run_arg, results_root)
        print("=== {} ===".format(run_path.name))
        print("path: {}".format(run_path))
        rows = load_round_rows(run_path)
        if not rows:
            print("no sequential_rounds/*.json found")
        elif args.summary_only:
            print(summarize_rows(rows))
        else:
            print(make_table(rows))
            print()
            print(summarize_rows(rows))
        if idx != len(args.run) - 1:
            print()


if __name__ == "__main__":
    main()
