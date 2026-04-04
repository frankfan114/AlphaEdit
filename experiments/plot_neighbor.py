#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import math

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot relationship between neighborhood hard correctness (k/N) "
            "and mean soft probability margin (NLL_true - NLL_new)."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing *_edits-case_*.json files",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        required=True,
        help="Path to output scatter plot (.png)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_png = Path(args.output_png)

    assert input_dir.exists() and input_dir.is_dir(), f"Invalid input dir: {input_dir}"

    xs = []  # k / N
    ys = []  # mean(NLL_true - NLL_new)

    # --------- collect ----------
    for json_file in sorted(input_dir.glob("*_edits-case_*.json")):
        with open(json_file, "r") as f:
            data = json.load(f)

        case_id = data.get("case_id", "UNKNOWN")
        post = data.get("post", {})

        corrects = post.get("neighborhood_prompts_correct")
        probs = post.get("neighborhood_prompts_probs")

        if corrects is None or probs is None:
            print(f"[WARN] Missing neighborhood data in {json_file.name}, skip")
            continue

        if len(corrects) == 0 or len(probs) == 0:
            print(f"[WARN] Empty neighborhood data in {json_file.name}, skip")
            continue

        if len(corrects) != len(probs):
            print(
                f"[WARN] Length mismatch in {json_file.name} "
                f"(correct={len(corrects)}, probs={len(probs)}), skip"
            )
            continue

        # ---- x: hard correctness score ----
        k = sum(1 for b in corrects if b)
        N = len(corrects)
        x = k / N

        # ---- y: mean soft margin ----
        deltas = []
        for p in probs:
            if "target_true" not in p or "target_new" not in p:
                continue
            deltas.append(math.log(p["target_true"] / p["target_new"]))


        if len(deltas) == 0:
            print(f"[WARN] No valid prob entries in {json_file.name}, skip")
            continue

        y = sum(deltas) / len(deltas)

        xs.append(x)
        ys.append(y)

    if len(xs) == 0:
        raise RuntimeError("No valid cases collected; nothing to plot.")

    # --------- plot ----------
    plt.figure(figsize=(6, 4.5))
    plt.scatter(xs, ys, alpha=0.7)

    plt.xlabel("Neighborhood correctness score (k / N)")
    plt.ylabel("Mean probability margin (NLL_true - NLL_new)")
    plt.title("Hard vs Soft Neighborhood Specificity")

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()

    print(f"[OK] Plot saved to: {output_png}")
    print(f"[INFO] Plotted {len(xs)} cases")


if __name__ == "__main__":
    main()
