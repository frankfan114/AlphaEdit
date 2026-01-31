import argparse
import json
from pathlib import Path


def score_bin(scores):
    """
    Group by exact neighborhood success ratio k/N
    """
    k = sum(scores)
    N = len(scores)
    return f"{k}/{N}", k, N


def main():
    parser = argparse.ArgumentParser(
        description="Group AlphaEdit cases by exact neighborhood success k/N"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing *_edits-case_*.json files",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output grouped json file",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    assert input_dir.exists() and input_dir.is_dir(), f"Invalid input dir: {input_dir}"

    grouped = {}

    # --------- collect ----------
    for json_file in sorted(input_dir.glob("*_edits-case_*.json")):
        with open(json_file, "r") as f:
            data = json.load(f)

        case_id = data.get("case_id")
        post = data.get("post", {})

        if "neighborhood_prompts_correct" not in post:
            print(f"[WARN] Missing neighborhood data in {json_file.name}")
            continue

        bools = post["neighborhood_prompts_correct"]
        scores = [1 if b else 0 for b in bools]

        bin_label, k, N = score_bin(scores)

        grouped.setdefault(bin_label, []).append(
            {
                "case_id": case_id,
                # "k": k,
                # "N": N,
                "source_file": json_file.name,
            }
        )

    # --------- sort ----------
    grouped_sorted = {}

    # 先按 k 排 bin（0/10, 1/10, ...）
    for bin_label in sorted(grouped.keys(), key=lambda x: int(x.split("/")[0])):
        # 再按 case_id 升序排每个 bin 内部
        grouped_sorted[bin_label] = sorted(
            grouped[bin_label], key=lambda x: x["case_id"]
        )

    # --------- write ----------
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(grouped_sorted, f, indent=2)

    print(f"[OK] Grouped cases written to:\n{output_file}")

    print("\nSummary (Neighborhood Success k/N):")
    for k, v in grouped_sorted.items():
        print(f"  {k:>4s}: {len(v)} cases")


if __name__ == "__main__":
    main()
