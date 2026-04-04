#!/usr/bin/env python3
"""
Compare per-layer null-space projection matrices saved in null_space_project.pt.

Outputs:
- Per-layer sanity checks (symmetry/idempotence/rank)
- Pairwise metrics between layers:
  - Frobenius distance
  - Subspace overlap (normalized trace)
  - Mean principal angle (degrees)
"""


import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import torch


def symmetrize(m: torch.Tensor) -> torch.Tensor:
    return (m + m.T) / 2.0


def proj_rank(m: torch.Tensor, eig_threshold: float) -> int:
    ms = symmetrize(m).to(dtype=torch.float64, device="cpu").contiguous()
    # Use singular values for robustness across LAPACK/MKL backends.
    s = torch.linalg.svdvals(ms)
    return int((s > eig_threshold).sum().item())


def proj_basis(m: torch.Tensor, eig_threshold: float) -> torch.Tensor:
    ms = symmetrize(m).to(dtype=torch.float64, device="cpu").contiguous()
    # For symmetric PSD projection matrices, left singular vectors span the same subspace.
    u, s, _ = torch.linalg.svd(ms, full_matrices=False)
    keep = s > eig_threshold
    if int(keep.sum().item()) == 0:
        return torch.empty((m.shape[0], 0), dtype=torch.float64)
    return u[:, keep]


def per_layer_checks(p: torch.Tensor, eig_threshold: float) -> List[Tuple[int, int, float, float]]:
    rows = []
    for i in range(p.shape[0]):
        pi = p[i]
        rank_i = proj_rank(pi, eig_threshold)
        sym_err = torch.norm(pi - pi.T, p="fro").item()
        idem_err = torch.norm(pi @ pi - pi, p="fro").item()
        rows.append((i, rank_i, sym_err, idem_err))
    return rows


def pairwise_metrics(
    p: torch.Tensor, eig_threshold: float
) -> List[Tuple[int, int, float, float, float]]:
    l = p.shape[0]
    bases = [proj_basis(p[i], eig_threshold) for i in range(l)]
    ranks = [b.shape[1] for b in bases]

    rows = []
    for i in range(l):
        for j in range(i + 1, l):
            pi, pj = p[i], p[j]
            fro = torch.norm(pi - pj, p="fro").item()

            denom = max(1, min(ranks[i], ranks[j]))
            overlap = (torch.trace(pi @ pj).item()) / float(denom)

            if ranks[i] == 0 or ranks[j] == 0:
                mean_angle_deg = float("nan")
            else:
                s = torch.linalg.svdvals(bases[i].T @ bases[j]).clamp(-1.0, 1.0)
                angles = torch.arccos(s) * (180.0 / torch.pi)
                mean_angle_deg = float(angles.mean().item())

            rows.append((i, j, fro, overlap, mean_angle_deg))
    return rows


def write_csv(path: Path, header: List[str], rows: List[Tuple]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    here = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Compare per-layer null-space projections.")
    parser.add_argument(
        "--pt",
        type=Path,
        default=here / "null_space_project.pt",
        help="Path to null_space_project.pt",
    )
    parser.add_argument(
        "--eig-threshold",
        type=float,
        default=0.5,
        help="Eigenvalue threshold used to estimate projection rank/basis.",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="nullspace_compare",
        help="Prefix of output CSV files (saved next to the .pt file).",
    )
    args = parser.parse_args()

    p = torch.load(args.pt, map_location="cpu")
    if not isinstance(p, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor in {args.pt}, got {type(p)}.")
    if p.ndim != 3 or p.shape[1] != p.shape[2]:
        raise ValueError(f"Expected [L, d, d] tensor, got shape {tuple(p.shape)}.")

    layer_rows = per_layer_checks(p, args.eig_threshold)
    pair_rows = pairwise_metrics(p, args.eig_threshold)

    out_dir = args.pt.resolve().parent
    layer_csv = out_dir / f"{args.out_prefix}_per_layer.csv"
    pair_csv = out_dir / f"{args.out_prefix}_pairwise.csv"

    write_csv(
        layer_csv,
        ["layer_idx", "rank_est", "symmetry_error_fro", "idempotence_error_fro"],
        layer_rows,
    )
    write_csv(
        pair_csv,
        ["layer_i", "layer_j", "fro_distance", "overlap_norm_trace", "mean_principal_angle_deg"],
        pair_rows,
    )

    print(f"Loaded tensor shape: {tuple(p.shape)} from {args.pt}")
    print(f"Wrote per-layer checks: {layer_csv}")
    print(f"Wrote pairwise metrics: {pair_csv}")

    # Console summary for quick inspection
    print("\nPer-layer summary:")
    for layer_idx, rank_est, sym_err, idem_err in layer_rows:
        print(
            f"  layer {layer_idx}: rank={rank_est}, "
            f"sym_err={sym_err:.3e}, idem_err={idem_err:.3e}"
        )


if __name__ == "__main__":
    main()
