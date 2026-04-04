"""
Delta-level clipping methods for numerical stability in sequential editing.

Two methods:
  - Frobenius clipping:  ΔW' = ΔW · min(1, τ_F / ‖ΔW‖_F)
  - Spectral clipping:   ΔW' = ΔW · min(1, τ_2 / ‖ΔW‖_2)

Both operate on ΔW *before* it is applied to W, preserving the direction of the update.
"""

from typing import Dict, Optional

import torch


def clip_delta_frobenius(
    delta: torch.Tensor,
    tau_f: float,
) -> Dict[str, object]:
    """
    Frobenius-norm clipping on the update matrix ΔW.

    ΔW' = ΔW · min(1, τ_F / ‖ΔW‖_F)

    Args:
        delta:  The raw update matrix ΔW (on any device).
        tau_f:  Frobenius-norm threshold τ_F.

    Returns dict with:
        clipped_delta   – the (possibly scaled) ΔW'
        pre_clip_norm   – ‖ΔW‖_F before clipping
        post_clip_norm  – ‖ΔW'‖_F after clipping
        scale           – the scalar multiplier applied
        clipped         – bool, whether scaling was applied
    """
    pre_norm = torch.linalg.norm(delta.float(), ord="fro").item()
    if pre_norm > tau_f and pre_norm > 0.0:
        scale = tau_f / pre_norm
        clipped_delta = delta * scale
        clipped = True
    else:
        scale = 1.0
        clipped_delta = delta
        clipped = False

    post_norm = torch.linalg.norm(clipped_delta.float(), ord="fro").item()
    return {
        "clipped_delta": clipped_delta,
        "pre_clip_norm": pre_norm,
        "post_clip_norm": post_norm,
        "scale": scale,
        "clipped": clipped,
    }


def clip_delta_spectral(
    delta: torch.Tensor,
    tau_2: float,
) -> Dict[str, object]:
    """
    Spectral-norm clipping on the update matrix ΔW.

    ΔW' = ΔW · min(1, τ_2 / ‖ΔW‖_2)

    ‖ΔW‖_2 is the largest singular value of ΔW.

    Args:
        delta:  The raw update matrix ΔW (on any device).
        tau_2:  Spectral-norm threshold τ_2.

    Returns dict with:
        clipped_delta        – the (possibly scaled) ΔW'
        pre_clip_spectral    – ‖ΔW‖_2 before clipping
        post_clip_spectral   – ‖ΔW'‖_2 after clipping
        scale                – the scalar multiplier applied
        clipped              – bool, whether scaling was applied
    """
    # torch.linalg.norm with ord=2 returns the largest singular value
    pre_spectral = torch.linalg.norm(delta.float(), ord=2).item()
    if pre_spectral > tau_2 and pre_spectral > 0.0:
        scale = tau_2 / pre_spectral
        clipped_delta = delta * scale
        clipped = True
    else:
        scale = 1.0
        clipped_delta = delta
        clipped = False

    post_spectral = torch.linalg.norm(clipped_delta.float(), ord=2).item()
    return {
        "clipped_delta": clipped_delta,
        "pre_clip_spectral": pre_spectral,
        "post_clip_spectral": post_spectral,
        "scale": scale,
        "clipped": clipped,
    }
