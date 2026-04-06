"""
Diagnostic utilities for analyzing AlphaEdit sequential editing.

Computes per-step, per-layer metrics that reveal *why* AlphaEdit experiences
late-stage Edit-Success (ES) decay during long sequential editing runs.

Key quantities
--------------
conflict_sub(t)
    Fraction of the current edit's key-vector energy that lies in the
    *blocked* directions — the complement of the null-space (i.e. the
    high-eigenvalue subspace that AlphaEdit forbids updating).
    A rising trend means later edits increasingly point in directions
    AlphaEdit cannot modify.

    Definition (per layer ℓ, step t):
        conflict_sub(t) = 1 - ‖P_ℓ k_t‖² / ‖k_t‖²  =  ‖(I-P_ℓ) k_t‖² / ‖k_t‖²
    where
        P_ℓ  = U₀ U₀ᵀ   (fixed null-space projector, never updated)
        k_t  = mean key vector for the current edit (mean over context copies)

    Implementation note:
        We use P directly.  Since P is an orthogonal projector (P²=P, Pᵀ=P),
        ‖P k‖² = kᵀ Pᵀ P k = kᵀ P k.  No need to factor P back to U₀.

block_ratio(t)
    Fraction of the *raw (unconstrained) update* Frobenius energy that AlphaEdit
    blocks by the null-space projector.

    We define the "raw update" as the MEMIT-style closed-form update that
    *ignores* the null-space constraint:
        Δ_raw  = (K Kᵀ + C + λI)⁻¹ K Rᵀ
    AlphaEdit's actual update instead forces the row space into the null-space
    via P (left-multiplied on the key dimension):
        Δ_allow = P @ Δ_raw          ← components in null-space (allowed)
        Δ_block = (I - P) @ Δ_raw   ← components blocked by AlphaEdit

    Definition:
        block_ratio(t) = ‖Δ_block‖_F / (‖Δ_raw‖_F + ε)

    Projection direction: LEFT multiplication on the [d_key, d_out] matrix.
    This mirrors AlphaEdit's formula where P appears left-multiplied:
        P (K Kᵀ + C) and P K Rᵀ

    A block_ratio rising in sync with ES decay would confirm that AlphaEdit's
    null-space constraint increasingly suppresses the desired updates.

Usage (called from AlphaEdit_main.py with enable_diagnostics=True)
-------------------------------------------------------------------
    from .diagnostics import compute_conflict_sub, compute_block_ratio, protected_rank_from_projector
"""

import warnings
from typing import Dict

import torch


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_conflict_sub(
    layer_ks: torch.Tensor,    # [d_key, n]  key vectors (already transposed)
    P_layer: torch.Tensor,     # [d_key, d_key]  fixed protected-subspace projector
) -> Dict:
    """
    Compute conflict_sub(t) for a single layer.

    conflict_sub(t) = ‖P k_t‖² / ‖k_t‖²

    where k_t is the mean key vector across all n context copies for this edit.

    Args:
        layer_ks:  [d_key, n]  – the key-vector matrix for the current edit
                   (output of compute_ks().T inside apply_AlphaEdit_to_model)
        P_layer:   [d_key, d_key]  – the fixed orthogonal projector onto the
                   protected null-space.  P = U₀ U₀ᵀ where U₀ holds the
                   eigenvectors of the pre-training covariance that correspond
                   to eigenvalues below hparams.nullspace_threshold.

    Returns:
        dict with keys:
            conflict_sub  – scalar in [0, 1]
            k_t_norm      – ‖k_t‖₂  (for reference)
    """
    if layer_ks.ndim == 1:
        k_t = layer_ks.detach().float()
    else:
        # [d_key, n] → mean over context/request columns → [d_key]
        k_t = layer_ks.detach().float().mean(dim=1)

    P = P_layer.detach().float().to(k_t.device)

    Pk = P @ k_t                             # [d_key]
    k_norm_sq  = float((k_t @ k_t).item())
    Pk_norm_sq = float((Pk @ Pk).item())

    return {
        "conflict_sub": 1.0 - Pk_norm_sq / (k_norm_sq + 1e-12),  # fraction of k in BLOCKED (complement) directions
        "k_t_norm":     float(k_norm_sq ** 0.5),
    }


def compute_block_ratio(
    layer_ks: torch.Tensor,    # [d_key, n]      key vectors (transposed)
    resid:    torch.Tensor,    # [d_out, n]      distributed residual
    cache_c:  torch.Tensor,    # [d_key, d_key]  accumulated sequential covariance
    P_layer:  torch.Tensor,    # [d_key, d_key]  fixed protected-subspace projector
    L2:       float,           # L2 regularisation coefficient (hparams.L2)
) -> Dict:
    """
    Compute block_ratio(t) for a single layer.

    Computes the MEMIT-style raw update (without null-space constraint) and
    measures what fraction of its Frobenius energy would be projected out by
    AlphaEdit's null-space projector.

    Δ_raw   = (K Kᵀ + C + λI)⁻¹  K Rᵀ       [d_key × d_out]
    Δ_allow = P @ Δ_raw                        [left-mult on key dimension]
    Δ_block = (I − P) @ Δ_raw  =  Δ_raw − Δ_allow

    block_ratio = ‖Δ_block‖_F / (‖Δ_raw‖_F + ε)

    Args:
        layer_ks:  [d_key, n]      – key-vector matrix
        resid:     [d_out, n]      – residual targets (distributed across layers)
        cache_c:   [d_key, d_key]  – sequential covariance accumulator
        P_layer:   [d_key, d_key]  – fixed orthogonal projector
        L2:        float           – L2 reg. (same as hparams.L2)

    Returns:
        dict with keys:
            block_ratio      – scalar ≥ 0
            delta_raw_fro    – ‖Δ_raw‖_F
            delta_allow_fro  – ‖Δ_allow‖_F
            delta_block_fro  – ‖Δ_block‖_F
            solve_failed     – bool (True if numerical solve raised an exception)
    """
    d = layer_ks.shape[0]
    device = layer_ks.device

    try:
        K = layer_ks.detach().float()
        R = resid.detach().float()
        C = cache_c.detach().float().to(device)
        P = P_layer.detach().float().to(device)

        # MEMIT-style system (no P constraint):  A x = b
        A_raw = K @ K.T + C + L2 * torch.eye(d, dtype=torch.float32, device=device)
        b_raw = K @ R.T                         # [d_key, d_out]

        delta_raw   = torch.linalg.solve(A_raw, b_raw)   # [d_key, d_out]
        delta_allow = P @ delta_raw                        # [d_key, d_out]  (LEFT mult)
        delta_block = delta_raw - delta_allow              # [d_key, d_out]

        raw_fro   = float(torch.linalg.norm(delta_raw,   ord="fro").item())
        allow_fro = float(torch.linalg.norm(delta_allow, ord="fro").item())
        block_fro = float(torch.linalg.norm(delta_block, ord="fro").item())

        del K, R, C, P, A_raw, b_raw, delta_raw, delta_allow, delta_block
        torch.cuda.empty_cache()

        return {
            "block_ratio":    block_fro / (raw_fro + 1e-12),
            "delta_raw_fro":  raw_fro,
            "delta_allow_fro": allow_fro,
            "delta_block_fro": block_fro,
            "solve_failed":   False,
        }

    except Exception as exc:
        warnings.warn(
            f"[diagnostics] compute_block_ratio: linear solve failed – {exc}. "
            "block_ratio will be NaN for this step/layer."
        )
        return {
            "block_ratio":    float("nan"),
            "delta_raw_fro":  float("nan"),
            "delta_allow_fro": float("nan"),
            "delta_block_fro": float("nan"),
            "solve_failed":   True,
        }


def compute_rhs_block_ratio(
    layer_ks: torch.Tensor,    # [d_key, n]
    resid:    torch.Tensor,    # [d_out, n]
    P_layer:  torch.Tensor,    # [d_key, d_key]
) -> Dict:
    """
    Compute rhs_block_ratio(t) — a simpler, C-independent measure of how much
    of the edit's target signal is blocked by the null-space projector.

    Instead of solving the full linear system, we look directly at the
    right-hand side signal b = K Rᵀ and decompose it via P:

        b_total  = K Rᵀ                    [d_key, d_out]
        b_allow  = P @ b_total             (component in null-space, allowed)
        b_block  = (I − P) @ b_total       (component blocked by AlphaEdit)

        rhs_block_ratio = ‖b_block‖_F / (‖b_total‖_F + ε)

    Unlike block_ratio, this does NOT depend on cache_c or the linear solve.
    It purely reflects the geometric alignment of the current edit's key-target
    direction with the protected subspace — making it cleaner for tracking
    how the edit intent conflicts with the null-space constraint over time.
    """
    try:
        K = layer_ks.detach().float()
        R = resid.detach().float()
        P = P_layer.detach().float().to(K.device)

        b_total = K @ R.T                  # [d_key, d_out]
        b_allow = P @ b_total              # [d_key, d_out]
        b_block = b_total - b_allow        # [d_key, d_out]

        total_fro = float(torch.linalg.norm(b_total, ord="fro").item())
        allow_fro = float(torch.linalg.norm(b_allow, ord="fro").item())
        block_fro = float(torch.linalg.norm(b_block, ord="fro").item())

        return {
            "rhs_block_ratio":  block_fro / (total_fro + 1e-12),
            "rhs_total_fro":    total_fro,
            "rhs_allow_fro":    allow_fro,
            "rhs_block_fro":    block_fro,
        }
    except Exception as exc:
        warnings.warn(f"[diagnostics] compute_rhs_block_ratio failed: {exc}")
        return {
            "rhs_block_ratio": float("nan"),
            "rhs_total_fro":   float("nan"),
            "rhs_allow_fro":   float("nan"),
            "rhs_block_fro":   float("nan"),
        }


def protected_rank_from_projector(P_layer: torch.Tensor) -> int:
    """
    Estimate the rank of the protected null-space from the projector matrix P.

    Since P is an orthogonal projector (P² = P, Pᵀ = P), its eigenvalues are
    either 0 or 1, so rank(P) = trace(P).

    Args:
        P_layer:  [d, d]  – the projector (can be on any device)

    Returns:
        Integer rank of the protected subspace.
    """
    return int(round(float(P_layer.detach().float().trace().item())))
