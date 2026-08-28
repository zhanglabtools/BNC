from __future__ import annotations

import torch


def participation_rank(
    matrix: torch.Tensor,
    *,
    center_rows: bool = True,
    eps: float = 1e-30,
) -> torch.Tensor:
    """Energy participation rank of a matrix.

    The public convention is K x d rows. ``center_rows=True`` subtracts the
    mean across the K rows before computing singular values.
    """

    if matrix.ndim != 2:
        raise ValueError(f"participation rank requires a rank-2 matrix, got {matrix.shape}")
    work = matrix
    if center_rows:
        work = work - work.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(work)
    energy = singular_values.square()
    total = energy.sum()
    if total <= eps:
        raise ValueError("participation rank is undefined for a zero-energy matrix")
    return total.square() / energy.square().sum().clamp_min(eps)


def classifier_participation_rank(classifier_columns: torch.Tensor) -> torch.Tensor:
    """Participation rank for a d x K classifier, centered across classes."""
    if classifier_columns.ndim != 2:
        raise ValueError("classifier columns must be rank two")
    centered = classifier_columns - classifier_columns.mean(dim=1, keepdim=True)
    return participation_rank(centered, center_rows=False)


def spectrum_summary(class_rows: torch.Tensor) -> dict[str, float]:
    centered = class_rows.to(torch.float64) - class_rows.to(torch.float64).mean(
        dim=0, keepdim=True
    )
    singular_values = torch.linalg.svdvals(centered)
    singular_values = singular_values[singular_values > 1e-12]
    if singular_values.numel() == 0:
        raise ValueError("spectrum is degenerate")
    energy = singular_values.square()
    participation = energy.sum().square() / energy.square().sum().clamp_min(1e-30)
    probability = singular_values / singular_values.sum().clamp_min(1e-30)
    entropy = torch.exp(-(probability * torch.log(probability.clamp_min(1e-30))).sum())
    tail = (
        energy[2:].sum() / energy.sum().clamp_min(1e-30)
        if energy.numel() > 2
        else energy.new_zeros(())
    )
    return {
        "participation_rank": float(participation.item()),
        "entropy_rank": float(entropy.item()),
        "top2_tail": float(tail.item()),
    }

