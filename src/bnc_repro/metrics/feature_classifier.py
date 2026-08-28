from __future__ import annotations

import torch

from .token_geometry import centered_unit_rows


def class_means(features: torch.Tensor, labels: torch.Tensor, modulus: int) -> torch.Tensor:
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels have incompatible shapes")
    sums = torch.zeros(
        (modulus, features.shape[1]), dtype=features.dtype, device=features.device
    )
    sums.index_add_(0, labels, features)
    counts = torch.bincount(labels, minlength=modulus)
    if bool(torch.any(counts == 0).item()):
        missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"balanced grid is missing classes: {missing}")
    return sums / counts.to(features.dtype).unsqueeze(1)


def centered_feature_classifier_alignment(
    means: torch.Tensor,
    classifier_rows: torch.Tensor,
    permutations: torch.Tensor | None = None,
) -> tuple[float, float | None]:
    """Mean corresponding-class cosine after separate across-class centering."""
    means_unit = centered_unit_rows(means.to(torch.float64))
    classifier_unit = centered_unit_rows(classifier_rows.to(torch.float64))
    if means_unit.shape != classifier_unit.shape:
        raise ValueError(
            f"feature means and classifier rows differ: {means_unit.shape} != {classifier_unit.shape}"
        )
    matched = float(torch.sum(means_unit * classifier_unit, dim=1).mean().item())
    if permutations is None:
        return matched, None
    if permutations.ndim != 2 or permutations.shape[1] != means.shape[0]:
        raise ValueError("permutations must have shape n_controls x K")
    shuffled = torch.sum(
        means_unit.unsqueeze(0) * classifier_unit[permutations], dim=2
    ).mean()
    return matched, float(shuffled.item())

