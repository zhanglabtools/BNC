from __future__ import annotations

import torch
import torch.nn.functional as F


def centered_unit_rows(matrix: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError(f"expected a K x d matrix, got {matrix.shape}")
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    norms = torch.linalg.vector_norm(centered, dim=1, keepdim=True)
    if bool(torch.any(norms <= eps).item()):
        bad = torch.nonzero(norms[:, 0] <= eps, as_tuple=False).flatten().tolist()
        raise ValueError(f"centered representation has near-zero rows: {bad}")
    return centered / norms


def token_geometry_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    """Pearson correlation of strict-upper pairwise-cosine geometries."""
    if first.shape[0] != second.shape[0]:
        raise ValueError("token matrices must have the same number of rows")
    first_unit = centered_unit_rows(first.to(torch.float64))
    second_unit = centered_unit_rows(second.to(torch.float64))
    first_cosines = first_unit @ first_unit.T
    second_cosines = second_unit @ second_unit.T
    indices = torch.triu_indices(first.shape[0], first.shape[0], offset=1)
    first_values = first_cosines[indices[0], indices[1]]
    second_values = second_cosines[indices[0], indices[1]]
    first_values = first_values - first_values.mean()
    second_values = second_values - second_values.mean()
    denominator = (
        torch.linalg.vector_norm(first_values)
        * torch.linalg.vector_norm(second_values)
    )
    if denominator <= 1e-20:
        raise ValueError("token-geometry Pearson correlation is degenerate")
    return float(((first_values @ second_values) / denominator).item())


def fixed_shuffle_control(
    first: torch.Tensor,
    second: torch.Tensor,
    permutations: torch.Tensor,
) -> float:
    if permutations.ndim != 2 or permutations.shape[1] != first.shape[0]:
        raise ValueError("permutations must have shape n_controls x K")
    values = [token_geometry_correlation(first, second[p]) for p in permutations]
    return float(torch.tensor(values, dtype=torch.float64).mean().item())

