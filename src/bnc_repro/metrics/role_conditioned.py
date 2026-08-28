from __future__ import annotations

import torch


def role_conditioned_codes(
    features: torch.Tensor,
    pairs: torch.Tensor,
    modulus: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return K x d role means after subtracting the complete-grid mean."""
    if features.ndim != 2 or pairs.shape != (features.shape[0], 2):
        raise ValueError("features and ordered pairs have incompatible shapes")
    global_mean = features.mean(dim=0, keepdim=True)
    results: list[torch.Tensor] = []
    for role in (0, 1):
        indices = pairs[:, role]
        sums = torch.zeros(
            (modulus, features.shape[1]), dtype=features.dtype, device=features.device
        )
        sums.index_add_(0, indices, features)
        counts = torch.bincount(indices, minlength=modulus).to(features.dtype)
        if bool(torch.any(counts == 0).item()):
            raise ValueError("role-conditioned code is missing symbols")
        results.append(sums / counts[:, None] - global_mean)
    return results[0], results[1]


def effective_participation_dimension(rx: torch.Tensor, ry: torch.Tensor) -> torch.Tensor:
    from .participation import participation_rank

    return 0.5 * (
        participation_rank(rx, center_rows=False)
        + participation_rank(ry, center_rows=False)
    )


def participation_target_penalty(
    rx: torch.Tensor,
    ry: torch.Tensor,
    target: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    from .participation import participation_rank

    ranks = torch.stack(
        [
            participation_rank(rx, center_rows=False),
            participation_rank(ry, center_rows=False),
        ]
    )
    penalty = 0.5 * torch.log(ranks / target).square().sum()
    return penalty, ranks.mean()

