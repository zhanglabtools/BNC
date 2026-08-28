"""Small, deterministic walkthrough of the public geometry metrics."""

from __future__ import annotations

import torch

from bnc_repro.metrics.bcs import best_cyclic_score
from bnc_repro.metrics.participation import participation_rank, spectrum_summary
from bnc_repro.metrics.token_geometry import (
    fixed_shuffle_control,
    token_geometry_correlation,
)
from bnc_repro.theory.geometry import cyclic_code


def main() -> None:
    modulus = 17
    dtype = torch.float64
    code = torch.as_tensor(cyclic_code(modulus), dtype=dtype)
    rotation = torch.tensor([[0.6, -0.8], [0.8, 0.6]], dtype=dtype)
    rotated = code @ rotation

    generator = torch.Generator().manual_seed(20260828)
    permutations = torch.stack(
        [torch.randperm(modulus, generator=generator) for _ in range(8)]
    )

    result = {
        "best_cyclic_score": best_cyclic_score(rotated, modulus),
        "rotated_geometry_correlation": token_geometry_correlation(code, rotated),
        "fixed_shuffle_control": fixed_shuffle_control(code, rotated, permutations),
        "participation_rank": float(participation_rank(rotated).item()),
        **spectrum_summary(rotated),
    }
    for name, value in result.items():
        print(f"{name}: {value:.6f}")

    assert result["best_cyclic_score"] > 0.999999
    assert result["rotated_geometry_correlation"] > 0.999999
    assert result["participation_rank"] > 1.999999
    assert result["fixed_shuffle_control"] < 0.9


if __name__ == "__main__":
    main()
