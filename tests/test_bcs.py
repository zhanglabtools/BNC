from __future__ import annotations

import math

import torch

from bnc_repro.metrics.bcs import best_cyclic_score


def ring(modulus: int, multiplier: int = 1) -> torch.Tensor:
    labels = torch.arange(modulus, dtype=torch.float64)
    phase = 2 * math.pi * multiplier * labels / modulus
    return torch.stack((torch.cos(phase), torch.sin(phase)), dim=1)


def test_perfect_ring_is_one() -> None:
    assert best_cyclic_score(ring(97), 97) > 1 - 1e-12


def test_rotation_reflection_and_automorphism_invariance() -> None:
    base = ring(79, multiplier=7)
    angle = 0.713
    rotation = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=torch.float64,
    )
    variants = (base @ rotation, (base @ rotation) * torch.tensor([1.0, -1.0]))
    for variant in variants:
        assert best_cyclic_score(variant, 79) > 1 - 1e-12


def test_bcs_bounds_for_random_matrix() -> None:
    generator = torch.Generator().manual_seed(4)
    score = best_cyclic_score(torch.randn(17, 8, generator=generator), 17)
    assert 0.0 <= score <= 1.0

