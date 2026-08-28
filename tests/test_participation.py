from __future__ import annotations

import math

import torch

from bnc_repro.metrics.participation import participation_rank


def test_uniform_circle_has_dimension_two() -> None:
    phase = 2 * math.pi * torch.arange(101, dtype=torch.float64) / 101
    circle = torch.stack((torch.cos(phase), torch.sin(phase)), dim=1)
    assert torch.isclose(participation_rank(circle), torch.tensor(2.0, dtype=torch.float64), atol=1e-12)


def test_equal_energy_rank_r_has_participation_r() -> None:
    for rank in (1, 2, 5, 9):
        matrix = torch.eye(rank, dtype=torch.float64)
        value = participation_rank(matrix, center_rows=False)
        assert torch.isclose(value, torch.tensor(float(rank), dtype=torch.float64), atol=1e-12)

