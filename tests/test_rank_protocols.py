from __future__ import annotations

import math

import torch

from bnc_repro.metrics.role_conditioned import role_conditioned_codes
from bnc_repro.models.registry import build_model
from bnc_repro.protocols.rank_common import factorize_dense_model
from bnc_repro.protocols.rank_homotopy import cosine_tail_gate


def test_rank2_head_numerical_rank() -> None:
    model = build_model("mlp", 17, 1, {"mlp_embedding_dim": 16, "mlp_hidden_dim": 12})
    factorized = factorize_dense_model(model, 2, 2)
    assert int(torch.linalg.matrix_rank(factorized.classifier()).item()) <= 2


def test_tail_gate_endpoints_midpoint_and_monotonicity() -> None:
    assert cosine_tail_gate(1, 1, 6000) == 1.0
    assert cosine_tail_gate(6000, 1, 6000) == 0.0
    midpoint = cosine_tail_gate((1 + 6000) // 2, 1, 6000)
    assert 0.49 < midpoint < 0.51
    values = [cosine_tail_gate(epoch, 1, 6000) for epoch in range(1, 6001, 137)]
    assert all(first >= second for first, second in zip(values, values[1:]))


def test_role_conditioned_codes_match_manual_example() -> None:
    pairs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    features = torch.tensor([[0.0], [2.0], [4.0], [6.0]])
    rx, ry = role_conditioned_codes(features, pairs, 2)
    assert torch.allclose(rx, torch.tensor([[-2.0], [2.0]]))
    assert torch.allclose(ry, torch.tensor([[-1.0], [1.0]]))

