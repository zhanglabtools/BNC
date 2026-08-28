from __future__ import annotations

import torch

from bnc_repro.metrics.feature_classifier import centered_feature_classifier_alignment
from bnc_repro.metrics.token_geometry import token_geometry_correlation


def sample_means() -> torch.Tensor:
    return torch.tensor(
        [[2.0, -1.0, 0.5], [-0.5, 1.5, 2.0], [1.0, 2.5, -1.0], [-2.0, -0.5, 1.5]],
        dtype=torch.float64,
    )


def test_token_geometry_is_invariant_to_common_orthogonal_transform() -> None:
    generator = torch.Generator().manual_seed(11)
    matrix = torch.randn(17, 6, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(6, 6, generator=generator, dtype=torch.float64))
    assert abs(token_geometry_correlation(matrix, matrix @ q) - 1.0) < 1e-12


def test_centered_alignment_positive_negative_and_offset_invariance() -> None:
    means = sample_means()
    permutations = torch.tensor([[1, 2, 3, 0], [3, 0, 1, 2]])
    positive, _ = centered_feature_classifier_alignment(
        means + torch.tensor([90.0, -17.0, 6.0]),
        2.5 * means + torch.tensor([-5.0, 42.0, 13.0]),
        permutations,
    )
    negative, _ = centered_feature_classifier_alignment(
        means, -2.5 * means + torch.tensor([8.0, -3.0, 11.0]), permutations
    )
    assert abs(positive - 1.0) < 1e-12
    assert abs(negative + 1.0) < 1e-12


def test_shuffled_control_matches_manual_computation() -> None:
    means = sample_means()
    classifier = torch.tensor(
        [[1.5, 0.2, -0.4], [-0.3, 1.8, 0.6], [0.4, -0.7, 2.1], [-1.2, -0.6, -0.8]],
        dtype=torch.float64,
    )
    permutations = torch.tensor([[1, 2, 3, 0], [3, 0, 1, 2]])
    _, shuffled = centered_feature_classifier_alignment(means, classifier, permutations)
    means_unit = (means - means.mean(0))
    means_unit = means_unit / torch.linalg.vector_norm(means_unit, dim=1, keepdim=True)
    classifier_unit = classifier - classifier.mean(0)
    classifier_unit = classifier_unit / torch.linalg.vector_norm(classifier_unit, dim=1, keepdim=True)
    manual = torch.stack(
        [(means_unit * classifier_unit[p]).sum(1).mean() for p in permutations]
    ).mean()
    assert shuffled is not None
    assert abs(shuffled - float(manual)) < 1e-12


def test_degenerate_centered_row_is_rejected() -> None:
    degenerate = torch.tensor([[1.0, 1.0], [0.0, 0.0], [2.0, 2.0]])
    try:
        centered_feature_classifier_alignment(degenerate, degenerate)
    except ValueError as error:
        assert "near-zero rows" in str(error)
    else:
        raise AssertionError("degenerate centered rows were accepted")

