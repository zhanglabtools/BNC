from __future__ import annotations

import numpy as np

from bnc_repro.theory.geometry import candidate_crossover_scale, cyclic_code, simplex_etf_gram


def test_cyclic_code_is_unit_norm_and_centered() -> None:
    code = cyclic_code(97)
    assert np.allclose(np.linalg.norm(code, axis=1), 1.0)
    assert np.allclose(code.mean(axis=0), 0.0, atol=1e-14)


def test_simplex_etf_gram_has_expected_spectrum() -> None:
    gram = simplex_etf_gram(11)
    constant = np.ones(11)
    centered = np.arange(11, dtype=float) - 5.0
    assert np.allclose(gram @ constant, 0.0, atol=1e-12)
    assert np.allclose(gram @ centered, (11 / 10) * centered)


def test_candidate_crossover_has_inverse_class_scaling() -> None:
    first = candidate_crossover_scale(50, 2.0, 0.4)
    second = candidate_crossover_scale(100, 2.0, 0.4)
    assert np.isclose(first / second, 2.0)
