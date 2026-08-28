from __future__ import annotations

import numpy as np


def cyclic_code(classes: int, multiplier: int = 1) -> np.ndarray:
    labels = np.arange(classes)
    phase = 2.0 * np.pi * multiplier * labels / classes
    return np.stack((np.cos(phase), np.sin(phase)), axis=1)


def simplex_etf_gram(classes: int) -> np.ndarray:
    if classes < 2:
        raise ValueError("classes must be at least two")
    identity = np.eye(classes)
    return classes / (classes - 1) * (identity - np.ones((classes, classes)) / classes)


def candidate_crossover_scale(classes: int, cross_entropy_gap: float, complexity_gap_per_class: float) -> float:
    """Candidate-level O(1/K) surrogate threshold, not an AdamW coefficient."""
    if classes <= 0 or complexity_gap_per_class <= 0:
        raise ValueError("classes and complexity gap must be positive")
    return cross_entropy_gap / (classes * complexity_gap_per_class)

