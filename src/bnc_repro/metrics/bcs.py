from __future__ import annotations

import math

import numpy as np
import torch


def pca2_coordinates(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2 or matrix.shape[0] < 3:
        raise ValueError(f"BCS requires a K x d matrix with K >= 3, got {matrix.shape}")
    centered = matrix.to(torch.float64) - matrix.to(torch.float64).mean(dim=0, keepdim=True)
    u, singular_values, _ = torch.linalg.svd(centered, full_matrices=False)
    if singular_values.numel() < 2:
        raise ValueError("BCS requires at least two available PCA directions")
    coordinates = u[:, :2] * singular_values[:2]
    radius = torch.sqrt(coordinates.square().sum(dim=1).mean())
    if not torch.isfinite(radius) or radius <= 1e-20:
        raise ValueError("BCS is undefined for a degenerate centered matrix")
    return coordinates / radius


def best_cyclic_score(matrix: torch.Tensor, modulus: int | None = None) -> float:
    """Exact July-2026 BCS with automorphism and orientation search.

    Rows are tokens/classes. The score centers rows, obtains SVD PCA coordinates
    ``U[:, :2] * S[:2]``, RMS-normalizes, projects every point to the unit
    circle, and searches every unit of Z_K and both orientations.
    """

    modulus = int(matrix.shape[0] if modulus is None else modulus)
    if matrix.shape[0] != modulus:
        raise ValueError(f"matrix has {matrix.shape[0]} rows but modulus={modulus}")
    coordinates = pca2_coordinates(matrix).detach().cpu().numpy()
    complex_coordinates = coordinates[:, 0] + 1j * coordinates[:, 1]
    magnitudes = np.abs(complex_coordinates)
    if np.any(magnitudes <= 1e-12):
        raise ValueError("BCS unit-circle projection encountered a near-zero point")
    complex_coordinates = complex_coordinates / magnitudes
    labels = np.arange(modulus)
    best = 0.0
    for multiplier in range(1, modulus):
        if math.gcd(multiplier, modulus) != 1:
            continue
        phase = 2.0 * np.pi * ((multiplier * labels) % modulus) / modulus
        target = np.exp(1j * phase)
        best = max(
            best,
            float(abs(np.mean(complex_coordinates * np.conjugate(target)))),
            float(abs(np.mean(complex_coordinates * target))),
        )
    return float(min(max(best, 0.0), 1.0))

