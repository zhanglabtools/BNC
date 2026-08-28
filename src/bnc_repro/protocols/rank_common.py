from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from bnc_repro.models.base import (
    ModularArchitecture,
    classifier_columns_from_rows,
    load_legacy_state_dict,
)
from bnc_repro.models.registry import build_model


def balanced_centered_factors(
    classifier_columns: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Balanced factors of the best class-centered rank-r approximation."""
    if classifier_columns.ndim != 2:
        raise ValueError("classifier must use the d x K column convention")
    centered = classifier_columns - classifier_columns.mean(dim=1, keepdim=True)
    u, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    effective = min(int(rank), singular_values.numel())
    scale = torch.sqrt(singular_values[:effective].clamp_min(0))
    return u[:, :effective] * scale[None, :], scale[:, None] * vh[:effective]


def cosine_learning_rate(epoch: int, total: int, maximum: float, minimum: float) -> float:
    progress = min(max(epoch / total, 0.0), 1.0)
    return minimum + 0.5 * (maximum - minimum) * (1.0 + math.cos(math.pi * progress))


def dense_checkpoint_path(
    dense_root: Path, architecture: str, modulus: int, seed: int
) -> Path:
    candidates = [
        dense_root / architecture / f"K{modulus}" / f"seed_{seed}" / "model_final.pt",
        dense_root / "dense" / architecture / f"K{modulus}" / f"seed_{seed}" / "model_final.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "dense checkpoint is required and was not supplied; tried: "
        + ", ".join(str(path) for path in candidates)
    )


def load_dense_model(
    config: dict[str, Any], architecture: str, modulus: int, seed: int, device: torch.device
) -> ModularArchitecture:
    dense_root = Path(config.get("dense_root", "outputs/dense")).resolve()
    checkpoint = dense_checkpoint_path(dense_root, architecture, modulus, seed)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("model_state_dict", payload)
    model = build_model(architecture, modulus, seed, config.get("model", {})).to(device)
    load_legacy_state_dict(model, state)
    return model


def remove_dense_classifier(model: ModularArchitecture) -> None:
    if hasattr(model, "W_U"):
        delattr(model, "W_U")
    elif hasattr(model, "head"):
        delattr(model, "head")
    else:
        raise TypeError("unsupported dense classifier layout")


class FactorizedArchitecture(nn.Module):
    """Canonical architecture body with explicit d x K classifier factors."""

    def __init__(
        self,
        base: ModularArchitecture,
        classifier_columns: torch.Tensor,
        rank: int,
        target_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.architecture = base.architecture
        self.modulus = base.modulus
        A, B = balanced_centered_factors(classifier_columns, rank)
        self.A = nn.Parameter(A.clone())
        self.B = nn.Parameter(B.clone())
        self.max_rank = int(A.shape[1])
        self.target_rank = int(target_rank if target_rank is not None else rank)
        self.tail_gate = 1.0
        remove_dense_classifier(base)
        self.base = base

    def hidden(self, pairs: torch.Tensor) -> torch.Tensor:
        return self.base.penultimate_features(pairs)

    def classifier(self) -> torch.Tensor:
        gates = self.A.new_ones(self.max_rank)
        gates[self.target_rank :] = self.tail_gate
        return (self.A * gates[None, :]) @ self.B

    def target_classifier(self) -> torch.Tensor:
        return self.A[:, : self.target_rank] @ self.B[: self.target_rank]

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        return self.hidden(pairs) @ self.classifier()

    def tail_factor_regularizer(self) -> torch.Tensor:
        tail_A = self.A[:, self.target_rank :]
        tail_B = self.B[self.target_rank :]
        if tail_A.numel() == 0 or tail_B.numel() == 0:
            return self.A.new_zeros(())
        return 0.5 * (tail_A.square().mean() + tail_B.square().mean())


def factorize_dense_model(
    model: ModularArchitecture, rank: int, target_rank: int | None = None
) -> FactorizedArchitecture:
    columns = classifier_columns_from_rows(model.classifier_matrix().detach().clone())
    return FactorizedArchitecture(model, columns, rank, target_rank)

