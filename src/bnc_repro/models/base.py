from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
import torch.nn as nn


class ModularArchitecture(nn.Module, ABC):
    architecture: str
    modulus: int

    @abstractmethod
    def forward(self, pairs: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def penultimate_features(self, pairs: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def embedding_matrix(self) -> torch.Tensor: ...

    @abstractmethod
    def classifier_matrix(self) -> torch.Tensor: ...

    @abstractmethod
    def embedding_parameters(self) -> list[nn.Parameter]: ...


def classifier_columns_from_rows(rows: torch.Tensor) -> torch.Tensor:
    """Convert the public K x d class-row convention to d x K classifier columns."""
    if rows.ndim != 2:
        raise ValueError(f"classifier rows must be rank two, got {rows.shape}")
    return rows.transpose(0, 1).contiguous()


def classifier_rows_from_columns(columns: torch.Tensor) -> torch.Tensor:
    """Convert d x K classifier columns to the public K x d class-row convention."""
    if columns.ndim != 2:
        raise ValueError(f"classifier columns must be rank two, got {columns.shape}")
    return columns.transpose(0, 1).contiguous()


def migrate_legacy_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove common wrappers while preserving canonical July-2026 parameter names."""
    migrated: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        normalized = key
        for prefix in ("module.", "model."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
        migrated[normalized] = value
    return migrated


def load_legacy_state_dict(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    result = model.load_state_dict(migrate_legacy_state_dict(state), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"state-dict migration failed: {result}")

