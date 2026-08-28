from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModularArchitecture


class RoleSpecificMLP(ModularArchitecture):
    """Bias-free one-hidden-layer ReLU MLP with separate operand codebooks."""

    architecture = "mlp"

    def __init__(self, modulus: int, embedding_dim: int, hidden_dim: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.modulus = int(modulus)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.W_x = nn.Parameter(torch.randn(modulus, embedding_dim) / math.sqrt(embedding_dim))
        self.W_y = nn.Parameter(torch.randn(modulus, embedding_dim) / math.sqrt(embedding_dim))
        self.W = nn.Parameter(
            torch.randn(2 * embedding_dim, hidden_dim) / math.sqrt(2 * embedding_dim)
        )
        self.W_U = nn.Parameter(torch.randn(hidden_dim, modulus) / math.sqrt(hidden_dim))

    def penultimate_features(self, pairs: torch.Tensor) -> torch.Tensor:
        first = self.W_x[pairs[:, 0]]
        second = self.W_y[pairs[:, 1]]
        return F.relu(torch.cat((first, second), dim=1) @ self.W)

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        return self.penultimate_features(pairs) @ self.W_U

    def embedding_matrix(self) -> torch.Tensor:
        return self.W_x

    def classifier_matrix(self) -> torch.Tensor:
        return self.W_U.T

    def embedding_parameters(self) -> list[nn.Parameter]:
        return [self.W_x, self.W_y]

