from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModularArchitecture


class CausalTransformer(ModularArchitecture):
    """Canonical bias-free, no-LayerNorm, one-block causal Transformer."""

    architecture = "transformer"

    def __init__(
        self,
        modulus: int,
        seed: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_head: int = 32,
        d_mlp: int = 512,
    ) -> None:
        super().__init__()
        if n_heads * d_head != d_model:
            raise ValueError("canonical attention requires n_heads * d_head == d_model")
        torch.manual_seed(seed)
        self.modulus = int(modulus)
        self.K = self.modulus
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = int(d_head)
        self.d_mlp = int(d_mlp)
        self.W_E = nn.Parameter(torch.randn(modulus + 1, d_model) / math.sqrt(d_model))
        self.W_pos = nn.Parameter(torch.randn(3, d_model) / math.sqrt(d_model))
        self.W_Q = nn.Parameter(torch.randn(n_heads, d_model, d_head) / math.sqrt(d_model))
        self.W_K = nn.Parameter(torch.randn(n_heads, d_model, d_head) / math.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(n_heads, d_model, d_head) / math.sqrt(d_model))
        self.W_O = nn.Parameter(
            torch.randn(n_heads, d_head, d_model) / math.sqrt(n_heads * d_head)
        )
        self.W_in = nn.Parameter(torch.randn(d_model, d_mlp) / math.sqrt(d_model))
        self.W_out = nn.Parameter(torch.randn(d_mlp, d_model) / math.sqrt(d_mlp))
        self.W_U = nn.Parameter(torch.randn(d_model, modulus) / math.sqrt(d_model))

    def penultimate_features(self, pairs: torch.Tensor) -> torch.Tensor:
        equals = torch.full(
            (pairs.shape[0],), self.modulus, dtype=torch.long, device=pairs.device
        )
        tokens = torch.stack((pairs[:, 0], pairs[:, 1], equals), dim=1)
        residual = self.W_E[tokens] + self.W_pos[None, :, :]
        query = torch.einsum("bpd,hdf->bphf", residual, self.W_Q)
        key = torch.einsum("bpd,hdf->bphf", residual, self.W_K)
        value = torch.einsum("bpd,hdf->bphf", residual, self.W_V)
        scores = torch.einsum("bqhd,bkhd->bhqk", query, key) / math.sqrt(self.d_head)
        mask = torch.tril(torch.ones(3, 3, device=pairs.device, dtype=torch.bool))
        attention = torch.softmax(scores.masked_fill(~mask[None, None], -1e9), dim=-1)
        mixed = torch.einsum("bhqk,bkhd->bqhd", attention, value)
        residual = residual + torch.einsum("bqhd,hdf->bqf", mixed, self.W_O)
        residual = residual + F.relu(residual @ self.W_in) @ self.W_out
        return residual[:, -1]

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        return self.penultimate_features(pairs) @ self.W_U

    def embedding_matrix(self) -> torch.Tensor:
        return self.W_E[: self.modulus]

    def classifier_matrix(self) -> torch.Tensor:
        return self.W_U.T

    def embedding_parameters(self) -> list[nn.Parameter]:
        return [self.W_E]

