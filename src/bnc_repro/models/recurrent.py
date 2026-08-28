from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import ModularArchitecture


class SharedRecurrent(ModularArchitecture):
    """Shared K+1 codebook with an independent bias-free LSTM/RNN classifier."""

    def __init__(
        self,
        architecture: str,
        modulus: int,
        seed: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if architecture not in {"lstm", "rnn"}:
            raise ValueError(architecture)
        if embedding_dim != hidden_dim:
            raise ValueError("canonical recurrent models use equal embedding/hidden dimensions")
        torch.manual_seed(seed)
        self.architecture = architecture
        self.modulus = int(modulus)
        self.K = self.modulus
        self.embedding = nn.Embedding(modulus + 1, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=1.0 / math.sqrt(embedding_dim))
        if architecture == "lstm":
            self.recurrent = nn.LSTM(
                embedding_dim, hidden_dim, num_layers=1, bias=False, batch_first=True
            )
        else:
            self.recurrent = nn.RNN(
                embedding_dim,
                hidden_dim,
                num_layers=1,
                nonlinearity="relu",
                bias=False,
                batch_first=True,
            )
        self.head = nn.Linear(hidden_dim, modulus, bias=False)

    def penultimate_features(self, pairs: torch.Tensor) -> torch.Tensor:
        equals = torch.full(
            (pairs.shape[0],), self.modulus, dtype=torch.long, device=pairs.device
        )
        tokens = torch.stack((pairs[:, 0], pairs[:, 1], equals), dim=1)
        output, _ = self.recurrent(self.embedding(tokens))
        return output[:, -1]

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        return self.head(self.penultimate_features(pairs))

    def embedding_matrix(self) -> torch.Tensor:
        return self.embedding.weight[: self.modulus]

    def classifier_matrix(self) -> torch.Tensor:
        return self.head.weight

    def embedding_parameters(self) -> list[nn.Parameter]:
        return [self.embedding.weight]

