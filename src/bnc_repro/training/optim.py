from __future__ import annotations

import torch

from bnc_repro.models.base import ModularArchitecture


def grouped_adamw(
    model: ModularArchitecture,
    *,
    embedding_lr: float,
    other_lr: float,
    embedding_weight_decay: float,
    other_weight_decay: float,
    betas: tuple[float, float],
) -> torch.optim.AdamW:
    embedding = model.embedding_parameters()
    ids = {id(parameter) for parameter in embedding}
    other = [parameter for parameter in model.parameters() if id(parameter) not in ids]
    return torch.optim.AdamW(
        [
            {
                "params": embedding,
                "lr": embedding_lr,
                "weight_decay": embedding_weight_decay,
            },
            {"params": other, "lr": other_lr, "weight_decay": other_weight_decay},
        ],
        betas=betas,
    )

