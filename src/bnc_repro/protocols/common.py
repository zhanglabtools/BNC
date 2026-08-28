from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {requested}")
    return torch.device(requested)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


@torch.no_grad()
def loss_accuracy(
    model: torch.nn.Module, pairs: torch.Tensor, labels: torch.Tensor
) -> tuple[float, float]:
    model.eval()
    logits = model(pairs)
    return (
        float(F.cross_entropy(logits, labels).item()),
        float((logits.argmax(dim=-1) == labels).float().mean().item()),
    )


def geometric_checkpoint_epochs(epochs: int, requested: int) -> list[int]:
    if epochs < 1 or requested < 2:
        raise ValueError("epochs must be positive and requested must be at least two")
    values = np.rint(np.geomspace(1, epochs, requested)).astype(int)
    return [int(value) for value in np.unique(np.clip(values, 1, epochs))]


def fixed_permutations(modulus: int, seed: int, controls: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator().manual_seed(10000 + int(seed))
    return torch.stack(
        [torch.randperm(modulus, generator=generator) for _ in range(controls)]
    ).to(device)


def grid(config: dict[str, Any]) -> Iterable[tuple[str, int, int]]:
    cfg = config["grid"]
    for architecture in cfg["architectures"]:
        for modulus in cfg["moduli"]:
            for seed in cfg["seeds"]:
                yield str(architecture), int(modulus), int(seed)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]

