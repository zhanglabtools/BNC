from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModularSplit:
    all_pairs: torch.Tensor
    all_labels: torch.Tensor
    train_pairs: torch.Tensor
    train_labels: torch.Tensor
    test_pairs: torch.Tensor
    test_labels: torch.Tensor
    train_indices: torch.Tensor
    test_indices: torch.Tensor


def modular_pairs(modulus: int, device: str | torch.device = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    if modulus < 2:
        raise ValueError("modulus must be at least two")
    values = torch.arange(modulus, dtype=torch.long, device=device)
    pairs = torch.cartesian_prod(values, values)
    labels = (pairs[:, 0] + pairs[:, 1]) % modulus
    return pairs, labels


def split_modular_addition(
    modulus: int,
    train_fraction: float,
    split_seed: int,
    device: str | torch.device = "cpu",
) -> ModularSplit:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie in (0, 1)")
    cpu_pairs, cpu_labels = modular_pairs(modulus, "cpu")
    generator = torch.Generator().manual_seed(int(split_seed))
    permutation = torch.randperm(modulus * modulus, generator=generator)
    split = int(train_fraction * modulus * modulus)
    train_indices = permutation[:split]
    test_indices = permutation[split:]
    target = torch.device(device)
    return ModularSplit(
        all_pairs=cpu_pairs.to(target),
        all_labels=cpu_labels.to(target),
        train_pairs=cpu_pairs[train_indices].to(target),
        train_labels=cpu_labels[train_indices].to(target),
        test_pairs=cpu_pairs[test_indices].to(target),
        test_labels=cpu_labels[test_indices].to(target),
        train_indices=train_indices,
        test_indices=test_indices,
    )

