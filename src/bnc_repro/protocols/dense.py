from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from bnc_repro.data.modular_addition import split_modular_addition
from bnc_repro.models.registry import build_model
from bnc_repro.protocols.common import grid, repository_root, resolve_device, set_seed
from bnc_repro.training.checkpoints import (
    atomic_json,
    runtime_metadata,
    save_model,
    save_resolved_config,
)
from bnc_repro.training.optim import grouped_adamw


def run_dense_grid(config: dict[str, Any], output_root: Path) -> list[Path]:
    completed: list[Path] = []
    training = config["training"]
    device = resolve_device(str(config.get("device", "cuda")))
    for architecture, modulus, seed in grid(config):
        run_dir = output_root / "dense" / architecture / f"K{modulus}" / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        set_seed(seed)
        split_seed = int(config.get("split_seed", seed))
        split = split_modular_addition(modulus, float(config.get("train_fraction", 0.7)), split_seed, device)
        model = build_model(architecture, modulus, seed, config.get("model", {})).to(device)
        optimizer = grouped_adamw(
            model,
            embedding_lr=float(training["embedding_lr"]),
            other_lr=float(training["other_lr"]),
            embedding_weight_decay=float(training.get("embedding_weight_decay", 0.0)),
            other_weight_decay=float(training["other_weight_decay"]),
            betas=tuple(float(v) for v in training["betas"]),
        )
        save_resolved_config(run_dir / "config_resolved.yaml", config)
        np.savez(
            run_dir / "split_indices.npz",
            train_indices=split.train_indices.numpy(),
            test_indices=split.test_indices.numpy(),
        )
        atomic_json(
            run_dir / "status.json",
            {
                "status": "running",
                "architecture": architecture,
                "K": modulus,
                "seed": seed,
                "split_seed": split_seed,
                "runtime": runtime_metadata(device, repository_root()),
            },
        )
        model.train()
        epochs = int(training["epochs"])
        for _epoch in range(1, epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(split.train_pairs), split.train_labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite dense-training loss")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            test_accuracy = float(
                (model(split.test_pairs).argmax(dim=-1) == split.test_labels)
                .float()
                .mean()
                .item()
            )
        save_model(
            run_dir / "model_final.pt",
            model,
            {"architecture": architecture, "K": modulus, "seed": seed, "epoch": epochs},
        )
        atomic_json(
            run_dir / "status.json",
            {
                "status": "complete",
                "architecture": architecture,
                "K": modulus,
                "seed": seed,
                "epoch": epochs,
                "test_accuracy": test_accuracy,
                "runtime": runtime_metadata(device, repository_root()),
            },
        )
        completed.append(run_dir)
    return completed

