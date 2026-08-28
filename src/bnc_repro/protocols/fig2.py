from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from bnc_repro.data.modular_addition import split_modular_addition
from bnc_repro.metrics.bcs import best_cyclic_score
from bnc_repro.models.registry import build_model
from bnc_repro.protocols.common import (
    grid,
    loss_accuracy,
    repository_root,
    resolve_device,
    set_seed,
    write_csv,
)
from bnc_repro.training.checkpoints import atomic_json, runtime_metadata, save_model, save_resolved_config
from bnc_repro.training.optim import grouped_adamw


def _metric_row(model, architecture: str, modulus: int, seed: int, epoch: int, split) -> dict[str, Any]:
    train_loss, train_accuracy = loss_accuracy(model, split.train_pairs, split.train_labels)
    _, test_accuracy = loss_accuracy(model, split.test_pairs, split.test_labels)
    return {
        "architecture": architecture,
        "K": modulus,
        "seed": seed,
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "classifier_best_cyclic_score": best_cyclic_score(model.classifier_matrix(), modulus),
        "embedding_best_cyclic_score": best_cyclic_score(model.embedding_matrix(), modulus),
    }


def run_fig2_grid(config: dict[str, Any], output_root: Path) -> list[Path]:
    completed: list[Path] = []
    training = config["training"]
    device = resolve_device(str(config.get("device", "cuda")))
    for architecture, modulus, seed in grid(config):
        run_dir = output_root / "fig2" / architecture / f"K{modulus}" / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        set_seed(seed)
        split_seed = int(config.get("split_seed", seed))
        split = split_modular_addition(modulus, float(config.get("train_fraction", 0.7)), split_seed, device)
        model = build_model(architecture, modulus, seed, config.get("model", {})).to(device)
        optimizer = grouped_adamw(
            model,
            embedding_lr=float(training["embedding_lr"]),
            other_lr=float(training["other_lr"]),
            embedding_weight_decay=float(training["embedding_weight_decay"]),
            other_weight_decay=float(training["other_weight_decay"]),
            betas=tuple(float(v) for v in training["betas"]),
        )
        save_resolved_config(run_dir / "config_resolved.yaml", config)
        atomic_json(run_dir / "status.json", {"status": "running", "runtime": runtime_metadata(device, repository_root())})
        rows = [_metric_row(model, architecture, modulus, seed, 0, split)]
        epochs = int(training["epochs"])
        log_every = int(training["log_every"])
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(split.train_pairs), split.train_labels)
            loss.backward()
            optimizer.step()
            if epoch % log_every == 0 or epoch == epochs:
                rows.append(_metric_row(model, architecture, modulus, seed, epoch, split))
        write_csv(run_dir / "metrics.csv", rows)
        if bool(training.get("save_final_checkpoint", True)):
            save_model(run_dir / "model_final.pt", model, {"epoch": epochs})
        atomic_json(
            run_dir / "status.json",
            {"status": "complete", "epoch": epochs, "metric_rows": len(rows), "runtime": runtime_metadata(device, repository_root())},
        )
        completed.append(run_dir)
    return completed

