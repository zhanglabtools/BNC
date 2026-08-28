from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from bnc_repro.data.modular_addition import split_modular_addition
from bnc_repro.metrics.feature_classifier import class_means, centered_feature_classifier_alignment
from bnc_repro.metrics.token_geometry import fixed_shuffle_control, token_geometry_correlation
from bnc_repro.models.registry import build_model
from bnc_repro.protocols.common import (
    fixed_permutations,
    geometric_checkpoint_epochs,
    grid,
    loss_accuracy,
    repository_root,
    resolve_device,
    set_seed,
    write_csv,
)
from bnc_repro.training.checkpoints import atomic_json, runtime_metadata, save_resolved_config
from bnc_repro.training.optim import grouped_adamw


@torch.no_grad()
def _record(model, split, permutations, metrics: list[str], architecture: str, modulus: int, seed: int, epoch: int) -> dict[str, Any]:
    _, train_accuracy = loss_accuracy(model, split.train_pairs, split.train_labels)
    _, test_accuracy = loss_accuracy(model, split.test_pairs, split.test_labels)
    row: dict[str, Any] = {
        "architecture": architecture,
        "K": modulus,
        "seed": seed,
        "epoch": epoch,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
    }
    if "token_geometry" in metrics:
        embedding = model.embedding_matrix().detach().to(torch.float64)
        classifier = model.classifier_matrix().detach().to(torch.float64)
        row["matched_alignment"] = token_geometry_correlation(embedding, classifier)
        row["shuffled_alignment"] = fixed_shuffle_control(embedding, classifier, permutations)
    if "centered_feature_classifier" in metrics:
        features = model.penultimate_features(split.all_pairs).detach().to(torch.float64)
        means = class_means(features, split.all_labels, modulus)
        matched, shuffled = centered_feature_classifier_alignment(
            means, model.classifier_matrix().detach().to(torch.float64), permutations
        )
        row["centered_feature_classifier_alignment"] = matched
        row["shuffled_centered_feature_classifier_alignment"] = shuffled
    return row


def run_alignment_grid(config: dict[str, Any], output_root: Path) -> list[Path]:
    completed: list[Path] = []
    training = config["training"]
    metric_names = [str(value) for value in config["metrics"]["names"]]
    device = resolve_device(str(config.get("device", "cuda")))
    for architecture, modulus, seed in grid(config):
        run_dir = output_root / config["experiment"] / architecture / f"K{modulus}" / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        set_seed(seed)
        split = split_modular_addition(modulus, float(config.get("train_fraction", 0.7)), seed, device)
        model = build_model(architecture, modulus, seed, config.get("model", {})).to(device)
        optimizer = grouped_adamw(
            model,
            embedding_lr=float(training["embedding_lr"]),
            other_lr=float(training["other_lr"]),
            embedding_weight_decay=float(training["embedding_weight_decay"]),
            other_weight_decay=float(training["other_weight_decay"]),
            betas=tuple(float(v) for v in training["betas"]),
        )
        permutations = fixed_permutations(
            modulus, seed, int(config["metrics"]["shuffle_controls"]), device
        )
        schedule = geometric_checkpoint_epochs(
            int(training["epochs"]), int(training["requested_checkpoints"])
        )
        schedule_set = set(schedule)
        save_resolved_config(run_dir / "config_resolved.yaml", config)
        atomic_json(
            run_dir / "status.json",
            {"status": "running", "checkpoint_epochs": schedule, "runtime": runtime_metadata(device, repository_root())},
        )
        rows: list[dict[str, Any]] = []
        for epoch in range(1, int(training["epochs"]) + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(split.train_pairs), split.train_labels)
            loss.backward()
            optimizer.step()
            if epoch in schedule_set:
                rows.append(_record(model, split, permutations, metric_names, architecture, modulus, seed, epoch))
        write_csv(run_dir / "metrics.csv", rows)
        atomic_json(
            run_dir / "status.json",
            {"status": "complete", "metric_rows": len(rows), "checkpoint_epochs": schedule, "runtime": runtime_metadata(device, repository_root())},
        )
        completed.append(run_dir)
    return completed

