from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bnc_repro.data.modular_addition import split_modular_addition
from bnc_repro.models.registry import build_model
from bnc_repro.protocols.common import repository_root, resolve_device, set_seed
from bnc_repro.training.checkpoints import atomic_json, runtime_metadata, save_model, save_resolved_config


def _pca_coordinates(matrix: torch.Tensor) -> np.ndarray:
    centered = matrix.detach().to(torch.float64) - matrix.detach().to(torch.float64).mean(
        dim=0, keepdim=True
    )
    u, singular_values, _ = torch.linalg.svd(centered, full_matrices=False)
    coordinates = u[:, :2] * singular_values[:2]
    radius = torch.sqrt(coordinates.square().sum(dim=1).mean()).clamp_min(1e-20)
    return (coordinates / radius).cpu().numpy()


def _display_values(coordinates: np.ndarray) -> np.ndarray:
    modulus = coordinates.shape[0]
    values = coordinates[:, 0] + 1j * coordinates[:, 1]
    values = values / np.maximum(np.abs(values), 1e-12)
    labels = np.arange(modulus)
    best_score, best_multiplier, best_orientation = -1.0, 1, 1
    for multiplier in range(1, modulus):
        if math.gcd(multiplier, modulus) != 1:
            continue
        target = np.exp(2j * np.pi * ((multiplier * labels) % modulus) / modulus)
        for orientation, score in (
            (1, abs(np.mean(values * np.conjugate(target)))),
            (-1, abs(np.mean(values * target))),
        ):
            if score > best_score:
                best_score, best_multiplier, best_orientation = score, multiplier, orientation
    if best_orientation < 0:
        best_multiplier = (-best_multiplier) % modulus
    return (best_multiplier * labels) % modulus


def run_fig1_grid(config: dict, output_root: Path) -> list[Path]:
    device = resolve_device(str(config.get("device", "cuda")))
    training = config["training"]
    completed = []
    for seed in config["grid"]["seeds"]:
        seed = int(seed)
        modulus = int(config["grid"]["moduli"][0])
        run_dir = output_root / "fig1" / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        set_seed(seed)
        split = split_modular_addition(modulus, float(config.get("train_fraction", 0.7)), seed, device)
        model = build_model("mlp", modulus, seed, config["model"]).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["lr"]),
            betas=tuple(float(value) for value in training["betas"]),
            weight_decay=float(training["weight_decay"]),
        )
        save_resolved_config(run_dir / "config_resolved.yaml", config)
        for _epoch in range(1, int(training["epochs"]) + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(split.train_pairs), split.train_labels)
            loss.backward()
            optimizer.step()
        embedding = _pca_coordinates(model.embedding_matrix())
        classifier = _pca_coordinates(model.classifier_matrix())
        np.savez(
            run_dir / "pca_coordinates.npz",
            embedding_coordinates=embedding,
            classifier_coordinates=classifier,
            display_values=_display_values(embedding),
        )
        save_model(run_dir / "model_final.pt", model, {"epoch": int(training["epochs"])})
        atomic_json(
            run_dir / "status.json",
            {"status": "complete", "runtime": runtime_metadata(device, repository_root())},
        )
        completed.append(run_dir)
    return completed

