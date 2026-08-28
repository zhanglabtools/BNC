from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from bnc_repro.data.modular_addition import split_modular_addition
from bnc_repro.metrics.participation import spectrum_summary
from bnc_repro.metrics.role_conditioned import role_conditioned_codes
from bnc_repro.protocols.common import repository_root, resolve_device, set_seed, write_csv
from bnc_repro.protocols.rank_common import (
    FactorizedArchitecture,
    cosine_learning_rate,
    factorize_dense_model,
    load_dense_model,
)
from bnc_repro.training.checkpoints import atomic_json, runtime_metadata, save_model, save_resolved_config


def cosine_tail_gate(epoch: int, start: int, end: int) -> float:
    if epoch <= start:
        return 1.0
    if epoch >= end:
        return 0.0
    progress = (epoch - start) / (end - start)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def should_record_s1(epoch: int, total_epochs: int) -> bool:
    if epoch <= 20:
        return True
    if epoch <= 100:
        return epoch % 5 == 0
    if epoch <= 500:
        return epoch % 10 == 0
    if epoch <= 1000:
        return epoch % 25 == 0
    if epoch <= 7000:
        return epoch % 50 == 0
    return epoch % 100 == 0 or epoch == total_epochs


@torch.no_grad()
def _metric_row(model: FactorizedArchitecture, split, architecture: str, modulus: int, seed: int, lambda_reg: float, epoch: int, learning_rate: float) -> dict[str, Any]:
    model.eval()
    train_logits = model(split.train_pairs)
    test_logits = model(split.test_pairs)
    features = model.hidden(split.all_pairs)
    rx, ry = role_conditioned_codes(features, split.all_pairs, modulus)
    rx_spec = spectrum_summary(rx)
    ry_spec = spectrum_summary(ry)
    classifier_columns = model.classifier().to(torch.float64)
    classifier_spec = spectrum_summary(classifier_columns.T)
    singular_values = torch.linalg.svdvals(classifier_columns)
    tolerance = 1e-6 * singular_values.max().clamp_min(1e-30)
    numerical_rank = int((singular_values > tolerance).sum().item())
    row = {
        "architecture": architecture,
        "K": modulus,
        "seed": seed,
        "lambda_reg": lambda_reg,
        "epoch": epoch,
        "learning_rate": learning_rate,
        "train_loss": float(F.cross_entropy(train_logits, split.train_labels).item()),
        "test_loss": float(F.cross_entropy(test_logits, split.test_labels).item()),
        "train_accuracy": float((train_logits.argmax(-1) == split.train_labels).float().mean().item()),
        "test_accuracy": float((test_logits.argmax(-1) == split.test_labels).float().mean().item()),
        "rank_regularizer": float(model.tail_factor_regularizer().item()),
        "rx_participation_rank": rx_spec["participation_rank"],
        "ry_participation_rank": ry_spec["participation_rank"],
        "effective_mean_participation_rank": 0.5 * (rx_spec["participation_rank"] + ry_spec["participation_rank"]),
        "rx_entropy_rank": rx_spec["entropy_rank"],
        "ry_entropy_rank": ry_spec["entropy_rank"],
        "effective_mean_entropy_rank": 0.5 * (rx_spec["entropy_rank"] + ry_spec["entropy_rank"]),
        "rx_top2_tail": rx_spec["top2_tail"],
        "ry_top2_tail": ry_spec["top2_tail"],
        "effective_mean_top2_tail": 0.5 * (rx_spec["top2_tail"] + ry_spec["top2_tail"]),
        "classifier_participation_rank": classifier_spec["participation_rank"],
        "classifier_entropy_rank": classifier_spec["entropy_rank"],
        "classifier_top2_tail": classifier_spec["top2_tail"],
        "classifier_numerical_rank": numerical_rank,
        "tail_gate": float(model.tail_gate),
        "nominal_rank": model.target_rank if model.tail_gate == 0.0 else model.max_rank,
    }
    if model.tail_gate == 0.0 and numerical_rank > model.target_rank:
        raise RuntimeError("rank-homotopy final numerical rank invariant failed")
    return row


def run_rank_homotopy_grid(config: dict[str, Any], output_root: Path) -> list[Path]:
    completed: list[Path] = []
    device = resolve_device(str(config.get("device", "cuda")))
    training = config["training"]
    rank = config["rank"]
    for architecture in config["grid"]["architectures"]:
        for modulus in config["grid"]["moduli"]:
            for seed in config["grid"]["seeds"]:
                for lambda_reg in config["grid"]["lambdas"]:
                    architecture, modulus, seed = str(architecture), int(modulus), int(seed)
                    token = str(lambda_reg).replace(".", "p")
                    run_dir = output_root / "fig_s1" / architecture / f"K{modulus}" / f"seed_{seed}" / f"lambda_{token}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    set_seed(seed)
                    split = split_modular_addition(modulus, float(config.get("train_fraction", 0.7)), seed, device)
                    dense = load_dense_model(config, architecture, modulus, seed, device)
                    model = factorize_dense_model(
                        dense, int(rank["max_rank"]), int(rank["target_rank"])
                    ).to(device)
                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=float(training["lr"]),
                        weight_decay=float(training["weight_decay"]),
                        betas=tuple(float(v) for v in training["betas"]),
                    )
                    save_resolved_config(run_dir / "config_resolved.yaml", config)
                    rows = [_metric_row(model, split, architecture, modulus, seed, float(lambda_reg), 0, float(training["lr"]))]
                    epochs = int(training["epochs"])
                    for epoch in range(1, epochs + 1):
                        model.tail_gate = cosine_tail_gate(epoch, int(rank["gate_start"]), int(rank["gate_end"]))
                        learning_rate = cosine_learning_rate(epoch, epochs, float(training["lr"]), float(training["min_lr"]))
                        for group in optimizer.param_groups:
                            group["lr"] = learning_rate
                        model.train()
                        optimizer.zero_grad(set_to_none=True)
                        hidden = model.hidden(split.train_pairs)
                        full_loss = F.cross_entropy(hidden @ model.classifier(), split.train_labels)
                        target_loss = F.cross_entropy(hidden @ model.target_classifier(), split.train_labels)
                        loss = full_loss + float(training["target_loss_weight"]) * target_loss + float(lambda_reg) * model.tail_factor_regularizer()
                        if not torch.isfinite(loss):
                            raise FloatingPointError("non-finite rank-homotopy loss")
                        loss.backward()
                        optimizer.step()
                        if should_record_s1(epoch, epochs):
                            rows.append(_metric_row(model, split, architecture, modulus, seed, float(lambda_reg), epoch, learning_rate))
                    write_csv(run_dir / "metrics.csv", rows)
                    save_model(run_dir / "model_final.pt", model, {"epoch": epochs})
                    atomic_json(run_dir / "status.json", {"status": "complete", "metric_rows": len(rows), "runtime": runtime_metadata(device, repository_root())})
                    completed.append(run_dir)
    return completed

