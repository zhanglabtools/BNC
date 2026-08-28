from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from bnc_repro.data.modular_addition import split_modular_addition
from bnc_repro.metrics.participation import spectrum_summary
from bnc_repro.metrics.role_conditioned import participation_target_penalty, role_conditioned_codes
from bnc_repro.protocols.common import grid, repository_root, resolve_device, set_seed, write_csv
from bnc_repro.protocols.rank_common import cosine_learning_rate, factorize_dense_model, load_dense_model
from bnc_repro.training.checkpoints import atomic_json, runtime_metadata, save_model, save_resolved_config


def should_record_s2(epoch: int, total_epochs: int) -> bool:
    if epoch <= 200:
        return True
    if epoch <= 1000:
        return epoch % 5 == 0
    return epoch % 25 == 0 or epoch == total_epochs


def representation_tail_balance(model, rx: torch.Tensor, ry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    decision_basis = torch.linalg.qr(model.A, mode="reduced").Q
    identity = torch.eye(2, dtype=rx.dtype, device=rx.device) / 2.0
    tails: list[torch.Tensor] = []
    balances: list[torch.Tensor] = []
    for code in (rx, ry):
        projected = code @ decision_basis
        residual = code - projected @ decision_basis.T
        tails.append(residual.square().sum() / code.square().sum().clamp_min(1e-12))
        covariance = projected.T @ projected
        normalized = covariance / covariance.trace().clamp_min(1e-12)
        balances.append((normalized - identity).square().sum())
    return torch.stack(tails).mean(), torch.stack(balances).mean()


@torch.no_grad()
def _metric_row(model, split, architecture: str, modulus: int, seed: int, epoch: int, learning_rate: float) -> dict[str, Any]:
    model.eval()
    train_logits = model(split.train_pairs)
    test_logits = model(split.test_pairs)
    features = model.hidden(split.all_pairs)
    rx, ry = role_conditioned_codes(features, split.all_pairs, modulus)
    rx_spec, ry_spec = spectrum_summary(rx), spectrum_summary(ry)
    classifier = model.classifier().to(torch.float64)
    classifier_spec = spectrum_summary(classifier.T)
    singular_values = torch.linalg.svdvals(classifier)
    tolerance = 1e-6 * singular_values.max().clamp_min(1e-30)
    numerical_rank = int((singular_values > tolerance).sum().item())
    if numerical_rank > 2:
        raise RuntimeError(f"explicit rank-2 head has numerical rank {numerical_rank}")
    return {
        "architecture": architecture,
        "K": modulus,
        "seed": seed,
        "epoch": epoch,
        "learning_rate": learning_rate,
        "train_loss": float(F.cross_entropy(train_logits, split.train_labels).item()),
        "test_loss": float(F.cross_entropy(test_logits, split.test_labels).item()),
        "train_accuracy": float((train_logits.argmax(-1) == split.train_labels).float().mean().item()),
        "test_accuracy": float((test_logits.argmax(-1) == split.test_labels).float().mean().item()),
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
    }


def run_rank2_grid(config: dict[str, Any], output_root: Path) -> list[Path]:
    completed: list[Path] = []
    device = resolve_device(str(config.get("device", "cuda")))
    training, regularizer = config["training"], config["regularizer"]
    for architecture, modulus, seed in grid(config):
        run_dir = output_root / "fig_s2" / architecture / f"K{modulus}" / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        set_seed(seed)
        split = split_modular_addition(modulus, float(config.get("train_fraction", 0.7)), seed, device)
        dense = load_dense_model(config, architecture, modulus, seed, device)
        model = factorize_dense_model(dense, 2, 2).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["lr"]),
            weight_decay=float(training["weight_decay"]),
            betas=tuple(float(v) for v in training["betas"]),
        )
        save_resolved_config(run_dir / "config_resolved.yaml", config)
        rows = [_metric_row(model, split, architecture, modulus, seed, 0, float(training["lr"]))]
        epochs = int(training["epochs"])
        for epoch in range(1, epochs + 1):
            learning_rate = cosine_learning_rate(epoch, epochs, float(training["lr"]), float(training["min_lr"]))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            model.train()
            optimizer.zero_grad(set_to_none=True)
            all_features = model.hidden(split.all_pairs)
            train_features = all_features[split.train_indices.to(device)]
            classification_loss = F.cross_entropy(train_features @ model.classifier(), split.train_labels)
            rx, ry = role_conditioned_codes(all_features, split.all_pairs, modulus)
            participation_penalty, _ = participation_target_penalty(rx, ry, 2.0)
            tail_penalty, balance_penalty = representation_tail_balance(model, rx, ry)
            age = max(0, epoch - int(regularizer.get("start_epoch", 0)))
            ramp = int(regularizer["ramp_epochs"])
            scale = 0.0 if age == 0 else (1.0 if ramp == 0 else min(1.0, age / ramp))
            penalty = (
                float(regularizer["tail_weight"]) * tail_penalty
                + float(regularizer["balance_weight"]) * balance_penalty
                + float(regularizer["participation_weight"]) * participation_penalty
            )
            loss = classification_loss + float(regularizer["overall_coefficient"]) * scale * penalty
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite rank-2 fine-tuning loss")
            loss.backward()
            optimizer.step()
            if should_record_s2(epoch, epochs):
                rows.append(_metric_row(model, split, architecture, modulus, seed, epoch, learning_rate))
        write_csv(run_dir / "metrics.csv", rows)
        save_model(run_dir / "model_final.pt", model, {"epoch": epochs})
        atomic_json(run_dir / "status.json", {"status": "complete", "metric_rows": len(rows), "runtime": runtime_metadata(device, repository_root())})
        completed.append(run_dir)
    return completed

