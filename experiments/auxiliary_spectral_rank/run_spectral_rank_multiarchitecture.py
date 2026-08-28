#!/usr/bin/env python3
"""Soft rank-homotopy fine-tuning for four supplied architectures.

The dense models and architecture implementations come from the audited
``classifier_first_multiarchitecture_K79_K97_K113_20260721`` project.  This
script does not relabel an MLP run.  It initializes each real architecture's
classifier from a centered rank-16 SVD, then continuously gates channels 3--16
to zero while channels 1--2 remain active.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ARCHITECTURES = ("mlp", "transformer", "lstm", "rnn")
K_VALUES = (79, 97, 113)
SEEDS = (1, 2, 3, 4, 5)
METRIC_FIELDS = [
    "architecture", "K", "seed", "epoch", "learning_rate",
    "train_loss", "test_loss", "train_accuracy", "test_accuracy",
    "rx_participation_rank", "ry_participation_rank",
    "effective_mean_participation_rank",
    "rx_entropy_rank", "ry_entropy_rank", "effective_mean_entropy_rank",
    "rx_top2_tail", "ry_top2_tail", "effective_mean_top2_tail",
    "classifier_participation_rank", "classifier_entropy_rank",
    "classifier_top2_tail", "classifier_numerical_rank",
    "tail_gate", "nominal_rank",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES,
                        default=list(ARCHITECTURES))
    parser.add_argument("--k-values", nargs="+", type=int, default=list(K_VALUES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument(
        "--architecture-source",
        type=Path,
        default=root.parent / "classifier_first_multiarchitecture_K79_K97_K113_20260721"
        / "run_classifier_first_multiarchitecture.py",
    )
    parser.add_argument(
        "--dense-root",
        type=Path,
        default=root.parent / "classifier_first_multiarchitecture_K79_K97_K113_20260721"
        / "outputs",
    )
    parser.add_argument("--output-root", type=Path, default=root / "outputs")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--frac-train", type=float, default=0.7)
    parser.add_argument("--max-rank", type=int, default=16)
    parser.add_argument("--target-rank", type=int, default=2)
    parser.add_argument("--gate-start", type=int, default=1)
    parser.add_argument("--gate-end", type=int, default=6000)
    parser.add_argument("--target-loss-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=[])
    parser.add_argument("--max-parallel-per-gpu", type=int, default=1)
    parser.add_argument("--worker-architecture", choices=ARCHITECTURES)
    parser.add_argument("--worker-k", type=int)
    parser.add_argument("--worker-seed", type=int)
    args = parser.parse_args()
    worker_values = (args.worker_architecture, args.worker_k, args.worker_seed)
    if any(value is not None for value in worker_values) and not all(
        value is not None for value in worker_values
    ):
        parser.error("worker architecture, K, and seed must be supplied together")
    if args.epochs <= 0 or args.max_parallel_per_gpu <= 0:
        parser.error("epochs and max-parallel-per-gpu must be positive")
    if not 1 <= args.target_rank < args.max_rank:
        parser.error("target-rank must be positive and smaller than max-rank")
    if args.gate_end <= args.gate_start:
        parser.error("gate-end must be greater than gate-start")
    return args


def load_architecture_source(path: Path):
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Architecture source not found: {path}")
    spec = importlib.util.spec_from_file_location("supplied_multiarchitecture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import architecture source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_metrics(path: Path, max_epoch: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            epoch = int(raw["epoch"])
            if epoch > max_epoch:
                continue
            row: dict[str, Any] = {"architecture": raw["architecture"]}
            for field in METRIC_FIELDS[1:]:
                row[field] = int(raw[field]) if field in {
                    "K", "seed", "epoch", "classifier_numerical_rank", "nominal_rank"
                } else float(raw[field])
            rows.append(row)
    return rows


def balanced_centered_rank_factors(
    classifier: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return balanced factors for the best centered rank-r approximation."""
    centered = classifier - classifier.mean(dim=1, keepdim=True)
    u, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    effective_rank = min(rank, singular_values.numel())
    scale = torch.sqrt(singular_values[:effective_rank].clamp_min(0))
    return (
        u[:, :effective_rank] * scale[None, :],
        scale[:, None] * vh[:effective_rank],
    )


class SoftRankHomotopyArchitecture(nn.Module):
    """Supplied architecture body plus a gated centered-SVD classifier."""

    def __init__(
        self,
        base: nn.Module,
        architecture: str,
        classifier: torch.Tensor,
        max_rank: int,
        target_rank: int,
    ):
        super().__init__()
        self.base = base
        self.architecture = architecture
        self.K = int(classifier.shape[1])
        A, B = balanced_centered_rank_factors(classifier, max_rank)
        self.A = nn.Parameter(A.clone())
        self.B = nn.Parameter(B.clone())
        self.max_rank = int(A.shape[1])
        self.target_rank = int(target_rank)
        self.tail_gate = 1.0

        if architecture in {"mlp", "transformer"}:
            delattr(self.base, "W_U")
        else:
            delattr(self.base, "head")

    def classifier(self) -> torch.Tensor:
        gates = self.A.new_ones(self.max_rank)
        gates[self.target_rank :] = self.tail_gate
        return (self.A * gates[None, :]) @ self.B

    def target_classifier(self) -> torch.Tensor:
        return self.A[:, : self.target_rank] @ self.B[: self.target_rank]

    def hidden(self, pairs: torch.Tensor) -> torch.Tensor:
        if self.architecture == "mlp":
            first = self.base.W_x[pairs[:, 0]]
            second = self.base.W_y[pairs[:, 1]]
            return F.relu(torch.cat((first, second), dim=1) @ self.base.W)

        equals = torch.full(
            (pairs.shape[0],), self.K, dtype=torch.long, device=pairs.device
        )
        tokens = torch.stack((pairs[:, 0], pairs[:, 1], equals), dim=1)
        if self.architecture == "transformer":
            residual = self.base.W_E[tokens] + self.base.W_pos[None, :, :]
            query = torch.einsum("bpd,hdf->bphf", residual, self.base.W_Q)
            key = torch.einsum("bpd,hdf->bphf", residual, self.base.W_K)
            value = torch.einsum("bpd,hdf->bphf", residual, self.base.W_V)
            scores = torch.einsum("bqhd,bkhd->bhqk", query, key) / math.sqrt(self.base.d_head)
            mask = torch.tril(torch.ones(3, 3, device=pairs.device, dtype=torch.bool))
            attention = torch.softmax(scores.masked_fill(~mask[None, None], -1e9), dim=-1)
            mixed = torch.einsum("bhqk,bkhd->bqhd", attention, value)
            attention_out = torch.einsum("bqhd,hdf->bqf", mixed, self.base.W_O)
            residual = residual + attention_out
            residual = residual + F.relu(residual @ self.base.W_in) @ self.base.W_out
            return residual[:, -1]

        output, _ = self.base.recurrent(self.base.embedding(tokens))
        return output[:, -1]

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        return self.hidden(pairs) @ self.classifier()


def dense_classifier(model: nn.Module, architecture: str) -> torch.Tensor:
    if architecture in {"mlp", "transformer"}:
        return model.W_U.detach().clone()
    return model.head.weight.detach().T.clone()


def canonical_pairs(K: int, device: torch.device) -> torch.Tensor:
    values = torch.arange(K, dtype=torch.long, device=device)
    return torch.cartesian_prod(values, values)


def split_data(K: int, seed: int, frac_train: float, device: torch.device):
    pairs = canonical_pairs(K, device)
    labels = (pairs[:, 0] + pairs[:, 1]) % K
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(K * K, generator=generator).to(device)
    split = int(frac_train * K * K)
    train_idx, test_idx = permutation[:split], permutation[split:]
    return pairs[train_idx], pairs[test_idx], labels[train_idx], labels[test_idx], pairs


def spectrum_metrics(feature_by_symbol: torch.Tensor) -> dict[str, float]:
    centered = feature_by_symbol - feature_by_symbol.mean(dim=1, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    singular_values = singular_values[singular_values > 1e-12]
    if singular_values.numel() == 0:
        return {"participation_rank": 0.0, "entropy_rank": 0.0, "top2_tail": 0.0}
    energy = singular_values.square()
    participation = energy.sum().square() / energy.square().sum().clamp_min(1e-30)
    probability = singular_values / singular_values.sum().clamp_min(1e-30)
    entropy = torch.exp(-(probability * torch.log(probability.clamp_min(1e-30))).sum())
    tail = energy[2:].sum() / energy.sum().clamp_min(1e-30) if energy.numel() > 2 else energy.new_zeros(())
    return {
        "participation_rank": float(participation.item()),
        "entropy_rank": float(entropy.item()),
        "top2_tail": float(tail.item()),
    }


@torch.no_grad()
def loss_and_accuracy(model: nn.Module, data: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    logits = model(data)
    return float(F.cross_entropy(logits, labels).item()), float((logits.argmax(-1) == labels).float().mean().item())


@torch.no_grad()
def metric_row(
    model: SoftRankHomotopyArchitecture,
    architecture: str,
    K: int,
    seed: int,
    epoch: int,
    learning_rate: float,
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    test_data: torch.Tensor,
    test_labels: torch.Tensor,
    all_pairs: torch.Tensor,
) -> dict[str, Any]:
    model.eval()
    train_loss, train_accuracy = loss_and_accuracy(model, train_data, train_labels)
    test_loss, test_accuracy = loss_and_accuracy(model, test_data, test_labels)

    hidden = model.hidden(all_pairs).reshape(K, K, -1)
    global_mean = hidden.mean(dim=(0, 1), keepdim=False)
    rx = hidden.mean(dim=1) - global_mean[None, :]
    ry = hidden.mean(dim=0) - global_mean[None, :]
    rx_spec = spectrum_metrics(rx.T.to(torch.float64))
    ry_spec = spectrum_metrics(ry.T.to(torch.float64))
    classifier = model.classifier().to(torch.float64)
    classifier_spec = spectrum_metrics(classifier)
    singular_values = torch.linalg.svdvals(classifier)
    # ``A @ B`` is formed in float32, then promoted for diagnostics.  Use a
    # float32-relative tolerance so round-off singular values are not counted
    # as genuine classifier dimensions.
    tolerance = 1e-6 * singular_values.max().clamp_min(1e-30)
    numerical_rank = int((singular_values > tolerance).sum().item())

    row: dict[str, Any] = {
        "architecture": architecture,
        "K": K,
        "seed": seed,
        "epoch": epoch,
        "learning_rate": learning_rate,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
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
        "nominal_rank": model.target_rank if model.tail_gate == 0 else model.max_rank,
    }
    numeric = [float(value) for key, value in row.items() if key != "architecture"]
    if not np.isfinite(numeric).all():
        raise FloatingPointError(f"non-finite metric at epoch {epoch}")
    if model.tail_gate == 0 and (
        numerical_rank > model.target_rank or classifier_spec["top2_tail"] > 1e-10
    ):
        raise RuntimeError(
            f"final rank invariant failed: numerical_rank={numerical_rank}, "
            f"tail={classifier_spec['top2_tail']:.3e}"
        )
    return row


def should_record(epoch: int, total_epochs: int) -> bool:
    """Early-dense sampling for a logarithmic epoch axis."""
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


def cosine_tail_gate(epoch: int, start: int, end: int) -> float:
    if epoch <= start:
        return 1.0
    if epoch >= end:
        return 0.0
    progress = (epoch - start) / (end - start)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def cosine_lr(epoch: int, total_epochs: int, maximum: float, minimum: float) -> float:
    progress = min(max(epoch / total_epochs, 0.0), 1.0)
    return minimum + 0.5 * (maximum - minimum) * (1.0 + math.cos(math.pi * progress))


def save_resume(
    path: Path,
    model: SoftRankHomotopyArchitecture,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "epoch": epoch,
            "tail_gate": float(model.tail_gate),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        },
        temporary,
    )
    temporary.replace(path)


def train_one(args: argparse.Namespace, architecture: str, K: int, seed: int) -> None:
    source = load_architecture_source(args.architecture_source)
    if not source.is_prime(K):
        raise ValueError(f"K must be prime, got {K}")
    set_seed(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    dense_run = args.dense_root.resolve() / architecture / f"K{K}" / f"seed_{seed}"
    dense_checkpoint = dense_run / "model_final.pt"
    dense_status_path = dense_run / "status.json"
    if not dense_checkpoint.exists() or not dense_status_path.exists():
        raise FileNotFoundError(f"Missing dense run: {dense_run}")
    dense_status = json.loads(dense_status_path.read_text(encoding="utf-8"))
    if dense_status.get("status") != "complete":
        raise RuntimeError(f"Dense run is not complete: {dense_run}: {dense_status}")

    out = args.output_root.resolve() / "runs" / architecture / f"K{K}" / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "status.json"
    metrics_path = out / "metrics.csv"
    resume_path = out / "resume_state.pt"
    if args.resume and status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "complete" and int(status.get("epoch", -1)) >= args.epochs:
            print(f"skip complete {architecture} K={K} seed={seed}", flush=True)
            return

    base = source.build_model(architecture, K, seed).to(device)
    dense_state = torch.load(dense_checkpoint, map_location=device, weights_only=True)
    base.load_state_dict(dense_state)
    classifier = dense_classifier(base, architecture)
    model = SoftRankHomotopyArchitecture(
        base, architecture, classifier, args.max_rank, args.target_rank
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, args.beta2),
    )
    train_data, test_data, train_labels, test_labels, all_pairs = split_data(
        K, seed, args.frac_train, device
    )

    config = {
        "architecture": architecture,
        "architecture_source": str(args.architecture_source.resolve()),
        "dense_run": str(dense_run),
        "K": K,
        "seed": seed,
        "task": "(x + y) mod K",
        "max_rank": model.max_rank,
        "target_rank": model.target_rank,
        "classifier_parameterization": "(A * channel_gate) @ B",
        "centered_balanced_svd_initialization": True,
        "gate_start": args.gate_start,
        "gate_end": args.gate_end,
        "target_loss_weight": args.target_loss_weight,
        "output_bias": False,
        "frac_train": args.frac_train,
        "epochs": args.epochs,
        "optimizer": "AdamW",
        "lr": args.lr,
        "min_lr": args.min_lr,
        "scheduler": "cosine",
        "weight_decay": args.weight_decay,
        "betas": [0.9, args.beta2],
        "full_batch": True,
        "metric_schedule": {
            "epoch_1_to_20": 1,
            "epoch_25_to_100": 5,
            "epoch_110_to_500": 10,
            "epoch_525_to_1000": 25,
            "epoch_1050_to_7000": 50,
            "after_7000": 100,
        },
        "effective_code": "role-conditioned mean penultimate feature, globally centered",
        "entropy_rank_probability": "sigma_i / sum_j sigma_j",
    }
    atomic_json(out / "config.json", config)
    atomic_json(status_path, {"status": "running", "epoch": 0, "architecture": architecture, "K": K, "seed": seed})

    rows: list[dict[str, Any]] = []
    current_epoch = 0
    if args.resume and resume_path.exists():
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        current_epoch = int(checkpoint["epoch"])
        model.tail_gate = float(
            checkpoint.get(
                "tail_gate",
                cosine_tail_gate(current_epoch, args.gate_start, args.gate_end),
            )
        )
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state_all"]])
        np.random.set_state(checkpoint["numpy_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        rows = read_metrics(metrics_path, current_epoch)
        atomic_csv(metrics_path, rows)
    if current_epoch == 0 and not rows:
        rows.append(
            metric_row(
                model, architecture, K, seed, 0, args.lr,
                train_data, train_labels, test_data, test_labels, all_pairs,
            )
        )
        atomic_csv(metrics_path, rows)
        torch.save(model.state_dict(), out / "checkpoint_initial.pt")
        save_resume(resume_path, model, optimizer, 0)

    log_handle = (out / "training.log").open("a", encoding="utf-8", buffering=1)
    started = time.time()
    try:
        for epoch in range(current_epoch + 1, args.epochs + 1):
            model.tail_gate = cosine_tail_gate(epoch, args.gate_start, args.gate_end)
            learning_rate = cosine_lr(epoch, args.epochs, args.lr, args.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            model.train()
            optimizer.zero_grad(set_to_none=True)
            hidden = model.hidden(train_data)
            loss = F.cross_entropy(hidden @ model.classifier(), train_labels)
            if args.target_loss_weight > 0:
                target_logits = hidden @ model.target_classifier()
                loss = loss + args.target_loss_weight * F.cross_entropy(
                    target_logits, train_labels
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            if should_record(epoch, args.epochs):
                row = metric_row(
                    model, architecture, K, seed, epoch, learning_rate,
                    train_data, train_labels, test_data, test_labels, all_pairs,
                )
                rows.append(row)
                atomic_csv(metrics_path, rows)
                save_resume(resume_path, model, optimizer, epoch)
                atomic_json(status_path, {"status": "running", "epoch": epoch, "architecture": architecture, "K": K, "seed": seed})
                log_handle.write(
                    f"epoch={epoch} train={row['train_accuracy']:.6f} test={row['test_accuracy']:.6f} "
                    f"part={row['effective_mean_participation_rank']:.6f} "
                    f"classifier_pr={row['classifier_participation_rank']:.6f} "
                    f"gate={row['tail_gate']:.6f} "
                    f"entropy={row['effective_mean_entropy_rank']:.6f} "
                    f"rank={row['classifier_numerical_rank']} elapsed={time.time()-started:.1f}\n"
                )
        torch.save(model.state_dict(), out / "checkpoint_final.pt")
        atomic_json(status_path, {
            "status": "complete", "epoch": args.epochs, "architecture": architecture,
            "K": K, "seed": seed, "metric_rows": len(rows), "nan_detected": False,
        })
    except Exception as exc:
        error = traceback.format_exc()
        (out / "error.log").write_text(error, encoding="utf-8")
        atomic_json(status_path, {
            "status": "failed", "epoch": int(rows[-1]["epoch"]) if rows else 0,
            "architecture": architecture, "K": K, "seed": seed,
            "nan_detected": "non-finite" in str(exc), "error": repr(exc),
        })
        raise
    finally:
        log_handle.close()


def worker_command(args: argparse.Namespace, architecture: str, K: int, seed: int) -> list[str]:
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker-architecture", architecture, "--worker-k", str(K), "--worker-seed", str(seed),
        "--architecture-source", str(args.architecture_source.resolve()),
        "--dense-root", str(args.dense_root.resolve()),
        "--output-root", str(args.output_root.resolve()),
        "--epochs", str(args.epochs), "--lr", str(args.lr), "--min-lr", str(args.min_lr),
        "--weight-decay", str(args.weight_decay), "--beta2", str(args.beta2),
        "--frac-train", str(args.frac_train), "--device", "cuda:0",
        "--max-rank", str(args.max_rank), "--target-rank", str(args.target_rank),
        "--gate-start", str(args.gate_start), "--gate-end", str(args.gate_end),
        "--target-loss-weight", str(args.target_loss_weight),
    ]
    if args.resume:
        command.append("--resume")
    return command


def orchestrate(args: argparse.Namespace) -> None:
    if not args.gpu_ids:
        raise ValueError("Orchestrator mode requires --gpu-ids")
    jobs = [(a, K, seed) for a in args.architectures for K in args.k_values for seed in args.seeds]
    slots = [gpu for gpu in args.gpu_ids for _ in range(args.max_parallel_per_gpu)]
    active: list[tuple[subprocess.Popen, Any, str, int, int, int]] = []
    failures: list[tuple[str, int, int, int]] = []
    log_dir = args.output_root.resolve() / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    while jobs or active:
        used_slots = [entry[5] for entry in active]
        available = list(slots)
        for gpu in used_slots:
            available.remove(gpu)
        while jobs and available:
            architecture, K, seed = jobs.pop(0)
            gpu = available.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            handle = (log_dir / f"{architecture}_K{K}_seed{seed}.log").open("a", encoding="utf-8")
            process = subprocess.Popen(worker_command(args, architecture, K, seed), stdout=handle, stderr=subprocess.STDOUT, env=env)
            active.append((process, handle, architecture, K, seed, gpu))
            print(f"launched {architecture} K={K} seed={seed} gpu={gpu} pid={process.pid}", flush=True)
        time.sleep(2)
        still_active = []
        for process, handle, architecture, K, seed, gpu in active:
            code = process.poll()
            if code is None:
                still_active.append((process, handle, architecture, K, seed, gpu))
            else:
                handle.close()
                print(f"finished {architecture} K={K} seed={seed} exit={code}", flush=True)
                if code != 0:
                    failures.append((architecture, K, seed, code))
        active = still_active
    if failures:
        raise RuntimeError(f"failed jobs: {failures}")


def main() -> int:
    args = parse_args()
    if args.worker_architecture is not None:
        train_one(args, args.worker_architecture, args.worker_k, args.worker_seed)
    else:
        orchestrate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
