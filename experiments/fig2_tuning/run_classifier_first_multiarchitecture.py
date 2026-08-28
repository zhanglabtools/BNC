"""Run classifier-first Best-cyclic-score dynamics across four architectures.

The exact Best cyclic score and classifier-first optimizer grouping are kept
equivalent to the traced experiment.  The forward models are the role-specific
MLP, legacy causal Transformer, shared-codebook LSTM, and shared-codebook ReLU
RNN from the supplied multi-architecture code.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


METRIC_FIELDS = [
    "architecture",
    "K",
    "seed",
    "epoch",
    "train_loss",
    "train_accuracy",
    "test_accuracy",
    "classifier_best_cyclic_score",
    "embedding_best_cyclic_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=["mlp", "transformer", "lstm", "rnn"],
        default=["mlp", "transformer", "lstm", "rnn"],
    )
    parser.add_argument("--k-values", type=int, nargs="+", default=[79, 97, 113])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="fixed train/test split seed; default uses each run's initialization seed",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[])
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frac-train", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--embedding-lr", type=float, default=2e-4)
    parser.add_argument("--decoder-lr", type=float, default=2e-3)
    parser.add_argument("--embedding-weight-decay", type=float, default=0.0)
    parser.add_argument("--decoder-weight-decay", type=float, default=0.8)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--mlp-embedding-dim", type=int, default=256)
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--worker-k", type=int)
    parser.add_argument("--worker-seed", type=int)
    parser.add_argument("--worker-architecture", choices=["mlp", "transformer", "lstm", "rnn"])
    args = parser.parse_args()
    worker_values = (args.worker_architecture, args.worker_k, args.worker_seed)
    if any(value is not None for value in worker_values) and not all(
        value is not None for value in worker_values
    ):
        parser.error("--worker-architecture, --worker-k and --worker-seed must be supplied together")
    if not 0.0 < args.frac_train < 1.0:
        parser.error("--frac-train must be in (0, 1)")
    if args.log_every <= 0 or args.epochs <= 0:
        parser.error("epochs and log-every must be positive")
    if args.max_parallel <= 0:
        parser.error("--max-parallel must be positive")
    return args


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    for divisor in range(2, int(math.sqrt(value)) + 1):
        if value % divisor == 0:
            return False
    return True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_data(K: int, frac_train: float, seed: int, device: torch.device):
    elements = torch.arange(K, dtype=torch.long)
    pairs = torch.cartesian_prod(elements, elements)
    labels = (pairs[:, 0] + pairs[:, 1]) % K
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(pairs.shape[0], generator=generator)
    split = int(frac_train * pairs.shape[0])
    train_indices = permutation[:split]
    test_indices = permutation[split:]
    return (
        pairs[train_indices].to(device),
        pairs[test_indices].to(device),
        labels[train_indices].to(device),
        labels[test_indices].to(device),
        train_indices,
        test_indices,
    )


class AlignmentArchitecture(nn.Module):
    architecture: str

    def embedding_matrix(self) -> torch.Tensor:
        raise NotImplementedError

    def classifier_matrix(self) -> torch.Tensor:
        raise NotImplementedError

    def embedding_parameters(self) -> list[nn.Parameter]:
        raise NotImplementedError

    def architecture_config(self) -> dict:
        raise NotImplementedError


class RoleSpecificMLP(AlignmentArchitecture):
    """Role-specific, bias-free one-hidden-layer ReLU MLP from the supplied code."""

    architecture = "mlp"

    def __init__(self, K: int, seed: int, embedding_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.W_x = nn.Parameter(torch.randn(K, embedding_dim) / math.sqrt(embedding_dim))
        self.W_y = nn.Parameter(torch.randn(K, embedding_dim) / math.sqrt(embedding_dim))
        self.W = nn.Parameter(
            torch.randn(2 * embedding_dim, hidden_dim) / math.sqrt(2 * embedding_dim)
        )
        self.W_U = nn.Parameter(torch.randn(hidden_dim, K) / math.sqrt(hidden_dim))

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        first = self.W_x[pairs[:, 0]]
        second = self.W_y[pairs[:, 1]]
        hidden = F.relu(torch.cat((first, second), dim=1) @ self.W)
        return hidden @ self.W_U

    def embedding_matrix(self) -> torch.Tensor:
        return self.W_x

    def classifier_matrix(self) -> torch.Tensor:
        return self.W_U.T

    def embedding_parameters(self) -> list[nn.Parameter]:
        return [self.W_x, self.W_y]

    def architecture_config(self) -> dict:
        return {
            "model": "role-specific bias-free one-hidden-layer ReLU MLP",
            "sequence": "role-specific [a,b]",
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "reported_embedding": "W_x (first-input token embedding)",
        }


class LegacyTransformer(AlignmentArchitecture):
    """Bias-free, no-LayerNorm one-block causal Transformer from the supplied code."""

    architecture = "transformer"

    def __init__(self, K: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        d_model = 128
        n_heads = 4
        d_head = 32
        d_mlp = 512
        self.K = K
        self.d_head = d_head
        self.W_E = nn.Parameter(torch.randn(K + 1, d_model) / math.sqrt(d_model))
        self.W_pos = nn.Parameter(torch.randn(3, d_model) / math.sqrt(d_model))
        self.W_Q = nn.Parameter(torch.randn(n_heads, d_model, d_head) / math.sqrt(d_model))
        self.W_K = nn.Parameter(torch.randn(n_heads, d_model, d_head) / math.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(n_heads, d_model, d_head) / math.sqrt(d_model))
        self.W_O = nn.Parameter(
            torch.randn(n_heads, d_head, d_model) / math.sqrt(n_heads * d_head)
        )
        self.W_in = nn.Parameter(torch.randn(d_model, d_mlp) / math.sqrt(d_model))
        self.W_out = nn.Parameter(torch.randn(d_mlp, d_model) / math.sqrt(d_mlp))
        self.W_U = nn.Parameter(torch.randn(d_model, K) / math.sqrt(d_model))

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        equals = torch.full((pairs.shape[0],), self.K, dtype=torch.long, device=pairs.device)
        tokens = torch.stack((pairs[:, 0], pairs[:, 1], equals), dim=1)
        residual = self.W_E[tokens] + self.W_pos[None, :, :]
        query = torch.einsum("bpd,hdf->bphf", residual, self.W_Q)
        key = torch.einsum("bpd,hdf->bphf", residual, self.W_K)
        value = torch.einsum("bpd,hdf->bphf", residual, self.W_V)
        scores = torch.einsum("bqhd,bkhd->bhqk", query, key) / math.sqrt(self.d_head)
        mask = torch.tril(torch.ones(3, 3, device=pairs.device, dtype=torch.bool))
        attention = torch.softmax(scores.masked_fill(~mask[None, None], -1e9), dim=-1)
        mixed = torch.einsum("bhqk,bkhd->bqhd", attention, value)
        attention_out = torch.einsum("bqhd,hdf->bqf", mixed, self.W_O)
        residual = residual + attention_out
        residual = residual + F.relu(residual @ self.W_in) @ self.W_out
        return residual[:, -1] @ self.W_U


    def embedding_matrix(self) -> torch.Tensor:
        return self.W_E[: self.K]

    def classifier_matrix(self) -> torch.Tensor:
        return self.W_U.T

    def embedding_parameters(self) -> list[nn.Parameter]:
        return [self.W_E]

    def architecture_config(self) -> dict:
        return {
            "model": "shared-codebook bias-free one-block causal Transformer without LayerNorm",
            "sequence": "[a,b,=]",
            "d_model": 128,
            "n_heads": 4,
            "d_head": 32,
            "d_mlp": 512,
        }


class SharedBiasFreeRecurrent(AlignmentArchitecture):
    """Shared-codebook LSTM/RNN with an independent bias-free classifier."""

    def __init__(self, architecture: str, K: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.architecture = architecture
        self.K = K
        self.embedding = nn.Embedding(K + 1, 128)
        nn.init.normal_(self.embedding.weight, std=1.0 / math.sqrt(128))
        if architecture == "lstm":
            self.recurrent = nn.LSTM(128, 128, num_layers=1, bias=False, batch_first=True)
        elif architecture == "rnn":
            self.recurrent = nn.RNN(
                128,
                128,
                num_layers=1,
                nonlinearity="relu",
                bias=False,
                batch_first=True,
            )
        else:
            raise ValueError(architecture)
        self.head = nn.Linear(128, K, bias=False)

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        equals = torch.full((pairs.shape[0],), self.K, dtype=torch.long, device=pairs.device)
        tokens = torch.stack((pairs[:, 0], pairs[:, 1], equals), dim=1)
        output, _ = self.recurrent(self.embedding(tokens))
        return self.head(output[:, -1])

    def embedding_matrix(self) -> torch.Tensor:
        return self.embedding.weight[: self.K]

    def classifier_matrix(self) -> torch.Tensor:
        return self.head.weight

    def embedding_parameters(self) -> list[nn.Parameter]:
        return [self.embedding.weight]

    def architecture_config(self) -> dict:
        core = "LSTM" if self.architecture == "lstm" else "ReLU RNN"
        return {
            "model": f"shared-codebook bias-free one-layer {core} with independent classifier",
            "sequence": "[a,b,=]",
            "embedding_dim": 128,
            "hidden_dim": 128,
            "n_layers": 1,
            "classifier_tied": False,
        }


def build_model(
    architecture: str,
    K: int,
    seed: int,
    mlp_embedding_dim: int = 256,
    mlp_hidden_dim: int = 128,
) -> AlignmentArchitecture:
    if architecture == "mlp":
        return RoleSpecificMLP(K, seed, mlp_embedding_dim, mlp_hidden_dim)
    if architecture == "transformer":
        return LegacyTransformer(K, seed)
    return SharedBiasFreeRecurrent(architecture, K, seed)


def make_optimizer(model: AlignmentArchitecture, args: argparse.Namespace):
    embedding_parameters = model.embedding_parameters()
    embedding_ids = {id(parameter) for parameter in embedding_parameters}
    decoder_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in embedding_ids
    ]
    return torch.optim.AdamW(
        [
            {
                "params": embedding_parameters,
                "lr": args.embedding_lr,
                "weight_decay": args.embedding_weight_decay,
            },
            {
                "params": decoder_parameters,
                "lr": args.decoder_lr,
                "weight_decay": args.decoder_weight_decay,
            },
        ],
        betas=(0.9, args.beta2),
    )


def centered(matrix: torch.Tensor) -> torch.Tensor:
    return matrix - matrix.mean(dim=0, keepdim=True)


def pca2_coordinates(matrix: torch.Tensor) -> torch.Tensor:
    matrix = centered(matrix)
    u, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
    coordinates = u[:, :2] * singular_values[:2]
    radius = torch.sqrt(coordinates.square().sum(dim=1).mean()).clamp_min(1e-20)
    return coordinates / radius


def best_cyclic_score(matrix: torch.Tensor, K: int) -> float:
    """Exact automorphism/orientation search traced from the mod-97 code."""
    coordinates = pca2_coordinates(matrix).detach().cpu().numpy()
    complex_coordinates = coordinates[:, 0] + 1j * coordinates[:, 1]
    complex_coordinates /= np.maximum(np.abs(complex_coordinates), 1e-12)
    labels = np.arange(K)
    best = 0.0
    for multiplier in range(1, K):
        if math.gcd(multiplier, K) != 1:
            continue
        phase = 2 * np.pi * ((multiplier * labels) % K) / K
        target = np.exp(1j * phase)
        best = max(
            best,
            float(abs(np.mean(complex_coordinates * np.conjugate(target)))),
            float(abs(np.mean(complex_coordinates * target))),
        )
    return best


@torch.no_grad()
def evaluate(model: nn.Module, data: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    model.eval()
    logits = model(data)
    loss = float(F.cross_entropy(logits, labels).item())
    accuracy = float((logits.argmax(dim=-1) == labels).float().mean().item())
    return loss, accuracy


@torch.no_grad()
def metric_row(
    model: AlignmentArchitecture,
    architecture: str,
    K: int,
    seed: int,
    epoch: int,
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    test_data: torch.Tensor,
    test_labels: torch.Tensor,
) -> dict:
    train_loss, train_accuracy = evaluate(model, train_data, train_labels)
    _, test_accuracy = evaluate(model, test_data, test_labels)
    embedding = model.embedding_matrix().detach().to(torch.float64)
    classifier = model.classifier_matrix().detach().to(torch.float64)
    row = {
        "architecture": architecture,
        "K": K,
        "seed": seed,
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "classifier_best_cyclic_score": best_cyclic_score(classifier, K),
        "embedding_best_cyclic_score": best_cyclic_score(embedding, K),
    }
    if classifier.shape[0] != K or embedding.shape[0] != K:
        raise RuntimeError(f"metric extraction shape failure: {classifier.shape=} {embedding.shape=}")
    if not all(np.isfinite(float(row[field])) for field in METRIC_FIELDS[4:]):
        raise RuntimeError(f"non-finite metric at K={K}, seed={seed}, epoch={epoch}")
    if not (0.0 <= row["classifier_best_cyclic_score"] <= 1.000001):
        raise RuntimeError("classifier cyclic score outside [0,1]")
    if not (0.0 <= row["embedding_best_cyclic_score"] <= 1.000001):
        raise RuntimeError("embedding cyclic score outside [0,1]")
    return row


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_resume(path: Path, model: nn.Module, optimizer, epoch: int) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "epoch": epoch,
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


def read_rows(path: Path, max_epoch: int | None = None) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            row = {
                field: raw[field] if field == "architecture" else float(raw[field])
                for field in METRIC_FIELDS
            }
            row["K"] = int(row["K"])
            row["seed"] = int(row["seed"])
            row["epoch"] = int(row["epoch"])
            if max_epoch is None or row["epoch"] <= max_epoch:
                rows.append(row)
    return rows


def configure_logger(path: Path, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def train_one(args: argparse.Namespace, architecture: str, K: int, seed: int) -> None:
    if not is_prime(K):
        raise ValueError(f"K must be prime for the requested sweep, got {K}")
    output_root = args.output_dir.resolve()
    run_dir = output_root / architecture / f"K{K}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(
        run_dir / "training.log", f"{architecture}_K{K}_seed{seed}"
    )
    status_path = run_dir / "status.json"
    metrics_path = run_dir / "metrics.csv"
    resume_path = run_dir / "resume_state.pt"
    final_path = run_dir / "final_metrics.json"

    try:
        if status_path.exists() and args.resume:
            existing = json.loads(status_path.read_text(encoding="utf-8"))
            if (
                existing.get("status") == "complete"
                and int(existing.get("epoch", -1)) >= args.epochs
                and final_path.exists()
            ):
                logger.info("already complete; skipping")
                return
        set_seed(seed)
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device(args.device)
        split_seed = seed if args.split_seed is None else args.split_seed
        train_data, test_data, train_labels, test_labels, train_indices, test_indices = make_data(
            K, args.frac_train, split_seed, device
        )
        if set(train_indices.tolist()).intersection(test_indices.tolist()):
            raise RuntimeError("train/test split overlap")
        model = build_model(
            architecture,
            K,
            seed,
            mlp_embedding_dim=args.mlp_embedding_dim,
            mlp_hidden_dim=args.mlp_hidden_dim,
        ).to(device)
        optimizer = make_optimizer(model, args)
        embedding_shape = list(model.embedding_matrix().shape)
        classifier_shape = list(model.classifier_matrix().shape)
        config = {
            "architecture": architecture,
            "K": K,
            "seed": seed,
            "task": "(a,b) -> (a+b) mod K",
            **model.architecture_config(),
            "equals_token_id": None if architecture == "mlp" else K,
            "numeric_token_ids": [0, K - 1],
            "dataset_size": K * K,
            "train_size": int(train_data.shape[0]),
            "test_size": int(test_data.shape[0]),
            "frac_train": args.frac_train,
            "dropout": 0.0,
            "optimizer": "AdamW",
            "betas": [0.9, args.beta2],
            "embedding_lr": args.embedding_lr,
            "decoder_lr": args.decoder_lr,
            "embedding_weight_decay": args.embedding_weight_decay,
            "decoder_weight_decay": args.decoder_weight_decay,
            "scheduler": None,
            "loss": "cross_entropy",
            "full_batch": True,
            "max_epoch": args.epochs,
            "checkpoint_every": args.log_every,
            "onset_threshold": 0.9,
            "onset_consecutive_checkpoints": 2,
            "reported_embedding_vectors_shape": embedding_shape,
            "classifier_vectors_shape": classifier_shape,
            "python_seed": seed,
            "numpy_seed": seed,
            "torch_seed": seed,
            "cuda_seed": seed,
            "data_split_seed": split_seed,
            "model_initialization_seed": seed,
            "dataloader_seed": None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "device": str(device),
        }
        atomic_json(run_dir / "config.json", config)
        atomic_json(
            status_path,
            {
                "status": "running",
                "architecture": architecture,
                "K": K,
                "seed": seed,
                "epoch": 0,
            },
        )

        current_epoch = 0
        rows: list[dict] = []
        if args.resume and resume_path.exists():
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            current_epoch = int(checkpoint["epoch"])
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
            if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all"):
                torch.cuda.set_rng_state_all(
                    [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
                )
            np.random.set_state(checkpoint["numpy_rng_state"])
            random.setstate(checkpoint["python_rng_state"])
            rows = read_rows(metrics_path, current_epoch)
            atomic_csv(metrics_path, rows)
            logger.info("resumed from epoch=%d with %d metric rows", current_epoch, len(rows))
        if current_epoch == 0 and not rows:
            rows.append(
                metric_row(
                    model,
                    architecture,
                    K,
                    seed,
                    0,
                    train_data,
                    train_labels,
                    test_data,
                    test_labels,
                )
            )
            atomic_csv(metrics_path, rows)
            save_resume(resume_path, model, optimizer, 0)
            logger.info("epoch=0 classifier=%.6f embedding=%.6f", rows[-1]["classifier_best_cyclic_score"], rows[-1]["embedding_best_cyclic_score"])

        start = time.time()
        model.train()
        for epoch in range(current_epoch + 1, args.epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(train_data), train_labels)
            loss.backward()
            optimizer.step()
            if epoch % args.log_every == 0 or epoch == args.epochs:
                row = metric_row(
                    model,
                    architecture,
                    K,
                    seed,
                    epoch,
                    train_data,
                    train_labels,
                    test_data,
                    test_labels,
                )
                rows.append(row)
                atomic_csv(metrics_path, rows)
                save_resume(resume_path, model, optimizer, epoch)
                atomic_json(
                    status_path,
                    {
                        "status": "running",
                        "architecture": architecture,
                        "K": K,
                        "seed": seed,
                        "epoch": epoch,
                    },
                )
                logger.info(
                    "epoch=%d loss=%.6g train=%.4f test=%.4f classifier=%.6f embedding=%.6f elapsed=%.1fs",
                    epoch,
                    row["train_loss"],
                    row["train_accuracy"],
                    row["test_accuracy"],
                    row["classifier_best_cyclic_score"],
                    row["embedding_best_cyclic_score"],
                    time.time() - start,
                )
                model.train()

        torch.save(model.state_dict(), run_dir / "model_final.pt")
        final = dict(rows[-1])
        final["n_logged_checkpoints"] = len(rows)
        final["elapsed_seconds_this_invocation"] = time.time() - start
        atomic_json(final_path, final)
        atomic_json(
            status_path,
            {
                "status": "complete",
                "architecture": architecture,
                "K": K,
                "seed": seed,
                "epoch": args.epochs,
            },
        )
        logger.info("complete")
    except Exception as exc:
        error_text = traceback.format_exc()
        (run_dir / "error.log").write_text(error_text, encoding="utf-8")
        atomic_json(
            status_path,
            {
                "status": "failed",
                "architecture": architecture,
                "K": K,
                "seed": seed,
                "error": repr(exc),
            },
        )
        logger.exception("run failed")
        raise


def worker_command(
    args: argparse.Namespace, architecture: str, K: int, seed: int
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-architecture",
        architecture,
        "--worker-k",
        str(K),
        "--worker-seed",
        str(seed),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--device",
        "cuda:0" if args.gpu_ids else args.device,
        "--frac-train",
        str(args.frac_train),
        "--split-seed",
        str(seed if args.split_seed is None else args.split_seed),
        "--epochs",
        str(args.epochs),
        "--log-every",
        str(args.log_every),
        "--embedding-lr",
        str(args.embedding_lr),
        "--decoder-lr",
        str(args.decoder_lr),
        "--embedding-weight-decay",
        str(args.embedding_weight_decay),
        "--decoder-weight-decay",
        str(args.decoder_weight_decay),
        "--beta2",
        str(args.beta2),
        "--mlp-embedding-dim",
        str(args.mlp_embedding_dim),
        "--mlp-hidden-dim",
        str(args.mlp_hidden_dim),
    ]
    if args.resume:
        command.append("--resume")
    return command


def run_sweep(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    queue = [
        (architecture, K, seed)
        for architecture in args.architectures
        for K in args.k_values
        for seed in args.seeds
    ]
    active: list[
        tuple[subprocess.Popen, object, str, int, int, int | None]
    ] = []
    failures: list[tuple[str, int, int, int]] = []
    launched = 0
    while queue or active:
        while queue and len(active) < args.max_parallel:
            architecture, K, seed = queue.pop(0)
            gpu_id = args.gpu_ids[launched % len(args.gpu_ids)] if args.gpu_ids else None
            launched += 1
            environment = os.environ.copy()
            if gpu_id is not None:
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            log_handle = (log_dir / f"{architecture}_K{K}_seed{seed}.log").open(
                "a", encoding="utf-8"
            )
            process = subprocess.Popen(
                worker_command(args, architecture, K, seed),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            active.append((process, log_handle, architecture, K, seed, gpu_id))
            print(
                f"launched architecture={architecture} K={K} seed={seed} gpu={gpu_id}",
                flush=True,
            )
        still_active = []
        for process, log_handle, architecture, K, seed, gpu_id in active:
            code = process.poll()
            if code is None:
                still_active.append(
                    (process, log_handle, architecture, K, seed, gpu_id)
                )
            else:
                log_handle.close()
                print(
                    f"finished architecture={architecture} K={K} seed={seed} exit={code}",
                    flush=True,
                )
                if code != 0:
                    failures.append((architecture, K, seed, code))
        active = still_active
        if active:
            time.sleep(1)
    if failures:
        raise SystemExit(f"failed runs: {failures}")


def main() -> None:
    args = parse_args()
    if args.worker_k is not None:
        train_one(
            args, args.worker_architecture, args.worker_k, args.worker_seed
        )
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
