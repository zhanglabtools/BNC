"""Train one mod-K architecture and track feature-classifier alignment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ARCHITECTURE_DETAILS = {
    "mlp": {
        "variant": "role-specific bias-free one-hidden-layer ReLU MLP",
        "embedding_dim": 256,
        "hidden_dim": 128,
    },
    "transformer": {
        "variant": "shared-codebook bias-free one-block causal Transformer",
        "d_model": 128,
        "n_heads": 4,
        "d_head": 32,
        "d_mlp": 512,
    },
    "lstm": {
        "variant": "shared-codebook bias-free one-layer LSTM with independent classifier",
        "embedding_dim": 128,
        "hidden_dim": 128,
    },
    "rnn": {
        "variant": "shared-codebook bias-free one-layer ReLU RNN with independent classifier",
        "embedding_dim": 128,
        "hidden_dim": 128,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=tuple(ARCHITECTURE_DETAILS), required=True)
    parser.add_argument("--K", "--modulus", dest="K", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frac-train", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--n-log-checkpoints", type=int, default=241)
    parser.add_argument("--embedding-lr", type=float, default=1e-3)
    parser.add_argument("--decoder-lr", type=float, default=1e-3)
    parser.add_argument("--embedding-weight-decay", type=float, default=0.0)
    parser.add_argument("--decoder-weight-decay", type=float, default=0.4)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--n-shuffle-controls", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.K < 3:
        parser.error("--K/--modulus must be at least 3")
    if not 0.0 < args.frac_train < 1.0:
        parser.error("--frac-train must lie strictly between 0 and 1")
    return args


def geometric_checkpoint_epochs(epochs: int, requested: int) -> list[int]:
    """Rounded geometric schedule, dense per decade on a logarithmic x-axis."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if requested < 2:
        raise ValueError("--n-log-checkpoints must be at least 2")
    values = np.rint(np.geomspace(1, epochs, requested)).astype(int)
    values = np.unique(np.clip(values, 1, epochs))
    return [int(value) for value in values]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_data(K: int, frac_train: float, seed: int, device: torch.device):
    elements = torch.arange(K, dtype=torch.long)
    data = torch.cartesian_prod(elements, elements)
    labels = (data[:, 0] + data[:, 1]) % K
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(data.shape[0], generator=generator)
    data, labels = data[permutation], labels[permutation]
    train_size = int(frac_train * data.shape[0])
    return (
        data[:train_size].to(device),
        labels[:train_size].to(device),
        data[train_size:].to(device),
        labels[train_size:].to(device),
    )


class AlignmentArchitecture(nn.Module):
    def penultimate_features(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def embedding_matrix(self) -> torch.Tensor:
        raise NotImplementedError

    def classifier_matrix(self) -> torch.Tensor:
        raise NotImplementedError

    def embedding_parameters(self) -> list[nn.Parameter]:
        raise NotImplementedError


class RoleSpecificMLP(AlignmentArchitecture):
    """The role-specific MLP used by the supplied multi-architecture package."""

    def __init__(self, K: int, embedding_dim: int, hidden_dim: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.W_x = nn.Parameter(torch.randn(K, embedding_dim) / math.sqrt(embedding_dim))
        self.W_y = nn.Parameter(torch.randn(K, embedding_dim) / math.sqrt(embedding_dim))
        self.W = nn.Parameter(
            torch.randn(2 * embedding_dim, hidden_dim) / math.sqrt(2 * embedding_dim)
        )
        self.W_U = nn.Parameter(torch.randn(hidden_dim, K) / math.sqrt(hidden_dim))

    def penultimate_features(self, data: torch.Tensor) -> torch.Tensor:
        first = self.W_x[data[:, 0]]
        second = self.W_y[data[:, 1]]
        return F.relu(torch.cat((first, second), dim=1) @ self.W)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.penultimate_features(data) @ self.W_U

    def embedding_matrix(self) -> torch.Tensor:
        return self.W_x

    def classifier_matrix(self) -> torch.Tensor:
        return self.W_U.T

    def embedding_parameters(self) -> list[nn.Parameter]:
        return [self.W_x, self.W_y]


class LegacyTransformer(AlignmentArchitecture):
    """Bias-free, no-LayerNorm one-block Transformer from the supplied package."""

    def __init__(self, K: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.K = K
        self.n_heads = 4
        self.d_head = 32
        d_model = 128
        d_mlp = 512
        self.W_E = nn.Parameter(torch.randn(K + 1, d_model) / math.sqrt(d_model))
        self.W_pos = nn.Parameter(torch.randn(3, d_model) / math.sqrt(d_model))
        self.W_Q = nn.Parameter(
            torch.randn(self.n_heads, d_model, self.d_head) / math.sqrt(d_model)
        )
        self.W_K = nn.Parameter(
            torch.randn(self.n_heads, d_model, self.d_head) / math.sqrt(d_model)
        )
        self.W_V = nn.Parameter(
            torch.randn(self.n_heads, d_model, self.d_head) / math.sqrt(d_model)
        )
        self.W_O = nn.Parameter(
            torch.randn(self.n_heads, self.d_head, d_model)
            / math.sqrt(self.n_heads * self.d_head)
        )
        self.W_in = nn.Parameter(torch.randn(d_model, d_mlp) / math.sqrt(d_model))
        self.W_out = nn.Parameter(torch.randn(d_mlp, d_model) / math.sqrt(d_mlp))
        self.W_U = nn.Parameter(torch.randn(d_model, K) / math.sqrt(d_model))

    def penultimate_features(self, data: torch.Tensor) -> torch.Tensor:
        equals = torch.full((data.shape[0],), self.K, dtype=torch.long, device=data.device)
        tokens = torch.stack((data[:, 0], data[:, 1], equals), dim=1)
        residual = self.W_E[tokens] + self.W_pos[None, :, :]
        query = torch.einsum("bpd,hdf->bphf", residual, self.W_Q)
        key = torch.einsum("bpd,hdf->bphf", residual, self.W_K)
        value = torch.einsum("bpd,hdf->bphf", residual, self.W_V)
        scores = torch.einsum("bqhd,bkhd->bhqk", query, key) / math.sqrt(self.d_head)
        mask = torch.tril(torch.ones(3, 3, device=data.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask[None, None], -1e9)
        attention = torch.softmax(scores, dim=-1)
        mixed = torch.einsum("bhqk,bkhd->bqhd", attention, value)
        residual = residual + torch.einsum("bqhd,hdf->bqf", mixed, self.W_O)
        residual = residual + F.relu(residual @ self.W_in) @ self.W_out
        return residual[:, -1]

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.penultimate_features(data) @ self.W_U

    def embedding_matrix(self) -> torch.Tensor:
        return self.W_E[: self.K]

    def classifier_matrix(self) -> torch.Tensor:
        return self.W_U.T

    def embedding_parameters(self) -> list[nn.Parameter]:
        return [self.W_E]


class SharedBiasFreeRecurrent(AlignmentArchitecture):
    """Shared codebook LSTM/RNN with an independent bias-free classifier."""

    def __init__(self, architecture: str, K: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
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

    def penultimate_features(self, data: torch.Tensor) -> torch.Tensor:
        equals = torch.full((data.shape[0],), self.K, dtype=torch.long, device=data.device)
        tokens = torch.stack((data[:, 0], data[:, 1], equals), dim=1)
        output, _ = self.recurrent(self.embedding(tokens))
        return output[:, -1]

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.head(self.penultimate_features(data))

    def embedding_matrix(self) -> torch.Tensor:
        return self.embedding.weight[: self.K]

    def classifier_matrix(self) -> torch.Tensor:
        return self.head.weight

    def embedding_parameters(self) -> list[nn.Parameter]:
        return [self.embedding.weight]


def build_model(architecture: str, K: int, seed: int) -> AlignmentArchitecture:
    if architecture == "mlp":
        return RoleSpecificMLP(K, embedding_dim=256, hidden_dim=128, seed=seed)
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


@torch.no_grad()
def accuracy(model: nn.Module, data: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    return float((model(data).argmax(dim=-1) == labels).float().mean().item())


def means_by_class(
    features: torch.Tensor, labels: torch.Tensor, K: int
) -> torch.Tensor:
    sums = torch.zeros(
        (K, features.shape[1]), dtype=features.dtype, device=features.device
    )
    sums.index_add_(0, labels, features)
    counts = torch.bincount(labels, minlength=K)
    if not bool(torch.all(counts > 0).item()):
        missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"training split is missing classes: {missing}")
    return sums / counts.to(features.dtype).unsqueeze(1)


def classwise_cosine(class_means: torch.Tensor, classifier: torch.Tensor) -> float:
    means_unit = F.normalize(class_means, dim=1)
    classifier_unit = F.normalize(classifier, dim=1)
    return float(torch.sum(means_unit * classifier_unit, dim=1).mean().item())


@torch.no_grad()
def alignment_metrics(
    model: AlignmentArchitecture,
    all_data: torch.Tensor,
    all_labels: torch.Tensor,
    K: int,
    permutations: torch.Tensor,
) -> tuple[float, float]:
    features = model.penultimate_features(all_data).detach().to(torch.float64)
    class_means = means_by_class(features, all_labels, K)
    classifier = model.classifier_matrix().detach().to(torch.float64)
    matched = classwise_cosine(class_means, classifier)
    shuffled = np.mean(
        [
            classwise_cosine(class_means, classifier[permutation])
            for permutation in permutations
        ]
    )
    return matched, float(shuffled)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_data, train_labels, test_data, test_labels = make_data(
        args.K, args.frac_train, args.seed, device
    )
    elements = torch.arange(args.K, dtype=torch.long, device=device)
    all_data = torch.cartesian_prod(elements, elements)
    all_labels = (all_data[:, 0] + all_data[:, 1]) % args.K
    model = build_model(args.architecture, args.K, args.seed).to(device)
    optimizer = make_optimizer(model, args)
    shuffle_generator = torch.Generator().manual_seed(10000 + args.seed)
    permutations = torch.stack(
        [
            torch.randperm(args.K, generator=shuffle_generator)
            for _ in range(args.n_shuffle_controls)
        ]
    ).to(device)
    checkpoint_epochs = geometric_checkpoint_epochs(args.epochs, args.n_log_checkpoints)
    checkpoint_epoch_set = set(checkpoint_epochs)

    sequence = "role-specific [a,b]" if args.architecture == "mlp" else "[a,b,=]"
    config = {
        "architecture": args.architecture,
        **ARCHITECTURE_DETAILS[args.architecture],
        "K": args.K,
        "seed": args.seed,
        "frac_train": args.frac_train,
        "epochs": args.epochs,
        "checkpoint_schedule": "rounded geometric spacing from epoch 1 to final epoch",
        "n_log_checkpoints_requested": args.n_log_checkpoints,
        "n_log_checkpoints_actual": len(checkpoint_epochs),
        "checkpoint_epochs": checkpoint_epochs,
        "embedding_lr": args.embedding_lr,
        "decoder_lr": args.decoder_lr,
        "embedding_weight_decay": args.embedding_weight_decay,
        "decoder_weight_decay": args.decoder_weight_decay,
        "beta2": args.beta2,
        "n_shuffle_controls": args.n_shuffle_controls,
        "device_requested": args.device,
        "device_used": str(device),
        "task": "(a,b) -> (a+b) mod K",
        "sequence": sequence,
        "equals_token_id": None if args.architecture == "mlp" else args.K,
        "n_input_tokens": args.K if args.architecture == "mlp" else args.K + 1,
        "n_output_classes": args.K,
        "dataset_size": args.K * args.K,
        "train_size": int(train_data.shape[0]),
        "test_size": int(test_data.shape[0]),
        "optimizer": "AdamW",
        "optimizer_betas": [0.9, args.beta2],
        "full_batch": True,
        "shuffle_permutation_seed": 10000 + args.seed,
        "feature_split": "full balanced K-by-K input grid",
        "feature_definition": "penultimate class means grouped by target label",
        "metric": "mean raw classwise cosine between penultimate class means and classifier vectors",
        "rnn_classifier_tied": False,
        "checkpoint_saved": False,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    rows: list[dict] = []
    start = time.time()

    def record(epoch: int) -> None:
        train_acc = accuracy(model, train_data, train_labels)
        test_acc = accuracy(model, test_data, test_labels)
        matched, shuffled = alignment_metrics(
            model, all_data, all_labels, args.K, permutations
        )
        row = {
            "epoch": epoch,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "feature_classifier_alignment": matched,
            "shuffled_feature_classifier_alignment": shuffled,
        }
        rows.append(row)
        print(
            f"arch={args.architecture} K={args.K} seed={args.seed} epoch={epoch} "
            f"train={train_acc:.4f} test={test_acc:.4f} matched={matched:.4f} "
            f"shuffled={shuffled:.4f} elapsed={time.time() - start:.1f}s",
            flush=True,
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(train_data), train_labels)
        loss.backward()
        optimizer.step()
        if epoch in checkpoint_epoch_set:
            record(epoch)

    write_csv(output_dir / "metrics.csv", rows)
    final = rows[-1]
    summary = {
        "architecture": args.architecture,
        "variant": ARCHITECTURE_DETAILS[args.architecture]["variant"],
        "K": args.K,
        "seed": args.seed,
        "final_epoch": int(final["epoch"]),
        "final_matched_alignment": float(final["feature_classifier_alignment"]),
        "final_shuffled_alignment": float(
            final["shuffled_feature_classifier_alignment"]
        ),
        "final_gap": float(
            final["feature_classifier_alignment"]
            - final["shuffled_feature_classifier_alignment"]
        ),
        "final_train_acc": float(final["train_acc"]),
        "final_test_acc": float(final["test_acc"]),
        "n_logged_epochs": len(rows),
        "elapsed_seconds": time.time() - start,
        "checkpoint_saved": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
