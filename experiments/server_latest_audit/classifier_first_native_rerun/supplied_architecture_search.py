"""Automatic architecture search for the fixed-seed mod-97 experiment.

The original data split and evaluation target are kept fixed. Hyperparameters
are selected on a validation subset carved out of the original training set;
the untouched 30% test split is used only for the final retraining runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class TrialConfig:
    architecture: str
    model_dim: int
    hidden_dim: int
    layers: int
    lr: float
    weight_decay: float
    dropout: float = 0.0
    n_heads: int = 4
    ff_mult: int = 2
    nonlinearity: str = "tanh"
    input_style: str = "query"
    variant: str = "standard"

    @property
    def name(self) -> str:
        lr = str(self.lr).replace(".", "p")
        wd = str(self.weight_decay).replace(".", "p")
        suffix = ""
        if self.input_style != "query":
            suffix += f"_{self.input_style}"
        if self.variant != "standard":
            suffix += f"_{self.variant}"
        if self.architecture == "rnn" and self.nonlinearity != "tanh":
            suffix += f"_{self.nonlinearity}"
        return (
            f"{self.architecture}_d{self.model_dim}_h{self.hidden_dim}_"
            f"L{self.layers}_lr{lr}_wd{wd}_do{str(self.dropout).replace('.', 'p')}_"
            f"heads{self.n_heads}_ff{self.ff_mult}{suffix}"
        )


class RoleSequenceInputs(nn.Module):
    """Role-specific number embeddings plus a learned result-query token."""

    def __init__(
        self,
        mod: int,
        model_dim: int,
        dropout: float,
        seed: int,
        input_style: str,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.mod = mod
        self.model_dim = model_dim
        self.input_style = input_style
        self.E_x = nn.Embedding(mod, model_dim)
        self.E_y = nn.Embedding(mod, model_dim)
        self.query = nn.Parameter(torch.randn(model_dim) / math.sqrt(model_dim))
        self.position = nn.Parameter(
            torch.randn(3, model_dim) / math.sqrt(model_dim)
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.E_x.weight, std=1.0 / math.sqrt(model_dim))
        nn.init.normal_(self.E_y.weight, std=1.0 / math.sqrt(model_dim))

    def sequence(self, data: torch.Tensor) -> torch.Tensor:
        batch = data.shape[0]
        x = self.E_x(data[:, 0])
        y = self.E_y(data[:, 1])
        q = self.query.unsqueeze(0).expand(batch, -1)
        if self.input_style == "operands":
            seq = torch.stack((x, y), dim=1)
            return self.dropout(seq + self.position[:2].unsqueeze(0))
        seq = torch.stack((x, y, q), dim=1)
        return self.dropout(seq + self.position.unsqueeze(0))

    def embedding_matrix(self) -> torch.Tensor:
        return self.E_x.weight


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The result-query token is last, so a causal mask still exposes both operands.
        mask = torch.triu(
            torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        normed = self.ln1(x)
        attn, _ = self.attn(normed, normed, normed, attn_mask=mask, need_weights=False)
        x = x + attn
        return x + self.ff(self.ln2(x))


class ModularTransformer(RoleSequenceInputs):
    def __init__(self, mod: int, cfg: TrialConfig, seed: int):
        super().__init__(mod, cfg.model_dim, cfg.dropout, seed, cfg.input_style)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    cfg.model_dim, cfg.n_heads, cfg.ff_mult, cfg.dropout
                )
                for _ in range(cfg.layers)
            ]
        )
        self.final_norm = nn.LayerNorm(cfg.model_dim)
        self.head = nn.Linear(cfg.model_dim, mod, bias=False)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        x = self.sequence(data)
        for block in self.blocks:
            x = block(x)
        return self.head(self.final_norm(x[:, -1, :]))

    def classifier_matrix(self) -> torch.Tensor:
        return self.head.weight


class ModularRecurrent(RoleSequenceInputs):
    def __init__(self, mod: int, cfg: TrialConfig, seed: int):
        super().__init__(mod, cfg.model_dim, cfg.dropout, seed, cfg.input_style)
        recurrent_dropout = cfg.dropout if cfg.layers > 1 else 0.0
        if cfg.architecture == "lstm":
            self.recurrent = nn.LSTM(
                cfg.model_dim,
                cfg.hidden_dim,
                num_layers=cfg.layers,
                dropout=recurrent_dropout,
                batch_first=True,
            )
        elif cfg.architecture == "rnn":
            self.recurrent = nn.RNN(
                cfg.model_dim,
                cfg.hidden_dim,
                num_layers=cfg.layers,
                nonlinearity=cfg.nonlinearity,
                dropout=recurrent_dropout,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unsupported recurrent architecture: {cfg.architecture}")
        self.final_norm = nn.LayerNorm(cfg.hidden_dim)
        self.head = nn.Linear(cfg.hidden_dim, mod, bias=False)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        output, _ = self.recurrent(self.sequence(data))
        return self.head(self.final_norm(output[:, -1, :]))

    def classifier_matrix(self) -> torch.Tensor:
        return self.head.weight


class LegacyModularTransformer(nn.Module):
    """Bias-free one-block Transformer matching the original probe style."""

    def __init__(self, mod: int, cfg: TrialConfig, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        d_model = cfg.model_dim
        d_head = d_model // cfg.n_heads
        d_mlp = cfg.ff_mult * d_model
        self.mod = mod
        self.n_heads = cfg.n_heads
        self.d_head = d_head
        self.W_E = nn.Parameter(torch.randn(mod + 1, d_model) / math.sqrt(d_model))
        self.W_pos = nn.Parameter(torch.randn(3, d_model) / math.sqrt(d_model))
        self.W_Q = nn.Parameter(
            torch.randn(cfg.n_heads, d_model, d_head) / math.sqrt(d_model)
        )
        self.W_K = nn.Parameter(
            torch.randn(cfg.n_heads, d_model, d_head) / math.sqrt(d_model)
        )
        self.W_V = nn.Parameter(
            torch.randn(cfg.n_heads, d_model, d_head) / math.sqrt(d_model)
        )
        self.W_O = nn.Parameter(
            torch.randn(cfg.n_heads, d_head, d_model)
            / math.sqrt(cfg.n_heads * d_head)
        )
        self.W_in = nn.Parameter(torch.randn(d_model, d_mlp) / math.sqrt(d_model))
        self.W_out = nn.Parameter(torch.randn(d_mlp, d_model) / math.sqrt(d_mlp))
        self.W_U = nn.Parameter(torch.randn(d_model, mod) / math.sqrt(d_model))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        equals = self.mod * torch.ones(
            data.shape[0], dtype=torch.long, device=data.device
        )
        tokens = torch.stack((data[:, 0], data[:, 1], equals), dim=1)
        resid = self.W_E[tokens] + self.W_pos.unsqueeze(0)
        q = torch.einsum("bpd,hdf->bphf", resid, self.W_Q)
        k = torch.einsum("bpd,hdf->bphf", resid, self.W_K)
        v = torch.einsum("bpd,hdf->bphf", resid, self.W_V)
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.d_head)
        mask = torch.tril(torch.ones(3, 3, device=data.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), -1e9)
        attn = torch.softmax(scores, dim=-1)
        z = torch.einsum("bhqk,bkhd->bqhd", attn, v)
        attn_out = torch.einsum("bqhd,hdf->bqf", z, self.W_O)
        resid = resid + attn_out
        resid = resid + F.relu(resid @ self.W_in) @ self.W_out
        return resid[:, -1, :] @ self.W_U

    def embedding_matrix(self) -> torch.Tensor:
        return self.W_E[: self.mod]

    def classifier_matrix(self) -> torch.Tensor:
        return self.W_U.T


class SharedBiasFreeRecurrent(nn.Module):
    """Shared token codebook with a bias-free recurrent core and linear head."""

    def __init__(self, mod: int, cfg: TrialConfig, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.mod = mod
        self.embedding = nn.Embedding(mod + 1, cfg.model_dim)
        nn.init.normal_(self.embedding.weight, std=1.0 / math.sqrt(cfg.model_dim))
        recurrent_dropout = cfg.dropout if cfg.layers > 1 else 0.0
        if cfg.architecture == "lstm":
            self.recurrent = nn.LSTM(
                cfg.model_dim,
                cfg.hidden_dim,
                num_layers=cfg.layers,
                bias=False,
                dropout=recurrent_dropout,
                batch_first=True,
            )
        elif cfg.architecture == "rnn":
            self.recurrent = nn.RNN(
                cfg.model_dim,
                cfg.hidden_dim,
                num_layers=cfg.layers,
                nonlinearity=cfg.nonlinearity,
                bias=False,
                dropout=recurrent_dropout,
                batch_first=True,
            )
        else:
            raise ValueError(cfg.architecture)
        self.head = nn.Linear(cfg.hidden_dim, mod, bias=False)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        equals = self.mod * torch.ones(
            data.shape[0], dtype=torch.long, device=data.device
        )
        tokens = torch.stack((data[:, 0], data[:, 1], equals), dim=1)
        output, _ = self.recurrent(self.embedding(tokens))
        return self.head(output[:, -1, :])

    def embedding_matrix(self) -> torch.Tensor:
        return self.embedding.weight[: self.mod]

    def classifier_matrix(self) -> torch.Tensor:
        return self.head.weight


class SharedTiedRecurrent(nn.Module):
    """Bias-free recurrent model with standard input/output weight tying."""

    def __init__(self, mod: int, cfg: TrialConfig, seed: int):
        super().__init__()
        if cfg.model_dim != cfg.hidden_dim:
            raise ValueError("shared_tied requires model_dim == hidden_dim")
        torch.manual_seed(seed)
        self.mod = mod
        self.embedding = nn.Embedding(mod + 1, cfg.model_dim)
        nn.init.normal_(self.embedding.weight, std=1.0 / math.sqrt(cfg.model_dim))
        if cfg.architecture == "lstm":
            self.recurrent = nn.LSTM(
                cfg.model_dim,
                cfg.hidden_dim,
                num_layers=cfg.layers,
                bias=False,
                batch_first=True,
            )
        elif cfg.architecture == "rnn":
            self.recurrent = nn.RNN(
                cfg.model_dim,
                cfg.hidden_dim,
                num_layers=cfg.layers,
                nonlinearity=cfg.nonlinearity,
                bias=False,
                batch_first=True,
            )
        else:
            raise ValueError(cfg.architecture)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        equals = self.mod * torch.ones(
            data.shape[0], dtype=torch.long, device=data.device
        )
        tokens = torch.stack((data[:, 0], data[:, 1], equals), dim=1)
        output, _ = self.recurrent(self.embedding(tokens))
        return self.logit_scale * output[:, -1, :] @ self.embedding.weight[: self.mod].T

    def embedding_matrix(self) -> torch.Tensor:
        return self.embedding.weight[: self.mod]

    def classifier_matrix(self) -> torch.Tensor:
        return self.embedding.weight[: self.mod]


def build_model(mod: int, cfg: TrialConfig, seed: int) -> nn.Module:
    if cfg.architecture == "transformer":
        if cfg.variant == "legacy":
            return LegacyModularTransformer(mod, cfg, seed)
        return ModularTransformer(mod, cfg, seed)
    if cfg.variant == "shared_biasfree":
        return SharedBiasFreeRecurrent(mod, cfg, seed)
    if cfg.variant == "shared_tied":
        return SharedTiedRecurrent(mod, cfg, seed)
    return ModularRecurrent(mod, cfg, seed)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_data(mod: int, frac_train: float, seed: int, device: str):
    elems = torch.arange(mod, dtype=torch.long)
    data = torch.cartesian_prod(elems, elems)
    labels = (data[:, 0] + data[:, 1]) % mod
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(data.shape[0], generator=generator)
    data, labels = data[order], labels[order]
    n_train = int(frac_train * data.shape[0])
    return (
        data[:n_train].to(device),
        labels[:n_train].to(device),
        data[n_train:].to(device),
        labels[n_train:].to(device),
    )


def split_search_train(
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    validation_fraction: float,
):
    n_search = int((1.0 - validation_fraction) * train_data.shape[0])
    return (
        train_data[:n_search],
        train_labels[:n_search],
        train_data[n_search:],
        train_labels[n_search:],
    )


@torch.no_grad()
def evaluate(model, data, labels) -> dict[str, float]:
    model.eval()
    logits = model(data)
    loss = F.cross_entropy(logits.double(), labels).item()
    acc = (logits.argmax(dim=-1) == labels).float().mean().item()
    return {"loss": loss, "acc": acc}


def scaled_pca(matrix: np.ndarray):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(matrix)
    coords -= coords.mean(axis=0, keepdims=True)
    rms_radius = np.sqrt(np.mean(np.sum(coords**2, axis=1)))
    if rms_radius > 0:
        coords /= rms_radius
    return coords, pca.explained_variance_ratio_.tolist()


def best_automorphism(coords: np.ndarray, mod: int) -> dict[str, float | int]:
    centered = coords - coords.mean(axis=0, keepdims=True)
    z = centered[:, 0] + 1j * centered[:, 1]
    denom = np.abs(z)
    denom[denom == 0] = 1.0
    z = z / denom
    labels = np.arange(mod)
    best: dict[str, float | int] = {
        "score": -1.0,
        "multiplier": 1,
        "orientation": 1,
    }
    for k in range(1, mod):
        if math.gcd(k, mod) != 1:
            continue
        phase = 2 * np.pi * ((k * labels) % mod) / mod
        for orientation in (1, -1):
            target = np.exp(1j * orientation * phase)
            score = abs(np.mean(z * np.conjugate(target)))
            if score > float(best["score"]):
                best = {
                    "score": float(score),
                    "multiplier": int(k),
                    "orientation": int(orientation),
                }
    return best


def circle_metrics(coords: np.ndarray, mod: int, auto: dict[str, Any]):
    centered = coords - coords.mean(axis=0, keepdims=True)
    radii = np.sqrt(np.sum(centered**2, axis=1))
    angles = np.arctan2(centered[:, 1], centered[:, 0])
    labels = np.arange(mod)
    target = (
        int(auto["orientation"])
        * 2
        * np.pi
        * ((int(auto["multiplier"]) * labels) % mod)
        / mod
    )
    offset = np.angle(np.mean(np.exp(1j * (angles - target))))
    errors = np.angle(np.exp(1j * (angles - target - offset)))
    return {
        "radial_cv": float(np.std(radii) / (np.mean(radii) + 1e-12)),
        "angular_rmse": float(np.sqrt(np.mean(errors**2))),
        "angular_mae": float(np.mean(np.abs(errors))),
    }


@torch.no_grad()
def geometry_metrics(model: nn.Module, mod: int) -> dict[str, Any]:
    embed = model.embedding_matrix().detach().cpu().numpy()
    classifier = model.classifier_matrix().detach().cpu().numpy()
    embed_coords, embed_var = scaled_pca(embed)
    classifier_coords, classifier_var = scaled_pca(classifier)
    embed_auto = best_automorphism(embed_coords, mod)
    classifier_auto = best_automorphism(classifier_coords, mod)
    return {
        "embedding_coords": embed_coords,
        "classifier_coords": classifier_coords,
        "embedding_pca_var": embed_var,
        "classifier_pca_var": classifier_var,
        "embedding_automorphism": embed_auto,
        "classifier_automorphism": classifier_auto,
        "embedding_circle": circle_metrics(embed_coords, mod, embed_auto),
        "classifier_circle": circle_metrics(classifier_coords, mod, classifier_auto),
        "circle_mean": 0.5
        * (float(embed_auto["score"]) + float(classifier_auto["score"])),
    }


def objective(train: dict[str, float], valid: dict[str, float], circle: float) -> float:
    # Generalization dominates, followed by fitting and cyclic geometry.
    return (
        4.0 * valid["acc"]
        + train["acc"]
        + 0.75 * circle
        - 0.03 * min(valid["loss"], 20.0)
    )


def save_history(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def train_to_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    eval_data: torch.Tensor,
    eval_labels: torch.Tensor,
    start_epoch: int,
    target_epoch: int,
    log_every: int,
    history: list[dict[str, float]],
    start_time: float,
):
    for epoch in range(start_epoch + 1, target_epoch + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(train_data).double(), train_labels)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % log_every == 0 or epoch == target_epoch:
            train_metrics = evaluate(model, train_data, train_labels)
            eval_metrics = evaluate(model, eval_data, eval_labels)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "eval_loss": eval_metrics["loss"],
                    "train_acc": train_metrics["acc"],
                    "eval_acc": eval_metrics["acc"],
                    "elapsed_sec": time.time() - start_time,
                }
            )
    return history


def optimizer_for(model: nn.Module, cfg: TrialConfig):
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.98),
        weight_decay=cfg.weight_decay,
    )


def run_search_stage(
    cfg: TrialConfig,
    stage: int,
    target_epoch: int,
    root: Path,
    mod: int,
    seed: int,
    device: str,
    search_train_data: torch.Tensor,
    search_train_labels: torch.Tensor,
    validation_data: torch.Tensor,
    validation_labels: torch.Tensor,
    log_every: int,
):
    trial_dir = root / "search" / cfg.architecture / cfg.name
    trial_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = trial_dir / "search_checkpoint.pt"
    set_seed(seed)
    model = build_model(mod, cfg, seed).to(device)
    optimizer = optimizer_for(model, cfg)
    history: list[dict[str, float]] = []
    start_epoch = 0
    elapsed_before = 0.0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"])
        history = checkpoint.get("history", [])
        elapsed_before = float(checkpoint.get("elapsed_sec", 0.0))
    start_time = time.time() - elapsed_before
    if start_epoch < target_epoch:
        history = train_to_epoch(
            model,
            optimizer,
            search_train_data,
            search_train_labels,
            validation_data,
            validation_labels,
            start_epoch,
            target_epoch,
            log_every,
            history,
            start_time,
        )
    train_metrics = evaluate(model, search_train_data, search_train_labels)
    valid_metrics = evaluate(model, validation_data, validation_labels)
    geometry = geometry_metrics(model, mod)
    score = objective(train_metrics, valid_metrics, geometry["circle_mean"])
    elapsed = time.time() - start_time
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": target_epoch,
        "history": history,
        "elapsed_sec": elapsed,
        "config": asdict(cfg),
    }
    torch.save(checkpoint, checkpoint_path)
    save_history(trial_dir / "history.csv", history)
    result = {
        **asdict(cfg),
        "trial_name": cfg.name,
        "stage": stage,
        "epochs": target_epoch,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "train_loss": train_metrics["loss"],
        "train_acc": train_metrics["acc"],
        "validation_loss": valid_metrics["loss"],
        "validation_acc": valid_metrics["acc"],
        "embedding_circle_score": geometry["embedding_automorphism"]["score"],
        "classifier_circle_score": geometry["classifier_automorphism"]["score"],
        "circle_mean": geometry["circle_mean"],
        "objective": score,
        "elapsed_sec": elapsed,
        "status": "ok",
    }
    (trial_dir / f"stage{stage}_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def config_catalog() -> dict[str, list[TrialConfig]]:
    return {
        "transformer": [
            TrialConfig("transformer", 128, 128, 1, 1e-3, 1.0, 0.0, 4, 2),
            TrialConfig("transformer", 128, 128, 2, 1e-3, 0.3, 0.0, 4, 4),
            TrialConfig("transformer", 192, 192, 1, 1e-3, 0.3, 0.0, 6, 2),
            TrialConfig("transformer", 192, 192, 2, 3e-4, 1.0, 0.0, 6, 2),
            TrialConfig("transformer", 256, 256, 1, 3e-4, 1.0, 0.0, 8, 2),
            TrialConfig("transformer", 128, 128, 2, 3e-3, 0.1, 0.0, 4, 2),
            TrialConfig("transformer", 256, 256, 2, 1e-3, 0.1, 0.0, 8, 2),
            TrialConfig("transformer", 192, 192, 2, 1e-3, 0.3, 0.1, 6, 4),
        ],
        "lstm": [
            TrialConfig("lstm", 128, 128, 1, 1e-3, 1.0),
            TrialConfig("lstm", 128, 256, 1, 1e-3, 0.3),
            TrialConfig("lstm", 256, 128, 1, 1e-3, 0.3),
            TrialConfig("lstm", 128, 128, 2, 3e-4, 1.0),
            TrialConfig("lstm", 256, 256, 1, 3e-4, 1.0),
            TrialConfig("lstm", 128, 256, 2, 1e-3, 0.1),
            TrialConfig("lstm", 256, 256, 2, 1e-3, 0.3, 0.1),
            TrialConfig("lstm", 128, 128, 1, 3e-3, 0.1),
        ],
        "rnn": [
            TrialConfig("rnn", 128, 128, 1, 1e-3, 1.0),
            TrialConfig("rnn", 128, 256, 1, 1e-3, 0.3),
            TrialConfig("rnn", 256, 128, 1, 1e-3, 0.3),
            TrialConfig("rnn", 128, 128, 2, 3e-4, 1.0),
            TrialConfig("rnn", 256, 256, 1, 3e-4, 1.0),
            TrialConfig("rnn", 128, 256, 2, 1e-3, 0.1),
            TrialConfig("rnn", 256, 256, 2, 1e-3, 0.3, 0.1),
            TrialConfig("rnn", 128, 128, 1, 3e-3, 0.1),
        ],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def display_values(mod: int, auto: dict[str, Any]) -> np.ndarray:
    multiplier = int(auto["multiplier"])
    if int(auto["orientation"]) < 0:
        multiplier = (-multiplier) % mod
    return (multiplier * np.arange(mod)) % mod


def save_pca_figure(geometry: dict[str, Any], mod: int, path: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), constrained_layout=True)
    for ax, key, auto_key, panel_title in (
        (axes[0], "embedding_coords", "embedding_automorphism", "Input embedding $E_x$"),
        (axes[1], "classifier_coords", "classifier_automorphism", "Output classifier"),
    ):
        coords = geometry[key]
        auto = geometry[auto_key]
        theta = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(theta), np.sin(theta), color="0.15", lw=0.6, alpha=0.55)
        points = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=display_values(mod, auto),
            cmap="hsv",
            vmin=0,
            vmax=mod - 1,
            s=24,
            edgecolors="0.25",
            linewidths=0.45,
        )
        ax.set_title(f"{panel_title}\ncyclic match={float(auto['score']):.4f}")
        ax.set_xlabel("PCA1")
        ax.set_ylabel("PCA2")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.grid(alpha=0.12)
    fig.colorbar(points, ax=axes, shrink=0.78, label=f"Integer (mod {mod})")
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_training_figure(history: list[dict[str, float]], path: Path, title: str):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), constrained_layout=True)
    axes[0].plot(epochs, [row["train_acc"] for row in history], label="train")
    axes[0].plot(epochs, [row["eval_acc"] for row in history], label="test")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[1].plot(epochs, [row["eval_loss"] for row in history], label="test")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-entropy (log scale)")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_final(
    cfg: TrialConfig,
    root: Path,
    mod: int,
    seed: int,
    device: str,
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    test_data: torch.Tensor,
    test_labels: torch.Tensor,
    epochs: int,
    log_every: int,
):
    final_dir = root / "final" / cfg.architecture
    final_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = final_dir / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    set_seed(seed)
    model = build_model(mod, cfg, seed).to(device)
    optimizer = optimizer_for(model, cfg)
    search_checkpoint_path = (
        root / "search" / cfg.architecture / cfg.name / "search_checkpoint.pt"
    )
    start_epoch = 0
    search_elapsed = 0.0
    if search_checkpoint_path.exists():
        checkpoint = torch.load(
            search_checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"])
        search_elapsed = float(checkpoint.get("elapsed_sec", 0.0))
    history: list[dict[str, float]] = []
    start_time = time.time()
    history = train_to_epoch(
        model,
        optimizer,
        train_data,
        train_labels,
        test_data,
        test_labels,
        start_epoch,
        epochs,
        log_every,
        history,
        start_time,
    )
    train_metrics = evaluate(model, train_data, train_labels)
    test_metrics = evaluate(model, test_data, test_labels)
    geometry = geometry_metrics(model, mod)
    elapsed = time.time() - start_time
    result = {
        **asdict(cfg),
        "trial_name": cfg.name,
        "continued_from_search_epoch": start_epoch,
        "epochs": epochs,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "train_loss": train_metrics["loss"],
        "train_acc": train_metrics["acc"],
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "embedding_circle_score": geometry["embedding_automorphism"]["score"],
        "classifier_circle_score": geometry["classifier_automorphism"]["score"],
        "circle_mean": geometry["circle_mean"],
        "embedding_pca_var_2d": sum(geometry["embedding_pca_var"]),
        "classifier_pca_var_2d": sum(geometry["classifier_pca_var"]),
        "embedding_radial_cv": geometry["embedding_circle"]["radial_cv"],
        "classifier_radial_cv": geometry["classifier_circle"]["radial_cv"],
        "embedding_angular_rmse": geometry["embedding_circle"]["angular_rmse"],
        "classifier_angular_rmse": geometry["classifier_circle"]["angular_rmse"],
        "embedding_multiplier": geometry["embedding_automorphism"]["multiplier"],
        "embedding_orientation": geometry["embedding_automorphism"]["orientation"],
        "classifier_multiplier": geometry["classifier_automorphism"]["multiplier"],
        "classifier_orientation": geometry["classifier_automorphism"]["orientation"],
        "final_finetune_elapsed_sec": elapsed,
        "search_elapsed_sec": search_elapsed,
        "elapsed_sec": elapsed + search_elapsed,
        "success_accuracy": bool(
            train_metrics["acc"] >= 0.999 and test_metrics["acc"] >= 0.999
        ),
        "success_mlp_like": bool(
            train_metrics["acc"] >= 0.999
            and test_metrics["acc"] >= 0.999
            and geometry["embedding_automorphism"]["score"] >= 0.98
            and geometry["classifier_automorphism"]["score"] >= 0.98
        ),
    }
    torch.save(
        {"model_state": model.state_dict(), "config": asdict(cfg)},
        final_dir / "model.pt",
    )
    save_history(final_dir / "history.csv", history)
    save_pca_figure(
        geometry,
        mod,
        final_dir / "pca.png",
        f"{cfg.architecture.upper()}: mod-{mod} terminal cyclic geometry",
    )
    save_training_figure(
        history,
        final_dir / "training.png",
        f"{cfg.architecture.upper()}: training and generalization",
    )
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def select_top(rows: list[dict[str, Any]], keep: int) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            row.get("status") == "ok",
            row.get("validation_acc", -1.0) >= 0.99,
            row.get("objective", -1e9),
        ),
        reverse=True,
    )[:keep]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mod", type=int, default=97)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--frac-train", type=float, default=0.7)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--stage1-epochs", type=int, default=600)
    parser.add_argument("--stage2-epochs", type=int, default=2500)
    parser.add_argument("--final-epochs", type=int, default=10000)
    parser.add_argument("--trials-per-architecture", type=int, default=8)
    parser.add_argument("--stage2-keep", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument(
        "--architectures", default="transformer,lstm,rnn", help="Comma-separated"
    )
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    train_data, train_labels, test_data, test_labels = make_data(
        args.mod, args.frac_train, args.seed, args.device
    )
    search_train_data, search_train_labels, valid_data, valid_labels = split_search_train(
        train_data, train_labels, args.validation_fraction
    )
    catalog = config_catalog()
    architectures = [x.strip() for x in args.architectures.split(",") if x.strip()]
    manifest = {
        "mod": args.mod,
        "seed": args.seed,
        "frac_train": args.frac_train,
        "original_train_pairs": int(train_data.shape[0]),
        "search_train_pairs": int(search_train_data.shape[0]),
        "validation_pairs": int(valid_data.shape[0]),
        "test_pairs": int(test_data.shape[0]),
        "stage1_epochs": args.stage1_epochs,
        "stage2_epochs": args.stage2_epochs,
        "final_epochs": args.final_epochs,
        "architectures": architectures,
        "selection_note": "Test split is untouched until final retraining.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    all_search_rows: list[dict[str, Any]] = []
    best_configs: dict[str, TrialConfig] = {}
    for architecture in architectures:
        configs = catalog[architecture][: args.trials_per_architecture]
        stage1_rows = []
        for cfg in configs:
            try:
                row = run_search_stage(
                    cfg,
                    1,
                    args.stage1_epochs,
                    args.output_dir,
                    args.mod,
                    args.seed,
                    args.device,
                    search_train_data,
                    search_train_labels,
                    valid_data,
                    valid_labels,
                    args.log_every,
                )
            except (RuntimeError, ValueError) as exc:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                row = {
                    **asdict(cfg),
                    "trial_name": cfg.name,
                    "stage": 1,
                    "epochs": args.stage1_epochs,
                    "status": "failed",
                    "error": str(exc),
                }
            stage1_rows.append(row)
            all_search_rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        promoted = select_top(stage1_rows, args.stage2_keep)
        stage2_rows = []
        config_by_name = {cfg.name: cfg for cfg in configs}
        for promoted_row in promoted:
            cfg = config_by_name[promoted_row["trial_name"]]
            try:
                row = run_search_stage(
                    cfg,
                    2,
                    args.stage2_epochs,
                    args.output_dir,
                    args.mod,
                    args.seed,
                    args.device,
                    search_train_data,
                    search_train_labels,
                    valid_data,
                    valid_labels,
                    args.log_every,
                )
            except (RuntimeError, ValueError) as exc:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                row = {
                    **asdict(cfg),
                    "trial_name": cfg.name,
                    "stage": 2,
                    "epochs": args.stage2_epochs,
                    "status": "failed",
                    "error": str(exc),
                }
            stage2_rows.append(row)
            all_search_rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        winner = select_top(stage2_rows, 1)[0]
        best_configs[architecture] = config_by_name[winner["trial_name"]]
        write_csv(args.output_dir / "search_results.csv", all_search_rows)
        (args.output_dir / "best_params.json").write_text(
            json.dumps(
                {name: asdict(cfg) for name, cfg in best_configs.items()}, indent=2
            ),
            encoding="utf-8",
        )

    final_rows = []
    for architecture in architectures:
        cfg = best_configs[architecture]
        result = run_final(
            cfg,
            args.output_dir,
            args.mod,
            args.seed,
            args.device,
            train_data,
            train_labels,
            test_data,
            test_labels,
            args.final_epochs,
            args.log_every,
        )
        final_rows.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    write_csv(args.output_dir / "final_comparison.csv", final_rows)
    (args.output_dir / "final_comparison.json").write_text(
        json.dumps(final_rows, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
