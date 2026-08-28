from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a resolved experiment configuration violates its profile."""


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    with source.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError(f"configuration must be a mapping: {source}")
    resolved = deepcopy(config)
    resolved["_config_path"] = str(source)
    validate_config(resolved)
    return resolved


def _expect(config: dict[str, Any], dotted: str, expected: Any) -> None:
    value: Any = config
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ConfigError(f"missing required setting: {dotted}")
        value = value[key]
    if value != expected:
        raise ConfigError(f"{dotted} must be {expected!r}, got {value!r}")


def validate_config(config: dict[str, Any]) -> None:
    experiment = config.get("experiment")
    protocol = config.get("protocol")
    if not experiment or not protocol:
        raise ConfigError("configuration requires experiment and protocol")
    if not config.get("formal", True):
        return

    common_grids = {
        "fig2": ([79, 97, 113], [1, 2, 3, 4, 5]),
        "dense": ([79, 97, 113], [1, 2, 3, 4, 5]),
        "fig_s2": ([79, 97, 113], [1, 2, 3, 4, 5]),
        "fig_s3": ([79, 97, 113], [1, 2, 3, 4, 5]),
        "fig_s4": ([79, 97, 113], [1, 2, 3, 4, 5]),
    }
    if experiment in common_grids:
        moduli, seeds = common_grids[experiment]
        _expect(config, "grid.moduli", moduli)
        _expect(config, "grid.seeds", seeds)

    if experiment.startswith("fig1"):
        _expect(config, "grid.moduli", [97])
        _expect(config, "model.architecture", "mlp")
        _expect(config, "model.embedding_dim", 256)
        _expect(config, "model.hidden_dim", 128)
        _expect(config, "training.epochs", 10000)
        _expect(config, "training.lr", 0.001)
        _expect(config, "training.weight_decay", 1.0)
        _expect(config, "training.betas", [0.9, 0.98])
    elif experiment == "fig2":
        _expect(config, "model.mlp_embedding_dim", 512)
        _expect(config, "model.mlp_hidden_dim", 512)
        _expect(config, "training.epochs", 10000)
        _expect(config, "training.log_every", 50)
        _expect(config, "training.embedding_lr", 0.0002)
        _expect(config, "training.other_lr", 0.003)
        _expect(config, "training.embedding_weight_decay", 0.0)
        _expect(config, "training.other_weight_decay", 0.4)
        _expect(config, "training.betas", [0.9, 0.98])
    elif experiment == "dense":
        _expect(config, "model.mlp_embedding_dim", 256)
        _expect(config, "model.mlp_hidden_dim", 128)
        _expect(config, "training.epochs", 10000)
        _expect(config, "training.embedding_lr", 0.0002)
        _expect(config, "training.other_lr", 0.002)
        _expect(config, "training.other_weight_decay", 0.8)
        _expect(config, "training.betas", [0.9, 0.98])
    elif experiment == "fig_s1":
        _expect(config, "grid.moduli", [97])
        _expect(config, "grid.seeds", [1, 2, 3, 4, 5])
        _expect(config, "grid.lambdas", [0.01, 0.1, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
        _expect(config, "rank.max_rank", 16)
        _expect(config, "rank.target_rank", 2)
        _expect(config, "rank.gate_start", 1)
        _expect(config, "rank.gate_end", 6000)
        _expect(config, "training.epochs", 10000)
        _expect(config, "training.target_loss_weight", 1.0)
        _expect(config, "training.weight_decay", 0.5)
    elif experiment == "fig_s2":
        _expect(config, "training.epochs", 6000)
        _expect(config, "training.weight_decay", 0.5)
        _expect(config, "regularizer.overall_coefficient", 1.0)
        _expect(config, "regularizer.tail_weight", 0.0)
        _expect(config, "regularizer.balance_weight", 0.0)
        _expect(config, "regularizer.participation_weight", 5.0)
        _expect(config, "regularizer.ramp_epochs", 200)
    elif experiment in {"fig_s3", "fig_s4"}:
        _expect(config, "training.epochs", 20000)
        _expect(config, "training.requested_checkpoints", 321)
        _expect(config, "training.embedding_lr", 0.001)
        _expect(config, "training.other_lr", 0.001)
        _expect(config, "training.embedding_weight_decay", 0.0)
        _expect(config, "training.other_weight_decay", 0.4)
        _expect(config, "training.betas", [0.9, 0.999])
        _expect(config, "metrics.shuffle_controls", 16)

