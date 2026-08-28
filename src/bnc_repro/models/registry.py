from __future__ import annotations

from typing import Any

from .base import ModularArchitecture
from .mlp import RoleSpecificMLP
from .recurrent import SharedRecurrent
from .transformer import CausalTransformer


ARCHITECTURES = ("mlp", "transformer", "lstm", "rnn")


def build_model(
    architecture: str,
    modulus: int,
    seed: int,
    model_config: dict[str, Any] | None = None,
) -> ModularArchitecture:
    cfg = model_config or {}
    if architecture == "mlp":
        embedding_dim = int(cfg.get("mlp_embedding_dim", cfg.get("embedding_dim", 256)))
        hidden_dim = int(cfg.get("mlp_hidden_dim", cfg.get("hidden_dim", 128)))
        return RoleSpecificMLP(modulus, embedding_dim, hidden_dim, seed)
    if architecture == "transformer":
        return CausalTransformer(
            modulus,
            seed,
            d_model=int(cfg.get("d_model", 128)),
            n_heads=int(cfg.get("n_heads", 4)),
            d_head=int(cfg.get("d_head", 32)),
            d_mlp=int(cfg.get("d_mlp", 512)),
        )
    if architecture in {"lstm", "rnn"}:
        return SharedRecurrent(
            architecture,
            modulus,
            seed,
            embedding_dim=int(cfg.get("recurrent_embedding_dim", 128)),
            hidden_dim=int(cfg.get("recurrent_hidden_dim", 128)),
        )
    raise ValueError(f"unknown architecture: {architecture}")

