from __future__ import annotations

import copy

import torch

from bnc_repro.config import ConfigError, validate_config
from bnc_repro.models.base import load_legacy_state_dict
from bnc_repro.models.registry import ARCHITECTURES, build_model


def test_all_architectures_forward_and_public_shapes() -> None:
    pairs = torch.tensor([[0, 1], [3, 4], [7, 2]])
    for architecture in ARCHITECTURES:
        model = build_model(
            architecture, 17, 1, {"mlp_embedding_dim": 32, "mlp_hidden_dim": 24}
        )
        assert model(pairs).shape == (3, 17)
        assert model.penultimate_features(pairs).shape[0] == 3
        assert model.embedding_matrix().shape[0] == 17
        assert model.classifier_matrix().shape[0] == 17
        assert model.classifier_matrix().data_ptr() != model.embedding_matrix().data_ptr()


def test_legacy_state_dict_names_load_without_migration_loss() -> None:
    for architecture in ARCHITECTURES:
        source = build_model(architecture, 17, 3)
        target = build_model(architecture, 17, 3)
        wrapped = {f"module.{key}": value.clone() for key, value in source.state_dict().items()}
        load_legacy_state_dict(target, wrapped)
        for key, value in source.state_dict().items():
            assert torch.equal(value, target.state_dict()[key])


def test_formal_s2_rejects_script_defaults() -> None:
    config = {
        "experiment": "fig_s2",
        "protocol": "rank2_finetune",
        "formal": True,
        "grid": {
            "architectures": ["mlp", "transformer", "lstm", "rnn"],
            "moduli": [79, 97, 113],
            "seeds": [1, 2, 3, 4, 5],
        },
        "training": {"epochs": 6000, "weight_decay": 0.5},
        "regularizer": {
            "overall_coefficient": 1.0,
            "tail_weight": 5.0,
            "balance_weight": 0.5,
            "participation_weight": 1.0,
            "ramp_epochs": 200,
        },
    }
    try:
        validate_config(config)
    except ConfigError as error:
        assert "tail_weight" in str(error)
    else:
        raise AssertionError("formal S2 accepted the non-formal argparse defaults")

