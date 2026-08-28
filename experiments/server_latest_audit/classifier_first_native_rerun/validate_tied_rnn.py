"""Show why the supplied tied RNN cannot test classifier-first timing."""

from __future__ import annotations

import json

import numpy as np

from supplied_architecture_search import (
    SharedTiedRecurrent,
    TrialConfig,
    best_automorphism,
    scaled_pca,
)


def main() -> None:
    config = TrialConfig(
        architecture="rnn",
        model_dim=128,
        hidden_dim=128,
        layers=1,
        lr=1e-3,
        weight_decay=3.0,
        nonlinearity="relu",
        variant="shared_tied",
    )
    model = SharedTiedRecurrent(mod=97, cfg=config, seed=1)
    embedding = model.embedding_matrix().detach().cpu().numpy()
    classifier = model.classifier_matrix().detach().cpu().numpy()
    embedding_score = float(best_automorphism(scaled_pca(embedding)[0], 97)["score"])
    classifier_score = float(best_automorphism(scaled_pca(classifier)[0], 97)["score"])
    print(
        json.dumps(
            {
                "architecture": "supplied shared_tied ReLU RNN",
                "K": 97,
                "seed": 1,
                "same_storage": bool(model.embedding_matrix().data_ptr() == model.classifier_matrix().data_ptr()),
                "max_abs_matrix_difference": float(np.max(np.abs(embedding - classifier))),
                "embedding_best_cyclic_score": embedding_score,
                "classifier_best_cyclic_score": classifier_score,
                "absolute_score_difference": abs(embedding_score - classifier_score),
                "classifier_first_identifiable": False,
                "reason": "embedding_matrix() and classifier_matrix() return the same parameter by construction",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
