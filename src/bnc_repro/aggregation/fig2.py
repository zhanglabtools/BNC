from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import read_run_metrics, write_frame


def first_sustained_onset(values: pd.Series, epochs: pd.Series, threshold: float = 0.9) -> int | None:
    mask = values.to_numpy() >= threshold
    points = epochs.to_numpy()
    for index in range(len(mask) - 1):
        if bool(mask[index] and mask[index + 1]):
            return int(points[index])
    return None


def aggregate_fig2(runs_root: str | Path, output_dir: str | Path) -> dict[str, Path]:
    raw = read_run_metrics(runs_root)
    required = {
        "architecture",
        "K",
        "seed",
        "epoch",
        "classifier_best_cyclic_score",
        "embedding_best_cyclic_score",
    }
    if not required.issubset(raw.columns):
        raise ValueError(f"Figure 2 metrics missing columns: {sorted(required - set(raw.columns))}")
    onset_rows = []
    for keys, group in raw.sort_values("epoch").groupby(["architecture", "K", "seed"]):
        classifier = first_sustained_onset(group["classifier_best_cyclic_score"], group["epoch"])
        embedding = first_sustained_onset(group["embedding_best_cyclic_score"], group["epoch"])
        onset_rows.append(
            {
                "architecture": keys[0],
                "K": keys[1],
                "seed": keys[2],
                "classifier_onset": classifier,
                "embedding_onset": embedding,
                "lead": None if classifier is None or embedding is None else embedding - classifier,
                "classifier_right_censored": classifier is None,
                "embedding_right_censored": embedding is None,
            }
        )
    aggregate = (
        raw.groupby(["architecture", "K", "epoch"], as_index=False)
        .agg(
            classifier_mean=("classifier_best_cyclic_score", "mean"),
            classifier_sample_sd=("classifier_best_cyclic_score", "std"),
            embedding_mean=("embedding_best_cyclic_score", "mean"),
            embedding_sample_sd=("embedding_best_cyclic_score", "std"),
            n_seeds=("seed", "nunique"),
        )
    )
    output = Path(output_dir)
    return {
        "raw": write_frame(raw, output / "raw_metrics.csv"),
        "aggregate": write_frame(aggregate, output / "aggregate_metrics.csv"),
        "onsets": write_frame(pd.DataFrame(onset_rows), output / "onset_by_run.csv"),
    }

