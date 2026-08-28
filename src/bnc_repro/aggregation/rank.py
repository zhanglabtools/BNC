from __future__ import annotations

from pathlib import Path

from .common import read_run_metrics, write_frame


def aggregate_rank(runs_root: str | Path, output: str | Path, *, figure: str) -> Path:
    raw = read_run_metrics(runs_root)
    if figure == "fig_s1":
        keys = ["architecture", "lambda_reg", "epoch"]
        metric = "classifier_participation_rank"
        count = "valid_seed_count"
    elif figure == "fig_s2":
        keys = ["architecture", "K", "epoch"]
        metric = "effective_mean_participation_rank"
        count = "valid_seed_count"
    else:
        raise ValueError(figure)
    summary = raw.groupby(keys, as_index=False).agg(
        **{
            f"{metric}_mean": (metric, "mean"),
            f"{metric}_std": (metric, "std"),
            count: ("seed", "nunique"),
        }
    )
    return write_frame(summary, output)

