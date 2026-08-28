from __future__ import annotations

from pathlib import Path

from .common import read_run_metrics, write_frame


def aggregate_alignment(
    runs_root: str | Path, output: str | Path, *, centered: bool
) -> Path:
    raw = read_run_metrics(runs_root)
    if centered:
        matched = "centered_feature_classifier_alignment"
        shuffled = "shuffled_centered_feature_classifier_alignment"
        output_columns = {
            f"{matched}_mean": "centered_matched_mean",
            f"{matched}_std": "centered_matched_std",
            f"{shuffled}_mean": "centered_shuffled_mean",
            f"{shuffled}_std": "centered_shuffled_std",
        }
    else:
        matched = "matched_alignment"
        shuffled = "shuffled_alignment"
        output_columns = {
            f"{matched}_mean": "matched_mean",
            f"{matched}_std": "matched_std",
            f"{shuffled}_mean": "shuffled_mean",
            f"{shuffled}_std": "shuffled_std",
        }
    summary = (
        raw.groupby(["architecture", "K", "epoch"], as_index=False)
        .agg(
            **{
                f"{matched}_mean": (matched, "mean"),
                f"{matched}_std": (matched, "std"),
                f"{shuffled}_mean": (shuffled, "mean"),
                f"{shuffled}_std": (shuffled, "std"),
                "n_seeds": ("seed", "nunique"),
            }
        )
        .rename(columns=output_columns)
    )
    return write_frame(summary, output)

