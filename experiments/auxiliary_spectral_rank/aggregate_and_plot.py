#!/usr/bin/env python3
"""Validate, aggregate, and plot the four-architecture rank-2 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter


ARCHITECTURES = ["mlp", "transformer", "lstm", "rnn"]
ARCHITECTURE_LABELS = {
    "mlp": "MLP",
    "transformer": "Transformer",
    "lstm": "LSTM",
    "rnn": "RNN",
}
K_VALUES = [79, 97, 113]
SEEDS = [1, 2, 3, 4, 5]
COLORS = {79: "#0072B2", 97: "#D55E00", 113: "#009E73"}
METRICS = {
    "classifier_participation_rank": "classifier",
    "effective_mean_participation_rank": "effective_embedding",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=root / "outputs")
    parser.add_argument("--output-root", type=Path, default=root / "artifacts")
    parser.add_argument("--font-family", default="Arial")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show-std", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def configure_style(font_family: str) -> str:
    font_path = font_manager.findfont(font_family, fallback_to_default=False)
    mpl.rcParams.update({
        "font.family": font_family,
        "font.size": 20,
        "axes.labelsize": 22,
        "xtick.labelsize": 19,
        "ytick.labelsize": 19,
        "legend.fontsize": 20,
        "axes.linewidth": 1.25,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return font_path


def expected_epochs(total: int = 10000) -> list[int]:
    values = [0]
    values.extend(range(1, min(total, 20) + 1))
    if total > 20:
        values.extend(range(25, min(total, 100) + 1, 5))
    if total > 100:
        values.extend(range(110, min(total, 500) + 1, 10))
    if total > 500:
        values.extend(range(525, min(total, 1000) + 1, 25))
    if total > 1000:
        values.extend(range(1050, min(total, 7000) + 1, 50))
    if total > 7000:
        values.extend(range(7100, total + 1, 100))
    if values[-1] != total:
        values.append(total)
    return values


def load_and_validate(input_root: Path) -> pd.DataFrame:
    frames = []
    expected = expected_epochs()
    problems: list[str] = []
    for architecture in ARCHITECTURES:
        for K in K_VALUES:
            for seed in SEEDS:
                run = input_root / "runs" / architecture / f"K{K}" / f"seed_{seed}"
                status_path = run / "status.json"
                metrics_path = run / "metrics.csv"
                if not status_path.exists() or not metrics_path.exists():
                    problems.append(f"missing {run}")
                    continue
                status = json.loads(status_path.read_text(encoding="utf-8"))
                frame = pd.read_csv(metrics_path)
                if status.get("status") != "complete":
                    problems.append(f"not complete {run}: {status.get('status')}")
                if frame["epoch"].astype(int).tolist() != expected:
                    problems.append(f"epoch schedule mismatch {run}: {len(frame)} rows")
                required = list(METRICS) + [
                    "classifier_numerical_rank", "classifier_top2_tail",
                    "tail_gate", "train_accuracy", "test_accuracy",
                ]
                if frame[required].isna().any().any() or not np.isfinite(frame[required].to_numpy(dtype=float)).all():
                    problems.append(f"non-finite metrics {run}")
                final = frame.iloc[-1]
                if (
                    float(final["tail_gate"]) != 0.0
                    or int(final["classifier_numerical_rank"]) > 2
                    or float(final["classifier_top2_tail"]) > 1e-10
                ):
                    problems.append(f"final rank-2 invariant failed {run}")
                frames.append(frame)
    if problems:
        raise RuntimeError("Validation failed:\n" + "\n".join(problems))
    raw = pd.concat(frames, ignore_index=True)
    if len(frames) != len(ARCHITECTURES) * len(K_VALUES) * len(SEEDS):
        raise RuntimeError(f"Expected 60 runs, loaded {len(frames)}")
    seed_counts = raw.groupby(["architecture", "K", "epoch"])["seed"].nunique()
    if not (seed_counts == 5).all():
        raise RuntimeError("At least one aggregate point lacks five seeds")
    return raw


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    aggregation = {
        "seed": ["nunique"],
        "train_accuracy": ["mean", "std"],
        "test_accuracy": ["mean", "std"],
    }
    for metric in METRICS:
        aggregation[metric] = ["mean", "std"]
    result = raw.groupby(["architecture", "K", "epoch"], as_index=False).agg(aggregation)
    result.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple) else column
        for column in result.columns
    ]
    return result.rename(columns={
        "architecture_": "architecture", "K_": "K", "epoch_": "epoch",
        "seed_nunique": "valid_seed_count",
    })


def apply_log_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.set_xlim(1, 10000)
    ticks = [1, 10, 100, 1000, 10000]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["1", "10", "100", "1k", "10k"])


def plot_metric(
    summary: pd.DataFrame,
    output_root: Path,
    dpi: int,
    show_std: bool,
    metric: str,
    filename: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16.8, 10.5), sharex=True, sharey=True)
    finite = summary[f"{metric}_mean"].to_numpy(dtype=float)
    deviations = summary[f"{metric}_std"].fillna(0).to_numpy(dtype=float)
    y_min = max(0.0, float(np.nanmin(finite - deviations)) - 0.35)
    y_max = float(np.nanmax(finite + deviations)) + 0.6

    for ax, architecture in zip(axes.ravel(), ARCHITECTURES):
        for K in K_VALUES:
            curve = summary.loc[
                (summary["architecture"] == architecture) & (summary["K"] == K)
                & (summary["epoch"] >= 1)
            ].sort_values("epoch")
            x = curve["epoch"].to_numpy()
            mean = curve[f"{metric}_mean"].to_numpy()
            std = curve[f"{metric}_std"].fillna(0).to_numpy()
            ax.plot(
                x, mean, color=COLORS[K], linewidth=3.0,
                solid_joinstyle="round", solid_capstyle="round", antialiased=True,
            )
            if show_std:
                ax.fill_between(
                    x, mean - std, mean + std,
                    color=COLORS[K], alpha=0.14, linewidth=0,
                )

        apply_log_axis(ax)
        ax.set_ylim(y_min, y_max)
        ax.grid(False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(1.25)
        # Shared axes normally suppress the top-row x labels and right-column
        # y labels. Keep the shared scale, but number both axes in every panel.
        ax.tick_params(
            axis="both", which="major", width=1.1, length=5,
            labelsize=19, labelbottom=True, labelleft=True,
        )
        ax.set_xlabel("Epoch", fontsize=22, fontfamily="Arial", labelpad=9)
        ax.set_ylabel(
            "Participation rank", fontsize=22,
            fontfamily="Arial", labelpad=10,
        )
        ax.set_title(
            ARCHITECTURE_LABELS[architecture], fontsize=25,
            fontweight="bold", pad=12, fontfamily="Arial",
        )

    handles = [
        Line2D([0], [0], color=COLORS[K], lw=3.2, label=f"K = {K}")
        for K in K_VALUES
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
        ncol=3, frameon=False, handlelength=3.0,
        prop={"family": "Arial", "size": 20},
    )
    fig.subplots_adjust(
        left=0.09, right=0.985, bottom=0.09, top=0.89,
        wspace=0.26, hspace=0.43,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    stem = output_root / filename
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    font_path = configure_style(args.font_family)
    raw = load_and_validate(args.input_root.resolve())
    summary = aggregate(raw)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_root / "all_runs_metrics.csv", index=False)
    summary.to_csv(output_root / "mean_std_trajectory.csv", index=False)
    final = summary.loc[summary["epoch"] == summary["epoch"].max()].copy()
    final.to_csv(output_root / "final_metrics.csv", index=False)
    plot_metric(
        summary, output_root, args.dpi, args.show_std,
        "classifier_participation_rank",
        "classifier_rank_dynamics_MLP_Transformer_LSTM_RNN_K79_K97_K113",
    )
    plot_metric(
        summary, output_root, args.dpi, args.show_std,
        "effective_mean_participation_rank",
        "effective_embedding_rank_dynamics_MLP_Transformer_LSTM_RNN_K79_K97_K113",
    )
    audit = {
        "validated_runs": 60,
        "architectures": ARCHITECTURES,
        "K_values": K_VALUES,
        "seeds": SEEDS,
        "metric_rows_per_run": len(expected_epochs()),
        "font": args.font_family,
        "font_path": font_path,
        "horizontal_grid_lines": False,
        "title": None,
        "caption_inside_figure": False,
        "x_scale": "log(epoch), epoch >= 1",
        "axis_tick_numbers": "shown on both axes in every panel",
        "axis_labels": "Epoch and Participation rank centered on every panel axis",
        "final_rank_2_invariant": True,
        "figures": ["classifier rank dynamics", "effective embedding rank dynamics"],
    }
    (output_root / "figure_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
