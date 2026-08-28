"""Plot the frozen tuned MLP and the complete MLP/Transformer/LSTM/RNN figure."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


KS = (79, 97, 113)
ROWS = ("mlp", "transformer", "lstm", "rnn")
LABELS = {"mlp": "MLP", "transformer": "Transformer", "lstm": "LSTM", "rnn": "RNN"}
RED = "#D94B3D"
BLUE = "#2166AC"
INK = "#20252B"
FONT_SIZE = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp-aggregate", type=Path, required=True)
    parser.add_argument("--other-aggregate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_mlp(path: Path) -> dict[tuple[str, int], list[dict]]:
    grouped = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[("mlp", int(row["K"]))].append(
                {
                    "epoch": int(row["epoch"]),
                    "classifier_mean": float(row["classifier_mean"]),
                    "classifier_sd": float(row["classifier_sample_sd"]),
                    "embedding_mean": float(row["embedding_mean"]),
                    "embedding_sd": float(row["embedding_sample_sd"]),
                }
            )
    return grouped


def load_other(path: Path) -> dict[tuple[str, int], list[dict]]:
    grouped = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            architecture = row["architecture"]
            if architecture not in {"transformer", "lstm", "rnn"}:
                continue
            grouped[(architecture, int(row["K"]))].append(
                {
                    "epoch": int(row["epoch"]),
                    "classifier_mean": float(row["classifier_mean"]),
                    "classifier_sd": float(row["classifier_sample_sd"]),
                    "embedding_mean": float(row["embedding_mean"]),
                    "embedding_sd": float(row["embedding_sample_sd"]),
                }
            )
    return grouped


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
        }
    )


def draw_panel(axis, rows: list[dict]) -> None:
    epoch = np.asarray([row["epoch"] for row in rows], dtype=float)
    classifier = np.asarray([row["classifier_mean"] for row in rows])
    classifier_sd = np.asarray([row["classifier_sd"] for row in rows])
    embedding = np.asarray([row["embedding_mean"] for row in rows])
    embedding_sd = np.asarray([row["embedding_sd"] for row in rows])
    axis.fill_between(epoch, np.clip(classifier - classifier_sd, 0, 1), np.clip(classifier + classifier_sd, 0, 1), color=RED, alpha=0.16, linewidth=0)
    axis.fill_between(epoch, np.clip(embedding - embedding_sd, 0, 1), np.clip(embedding + embedding_sd, 0, 1), color=BLUE, alpha=0.16, linewidth=0)
    axis.plot(epoch, classifier, color=RED, linewidth=2.6)
    axis.plot(epoch, embedding, color=BLUE, linewidth=2.6)
    axis.set_xscale("symlog", linthresh=100)
    axis.set_xlim(0, 10000)
    axis.set_ylim(0.15, 1.02)
    axis.set_xticks([0, 100, 300, 1000, 3000, 10000])
    axis.set_xticklabels(["0", "100", "300", "1k", "3k", "10k"])
    axis.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    axis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#AAB1B8")
    axis.tick_params(axis="both", colors=INK, labelbottom=True, labelleft=True)


def legend(figure) -> None:
    figure.legend(
        handles=[Line2D([0], [0], color=RED, linewidth=3, label="Classifier"), Line2D([0], [0], color=BLUE, linewidth=3, label="Embedding")],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.012),
        handlelength=2.8,
        columnspacing=2.2,
    )


def save(figure, output_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        path = output_dir / f"{stem}.{suffix}"
        options = {"dpi": 300} if suffix == "png" else {}
        figure.savefig(path, bbox_inches="tight", facecolor="white", **options)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    grouped = load_mlp(args.mlp_aggregate.resolve())
    grouped.update(load_other(args.other_aggregate.resolve()))
    expected = {(architecture, K) for architecture in ROWS for K in KS}
    if set(grouped) != expected:
        raise ValueError(f"expected {sorted(expected)}; found {sorted(grouped)}")
    for rows in grouped.values():
        rows.sort(key=lambda row: row["epoch"])
        if [row["epoch"] for row in rows] != list(range(0, 10001, 50)):
            raise ValueError("checkpoint grid mismatch")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    mlp_figure, mlp_axes = plt.subplots(1, 3, figsize=(20, 5.4), sharex=True, sharey=True)
    for column, K in enumerate(KS):
        draw_panel(mlp_axes[column], grouped[("mlp", K)])
        mlp_axes[column].set_title(f"K = {K}", pad=10, fontweight="bold")
        mlp_axes[column].set_xlabel("Epoch", labelpad=8)
    mlp_axes[0].set_ylabel("Best cyclic score", labelpad=10)
    legend(mlp_figure)
    mlp_figure.subplots_adjust(left=0.075, right=0.985, top=0.93, bottom=0.22, wspace=0.20)
    save(mlp_figure, output_dir, "classifier_first_tuned_MLP_K79_K97_K113")

    figure, axes = plt.subplots(4, 3, figsize=(20, 18), sharex=True, sharey=True)
    for row_index, architecture in enumerate(ROWS):
        for column_index, K in enumerate(KS):
            axis = axes[row_index, column_index]
            draw_panel(axis, grouped[(architecture, K)])
            if row_index == 0:
                axis.set_title(f"K = {K}", pad=10, fontweight="bold")
            if column_index == 0:
                axis.set_ylabel(f"{LABELS[architecture]}\nBest cyclic score", labelpad=10)
            if row_index == len(ROWS) - 1:
                axis.set_xlabel("Epoch", labelpad=8)
    legend(figure)
    figure.subplots_adjust(left=0.10, right=0.985, top=0.97, bottom=0.075, hspace=0.28, wspace=0.20)
    save(figure, output_dir, "classifier_first_tuned_MLP_Transformer_LSTM_RNN_K79_K97_K113")


if __name__ == "__main__":
    main()
