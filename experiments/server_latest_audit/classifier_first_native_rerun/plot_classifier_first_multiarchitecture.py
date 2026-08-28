"""Plot one 4x3 classifier-first Best-cyclic-score figure."""

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


ARCHITECTURES = ("mlp", "transformer", "lstm", "rnn")
LABELS = {"mlp": "MLP", "transformer": "Transformer", "lstm": "LSTM", "rnn": "RNN"}
KS = (79, 97, 113)
CLASSIFIER_COLOR = "#D94B3D"
EMBEDDING_COLOR = "#2166AC"
INK = "#20252B"
FONT_SIZE = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[tuple[str, int], list[dict]]:
    grouped = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[(row["architecture"], int(row["K"]))].append(
                {
                    "epoch": int(row["epoch"]),
                    "classifier_mean": float(row["classifier_mean"]),
                    "classifier_sd": float(row["classifier_sample_sd"]),
                    "embedding_mean": float(row["embedding_mean"]),
                    "embedding_sd": float(row["embedding_sample_sd"]),
                }
            )
    for rows in grouped.values():
        rows.sort(key=lambda row: row["epoch"])
    expected = {(architecture, K) for architecture in ARCHITECTURES for K in KS}
    if set(grouped) != expected:
        raise ValueError(f"expected {sorted(expected)}; found {sorted(grouped)}")
    return grouped


def main() -> None:
    args = parse_args()
    grouped = load(args.aggregate_csv.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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
    figure, axes = plt.subplots(
        len(ARCHITECTURES),
        len(KS),
        figsize=(20, 18),
        sharex=True,
        sharey=True,
    )
    for row_index, architecture in enumerate(ARCHITECTURES):
        for column_index, K in enumerate(KS):
            axis = axes[row_index, column_index]
            rows = grouped[(architecture, K)]
            epoch = np.asarray([row["epoch"] for row in rows], dtype=float)
            classifier = np.asarray([row["classifier_mean"] for row in rows])
            classifier_sd = np.asarray([row["classifier_sd"] for row in rows])
            embedding = np.asarray([row["embedding_mean"] for row in rows])
            embedding_sd = np.asarray([row["embedding_sd"] for row in rows])
            axis.fill_between(
                epoch,
                np.clip(classifier - classifier_sd, 0.0, 1.0),
                np.clip(classifier + classifier_sd, 0.0, 1.0),
                color=CLASSIFIER_COLOR,
                alpha=0.16,
                linewidth=0,
            )
            axis.fill_between(
                epoch,
                np.clip(embedding - embedding_sd, 0.0, 1.0),
                np.clip(embedding + embedding_sd, 0.0, 1.0),
                color=EMBEDDING_COLOR,
                alpha=0.16,
                linewidth=0,
            )
            axis.plot(epoch, classifier, color=CLASSIFIER_COLOR, linewidth=2.6)
            axis.plot(epoch, embedding, color=EMBEDDING_COLOR, linewidth=2.6)
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
            if row_index == 0:
                axis.set_title(f"K = {K}", pad=10, fontweight="bold")
            if column_index == 0:
                axis.set_ylabel(
                    f"{LABELS[architecture]}\nBest cyclic score", labelpad=10
                )
            if row_index == len(ARCHITECTURES) - 1:
                axis.set_xlabel("Epoch", labelpad=8)

    legend = [
        Line2D([0], [0], color=CLASSIFIER_COLOR, linewidth=3, label="Classifier"),
        Line2D([0], [0], color=EMBEDDING_COLOR, linewidth=3, label="Embedding"),
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.012),
        handlelength=2.8,
        columnspacing=2.2,
    )
    figure.subplots_adjust(
        left=0.10,
        right=0.985,
        top=0.97,
        bottom=0.075,
        hspace=0.28,
        wspace=0.20,
    )
    stem = "classifier_first_MLP_Transformer_LSTM_RNN_K79_K97_K113_native_corrected"
    paths = [output_dir / f"{stem}.{suffix}" for suffix in ("png", "pdf", "svg")]
    for path in paths:
        options = {"dpi": 300} if path.suffix == ".png" else {}
        figure.savefig(path, bbox_inches="tight", facecolor="white", **options)
    plt.close(figure)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
