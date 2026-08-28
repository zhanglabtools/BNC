"""Plot four-panel feature-classifier alignment trajectories."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


ORDER = ("mlp", "transformer", "lstm", "rnn")
LABELS = {"mlp": "MLP", "transformer": "Transformer", "lstm": "LSTM", "rnn": "RNN"}
COLORS = {79: "#1769AA", 97: "#D95F02", 113: "#238B45"}
INK = "#20252B"
TEXT_SIZE = 21
PANEL_TITLE_SIZE = 27


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_summary(path: Path):
    grouped = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["architecture"], int(row["K"]))
            grouped[key].append(
                {
                    "epoch": int(row["epoch"]),
                    "matched_mean": float(row["matched_mean"]),
                    "matched_std": float(row["matched_std"]),
                    "shuffled_mean": float(row["shuffled_mean"]),
                    "shuffled_std": float(row["shuffled_std"]),
                }
            )
    for rows in grouped.values():
        rows.sort(key=lambda row: row["epoch"])
    expected = {(architecture, K) for architecture in ORDER for K in COLORS}
    if set(grouped) != expected:
        raise ValueError(f"expected {sorted(expected)}; found {sorted(grouped)}")
    return grouped


def arrays(rows):
    fields = ("epoch", "matched_mean", "matched_std", "shuffled_mean", "shuffled_std")
    return tuple(np.asarray([row[field] for row in rows], dtype=float) for field in fields)


def style_axis(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(INK)
    axis.spines[["left", "bottom"]].set_linewidth(1.1)
    axis.grid(False)
    axis.tick_params(axis="both", labelsize=TEXT_SIZE, colors=INK, width=1.0, length=5)


def format_epoch_tick(tick: int) -> str:
    if tick >= 1000 and tick % 1000 == 0:
        return f"{tick // 1000}k"
    return f"{tick:,}"


def main() -> None:
    args = parse_args()
    grouped = load_summary(args.summary_csv.resolve())
    max_epoch = int(max(row["epoch"] for rows in grouped.values() for row in rows))
    lower_envelope_min = min(
        row["matched_mean"] - row["matched_std"]
        for rows in grouped.values()
        for row in rows
    )
    # Leave visible breathing room below the lowest mean - s.d. envelope. Some
    # recurrent/Transformer runs exhibit genuine late negative excursions.
    y_min = min(-0.08, np.floor(lower_envelope_min * 10.0) / 10.0 - 0.02)
    y_ticks = np.arange(np.ceil(y_min / 0.2) * 0.2, 1.01, 0.2)
    decade_ticks = [
        10**power
        for power in range(int(np.floor(np.log10(max_epoch))) + 1)
        if 10**power <= max_epoch
    ]
    x_ticks = decade_ticks + ([] if max_epoch in decade_ticks else [max_epoch])
    x_tick_labels = [format_epoch_tick(tick) for tick in x_ticks]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": TEXT_SIZE,
            "axes.labelsize": TEXT_SIZE,
            "xtick.labelsize": TEXT_SIZE,
            "ytick.labelsize": TEXT_SIZE,
            "legend.fontsize": TEXT_SIZE,
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(17.8, 10.4), sharex=True, sharey=True)

    for axis, architecture in zip(axes.ravel(), ORDER):
        for K, color in COLORS.items():
            epoch, matched, matched_std, _, _ = arrays(grouped[(architecture, K)])
            axis.fill_between(
                epoch,
                matched - matched_std,
                matched + matched_std,
                color=color,
                alpha=0.13,
                linewidth=0,
            )
            axis.plot(epoch, matched, color=color, linewidth=2.7)
        axis.set_title(
            LABELS[architecture], fontsize=PANEL_TITLE_SIZE, fontweight="bold", pad=14
        )
        axis.set_xscale("log", base=10)
        axis.set_xlim(1, max_epoch)
        axis.set_xticks(x_ticks)
        axis.set_xticklabels(x_tick_labels)
        axis.set_ylim(y_min, 1.02)
        axis.set_yticks(y_ticks)
        axis.tick_params(axis="y", labelleft=True)
        axis.tick_params(axis="x", labelbottom=True)
        style_axis(axis)
        if max_epoch not in decade_ticks and max_epoch / decade_ticks[-1] < 3:
            # Keep both late labels readable at the same font size.
            tick_texts = axis.get_xticklabels()
            tick_texts[-2].set_horizontalalignment("right")
            tick_texts[-1].set_horizontalalignment("left")

    figure.supxlabel("Epoch", fontsize=TEXT_SIZE + 1, y=0.082)
    figure.supylabel(
        "Feature-classifier alignment", fontsize=TEXT_SIZE + 1, x=0.025
    )

    legend_handles = [
        Line2D([0], [0], color=COLORS[K], linewidth=2.8, label=f"K = {K}")
        for K in COLORS
    ]
    legend_handles.append(
        Patch(facecolor="#7B8794", alpha=0.18, edgecolor="none", label="Mean ± s.d.")
    )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        frameon=False,
        fontsize=TEXT_SIZE,
        handlelength=2.7,
        columnspacing=2.0,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        top=0.965,
        bottom=0.16,
        hspace=0.24,
        wspace=0.18,
    )
    stem = (
        "feature_classifier_alignment_MLP_Transformer_LSTM_RNN_"
        f"logx_densecheckpoints_{max_epoch // 1000}kepochs"
    )
    paths = [output_dir / f"{stem}.{suffix}" for suffix in ("png", "pdf", "svg")]
    for path in paths:
        options = {"dpi": 300} if path.suffix == ".png" else {}
        figure.savefig(path, bbox_inches="tight", facecolor="white", **options)
    plt.close(figure)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
