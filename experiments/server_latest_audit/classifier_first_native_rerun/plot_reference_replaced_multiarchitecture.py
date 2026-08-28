"""Build a unified 4x3 figure with the audited reference Transformer as row one."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROWS = ("reference_transformer", "transformer", "lstm", "rnn")
ROW_LABELS = {
    "reference_transformer": "Reference Transformer\n($d_{model}=256$)",
    "transformer": "Transformer\n($d_{model}=128$)",
    "lstm": "LSTM",
    "rnn": "RNN",
}
KS = (79, 97, 113)
CLASSIFIER_COLOR = "#D94B3D"
EMBEDDING_COLOR = "#2166AC"
INK = "#20252B"
FONT_SIZE = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-aggregate", type=Path, required=True)
    parser.add_argument("--multiarchitecture-aggregate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference(path: Path) -> dict[tuple[str, int], list[dict[str, float]]]:
    grouped: dict[tuple[str, int], list[dict[str, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            K = int(row["K"])
            grouped[("reference_transformer", K)].append(
                {
                    "epoch": int(row["epoch"]),
                    "classifier_mean": float(row["classifier_mean"]),
                    "classifier_sd": float(row["classifier_std"]),
                    "embedding_mean": float(row["embedding_mean"]),
                    "embedding_sd": float(row["embedding_std"]),
                }
            )
    return grouped


def load_multiarchitecture(path: Path) -> dict[tuple[str, int], list[dict[str, float]]]:
    grouped: dict[tuple[str, int], list[dict[str, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            architecture = row["architecture"]
            if architecture not in {"transformer", "lstm", "rnn"}:
                continue
            K = int(row["K"])
            grouped[(architecture, K)].append(
                {
                    "epoch": int(row["epoch"]),
                    "classifier_mean": float(row["classifier_mean"]),
                    "classifier_sd": float(row["classifier_sample_sd"]),
                    "embedding_mean": float(row["embedding_mean"]),
                    "embedding_sd": float(row["embedding_sample_sd"]),
                }
            )
    return grouped


def validate(grouped: dict[tuple[str, int], list[dict[str, float]]]) -> None:
    expected = {(architecture, K) for architecture in ROWS for K in KS}
    if set(grouped) != expected:
        raise ValueError(f"expected {sorted(expected)}; found {sorted(grouped)}")
    expected_epochs = list(range(0, 10001, 50))
    for key, rows in grouped.items():
        rows.sort(key=lambda row: int(row["epoch"]))
        epochs = [int(row["epoch"]) for row in rows]
        if epochs != expected_epochs:
            raise ValueError(f"checkpoint grid mismatch for {key}")
        for row in rows:
            values = [row[name] for name in ("classifier_mean", "classifier_sd", "embedding_mean", "embedding_sd")]
            if not all(np.isfinite(values)):
                raise ValueError(f"non-finite value for {key} at epoch {row['epoch']}")


def main() -> None:
    args = parse_args()
    reference_path = args.reference_aggregate.resolve()
    multi_path = args.multiarchitecture_aggregate.resolve()
    grouped = load_reference(reference_path)
    grouped.update(load_multiarchitecture(multi_path))
    validate(grouped)

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
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(4, 3, figsize=(20, 18), sharex=True, sharey=True)
    for row_index, architecture in enumerate(ROWS):
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
                axis.set_ylabel(f"{ROW_LABELS[architecture]}\nBest cyclic score", labelpad=10)
            if row_index == len(ROWS) - 1:
                axis.set_xlabel("Epoch", labelpad=8)

    figure.legend(
        handles=[
            Line2D([0], [0], color=CLASSIFIER_COLOR, linewidth=3, label="Classifier"),
            Line2D([0], [0], color=EMBEDDING_COLOR, linewidth=3, label="Embedding"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.012),
        handlelength=2.8,
        columnspacing=2.2,
    )
    figure.subplots_adjust(left=0.12, right=0.985, top=0.97, bottom=0.075, hspace=0.28, wspace=0.20)
    stem = output_dir / "classifier_first_ReferenceTransformer_Transformer_LSTM_RNN_K79_K97_K113"
    for suffix in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{suffix}")
        options = {"dpi": 300} if suffix == "png" else {}
        figure.savefig(path, bbox_inches="tight", facecolor="white", **options)
    plt.close(figure)

    manifest = {
        "rows": list(ROWS),
        "K_values": list(KS),
        "checkpoint_grid": "0,50,...,10000",
        "reference_source": str(reference_path),
        "reference_source_sha256": sha256(reference_path),
        "multiarchitecture_source": str(multi_path),
        "multiarchitecture_source_sha256": sha256(multi_path),
        "reference_row_is_audited_model": "one-layer causal Transformer, d_model=256",
        "format": {
            "font": "Arial",
            "font_size": FONT_SIZE,
            "global_title": False,
            "threshold_line": False,
            "grid": False,
            "embedded_caption": False,
            "bands": "mean ± sample SD across five seeds",
        },
    }
    (output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(stem)


if __name__ == "__main__":
    main()
