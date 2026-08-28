from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import configure_font, save_formats


def plot_fig1_coordinates(
    data: str | Path, output_stem: str | Path, *, font_family: str = "Arial"
) -> list[Path]:
    configure_font(font_family)
    payload = np.load(data)
    embedding = payload["embedding_coordinates"]
    classifier = payload["classifier_coordinates"]
    labels = payload["display_values"] if "display_values" in payload else np.arange(embedding.shape[0])
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), constrained_layout=True)
    scatter = None
    for axis, coordinates, title in zip(
        axes, (embedding, classifier), ("First-role embedding", "Classifier"), strict=True
    ):
        theta = np.linspace(0, 2 * np.pi, 500)
        axis.plot(np.cos(theta), np.sin(theta), color="black", lw=0.5, alpha=0.6)
        scatter = axis.scatter(
            coordinates[:, 0], coordinates[:, 1], c=labels, cmap="Blues", s=24,
            edgecolors="0.25", linewidths=0.5,
        )
        axis.set_aspect("equal")
        axis.set_xlim(-1.25, 1.25)
        axis.set_ylim(-1.25, 1.25)
        axis.set_xlabel("PCA1")
        axis.set_title(title)
    axes[0].set_ylabel("PCA2")
    if scatter is not None:
        fig.colorbar(scatter, ax=axes, label=f"Integer (mod {embedding.shape[0]})")
    return save_formats(fig, output_stem)

