from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import configure_font, save_formats


def plot_fig2_mlp(
    data: str | Path,
    output_stem: str | Path,
    *,
    font_family: str = "Arial",
) -> list[Path]:
    configure_font(font_family)
    frame = pd.read_csv(data)
    if "architecture" in frame.columns:
        frame = frame[frame["architecture"] == "mlp"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    for axis, modulus in zip(axes, (79, 97, 113), strict=True):
        group = frame[frame["K"] == modulus].sort_values("epoch")
        if group.empty:
            raise ValueError(f"Figure 2 aggregate is missing K={modulus}")
        epoch = group["epoch"].to_numpy(float)
        for prefix, color, label in (
            ("classifier", "#ef3b2c", "Classifier"),
            ("embedding", "#2171b5", "Embedding"),
        ):
            mean = group[f"{prefix}_mean"].to_numpy(float)
            sd = group[f"{prefix}_sample_sd"].to_numpy(float)
            axis.plot(epoch, mean, color=color, lw=1.5, label=label)
            axis.fill_between(epoch, mean - sd, mean + sd, color=color, alpha=0.16, linewidth=0)
        axis.set_xscale("symlog", linthresh=50)
        axis.set_ylim(0, 1.02)
        axis.set_title(f"K = {modulus}")
        axis.set_xlabel("Epoch")
    axes[0].set_ylabel("Best cyclic score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    return save_formats(fig, output_stem)

