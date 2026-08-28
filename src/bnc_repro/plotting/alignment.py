from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import ARCHITECTURES, ARCHITECTURE_LABELS, K_COLORS, configure_font, save_formats


def plot_alignment(
    data: str | Path,
    output_stem: str | Path,
    *,
    centered: bool,
    font_family: str = "Arial",
) -> list[Path]:
    configure_font(font_family)
    frame = pd.read_csv(data)
    mean_column = "centered_matched_mean" if centered else "matched_mean"
    std_column = "centered_matched_std" if centered else "matched_std"
    ylabel = "Centered feature-classifier alignment" if centered else "Embedding-classifier alignment"
    moduli = sorted(int(value) for value in frame["K"].dropna().unique())
    fallback_colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for axis, architecture in zip(axes.flat, ARCHITECTURES, strict=True):
        subset = frame[frame["architecture"] == architecture]
        for color_index, modulus in enumerate(moduli):
            group = subset[subset["K"] == modulus].sort_values("epoch")
            if group.empty:
                continue
            epoch = group["epoch"].to_numpy(float)
            mean = group[mean_column].to_numpy(float)
            sd = group[std_column].to_numpy(float)
            color = K_COLORS.get(modulus, fallback_colors(color_index % 10))
            axis.plot(epoch, mean, color=color, lw=1.5, label=f"K = {modulus}")
            axis.fill_between(epoch, mean - sd, mean + sd, color=color, alpha=0.13, linewidth=0)
        if (subset["epoch"] > 0).any():
            axis.set_xscale("log")
        axis.set_title(ARCHITECTURE_LABELS[architecture])
        axis.set_xlabel("Epoch")
    for axis in axes[:, 0]:
        axis.set_ylabel(ylabel)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return save_formats(fig, output_stem)
