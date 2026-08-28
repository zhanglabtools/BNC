from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import ARCHITECTURES, ARCHITECTURE_LABELS, K_COLORS, configure_font, save_formats


def plot_fig_s2(data: str | Path, output_stem: str | Path, *, font_family: str = "Arial") -> list[Path]:
    configure_font(font_family)
    frame = pd.read_csv(data)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    y_limits = {"mlp": (0, 25), "transformer": (0, 10), "lstm": (0, 15), "rnn": (0, 10)}
    for axis, architecture in zip(axes, ARCHITECTURES, strict=True):
        subset = frame[frame["architecture"] == architecture]
        for modulus in (79, 97, 113):
            group = subset[subset["K"] == modulus].sort_values("epoch")
            x = np.log1p(group["epoch"].to_numpy(float))
            mean = group["effective_mean_participation_rank_mean"].to_numpy(float)
            sd = group["effective_mean_participation_rank_std"].to_numpy(float)
            axis.plot(x, mean, color=K_COLORS[modulus], lw=1.5, label=f"K={modulus}")
            axis.fill_between(x, mean - sd, mean + sd, color=K_COLORS[modulus], alpha=0.13, linewidth=0)
        ticks = np.array([0, 10, 50, 200, 1000, 5000], dtype=float)
        axis.set_xticks(np.log1p(ticks), [str(int(value)) for value in ticks])
        axis.set_ylim(*y_limits[architecture])
        axis.set_title(ARCHITECTURE_LABELS[architecture])
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Effective dimension")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return save_formats(fig, output_stem)

