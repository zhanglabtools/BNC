from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import ARCHITECTURES, ARCHITECTURE_LABELS, configure_font, save_formats


def plot_fig_s1(data: str | Path, output_stem: str | Path, *, font_family: str = "Arial") -> list[Path]:
    configure_font(font_family)
    frame = pd.read_csv(data)
    lambdas = sorted(frame["lambda_reg"].unique())
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.9, len(lambdas)))
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    for axis, architecture in zip(axes.flat, ARCHITECTURES, strict=True):
        subset = frame[frame["architecture"] == architecture]
        for color, coefficient in zip(colors, lambdas, strict=True):
            group = subset[subset["lambda_reg"] == coefficient].sort_values("epoch")
            x = group["epoch"].to_numpy(float)
            mean = group["classifier_participation_rank_mean"].to_numpy(float)
            sd = group["classifier_participation_rank_std"].to_numpy(float)
            axis.plot(x, mean, color=color, lw=1.3, label=f"λ={coefficient:g}")
            axis.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.10, linewidth=0)
        axis.set_xscale("log")
        axis.set_title(ARCHITECTURE_LABELS[architecture])
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Participation rank")
        if architecture in {"transformer", "rnn"}:
            axis.set_ylim(top=10)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=8, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_formats(fig, output_stem)

