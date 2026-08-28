from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


_FONT_WARNING_EMITTED = False


def configure_font(requested: str = "Arial") -> str:
    global _FONT_WARNING_EMITTED
    names = {font.name for font in fm.fontManager.ttflist}
    if requested in names:
        selected = requested
    else:
        selected = "DejaVu Sans"
        if not _FONT_WARNING_EMITTED:
            warnings.warn(
                f"font {requested!r} is unavailable; falling back to {selected!r}",
                stacklevel=2,
            )
            _FONT_WARNING_EMITTED = True
    plt.rcParams.update({"font.family": selected, "axes.unicode_minus": False})
    return selected


def save_formats(fig, output_stem: str | Path, dpi: int = 300) -> list[Path]:
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in (".png", ".pdf", ".svg"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


ARCHITECTURES = ("mlp", "transformer", "lstm", "rnn")
ARCHITECTURE_LABELS = {
    "mlp": "MLP",
    "transformer": "Transformer",
    "lstm": "LSTM",
    "rnn": "RNN",
}
K_COLORS = {79: "#1565c0", 97: "#ef6c00", 113: "#00897b"}

