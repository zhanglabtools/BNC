from __future__ import annotations

import pandas as pd

from bnc_repro.plotting.alignment import plot_alignment


def test_alignment_plot_accepts_non_paper_modulus(tmp_path) -> None:
    rows = []
    for architecture in ("mlp", "transformer", "lstm", "rnn"):
        for epoch in (1, 2):
            rows.append(
                {
                    "architecture": architecture,
                    "K": 17,
                    "epoch": epoch,
                    "matched_mean": 0.1 * epoch,
                    "matched_std": 0.01,
                }
            )
    data = tmp_path / "summary.csv"
    pd.DataFrame(rows).to_csv(data, index=False)
    outputs = plot_alignment(data, tmp_path / "alignment", centered=False)
    assert {path.suffix for path in outputs} == {".png", ".pdf", ".svg"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
