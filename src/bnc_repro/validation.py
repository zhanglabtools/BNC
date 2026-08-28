from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from bnc_repro.protocols.common import geometric_checkpoint_epochs
from bnc_repro.protocols.rank2_finetune import should_record_s2
from bnc_repro.protocols.rank_homotopy import should_record_s1


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, compression="infer")


def _schedule_count(predicate, total: int) -> int:
    return sum(predicate(epoch, total) for epoch in range(total + 1))


def validate_figure(figure: str, paper_data_root: str | Path) -> dict[str, Any]:
    root = Path(paper_data_root)
    if figure == "fig1":
        metrics = root / "fig1" / "bundled_reference_metrics.json"
        reference = root / "fig1" / "bundled_reference.png"
        if not metrics.exists() or not reference.exists():
            raise FileNotFoundError("Figure 1 bundled reference artifacts are incomplete")
        return {
            "figure": figure,
            "status": "reference-artifact-only",
            "plot_only": "blocked-by-missing-coordinate-data-or-checkpoint",
        }
    if figure == "fig2":
        raw = _read(root / "fig2" / "mlp_raw_metrics.csv")
        aggregate = _read(root / "fig2" / "mlp_aggregate_metrics.csv")
        if len(raw) != 3015 or len(aggregate) != 603:
            raise ValueError(f"Figure 2 row mismatch: raw={len(raw)} aggregate={len(aggregate)}")
        if set(raw.groupby(["K", "seed"]).size()) != {201}:
            raise ValueError("Figure 2 must have 201 checkpoints per MLP run")
        return {
            "figure": figure,
            "status": "mlp-data-validated",
            "raw_rows": len(raw),
            "aggregate_rows": len(aggregate),
            "four_architecture_plot": "blocked-by-missing-other-aggregate",
        }
    if figure == "fig_s1":
        raw = _read(root / "fig_s1" / "all_runs_metrics.csv.gz")
        summary = _read(root / "fig_s1" / "mean_std_trajectory.csv")
        if len(raw) != 39520 or len(summary) != 7904:
            raise ValueError(f"S1 row mismatch: raw={len(raw)} summary={len(summary)}")
        if _schedule_count(should_record_s1, 10000) != 247:
            raise AssertionError("S1 schedule implementation does not produce 247 checkpoints")
        return {"figure": figure, "status": "validated", "raw_rows": len(raw), "summary_rows": len(summary)}
    if figure == "fig_s2":
        raw = _read(root / "fig_s2" / "all_runs_metrics.csv.gz")
        summary = _read(root / "fig_s2" / "mean_std_trajectory.csv")
        if len(raw) != 33660 or len(summary) != 6732:
            raise ValueError(f"S2 row mismatch: raw={len(raw)} summary={len(summary)}")
        if _schedule_count(should_record_s2, 6000) != 561:
            raise AssertionError("S2 schedule implementation does not produce 561 checkpoints")
        return {"figure": figure, "status": "validated", "raw_rows": len(raw), "summary_rows": len(summary)}
    if figure == "fig_s3":
        summary = _read(root / "fig_s3" / "alignment_summary.csv")
        if len(summary) != 2880:
            raise ValueError(f"S3 summary must contain 2880 rows, got {len(summary)}")
        if len(geometric_checkpoint_epochs(20000, 321)) != 240:
            raise AssertionError("S3/S4 geometric schedule must contain 240 epochs")
        if set(summary.groupby(["architecture", "K"]).size()) != {240}:
            raise ValueError("S3 must have 240 rows for every architecture/K group")
        return {"figure": figure, "status": "validated", "summary_rows": len(summary)}
    if figure == "fig_s4":
        raw = _read(root / "fig_s4" / "all_runs_metrics.csv.gz")
        summary = _read(root / "fig_s4" / "centered_alignment_summary.csv")
        if len(raw) != 14400 or len(summary) != 2880:
            raise ValueError(f"S4 row mismatch: raw={len(raw)} summary={len(summary)}")
        groups = raw.groupby(["architecture", "K", "seed"])
        if len(groups) != 60 or set(groups.size()) != {240}:
            raise ValueError("S4 must contain 60 runs with 240 common epochs each")
        return {"figure": figure, "status": "validated", "raw_rows": len(raw), "summary_rows": len(summary)}
    raise ValueError(f"unknown figure: {figure}")


def validate_all(paper_data_root: str | Path) -> list[dict[str, Any]]:
    return [validate_figure(name, paper_data_root) for name in ("fig1", "fig2", "fig_s1", "fig_s2", "fig_s3", "fig_s4")]

