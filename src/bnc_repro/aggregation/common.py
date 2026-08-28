from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_run_metrics(runs_root: str | Path) -> pd.DataFrame:
    paths = sorted(Path(runs_root).rglob("metrics.csv"))
    if not paths:
        raise FileNotFoundError(f"no metrics.csv files found below {runs_root}")
    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def write_frame(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination

