from __future__ import annotations

from pathlib import Path
from typing import Any


def execute_config(config: dict[str, Any], output_root: str | Path | None = None) -> list[Path]:
    protocol = config["protocol"]
    destination = Path(output_root or config.get("output_root", "outputs")).resolve()
    if protocol == "fig1_pca":
        from bnc_repro.protocols.fig1 import run_fig1_grid

        return run_fig1_grid(config, destination)
    if protocol == "dense":
        from bnc_repro.protocols.dense import run_dense_grid

        return run_dense_grid(config, destination)
    if protocol == "fig2_bcs":
        from bnc_repro.protocols.fig2 import run_fig2_grid

        return run_fig2_grid(config, destination)
    if protocol == "alignment":
        from bnc_repro.protocols.alignment import run_alignment_grid

        return run_alignment_grid(config, destination)
    if protocol == "rank_homotopy":
        from bnc_repro.protocols.rank_homotopy import run_rank_homotopy_grid

        return run_rank_homotopy_grid(config, destination)
    if protocol == "rank2_finetune":
        from bnc_repro.protocols.rank2_finetune import run_rank2_grid

        return run_rank2_grid(config, destination)
    raise ValueError(f"unsupported protocol: {protocol}")
