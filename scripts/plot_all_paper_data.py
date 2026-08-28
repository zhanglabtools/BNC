from __future__ import annotations

import json
from pathlib import Path

from bnc_repro.plotting.alignment import plot_alignment
from bnc_repro.plotting.fig2 import plot_fig2_mlp
from bnc_repro.plotting.fig_s1 import plot_fig_s1
from bnc_repro.plotting.fig_s2 import plot_fig_s2


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = root / "paper_data"
    figures = root / "figures" / "paper_data"
    statuses = {
        "fig1": {
            "status": "blocked",
            "reason": "bundled ZIP has a PNG and metrics but no coordinate data/checkpoint",
        },
        "fig2_full": {
            "status": "blocked",
            "reason": "other-architecture aggregate CSV was not supplied",
        },
    }
    jobs = {
        "fig2_mlp": plot_fig2_mlp(data / "fig2" / "mlp_aggregate_metrics.csv", figures / "fig2_mlp"),
        "fig_s1": plot_fig_s1(data / "fig_s1" / "mean_std_trajectory.csv", figures / "fig_s1"),
        "fig_s2": plot_fig_s2(data / "fig_s2" / "mean_std_trajectory.csv", figures / "fig_s2"),
        "fig_s3": plot_alignment(data / "fig_s3" / "alignment_summary.csv", figures / "fig_s3", centered=False),
        "fig_s4": plot_alignment(data / "fig_s4" / "centered_alignment_summary.csv", figures / "fig_s4", centered=True),
    }
    for name, paths in jobs.items():
        statuses[name] = {"status": "created", "files": [str(path.relative_to(root)) for path in paths]}
    figures.mkdir(parents=True, exist_ok=True)
    report = figures / "plot_status.json"
    report.write_text(json.dumps(statuses, indent=2), encoding="utf-8")
    print(json.dumps(statuses, indent=2))


if __name__ == "__main__":
    main()

