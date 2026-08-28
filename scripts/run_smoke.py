from __future__ import annotations

import argparse
from pathlib import Path

from bnc_repro.aggregation.alignment import aggregate_alignment
from bnc_repro.config import load_config
from bnc_repro.plotting.alignment import plot_alignment
from bnc_repro.training.engine import execute_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the four-architecture CPU smoke test")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output or (root / "outputs" / "smoke")
    if not output.is_absolute():
        output = root / output
    execute_config(load_config(root / "configs" / "smoke" / "cpu_all_architectures.yaml"), output)
    runs = output / "smoke_alignment"
    aggregate_alignment(runs, output / "token_geometry_summary.csv", centered=False)
    aggregate_alignment(runs, output / "centered_feature_summary.csv", centered=True)
    plot_alignment(output / "token_geometry_summary.csv", output / "token_geometry", centered=False)
    plot_alignment(output / "centered_feature_summary.csv", output / "centered_feature", centered=True)
    print(f"CPU smoke completed: {output}")


if __name__ == "__main__":
    main()
