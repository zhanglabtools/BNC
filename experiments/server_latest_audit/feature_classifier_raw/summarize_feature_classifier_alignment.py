"""Aggregate five-seed feature-classifier trajectories by architecture and K."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ARCHITECTURES = ("mlp", "transformer", "lstm", "rnn")
FIELDS = {
    "epoch",
    "train_acc",
    "test_acc",
    "feature_classifier_alignment",
    "shuffled_feature_classifier_alignment",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--architectures", nargs="+", default=list(ARCHITECTURES))
    parser.add_argument("--Ks", type=int, nargs="+", default=[79, 97, 113])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    return parser.parse_args()


def read_metrics(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing = FIELDS.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")
    return [
        {
            "epoch": int(row["epoch"]),
            "train_acc": float(row["train_acc"]),
            "test_acc": float(row["test_acc"]),
            "feature_classifier_alignment": float(
                row["feature_classifier_alignment"]
            ),
            "shuffled_feature_classifier_alignment": float(
                row["shuffled_feature_classifier_alignment"]
            ),
        }
        for row in rows
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    runs_root = args.runs_root.resolve()
    artifacts_root = args.artifacts_root.resolve()
    artifacts_root.mkdir(parents=True, exist_ok=True)
    by_epoch: list[dict] = []
    final_rows: list[dict] = []

    for architecture in args.architectures:
        for K in args.Ks:
            runs = []
            summaries = []
            for seed in args.seeds:
                run_dir = runs_root / f"alignment_{architecture}_K{K}_seed{seed}"
                metrics_path = run_dir / "metrics.csv"
                summary_path = run_dir / "summary.json"
                if not metrics_path.exists() or not summary_path.exists():
                    raise FileNotFoundError(f"incomplete run: {run_dir}")
                rows = read_metrics(metrics_path)
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                expected = (architecture, K, seed)
                observed = (summary["architecture"], summary["K"], summary["seed"])
                if observed != expected:
                    raise ValueError(f"identity mismatch in {run_dir}: {observed} != {expected}")
                if rows[-1]["epoch"] != summary["final_epoch"]:
                    raise ValueError(f"metrics/summary epoch mismatch: {run_dir}")
                runs.append(rows)
                summaries.append(summary)

            epoch_grid = [row["epoch"] for row in runs[0]]
            if any([row["epoch"] for row in run] != epoch_grid for run in runs[1:]):
                raise ValueError(f"logged epoch grids differ for {architecture}, K={K}")

            for index, epoch in enumerate(epoch_grid):
                matched = np.asarray(
                    [run[index]["feature_classifier_alignment"] for run in runs],
                    dtype=float,
                )
                shuffled = np.asarray(
                    [
                        run[index]["shuffled_feature_classifier_alignment"]
                        for run in runs
                    ],
                    dtype=float,
                )
                by_epoch.append(
                    {
                        "architecture": architecture,
                        "K": K,
                        "epoch": epoch,
                        "matched_mean": float(matched.mean()),
                        "matched_std": float(matched.std(ddof=1)),
                        "shuffled_mean": float(shuffled.mean()),
                        "shuffled_std": float(shuffled.std(ddof=1)),
                        "n_seeds": len(runs),
                    }
                )

            def values(name: str) -> np.ndarray:
                return np.asarray([row[name] for row in summaries], dtype=float)

            final_matched = values("final_matched_alignment")
            final_shuffled = values("final_shuffled_alignment")
            final_gap = values("final_gap")
            final_test = values("final_test_acc")
            final_rows.append(
                {
                    "architecture": architecture,
                    "K": K,
                    "final_matched_mean": float(final_matched.mean()),
                    "final_matched_std": float(final_matched.std(ddof=1)),
                    "final_shuffled_mean": float(final_shuffled.mean()),
                    "final_shuffled_std": float(final_shuffled.std(ddof=1)),
                    "final_gap_mean": float(final_gap.mean()),
                    "final_gap_std": float(final_gap.std(ddof=1)),
                    "final_test_acc_mean": float(final_test.mean()),
                    "final_test_acc_std": float(final_test.std(ddof=1)),
                    "n_seeds": len(runs),
                }
            )

    write_csv(
        artifacts_root / "feature_classifier_summary_by_architecture_epoch.csv",
        by_epoch,
    )
    write_csv(
        artifacts_root / "final_feature_classifier_summary_by_architecture.csv",
        final_rows,
    )
    print(f"wrote {len(by_epoch)} epoch rows and {len(final_rows)} final rows")


if __name__ == "__main__":
    main()
