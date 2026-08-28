"""Aggregate four-architecture classifier-first Best-cyclic-score trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
from collections import defaultdict
from pathlib import Path


ARCHITECTURES = ("mlp", "transformer", "lstm", "rnn")
KS = (79, 97, 113)
SEEDS = (1, 2, 3, 4, 5)
RAW_FIELDS = (
    "architecture",
    "K",
    "seed",
    "epoch",
    "train_loss",
    "train_accuracy",
    "test_accuracy",
    "classifier_best_cyclic_score",
    "embedding_best_cyclic_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--consecutive", type=int, default=2)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "sample_sd": None, "variance": None, "median": None, "n": 0}
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) >= 2 else None,
        "variance": statistics.variance(values) if len(values) >= 2 else None,
        "median": statistics.median(values),
        "n": len(values),
    }


def onset(rows: list[dict], field: str, threshold: float, consecutive: int) -> int | None:
    values = [float(row[field]) for row in rows]
    epochs = [int(row["epoch"]) for row in rows]
    for end in range(consecutive - 1, len(values)):
        start = end - consecutive + 1
        if all(value >= threshold for value in values[start : end + 1]):
            return epochs[start]
    return None


def read_run(path: Path, architecture: str, K: int, seed: int) -> list[dict]:
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete" or int(status.get("epoch", -1)) != 10000:
        raise RuntimeError(f"incomplete run: {path}: {status}")
    with (path / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 201:
        raise RuntimeError(f"expected 201 checkpoints in {path}, found {len(rows)}")
    parsed = []
    for row in rows:
        item = {
            "architecture": row["architecture"],
            "K": int(row["K"]),
            "seed": int(row["seed"]),
            "epoch": int(row["epoch"]),
            "train_loss": float(row["train_loss"]),
            "train_accuracy": float(row["train_accuracy"]),
            "test_accuracy": float(row["test_accuracy"]),
            "classifier_best_cyclic_score": float(row["classifier_best_cyclic_score"]),
            "embedding_best_cyclic_score": float(row["embedding_best_cyclic_score"]),
        }
        if (item["architecture"], item["K"], item["seed"]) != (architecture, K, seed):
            raise RuntimeError(f"identity mismatch in {path}")
        if not all(math.isfinite(item[field]) for field in RAW_FIELDS[4:]):
            raise RuntimeError(f"non-finite metric in {path}")
        parsed.append(item)
    if [row["epoch"] for row in parsed] != list(range(0, 10001, 50)):
        raise RuntimeError(f"checkpoint grid mismatch in {path}")
    return parsed


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    by_run: dict[tuple[str, int, int], list[dict]] = {}
    for architecture in ARCHITECTURES:
        for K in KS:
            for seed in SEEDS:
                run_dir = output_dir / architecture / f"K{K}" / f"seed_{seed}"
                rows = read_run(run_dir, architecture, K, seed)
                by_run[(architecture, K, seed)] = rows
                all_rows.extend(rows)

    write_csv(artifacts_dir / "raw_metrics.csv", all_rows)

    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in all_rows:
        grouped[(row["architecture"], row["K"], row["epoch"])].append(row)
    aggregate_rows = []
    for architecture in ARCHITECTURES:
        for K in KS:
            for epoch in range(0, 10001, 50):
                rows = grouped[(architecture, K, epoch)]
                classifier = [row["classifier_best_cyclic_score"] for row in rows]
                embedding = [row["embedding_best_cyclic_score"] for row in rows]
                accuracy = [row["test_accuracy"] for row in rows]
                aggregate_rows.append(
                    {
                        "architecture": architecture,
                        "K": K,
                        "epoch": epoch,
                        "classifier_mean": statistics.mean(classifier),
                        "classifier_sample_sd": statistics.stdev(classifier),
                        "classifier_variance": statistics.variance(classifier),
                        "embedding_mean": statistics.mean(embedding),
                        "embedding_sample_sd": statistics.stdev(embedding),
                        "embedding_variance": statistics.variance(embedding),
                        "test_accuracy_mean": statistics.mean(accuracy),
                        "test_accuracy_sample_sd": statistics.stdev(accuracy),
                        "n_seeds": len(rows),
                    }
                )
    write_csv(artifacts_dir / "aggregate_metrics.csv", aggregate_rows)

    onset_rows = []
    summary = {
        "experiment": "classifier-first Best cyclic score across supplied architectures",
        "architectures": list(ARCHITECTURES),
        "K_values": list(KS),
        "seeds": list(SEEDS),
        "threshold": args.threshold,
        "consecutive_checkpoints": args.consecutive,
        "checkpoint_every": 50,
        "groups": {},
    }
    final_rows = []
    for architecture in ARCHITECTURES:
        summary["groups"][architecture] = {}
        for K in KS:
            group_onsets = []
            for seed in SEEDS:
                rows = by_run[(architecture, K, seed)]
                classifier_onset = onset(
                    rows, "classifier_best_cyclic_score", args.threshold, args.consecutive
                )
                embedding_onset = onset(
                    rows, "embedding_best_cyclic_score", args.threshold, args.consecutive
                )
                lead = (
                    embedding_onset - classifier_onset
                    if classifier_onset is not None and embedding_onset is not None
                    else None
                )
                group_onsets.append(
                    {
                        "architecture": architecture,
                        "K": K,
                        "seed": seed,
                        "classifier_onset": classifier_onset,
                        "embedding_onset": embedding_onset,
                        "lead": lead,
                        "classifier_first": lead is not None and lead > 0,
                        "classifier_right_censored": classifier_onset is None,
                        "embedding_right_censored": embedding_onset is None,
                    }
                )
            onset_rows.extend(group_onsets)
            classifier_values = [
                row["classifier_onset"]
                for row in group_onsets
                if row["classifier_onset"] is not None
            ]
            embedding_values = [
                row["embedding_onset"]
                for row in group_onsets
                if row["embedding_onset"] is not None
            ]
            lead_values = [row["lead"] for row in group_onsets if row["lead"] is not None]
            final_accuracy = [
                by_run[(architecture, K, seed)][-1]["test_accuracy"] for seed in SEEDS
            ]
            final_classifier = [
                by_run[(architecture, K, seed)][-1]["classifier_best_cyclic_score"]
                for seed in SEEDS
            ]
            final_embedding = [
                by_run[(architecture, K, seed)][-1]["embedding_best_cyclic_score"]
                for seed in SEEDS
            ]
            group = {
                "classifier_onset": sample_stats(classifier_values),
                "embedding_onset": sample_stats(embedding_values),
                "lead": sample_stats(lead_values),
                "classifier_first_count": sum(row["classifier_first"] for row in group_onsets),
                "complete_onset_pairs": len(lead_values),
                "classifier_right_censored_count": sum(
                    row["classifier_right_censored"] for row in group_onsets
                ),
                "embedding_right_censored_count": sum(
                    row["embedding_right_censored"] for row in group_onsets
                ),
                "final_test_accuracy": sample_stats(final_accuracy),
                "final_classifier_score": sample_stats(final_classifier),
                "final_embedding_score": sample_stats(final_embedding),
            }
            summary["groups"][architecture][str(K)] = group
            final_rows.append(
                {
                    "architecture": architecture,
                    "K": K,
                    "classifier_first_count": group["classifier_first_count"],
                    "complete_onset_pairs": group["complete_onset_pairs"],
                    "classifier_right_censored": group["classifier_right_censored_count"],
                    "embedding_right_censored": group["embedding_right_censored_count"],
                    "classifier_onset_mean": group["classifier_onset"]["mean"],
                    "embedding_onset_mean": group["embedding_onset"]["mean"],
                    "lead_mean": group["lead"]["mean"],
                    "lead_sample_sd": group["lead"]["sample_sd"],
                    "lead_variance": group["lead"]["variance"],
                    "final_test_accuracy_mean": group["final_test_accuracy"]["mean"],
                }
            )

    write_csv(artifacts_dir / "onset_by_run.csv", onset_rows)
    write_csv(artifacts_dir / "final_summary_by_architecture_K.csv", final_rows)
    (artifacts_dir / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "completed_runs": len(by_run),
        "raw_metric_rows": len(all_rows),
        "aggregate_rows": len(aggregate_rows),
    }
    (artifacts_dir / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    print(
        f"aggregated {len(by_run)} runs, {len(all_rows)} raw rows, "
        f"{len(aggregate_rows)} aggregate rows"
    )


if __name__ == "__main__":
    main()
