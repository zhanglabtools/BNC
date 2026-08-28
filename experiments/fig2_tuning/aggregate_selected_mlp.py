"""Aggregate the frozen MLP configuration across K=79,97,113 and five seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


KS = (79, 97, 113)
SEEDS = (1, 2, 3, 4, 5)
THRESHOLD = 0.90
CONSECUTIVE = 2


def onset(rows: list[dict], field: str) -> int | None:
    for index in range(len(rows) - CONSECUTIVE + 1):
        if all(float(rows[j][field]) >= THRESHOLD for j in range(index, index + CONSECUTIVE)):
            return int(rows[index]["epoch"])
    return None


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values) if values else None,
        "sample_sd": statistics.stdev(values) if len(values) >= 2 else None,
        "median": statistics.median(values) if values else None,
        "n": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    by_run = {}
    raw_rows = []
    configs = []
    for K in KS:
        for seed in SEEDS:
            run_dir = run_root / f"K{K}" / f"seed_{seed}"
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            if status.get("status") != "complete" or int(status.get("epoch", -1)) != 10000:
                raise RuntimeError(f"incomplete run {run_dir}: {status}")
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            configs.append(config)
            with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 201 or [int(row["epoch"]) for row in rows] != list(range(0, 10001, 50)):
                raise RuntimeError(f"checkpoint grid mismatch: {run_dir}")
            parsed = []
            for row in rows:
                item = {
                    "K": K,
                    "seed": seed,
                    "epoch": int(row["epoch"]),
                    "classifier_score": float(row["classifier_best_cyclic_score"]),
                    "embedding_score": float(row["embedding_best_cyclic_score"]),
                    "train_loss": float(row["train_loss"]),
                    "train_accuracy": float(row["train_accuracy"]),
                    "test_accuracy": float(row["test_accuracy"]),
                }
                if not all(math.isfinite(item[key]) for key in ("classifier_score", "embedding_score", "train_loss", "train_accuracy", "test_accuracy")):
                    raise RuntimeError(f"non-finite metric: {run_dir}")
                parsed.append(item)
                raw_rows.append(item)
            by_run[(K, seed)] = parsed
    write_csv(output_dir / "raw_metrics.csv", raw_rows)

    aggregate_rows = []
    for K in KS:
        for index, epoch in enumerate(range(0, 10001, 50)):
            classifier = [by_run[(K, seed)][index]["classifier_score"] for seed in SEEDS]
            embedding = [by_run[(K, seed)][index]["embedding_score"] for seed in SEEDS]
            aggregate_rows.append(
                {
                    "K": K,
                    "epoch": epoch,
                    "classifier_mean": statistics.mean(classifier),
                    "classifier_sample_sd": statistics.stdev(classifier),
                    "embedding_mean": statistics.mean(embedding),
                    "embedding_sample_sd": statistics.stdev(embedding),
                    "n_seeds": 5,
                }
            )
    write_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)

    onset_rows = []
    summary = {"K_values": list(KS), "seeds": list(SEEDS), "threshold": THRESHOLD, "consecutive": CONSECUTIVE, "by_K": {}}
    for K in KS:
        group = []
        for seed in SEEDS:
            rows = by_run[(K, seed)]
            classifier = onset(rows, "classifier_score")
            embedding = onset(rows, "embedding_score")
            lead = embedding - classifier if classifier is not None and embedding is not None else None
            record = {
                "K": K,
                "seed": seed,
                "classifier_onset": classifier,
                "embedding_onset": embedding,
                "lead": lead,
                "classifier_first": lead is not None and lead > 0,
                "classifier_censored": classifier is None,
                "embedding_censored": embedding is None,
                "final_test_accuracy": rows[-1]["test_accuracy"],
            }
            onset_rows.append(record)
            group.append(record)
        leads = [row["lead"] for row in group if row["lead"] is not None]
        summary["by_K"][str(K)] = {
            "classifier_first_count": sum(row["classifier_first"] for row in group),
            "complete_onset_pairs": len(leads),
            "classifier_right_censored_count": sum(row["classifier_censored"] for row in group),
            "embedding_right_censored_count": sum(row["embedding_censored"] for row in group),
            "lead": stats(leads),
            "classifier_onset": stats([row["classifier_onset"] for row in group if row["classifier_onset"] is not None]),
            "embedding_onset": stats([row["embedding_onset"] for row in group if row["embedding_onset"] is not None]),
            "final_test_accuracy": stats([row["final_test_accuracy"] for row in group]),
            "onsets_by_seed": group,
        }
    write_csv(output_dir / "onset_by_run.csv", onset_rows)
    config_fields = ("model", "embedding_dim", "hidden_dim", "embedding_lr", "decoder_lr", "embedding_weight_decay", "decoder_weight_decay", "betas", "max_epoch", "checkpoint_every")
    reference = {key: configs[0][key] for key in config_fields}
    if any({key: config[key] for key in config_fields} != reference for config in configs[1:]):
        raise RuntimeError("frozen hyperparameters differ across final runs")
    summary["frozen_config"] = reference
    (output_dir / "aggregate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
