"""Leakage-controlled hyperparameter search for MLP classifier-first dynamics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


EPOCHS = 10_000
LOG_EVERY = 50
THRESHOLD = 0.90
CONSECUTIVE = 2
DISCOVERY_K = 97
DISCOVERY_SEEDS = (1, 2, 3)
REFERENCE_TARGET = {
    "classifier_onset_mean": 200.0,
    "embedding_onset_mean": 3180.0,
    "lead_mean": 2980.0,
}
CANDIDATES = (
    {"name": "e2e-4_d2e-3_wd0p8", "embedding_lr": 2e-4, "decoder_lr": 2e-3, "decoder_wd": 0.8},
    {"name": "e1e-4_d2e-3_wd0p8", "embedding_lr": 1e-4, "decoder_lr": 2e-3, "decoder_wd": 0.8},
    {"name": "e5e-5_d2e-3_wd0p8", "embedding_lr": 5e-5, "decoder_lr": 2e-3, "decoder_wd": 0.8},
    {"name": "e2e-5_d2e-3_wd0p8", "embedding_lr": 2e-5, "decoder_lr": 2e-3, "decoder_wd": 0.8},
    {"name": "e1e-4_d4e-3_wd0p8", "embedding_lr": 1e-4, "decoder_lr": 4e-3, "decoder_wd": 0.8},
    {"name": "e5e-5_d4e-3_wd0p8", "embedding_lr": 5e-5, "decoder_lr": 4e-3, "decoder_wd": 0.8},
    {"name": "e1e-4_d2e-3_wd1p2", "embedding_lr": 1e-4, "decoder_lr": 2e-3, "decoder_wd": 1.2},
    {"name": "e5e-5_d2e-3_wd1p2", "embedding_lr": 5e-5, "decoder_lr": 2e-3, "decoder_wd": 1.2},
    {"name": "e5e-5_d4e-3_wd1p2", "embedding_lr": 5e-5, "decoder_lr": 4e-3, "decoder_wd": 1.2},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("launch-discovery", "summarize-discovery"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--max-parallel", type=int, default=6)
    return parser.parse_args()


def onset(rows: list[dict[str, str]], field: str) -> int | None:
    values = [float(row[field]) for row in rows]
    epochs = [int(row["epoch"]) for row in rows]
    for index in range(len(rows) - CONSECUTIVE + 1):
        if all(value >= THRESHOLD for value in values[index : index + CONSECUTIVE]):
            return epochs[index]
    return None


def worker_command(root: Path, candidate: dict, seed: int) -> tuple[list[str], Path]:
    output_dir = root / "discovery" / candidate["name"] / "outputs"
    log_path = root / "discovery" / candidate["name"] / "logs" / f"K{DISCOVERY_K}_seed{seed}.log"
    command = [
        sys.executable,
        str((root / "run_classifier_first_multiarchitecture.py").resolve()),
        "--worker-architecture", "mlp",
        "--worker-k", str(DISCOVERY_K),
        "--worker-seed", str(seed),
        "--output-dir", str(output_dir.resolve()),
        "--device", "cuda:0",
        "--frac-train", "0.7",
        "--epochs", str(EPOCHS),
        "--log-every", str(LOG_EVERY),
        "--embedding-lr", str(candidate["embedding_lr"]),
        "--decoder-lr", str(candidate["decoder_lr"]),
        "--embedding-weight-decay", "0.0",
        "--decoder-weight-decay", str(candidate["decoder_wd"]),
        "--beta2", "0.98",
        "--resume",
    ]
    if "embedding_dim" in candidate:
        command.extend(["--mlp-embedding-dim", str(candidate["embedding_dim"])])
    if "hidden_dim" in candidate:
        command.extend(["--mlp-hidden-dim", str(candidate["hidden_dim"])])
    return command, log_path


def complete(root: Path, candidate: dict, seed: int) -> bool:
    status_path = root / "discovery" / candidate["name"] / "outputs" / "mlp" / f"K{DISCOVERY_K}" / f"seed_{seed}" / "status.json"
    if not status_path.exists():
        return False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    return status.get("status") == "complete" and int(status.get("epoch", -1)) == EPOCHS


def launch_discovery(root: Path, gpu_ids: list[int], max_parallel: int) -> None:
    tasks = [(candidate, seed) for candidate in CANDIDATES for seed in DISCOVERY_SEEDS if not complete(root, candidate, seed)]
    active: list[tuple[subprocess.Popen, object, dict, int, int]] = []
    failures: list[tuple[str, int, int]] = []
    launched = 0
    while tasks or active:
        while tasks and len(active) < max_parallel:
            candidate, seed = tasks.pop(0)
            gpu = gpu_ids[launched % len(gpu_ids)]
            launched += 1
            command, log_path = worker_command(root, candidate, seed)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=environment)
            active.append((process, handle, candidate, seed, gpu))
            print(f"launched {candidate['name']} seed={seed} gpu={gpu}", flush=True)
        retained = []
        for process, handle, candidate, seed, gpu in active:
            code = process.poll()
            if code is None:
                retained.append((process, handle, candidate, seed, gpu))
                continue
            handle.close()
            print(f"finished {candidate['name']} seed={seed} gpu={gpu} exit={code}", flush=True)
            if code != 0:
                failures.append((candidate["name"], seed, code))
        active = retained
        if active:
            time.sleep(1)
    if failures:
        raise SystemExit(f"failed discovery runs: {failures}")


def summarize_discovery(root: Path) -> None:
    per_run = []
    per_candidate = []
    for candidate in CANDIDATES:
        rows_for_candidate = []
        for seed in DISCOVERY_SEEDS:
            run_dir = root / "discovery" / candidate["name"] / "outputs" / "mlp" / f"K{DISCOVERY_K}" / f"seed_{seed}"
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            if status.get("status") != "complete":
                raise RuntimeError(f"incomplete run: {run_dir}")
            with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            classifier = onset(rows, "classifier_best_cyclic_score")
            embedding = onset(rows, "embedding_best_cyclic_score")
            lead = embedding - classifier if classifier is not None and embedding is not None else None
            final_accuracy = float(rows[-1]["test_accuracy"])
            record = {
                "candidate": candidate["name"],
                "seed": seed,
                "classifier_onset": classifier,
                "embedding_onset": embedding,
                "lead": lead,
                "classifier_first_complete": lead is not None and lead > 0,
                "classifier_first_lower_bound": classifier is not None and embedding is None,
                "final_test_accuracy": final_accuracy,
            }
            per_run.append(record)
            rows_for_candidate.append(record)

        complete_leads = [row["lead"] for row in rows_for_candidate if row["lead"] is not None]
        classifier_onsets = [row["classifier_onset"] for row in rows_for_candidate if row["classifier_onset"] is not None]
        embedding_onsets = [row["embedding_onset"] for row in rows_for_candidate if row["embedding_onset"] is not None]
        accuracy_pass = sum(row["final_test_accuracy"] >= 0.99 for row in rows_for_candidate)
        complete_first = sum(row["classifier_first_complete"] for row in rows_for_candidate)
        lower_bound_first = sum(row["classifier_first_lower_bound"] for row in rows_for_candidate)
        complete_pairs = len(complete_leads)
        lead_mean = statistics.mean(complete_leads) if complete_leads else None
        classifier_mean = statistics.mean(classifier_onsets) if classifier_onsets else None
        embedding_mean = statistics.mean(embedding_onsets) if embedding_onsets else None
        target_distance = float("inf")
        if lead_mean is not None and classifier_mean is not None and embedding_mean is not None:
            target_distance = (
                ((classifier_mean - REFERENCE_TARGET["classifier_onset_mean"]) / 500.0) ** 2
                + ((embedding_mean - REFERENCE_TARGET["embedding_onset_mean"]) / 3000.0) ** 2
                + ((lead_mean - REFERENCE_TARGET["lead_mean"]) / 3000.0) ** 2
            )
        eligible = accuracy_pass == len(DISCOVERY_SEEDS)
        selection_key = (
            int(eligible),
            complete_first,
            lower_bound_first,
            complete_pairs,
            -target_distance,
        )
        per_candidate.append(
            {
                **candidate,
                "accuracy_pass_count": accuracy_pass,
                "classifier_first_complete_count": complete_first,
                "classifier_first_lower_bound_count": lower_bound_first,
                "complete_onset_pairs": complete_pairs,
                "classifier_onset_mean": classifier_mean,
                "embedding_onset_mean": embedding_mean,
                "lead_mean": lead_mean,
                "target_distance": target_distance,
                "eligible": eligible,
                "selection_key": selection_key,
            }
        )

    per_candidate.sort(key=lambda row: row["selection_key"], reverse=True)
    selected = per_candidate[0]
    output_dir = root / "discovery_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_run.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_run[0]))
        writer.writeheader()
        writer.writerows(per_run)
    serializable = []
    for row in per_candidate:
        item = dict(row)
        item["selection_key"] = list(item["selection_key"])
        serializable.append(item)
    (output_dir / "candidate_ranking.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    selected_config = {
        "selection_scope": {"K": DISCOVERY_K, "seeds": list(DISCOVERY_SEEDS), "held_out": "K97 seeds 4-5 and all K79/K113 runs"},
        "selection_rule": "accuracy eligibility, complete classifier-first count, right-censored lower-bound evidence, complete pairs, then distance to original K97 onset target",
        "selected": {
            key: selected[key]
            for key in ("name", "embedding_lr", "decoder_lr", "decoder_wd", "embedding_dim", "hidden_dim")
            if key in selected
        },
        "discovery_metrics": {key: selected[key] for key in ("accuracy_pass_count", "classifier_first_complete_count", "classifier_first_lower_bound_count", "complete_onset_pairs", "classifier_onset_mean", "embedding_onset_mean", "lead_mean", "target_distance")},
    }
    (output_dir / "selected_config.json").write_text(json.dumps(selected_config, indent=2), encoding="utf-8")
    print(json.dumps(selected_config, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if args.command == "launch-discovery":
        launch_discovery(root, args.gpu_ids, args.max_parallel)
    else:
        summarize_discovery(root)


if __name__ == "__main__":
    main()
