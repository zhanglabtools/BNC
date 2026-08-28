#!/usr/bin/env python3
"""Launch the K=97 regularization-coefficient ablation on shared GPU slots."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ARCHITECTURES = ("mlp", "transformer", "lstm", "rnn")
SEEDS = (1, 2, 3, 4, 5)
DEFAULT_LAMBDAS = (0.01, 0.1, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
K = 97


def lambda_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambdas", nargs="+", type=float, default=list(DEFAULT_LAMBDAS))
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--output-root", type=Path, default=root / "lambda_ablation_K97_outputs")
    parser.add_argument(
        "--architecture-source",
        type=Path,
        default=root.parent / "classifier_first_multiarchitecture_K79_K97_K113_20260721"
        / "run_classifier_first_multiarchitecture.py",
    )
    parser.add_argument(
        "--dense-root",
        type=Path,
        default=root.parent / "classifier_first_multiarchitecture_K79_K97_K113_20260721"
        / "outputs",
    )
    parser.add_argument("--gpu-ids", nargs="+", type=int, required=True)
    parser.add_argument("--max-parallel-per-gpu", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if any(value < 0 for value in args.lambdas):
        parser.error("lambda values must be non-negative")
    if args.max_parallel_per_gpu <= 0:
        parser.error("max-parallel-per-gpu must be positive")
    return args


def fixed_training_args() -> list[str]:
    """Formal baseline settings; lambda is the only varied training parameter."""
    return [
        "--epochs", "6000",
        "--lr", "0.001",
        "--min-lr", "0.00001",
        "--weight-decay", "0.5",
        "--beta2", "0.98",
        "--frac-train", "0.7",
        "--collapse-tail-weight", "0",
        "--collapse-balance-weight", "0",
        "--collapse-participation-weight", "5",
        "--collapse-start-epoch", "0",
        "--collapse-ramp-epochs", "200",
    ]


def worker_command(
    args: argparse.Namespace,
    architecture: str,
    seed: int,
    regularization_coefficient: float,
) -> list[str]:
    output = args.output_root.resolve() / f"lambda_{lambda_token(regularization_coefficient)}"
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_rank2_representation_collapse.py"),
        "--worker-architecture", architecture,
        "--worker-k", str(K),
        "--worker-seed", str(seed),
        "--architecture-source", str(args.architecture_source.resolve()),
        "--dense-root", str(args.dense_root.resolve()),
        "--output-root", str(output),
        "--device", "cuda:0",
        "--regularization-coefficient", str(regularization_coefficient),
        *fixed_training_args(),
    ]
    if args.resume:
        command.append("--resume")
    return command


def is_complete(
    root: Path, architecture: str, seed: int, regularization_coefficient: float
) -> bool:
    status_path = (
        root.resolve()
        / f"lambda_{lambda_token(regularization_coefficient)}"
        / "runs"
        / architecture
        / f"K{K}"
        / f"seed_{seed}"
        / "status.json"
    )
    if not status_path.exists():
        return False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    return status.get("status") == "complete" and int(status.get("epoch", -1)) >= 6000


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "regularization coefficient ablation",
        "K": K,
        "lambdas": args.lambdas,
        "architectures": args.architectures,
        "seeds": args.seeds,
        "varied_parameter": "regularization_coefficient",
        "fixed_training_parameters": {
            "epochs": 6000,
            "optimizer": "AdamW",
            "lr": 0.001,
            "min_lr": 0.00001,
            "scheduler": "cosine",
            "weight_decay": 0.5,
            "beta2": 0.98,
            "frac_train": 0.7,
            "collapse_tail_weight": 0.0,
            "collapse_balance_weight": 0.0,
            "collapse_participation_weight": 5.0,
            "collapse_start_epoch": 0,
            "collapse_ramp_epochs": 200,
        },
        "job_count": len(args.lambdas) * len(args.architectures) * len(args.seeds),
    }
    (output_root / "ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    jobs = [
        (regularization_coefficient, architecture, seed)
        for seed in args.seeds
        for architecture in args.architectures
        for regularization_coefficient in args.lambdas
        if not is_complete(output_root, architecture, seed, regularization_coefficient)
    ]
    slots = [gpu for gpu in args.gpu_ids for _ in range(args.max_parallel_per_gpu)]
    active: list[
        tuple[subprocess.Popen[Any], Any, float, str, int, int]
    ] = []
    failures: list[tuple[float, str, int, int]] = []
    log_root = output_root / "launcher_logs"
    log_root.mkdir(parents=True, exist_ok=True)

    while jobs or active:
        available = list(slots)
        for entry in active:
            available.remove(entry[5])
        while jobs and available:
            regularization_coefficient, architecture, seed = jobs.pop(0)
            gpu = available.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            token = lambda_token(regularization_coefficient)
            log_path = log_root / f"lambda_{token}_{architecture}_K{K}_seed{seed}.log"
            handle = log_path.open("a", encoding="utf-8")
            process = subprocess.Popen(
                worker_command(args, architecture, seed, regularization_coefficient),
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
            active.append(
                (process, handle, regularization_coefficient, architecture, seed, gpu)
            )
            print(
                f"launched lambda={regularization_coefficient:g} {architecture} "
                f"K={K} seed={seed} gpu={gpu} pid={process.pid}",
                flush=True,
            )

        time.sleep(2)
        still_active = []
        for process, handle, regularization_coefficient, architecture, seed, gpu in active:
            return_code = process.poll()
            if return_code is None:
                still_active.append(
                    (process, handle, regularization_coefficient, architecture, seed, gpu)
                )
                continue
            handle.close()
            print(
                f"finished lambda={regularization_coefficient:g} {architecture} "
                f"K={K} seed={seed} exit={return_code}",
                flush=True,
            )
            if return_code != 0:
                failures.append(
                    (regularization_coefficient, architecture, seed, return_code)
                )
        active = still_active

    if failures:
        raise RuntimeError(f"failed jobs: {failures}")
    print("all lambda-ablation jobs complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
