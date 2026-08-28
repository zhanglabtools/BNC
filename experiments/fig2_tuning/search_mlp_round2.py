"""Second discovery-only search round: decoder LR/weight-decay refinement."""

from __future__ import annotations

import argparse
from pathlib import Path

import search_mlp_classifier_first as core


ROUND2_CANDIDATES = tuple(
    {
        "name": f"e2e-4_d{decoder_name}_wd{wd_name}",
        "embedding_lr": 2e-4,
        "decoder_lr": decoder_lr,
        "decoder_wd": decoder_wd,
    }
    for decoder_name, decoder_lr in (("1e-3", 1e-3), ("2e-3", 2e-3), ("3e-3", 3e-3))
    for wd_name, decoder_wd in (("0", 0.0), ("0p2", 0.2), ("0p4", 0.4), ("0p6", 0.6))
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("launch", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[4, 5])
    parser.add_argument("--max-parallel", type=int, default=6)
    args = parser.parse_args()
    core.CANDIDATES = ROUND2_CANDIDATES
    if args.command == "launch":
        core.launch_discovery(args.root.resolve(), args.gpu_ids, args.max_parallel)
    else:
        core.summarize_discovery(args.root.resolve())


if __name__ == "__main__":
    main()
