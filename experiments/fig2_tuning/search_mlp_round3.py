"""Third discovery-only search round: MLP embedding/hidden widths."""

from __future__ import annotations

import argparse
from pathlib import Path

import search_mlp_classifier_first as core


WIDTHS = (
    (128, 64),
    (128, 128),
    (256, 64),
    (256, 128),
    (256, 256),
    (256, 512),
    (512, 128),
    (512, 256),
    (512, 512),
)
ROUND3_CANDIDATES = tuple(
    {
        "name": f"emb{embedding_dim}_hid{hidden_dim}_e2e-4_d3e-3_wd0p4",
        "embedding_lr": 2e-4,
        "decoder_lr": 3e-3,
        "decoder_wd": 0.4,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
    }
    for embedding_dim, hidden_dim in WIDTHS
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("launch", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[4, 5])
    parser.add_argument("--max-parallel", type=int, default=6)
    args = parser.parse_args()
    core.CANDIDATES = ROUND3_CANDIDATES
    if args.command == "launch":
        core.launch_discovery(args.root.resolve(), args.gpu_ids, args.max_parallel)
    else:
        core.summarize_discovery(args.root.resolve())


if __name__ == "__main__":
    main()
