from __future__ import annotations

import argparse
import json
from pathlib import Path

from bnc_repro.config import load_config


FIGURES = ("fig1", "fig2", "fig_s1", "fig_s2", "fig_s3", "fig_s4")


def _aggregate(args: argparse.Namespace) -> None:
    if args.figure == "fig2":
        from bnc_repro.aggregation.fig2 import aggregate_fig2

        result = aggregate_fig2(args.runs_root, args.output)
    elif args.figure in {"fig_s3", "fig_s4"}:
        from bnc_repro.aggregation.alignment import aggregate_alignment

        result = aggregate_alignment(
            args.runs_root, args.output, centered=args.figure == "fig_s4"
        )
    elif args.figure in {"fig_s1", "fig_s2"}:
        from bnc_repro.aggregation.rank import aggregate_rank

        result = aggregate_rank(args.runs_root, args.output, figure=args.figure)
    else:
        raise SystemExit("Figure 1 aggregation requires a trained checkpoint and is handled by train")
    print(result)


def _plot(args: argparse.Namespace) -> None:
    if args.figure == "fig1":
        from bnc_repro.plotting.fig1 import plot_fig1_coordinates

        paths = plot_fig1_coordinates(args.data, args.output, font_family=args.font_family)
    elif args.figure == "fig2":
        from bnc_repro.plotting.fig2 import plot_fig2_mlp

        paths = plot_fig2_mlp(args.data, args.output, font_family=args.font_family)
    elif args.figure == "fig_s1":
        from bnc_repro.plotting.fig_s1 import plot_fig_s1

        paths = plot_fig_s1(args.data, args.output, font_family=args.font_family)
    elif args.figure == "fig_s2":
        from bnc_repro.plotting.fig_s2 import plot_fig_s2

        paths = plot_fig_s2(args.data, args.output, font_family=args.font_family)
    else:
        from bnc_repro.plotting.alignment import plot_alignment

        paths = plot_alignment(
            args.data,
            args.output,
            centered=args.figure == "fig_s4",
            font_family=args.font_family,
        )
    print(json.dumps([str(path) for path in paths], indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bnc-repro")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--output", type=Path)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--figure", choices=FIGURES, required=True)
    aggregate.add_argument("--runs-root", required=True, type=Path)
    aggregate.add_argument("--output", required=True, type=Path)

    plot = subparsers.add_parser("plot")
    plot.add_argument("--figure", choices=FIGURES, required=True)
    plot.add_argument("--data", required=True, type=Path)
    plot.add_argument("--output", required=True, type=Path)
    plot.add_argument("--font-family", default="Arial")

    validate = subparsers.add_parser("validate")
    validate.add_argument("--figure", choices=(*FIGURES, "all"), required=True)
    validate.add_argument("--data", type=Path, default=Path("paper_data"))

    args = parser.parse_args(argv)
    if args.command == "train":
        from bnc_repro.training.engine import execute_config

        paths = execute_config(load_config(args.config), args.output)
        print(json.dumps([str(path) for path in paths], indent=2))
    elif args.command == "aggregate":
        _aggregate(args)
    elif args.command == "plot":
        _plot(args)
    else:
        from bnc_repro.validation import validate_all, validate_figure

        result = validate_all(args.data) if args.figure == "all" else validate_figure(args.figure, args.data)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

