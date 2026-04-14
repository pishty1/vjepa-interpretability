from __future__ import annotations

import argparse

from .extract_stage import add_extract_parser
from .heatmaps_stage import add_heatmaps_parser
from .model_stage import add_model_parser
from .stats_stage import add_stats_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the V-JEPA 2.1 temporal analysis pipeline as independent stages."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_extract_parser(subparsers)
    add_model_parser(subparsers)
    add_heatmaps_parser(subparsers)
    add_stats_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
