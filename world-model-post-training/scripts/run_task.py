"""CLI entry point for repo-level task dispatch."""

from __future__ import annotations

import argparse

from verl_post_training.launch import load_task_config, resolve_dispatch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to a repo-level YAML task config.")
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    config = load_task_config(args.config)
    return resolve_dispatch(config)


if __name__ == "__main__":
    main()
