"""Command line interface for SegCraft."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict

from segcraft import evaluate, load_config, predict, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="segcraft",
        description="SegCraft: config-driven semantic segmentation framework",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["validate", "train", "evaluate", "predict"],
        default="validate",
        help="Execution mode. Default is validate.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--preset",
        type=Path,
        default=None,
        help="Optional preset override YAML (e.g. configs/presets/fast_dev.yaml).",
    )
    parser.add_argument(
        "--local",
        type=Path,
        default=Path("configs/local.yaml"),
        help="Optional machine-local override YAML (applied if file exists).",
    )
    parser.add_argument("--print-config", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config, preset_path=args.preset, local_path=args.local)
    if args.print_config:
        print(json.dumps(config, indent=2))

    handlers: Dict[str, Callable[..., Dict[str, Any]]] = {
        "train": train,
        "evaluate": evaluate,
        "predict": predict,
    }

    if args.mode == "validate":
        print("SegCraft config is valid.")
        return

    summary = handlers[args.mode](args.config, preset_path=args.preset, local_path=args.local)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
