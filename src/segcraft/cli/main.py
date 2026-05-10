"""Command line interface for SegCraft."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from importlib.resources import as_file, files
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterator

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
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Base config YAML. Defaults to configs/base.yaml when present, "
            "otherwise uses the packaged base config."
        ),
    )
    parser.add_argument(
        "--preset",
        type=Path,
        default=None,
        help="Optional preset name or YAML path (e.g. fast_dev or configs/presets/fast_dev.yaml).",
    )
    parser.add_argument(
        "--local",
        type=Path,
        default=Path("configs/local.yaml"),
        help="Optional machine-local override YAML (applied if file exists).",
    )
    parser.add_argument("--print-config", action="store_true")
    return parser


@contextmanager
def resolve_config_path(config_path: Path | None = None) -> Iterator[Path]:
    """Resolve the CLI's default config for source-tree and installed-package use."""
    if config_path is not None:
        yield config_path
        return

    local_default = Path("configs/base.yaml")
    if local_default.exists():
        yield local_default
        return

    with as_file(files("segcraft.templates").joinpath("base.yaml")) as packaged_default:
        yield packaged_default


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    with resolve_config_path(args.config) as config_path:
        config = load_config(config_path, preset_path=args.preset, local_path=args.local)
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

        summary = handlers[args.mode](config_path, preset_path=args.preset, local_path=args.local)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
