"""Public API for SegCraft."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .config.loader import load_and_validate_config, load_config_object as load_typed_config
from .config.schema import SegCraftConfig
from .engine import evaluate as run_evaluate
from .engine import predict as run_predict
from .engine import train as run_train


def load_config(
    config_path: str | Path,
    preset_path: str | Path | None = None,
    local_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Load and validate SegCraft config with optional preset/local overrides."""
    return load_and_validate_config(config_path, preset_path=preset_path, local_path=local_path)


def load_config_object(
    config_path: str | Path,
    preset_path: str | Path | None = None,
    local_path: str | Path | None = None,
) -> SegCraftConfig:
    """Load and validate SegCraft config as typed sections."""
    return load_typed_config(config_path, preset_path=preset_path, local_path=local_path)


def train(
    config_path: str | Path,
    preset_path: str | Path | None = None,
    local_path: str | Path | None = None,
) -> Dict[str, Any]:
    config = load_config(config_path, preset_path=preset_path, local_path=local_path)
    return run_train(config)


def evaluate(
    config_path: str | Path,
    preset_path: str | Path | None = None,
    local_path: str | Path | None = None,
) -> Dict[str, Any]:
    config = load_config(config_path, preset_path=preset_path, local_path=local_path)
    return run_evaluate(config)


def predict(
    config_path: str | Path,
    preset_path: str | Path | None = None,
    local_path: str | Path | None = None,
) -> Dict[str, Any]:
    config = load_config(config_path, preset_path=preset_path, local_path=local_path)
    return run_predict(config)
