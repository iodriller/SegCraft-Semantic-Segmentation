"""YAML config loading and validation for SegCraft."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .schema import validate_config


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # clear UX when env is missing deps
        raise ModuleNotFoundError(
            "PyYAML is required to read config files. Install with `pip install pyyaml`."
        ) from exc

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a YAML mapping/object: {path}")
    return data


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_and_validate_config(
    config_path: str | Path,
    preset_path: str | Path | None = None,
    local_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Load base config, then optional preset/local overlays, then validate."""
    config = _load_yaml(Path(config_path))

    if preset_path:
        # Friendly behavior: missing preset should fail fast so users don't silently run defaults.
        preset_cfg = _load_yaml(Path(preset_path))
        config = _deep_merge(config, preset_cfg)

    if local_path and Path(local_path).exists():
        local_cfg = _load_yaml(Path(local_path))
        config = _deep_merge(config, local_cfg)

    validate_config(config)
    return config
