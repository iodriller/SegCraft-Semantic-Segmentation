"""YAML config loading and validation for SegCraft."""

from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, Dict

from .schema import SegCraftConfig, parse_config, validate_config


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


def list_available_presets() -> list[str]:
    """Return preset names available from the source tree or installed package."""
    names = set()
    local_dir = Path("configs/presets")
    if local_dir.exists():
        names.update(path.stem for path in local_dir.glob("*.yaml"))

    try:
        preset_dir = files("segcraft.templates.presets")
        names.update(item.name.removesuffix(".yaml") for item in preset_dir.iterdir() if item.name.endswith(".yaml"))
    except ModuleNotFoundError:
        pass
    return sorted(names)


def _load_preset_yaml(preset_path: str | Path) -> Dict[str, Any]:
    candidate = Path(preset_path)
    if candidate.exists():
        return _load_yaml(candidate)

    preset_name = candidate.name
    if not preset_name.endswith((".yaml", ".yml")):
        preset_name = f"{preset_name}.yaml"

    local_candidate = Path("configs/presets") / preset_name
    if local_candidate.exists():
        return _load_yaml(local_candidate)

    try:
        resource = files("segcraft.templates.presets").joinpath(preset_name)
        if resource.is_file():
            with as_file(resource) as resource_path:
                return _load_yaml(resource_path)
    except ModuleNotFoundError:
        pass

    available = ", ".join(list_available_presets()) or "none"
    raise FileNotFoundError(f"Preset not found: {preset_path}. Available presets: {available}")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_merged_config(
    config_path: str | Path,
    preset_path: str | Path | None = None,
    local_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Load base config, then optional preset/local overlays."""
    config = _load_yaml(Path(config_path))

    if preset_path:
        # Friendly behavior: missing preset should fail fast so users don't silently run defaults.
        preset_cfg = _load_preset_yaml(preset_path)
        config = _deep_merge(config, preset_cfg)

    if local_path and Path(local_path).exists():
        local_cfg = _load_yaml(Path(local_path))
        config = _deep_merge(config, local_cfg)

    return config


def load_and_validate_config(
    config_path: str | Path,
    preset_path: str | Path | None = None,
    local_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Load and validate config, returning the merged mapping."""
    config = _load_merged_config(config_path, preset_path=preset_path, local_path=local_path)
    validate_config(config)
    return config


def load_config_object(
    config_path: str | Path,
    preset_path: str | Path | None = None,
    local_path: str | Path | None = None,
) -> SegCraftConfig:
    """Load and validate config, returning typed config sections."""
    return parse_config(_load_merged_config(config_path, preset_path=preset_path, local_path=local_path))
