"""Minimal schema validation for SegCraft configs."""

from typing import Any, Dict


REQUIRED_TOP_LEVEL_KEYS = {
    "task",
    "model",
    "data",
    "train",
    "eval",
    "predict",
    "runtime",
}

ALLOWED_TASK_TYPES = {"binary", "multiclass"}


class ConfigValidationError(ValueError):
    """Raised when config does not satisfy SegCraft schema."""


def _require_keys(section_name: str, section: Dict[str, Any], required: set[str]) -> None:
    missing = required - set(section.keys())
    if missing:
        formatted = ", ".join(sorted(missing))
        raise ConfigValidationError(f"Missing keys in '{section_name}': {formatted}")


def validate_config(config: Dict[str, Any]) -> None:
    missing = REQUIRED_TOP_LEVEL_KEYS - set(config.keys())
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ConfigValidationError(f"Missing required config sections: {missing_list}")

    task = config.get("task", {})
    task_type = task.get("type")
    if task_type not in ALLOWED_TASK_TYPES:
        options = ", ".join(sorted(ALLOWED_TASK_TYPES))
        raise ConfigValidationError(f"task.type must be one of: {options}")

    num_classes = task.get("num_classes")
    if not isinstance(num_classes, int) or num_classes < 1:
        raise ConfigValidationError("task.num_classes must be a positive integer")

    if task_type == "binary" and num_classes != 1:
        raise ConfigValidationError("For binary task.type, task.num_classes must be exactly 1")

    if task_type == "multiclass" and num_classes < 2:
        raise ConfigValidationError("For multiclass task.type, task.num_classes must be >= 2")

    _require_keys("model", config["model"], {"name"})
    _require_keys("train", config["train"], {"epochs", "optimizer", "learning_rate"})
    _require_keys("predict", config["predict"], {"input_path", "output_path"})
