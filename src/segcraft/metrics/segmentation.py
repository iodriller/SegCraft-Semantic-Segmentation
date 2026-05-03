"""Segmentation metric resolution helpers."""

from typing import Any, Dict, List

DEFAULT_BINARY_METRICS = ["iou", "dice", "pixel_accuracy"]
DEFAULT_MULTICLASS_METRICS = ["miou", "dice_macro", "pixel_accuracy"]


def resolve_metrics(task_config: Dict[str, Any], eval_config: Dict[str, Any]) -> List[str]:
    configured = eval_config.get("metrics")
    if configured:
        return list(configured)

    if task_config["type"] == "binary":
        return DEFAULT_BINARY_METRICS
    return DEFAULT_MULTICLASS_METRICS
