"""Train/Eval/Predict workflow scaffolding.

This is intentionally backend-agnostic and returns structured run summaries.
"""

from datetime import datetime, timezone
from typing import Any, Dict

from segcraft.metrics import resolve_metrics
from segcraft.models import build_model


LOSS_BY_TASK = {
    "binary": "bce_dice",
    "multiclass": "cross_entropy_dice",
}


def _resolve_loss(task_cfg: Dict[str, Any], train_cfg: Dict[str, Any]) -> str:
    chosen = train_cfg.get("loss")
    if chosen and str(chosen).lower() != "auto":
        return str(chosen)
    return LOSS_BY_TASK[task_cfg["type"]]


def _common_summary(mode: str, config: Dict[str, Any]) -> Dict[str, Any]:
    task = config["task"]
    model = build_model(config["model"], task)
    metrics = resolve_metrics(task, config["eval"])
    return {
        "mode": mode,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "model": model,
        "metrics": metrics,
    }


def train(config: Dict[str, Any]) -> Dict[str, Any]:
    summary = _common_summary("train", config)
    summary["train"] = {
        "epochs": config["train"]["epochs"],
        "optimizer": config["train"]["optimizer"],
        "learning_rate": config["train"]["learning_rate"],
        "loss": _resolve_loss(config["task"], config["train"]),
    }
    return summary


def evaluate(config: Dict[str, Any]) -> Dict[str, Any]:
    summary = _common_summary("evaluate", config)
    summary["eval"] = {"metrics": summary["metrics"]}
    return summary


def predict(config: Dict[str, Any]) -> Dict[str, Any]:
    summary = _common_summary("predict", config)
    summary["predict"] = {
        "input_path": config["predict"]["input_path"],
        "output_path": config["predict"]["output_path"],
        "overlay_alpha": config["predict"].get("overlay_alpha", 0.5),
    }
    return summary
