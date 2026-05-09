"""Train/Eval/Predict workflow entry points."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from segcraft.config import SegCraftConfig, parse_config
from segcraft.metrics import resolve_metrics
from segcraft.models import build_model
from segcraft.prediction import run_prediction
from segcraft.training import run_evaluation, run_training


LOSS_BY_TASK = {
    "binary": "bce_dice",
    "multiclass": "cross_entropy_dice",
}


def _resolve_loss(config: SegCraftConfig) -> str:
    chosen = config.train.loss
    if chosen and str(chosen).lower() != "auto":
        return str(chosen)
    return LOSS_BY_TASK[config.task.type]


def _common_summary(
    mode: str, config: Mapping[str, Any] | SegCraftConfig
) -> tuple[SegCraftConfig, dict[str, Any]]:
    cfg = parse_config(config)
    model = build_model(cfg.model, cfg.task)
    metrics = resolve_metrics(cfg.task.to_dict(), cfg.eval.to_dict())
    return cfg, {
        "mode": mode,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": cfg.task.to_dict(),
        "model": model,
        "metrics": metrics,
    }


def train(config: Mapping[str, Any] | SegCraftConfig) -> dict[str, Any]:
    cfg, summary = _common_summary("train", config)
    summary["train"] = {
        "epochs": cfg.train.epochs,
        "optimizer": cfg.train.optimizer,
        "learning_rate": cfg.train.learning_rate,
        "loss": _resolve_loss(cfg),
    }
    summary["train"].update(run_training(cfg))
    return summary


def evaluate(config: Mapping[str, Any] | SegCraftConfig) -> dict[str, Any]:
    cfg, summary = _common_summary("evaluate", config)
    summary["eval"] = {"metrics": summary["metrics"]}
    summary["eval"].update(run_evaluation(cfg))
    return summary


def predict(config: Mapping[str, Any] | SegCraftConfig) -> dict[str, Any]:
    cfg, summary = _common_summary("predict", config)
    summary["predict"] = {
        "input_path": cfg.predict.input_path,
        "output_path": cfg.predict.output_path,
        "overlay_alpha": cfg.predict.overlay_alpha,
    }
    if not Path(cfg.predict.input_path).exists():
        summary["predict"]["status"] = "input_missing"
        summary["predict"]["message"] = f"Prediction input path does not exist: {cfg.predict.input_path}"
        return summary

    summary["predict"].update(run_prediction(cfg))
    return summary
