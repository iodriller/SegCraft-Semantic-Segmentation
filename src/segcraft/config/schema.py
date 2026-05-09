"""Typed configuration and validation for SegCraft."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


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
ALLOWED_MODEL_BACKENDS = {"auto", "torchvision", "smp"}


class ConfigValidationError(ValueError):
    """Raised when a config does not satisfy the SegCraft schema."""


def _as_mapping(source: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = source.get(key)
    if not isinstance(value, Mapping):
        raise ConfigValidationError(f"'{key}' must be a mapping/object")
    return value


def _as_positive_int(section: str, key: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigValidationError(f"{section}.{key} must be a positive integer")
    return value


def _as_nonnegative_int(section: str, key: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigValidationError(f"{section}.{key} must be a non-negative integer")
    return value


def _as_positive_float(section: str, key: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigValidationError(f"{section}.{key} must be a positive number")
    return float(value)


def _as_string(section: str, key: str, value: Any, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ConfigValidationError(f"{section}.{key} must be a non-empty string")
    return value


def _as_string_list(section: str, key: str, value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigValidationError(f"{section}.{key} must be a list of strings")
    return value


def _as_image_size(value: Any) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(
            not isinstance(item, bool) and isinstance(item, int) and item > 0 for item in value
        )
    ):
        raise ConfigValidationError("data.image_size must be [height, width] with positive integers")
    return int(value[0]), int(value[1])


@dataclass(frozen=True)
class TaskConfig:
    type: str
    num_classes: int
    class_names: list[str] = field(default_factory=list)
    ignore_index: int | None = 255

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TaskConfig":
        task_type = _as_string("task", "type", data.get("type"))
        if task_type not in ALLOWED_TASK_TYPES:
            options = ", ".join(sorted(ALLOWED_TASK_TYPES))
            raise ConfigValidationError(f"task.type must be one of: {options}")

        num_classes = _as_positive_int("task", "num_classes", data.get("num_classes"))
        if task_type == "binary" and num_classes != 1:
            raise ConfigValidationError("For binary task.type, task.num_classes must be exactly 1")
        if task_type == "multiclass" and num_classes < 2:
            raise ConfigValidationError("For multiclass task.type, task.num_classes must be >= 2")

        class_names = _as_string_list("task", "class_names", data.get("class_names", []))
        valid_name_counts = {num_classes}
        if task_type == "binary":
            valid_name_counts.add(2)
        if class_names and len(class_names) not in valid_name_counts:
            raise ConfigValidationError(
                "task.class_names must match task.num_classes; binary configs may also use "
                "[background, foreground]"
            )

        ignore_index = data.get("ignore_index", 255)
        if ignore_index is not None and not isinstance(ignore_index, int):
            raise ConfigValidationError("task.ignore_index must be an integer or null")

        return cls(
            type=task_type,
            num_classes=num_classes,
            class_names=class_names,
            ignore_index=ignore_index,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "num_classes": self.num_classes,
            "class_names": list(self.class_names),
            "ignore_index": self.ignore_index,
        }


@dataclass(frozen=True)
class ModelConfig:
    name: str
    backend: str = "auto"
    encoder: str | None = None
    pretrained: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ModelConfig":
        name = _as_string("model", "name", data.get("name"))
        backend = _as_string("model", "backend", data.get("backend", "auto"))
        if backend not in ALLOWED_MODEL_BACKENDS:
            options = ", ".join(sorted(ALLOWED_MODEL_BACKENDS))
            raise ConfigValidationError(f"model.backend must be one of: {options}")

        encoder = data.get("encoder")
        if encoder is not None:
            encoder = _as_string("model", "encoder", encoder)

        pretrained = data.get("pretrained", True)
        if not isinstance(pretrained, bool):
            raise ConfigValidationError("model.pretrained must be true or false")

        return cls(name=name, backend=backend, encoder=encoder, pretrained=pretrained)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "encoder": self.encoder,
            "pretrained": self.pretrained,
        }


@dataclass(frozen=True)
class DataConfig:
    train_images: str = "data/train/images"
    train_masks: str = "data/train/masks"
    val_images: str = "data/val/images"
    val_masks: str = "data/val/masks"
    image_size: tuple[int, int] = (512, 512)
    batch_size: int = 4
    num_workers: int = 4
    mask_suffix: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DataConfig":
        return cls(
            train_images=_as_string(
                "data", "train_images", data.get("train_images", "data/train/images")
            ),
            train_masks=_as_string("data", "train_masks", data.get("train_masks", "data/train/masks")),
            val_images=_as_string("data", "val_images", data.get("val_images", "data/val/images")),
            val_masks=_as_string("data", "val_masks", data.get("val_masks", "data/val/masks")),
            image_size=_as_image_size(data.get("image_size", [512, 512])),
            batch_size=_as_positive_int("data", "batch_size", data.get("batch_size", 4)),
            num_workers=_as_nonnegative_int("data", "num_workers", data.get("num_workers", 4)),
            mask_suffix=_as_string(
                "data", "mask_suffix", data.get("mask_suffix", ""), allow_empty=True
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_images": self.train_images,
            "train_masks": self.train_masks,
            "val_images": self.val_images,
            "val_masks": self.val_masks,
            "image_size": list(self.image_size),
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "mask_suffix": self.mask_suffix,
        }


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    optimizer: str
    learning_rate: float
    loss: str = "auto"
    scheduler: str = "none"
    amp: bool = False
    resume_from: str | None = None
    early_stopping_patience: int | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TrainConfig":
        scheduler = _as_string("train", "scheduler", data.get("scheduler", "none")).lower()
        if scheduler not in {"none", "cosine", "step"}:
            raise ConfigValidationError("train.scheduler must be one of: none, cosine, step")

        amp = data.get("amp", False)
        if not isinstance(amp, bool):
            raise ConfigValidationError("train.amp must be true or false")

        resume_from = data.get("resume_from")
        if resume_from is not None:
            resume_from = _as_string("train", "resume_from", resume_from)

        patience = data.get("early_stopping_patience")
        if patience is not None:
            patience = _as_nonnegative_int("train", "early_stopping_patience", patience)

        return cls(
            epochs=_as_positive_int("train", "epochs", data.get("epochs")),
            optimizer=_as_string("train", "optimizer", data.get("optimizer")),
            learning_rate=_as_positive_float("train", "learning_rate", data.get("learning_rate")),
            loss=_as_string("train", "loss", data.get("loss", "auto")),
            scheduler=scheduler,
            amp=amp,
            resume_from=resume_from,
            early_stopping_patience=patience,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "loss": self.loss,
            "scheduler": self.scheduler,
            "amp": self.amp,
            "resume_from": self.resume_from,
            "early_stopping_patience": self.early_stopping_patience,
        }


@dataclass(frozen=True)
class EvalConfig:
    metrics: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EvalConfig":
        return cls(metrics=_as_string_list("eval", "metrics", data.get("metrics", [])))

    def to_dict(self) -> dict[str, Any]:
        return {"metrics": list(self.metrics)}


@dataclass(frozen=True)
class PredictConfig:
    input_path: str
    output_path: str
    overlay_alpha: float = 0.5
    annotate: bool = True
    save_video: bool = True
    video_fps: float = 6.0
    video_path: str | None = None
    preserve_audio: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PredictConfig":
        overlay_alpha = data.get("overlay_alpha", 0.5)
        if not isinstance(overlay_alpha, (int, float)) or not 0 <= overlay_alpha <= 1:
            raise ConfigValidationError("predict.overlay_alpha must be between 0 and 1")
        annotate = data.get("annotate", True)
        if not isinstance(annotate, bool):
            raise ConfigValidationError("predict.annotate must be true or false")
        save_video = data.get("save_video", True)
        if not isinstance(save_video, bool):
            raise ConfigValidationError("predict.save_video must be true or false")
        video_fps = _as_positive_float("predict", "video_fps", data.get("video_fps", 6.0))
        video_path = data.get("video_path")
        if video_path is not None:
            video_path = _as_string("predict", "video_path", video_path)
        preserve_audio = data.get("preserve_audio", True)
        if not isinstance(preserve_audio, bool):
            raise ConfigValidationError("predict.preserve_audio must be true or false")

        return cls(
            input_path=_as_string("predict", "input_path", data.get("input_path")),
            output_path=_as_string("predict", "output_path", data.get("output_path")),
            overlay_alpha=float(overlay_alpha),
            annotate=annotate,
            save_video=save_video,
            video_fps=video_fps,
            video_path=video_path,
            preserve_audio=preserve_audio,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "overlay_alpha": self.overlay_alpha,
            "annotate": self.annotate,
            "save_video": self.save_video,
            "video_fps": self.video_fps,
            "video_path": self.video_path,
            "preserve_audio": self.preserve_audio,
        }


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 42
    device: str = "auto"
    output_dir: str = "outputs"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RuntimeConfig":
        seed = data.get("seed", 42)
        if not isinstance(seed, int):
            raise ConfigValidationError("runtime.seed must be an integer")

        return cls(
            seed=seed,
            device=_as_string("runtime", "device", data.get("device", "auto")),
            output_dir=_as_string("runtime", "output_dir", data.get("output_dir", "outputs")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "device": self.device, "output_dir": self.output_dir}


@dataclass(frozen=True)
class SegCraftConfig:
    task: TaskConfig
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    eval: EvalConfig
    predict: PredictConfig
    runtime: RuntimeConfig

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "SegCraftConfig":
        missing = REQUIRED_TOP_LEVEL_KEYS - set(config.keys())
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ConfigValidationError(f"Missing required config sections: {missing_list}")

        return cls(
            task=TaskConfig.from_mapping(_as_mapping(config, "task")),
            model=ModelConfig.from_mapping(_as_mapping(config, "model")),
            data=DataConfig.from_mapping(_as_mapping(config, "data")),
            train=TrainConfig.from_mapping(_as_mapping(config, "train")),
            eval=EvalConfig.from_mapping(_as_mapping(config, "eval")),
            predict=PredictConfig.from_mapping(_as_mapping(config, "predict")),
            runtime=RuntimeConfig.from_mapping(_as_mapping(config, "runtime")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "model": self.model.to_dict(),
            "data": self.data.to_dict(),
            "train": self.train.to_dict(),
            "eval": self.eval.to_dict(),
            "predict": self.predict.to_dict(),
            "runtime": self.runtime.to_dict(),
        }


def parse_config(config: Mapping[str, Any] | SegCraftConfig) -> SegCraftConfig:
    if isinstance(config, SegCraftConfig):
        return config
    if not isinstance(config, Mapping):
        raise ConfigValidationError("Config must be a mapping/object")
    return SegCraftConfig.from_mapping(config)


def validate_config(config: Mapping[str, Any] | SegCraftConfig) -> None:
    parse_config(config)
