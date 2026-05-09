"""Model registry and optional backend factories."""

from __future__ import annotations

from typing import Any, Mapping

from segcraft.config import ModelConfig, TaskConfig


TORCHVISION_ALIASES = {
    "deeplabv3": "deeplabv3_resnet50",
    "deeplabv3_resnet50": "deeplabv3_resnet50",
    "deeplabv3_resnet101": "deeplabv3_resnet101",
    "fcn": "fcn_resnet50",
    "fcn_resnet50": "fcn_resnet50",
    "fcn_resnet101": "fcn_resnet101",
    "lraspp": "lraspp_mobilenet_v3_large",
    "lraspp_mobilenet_v3_large": "lraspp_mobilenet_v3_large",
}

TORCHVISION_WEIGHT_ENUMS = {
    "deeplabv3_resnet50": "DeepLabV3_ResNet50_Weights",
    "deeplabv3_resnet101": "DeepLabV3_ResNet101_Weights",
    "fcn_resnet50": "FCN_ResNet50_Weights",
    "fcn_resnet101": "FCN_ResNet101_Weights",
    "lraspp_mobilenet_v3_large": "LRASPP_MobileNet_V3_Large_Weights",
}

SMP_ALIASES = {
    "unet": "Unet",
    "unetplusplus": "UnetPlusPlus",
    "fpn": "FPN",
    "linknet": "Linknet",
    "pspnet": "PSPNet",
}

SUPPORTED_MODELS = set(TORCHVISION_ALIASES) | set(SMP_ALIASES)


def build_model(model_config: ModelConfig | Mapping[str, Any], task_config: TaskConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Return a lightweight model spec without importing heavy ML packages."""
    model = _coerce_model_config(model_config)
    task = _coerce_task_config(task_config)
    name = model.name.lower()
    backend = _resolve_backend(model.backend, name)
    factory = _resolve_factory(backend, name)

    return {
        "name": name,
        "backend": backend,
        "factory": factory,
        "encoder": model.encoder,
        "pretrained": model.pretrained,
        "num_classes": task.num_classes,
        "task_type": task.type,
    }


def create_model(model_config: ModelConfig | Mapping[str, Any], task_config: TaskConfig | Mapping[str, Any]) -> Any:
    """Instantiate the configured segmentation model.

    Heavy dependencies are optional. Install `.[torch]` for TorchVision models
    or `.[torch,smp]` for segmentation-models-pytorch models.
    """
    spec = build_model(model_config, task_config)
    if spec["backend"] == "torchvision":
        return _create_torchvision_model(spec)
    if spec["backend"] == "smp":
        return _create_smp_model(spec)
    raise ValueError(f"Unsupported model backend: {spec['backend']}")


def _coerce_model_config(config: ModelConfig | Mapping[str, Any]) -> ModelConfig:
    if isinstance(config, ModelConfig):
        return config
    return ModelConfig.from_mapping(config)


def _coerce_task_config(config: TaskConfig | Mapping[str, Any]) -> TaskConfig:
    if isinstance(config, TaskConfig):
        return config
    return TaskConfig.from_mapping(config)


def _resolve_backend(configured_backend: str, name: str) -> str:
    if configured_backend != "auto":
        return configured_backend
    if name in TORCHVISION_ALIASES:
        return "torchvision"
    if name in SMP_ALIASES:
        return "smp"
    _raise_unsupported_model(name)


def _resolve_factory(backend: str, name: str) -> str:
    if backend == "torchvision" and name in TORCHVISION_ALIASES:
        return TORCHVISION_ALIASES[name]
    if backend == "smp" and name in SMP_ALIASES:
        return SMP_ALIASES[name]
    _raise_unsupported_model(name, backend=backend)


def _raise_unsupported_model(name: str, backend: str | None = None) -> None:
    supported = ", ".join(sorted(SUPPORTED_MODELS))
    backend_text = f" for backend '{backend}'" if backend else ""
    raise ValueError(f"Unsupported model '{name}'{backend_text}. Supported models: {supported}")


def _create_torchvision_model(spec: Mapping[str, Any]) -> Any:
    try:
        from torchvision.models import segmentation
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TorchVision models require optional dependencies. Install with `pip install -e .[torch]`."
        ) from exc

    factory_name = spec["factory"]
    factory = getattr(segmentation, factory_name)
    weights = _resolve_torchvision_weights(segmentation, factory_name, spec)
    if weights is not None:
        return factory(weights=weights)

    kwargs = {"num_classes": spec["num_classes"], "weights": None}
    if spec["pretrained"]:
        kwargs["weights_backbone"] = "DEFAULT"
    try:
        return factory(**kwargs)
    except TypeError:
        legacy_kwargs = {
            "num_classes": spec["num_classes"],
            "pretrained": False,
            "pretrained_backbone": bool(spec["pretrained"]),
        }
        try:
            return factory(**legacy_kwargs)
        except TypeError:
            legacy_kwargs.pop("pretrained_backbone")
            return factory(**legacy_kwargs)


def _resolve_torchvision_weights(segmentation: Any, factory_name: str, spec: Mapping[str, Any]) -> Any:
    if not spec["pretrained"]:
        return None

    enum_name = TORCHVISION_WEIGHT_ENUMS.get(factory_name)
    if not enum_name or not hasattr(segmentation, enum_name):
        return None

    weights = getattr(segmentation, enum_name).DEFAULT
    categories = weights.meta.get("categories", [])
    if categories and len(categories) != spec["num_classes"]:
        return None
    return weights


def _create_smp_model(spec: Mapping[str, Any]) -> Any:
    try:
        import segmentation_models_pytorch as smp
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SMP models require optional dependencies. Install with `pip install -e .[torch,smp]`."
        ) from exc

    factory = getattr(smp, spec["factory"])
    encoder = spec.get("encoder") or "resnet34"
    encoder_weights = "imagenet" if spec["pretrained"] else None
    return factory(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=spec["num_classes"],
    )
