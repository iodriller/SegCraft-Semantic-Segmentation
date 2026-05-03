"""Model registry.

For now this module returns lightweight descriptors instead of heavy ML objects.
This keeps the project runnable while we layer in real training backends.
"""

from typing import Any, Dict


SUPPORTED_MODELS = {"unet", "deeplabv3", "fpn"}


def build_model(model_config: Dict[str, Any], task_config: Dict[str, Any]) -> Dict[str, Any]:
    name = str(model_config.get("name", "")).lower()
    if name not in SUPPORTED_MODELS:
        supported = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(f"Unsupported model '{name}'. Supported models: {supported}")

    # Human note: this is intentionally simple now; we'll swap this for a real backend object later.
    return {
        "name": name,
        "encoder": model_config.get("encoder"),
        "pretrained": bool(model_config.get("pretrained", True)),
        "num_classes": task_config["num_classes"],
        "task_type": task_config["type"],
    }
