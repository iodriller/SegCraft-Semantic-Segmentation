"""Image-folder prediction for semantic segmentation models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from segcraft.config import SegCraftConfig, parse_config
from segcraft.data import list_image_files
from segcraft.models import create_model
from segcraft.video import write_video_from_images


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def run_prediction(config: Mapping[str, Any] | SegCraftConfig) -> dict[str, Any]:
    cfg = parse_config(config)
    input_path = Path(cfg.predict.input_path)
    image_paths = list_image_files(input_path)
    if not image_paths:
        raise ValueError(f"No images found under: {input_path}")

    output_dir = Path(cfg.predict.output_path)
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    torch = _torch()
    device = _resolve_device(cfg.runtime.device, torch)
    model = create_model(cfg.model, cfg.task).to(device)
    model.eval()

    artifacts = []
    with torch.inference_mode():
        for image_path in image_paths:
            artifact = _predict_one(
                image_path=image_path,
                model=model,
                device=device,
                image_size=cfg.data.image_size,
                num_classes=cfg.task.num_classes,
                overlay_alpha=cfg.predict.overlay_alpha,
                masks_dir=masks_dir,
                overlays_dir=overlays_dir,
            )
            artifacts.append(artifact)

    summary = {
        "status": "completed",
        "device": str(device),
        "images_found": len(image_paths),
        "images_processed": len(artifacts),
        "masks_dir": str(masks_dir),
        "overlays_dir": str(overlays_dir),
        "sample_outputs": artifacts[:5],
    }
    if cfg.predict.save_video:
        video_path = Path(cfg.predict.video_path) if cfg.predict.video_path else output_dir / "overlay.mp4"
        summary["overlay_video"] = write_video_from_images(
            overlays_dir,
            video_path,
            fps=cfg.predict.video_fps,
        )
    return summary


def _predict_one(
    *,
    image_path: Path,
    model: Any,
    device: Any,
    image_size: tuple[int, int],
    num_classes: int,
    overlay_alpha: float,
    masks_dir: Path,
    overlays_dir: Path,
) -> dict[str, str]:
    image = _pil().open(image_path).convert("RGB")
    original_size = image.size
    batch = _preprocess(image, image_size).to(device)
    output = model(batch)
    logits = output["out"] if isinstance(output, dict) else output
    prediction = logits.argmax(dim=1)[0].detach().cpu().numpy().astype("uint8")

    mask = _pil().fromarray(prediction).resize(original_size, resample=_pil().NEAREST)
    mask.putpalette(_palette(num_classes))
    mask_path = masks_dir / f"{image_path.stem}.png"
    mask.save(mask_path)

    overlay = _overlay(image, mask, num_classes=num_classes, alpha=overlay_alpha)
    overlay_path = overlays_dir / f"{image_path.stem}.jpg"
    overlay.save(overlay_path, quality=92)

    return {
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
    }


def _preprocess(image: Any, image_size: tuple[int, int]) -> Any:
    import numpy as np

    torch = _torch()
    height, width = image_size
    resized = image.resize((width, height), resample=_pil().BILINEAR)
    array = np.asarray(resized).astype("float32") / 255.0
    array = (array - np.asarray(IMAGENET_MEAN, dtype="float32")) / np.asarray(
        IMAGENET_STD, dtype="float32"
    )
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def _overlay(image: Any, mask: Any, *, num_classes: int, alpha: float) -> Any:
    import numpy as np

    mask_index = np.asarray(mask).astype("int64")
    colors = np.asarray(_palette(num_classes), dtype="uint8").reshape(-1, 3)
    color_mask = colors[mask_index % len(colors)]
    image_array = np.asarray(image).astype("float32")
    blended = image_array.copy()
    foreground = mask_index > 0
    blended[foreground] = (
        image_array[foreground] * (1.0 - alpha) + color_mask[foreground].astype("float32") * alpha
    )
    return _pil().fromarray(blended.astype("uint8"))


def _resolve_device(requested: str, torch: Any) -> Any:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device is 'cuda', but CUDA is not available")
    return torch.device(requested)


def _palette(num_classes: int) -> list[int]:
    colors = []
    for class_id in range(max(num_classes, 1)):
        label = class_id
        r = g = b = 0
        for bit in range(8):
            r |= ((label >> 0) & 1) << (7 - bit)
            g |= ((label >> 1) & 1) << (7 - bit)
            b |= ((label >> 2) & 1) << (7 - bit)
            label >>= 3
        colors.extend([r, g, b])
    colors.extend([0] * (768 - len(colors)))
    return colors[:768]


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Prediction requires PyTorch and TorchVision. Install with `pip install -e .[torch]`."
        ) from exc
    return torch


def _pil() -> Any:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Prediction requires Pillow. Install with `pip install -e .[torch]`."
        ) from exc
    return Image
