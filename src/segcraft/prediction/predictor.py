"""Image-folder prediction for semantic segmentation models."""

from __future__ import annotations

import json
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
        for frame_index, image_path in enumerate(image_paths):
            artifact = _predict_one(
                image_path=image_path,
                frame_index=frame_index,
                model=model,
                model_name=cfg.model.name,
                device=device,
                image_size=cfg.data.image_size,
                num_classes=cfg.task.num_classes,
                class_names=cfg.task.class_names,
                overlay_alpha=cfg.predict.overlay_alpha,
                annotate=cfg.predict.annotate,
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
    summary_path = output_dir / "summary.json"
    summary["summary_path"] = str(summary_path)
    full_summary = dict(summary)
    full_summary["outputs"] = artifacts
    summary_path.write_text(json.dumps(full_summary, indent=2), encoding="utf-8")
    return summary


def _predict_one(
    *,
    image_path: Path,
    frame_index: int,
    model: Any,
    model_name: str,
    device: Any,
    image_size: tuple[int, int],
    num_classes: int,
    class_names: list[str],
    overlay_alpha: float,
    annotate: bool,
    masks_dir: Path,
    overlays_dir: Path,
) -> dict[str, Any]:
    image = _pil().open(image_path).convert("RGB")
    original_size = image.size
    batch = _preprocess(image, image_size).to(device)
    output = model(batch)
    logits = output["out"] if isinstance(output, dict) else output
    prediction = logits.argmax(dim=1)[0].detach().cpu().numpy().astype("uint8")

    mask = _pil().fromarray(prediction).resize(original_size, resample=_pil().NEAREST)
    mask_index = _np().asarray(mask).astype("int64")
    classes = _class_summary(mask_index, class_names, max_items=5)
    mask.putpalette(_palette(num_classes))
    mask_path = masks_dir / f"{image_path.stem}.png"
    mask.save(mask_path)

    overlay = _overlay(image, mask_index, num_classes=num_classes, alpha=overlay_alpha)
    if annotate:
        overlay = _annotate_overlay(
            overlay,
            frame_index=frame_index,
            image_name=image_path.name,
            model_name=model_name,
            classes=classes,
            num_classes=num_classes,
        )
    overlay_path = overlays_dir / f"{image_path.stem}.jpg"
    overlay.save(overlay_path, quality=92)

    return {
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "classes": classes,
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


def _overlay(image: Any, mask_index: Any, *, num_classes: int, alpha: float) -> Any:
    colors = _np().asarray(_palette(num_classes), dtype="uint8").reshape(-1, 3)
    color_mask = colors[mask_index % len(colors)]
    image_array = _np().asarray(image).astype("float32")
    blended = image_array.copy()
    foreground = mask_index > 0
    blended[foreground] = (
        image_array[foreground] * (1.0 - alpha) + color_mask[foreground].astype("float32") * alpha
    )
    return _pil().fromarray(blended.astype("uint8"))


def _annotate_overlay(
    image: Any,
    *,
    frame_index: int,
    image_name: str,
    model_name: str,
    classes: list[dict[str, Any]],
    num_classes: int,
) -> Any:
    from PIL import ImageDraw, ImageFont

    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    width, height = canvas.size
    panel_height = 18 + 15 * max(len(classes), 1)
    panel_width = min(max(250, int(width * 0.46)), width - 8)
    x0 = 8
    y0 = max(8, height - panel_height - 8)
    draw.rounded_rectangle(
        (x0, y0, x0 + panel_width, y0 + panel_height),
        radius=4,
        fill=(0, 0, 0, 150),
    )
    title = f"SegCraft | {model_name} | frame {frame_index:04d}"
    draw.text((x0 + 8, y0 + 6), title[:58], fill=(255, 255, 255, 255), font=font)
    draw.text((x0 + 8, y0 + 20), image_name[:58], fill=(210, 210, 210, 255), font=font)

    if not classes:
        draw.text((x0 + 8, y0 + 36), "classes: background only", fill=(230, 230, 230, 255), font=font)
        return canvas.convert("RGB")

    palette = _palette(num_classes)
    for row, item in enumerate(classes):
        y = y0 + 36 + row * 15
        color = _palette_color(palette, item["class_id"])
        draw.rectangle((x0 + 8, y + 2, x0 + 18, y + 12), fill=tuple(color) + (255,))
        text = f"{item['name']}: {item['percent']:.1f}%"
        draw.text((x0 + 24, y), text[:44], fill=(245, 245, 245, 255), font=font)
    return canvas.convert("RGB")


def _class_summary(mask_index: Any, class_names: list[str], *, max_items: int) -> list[dict[str, Any]]:
    values, counts = _np().unique(mask_index, return_counts=True)
    total = max(int(mask_index.size), 1)
    rows = []
    for value, count in zip(values.tolist(), counts.tolist()):
        class_id = int(value)
        if class_id == 0:
            continue
        rows.append(
            {
                "class_id": class_id,
                "name": class_names[class_id] if class_id < len(class_names) else f"class_{class_id}",
                "pixels": int(count),
                "percent": round((int(count) / total) * 100.0, 3),
            }
        )
    rows.sort(key=lambda item: item["pixels"], reverse=True)
    return rows[:max_items]


def _palette_color(palette: list[int], class_id: int) -> list[int]:
    offset = (class_id % 256) * 3
    return palette[offset : offset + 3]


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


def _np() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Prediction requires NumPy. Install with `pip install -e .[torch]`."
        ) from exc
    return np
