"""Prediction workflows for images and videos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from segcraft.config import SegCraftConfig, parse_config
from segcraft.data import list_image_files
from segcraft.models import create_model
from segcraft.video import (
    copy_video_file,
    is_video_file,
    mux_audio_from_source,
    probe_video,
    verify_video,
    write_side_by_side_video,
    write_video_from_images,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def run_prediction(config: Mapping[str, Any] | SegCraftConfig) -> dict[str, Any]:
    cfg = parse_config(config)
    input_path = Path(cfg.predict.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Prediction input path does not exist: {input_path}")
    if is_video_file(input_path):
        return _run_video_prediction(cfg, input_path)
    return _run_image_prediction(cfg, input_path)


def _run_image_prediction(cfg: SegCraftConfig, input_path: Path) -> dict[str, Any]:
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
    class_totals: dict[int, dict[str, Any]] = {}
    with torch.inference_mode():
        for frame_index, image_path in enumerate(image_paths):
            artifact = _predict_one(
                image_path=image_path,
                frame_index=frame_index,
                model=model,
                model_name=cfg.model.name,
                device=device,
                image_size=cfg.data.image_size,
                task_type=cfg.task.type,
                num_classes=cfg.task.num_classes,
                class_names=cfg.task.class_names,
                overlay_alpha=cfg.predict.overlay_alpha,
                annotate=cfg.predict.annotate,
                masks_dir=masks_dir,
                overlays_dir=overlays_dir,
            )
            artifacts.append(artifact)
            _update_class_totals(class_totals, artifact["classes"], artifact["total_pixels"])

    summary = {
        "status": "completed",
        "input_type": "images",
        "device": str(device),
        "images_found": len(image_paths),
        "images_processed": len(artifacts),
        "masks_dir": str(masks_dir),
        "overlays_dir": str(overlays_dir),
        "class_summary": _finalize_class_totals(class_totals),
        "sample_outputs": artifacts[:5],
    }
    if cfg.predict.save_video:
        video_path = _prediction_video_path(cfg, output_dir)
        summary["overlay_video"] = write_video_from_images(
            overlays_dir,
            video_path,
            fps=cfg.predict.video_fps,
        )
    _write_prediction_summary(output_dir, summary, outputs=artifacts)
    return summary


def _run_video_prediction(cfg: SegCraftConfig, input_path: Path) -> dict[str, Any]:
    output_dir = Path(cfg.predict.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch = _torch()
    device = _resolve_device(cfg.runtime.device, torch)
    model = create_model(cfg.model, cfg.task).to(device)
    model.eval()

    cv2 = _cv2()
    video_info = probe_video(input_path)
    fps = float(video_info["fps"] or cfg.predict.video_fps)
    source_width, source_height = video_info["size"]
    video_path = _prediction_video_path(cfg, output_dir)
    original_video = copy_video_file(input_path, _original_video_path(input_path, output_dir))

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")
    if source_width <= 0 or source_height <= 0:
        ok, first_frame = cap.read()
        if not ok:
            cap.release()
            raise ValueError(f"No frames could be read from video: {input_path}")
        source_height, source_width = first_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    output_width, output_height = _even_size(source_width, source_height)

    writer = None
    write_target = _silent_video_path(video_path) if cfg.predict.preserve_audio else video_path
    codec = _video_codec(video_path)
    if cfg.predict.save_video:
        write_target.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(write_target),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (output_width, output_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {write_target} with codec {codec}")

    samples = []
    class_totals: dict[int, dict[str, Any]] = {}
    frames_processed = 0
    try:
        with torch.inference_mode():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                image = _pil().fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = _predict_image(
                    image=image,
                    frame_index=frames_processed,
                    image_name=input_path.name,
                    model=model,
                    model_name=cfg.model.name,
                    device=device,
                    image_size=cfg.data.image_size,
                    task_type=cfg.task.type,
                    num_classes=cfg.task.num_classes,
                    class_names=cfg.task.class_names,
                    overlay_alpha=cfg.predict.overlay_alpha,
                    annotate=cfg.predict.annotate,
                )

                if writer is not None:
                    overlay = _np().asarray(result["overlay"])
                    overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                    if overlay.shape[:2] != (output_height, output_width):
                        overlay = cv2.resize(overlay, (output_width, output_height))
                    writer.write(overlay)

                if len(samples) < 5:
                    samples.append(
                        {
                            "frame_index": frames_processed,
                            "classes": result["classes"],
                        }
                    )
                _update_class_totals(class_totals, result["classes"], result["total_pixels"])
                frames_processed += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if frames_processed == 0:
        raise ValueError(f"No frames could be read from video: {input_path}")

    overlay_video = None
    comparison_video = None
    audio = {"status": "disabled", "preserved": False}
    if cfg.predict.save_video:
        if cfg.predict.preserve_audio:
            audio = mux_audio_from_source(input_path, write_target, video_path)
        else:
            verify_video(video_path)
        verify_video(video_path)
        overlay_video = {
            "video_path": str(video_path),
            "frames": frames_processed,
            "fps": fps,
            "duration_seconds": round(frames_processed / fps, 3) if fps > 0 else None,
            "size": [output_width, output_height],
            "codec": codec,
            "audio": audio,
        }
        comparison_video = write_side_by_side_video(
            original_video["video_path"],
            video_path,
            output_dir / "comparison.mp4",
            fps=fps,
            codec=codec,
            left_label="Original dashcam",
            right_label="Semantic segmentation overlay",
        )

    summary = {
        "status": "completed",
        "input_type": "video",
        "device": str(device),
        "source_video": video_info,
        "original_video": original_video,
        "frames_processed": frames_processed,
        "class_summary": _finalize_class_totals(class_totals),
        "sample_outputs": samples,
        "overlay_video": overlay_video,
        "comparison_video": comparison_video,
    }
    _write_prediction_summary(output_dir, summary, outputs=samples)
    return summary


def _predict_one(
    *,
    image_path: Path,
    frame_index: int,
    model: Any,
    model_name: str,
    device: Any,
    image_size: tuple[int, int],
    task_type: str,
    num_classes: int,
    class_names: list[str],
    overlay_alpha: float,
    annotate: bool,
    masks_dir: Path,
    overlays_dir: Path,
) -> dict[str, Any]:
    image = _pil().open(image_path).convert("RGB")
    result = _predict_image(
        image=image,
        frame_index=frame_index,
        image_name=image_path.name,
        model=model,
        model_name=model_name,
        device=device,
        image_size=image_size,
        task_type=task_type,
        num_classes=num_classes,
        class_names=class_names,
        overlay_alpha=overlay_alpha,
        annotate=annotate,
    )

    mask = result["mask"]
    mask.putpalette(_palette(_mask_class_count(task_type, num_classes)))
    mask_path = masks_dir / f"{image_path.stem}.png"
    mask.save(mask_path)

    overlay_path = overlays_dir / f"{image_path.stem}.jpg"
    result["overlay"].save(overlay_path, quality=92)

    return {
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "classes": result["classes"],
        "total_pixels": result["total_pixels"],
    }


def _predict_image(
    *,
    image: Any,
    frame_index: int,
    image_name: str,
    model: Any,
    model_name: str,
    device: Any,
    image_size: tuple[int, int],
    task_type: str,
    num_classes: int,
    class_names: list[str],
    overlay_alpha: float,
    annotate: bool,
) -> dict[str, Any]:
    original_size = image.size
    batch = _preprocess(image, image_size).to(device)
    output = model(batch)
    logits = output["out"] if isinstance(output, dict) else output
    prediction, confidence_by_class = _prediction_from_logits(logits, task_type)

    mask = _pil().fromarray(prediction).resize(original_size, resample=_pil().NEAREST)
    mask_index = _np().asarray(mask).astype("int64")
    class_count = _mask_class_count(task_type, num_classes)
    classes = _class_summary(
        mask_index,
        class_names,
        confidence_by_class=confidence_by_class,
        max_items=5,
    )

    overlay = _overlay(image, mask_index, num_classes=class_count, alpha=overlay_alpha)
    if annotate:
        overlay = _annotate_overlay(
            overlay,
            frame_index=frame_index,
            image_name=image_name,
            model_name=model_name,
            classes=classes,
            num_classes=class_count,
        )

    return {
        "mask": mask,
        "overlay": overlay,
        "classes": classes,
        "total_pixels": int(mask_index.size),
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


def _prediction_from_logits(logits: Any, task_type: str) -> tuple[Any, dict[int, dict[str, float]]]:
    torch = _torch()
    if task_type == "binary":
        foreground = torch.sigmoid(logits[:, 0])[0].detach().cpu().numpy()
        prediction = (foreground >= 0.5).astype("uint8")
        values = foreground[prediction == 1]
        confidence = {1: _confidence_stats(values)} if values.size else {}
        return prediction, confidence

    probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    prediction = probabilities.argmax(axis=0).astype("uint8")
    confidence = {}
    for class_id in _np().unique(prediction).tolist():
        class_id = int(class_id)
        if class_id == 0 or class_id >= probabilities.shape[0]:
            continue
        values = probabilities[class_id][prediction == class_id]
        if values.size:
            confidence[class_id] = _confidence_stats(values)
    return prediction, confidence


def _confidence_stats(values: Any) -> dict[str, float]:
    return {
        "mean_confidence": round(float(values.mean()), 4),
        "max_confidence": round(float(values.max()), 4),
    }


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
    panel_height = 42 + 16 * max(len(classes), 1)
    panel_width = min(max(300, int(width * 0.52)), width - 8)
    x0 = 8
    y0 = max(8, height - panel_height - 8)
    draw.rounded_rectangle(
        (x0, y0, x0 + panel_width, y0 + panel_height),
        radius=4,
        fill=(0, 0, 0, 150),
    )
    title = f"SegCraft | {model_name} | frame {frame_index:04d}"
    draw.text((x0 + 8, y0 + 6), title[:66], fill=(255, 255, 255, 255), font=font)
    draw.text((x0 + 8, y0 + 20), image_name[:66], fill=(210, 210, 210, 255), font=font)

    if not classes:
        draw.text((x0 + 8, y0 + 38), "classes: background only", fill=(230, 230, 230, 255), font=font)
        return canvas.convert("RGB")

    palette = _palette(num_classes)
    for row, item in enumerate(classes):
        y = y0 + 38 + row * 16
        color = _palette_color(palette, item["class_id"])
        draw.rectangle((x0 + 8, y + 2, x0 + 18, y + 12), fill=tuple(color) + (255,))
        confidence = item.get("mean_confidence")
        suffix = f" | conf {confidence:.2f}" if confidence is not None else ""
        text = f"{item['name']}: {item['percent']:.1f}%{suffix}"
        draw.text((x0 + 24, y), text[:52], fill=(245, 245, 245, 255), font=font)
    return canvas.convert("RGB")


def _class_summary(
    mask_index: Any,
    class_names: list[str],
    *,
    confidence_by_class: dict[int, dict[str, float]],
    max_items: int,
) -> list[dict[str, Any]]:
    values, counts = _np().unique(mask_index, return_counts=True)
    total = max(int(mask_index.size), 1)
    rows = []
    for value, count in zip(values.tolist(), counts.tolist()):
        class_id = int(value)
        if class_id == 0:
            continue
        row = {
            "class_id": class_id,
            "name": _class_name(class_id, class_names),
            "pixels": int(count),
            "percent": round((int(count) / total) * 100.0, 3),
        }
        row.update(confidence_by_class.get(class_id, {}))
        rows.append(row)
    rows.sort(key=lambda item: item["pixels"], reverse=True)
    return rows[:max_items]


def _update_class_totals(
    totals: dict[int, dict[str, Any]],
    classes: list[dict[str, Any]],
    total_pixels: int,
) -> None:
    totals.setdefault(0, {"total_pixels": 0})["total_pixels"] += total_pixels
    for item in classes:
        class_id = int(item["class_id"])
        row = totals.setdefault(
            class_id,
            {
                "class_id": class_id,
                "name": item["name"],
                "pixels": 0,
                "confidence_pixels": 0,
                "confidence_sum": 0.0,
                "max_confidence": None,
            },
        )
        pixels = int(item["pixels"])
        row["pixels"] += pixels
        if "mean_confidence" in item:
            row["confidence_pixels"] += pixels
            row["confidence_sum"] += float(item["mean_confidence"]) * pixels
            max_confidence = float(item.get("max_confidence", item["mean_confidence"]))
            if row["max_confidence"] is None or max_confidence > row["max_confidence"]:
                row["max_confidence"] = max_confidence


def _finalize_class_totals(totals: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    total_pixels = max(int(totals.get(0, {}).get("total_pixels", 0)), 1)
    rows = []
    for class_id, item in totals.items():
        if class_id == 0:
            continue
        row = {
            "class_id": class_id,
            "name": item["name"],
            "pixels": int(item["pixels"]),
            "percent": round((int(item["pixels"]) / total_pixels) * 100.0, 3),
        }
        if item["confidence_pixels"]:
            row["mean_confidence"] = round(
                item["confidence_sum"] / item["confidence_pixels"],
                4,
            )
            row["max_confidence"] = round(float(item["max_confidence"]), 4)
        rows.append(row)
    rows.sort(key=lambda item: item["pixels"], reverse=True)
    return rows


def _class_name(class_id: int, class_names: list[str]) -> str:
    if class_id < len(class_names):
        return class_names[class_id]
    if class_id == 1:
        return "foreground"
    return f"class_{class_id}"


def _palette_color(palette: list[int], class_id: int) -> list[int]:
    offset = (class_id % 256) * 3
    return palette[offset : offset + 3]


def _prediction_video_path(cfg: SegCraftConfig, output_dir: Path) -> Path:
    video_path = Path(cfg.predict.video_path) if cfg.predict.video_path else output_dir / "overlay.mp4"
    if not video_path.suffix:
        video_path = video_path.with_suffix(".mp4")
    return video_path


def _original_video_path(input_path: Path, output_dir: Path) -> Path:
    suffix = input_path.suffix if input_path.suffix else ".mp4"
    return output_dir / f"original{suffix.lower()}"


def _silent_video_path(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}.no_audio{video_path.suffix}")


def _video_codec(video_path: Path) -> str:
    if video_path.suffix.lower() == ".avi":
        return "MJPG"
    return "mp4v"


def _even_size(width: int, height: int) -> tuple[int, int]:
    width = max(width - (width % 2), 2)
    height = max(height - (height % 2), 2)
    return width, height


def _mask_class_count(task_type: str, num_classes: int) -> int:
    return 2 if task_type == "binary" else num_classes


def _write_prediction_summary(
    output_dir: Path,
    summary: dict[str, Any],
    *,
    outputs: list[dict[str, Any]],
) -> None:
    summary_path = output_dir / "summary.json"
    summary["summary_path"] = str(summary_path)
    full_summary = dict(summary)
    full_summary["outputs"] = outputs
    summary_path.write_text(json.dumps(full_summary, indent=2), encoding="utf-8")


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


def _cv2() -> Any:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Video prediction requires OpenCV. Install with `pip install -e .[video]`."
        ) from exc
    return cv2
