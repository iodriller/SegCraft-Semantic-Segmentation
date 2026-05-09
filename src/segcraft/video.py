"""Small video helpers used by the examples."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from segcraft.data import list_image_files


def download_youtube(
    url: str,
    output_path: str | Path,
    *,
    format_selector: str = "mp4[height<=360]/mp4/best",
) -> Path:
    """Download a YouTube video with yt-dlp and return the output path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-f",
        format_selector,
        "-o",
        str(output_path),
        url,
    ]
    subprocess.run(command, check=True)
    return output_path


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    every_seconds: float = 1.0,
    max_frames: int | None = None,
    clear: bool = True,
) -> dict[str, Any]:
    """Extract frames from a video at a simple fixed interval."""
    cv2 = _cv2()
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if clear:
        for old_file in output_dir.glob("*.jpg"):
            old_file.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    step = max(int(round(fps * every_seconds)), 1)

    saved = 0
    index = 0
    while max_frames is None or saved < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            cv2.imwrite(str(output_dir / f"frame_{saved:04d}.jpg"), frame)
            saved += 1
        index += 1
    cap.release()

    return {
        "video_path": str(video_path),
        "frames_dir": str(output_dir),
        "fps": round(float(fps), 2),
        "frame_count": frame_count,
        "size": [width, height],
        "saved_frames": saved,
    }


def write_video_from_images(
    image_dir: str | Path,
    output_path: str | Path,
    *,
    fps: float = 6.0,
    codec: str | None = None,
    verify: bool = True,
) -> dict[str, Any]:
    """Write a video from a folder of sorted image files."""
    cv2 = _cv2()
    image_paths = list_image_files(image_dir)
    if not image_paths:
        raise ValueError(f"No images found under: {image_dir}")

    first = cv2.imread(str(image_paths[0]))
    if first is None:
        raise ValueError(f"Could not read image: {image_paths[0]}")

    height, width = first.shape[:2]
    if width % 2:
        width -= 1
    if height % 2:
        height -= 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chosen_codec = codec or _default_codec(output_path)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*chosen_codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path} with codec {chosen_codec}")

    try:
        for image_path in image_paths:
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise ValueError(f"Could not read image: {image_path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()

    if verify:
        _verify_video(output_path)

    return {
        "video_path": str(output_path),
        "frames": len(image_paths),
        "fps": fps,
        "size": [width, height],
        "codec": chosen_codec,
    }


def _default_codec(output_path: Path) -> str:
    if output_path.suffix.lower() == ".avi":
        return "MJPG"
    return "mp4v"


def _verify_video(path: Path) -> None:
    cv2 = _cv2()
    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if not cap.isOpened() or not ok or frame is None or frame_count < 1:
            raise RuntimeError(f"Video was written but could not be read back: {path}")
    finally:
        cap.release()


def _cv2() -> Any:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Video helpers require OpenCV. Install with `pip install -e .[video]`."
        ) from exc
    return cv2
