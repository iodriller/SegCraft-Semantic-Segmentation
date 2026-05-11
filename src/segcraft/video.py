"""Small video helpers used by the examples."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from segcraft.data import list_image_files
from segcraft.runtime import INSTALL_HINTS


VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
DEFAULT_YOUTUBE_FORMAT = "mp4[height<=360]/mp4/best"


class DownloadCancelled(RuntimeError):
    """Raised when a cancellable download is stopped by the caller."""


def download_youtube(
    url: str,
    output_path: str | Path,
    *,
    format_selector: str = DEFAULT_YOUTUBE_FORMAT,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    should_stop: Callable[[], bool] | None = None,
) -> Path:
    """Download a YouTube video with yt-dlp and return the output path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {"url": url, "format_selector": format_selector}
    output_metadata_path = _download_metadata_path(output_path)
    if use_cache and _download_matches(output_path, output_metadata_path, metadata):
        return output_path

    cache_path = None
    cache_metadata_path = None
    if use_cache and cache_dir is not None:
        cache_path = _cached_download_path(cache_dir, url, format_selector, output_path.suffix or ".mp4")
        cache_metadata_path = _download_metadata_path(cache_path)
        if _download_matches(cache_path, cache_metadata_path, metadata):
            shutil.copy2(cache_path, output_path)
            _write_json(output_metadata_path, metadata)
            return output_path

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
    _run_download(command, should_stop=should_stop)
    _write_json(output_metadata_path, metadata)

    if cache_path is not None and cache_metadata_path is not None and _usable_file(output_path):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, cache_path)
        _write_json(cache_metadata_path, metadata)

    return output_path


def _run_download(command: list[str], *, should_stop: Callable[[], bool] | None = None) -> None:
    if should_stop is None:
        subprocess.run(command, check=True)
        return

    process = subprocess.Popen(command)
    while True:
        return_code = process.poll()
        if return_code is not None:
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)
            return
        if should_stop():
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover - depends on yt-dlp process state
                process.kill()
                process.wait()
            raise DownloadCancelled("Download was canceled.")
        time.sleep(0.5)


def _cached_download_path(cache_dir: str | Path, url: str, format_selector: str, suffix: str) -> Path:
    key = hashlib.sha256(f"{url}\n{format_selector}".encode("utf-8")).hexdigest()[:24]
    safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return Path(cache_dir) / f"{key}{safe_suffix}"


def _download_metadata_path(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.name}.segcraft-download.json")


def _download_matches(video_path: Path, metadata_path: Path, expected: dict[str, str]) -> bool:
    if not _usable_file(video_path) or not metadata_path.exists():
        return False
    try:
        actual = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(actual.get(key) == value for key, value in expected.items())


def _usable_file(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def is_video_file(path: str | Path) -> bool:
    """Return true when a path looks like a video file SegCraft can stream."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def probe_video(video_path: str | Path) -> dict[str, Any]:
    """Read basic video metadata with OpenCV."""
    cv2 = _cv2()
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()

    duration_seconds = (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0
    return {
        "video_path": str(video_path),
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "duration_seconds": round(duration_seconds, 3),
        "size": [width, height],
        "has_audio": _has_audio(video_path),
    }


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


def mux_audio_from_source(
    source_video: str | Path,
    silent_video: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Copy source audio into an overlay video when ffmpeg is available."""
    source_video = Path(source_video)
    silent_video = Path(silent_video)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        _replace_file(silent_video, output_path)
        return {
            "status": "ffmpeg_missing",
            "preserved": False,
            "message": "ffmpeg was not found; wrote the overlay video without source audio.",
        }

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(silent_video),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not output_path.exists():
        _replace_file(silent_video, output_path)
        return {
            "status": "mux_failed",
            "preserved": False,
            "message": "ffmpeg could not mux source audio; wrote the overlay video without audio.",
        }

    try:
        silent_video.unlink()
    except FileNotFoundError:
        pass

    preserved = _has_audio(output_path)
    message = (
        "Source audio was muxed into the overlay video."
        if preserved
        else "No readable source audio stream was found; wrote the overlay video without audio."
    )
    return {
        "status": "completed",
        "preserved": preserved,
        "message": message,
    }


def copy_video_file(
    source_video: str | Path,
    output_path: str | Path,
    *,
    verify: bool = True,
) -> dict[str, Any]:
    """Copy the source video next to prediction outputs for easy comparison."""
    source_video = Path(source_video)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if source_video.resolve() != output_path.resolve():
        shutil.copy2(source_video, output_path)

    if verify:
        _verify_video(output_path)

    metadata = probe_video(output_path)
    metadata["source_path"] = str(source_video)
    return metadata


def write_side_by_side_video(
    left_video: str | Path,
    right_video: str | Path,
    output_path: str | Path,
    *,
    fps: float | None = None,
    codec: str | None = None,
    left_label: str = "Original",
    right_label: str = "SegCraft overlay",
    verify: bool = True,
) -> dict[str, Any]:
    """Write a side-by-side comparison video from two readable video files."""
    cv2 = _cv2()
    left_video = Path(left_video)
    right_video = Path(right_video)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    left = cv2.VideoCapture(str(left_video))
    right = cv2.VideoCapture(str(right_video))
    if not left.isOpened():
        raise FileNotFoundError(f"Could not open video: {left_video}")
    if not right.isOpened():
        left.release()
        raise FileNotFoundError(f"Could not open video: {right_video}")

    chosen_fps = fps or right.get(cv2.CAP_PROP_FPS) or left.get(cv2.CAP_PROP_FPS) or 6.0
    ok_left, left_frame = left.read()
    ok_right, right_frame = right.read()
    if not ok_left or left_frame is None or not ok_right or right_frame is None:
        left.release()
        right.release()
        raise RuntimeError("Could not read the first frames for comparison video")

    target_height, target_width = right_frame.shape[:2]
    target_width, target_height = _even_size(target_width, target_height)
    output_size = (target_width * 2, target_height)
    chosen_codec = codec or _default_codec(output_path)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*chosen_codec),
        chosen_fps,
        output_size,
    )
    if not writer.isOpened():
        left.release()
        right.release()
        raise RuntimeError(f"Could not open video writer for {output_path} with codec {chosen_codec}")

    frames = 0
    try:
        while ok_left and ok_right:
            left_prepared = _prepare_comparison_frame(left_frame, target_width, target_height)
            right_prepared = _prepare_comparison_frame(right_frame, target_width, target_height)
            _draw_video_label(left_prepared, left_label)
            _draw_video_label(right_prepared, right_label)
            writer.write(cv2.hconcat([left_prepared, right_prepared]))
            frames += 1
            ok_left, left_frame = left.read()
            ok_right, right_frame = right.read()
    finally:
        writer.release()
        left.release()
        right.release()

    if frames < 1:
        raise RuntimeError(f"No frames were written to comparison video: {output_path}")
    if verify:
        _verify_video(output_path)

    return {
        "video_path": str(output_path),
        "frames": frames,
        "fps": float(chosen_fps),
        "duration_seconds": round(frames / float(chosen_fps), 3) if chosen_fps else None,
        "size": [output_size[0], output_size[1]],
        "codec": chosen_codec,
        "left_video": str(left_video),
        "right_video": str(right_video),
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


def verify_video(path: str | Path) -> None:
    """Raise when a written video cannot be read back."""
    _verify_video(Path(path))


def _replace_file(source: Path, target: Path) -> None:
    if source.resolve() == target.resolve():
        return
    if target.exists():
        target.unlink()
    source.replace(target)


def _even_size(width: int, height: int) -> tuple[int, int]:
    width = max(width - (width % 2), 2)
    height = max(height - (height % 2), 2)
    return width, height


def _prepare_comparison_frame(frame: Any, width: int, height: int) -> Any:
    cv2 = _cv2()
    if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height))
    if width % 2 or height % 2:
        frame = frame[: height - (height % 2), : width - (width % 2)]
    return frame


def _draw_video_label(frame: Any, label: str) -> None:
    cv2 = _cv2()
    x0, y0 = 10, 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    cv2.rectangle(
        frame,
        (x0 - 4, y0 - 4),
        (x0 + text_width + 8, y0 + text_height + baseline + 8),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x0, y0 + text_height + 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _has_audio(video_path: Path) -> bool | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None

    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return None
    return "audio" in result.stdout.lower()


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
            f"Video helpers require OpenCV. {INSTALL_HINTS['video']}"
        ) from exc
    return cv2
