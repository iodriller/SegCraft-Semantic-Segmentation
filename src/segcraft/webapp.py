"""Minimal FastAPI app for running SegCraft video prediction jobs."""

import copy
from html import escape
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

from segcraft.config.loader import list_available_presets, load_and_validate_config
from segcraft.cli.main import resolve_config_path
from segcraft.prediction import run_prediction
from segcraft.runtime import INSTALL_HINTS, collect_runtime_diagnostics
from segcraft.video import download_youtube


JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.RLock()
DEFAULT_WORK_DIR = Path("outputs/web")
DOWNLOADABLE_FILES = ("comparison.mp4", "overlay.mp4", "original.mp4", "summary.json")


def create_app():
    try:
        from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.responses import FileResponse, HTMLResponse
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"The web app requires optional dependencies. {INSTALL_HINTS['app']}"
        ) from exc

    app = FastAPI(title="SegCraft")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _index_html()

    @app.get("/runtime")
    def runtime() -> dict[str, Any]:
        return collect_runtime_diagnostics()

    @app.post("/jobs")
    async def create_job(
        background_tasks: BackgroundTasks,
        video_file: UploadFile | None = File(default=None),
        youtube_url: str = Form(default=""),
        config_path: str = Form(default=""),
        preset_name: str = Form(default="pascal_video"),
        preset_path: str = Form(default=""),
        device: str = Form(default="auto"),
        image_height: int = Form(default=360),
        image_width: int = Form(default=640),
        max_seconds: float = Form(default=30.0),
        frame_stride: int = Form(default=2),
        preserve_audio: bool = Form(default=True),
    ) -> dict[str, str]:
        if video_file is None and not youtube_url.strip():
            raise HTTPException(status_code=400, detail="Upload a video or provide a YouTube URL.")

        job_id = uuid.uuid4().hex[:12]
        job_dir = DEFAULT_WORK_DIR / "jobs" / job_id
        input_dir = job_dir / "input"
        output_dir = job_dir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        input_path = None
        if video_file is not None and video_file.filename:
            safe_name = Path(video_file.filename).name
            input_path = input_dir / safe_name
            with input_path.open("wb") as handle:
                shutil.copyfileobj(video_file.file, handle)

        params = {
            "job_id": job_id,
            "job_dir": str(job_dir),
            "input_path": str(input_path) if input_path else None,
            "youtube_url": youtube_url.strip() or None,
            "download_path": str(input_dir / "youtube.mp4"),
            "output_dir": str(output_dir),
            "config_path": config_path.strip() or None,
            "preset_name": preset_name.strip() or None,
            "preset_path": preset_path.strip() or None,
            "device": device,
            "image_size": [max(int(image_height), 1), max(int(image_width), 1)],
            "max_seconds": max(float(max_seconds), 0.1),
            "frame_stride": max(int(frame_stride), 1),
            "preserve_audio": bool(preserve_audio),
        }
        _set_job(
            job_id,
            id=job_id,
            status="queued",
            progress={"stage": "queued", "completed": 0, "total": None, "percent": 0, "message": "Queued"},
            output_dir=str(output_dir),
            downloads={},
        )
        background_tasks.add_task(_run_job, params)
        return {"job_id": job_id}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = _get_job(job_id)
        job["downloads"] = _download_links(job_id)
        return job

    @app.get("/jobs/{job_id}/download/{filename}")
    def download(job_id: str, filename: str):
        if filename not in DOWNLOADABLE_FILES:
            raise HTTPException(status_code=404, detail="Unknown artifact")
        job = _get_job(job_id)
        output_dir = Path(job["output_dir"])
        path = output_dir / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail="Artifact not ready")
        return FileResponse(path, filename=filename)

    return app


def _run_job(params: dict[str, Any]) -> None:
    job_id = params["job_id"]
    try:
        _set_job(job_id, status="running")
        input_path = params.get("input_path")
        if params.get("youtube_url"):
            _set_job(
                job_id,
                progress={
                    "stage": "downloading",
                    "completed": 0,
                    "total": None,
                    "percent": 0,
                    "message": "Downloading source video",
                },
            )
            input_path = str(download_youtube(params["youtube_url"], params["download_path"]))

        cfg = _job_config(params, input_path)
        summary = run_prediction(cfg, progress_callback=lambda event: _set_job(job_id, progress=event))
        _set_job(
            job_id,
            status="completed",
            summary=summary,
            progress={
                "stage": "completed",
                "completed": summary.get("frames_processed", summary.get("images_processed", 1)),
                "total": summary.get("frames_processed", summary.get("images_processed", 1)),
                "percent": 100,
                "message": "Prediction complete",
            },
        )
    except Exception as exc:  # pragma: no cover - exercised by real app failures
        _set_job(
            job_id,
            status="failed",
            error=str(exc),
            progress={
                "stage": "failed",
                "completed": 0,
                "total": None,
                "percent": 0,
                "message": str(exc),
            },
        )


def _job_config(params: dict[str, Any], input_path: str | None) -> dict[str, Any]:
    if input_path is None:
        raise ValueError("No input video was provided")

    config_path = params.get("config_path")
    with resolve_config_path(Path(config_path) if config_path else None) as base_config:
        cfg = load_and_validate_config(
            base_config,
            preset_path=_resolve_preset(params),
            local_path=None,
        )

    cfg = copy.deepcopy(cfg)
    cfg.setdefault("data", {})["image_size"] = params["image_size"]
    cfg.setdefault("predict", {}).update(
        {
            "input_path": input_path,
            "output_path": params["output_dir"],
            "video_max_seconds": params["max_seconds"],
            "video_frame_stride": params["frame_stride"],
            "preserve_audio": params["preserve_audio"],
            "save_video": True,
        }
    )
    cfg.setdefault("runtime", {})["device"] = params["device"]
    return cfg


def _resolve_preset(params: dict[str, Any]) -> str | None:
    return params.get("preset_path") or params.get("preset_name")


def _set_job(job_id: str, **updates: Any) -> None:
    with JOBS_LOCK:
        job = JOBS.setdefault(job_id, {"id": job_id})
        job.update(updates)


def _get_job(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        if job_id not in JOBS:
            try:
                from fastapi import HTTPException
            except ModuleNotFoundError:
                raise KeyError(job_id)
            raise HTTPException(status_code=404, detail="Unknown job")
        return copy.deepcopy(JOBS[job_id])


def _download_links(job_id: str) -> dict[str, str]:
    job = _get_job(job_id)
    output_dir = Path(job["output_dir"])
    links = {}
    for filename in DOWNLOADABLE_FILES:
        if (output_dir / filename).exists():
            links[filename] = f"/jobs/{job_id}/download/{filename}"
    return links


def _index_html() -> str:
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SegCraft</title>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f7f7f4; color: #181a1f; }
    main { max-width: 1040px; margin: 0 auto; padding: 28px 20px 44px; }
    h1 { font-size: 32px; line-height: 1.1; margin: 0 0 6px; letter-spacing: 0; }
    h2 { font-size: 16px; margin: 0 0 12px; letter-spacing: 0; }
    p { color: #5a5f68; line-height: 1.5; }
    form, .panel { background: #fff; border: 1px solid #d9ded6; border-radius: 8px; padding: 18px; box-shadow: 0 1px 2px rgba(20,24,20,.04); }
    label { display: grid; gap: 6px; color: #333945; font-size: 14px; font-weight: 600; }
    input, select { border: 1px solid #c7cec6; border-radius: 6px; padding: 10px 11px; font: inherit; min-width: 0; background: #fff; }
    .grid { display: grid; gap: 14px; grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .source-grid { display: grid; gap: 14px; grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .source-box { border: 1px solid #d9ded6; border-radius: 8px; padding: 14px; background: #fbfbf8; }
    .full { grid-column: 1 / -1; }
    .hint { color: #6d736c; font-size: 13px; font-weight: 500; }
    .runtime { margin: 0 0 16px; font-size: 14px; }
    button { border: 0; border-radius: 6px; background: #176a5f; color: white; padding: 11px 14px; font: inherit; font-weight: 700; cursor: pointer; }
    button:disabled { background: #8a9a95; cursor: wait; }
    .bar { height: 12px; background: #e7eae4; border-radius: 999px; overflow: hidden; }
    .fill { width: 0%; height: 100%; background: #cf8a28; transition: width .25s ease; }
    .row { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .downloads { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .downloads a { color: #176a5f; border: 1px solid #bad5cb; border-radius: 6px; padding: 8px 10px; text-decoration: none; font-weight: 600; }
    @media (max-width: 720px) { .grid, .source-grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <h1>SegCraft Video</h1>
    <p id="runtime" class="runtime">Checking runtime...</p>
    <form id="job-form" class="grid">
      <section class="full">
        <h2>Source</h2>
        <div class="source-grid">
          <label class="source-box">Upload video <input name="video_file" type="file" accept="video/*"><span class="hint">Use this for a local MP4, MOV, AVI, or similar file.</span></label>
          <label class="source-box">YouTube URL <input name="youtube_url" type="url" placeholder="https://www.youtube.com/watch?v=..."><span class="hint">Use this instead of a file upload.</span></label>
        </div>
      </section>
      <label>Preset <select name="preset_name">__PRESET_OPTIONS__</select></label>
      <label>Custom preset <input name="preset_path" type="text" placeholder="cityscapes_video or configs/presets/quality.yaml"></label>
      <label>Device <select name="device"><option>auto</option><option>cpu</option><option>cuda</option></select></label>
      <label>Max seconds <input name="max_seconds" type="number" min="1" step="1" value="30"></label>
      <label>Frame stride <input name="frame_stride" type="number" min="1" step="1" value="2"></label>
      <label>Image height <input name="image_height" type="number" min="32" step="1" value="360"></label>
      <label>Image width <input name="image_width" type="number" min="32" step="1" value="640"></label>
      <label>Config path <input name="config_path" type="text" placeholder="configs/base.yaml or leave blank"></label>
      <label><span>Preserve audio</span><select name="preserve_audio"><option value="true">true</option><option value="false">false</option></select></label>
      <div class="full"><button id="submit" type="submit">Start job</button></div>
    </form>
    <section class="panel" style="margin-top:16px">
      <div class="row"><strong id="status">Idle</strong><span id="percent">0%</span></div>
      <div class="bar" style="margin-top:12px"><div id="fill" class="fill"></div></div>
      <p id="message">No job running.</p>
      <div id="downloads" class="downloads"></div>
    </section>
  </main>
  <script>
    const form = document.getElementById('job-form');
    const submit = document.getElementById('submit');
    const statusEl = document.getElementById('status');
    const percentEl = document.getElementById('percent');
    const fillEl = document.getElementById('fill');
    const messageEl = document.getElementById('message');
    const downloadsEl = document.getElementById('downloads');
    const runtimeEl = document.getElementById('runtime');
    let timer = null;

    loadRuntime();

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      submit.disabled = true;
      downloadsEl.innerHTML = '';
      const response = await fetch('/jobs', { method: 'POST', body: new FormData(form) });
      if (!response.ok) {
        messageEl.textContent = await response.text();
        submit.disabled = false;
        return;
      }
      const payload = await response.json();
      poll(payload.job_id);
      timer = setInterval(() => poll(payload.job_id), 1000);
    });

    async function poll(jobId) {
      const response = await fetch(`/jobs/${jobId}`);
      const job = await response.json();
      const progress = job.progress || {};
      const percent = progress.percent ?? 0;
      statusEl.textContent = job.status || 'unknown';
      percentEl.textContent = `${Math.round(percent)}%`;
      fillEl.style.width = `${percent}%`;
      messageEl.textContent = progress.message || '';
      downloadsEl.innerHTML = '';
      Object.entries(job.downloads || {}).forEach(([name, url]) => {
        const link = document.createElement('a');
        link.href = url;
        link.textContent = `Download ${name}`;
        downloadsEl.appendChild(link);
      });
      if (job.status === 'completed' || job.status === 'failed') {
        clearInterval(timer);
        submit.disabled = false;
      }
    }

    async function loadRuntime() {
      try {
        const response = await fetch('/runtime');
        const runtime = await response.json();
        const torch = runtime.torch || {};
        if (!torch.installed) {
          runtimeEl.textContent = 'Torch is not installed in this Python environment. Install segcraft[web] before running jobs.';
          return;
        }
        const deviceNames = torch.device_names || [];
        if (torch.cuda_available) {
          runtimeEl.textContent = `Torch ${torch.torch_version}; CUDA ${torch.cuda_version}; ${deviceNames.join(', ') || 'CUDA device ready'}`;
          return;
        }
        const cudaBuild = torch.cuda_version ? `CUDA build ${torch.cuda_version}` : 'CPU-only Torch build';
        runtimeEl.textContent = `Torch ${torch.torch_version}; ${cudaBuild}; CUDA unavailable in ${runtime.python_executable}`;
      } catch (error) {
        runtimeEl.textContent = 'Runtime status unavailable.';
      }
    }
  </script>
</body>
</html>"""
    return html.replace("__PRESET_OPTIONS__", _preset_options_html())


def _preset_options_html() -> str:
    names = list_available_presets()
    preferred = "pascal_video" if "pascal_video" in names else (names[0] if names else "")
    options = ['<option value="">base only</option>']
    for name in names:
        selected = " selected" if name == preferred else ""
        options.append(f'<option value="{escape(name)}"{selected}>{escape(name)}</option>')
    return "\n".join(options)


def main() -> None:
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"The web app requires optional dependencies. {INSTALL_HINTS['app']}"
        ) from exc

    uvicorn.run("segcraft.webapp:create_app", factory=True, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
