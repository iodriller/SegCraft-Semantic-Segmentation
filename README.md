# SegCraft

SegCraft is a small, config-first semantic segmentation toolkit. The goal is
simple: keep experiments readable, move project choices into YAML, and expose a
Python API that notebooks and scripts can share.

The current package supports typed configuration, paired image/mask datasets,
model selection, supervised train/evaluate loops, image/video prediction,
small video helpers, CLI entry points, and tests.

![SegCraft GPU demo: original dashcam video beside semantic segmentation overlay](assets/segcraft-gpu-demo.gif)

## Install

After the first release is published:

```bash
pip install segcraft
```

Install directly from GitHub today:

```bash
pip install "segcraft @ git+https://github.com/iodriller/SegCraft-Semantic-Segmentation.git"
```

Local development install:

```bash
pip install -e .
```

Install the image/model stack for TorchVision prediction:

```bash
pip install -e ".[torch]"
```

Optional extras:

```bash
pip install -e ".[torch,smp]"   # Unet, FPN, Linknet, PSPNet via segmentation-models-pytorch
pip install -e ".[torch,transformers]"  # Hugging Face semantic segmentation models
pip install -e ".[video]"       # YouTube download/frame/video helpers
pip install -e ".[app]"         # FastAPI demo app
pip install -e ".[dev]"         # test runner
```

For NVIDIA GPUs, install a CUDA-enabled PyTorch wheel after the editable
install. Check the current command on the [PyTorch install page](https://pytorch.org/get-started/locally/);
for CUDA 12.8 it looks like:

```bash
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Quick Start

Validate a config:

```bash
segcraft validate
```

When you run from a source checkout, SegCraft uses `configs/base.yaml`. When
installed from a wheel, the CLI falls back to the packaged base config.

Run prediction on an image folder or a video file:

```bash
segcraft predict --config configs/base.yaml --local configs/local.yaml
```

`configs/local.yaml` should point `predict.input_path` at your images or video
and `predict.output_path` at the folder where outputs should be saved.

Launch the optional web UI:

```bash
pip install -e ".[torch,video,app]"
segcraft-web
```

Open `http://127.0.0.1:8000`, upload a video or paste a YouTube URL, then
download `comparison.mp4`, `overlay.mp4`, `original.mp4`, or `summary.json`
when the job finishes.

Train/evaluate use the paired image and mask paths from the config. If those
paths do not exist yet, the commands return a clear `data_missing` summary.
Training writes `checkpoints/best.pt`, `checkpoints/last.pt`, and compact JSON
history files under `runtime.output_dir`.

```bash
segcraft train --config configs/base.yaml --preset configs/presets/fast_dev.yaml
segcraft evaluate --config configs/base.yaml --preset configs/presets/quality.yaml
```

## YouTube Demo

Install the extras used by the Cityscapes video demo:

```bash
pip install -e ".[torch,transformers,video]"
```

Prepare a video. On CPU, start with a short clip:

```python
from pathlib import Path
from segcraft.video import download_youtube

url = "https://www.youtube.com/watch?v=BHYOo3JCuvk"
video_path = download_youtube(url, "data/demo/video.mp4")

Path("configs/local.yaml").write_text(
    "data:\n"
    "  image_size: [360, 640]\n"
    "predict:\n"
    "  input_path: data/demo/video.mp4\n"
    "  output_path: outputs/demo_predictions\n"
    "  video_max_seconds: 60\n"
    "  video_frame_stride: 1\n"
    "  preserve_audio: true\n"
    "runtime:\n"
    "  device: auto\n"
    "  output_dir: outputs/demo\n",
    encoding="utf-8",
)
```

Run prediction:

```bash
segcraft predict --config configs/base.yaml --preset configs/presets/cityscapes_video.yaml --local configs/local.yaml
```

For video input, the output folder contains:

- `original.mp4`: the source frames used for prediction
- `overlay.mp4`: the same video with semantic segmentation overlays and class/confidence metadata
- `comparison.mp4`: original and processed videos side by side
- `summary.json`: frame counts, video metadata, class coverage, and mean/max confidence values

If `ffmpeg` is available and `predict.preserve_audio` is enabled, source audio
is copied into `overlay.mp4`. The comparison video is written without audio.

For image-folder input, SegCraft writes indexed mask PNGs under `masks/`,
overlay JPGs under `overlays/`, an optional overlay video, and the same summary
metadata.

For video demos, keep `data.image_size` close to the source aspect ratio. The
example above uses `[360, 640]` for a 16:9 clip.

Video sampling controls live under `predict`:

```yaml
predict:
  video_max_seconds: 60    # null means process the whole video
  video_frame_stride: 1    # 1 means every frame; 2 means every other frame
```

Direct video prediction preserves the source FPS when `video_frame_stride` is
`1`. If you increase the stride, SegCraft lowers the output FPS so the processed
video keeps the same real-time pace instead of speeding up.

Runtime device selection lives under `runtime`:

```yaml
runtime:
  device: auto  # auto, cuda, or cpu
```

`auto` uses CUDA when the installed PyTorch build can see a GPU, otherwise it
uses CPU. Prediction also records `device_fallback` in `summary.json` if model
startup falls back from CUDA to CPU. Use `cuda` when you want a hard failure if
GPU is not available.

Display controls live under `predict.display`:

```yaml
predict:
  display:
    palette: vivid           # vivid or pascal
    show_panel: true
    show_floating_labels: false # turn on labels near large predicted regions
    show_confidence: true
    show_percentages: true
    max_classes: 8
    max_labels: 10
    label_min_pixels: 2400   # ignore tiny regions when floating labels are enabled
    label_move_threshold: 96 # pixels; ignore smaller label movements frame to frame
    label_smoothing: 0.85    # 0 disables smoothing, higher values move labels more slowly
    panel_position: top_right
```

The detectable classes are the model's labels. Configure the display names in
`task.class_names`; keep the list aligned with the model checkpoint. For the
Cityscapes preset, the classes are `road`, `sidewalk`, `building`, `wall`,
`fence`, `pole`, `traffic_light`, `traffic_sign`, `vegetation`, `terrain`,
`sky`, `person`, `rider`, `car`, `truck`, `bus`, `train`, `motorcycle`, and
`bicycle`.

## Configuration

SegCraft uses one main config with optional overlays:

```text
configs/base.yaml
configs/presets/fast_dev.yaml
configs/presets/quality.yaml
configs/presets/cityscapes_video.yaml
configs/presets/binary_quickstart.yaml
configs/local.example.yaml
```

Merge order is:

1. base config
2. optional preset
3. optional local config

Use `configs/local.yaml` for machine-specific paths. It is ignored by git.

Useful training controls live in the same `train` section:

```yaml
train:
  scheduler: cosine        # none, cosine, or step
  amp: true                # enabled only when running on CUDA
  resume_from: outputs/checkpoints/last.pt
  early_stopping_patience: 8
```

## Python API

```python
from segcraft import load_config, load_config_object
from segcraft.data import pair_image_masks
from segcraft.models import build_model, create_model
from segcraft.prediction import run_prediction

config = load_config("configs/base.yaml")
typed = load_config_object("configs/base.yaml")

pairs = pair_image_masks(typed.data.train_images, typed.data.train_masks)
model_spec = build_model(typed.model, typed.task)

# Requires: pip install -e ".[torch]"
model = create_model(typed.model, typed.task)
```

`load_config` returns the merged dictionary for simple scripts. `load_config_object`
returns typed config sections for library code.

Prediction also accepts a lightweight progress callback:

```python
events = []
summary = run_prediction(config, progress_callback=events.append)
```

## Notebooks

- `notebooks/01_quickstart.ipynb`: YouTube/video-to-overlay demo.
- `notebooks/02_config_and_api.ipynb`: minimal config, dataset pairing, and model-spec API tour.
- `notebooks/03_web_app.ipynb`: launch and use the optional FastAPI demo app.

## Supported Model Specs

TorchVision:

- `deeplabv3`, `deeplabv3_resnet50`, `deeplabv3_resnet101`
- `fcn`, `fcn_resnet50`, `fcn_resnet101`
- `lraspp`, `lraspp_mobilenet_v3_large`

segmentation-models-pytorch:

- `unet`
- `unetplusplus`
- `fpn`
- `linknet`
- `pspnet`

Transformers:

- Hugging Face model ids such as `nvidia/segformer-b0-finetuned-cityscapes-1024-1024`

## Repository Layout

```text
src/segcraft/
  api.py
  cli/main.py
  config/
  data/
  engine/
  metrics/
  models/
  prediction/
  training.py
  video.py
configs/
assets/
notebooks/
tests/
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

Build and check release artifacts:

```bash
python -m build
twine check dist/*
```
