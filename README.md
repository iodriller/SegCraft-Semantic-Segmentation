# SegCraft

SegCraft is a small, config-first semantic segmentation toolkit. The goal is
simple: keep experiments readable, move project choices into YAML, and expose a
Python API that notebooks and scripts can share.

The current package supports typed configuration, paired image/mask datasets,
model selection, supervised train/evaluate loops, image/video prediction,
small video helpers, CLI entry points, and tests.

## Install

Core install:

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
pip install -e ".[video]"       # YouTube download/frame/video helpers
pip install -e ".[dev]"         # test runner
```

## Quick Start

Validate a config:

```bash
segcraft validate --config configs/base.yaml
```

Run prediction on an image folder or a video file:

```bash
segcraft predict --config configs/base.yaml --local configs/local.yaml
```

`configs/local.yaml` should point `predict.input_path` at your images or video
and `predict.output_path` at the folder where outputs should be saved.

Train/evaluate use the paired image and mask paths from the config. If those
paths do not exist yet, the commands return a clear `data_missing` summary.
Training writes `checkpoints/best.pt`, `checkpoints/last.pt`, and compact JSON
history files under `runtime.output_dir`.

```bash
segcraft train --config configs/base.yaml --preset configs/presets/fast_dev.yaml
segcraft evaluate --config configs/base.yaml --preset configs/presets/quality.yaml
```

## YouTube Demo

Install the extras used by the demo:

```bash
pip install -e ".[torch,video]"
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
    "  preserve_audio: true\n"
    "runtime:\n"
    "  output_dir: outputs/demo\n",
    encoding="utf-8",
)
```

Run prediction:

```bash
segcraft predict --config configs/base.yaml --preset configs/presets/fast_dev.yaml --local configs/local.yaml
```

For video input, the command writes a new annotated overlay video to
`outputs/demo_predictions/overlay.mp4` at the source FPS. If `ffmpeg` is
available and `predict.preserve_audio` is enabled, source audio is copied into
the overlay video. `summary.json` records frame counts, video metadata, class
coverage, and mean/max confidence for visible classes.

For image-folder input, SegCraft writes indexed mask PNGs under `masks/`,
overlay JPGs under `overlays/`, an optional overlay video, and the same summary
metadata.

For video demos, keep `data.image_size` close to the source aspect ratio. The
example above uses `[360, 640]` for a 16:9 clip.

## Configuration

SegCraft uses one main config with optional overlays:

```text
configs/base.yaml
configs/presets/fast_dev.yaml
configs/presets/quality.yaml
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

config = load_config("configs/base.yaml")
typed = load_config_object("configs/base.yaml")

pairs = pair_image_masks(typed.data.train_images, typed.data.train_masks)
model_spec = build_model(typed.model, typed.task)

# Requires: pip install -e ".[torch]"
model = create_model(typed.model, typed.task)
```

`load_config` returns the merged dictionary for simple scripts. `load_config_object`
returns typed config sections for library code.

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
notebooks/
tests/
legacy/
```

The `legacy/` directory keeps the original script-based workflow as reference
while the package version replaces it with maintained modules.

## Development

```bash
pip install -e ".[dev]"
pytest
```
