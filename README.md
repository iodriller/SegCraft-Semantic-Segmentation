# SegCraft

SegCraft is a small, config-first semantic segmentation toolkit. The goal is
simple: keep experiments readable, move project choices into YAML, and expose a
Python API that notebooks and scripts can share.

The current package supports typed configuration, dataset discovery, model
selection, real image-folder prediction, small video helpers, CLI entry points,
and tests. Training and evaluation are still lightweight workflow summaries
while the data/model pieces settle.

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

Run prediction on an image folder:

```bash
segcraft predict --config configs/base.yaml --local configs/local.yaml
```

`configs/local.yaml` should point `predict.input_path` at your images and
`predict.output_path` at the folder where masks and overlays should be saved.

Train/evaluate currently return structured summaries:

```bash
segcraft train --config configs/base.yaml --preset configs/presets/fast_dev.yaml
segcraft evaluate --config configs/base.yaml --preset configs/presets/quality.yaml
```

## YouTube Demo

Install the extras used by the demo:

```bash
pip install -e ".[torch,video]"
```

Prepare a short video clip:

```python
from pathlib import Path
from segcraft.video import download_youtube, extract_frames

url = "https://www.youtube.com/watch?v=ZpYBDJv5KAU"
video_path = download_youtube(url, "data/demo/video.mp4")
extract_frames(video_path, "data/demo/frames", every_seconds=1.0, max_frames=10)

Path("configs/local.yaml").write_text(
    "predict:\n"
    "  input_path: data/demo/frames\n"
    "  output_path: outputs/demo_predictions\n"
    "runtime:\n"
    "  output_dir: outputs/demo\n",
    encoding="utf-8",
)
```

Run prediction:

```bash
segcraft predict --config configs/base.yaml --preset configs/presets/fast_dev.yaml --local configs/local.yaml
```

The command writes indexed mask PNGs under `outputs/demo_predictions/masks` and
overlay JPGs under `outputs/demo_predictions/overlays`.

To stitch the overlays back into a quick MP4:

```python
from segcraft.video import write_video_from_images

write_video_from_images("outputs/demo_predictions/overlays", "outputs/demo_predictions/overlay.mp4", fps=4)
```

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
