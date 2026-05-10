# SegCraft

SegCraft is a config-first semantic segmentation toolkit for training,
evaluating, and running image/video prediction from the same YAML setup.

![SegCraft GPU demo: original dashcam video beside semantic segmentation overlay](assets/segcraft-gpu-demo.gif)

## Install

```bash
pip install segcraft
```

Install only the extras you need:

```bash
pip install "segcraft[torch]"                    # prediction/training with TorchVision
pip install "segcraft[torch,smp]"                # segmentation-models-pytorch
pip install "segcraft[torch,transformers]"       # Hugging Face segmentation models
pip install "segcraft[torch,transformers,video]" # video files and YouTube helpers
pip install "segcraft[torch,transformers,video,app]" # optional FastAPI UI
```

From a checkout:

```bash
pip install -e ".[torch,transformers,video,app,dev]"
```

For NVIDIA GPUs, install the CUDA-enabled PyTorch wheel that matches your
system from the PyTorch install page.

## CLI

```bash
segcraft validate
segcraft predict --preset cityscapes_video --local configs/local.yaml
segcraft train --preset fast_dev --local configs/local.yaml
segcraft evaluate --preset quality --local configs/local.yaml
```

`configs/local.yaml` is for machine-specific paths and is ignored by git.
Start from `configs/local.example.yaml`.

## Web App

```bash
segcraft-web
```

Open `http://127.0.0.1:8000`. The UI accepts either a video upload or a
YouTube URL, lets you choose a preset or type a custom preset path/name, shows
job progress, and exposes downloads for the generated outputs.

## Notebooks

- `notebooks/01_quickstart.ipynb`: video prediction demo.
- `notebooks/02_config_and_api.ipynb`: config and API basics.
- `notebooks/03_web_app.ipynb`: launching the optional FastAPI app.

## Presets

SegCraft merges configs in this order:

1. `configs/base.yaml`
2. optional preset
3. optional local config

Preset names work in the CLI, Python API, and web app:

- `fast_dev`: tiny CPU training run.
- `quality`: longer training settings with scheduler and metrics.
- `binary_quickstart`: binary foreground/background setup.
- `pascal_video`: TorchVision PASCAL/VOC video prediction.
- `cityscapes_video`: SegFormer Cityscapes video prediction.
- `cpu_video_demo`: short Cityscapes CPU demo settings.
- `ade20k_video`: SegFormer ADE20K video prediction.
- `smp_unet_resnet34`: SMP Unet training setup.

`task.num_classes` controls trainable model heads. `task.class_names` only
controls display names; if labels are missing or do not match the model,
SegCraft falls back to `class_<id>` names during prediction.

## Python API

```python
from segcraft import load_config, load_config_object, list_available_presets
from segcraft.prediction import run_prediction

print(list_available_presets())

config = load_config("configs/base.yaml", preset_path="cityscapes_video")
typed = load_config_object("configs/base.yaml", preset_path="cityscapes_video")

events = []
summary = run_prediction(config, progress_callback=events.append)
```

## Outputs

Video prediction writes:

- `original.mp4`
- `overlay.mp4`
- `comparison.mp4`
- `summary.json`

Image-folder prediction writes masks, overlays, an optional overlay video, and
the same summary metadata.

## Development

```bash
pip install -e ".[dev]"
pytest
python -m build
twine check dist/*
```

Publishing uses `.github/workflows/release.yml` with GitHub trusted publishing.
Configure the PyPI/TestPyPI publisher for owner `iodriller`, repository
`SegCraft-Semantic-Segmentation`, workflow `release.yml`, and no environment,
then run the workflow manually for the target repository.
