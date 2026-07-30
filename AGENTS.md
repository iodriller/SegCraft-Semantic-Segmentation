# AGENTS.md

## Project

SegCraft is a configuration-first semantic-segmentation toolkit. The same merged
YAML configuration drives training, evaluation, image/video prediction, the
Python API, and the optional FastAPI web application.

Configuration merges in this order: `configs/base.yaml`, an optional preset, then
the ignored machine-specific `configs/local.yaml`. Package code lives under
`src/segcraft/`; tests live under `tests/`; notebooks are examples, not the
canonical implementation.

## Commands

Development:

```bash
pip install -e ".[web,dev]"
segcraft validate
segcraft doctor
pytest
python -m build
twine check dist/*
```

Representative workflows:

```bash
segcraft predict --preset cityscapes_video --local configs/local.yaml
segcraft train --preset fast_dev --local configs/local.yaml
segcraft evaluate --preset quality --local configs/local.yaml
segcraft-web
```

Do not publish to PyPI/TestPyPI or trigger release workflows unless explicitly
requested.

## Project Rules

- Keep CLI, Python API, and web app behavior aligned around the same configuration
  loader and preset resolution.
- `task.num_classes` controls model heads; `task.class_names` controls display
  labels. Do not conflate them.
- Preserve `runtime.device: auto` fallback and report the actual Torch/CUDA/device
  state rather than assuming GPU availability.
- Keep optional model ecosystems behind their extras. Core config parsing should
  not require Torch, Transformers, SMP, video, or web dependencies.
- Avoid network/model downloads in deterministic tests. Make remote weights,
  YouTube access, and GPU checks explicit.
- Do not commit local paths, datasets, outputs, downloaded weights, build
  artifacts, notebook output, or large generated media.
- Maintain summary/output compatibility when changing prediction artifacts.

## Verification

- Documentation or guidance only: verify referenced paths and run
  `git diff --check`; application tests are not required.
- Core Python behavior: run the focused test and `pytest`.
- Config or CLI changes: also run `segcraft validate`.
- Runtime/device changes: run `segcraft doctor` in the actual target environment.
- Packaging changes: run `python -m build` and `twine check dist/*`.
- Web/video/model-backend changes: run only the relevant optional lane and report
  missing extras, models, network, or GPU access explicitly.

Never describe CPU/GPU, model, or dataset results that were not observed.

## Git and Safety

- Preserve unrelated changes and keep commits focused.
- Use the configured repository-owner identity.
- Do not add assistant names, co-author trailers, session links, or tool
  attribution to Git artifacts.
- Confirm licenses and redistribution rights before adding datasets, weights,
  videos, or generated examples.
