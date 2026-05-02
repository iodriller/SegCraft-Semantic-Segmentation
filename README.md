# SegCraft

SegCraft is a config-driven semantic segmentation framework scaffold. It is installable, testable, and designed so users can run workflows by changing configuration instead of modifying code.

## Project completion status

✅ Step 1: package foundation and installability scaffold  
✅ Step 2: unified configuration layout and validation  
✅ Step 3: core library API + CLI workflow modes  
✅ Step 4: task-aware segmentation setup (binary vs multiclass)  
✅ Step 5: notebook quickstart using the installable library API  
✅ Step 6: quality gates (tests + CI)  
✅ Step 7: migration docs + release polish + missing-piece review

## Install

```bash
pip install -e .
```

## Quick start

Validate base config:

```bash
segcraft validate --config configs/base.yaml
```

Train/evaluate/predict with overrides:

```bash
segcraft train --config configs/base.yaml --preset configs/presets/fast_dev.yaml --local configs/local.yaml
segcraft evaluate --config configs/base.yaml --preset configs/presets/quality.yaml --local configs/local.yaml
segcraft predict --config configs/base.yaml --local configs/local.yaml
```

## Example notebook

- `notebooks/01_quickstart.ipynb` now shows a concise YouTube -> frames -> SegCraft prediction flow.
- It stays practical and short, with inline comments and a fast path for first-time users.

## Configuration strategy

- `configs/base.yaml`: full config for standard use.
- `configs/presets/*.yaml`: optional overrides (`fast_dev`, `quality`, `binary_quickstart`).
- `configs/local.example.yaml`: machine-specific template; copy to `configs/local.yaml`.

Merge order:
1. base config
2. optional preset
3. optional local config

## Task types

`task.type` supports:
- `binary` (requires `task.num_classes: 1`)
- `multiclass` (requires `task.num_classes >= 2`)

Loss defaults when `train.loss: auto`:
- binary -> `bce_dice`
- multiclass -> `cross_entropy_dice`

Metrics default by task type when `eval.metrics` is empty.

## Quality checks

- Unit tests: `tests/test_schema.py`, `tests/test_merge.py`, `tests/test_workflows.py`
- CI: `.github/workflows/ci.yml`

Run locally:

```bash
python -m pytest
```

## Migration and release docs

- Migration guide: `MIGRATION.md`
- Changelog: `CHANGELOG.md`

## Repository layout

```text
src/segcraft/
  api.py
  cli/main.py
  config/{loader.py,schema.py}
  engine/workflows.py
  metrics/segmentation.py
  models/registry.py
configs/
  base.yaml
  presets/{fast_dev.yaml,quality.yaml,binary_quickstart.yaml}
  local.example.yaml
notebooks/
  01_quickstart.ipynb
tests/
  test_schema.py
  test_merge.py
  test_workflows.py
.github/workflows/
  ci.yml
legacy/
  <previous script-based workflow>
```

## Missing pieces (intentional for next milestone)

To make this a full training framework beyond scaffold mode, next milestone should add:
- real dataset/dataloader module
- real model backend wiring (PyTorch)
- checkpoint save/load and experiment tracking
- visualization and export utilities

That work is now straightforward because the package/config/test structure is in place.
