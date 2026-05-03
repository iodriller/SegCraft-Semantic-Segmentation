# Migration Guide (Legacy Scripts -> SegCraft)

This project moved from one-off scripts to a config-driven package.

## Mapping

| Legacy script | SegCraft replacement |
|---|---|
| `legacy/process_and_run.py` | `segcraft train --config configs/base.yaml` |
| `legacy/semantic_images.py` | `segcraft predict --config configs/base.yaml` |
| `legacy/separate_to_jpgs.py` | keep as preprocessing utility in `legacy/` for now |
| `legacy/combine_and_make_video.py` | keep as postprocessing utility in `legacy/` for now |
| `legacy/youtube_download.py` | keep as data-acquisition utility in `legacy/` for now |

## Recommended migration path

1. Copy `configs/local.example.yaml` -> `configs/local.yaml` and set your data/output paths.
2. Validate configuration:
   ```bash
   segcraft validate --config configs/base.yaml --local configs/local.yaml
   ```
3. Start with a quick preset:
   ```bash
   segcraft train --config configs/base.yaml --preset configs/presets/fast_dev.yaml --local configs/local.yaml
   ```
4. For binary segmentation, use:
   ```bash
   segcraft train --config configs/base.yaml --preset configs/presets/binary_quickstart.yaml --local configs/local.yaml
   ```

## What remains in legacy

Some old scripts are intentionally preserved until future data/pipeline modules are implemented. The SegCraft path is the default path for new users.
