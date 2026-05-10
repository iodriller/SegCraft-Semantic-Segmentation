# Changelog

## Unreleased

### Added
- Optional FastAPI web app for video upload, YouTube input, progress, and output downloads.
- Prediction progress callback support.
- Optional-extra smoke CI and release publishing workflow.
- Real image prediction test using a dummy model when PyTorch is installed.
- Release metadata for PyPI packaging.
- Lightweight API notebook for config loading, model specs, and dataset pairing.
- Packaged CLI base config so `segcraft validate` works after wheel install.
- Direct video prediction that writes annotated overlay MP4 files from video input.
- Original-video copies and side-by-side comparison MP4 files for video prediction.
- Cityscapes video preset using SegFormer through the optional Transformers backend.
- Configurable overlay display controls for panels, region labels, confidence, percentages, and palette.
- Video sampling controls for first-N-seconds prediction and frame stride.
- Stabilized video label placement to reduce frame-to-frame jitter.
- Explicit floating-label toggle and concise GPU setup notes.
- Per-class confidence metadata in prediction overlays and `summary.json`.
- Minimal training controls for schedulers, AMP, checkpoint resume, early stopping, and JSON run summaries.

### Changed
- Video examples now use a 16:9 prediction size to avoid square-frame distortion.
- Prediction overlays now use a vivid class palette by default.

## [0.1.0] - 2026-05-02

### Added
- Installable `segcraft` package scaffold under `src/segcraft`.
- Config system with schema validation and base/preset/local merging.
- Typed config sections through `load_config_object`.
- Dataset file discovery and image/mask pairing helpers.
- Optional model factories for TorchVision and segmentation-models-pytorch.
- Real image-folder prediction with mask and overlay outputs.
- Automatic overlay video writing for prediction outputs.
- Annotated prediction overlays with frame metadata and per-frame class summaries.
- Prediction `summary.json` with per-frame artifact paths and detected classes.
- Small video helpers for YouTube download, frame extraction, and verified video writing.
- Minimal supervised train/evaluate loops for paired image/mask datasets.
- CLI modes: `validate`, `train`, `evaluate`, `predict`.
- Workflow scaffolding with task-aware loss/metric defaults.
- Example notebook: `notebooks/01_quickstart.ipynb`.
- Tests for schema, merge semantics, and workflow summaries.
- CI workflow to run pytest.

### Changed
- README rewritten around package-first usage.
