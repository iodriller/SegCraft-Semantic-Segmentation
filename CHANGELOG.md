# Changelog

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
- Migration guide from legacy scripts.

### Changed
- Legacy script workflow moved to `legacy/`.
- README rewritten around package-first usage.
