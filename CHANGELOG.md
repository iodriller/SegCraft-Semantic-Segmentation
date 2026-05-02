# Changelog

## [0.1.0] - 2026-05-02

### Added
- Installable `segcraft` package scaffold under `src/segcraft`.
- Config system with schema validation and base/preset/local merging.
- CLI modes: `validate`, `train`, `evaluate`, `predict`.
- Workflow scaffolding with task-aware loss/metric defaults.
- Example notebook: `notebooks/01_quickstart.ipynb`.
- Tests for schema, merge semantics, and workflow summaries.
- CI workflow to run pytest.
- Migration guide from legacy scripts.

### Changed
- Legacy script workflow moved to `legacy/`.
- README rewritten around package-first usage.
