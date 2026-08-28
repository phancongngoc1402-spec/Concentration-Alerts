# Changelog

## Manuscript-aligned revision (2026-08-28)

This repository revision aligns the reproducibility package with the revised manuscript and supplementary material.

### Added

- Mass-weighted **target-mass capture** for the locked 2025 -> January-July 2026 period-relative top-1% holdout.
- Conditional calendar-day bootstrap intervals for target-mass capture in Main Table 3.
- Target-mass-capture point estimates for the fixed historical P99 target.
- Conditional later-period bootstrap for the fixed historical P99 target, including target support, recall, precision, target-mass capture, alert rate, and zero-target replicate count (`TableS7b_fixed_P99_bootstrap.csv`).
- Optional `--revision-bootstrap` validation mode in `validate_key_results.py`.

### Clarified

- The 90% recall level is an operational design benchmark, not a regulatory cutoff.
- Fixed historical P99 performance is a different estimand from period-relative prioritization and must be interpreted with target support and precision.
- The observed concentration-flow distribution shift is descriptive and does not identify whether process, hydraulic, calibration, sensor-servicing, or flowmeter changes caused it.

### Reproducibility

The revision-added bootstrap uses the documented seed in `analysis/config.py` and the same resampled 2026 days for the period-relative target-mass-capture and fixed-P99 metrics within each replicate. This reproduces the percentile intervals reported in the revised manuscript.
