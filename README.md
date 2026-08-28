# Reproducibility package: concentration-only alerts and mass-ranked effluent extremes

This repository contains the analysis code and cleaned five-minute monitoring dataset used for the manuscript:

> **Temporal Transferability of Concentration-Only Alerts for Mass-Ranked Extremes in High-Frequency Industrial Effluent Monitoring**

The analysis tests whether concentration-only alert thresholds calibrated in an earlier operating period retain their ability to retrieve later **mass-ranked** effluent extremes. The primary design is chronological: thresholds are developed using 2025 observations, locked, and evaluated without re-estimation in January-July 2026. This release is aligned with the revised manuscript and includes the added **target-mass-capture** endpoint and conditional uncertainty analysis for the fixed historical P99 target.

## Repository contents

```text
.
├── Dataset.xlsx
├── README.md
├── CHANGELOG.md
├── DATA_DICTIONARY.md
├── requirements.txt
├── run_all.py
├── Supplementary_Code_Reproducibility.py
├── validate_key_results.py
├── analysis/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── core.py
│   ├── bootstrap.py
│   ├── robustness.py
│   ├── temporal.py
│   └── figures.py
├── outputs/
│   └── .gitkeep
└── .github/
    └── workflows/
        └── reproducibility.yml
```

See `DATA_DICTIONARY.md` for the workbook fields and status labels used by the analysis.

`Dataset.xlsx` is the cleaned analytical workbook. The code does not reconstruct the upstream sensor-ingestion or workbook-cleaning process; it reproduces the analyses reported from the supplied cleaned workbook.

## Data integrity

Expected workbook:

- File: `Dataset.xlsx`
- Analysis sheet: `Data_Tong_hop`
- Unique five-minute timestamps: **148,906**
- Observation window: **2025-01-01 00:00 to 2026-07-31 23:55**, Vietnam Standard Time (UTC+7)
- SHA-256: `253fd20f2e26f9e5b5cae57a4ed1140433f01ef141805e20b7b1f11b226fa283`

The workbook retains the legacy field name `NH4+ - Giá trị (mg/L)`. In the study documentation this archived channel is treated as **NH4-N (as N)**, so the analysis does not apply an NH4+/NH4-N chemical-form conversion.

## Analysis conventions

The implementation follows the manuscript definitions:

- Accepted chemistry statuses: `Hoạt động tổt` and `Vượt qui chuẩn`.
- Primary mass-ranking records require an accepted parameter status, positive chemistry, and positive effluent flow.
- Five-minute load rate is calculated as `C * Q / 1000` in kg/h (kg N/h for NH4-N).
- Five-minute interval mass is `load_rate * 5/60`.
- Missing calendar slots are not converted to zero flow and are not gap-filled for annual mass estimation.
- Top-load and top-concentration sets use stable descending sorting.
- Threshold comparisons are inclusive (`C >= threshold`); all ties at a threshold are flagged.
- The primary target is the period-relative top 1% of five-minute interval mass, with a 90% recall design benchmark. The 90% value is an operational benchmark, not a regulatory cutoff; 80-95% sensitivity grids are also reported.
- **Target-mass capture** is the mass-weighted companion to interval recall: mass captured among correctly flagged members of the target set divided by total mass within that target set.
- Concentration-screening burden is evaluated on a 0.5-percentage-point grid.
- The primary deployment test locks the 2025 threshold and evaluates it in January-July 2026.
- The fixed-P99 sensitivity instead locks the 2025 P99 **absolute interval-mass boundary** and evaluates the corresponding 2026 target. Its later-period uncertainty is obtained by resampling 2026 calendar days while both the observed 2025 concentration threshold and P99 boundary remain fixed.
- Jensen-Shannon distance uses `log1p(C)` and `log1p(Q)`, pooled 25 x 25 quantile bins, duplicate-edge removal, a 0.5 pseudocount per cell, and base-2 Jensen-Shannon distance.
- Primary uncertainty uses calendar-day resampling. The nested bootstrap resamples 2025 days, re-derives the threshold, and independently resamples 2026 days. The revision-added mass-capture/fixed-P99 bootstrap uses the documented shared seed in `analysis/config.py` to reproduce the revised manuscript intervals exactly.
- Moving-block sensitivity uses circular 1-, 3-, and 7-day blocks.
- Secondary hourly and event analyses use 500 bootstrap replicates by default.

All bootstrap seeds are defined in `analysis/config.py`.

## Software environment

The package was tested with Python **3.13.5**. `requirements.txt` records the exact Python-package versions used for this reproducibility release.

Create an isolated environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Reproduce the manuscript analyses

Place `Dataset.xlsx` in the repository root and run:

```bash
python run_all.py --input Dataset.xlsx --output outputs
```

The manuscript settings are the defaults:

- 1,000 replicates for primary bootstrap analyses;
- 500 replicates for secondary/moving-block analyses;
- 300-dpi PNG figure export.

For an explicit command:

```bash
python run_all.py \
  --input Dataset.xlsx \
  --output outputs \
  --bootstrap 1000 \
  --secondary-bootstrap 500
```

On Windows PowerShell, the same command can be entered on one line:

```powershell
python run_all.py --input Dataset.xlsx --output outputs --bootstrap 1000 --secondary-bootstrap 500
```

### Faster smoke run

For installation/CI checking without waiting for all bootstrap analyses and figures:

```bash
python run_all.py \
  --input Dataset.xlsx \
  --output outputs_smoke \
  --bootstrap 100 \
  --secondary-bootstrap 50 \
  --skip-secondary \
  --skip-figures
```

Bootstrap confidence intervals from a smoke run are **not** the manuscript intervals because the replicate count is deliberately reduced.

## Validate key deterministic results

Before a public release, run:

```bash
python validate_key_results.py --input Dataset.xlsx
```

This checks the workbook checksum and the principal deterministic point estimates used in the manuscript, including sample sizes, locked thresholds, holdout performance, target-mass capture, season-matched validation, and fixed-P99 sensitivity.

To reproduce and validate the revision-added 1,000-replicate target-mass-capture/fixed-P99 bootstrap endpoints:

```bash
python validate_key_results.py --input Dataset.xlsx --revision-bootstrap
```

To additionally validate the slower hourly and event-segmentation point estimates:

```bash
python validate_key_results.py --input Dataset.xlsx --full
```

The default validator focuses on deterministic point estimates. The optional `--revision-bootstrap` check is provided because the new percentile endpoints are explicitly reported in the revised manuscript and Supplementary Table S7b.

## Output-to-manuscript map

After `run_all.py`, the following files are written under `outputs/`.

| Output | Manuscript location |
|---|---|
| `main/Table1_primary_sample.csv` | Table 1 |
| `main/Table2_concentration_load_mismatch.csv` | Table 2 |
| `main/Table3_locked_threshold_holdout.csv` | Table 3, including target-mass capture and its conditional CI |
| `main/Table4_season_matched.csv` | Table 4 |
| `main/Joint_distribution_shift.csv` | Results: joint concentration-flow regime shift |
| `supplement/TableS1_status_audit.csv` | Table S1 |
| `supplement/TableS2a_decision_grid_2025.csv` | Table S2a |
| `supplement/TableS2b_decision_grid_2026.csv` | Table S2b |
| `supplement/TableS3_overlap_jaccard.csv` | Table S3 |
| `supplement/TableS3b_overlap_difference_bootstrap.csv` | Difference-CI text accompanying Table S3 |
| `supplement/TableS4_expanding_origin.csv` | Table S4 |
| `supplement/TableS4b_quarter_fixed_threshold.csv` | Quarter diagnostics accompanying S4 |
| `supplement/TableS5_low_flow_sensitivity.csv` | Table S5 |
| `supplement/TableS6_influential_day_high_coverage.csv` | Table S6 |
| `supplement/TableS7_fixed_P99_target.csv` | Table S7 and fixed-target target-mass-capture point estimates |
| `supplement/TableS7b_fixed_P99_bootstrap.csv` | Table S7b: conditional fixed-P99 target uncertainty, target support, precision and target-mass capture |
| `supplement/TableS8_nested_threshold_bootstrap.csv` | Table S8, nested threshold-estimation uncertainty |
| `supplement/TableS8b_moving_block_sensitivity.csv` | Table S8, moving-block sensitivity |
| `supplement/TableS9_hourly_aggregation.csv` | Table S9 |
| `supplement/TableS10_event_segmentation.csv` | Table S10 |
| `supplement/TableS11_event_bootstrap.csv` | Table S11 |
| `supplement/Month_level_locked_threshold.csv` | Month-level fixed-threshold diagnostics / Figure 2 source |
| `figures/Figure1_recall_alert_curves.png` | Figure 1 |
| `figures/Figure2_monthly_transfer.png` | Figure 2 |
| `figures/Figure3_joint_regime_shift.png` | Figure 3 |
| `figures/FigureS1_coverage_status.png` | Figure S1 |
| `figures/FigureS2_hourly_aggregation_error.png` | Figure S2 |
| `figures/FigureS3_event_mass_segmentation.png` | Figure S3 |
| `analysis_results.json` | Machine-readable summary of principal outputs and dataset checksum |

## Entry points

`run_all.py` is the recommended entry point. `Supplementary_Code_Reproducibility.py` is retained as a compatibility wrapper so that the repository still offers the single-script filename used in earlier versions of the project.

Useful options:

```text
--input PATH                 cleaned workbook (default: Dataset.xlsx)
--output DIR                 output directory (default: outputs)
--bootstrap N                primary bootstrap replicates (default: 1000)
--secondary-bootstrap N      secondary/moving-block replicates (default: 500)
--skip-secondary             skip hourly and event analyses
--skip-figures               skip PNG figure generation
```

## Revised-manuscript additions reproduced by this release

The final revision added two analyses that were not present in the earlier repository package:

1. **Mass-weighted target capture for the primary locked holdout.** The locked 2025 thresholds capture 57.0% of COD target mass, 67.1% of TSS target mass, and 99.6% of NH4-N target mass in January-July 2026. The corresponding conditional day-bootstrap 95% intervals are 18.9-83.7%, 42.8-87.0%, and 96.9-100.0%. These values are written to `main/Table3_locked_threshold_holdout.csv`.
2. **Conditional uncertainty for the fixed historical P99 target.** The observed later target counts are 26 (COD), 83 (TSS), and 1,088 (NH4-N). `supplement/TableS7b_fixed_P99_bootstrap.csv` reports bootstrap target support together with recall, precision and target-mass capture. Six of 1,000 COD resamples contain no interval above the locked historical boundary, matching the caveat reported in the revised Supplementary Material.

The new bootstrap stream is intentionally separated from the legacy fixed-threshold recall/precision stream so the repository reproduces the exact percentile intervals printed in the revised manuscript while preserving the earlier published conditional intervals.

## Reproducibility notes

1. **Period-relative versus absolute targets.** The primary endpoint ranks mass within each evaluation period; the fixed-P99 sensitivity asks a different question by carrying a 2025 absolute interval-mass boundary into 2026. The two analyses should not be interpreted as the same estimand. High fixed-target recall must also be interpreted alongside later target support and precision.
2. **Conditional versus threshold-estimation uncertainty.** The primary locked-threshold CI conditions on the observed 2025 threshold. Table S8 adds nested resampling to propagate development-period threshold-estimation uncertainty.
3. **Serial dependence.** Five-minute observations are not treated as independent replicates. Day-block and moving-block analyses preserve temporal clustering at the chosen block scale.
4. **Monitoring versus legal compliance.** Station-configured exceedance labels are used as operational metadata. This repository does not reconstruct permit-specific averaging rules or retrospectively adjudicate legal violations.
5. **Observed regime shift is descriptive.** The repository reproduces the change in the observed joint concentration-flow distribution, but the available data cannot separate process/hydraulic change from calibration, sensor-servicing, or flowmeter contributions.
6. **Single-site scope.** Numerical thresholds and effect sizes are site- and period-specific. The portable contribution is the evaluation workflow, not a universal COD, TSS, or NH4-N concentration cutoff.

## Data and code availability statement

A manuscript-compatible statement is:

> The cleaned analytical dataset and code required to reproduce the reported analyses are available in this repository. The repository includes fixed random seeds, a pinned software environment, machine-readable outputs, and a validation script for principal deterministic results.

For long-term publication, create a versioned GitHub release and, if possible, archive that exact release with a persistent DOI (for example via Zenodo). Update the manuscript Data Availability statement with the archived DOI after it is minted.

## License

No software/data license is selected in this package because licensing is an author decision. **Before public release, add an explicit license appropriate for both the code and the dataset.** Do not assume that a public GitHub repository automatically grants reuse rights.
