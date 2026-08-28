from pathlib import Path

SHEET = "Data_Tong_hop"
TIME_COL = "Thời gian ghi nhận"
QCOL = "Flow out 1 - Giá trị (m3/h)"

POLLUTANTS = {
    "COD": {
        "value_col": "COD - Giá trị (mg/L)",
        "status_col": "COD - Trạng thái",
        "display": "COD",
        "conc_unit": "mg/L",
        "mass_unit": "kg per 5 min",
    },
    "TSS": {
        "value_col": "TSS - Giá trị (mg/L)",
        "status_col": "TSS - Trạng thái",
        "display": "TSS",
        "conc_unit": "mg/L",
        "mass_unit": "kg per 5 min",
    },
    "NH4-N": {
        # The workbook retains a legacy NH4+ field name. Site documentation for
        # this study confirms that the archived channel is NH4-N (as N), so no
        # NH4+ <-> NH4-N conversion is applied.
        "value_col": "NH4+ - Giá trị (mg/L)",
        "status_col": "NH4+ - Trạng thái",
        "display": "NH$_4$-N",
        "conc_unit": "mg N/L",
        "mass_unit": "kg N per 5 min",
    },
}

STATUS_CHANNELS = {
    "COD": ("COD - Giá trị (mg/L)", "COD - Trạng thái"),
    "TSS": ("TSS - Giá trị (mg/L)", "TSS - Trạng thái"),
    "pH": ("pH - Giá trị", "pH - Trạng thái"),
    "NH4-N": ("NH4+ - Giá trị (mg/L)", "NH4+ - Trạng thái"),
    "Temperature": ("Temp - Giá trị (oC)", "Temp - Trạng thái"),
}

ACCEPTED_STATUSES = {"Hoạt động tổt", "Vượt qui chuẩn"}
EXCEEDANCE_STATUS = "Vượt qui chuẩn"
CALIBRATION_STATUS = "Hiệu chuẩn"
DEVICE_ERROR_STATUS = "Lỗi thiết bị"

DEVELOPMENT_START = "2025-01-01"
DEVELOPMENT_END = "2025-12-31 23:55:00"
HOLDOUT_START = "2026-01-01"
HOLDOUT_END = "2026-07-31 23:55:00"
SEASON_END_MONTH = 7

PRIMARY_LOAD_FRACTION = 0.01
PRIMARY_RECALL_TARGET = 0.90
ALERT_GRID_STEP = 0.005
LOAD_FRACTIONS = (0.005, 0.01, 0.02, 0.05)
RECALL_TARGETS = (0.80, 0.90, 0.95)
LOW_FLOW_CUTS = (0.0, 0.5, 1.0)
EVENT_BRIDGES_MIN = (0, 5, 10, 15, 30, 60)
HIGH_COVERAGE_THRESHOLD = 0.90
INFLUENTIAL_DAY = "2026-07-13"

PRIMARY_BOOTSTRAP_REPLICATES = 1000
SECONDARY_BOOTSTRAP_REPLICATES = 500

# Seeds used for the conditional locked-threshold holdout CIs in the current
# manuscript/repository implementation.
FIXED_HOLDOUT_SEEDS = {
    "COD": 20260829,
    "TSS": 20260829,
    "NH4-N": 20260834,
}

# Shared RNG stream used for the revision-added target-mass-capture bootstrap
# and the conditional fixed-P99 later-period bootstrap (Table S7b). Within each
# replicate the same resampled 2026 days support both estimands; the stream then
# advances through pollutants in order so manuscript v2 intervals reproduce.
REVISION_METRICS_BOOTSTRAP_SEED = 20260828

# Deterministic seeds for the remaining bootstrap analyses. The nested and
# moving-block seeds match the analysis used to generate Supplementary Table S8
# and the moving-block sensitivity reported in the revised manuscript.
NESTED_BASE_SEED = 725911
YEAR_BOOTSTRAP_BASE_SEED = 20260831
OVERLAP_DIFF_BASE_SEED = 20260911
HOURLY_BOOTSTRAP_BASE_SEED = 20260921
EVENT_BOOTSTRAP_BASE_SEED = 20260931

POLLUTANT_OFFSETS = {"COD": 0, "TSS": 1, "NH4-N": 2}

DEFAULT_OUTPUT_DIR = Path("outputs")
