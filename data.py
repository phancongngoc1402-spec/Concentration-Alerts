from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .config import (
    ACCEPTED_STATUSES,
    CALIBRATION_STATUS,
    DEVICE_ERROR_STATUS,
    EXCEEDANCE_STATUS,
    POLLUTANTS,
    QCOL,
    SHEET,
    STATUS_CHANNELS,
    TIME_COL,
)


@dataclass(frozen=True)
class PeriodMasks:
    development_2025: pd.Series
    holdout_2026: pd.Series
    jan_jul_2025: pd.Series
    jan_jul_2026: pd.Series


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the cleaned workbook and add time fields used by the analyses."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_excel(path, sheet_name=SHEET, engine="openpyxl")
    if TIME_COL not in df or QCOL not in df:
        raise KeyError(f"Expected columns {TIME_COL!r} and {QCOL!r} in sheet {SHEET!r}.")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    if df[TIME_COL].isna().any():
        n_bad = int(df[TIME_COL].isna().sum())
        raise ValueError(f"Found {n_bad} unparseable timestamps in {TIME_COL!r}.")

    if df[TIME_COL].duplicated().any():
        n_dup = int(df[TIME_COL].duplicated().sum())
        raise ValueError(f"Expected unique timestamps; found {n_dup} duplicates.")

    df = df.sort_values(TIME_COL).reset_index(drop=True)
    df[QCOL] = pd.to_numeric(df[QCOL], errors="coerce")
    df["time"] = df[TIME_COL]
    df["date"] = df["time"].dt.floor("D")
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.to_period("M").astype(str)
    df["quarter"] = df["time"].dt.to_period("Q").astype(str)
    return df


def period_masks(df: pd.DataFrame) -> PeriodMasks:
    development_2025 = df["year"].eq(2025)
    holdout_2026 = df["year"].eq(2026) & df["time"].lt(pd.Timestamp("2026-08-01"))
    jan_jul_2025 = df["time"].ge(pd.Timestamp("2025-01-01")) & df["time"].lt(pd.Timestamp("2025-08-01"))
    jan_jul_2026 = df["time"].ge(pd.Timestamp("2026-01-01")) & df["time"].lt(pd.Timestamp("2026-08-01"))
    return PeriodMasks(development_2025, holdout_2026, jan_jul_2025, jan_jul_2026)


def valid_pollutant(
    df: pd.DataFrame,
    pollutant: str,
    mask: Optional[pd.Series | np.ndarray] = None,
    qcut: float = 0.0,
) -> pd.DataFrame:
    """Return the analysis-eligible positive-flow records for one pollutant.

    Primary eligibility follows the manuscript: accepted sensor status, positive
    chemistry, and effluent flow greater than the requested flow cutoff.
    """
    if pollutant not in POLLUTANTS:
        raise KeyError(f"Unknown pollutant {pollutant!r}; choose from {list(POLLUTANTS)}")

    spec = POLLUTANTS[pollutant]
    ccol, scol = spec["value_col"], spec["status_col"]
    c = pd.to_numeric(df[ccol], errors="coerce")
    q = pd.to_numeric(df[QCOL], errors="coerce")
    m = df[scol].isin(ACCEPTED_STATUSES) & c.gt(0) & q.gt(qcut)
    if mask is not None:
        m &= np.asarray(mask, dtype=bool)

    out = df.loc[m, ["time", "date", "year", "month", "quarter"]].copy()
    out["C"] = c.loc[m].to_numpy(float)
    out["Q"] = q.loc[m].to_numpy(float)
    out["load_rate"] = out["C"] * out["Q"] / 1000.0
    out["interval_mass"] = out["load_rate"] * (5.0 / 60.0)
    return out.reset_index(drop=True)


def primary_valid_mask(df: pd.DataFrame, value_col: str, status_col: str) -> pd.Series:
    c = pd.to_numeric(df[value_col], errors="coerce")
    return df[status_col].isin(ACCEPTED_STATUSES) & c.gt(0)


def status_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce Supplementary Table S1 status-aware QC summary."""
    rows = []
    for label, (value_col, status_col) in STATUS_CHANNELS.items():
        primary = primary_valid_mask(df, value_col, status_col)
        flagged = primary & df[status_col].eq(EXCEEDANCE_STATUS)
        calibration = df[status_col].eq(CALIBRATION_STATUS)
        device_error = df[status_col].eq(DEVICE_ERROR_STATUS)
        rows.append(
            {
                "Parameter": label,
                "Primary-valid n": int(primary.sum()),
                "Flagged n": int(flagged.sum()),
                "Flagged (%)": float(flagged.sum() / primary.sum() * 100.0) if primary.sum() else np.nan,
                "Calibration-labelled (h)": float(calibration.sum() * 5.0 / 60.0),
                "Device-error (h)": float(device_error.sum() * 5.0 / 60.0),
            }
        )
    return pd.DataFrame(rows)


def calendar_coverage(df: pd.DataFrame) -> dict:
    """Return overall, period, and monthly five-minute calendar coverage."""
    first = df["time"].min()
    last = df["time"].max()
    full = pd.date_range(first, last, freq="5min")
    observed = pd.DatetimeIndex(df["time"])

    overall = len(observed) / len(full)
    masks = period_masks(df)

    def expected_between(start: str, end_exclusive: str) -> int:
        return len(pd.date_range(pd.Timestamp(start), pd.Timestamp(end_exclusive) - pd.Timedelta(minutes=5), freq="5min"))

    n25_obs = int(masks.development_2025.sum())
    n26_obs = int(masks.holdout_2026.sum())
    n25_exp = expected_between("2025-01-01", "2026-01-01")
    n26_exp = expected_between("2026-01-01", "2026-08-01")

    periods = {
        "overall": {"observed": int(len(observed)), "expected": int(len(full)), "coverage": float(overall)},
        "2025": {"observed": n25_obs, "expected": n25_exp, "coverage": float(n25_obs / n25_exp)},
        "2026_JanJul": {"observed": n26_obs, "expected": n26_exp, "coverage": float(n26_obs / n26_exp)},
    }

    monthly_rows = []
    month_range = pd.period_range(first.to_period("M"), last.to_period("M"), freq="M")
    counts = df.groupby("month").size()
    for p in month_range:
        start = p.start_time
        end = p.end_time.floor("min")
        expected = len(pd.date_range(start, end, freq="5min"))
        key = str(p)
        obs = int(counts.get(key, 0))
        monthly_rows.append({"month": key, "observed": obs, "expected": expected, "coverage": obs / expected})

    return {
        "first_timestamp": str(first),
        "last_timestamp": str(last),
        "periods": periods,
        "monthly": pd.DataFrame(monthly_rows),
    }


def monthly_status_fractions(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly station-configured exceedance fractions among primary-valid records."""
    rows = []
    for label, (value_col, status_col) in STATUS_CHANNELS.items():
        primary = primary_valid_mask(df, value_col, status_col)
        flagged = primary & df[status_col].eq(EXCEEDANCE_STATUS)
        tmp = pd.DataFrame({"month": df["month"], "primary": primary.astype(int), "flagged": flagged.astype(int)})
        grp = tmp.groupby("month", as_index=False).sum()
        grp["flagged_fraction"] = np.where(grp["primary"].gt(0), grp["flagged"] / grp["primary"], np.nan)
        grp["parameter"] = label
        rows.append(grp[["month", "parameter", "primary", "flagged", "flagged_fraction"]])
    return pd.concat(rows, ignore_index=True)


def zero_flow_summary(df: pd.DataFrame) -> dict:
    """Describe observed zero/near-zero Flow out states and zero-flow run lengths."""
    q = pd.to_numeric(df[QCOL], errors="coerce")
    counts = {
        "q_eq_0": int(q.eq(0).sum()),
        "0_lt_q_le_0_5": int((q.gt(0) & q.le(0.5)).sum()),
        "0_5_lt_q_le_1": int((q.gt(0.5) & q.le(1.0)).sum()),
        "q_gt_1": int(q.gt(1.0).sum()),
    }

    x = df[["time", QCOL]].sort_values("time").reset_index(drop=True)
    runs = []
    current = 0
    prev = None
    for t, qv in x[["time", QCOL]].itertuples(index=False, name=None):
        contiguous = prev is not None and (t - prev) == pd.Timedelta(minutes=5)
        if not contiguous and current:
            runs.append(current)
            current = 0
        if pd.notna(qv) and qv == 0:
            current += 1
        else:
            if current:
                runs.append(current)
                current = 0
        prev = t
    if current:
        runs.append(current)

    arr = np.asarray(runs, float) * 5.0
    return {
        **counts,
        "zero_flow_runs": int(len(arr)),
        "median_run_min": float(np.median(arr)) if len(arr) else np.nan,
        "p90_run_min": float(np.percentile(arr, 90)) if len(arr) else np.nan,
        "max_run_min": float(np.max(arr)) if len(arr) else np.nan,
    }
