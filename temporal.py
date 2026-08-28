from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .config import (
    ACCEPTED_STATUSES,
    EVENT_BOOTSTRAP_BASE_SEED,
    EVENT_BRIDGES_MIN,
    HOURLY_BOOTSTRAP_BASE_SEED,
    POLLUTANTS,
    POLLUTANT_OFFSETS,
    QCOL,
    SECONDARY_BOOTSTRAP_REPLICATES,
)
from .data import valid_pollutant


def hourly_error_records(dat: pd.DataFrame) -> pd.DataFrame:
    """Return complete positive-flow hours and product-of-means error records.

    The implementation is vectorized. A complete hour must contain all twelve
    unique five-minute slots (00, 05, ..., 55) on the expected time grid.
    """
    if len(dat) == 0:
        return pd.DataFrame(
            columns=[
                "hour", "date", "direct_mass", "approx_mass",
                "relative_error_pct", "absolute_error_pct",
            ]
        )

    x = dat[["time", "C", "Q", "interval_mass"]].copy()
    x["hour"] = x["time"].dt.floor("h")
    x["slot"] = x["time"].dt.minute // 5
    x["on_grid"] = (x["time"].dt.second.eq(0) & x["time"].dt.microsecond.eq(0) & x["time"].dt.minute.mod(5).eq(0))

    grp = x.groupby("hour", sort=True).agg(
        n=("time", "size"),
        unique_slots=("slot", "nunique"),
        on_grid=("on_grid", "all"),
        direct_mass=("interval_mass", "sum"),
        mean_c=("C", "mean"),
        mean_q=("Q", "mean"),
    )
    grp = grp.loc[grp["n"].eq(12) & grp["unique_slots"].eq(12) & grp["on_grid"]].copy()
    grp = grp.loc[grp["direct_mass"].gt(0)].copy()
    grp["approx_mass"] = grp["mean_c"] * grp["mean_q"] / 1000.0
    grp["relative_error_pct"] = (grp["approx_mass"] - grp["direct_mass"]) / grp["direct_mass"] * 100.0
    grp["absolute_error_pct"] = grp["relative_error_pct"].abs()
    grp = grp.reset_index()
    grp["date"] = grp["hour"].dt.floor("D")
    return grp[["hour", "date", "direct_mass", "approx_mass", "relative_error_pct", "absolute_error_pct"]]


def hourly_aggregation_metrics(records: pd.DataFrame) -> dict:
    if len(records) == 0:
        return {
            "complete_hours": 0,
            "p95_abs_error_pct": np.nan,
            "p99_abs_error_pct": np.nan,
            "cumulative_bias_pct": np.nan,
        }
    abs_err = records["absolute_error_pct"].to_numpy(float)
    direct = records["direct_mass"].to_numpy(float)
    approx = records["approx_mass"].to_numpy(float)
    return {
        "complete_hours": int(len(records)),
        "p95_abs_error_pct": float(np.percentile(abs_err, 95)),
        "p99_abs_error_pct": float(np.percentile(abs_err, 99)),
        "cumulative_bias_pct": float((approx.sum() - direct.sum()) / direct.sum() * 100.0),
    }


def bootstrap_hourly_metrics(records: pd.DataFrame, pollutant: str, B: int = SECONDARY_BOOTSTRAP_REPLICATES) -> dict:
    groups = [g.reset_index(drop=True) for _, g in records.groupby("date", sort=True)]
    rng = np.random.default_rng(HOURLY_BOOTSTRAP_BASE_SEED + POLLUTANT_OFFSETS[pollutant])
    vals = np.empty((B, 3), float)
    for b in range(B):
        idx = rng.integers(0, len(groups), size=len(groups))
        sample = pd.concat([groups[i] for i in idx], ignore_index=True)
        m = hourly_aggregation_metrics(sample)
        vals[b] = [m["p95_abs_error_pct"], m["p99_abs_error_pct"], m["cumulative_bias_pct"]]
    return {
        "B": int(B),
        "p95_ci95": np.nanpercentile(vals[:, 0], [2.5, 97.5]).tolist(),
        "p99_ci95": np.nanpercentile(vals[:, 1], [2.5, 97.5]).tolist(),
        "bias_ci95": np.nanpercentile(vals[:, 2], [2.5, 97.5]).tolist(),
    }


def hourly_summary(df: pd.DataFrame, B: int = SECONDARY_BOOTSTRAP_REPLICATES) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    record_map = {}
    for pol in POLLUTANTS:
        dat = valid_pollutant(df, pol)
        records = hourly_error_records(dat)
        record_map[pol] = records
        point = hourly_aggregation_metrics(records)
        boot = bootstrap_hourly_metrics(records, pol, B=B)
        rows.append(
            {
                "Channel": pol,
                "Complete hours": point["complete_hours"],
                "P95 |error| (%)": point["p95_abs_error_pct"],
                "P95 CI low": boot["p95_ci95"][0],
                "P95 CI high": boot["p95_ci95"][1],
                "P99 |error| (%)": point["p99_abs_error_pct"],
                "P99 CI low": boot["p99_ci95"][0],
                "P99 CI high": boot["p99_ci95"][1],
                "Cumulative bias (%)": point["cumulative_bias_pct"],
                "Bias CI low": boot["bias_ci95"][0],
                "Bias CI high": boot["bias_ci95"][1],
            }
        )
    return pd.DataFrame(rows), record_map


def _hydraulic_event_labels(df: pd.DataFrame, bridge_min: int):
    """Return event labels on the observed time grid and each event start date."""
    x = df.copy().sort_values("time").reset_index(drop=True)
    times = x["time"].to_numpy()
    q = pd.to_numeric(x[QCOL], errors="coerce").to_numpy(float)
    n = len(x)
    labels = np.full(n, -1, dtype=np.int32)
    max_slots = int(bridge_min / 5)
    current = -1
    next_event = 0
    zero_buffer = []
    start_indices = []

    for i in range(n):
        contiguous = i > 0 and (pd.Timestamp(times[i]) - pd.Timestamp(times[i - 1]) == pd.Timedelta(minutes=5))
        if i > 0 and not contiguous:
            current = -1
            zero_buffer = []

        if np.isfinite(q[i]) and q[i] > 0:
            if zero_buffer:
                if current >= 0 and len(zero_buffer) <= max_slots:
                    labels[np.asarray(zero_buffer, dtype=int)] = current
                else:
                    current = -1
                zero_buffer = []
            if current < 0:
                current = next_event
                next_event += 1
                start_indices.append(i)
            labels[i] = current
        else:
            if current >= 0:
                zero_buffer.append(i)

    start_dates = pd.to_datetime(times[np.asarray(start_indices, dtype=int)]).floor("D") if start_indices else pd.DatetimeIndex([])
    return x, labels, start_dates


def _pollutant_mass_array(sorted_df: pd.DataFrame, pollutant: str) -> np.ndarray:
    spec = POLLUTANTS[pollutant]
    c = pd.to_numeric(sorted_df[spec["value_col"]], errors="coerce").to_numpy(float)
    q = pd.to_numeric(sorted_df[QCOL], errors="coerce").to_numpy(float)
    status = sorted_df[spec["status_col"]].to_numpy(object)
    valid_status = np.isin(status, list(ACCEPTED_STATUSES))
    valid = valid_status & np.isfinite(c) & (c > 0) & np.isfinite(q) & (q > 0)
    out = np.zeros(len(sorted_df), dtype=float)
    out[valid] = c[valid] * q[valid] / 1000.0 * (5.0 / 60.0)
    return out


def event_records_from_labels(sorted_df: pd.DataFrame, labels: np.ndarray, start_dates, pollutant: str) -> pd.DataFrame:
    n_events = len(start_dates)
    mass = _pollutant_mass_array(sorted_df, pollutant)
    ok = labels >= 0
    event_mass = np.bincount(labels[ok], weights=mass[ok], minlength=n_events).astype(float)
    return pd.DataFrame(
        {
            "event_id": np.arange(n_events, dtype=int),
            "start_date": pd.DatetimeIndex(start_dates),
            "mass": event_mass,
        }
    )


def event_records(df: pd.DataFrame, bridge_min: int, pollutant: str) -> pd.DataFrame:
    sorted_df, labels, start_dates = _hydraulic_event_labels(df, bridge_min)
    return event_records_from_labels(sorted_df, labels, start_dates, pollutant)


def event_segmentation_summary(df: pd.DataFrame, bridges=EVENT_BRIDGES_MIN) -> tuple[pd.DataFrame, dict[tuple[str, int], pd.DataFrame]]:
    cache = {}
    rows = []
    for bridge in bridges:
        sorted_df, labels, start_dates = _hydraulic_event_labels(df, bridge)
        row = {"Max zero-flow bridge (min)": bridge, "Event n": int(len(start_dates))}
        for pol in POLLUTANTS:
            rec = event_records_from_labels(sorted_df, labels, start_dates, pol)
            cache[(pol, bridge)] = rec
            row[f"{pol} P90 mass"] = float(np.percentile(rec["mass"], 90))
        rows.append(row)
    return pd.DataFrame(rows), cache


def bootstrap_event_contrast(
    records0: pd.DataFrame,
    records30: pd.DataFrame,
    pollutant: str,
    B: int = SECONDARY_BOOTSTRAP_REPLICATES,
) -> dict:
    """Day-block bootstrap of event masses using event start date as block label."""
    days = sorted(set(records0["start_date"]).union(records30["start_date"]))
    g0 = {d: records0.loc[records0["start_date"].eq(d), "mass"].to_numpy(float) for d in days}
    g30 = {d: records30.loc[records30["start_date"].eq(d), "mass"].to_numpy(float) for d in days}
    rng = np.random.default_rng(EVENT_BOOTSTRAP_BASE_SEED + POLLUTANT_OFFSETS[pollutant])
    vals = np.empty((B, 3), float)
    for b in range(B):
        sampled = [days[i] for i in rng.integers(0, len(days), size=len(days))]
        pieces0 = [g0[d] for d in sampled if len(g0[d])]
        pieces30 = [g30[d] for d in sampled if len(g30[d])]
        a0 = np.concatenate(pieces0) if pieces0 else np.asarray([], float)
        a30 = np.concatenate(pieces30) if pieces30 else np.asarray([], float)
        p0 = float(np.percentile(a0, 90))
        p30 = float(np.percentile(a30, 90))
        rel = (p30 - p0) / p0 * 100.0 if p0 > 0 else np.nan
        vals[b] = [p0, p30, rel]
    return {
        "B": int(B),
        "p90_0_ci95": np.nanpercentile(vals[:, 0], [2.5, 97.5]).tolist(),
        "p90_30_ci95": np.nanpercentile(vals[:, 1], [2.5, 97.5]).tolist(),
        "relative_change_ci95": np.nanpercentile(vals[:, 2], [2.5, 97.5]).tolist(),
    }


def event_uncertainty_table(df: pd.DataFrame, event_cache=None, B: int = SECONDARY_BOOTSTRAP_REPLICATES) -> pd.DataFrame:
    if event_cache is None:
        _, event_cache = event_segmentation_summary(df, bridges=(0, 30))
    rows = []
    for pol in POLLUTANTS:
        r0 = event_cache.get((pol, 0))
        r30 = event_cache.get((pol, 30))
        if r0 is None:
            r0 = event_records(df, 0, pol)
        if r30 is None:
            r30 = event_records(df, 30, pol)
        p0 = float(np.percentile(r0["mass"], 90))
        p30 = float(np.percentile(r30["mass"], 90))
        rel = (p30 - p0) / p0 * 100.0
        boot = bootstrap_event_contrast(r0, r30, pol, B=B)
        rows.append(
            {
                "Channel": pol,
                "P90 mass 0-min": p0,
                "P90 0-min CI low": boot["p90_0_ci95"][0],
                "P90 0-min CI high": boot["p90_0_ci95"][1],
                "P90 mass 30-min": p30,
                "P90 30-min CI low": boot["p90_30_ci95"][0],
                "P90 30-min CI high": boot["p90_30_ci95"][1],
                "Relative change (%)": rel,
                "Relative change CI low": boot["relative_change_ci95"][0],
                "Relative change CI high": boot["relative_change_ci95"][1],
            }
        )
    return pd.DataFrame(rows)
