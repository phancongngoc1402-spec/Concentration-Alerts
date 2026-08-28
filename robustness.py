from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import (
    ALERT_GRID_STEP,
    HIGH_COVERAGE_THRESHOLD,
    INFLUENTIAL_DAY,
    LOAD_FRACTIONS,
    LOW_FLOW_CUTS,
    POLLUTANTS,
    PRIMARY_LOAD_FRACTION,
    PRIMARY_RECALL_TARGET,
    RECALL_TARGETS,
)
from .core import (
    alert_requirement,
    evaluate_fixed_mass_target,
    evaluate_threshold,
    p99_mass_boundary,
    top_metrics,
)
from .data import calendar_coverage, period_masks, valid_pollutant


def primary_sample_table(df: pd.DataFrame) -> pd.DataFrame:
    masks = period_masks(df)
    rows = []
    for pol in POLLUTANTS:
        overall = valid_pollutant(df, pol)
        y25 = valid_pollutant(df, pol, masks.development_2025)
        y26 = valid_pollutant(df, pol, masks.holdout_2026)
        rows.append(
            {
                "Channel": pol,
                "Valid n overall": len(overall),
                "Valid n 2025": len(y25),
                "Valid n 2026": len(y26),
                "Median C 2025": float(y25["C"].median()),
                "Median C 2026": float(y26["C"].median()),
                "Median Q 2025": float(y25["Q"].median()),
                "Median Q 2026": float(y26["Q"].median()),
            }
        )
    return pd.DataFrame(rows)


def decision_grid(df: pd.DataFrame) -> pd.DataFrame:
    masks = period_masks(df)
    rows = []
    for pol in POLLUTANTS:
        for period, mask in [("2025", masks.development_2025), ("2026", masks.holdout_2026)]:
            dat = valid_pollutant(df, pol, mask)
            for lf in LOAD_FRACTIONS:
                tm = top_metrics(dat, lf)
                for rt in RECALL_TARGETS:
                    req = alert_requirement(dat, load_frac=lf, recall_target=rt, step=ALERT_GRID_STEP)
                    rows.append(
                        {
                            "Channel": pol,
                            "Period": period,
                            "Load target": lf,
                            "Recall target": rt,
                            "Overlap": tm["overlap"],
                            "Jaccard": tm["jaccard"],
                            "Nominal alert fraction": req["nominal_alert_fraction"],
                            "Threshold": req["threshold"],
                            "Realized alert fraction": req["realized_alert_fraction"],
                        }
                    )
    return pd.DataFrame(rows)


def overlap_grid(df: pd.DataFrame) -> pd.DataFrame:
    masks = period_masks(df)
    rows = []
    for pol in POLLUTANTS:
        y25 = valid_pollutant(df, pol, masks.development_2025)
        y26 = valid_pollutant(df, pol, masks.holdout_2026)
        for lf in LOAD_FRACTIONS:
            m25 = top_metrics(y25, lf)
            m26 = top_metrics(y26, lf)
            rows.append(
                {
                    "Channel": pol,
                    "Top fraction": lf,
                    "Overlap 2025": m25["overlap"],
                    "Jaccard 2025": m25["jaccard"],
                    "Overlap 2026": m26["overlap"],
                    "Jaccard 2026": m26["jaccard"],
                }
            )
    return pd.DataFrame(rows)


def locked_holdout_point_estimates(df: pd.DataFrame) -> pd.DataFrame:
    masks = period_masks(df)
    rows = []
    for pol in POLLUTANTS:
        y25 = valid_pollutant(df, pol, masks.development_2025)
        y26 = valid_pollutant(df, pol, masks.holdout_2026)
        req = alert_requirement(y25)
        e = evaluate_threshold(y26, req["threshold"])
        rows.append(
            {
                "Channel": pol,
                "2025 threshold": req["threshold"],
                "2025 nominal alert fraction": req["nominal_alert_fraction"],
                "2026 n": e["n"],
                "2026 alert rate": e["alert_rate"],
                "2026 recall": e["recall"],
                "2026 precision": e["precision"],
                "2026 target-mass capture": e["target_mass_capture"],
            }
        )
    return pd.DataFrame(rows)


def season_matched(df: pd.DataFrame) -> pd.DataFrame:
    masks = period_masks(df)
    rows = []
    for pol in POLLUTANTS:
        dev = valid_pollutant(df, pol, masks.jan_jul_2025)
        hold = valid_pollutant(df, pol, masks.jan_jul_2026)
        req = alert_requirement(dev)
        e = evaluate_threshold(hold, req["threshold"])
        rows.append(
            {
                "Channel": pol,
                "Jan-Jul 2025 threshold": req["threshold"],
                "2026 alert rate": e["alert_rate"],
                "2026 recall": e["recall"],
                "2026 precision": e["precision"],
            }
        )
    return pd.DataFrame(rows)


def monthly_fixed_holdout(df: pd.DataFrame) -> pd.DataFrame:
    masks = period_masks(df)
    rows = []
    for pol in POLLUTANTS:
        y25 = valid_pollutant(df, pol, masks.development_2025)
        threshold = alert_requirement(y25)["threshold"]
        for month in [f"2026-{m:02d}" for m in range(1, 8)]:
            dat = valid_pollutant(df, pol, df["month"].eq(month))
            e = evaluate_threshold(dat, threshold)
            rows.append(
                {
                    "Channel": pol,
                    "Evaluation month": month,
                    "Threshold": threshold,
                    "Alert": e["alert_rate"],
                    "Recall": e["recall"],
                    "Precision": e["precision"],
                }
            )
    return pd.DataFrame(rows)


def expanding_origin(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pol in POLLUTANTS:
        for month_num in range(1, 8):
            month = f"2026-{month_num:02d}"
            start = pd.Timestamp(f"2026-{month_num:02d}-01")
            end = start + pd.offsets.MonthBegin(1)
            dev_mask = df["time"].lt(start)
            eval_mask = df["time"].ge(start) & df["time"].lt(end)
            dev = valid_pollutant(df, pol, dev_mask)
            eva = valid_pollutant(df, pol, eval_mask)
            req = alert_requirement(dev)
            e = evaluate_threshold(eva, req["threshold"])
            rows.append(
                {
                    "Channel": pol,
                    "Evaluation month": month,
                    "Updated threshold": req["threshold"],
                    "Alert": e["alert_rate"],
                    "Recall": e["recall"],
                    "Precision": e["precision"],
                }
            )
    return pd.DataFrame(rows)


def quarter_fixed_holdout(df: pd.DataFrame) -> pd.DataFrame:
    masks = period_masks(df)
    windows = {
        "Q1": (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-01")),
        "Q2": (pd.Timestamp("2026-04-01"), pd.Timestamp("2026-07-01")),
        "July": (pd.Timestamp("2026-07-01"), pd.Timestamp("2026-08-01")),
    }
    rows = []
    for pol in POLLUTANTS:
        y25 = valid_pollutant(df, pol, masks.development_2025)
        threshold = alert_requirement(y25)["threshold"]
        for label, (start, end) in windows.items():
            dat = valid_pollutant(df, pol, df["time"].ge(start) & df["time"].lt(end))
            e = evaluate_threshold(dat, threshold)
            rows.append(
                {
                    "Channel": pol,
                    "Window": label,
                    "Alert": e["alert_rate"],
                    "Recall": e["recall"],
                    "Precision": e["precision"],
                }
            )
    return pd.DataFrame(rows)


def low_flow_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    masks = period_masks(df)
    rows = []
    for pol in POLLUTANTS:
        for qcut in LOW_FLOW_CUTS:
            y25 = valid_pollutant(df, pol, masks.development_2025, qcut=qcut)
            y26 = valid_pollutant(df, pol, masks.holdout_2026, qcut=qcut)
            req = alert_requirement(y25)
            e = evaluate_threshold(y26, req["threshold"])
            rows.append(
                {
                    "Channel": pol,
                    "Flow cutoff": qcut,
                    "2025 threshold": req["threshold"],
                    "2026 n": e["n"],
                    "Alert": e["alert_rate"],
                    "Recall": e["recall"],
                    "Precision": e["precision"],
                }
            )
    return pd.DataFrame(rows)


def influential_day_audit(df: pd.DataFrame) -> pd.DataFrame:
    masks = period_masks(df)
    d = pd.Timestamp(INFLUENTIAL_DAY)
    rows = []
    for pol in POLLUTANTS:
        y26 = valid_pollutant(df, pol, masks.holdout_2026 & df["date"].ne(d))
        tm = top_metrics(y26)
        req = alert_requirement(y26)
        rows.append(
            {
                "Channel": pol,
                "Top-1% overlap": tm["overlap"],
                "Alert fraction >=90% recall": req["nominal_alert_fraction"],
            }
        )
    return pd.DataFrame(rows)


def high_coverage_audit(df: pd.DataFrame) -> pd.DataFrame:
    coverage = calendar_coverage(df)["monthly"]
    eligible = set(coverage.loc[coverage["coverage"].ge(HIGH_COVERAGE_THRESHOLD), "month"])
    masks = period_masks(df)
    rows = []
    for pol in POLLUTANTS:
        dev_mask = masks.development_2025 & df["month"].isin(eligible)
        hold_mask = masks.holdout_2026 & df["month"].isin(eligible)
        dev = valid_pollutant(df, pol, dev_mask)
        hold = valid_pollutant(df, pol, hold_mask)
        req = alert_requirement(dev)
        e = evaluate_threshold(hold, req["threshold"])
        rows.append(
            {
                "Channel": pol,
                "Threshold": req["threshold"],
                "2026 alert": e["alert_rate"],
                "Recall": e["recall"],
                "Precision": e["precision"],
                "Eligible 2025 months": ",".join(sorted(m for m in eligible if m.startswith("2025-"))),
                "Eligible 2026 months": ",".join(sorted(m for m in eligible if m.startswith("2026-"))),
            }
        )
    return pd.DataFrame(rows)


def fixed_p99_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    masks = period_masks(df)
    rows = []
    for pol in POLLUTANTS:
        y25 = valid_pollutant(df, pol, masks.development_2025)
        y26 = valid_pollutant(df, pol, masks.holdout_2026)
        req = alert_requirement(y25)
        boundary = p99_mass_boundary(y25)
        e = evaluate_fixed_mass_target(y26, req["threshold"], boundary)
        rows.append(
            {
                "Channel": pol,
                "2025 P99 interval-mass boundary": boundary,
                "2026 target n": e["target_n"],
                "Locked concentration threshold": req["threshold"],
                "Alert": e["alert_rate"],
                "Recall": e["recall"],
                "Precision": e["precision"],
                "Target-mass capture": e["target_mass_capture"],
                "True positives": e["true_positives"],
            }
        )
    return pd.DataFrame(rows)
