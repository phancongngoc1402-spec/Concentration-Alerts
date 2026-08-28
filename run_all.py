#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.bootstrap import (
    bootstrap_fixed_holdout,
    bootstrap_revision_metrics,
    bootstrap_year_metrics,
    moving_block_holdout,
    nested_threshold_bootstrap,
)
from analysis.config import (
    POLLUTANTS,
    PRIMARY_BOOTSTRAP_REPLICATES,
    REVISION_METRICS_BOOTSTRAP_SEED,
    SECONDARY_BOOTSTRAP_REPLICATES,
)
from analysis.core import alert_requirement, jsd_2d, p99_mass_boundary, spearman_cq, top_metrics
from analysis.data import (
    calendar_coverage,
    load_dataset,
    period_masks,
    status_audit,
    valid_pollutant,
    zero_flow_summary,
)
from analysis.figures import (
    figure1_recall_alert_curves,
    figure2_monthly_transfer,
    figure3_joint_regime_shift,
    figure_s1_coverage_status,
    figure_s2_hourly_error,
    figure_s3_event_mass,
)
from analysis.robustness import (
    decision_grid,
    expanding_origin,
    fixed_p99_sensitivity,
    high_coverage_audit,
    influential_day_audit,
    locked_holdout_point_estimates,
    low_flow_sensitivity,
    monthly_fixed_holdout,
    overlap_grid,
    primary_sample_table,
    quarter_fixed_holdout,
    season_matched,
)
from analysis.temporal import event_segmentation_summary, event_uncertainty_table, hourly_summary


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.replace({np.nan: None}).to_dict(orient="records")
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def main():
    ap = argparse.ArgumentParser(description="Reproduce manuscript analyses, tables, and figures.")
    ap.add_argument("--input", default="Dataset.xlsx", help="Path to the cleaned Dataset.xlsx workbook")
    ap.add_argument("--output", default="outputs", help="Directory for regenerated tables and figures")
    ap.add_argument("--bootstrap", type=int, default=PRIMARY_BOOTSTRAP_REPLICATES, help="Primary bootstrap replicates (manuscript: 1000)")
    ap.add_argument("--secondary-bootstrap", type=int, default=SECONDARY_BOOTSTRAP_REPLICATES, help="Secondary bootstrap replicates (manuscript: 500)")
    ap.add_argument("--skip-secondary", action="store_true", help="Skip hourly/event secondary analyses for a faster smoke run")
    ap.add_argument("--skip-figures", action="store_true", help="Do not regenerate PNG figures")
    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    out = Path(args.output).resolve()
    main_dir = out / "main"
    supp_dir = out / "supplement"
    fig_dir = out / "figures"
    main_dir.mkdir(parents=True, exist_ok=True)
    supp_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path} ...")
    df = load_dataset(input_path)
    masks = period_masks(df)

    results = {
        "metadata": {
            "input": str(input_path),
            "sha256": sha256(input_path),
            "rows": int(len(df)),
            "first_timestamp": str(df["time"].min()),
            "last_timestamp": str(df["time"].max()),
            "primary_bootstrap_replicates": int(args.bootstrap),
            "secondary_bootstrap_replicates": int(args.secondary_bootstrap),
            "revision_metrics_bootstrap_seed": int(REVISION_METRICS_BOOTSTRAP_SEED),
            "manuscript_alignment": "revised submission-ready v2",
            "NH4N_note": "Workbook field name is legacy NH4+; archived study channel is interpreted as NH4-N (as N), with no chemical-form conversion.",
        }
    }

    # ------------------------------------------------------------------
    # Main manuscript tables
    # ------------------------------------------------------------------
    table1 = primary_sample_table(df)
    save_csv(table1, main_dir / "Table1_primary_sample.csv")

    table2_rows = []
    table3_rows = []
    drift_rows = []
    overlap_diff_rows = []
    nested_rows = []
    moving_rows = []

    # Prepare the locked development quantities once. The revision-added
    # bootstrap uses one documented RNG stream across pollutants so its output
    # exactly matches the target-mass-capture and fixed-P99 uncertainty values
    # reported in the revised manuscript.
    prepared = {}
    revision_payloads = {}
    for pol in POLLUTANTS:
        y25 = valid_pollutant(df, pol, masks.development_2025)
        y26 = valid_pollutant(df, pol, masks.holdout_2026)
        req25 = alert_requirement(y25)
        prepared[pol] = {"y25": y25, "y26": y26, "req25": req25}
        revision_payloads[pol] = {
            "y26": y26,
            "threshold": req25["threshold"],
            "mass_boundary": p99_mass_boundary(y25),
        }
    revision_boot = bootstrap_revision_metrics(revision_payloads, B=args.bootstrap)

    for pol in POLLUTANTS:
        y25 = prepared[pol]["y25"]
        y26 = prepared[pol]["y26"]
        req25 = prepared[pol]["req25"]
        tm25 = top_metrics(y25)
        tm26 = top_metrics(y26)
        req26 = alert_requirement(y26)

        boot25 = bootstrap_year_metrics(y25, pol, "2025", B=args.bootstrap, return_samples=True)
        boot26 = bootstrap_year_metrics(y26, pol, "2026", B=args.bootstrap, return_samples=True)
        table2_rows.append(
            {
                "Channel": pol,
                "Top-1% overlap 2025": tm25["overlap"],
                "Top-1% overlap 2025 CI low": boot25["overlap_ci95"][0],
                "Top-1% overlap 2025 CI high": boot25["overlap_ci95"][1],
                "Top-1% overlap 2026": tm26["overlap"],
                "Top-1% overlap 2026 CI low": boot26["overlap_ci95"][0],
                "Top-1% overlap 2026 CI high": boot26["overlap_ci95"][1],
                "Jaccard 2025": tm25["jaccard"],
                "Jaccard 2026": tm26["jaccard"],
                "Alert fraction >=90% recall 2025": req25["nominal_alert_fraction"],
                "Alert fraction 2025 CI low": boot25["burden_ci95"][0],
                "Alert fraction 2025 CI high": boot25["burden_ci95"][1],
                "Alert fraction >=90% recall 2026": req26["nominal_alert_fraction"],
                "Alert fraction 2026 CI low": boot26["burden_ci95"][0],
                "Alert fraction 2026 CI high": boot26["burden_ci95"][1],
            }
        )

        hold = locked_holdout_point_estimates(df).loc[lambda x: x["Channel"].eq(pol)].iloc[0]
        bfix = bootstrap_fixed_holdout(y26, float(hold["2025 threshold"]), pol, B=args.bootstrap)
        table3_rows.append(
            {
                "Channel": pol,
                "2025 threshold": float(hold["2025 threshold"]),
                "2026 alert rate": float(hold["2026 alert rate"]),
                "2026 alert CI low": bfix["alert_rate_ci95"][0],
                "2026 alert CI high": bfix["alert_rate_ci95"][1],
                "2026 recall": float(hold["2026 recall"]),
                "2026 recall CI low": bfix["recall_ci95"][0],
                "2026 recall CI high": bfix["recall_ci95"][1],
                "2026 precision": float(hold["2026 precision"]),
                "2026 precision CI low": bfix["precision_ci95"][0],
                "2026 precision CI high": bfix["precision_ci95"][1],
                "2026 target-mass capture": float(hold["2026 target-mass capture"]),
                "2026 target-mass capture CI low": revision_boot[pol]["period_relative_target_mass_capture_median_ci95"][0],
                "2026 target-mass capture CI high": revision_boot[pol]["period_relative_target_mass_capture_median_ci95"][2],
            }
        )

        j_all = jsd_2d(y25, y26)
        jj25 = valid_pollutant(df, pol, masks.jan_jul_2025)
        jj26 = valid_pollutant(df, pol, masks.jan_jul_2026)
        j_sm = jsd_2d(jj25, jj26)
        drift_rows.append(
            {
                "Channel": pol,
                "Jensen-Shannon 2025 vs 2026": j_all,
                "Jensen-Shannon Jan-Jul 2025 vs Jan-Jul 2026": j_sm,
                "Spearman C-Q 2025": spearman_cq(y25),
                "Spearman C-Q 2026": spearman_cq(y26),
                "Median C 2025": y25["C"].median(),
                "Median C 2026": y26["C"].median(),
                "Median Q 2025": y25["Q"].median(),
                "Median Q 2026": y26["Q"].median(),
            }
        )

        diff_samples = boot26["_overlap_samples"] - boot25["_overlap_samples"]
        diff_ci = np.nanpercentile(diff_samples, [2.5, 97.5])
        overlap_diff_rows.append(
            {
                "Channel": pol,
                "Overlap difference 2026-2025": tm26["overlap"] - tm25["overlap"],
                "CI low": float(diff_ci[0]),
                "CI high": float(diff_ci[1]),
            }
        )

        nested = nested_threshold_bootstrap(y25, y26, pol, B=args.bootstrap)
        nested_rows.append(
            {
                "Channel": pol,
                "Threshold median": nested["threshold_median_ci95"][1],
                "Threshold CI low": nested["threshold_median_ci95"][0],
                "Threshold CI high": nested["threshold_median_ci95"][2],
                "Alert median": nested["alert_rate_median_ci95"][1],
                "Alert CI low": nested["alert_rate_median_ci95"][0],
                "Alert CI high": nested["alert_rate_median_ci95"][2],
                "Recall median": nested["recall_median_ci95"][1],
                "Recall CI low": nested["recall_median_ci95"][0],
                "Recall CI high": nested["recall_median_ci95"][2],
                "Precision median": nested["precision_median_ci95"][1],
                "Precision CI low": nested["precision_median_ci95"][0],
                "Precision CI high": nested["precision_median_ci95"][2],
            }
        )

        mb = moving_block_holdout(y26, float(hold["2025 threshold"]), pol, B=args.secondary_bootstrap)
        moving_rows.append(mb)

    table2 = pd.DataFrame(table2_rows)
    table3 = pd.DataFrame(table3_rows)
    table4 = season_matched(df)
    drift = pd.DataFrame(drift_rows)
    save_csv(table2, main_dir / "Table2_concentration_load_mismatch.csv")
    save_csv(table3, main_dir / "Table3_locked_threshold_holdout.csv")
    save_csv(table4, main_dir / "Table4_season_matched.csv")
    save_csv(drift, main_dir / "Joint_distribution_shift.csv")

    # ------------------------------------------------------------------
    # Supplementary tables and robustness analyses
    # ------------------------------------------------------------------
    coverage = calendar_coverage(df)
    s1 = status_audit(df)
    save_csv(s1, supp_dir / "TableS1_status_audit.csv")
    save_csv(coverage["monthly"], supp_dir / "Monthly_calendar_coverage.csv")

    grid = decision_grid(df)
    s2a = grid.loc[grid["Period"].eq("2025")].copy()
    s2b = grid.loc[grid["Period"].eq("2026")].copy()
    save_csv(s2a, supp_dir / "TableS2a_decision_grid_2025.csv")
    save_csv(s2b, supp_dir / "TableS2b_decision_grid_2026.csv")

    s3 = overlap_grid(df)
    save_csv(s3, supp_dir / "TableS3_overlap_jaccard.csv")
    save_csv(pd.DataFrame(overlap_diff_rows), supp_dir / "TableS3b_overlap_difference_bootstrap.csv")

    s4 = expanding_origin(df)
    qtr = quarter_fixed_holdout(df)
    save_csv(s4, supp_dir / "TableS4_expanding_origin.csv")
    save_csv(qtr, supp_dir / "TableS4b_quarter_fixed_threshold.csv")

    s5 = low_flow_sensitivity(df)
    save_csv(s5, supp_dir / "TableS5_low_flow_sensitivity.csv")

    inf = influential_day_audit(df)
    high = high_coverage_audit(df)
    s6 = inf.merge(high, on="Channel", how="outer", suffixes=("_13Jul", "_highcoverage"))
    save_csv(s6, supp_dir / "TableS6_influential_day_high_coverage.csv")

    s7 = fixed_p99_sensitivity(df)
    save_csv(s7, supp_dir / "TableS7_fixed_P99_target.csv")

    s7_indexed = s7.set_index("Channel")
    s7b_rows = []
    for pol in POLLUTANTS:
        rb = revision_boot[pol]
        target_q = rb["fixed_target_n_median_ci95"]
        recall_q = rb["fixed_recall_median_ci95"]
        precision_q = rb["fixed_precision_median_ci95"]
        capture_q = rb["fixed_target_mass_capture_median_ci95"]
        alert_q = rb["fixed_alert_rate_median_ci95"]
        s7b_rows.append(
            {
                "Channel": pol,
                "Observed 2026 target n": int(s7_indexed.loc[pol, "2026 target n"]),
                "Bootstrap target n median": target_q[1],
                "Bootstrap target n CI low": target_q[0],
                "Bootstrap target n CI high": target_q[2],
                "Recall median": recall_q[1],
                "Recall CI low": recall_q[0],
                "Recall CI high": recall_q[2],
                "Precision median": precision_q[1],
                "Precision CI low": precision_q[0],
                "Precision CI high": precision_q[2],
                "Target-mass capture median": capture_q[1],
                "Target-mass capture CI low": capture_q[0],
                "Target-mass capture CI high": capture_q[2],
                "Alert rate median": alert_q[1],
                "Alert rate CI low": alert_q[0],
                "Alert rate CI high": alert_q[2],
                "Empty-target bootstrap replicates": rb["fixed_empty_target_replicates"],
            }
        )
    s7b = pd.DataFrame(s7b_rows)
    save_csv(s7b, supp_dir / "TableS7b_fixed_P99_bootstrap.csv")

    s8 = pd.DataFrame(nested_rows)
    s8b = pd.concat(moving_rows, ignore_index=True)
    save_csv(s8, supp_dir / "TableS8_nested_threshold_bootstrap.csv")
    save_csv(s8b, supp_dir / "TableS8b_moving_block_sensitivity.csv")

    zflow = zero_flow_summary(df)

    # ------------------------------------------------------------------
    # Secondary temporal-representation analyses
    # ------------------------------------------------------------------
    hourly = None
    events = None
    event_unc = None
    if not args.skip_secondary:
        print("Running hourly aggregation and event segmentation analyses ...")
        hourly, hourly_records = hourly_summary(df, B=args.secondary_bootstrap)
        save_csv(hourly, supp_dir / "TableS9_hourly_aggregation.csv")

        events, event_cache = event_segmentation_summary(df)
        save_csv(events, supp_dir / "TableS10_event_segmentation.csv")

        event_unc = event_uncertainty_table(df, event_cache=event_cache, B=args.secondary_bootstrap)
        save_csv(event_unc, supp_dir / "TableS11_event_bootstrap.csv")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    monthly_fixed = monthly_fixed_holdout(df)
    save_csv(monthly_fixed, supp_dir / "Month_level_locked_threshold.csv")

    if not args.skip_figures:
        print("Rendering manuscript figures ...")
        figure1_recall_alert_curves(df, fig_dir / "Figure1_recall_alert_curves.png")
        figure2_monthly_transfer(monthly_fixed, fig_dir / "Figure2_monthly_transfer.png")
        figure3_joint_regime_shift(df, fig_dir / "Figure3_joint_regime_shift.png")
        figure_s1_coverage_status(df, fig_dir / "FigureS1_coverage_status.png")
        if hourly is not None:
            figure_s2_hourly_error(hourly, fig_dir / "FigureS2_hourly_aggregation_error.png")
        if events is not None:
            figure_s3_event_mass(events, fig_dir / "FigureS3_event_mass_segmentation.png")

    results.update(
        {
            "coverage": coverage["periods"],
            "zero_flow_summary": zflow,
            "main_table1": table1,
            "main_table2": table2,
            "main_table3": table3,
            "main_table4": table4,
            "joint_distribution_shift": drift,
            "supplement_tableS7_fixed_P99": s7,
            "supplement_tableS7b_fixed_P99_bootstrap": s7b,
            "revision_bootstrap_details": revision_boot,
            "supplement_tableS8_nested": s8,
            "supplement_tableS8b_moving_blocks": s8b,
            "secondary_hourly": hourly if hourly is not None else [],
            "secondary_events": events if events is not None else [],
            "secondary_event_uncertainty": event_unc if event_unc is not None else [],
        }
    )

    (out / "analysis_results.json").write_text(
        json.dumps(_json_safe(results), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Done. Outputs written to {out}")


if __name__ == "__main__":
    main()
