#!/usr/bin/env python3
"""Validate deterministic manuscript results against the supplied workbook.

This is deliberately a point-estimate validator. Bootstrap percentile endpoints
are produced by ``run_all.py`` under the fixed seeds in ``analysis/config.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path

import numpy as np

from analysis.bootstrap import bootstrap_revision_metrics
from analysis.core import alert_requirement, p99_mass_boundary
from analysis.data import load_dataset, period_masks, valid_pollutant
from analysis.robustness import (
    fixed_p99_sensitivity,
    locked_holdout_point_estimates,
    primary_sample_table,
    season_matched,
)
from analysis.temporal import event_segmentation_summary, hourly_summary

EXPECTED_SHA256 = "253fd20f2e26f9e5b5cae57a4ed1140433f01ef141805e20b7b1f11b226fa283"
EXPECTED_ROWS = 148_906

EXPECTED_SAMPLE = {
    "COD": (113_068, 72_391, 40_677, 31.14, 21.77, 137.0, 103.9),
    "TSS": (114_780, 72_926, 41_854, 24.79, 22.44, 137.0, 104.8),
    "NH4-N": (114_627, 72_757, 41_870, 1.26, 2.82, 137.0, 104.7),
}

EXPECTED_LOCKED = {
    "COD": (47.43, 0.0120, 0.491, 0.409, 0.570),
    "TSS": (35.44, 0.0604, 0.599, 0.099, 0.671),
    "NH4-N": (3.31, 0.283, 0.993, 0.035, 0.996),
}

EXPECTED_SEASON = {
    "COD": (51.08, 0.0102, 0.410, 0.401),
    "TSS": (35.90, 0.0524, 0.568, 0.109),
    "NH4-N": (3.31, 0.283, 0.993, 0.035),
}

EXPECTED_P99 = {
    "COD": (0.795872, 26, 47.43, 1.000, 0.053, 1.000, 26),
    "TSS": (0.614351, 83, 35.44, 0.952, 0.031, 0.974, 79),
    "NH4-N": (0.052219, 1088, 3.31, 0.927, 0.085, 0.946, 1009),
}

EXPECTED_EVENTS = {
    0: (4622, 17.05, 15.00, 1.142),
    5: (4105, 19.28, 17.16, 1.270),
    10: (3860, 20.38, 17.76, 1.304),
    15: (3664, 21.59, 18.99, 1.343),
    30: (3339, 22.31, 20.28, 1.425),
    60: (2972, 21.44, 19.94, 1.273),
}


EXPECTED_REVISION_BOOTSTRAP = {
    "COD": {
        "period_tmc": (0.18929768, 0.55073775, 0.83746899),
        "fixed_n": (1.0, 24.0, 68.0),
        "fixed_recall": (1.0, 1.0, 1.0),
        "fixed_precision": (0.00358328, 0.04758131, 0.19401484),
        "fixed_tmc": (1.0, 1.0, 1.0),
        "empty": 6,
    },
    "TSS": {
        "period_tmc": (0.42804494, 0.65563080, 0.86996982),
        "fixed_n": (15.975, 76.0, 185.05),
        "fixed_recall": (0.6875, 0.95215201, 1.0),
        "fixed_precision": (0.00517191, 0.02959965, 0.06993748),
        "fixed_tmc": (0.69321185, 0.97287532, 1.0),
        "empty": 0,
    },
    "NH4-N": {
        "period_tmc": (0.96871066, 0.99574166, 1.0),
        "fixed_n": (764.975, 1071.0, 1483.025),
        "fixed_recall": (0.88210894, 0.92813442, 0.95931010),
        "fixed_precision": (0.06538292, 0.08435700, 0.10718753),
        "fixed_tmc": (0.89277780, 0.94414540, 0.97390326),
        "empty": 0,
    },
}

EXPECTED_HOURLY = {
    "COD": (7032, 2.38, 8.94, -0.013),
    "TSS": (7241, 2.53, 5.92, -0.161),
    "NH4-N": (7244, 0.50, 1.77, -0.057),
}


class Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes = 0

    def equal(self, name: str, actual, expected) -> None:
        if actual == expected:
            print(f"PASS  {name}: {actual}")
            self.passes += 1
        else:
            msg = f"FAIL  {name}: got {actual!r}, expected {expected!r}"
            print(msg)
            self.failures.append(msg)

    def close(self, name: str, actual: float, expected: float, atol: float) -> None:
        if np.isfinite(actual) and math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=0.0):
            print(f"PASS  {name}: {float(actual):.8g}")
            self.passes += 1
        else:
            msg = f"FAIL  {name}: got {actual!r}, expected {expected!r} +/- {atol}"
            print(msg)
            self.failures.append(msg)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_primary(df, check: Checker) -> None:
    t1 = primary_sample_table(df).set_index("Channel")
    for pol, exp in EXPECTED_SAMPLE.items():
        row = t1.loc[pol]
        check.equal(f"{pol} valid n overall", int(row["Valid n overall"]), exp[0])
        check.equal(f"{pol} valid n 2025", int(row["Valid n 2025"]), exp[1])
        check.equal(f"{pol} valid n 2026", int(row["Valid n 2026"]), exp[2])
        check.close(f"{pol} median C 2025", row["Median C 2025"], exp[3], 0.01)
        check.close(f"{pol} median C 2026", row["Median C 2026"], exp[4], 0.01)
        check.close(f"{pol} median Q 2025", row["Median Q 2025"], exp[5], 0.05)
        check.close(f"{pol} median Q 2026", row["Median Q 2026"], exp[6], 0.05)

    locked = locked_holdout_point_estimates(df).set_index("Channel")
    for pol, exp in EXPECTED_LOCKED.items():
        row = locked.loc[pol]
        check.close(f"{pol} locked threshold", row["2025 threshold"], exp[0], 0.01)
        check.close(f"{pol} 2026 locked alert", row["2026 alert rate"], exp[1], 0.0006)
        check.close(f"{pol} 2026 locked recall", row["2026 recall"], exp[2], 0.001)
        check.close(f"{pol} 2026 locked precision", row["2026 precision"], exp[3], 0.001)
        check.close(f"{pol} 2026 locked target-mass capture", row["2026 target-mass capture"], exp[4], 0.001)

    season = season_matched(df).set_index("Channel")
    for pol, exp in EXPECTED_SEASON.items():
        row = season.loc[pol]
        check.close(f"{pol} season-matched threshold", row["Jan-Jul 2025 threshold"], exp[0], 0.01)
        check.close(f"{pol} season-matched alert", row["2026 alert rate"], exp[1], 0.0006)
        check.close(f"{pol} season-matched recall", row["2026 recall"], exp[2], 0.001)
        check.close(f"{pol} season-matched precision", row["2026 precision"], exp[3], 0.001)

    p99 = fixed_p99_sensitivity(df).set_index("Channel")
    for pol, exp in EXPECTED_P99.items():
        row = p99.loc[pol]
        check.close(f"{pol} fixed-P99 mass boundary", row["2025 P99 interval-mass boundary"], exp[0], 0.000002)
        check.equal(f"{pol} fixed-P99 2026 target n", int(row["2026 target n"]), exp[1])
        check.close(f"{pol} fixed-P99 locked threshold", row["Locked concentration threshold"], exp[2], 0.01)
        check.close(f"{pol} fixed-P99 recall", row["Recall"], exp[3], 0.001)
        check.close(f"{pol} fixed-P99 precision", row["Precision"], exp[4], 0.001)
        check.close(f"{pol} fixed-P99 target-mass capture", row["Target-mass capture"], exp[5], 0.001)
        check.equal(f"{pol} fixed-P99 true positives", int(row["True positives"]), exp[6])



def validate_revision_bootstrap(df, check: Checker) -> None:
    masks = period_masks(df)
    payload = {}
    for pol in EXPECTED_REVISION_BOOTSTRAP:
        y25 = valid_pollutant(df, pol, masks.development_2025)
        y26 = valid_pollutant(df, pol, masks.holdout_2026)
        req = alert_requirement(y25)
        payload[pol] = {
            "y26": y26,
            "threshold": req["threshold"],
            "mass_boundary": p99_mass_boundary(y25),
        }

    res = bootstrap_revision_metrics(payload, B=1000)
    for pol, exp in EXPECTED_REVISION_BOOTSTRAP.items():
        got = res[pol]
        for label, key in [
            ("period target-mass capture", "period_relative_target_mass_capture_median_ci95"),
            ("fixed target n", "fixed_target_n_median_ci95"),
            ("fixed recall", "fixed_recall_median_ci95"),
            ("fixed precision", "fixed_precision_median_ci95"),
            ("fixed target-mass capture", "fixed_target_mass_capture_median_ci95"),
        ]:
            exp_key = {
                "period target-mass capture": "period_tmc",
                "fixed target n": "fixed_n",
                "fixed recall": "fixed_recall",
                "fixed precision": "fixed_precision",
                "fixed target-mass capture": "fixed_tmc",
            }[label]
            for i, qlabel in enumerate(("2.5%", "50%", "97.5%")):
                check.close(
                    f"{pol} revision bootstrap {label} {qlabel}",
                    got[key][i],
                    exp[exp_key][i],
                    5e-6,
                )
        check.equal(
            f"{pol} revision bootstrap empty fixed-target replicates",
            got["fixed_empty_target_replicates"],
            exp["empty"],
        )

def validate_temporal(df, check: Checker) -> None:
    events, _ = event_segmentation_summary(df)
    events = events.set_index("Max zero-flow bridge (min)")
    for bridge, exp in EXPECTED_EVENTS.items():
        row = events.loc[bridge]
        check.equal(f"events bridge={bridge} n", int(row["Event n"]), exp[0])
        check.close(f"events bridge={bridge} COD P90", row["COD P90 mass"], exp[1], 0.01)
        check.close(f"events bridge={bridge} TSS P90", row["TSS P90 mass"], exp[2], 0.01)
        check.close(f"events bridge={bridge} NH4-N P90", row["NH4-N P90 mass"], exp[3], 0.001)

    # B=1 is enough because only deterministic point columns are checked here.
    hourly, _ = hourly_summary(df, B=1)
    hourly = hourly.set_index("Channel")
    for pol, exp in EXPECTED_HOURLY.items():
        row = hourly.loc[pol]
        check.equal(f"{pol} complete hours", int(row["Complete hours"]), exp[0])
        check.close(f"{pol} hourly P95 |error|", row["P95 |error| (%)"], exp[1], 0.01)
        check.close(f"{pol} hourly P99 |error|", row["P99 |error| (%)"], exp[2], 0.01)
        check.close(f"{pol} hourly cumulative bias", row["Cumulative bias (%)"], exp[3], 0.001)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate deterministic manuscript results.")
    parser.add_argument("--input", default="Dataset.xlsx", help="Path to Dataset.xlsx")
    parser.add_argument("--full", action="store_true", help="Also check slower hourly/event point estimates")
    parser.add_argument(
        "--revision-bootstrap",
        action="store_true",
        help="Also reproduce and validate the 1,000-replicate target-mass-capture/fixed-P99 revision bootstrap",
    )
    args = parser.parse_args()

    path = Path(args.input).resolve()
    if not path.exists():
        print(f"ERROR: input workbook not found: {path}", file=sys.stderr)
        return 2

    check = Checker()
    digest = sha256(path)
    check.equal("Dataset SHA-256", digest, EXPECTED_SHA256)

    print(f"Loading {path} ...")
    df = load_dataset(path)
    check.equal("Dataset rows", len(df), EXPECTED_ROWS)
    validate_primary(df, check)
    if args.revision_bootstrap:
        validate_revision_bootstrap(df, check)
    if args.full:
        validate_temporal(df, check)

    print()
    if check.failures:
        print(f"VALIDATION FAILED: {len(check.failures)} failure(s), {check.passes} checks passed.")
        return 1
    print(f"VALIDATION PASSED: {check.passes} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
