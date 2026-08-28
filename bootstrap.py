from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from .config import (
    ALERT_GRID_STEP,
    FIXED_HOLDOUT_SEEDS,
    NESTED_BASE_SEED,
    OVERLAP_DIFF_BASE_SEED,
    POLLUTANT_OFFSETS,
    PRIMARY_BOOTSTRAP_REPLICATES,
    PRIMARY_LOAD_FRACTION,
    PRIMARY_RECALL_TARGET,
    REVISION_METRICS_BOOTSTRAP_SEED,
    SECONDARY_BOOTSTRAP_REPLICATES,
    YEAR_BOOTSTRAP_BASE_SEED,
)
from .core import alert_requirement, evaluate_threshold, top_metrics


def percentile_interval(values, probs=(2.5, 50.0, 97.5)) -> np.ndarray:
    return np.nanpercentile(np.asarray(values, float), probs)


def _day_groups(dat: pd.DataFrame) -> list[pd.DataFrame]:
    return [g.reset_index(drop=True) for _, g in dat.groupby("date", sort=True)]


def _bootstrap_weight_context(dat: pd.DataFrame):
    dates = pd.Categorical(dat["date"], categories=sorted(dat["date"].unique()), ordered=True)
    day_codes = dates.codes.astype(np.int32)
    n_days = len(dates.categories)
    c = dat["C"].to_numpy(float)
    mass = dat["interval_mass"].to_numpy(float)
    corder = np.argsort(-c, kind="mergesort")
    morder = np.argsort(-mass, kind="mergesort")

    # End position of each equal-concentration group in concentration-sorted order.
    cs = c[corder]
    group_end = np.empty(len(cs), dtype=np.int32)
    if len(cs):
        starts = np.r_[0, np.flatnonzero(cs[1:] != cs[:-1]) + 1]
        ends = np.r_[starts[1:] - 1, len(cs) - 1]
        for a, b in zip(starts, ends):
            group_end[a : b + 1] = b
    return day_codes, n_days, c, mass, corder, morder, group_end


def _allocate_top_multiplicity(sorted_idx, row_weights, m):
    sw = row_weights[sorted_idx]
    cum = np.cumsum(sw, dtype=np.int64)
    prev = cum - sw
    take_sorted = np.clip(m - prev, 0, sw).astype(np.int32)
    out = np.zeros_like(row_weights, dtype=np.int32)
    out[sorted_idx] = take_sorted
    return out


def _weights_from_day_sample(rng, day_codes, n_days):
    sampled = rng.integers(0, n_days, size=n_days)
    counts = np.bincount(sampled, minlength=n_days).astype(np.int32)
    return counts[day_codes]


def bootstrap_fixed_holdout(
    y26: pd.DataFrame,
    threshold: float,
    pollutant: str,
    B: int = PRIMARY_BOOTSTRAP_REPLICATES,
) -> dict:
    """Calendar-day bootstrap conditional on the observed locked threshold.

    A weighted-day implementation avoids rebuilding/sorting the full resampled
    table in every replicate while preserving calendar-day resampling.
    """
    rng = np.random.default_rng(FIXED_HOLDOUT_SEEDS[pollutant])
    day_codes, n_days, c, mass, corder, morder, group_end = _bootstrap_weight_context(y26)
    flags = c >= threshold
    vals = np.empty((B, 3), dtype=float)
    for b in range(B):
        w = _weights_from_day_sample(rng, day_codes, n_days)
        n = int(w.sum())
        m = int(math.ceil(PRIMARY_LOAD_FRACTION * n))
        target_mult = _allocate_top_multiplicity(morder, w, m)
        alerts = int(w[flags].sum())
        hit = int(target_mult[flags].sum())
        vals[b] = [alerts / n, hit / m, hit / alerts if alerts else np.nan]

    return {
        "B": int(B),
        "alert_rate_ci95": percentile_interval(vals[:, 0], (2.5, 97.5)).tolist(),
        "recall_ci95": percentile_interval(vals[:, 1], (2.5, 97.5)).tolist(),
        "precision_ci95": percentile_interval(vals[:, 2], (2.5, 97.5)).tolist(),
    }


def _top_overlap_and_burden_arrays(c, mass, frac=PRIMARY_LOAD_FRACTION, recall_target=PRIMARY_RECALL_TARGET, step=ALERT_GRID_STEP):
    c = np.asarray(c, float)
    mass = np.asarray(mass, float)
    n = len(c)
    m = int(math.ceil(frac * n))
    corder = np.argsort(-c, kind="mergesort")
    morder = np.argsort(-mass, kind="mergesort")
    target = np.zeros(n, dtype=bool)
    target[morder[:m]] = True
    overlap = float(np.count_nonzero(target[corder[:m]]) / m)

    rank = np.empty(n, dtype=np.int64)
    rank[corder] = np.arange(1, n + 1)
    target_ranks = np.sort(rank[target])
    required_hit = int(math.ceil(recall_target * m))
    exact_k = int(target_ranks[required_hit - 1])
    exact_frac = exact_k / n
    burden = math.ceil((exact_frac - 1e-12) / step) * step
    burden = min(1.0, max(step, burden))
    return overlap, float(burden)


def bootstrap_year_metrics(
    dat: pd.DataFrame,
    pollutant: str,
    year_label: str,
    B: int = PRIMARY_BOOTSTRAP_REPLICATES,
    return_samples: bool = False,
) -> dict:
    """Day bootstrap for top-1% overlap and 90%-recall nominal alert burden."""
    offset = POLLUTANT_OFFSETS[pollutant]
    year_offset = 0 if str(year_label).startswith("2025") else 100
    rng = np.random.default_rng(YEAR_BOOTSTRAP_BASE_SEED + 10 * offset + year_offset)
    day_codes, n_days, c, mass, corder, morder, group_end = _bootstrap_weight_context(dat)
    overlap = np.empty(B, float)
    burden = np.empty(B, float)

    for b in range(B):
        w = _weights_from_day_sample(rng, day_codes, n_days)
        n = int(w.sum())
        m = int(math.ceil(PRIMARY_LOAD_FRACTION * n))
        target_mult = _allocate_top_multiplicity(morder, w, m)
        conc_mult = _allocate_top_multiplicity(corder, w, m)
        overlap[b] = float(np.minimum(target_mult, conc_mult).sum() / m)

        wc = w[corder]
        tc = target_mult[corder]
        cumw = np.cumsum(wc, dtype=np.int64)
        cumt = np.cumsum(tc, dtype=np.int64)
        f = ALERT_GRID_STEP
        found = np.nan
        while f <= 1.0 + 1e-12:
            k = int(math.ceil(f * n))
            pos = int(np.searchsorted(cumw, k, side="left"))
            pos = min(pos, len(corder) - 1)
            endpos = int(group_end[pos])
            recall = cumt[endpos] / m
            if recall + 1e-12 >= PRIMARY_RECALL_TARGET:
                found = f
                break
            f += ALERT_GRID_STEP
        burden[b] = found

    result = {
        "B": int(B),
        "overlap_ci95": percentile_interval(overlap, (2.5, 97.5)).tolist(),
        "burden_ci95": percentile_interval(burden, (2.5, 97.5)).tolist(),
    }
    if return_samples:
        result["_overlap_samples"] = overlap
        result["_burden_samples"] = burden
    return result


def bootstrap_overlap_difference(
    y25: pd.DataFrame,
    y26: pd.DataFrame,
    pollutant: str,
    B: int = PRIMARY_BOOTSTRAP_REPLICATES,
) -> dict:
    """Independent day-block bootstrap for overlap(2026) - overlap(2025)."""
    rng = np.random.default_rng(OVERLAP_DIFF_BASE_SEED + POLLUTANT_OFFSETS[pollutant])
    g25 = _day_groups(y25)
    g26 = _day_groups(y26)
    diff = np.empty(B, float)
    for b in range(B):
        i25 = rng.integers(0, len(g25), size=len(g25))
        i26 = rng.integers(0, len(g26), size=len(g26))
        s25 = pd.concat([g25[i] for i in i25], ignore_index=True)
        s26 = pd.concat([g26[i] for i in i26], ignore_index=True)
        diff[b] = top_metrics(s26)["overlap"] - top_metrics(s25)["overlap"]
    ci = percentile_interval(diff, (2.5, 97.5))
    return {"B": int(B), "difference_ci95": ci.tolist()}


def bootstrap_revision_metrics(
    payloads: dict[str, dict],
    B: int = PRIMARY_BOOTSTRAP_REPLICATES,
) -> dict[str, dict]:
    """Revision-added bootstrap metrics reported in manuscript v2.

    This routine reproduces two additions made during the final manuscript
    revision using one fixed RNG stream (seed 20260828):

    1. target-mass capture for the period-relative top-1% holdout target; and
    2. later-period uncertainty for the fixed historical P99 target (Table S7b).

    Within each replicate, the *same resampled 2026 calendar days* are used for
    both estimands. The RNG then advances through pollutants in COD, TSS, NH4-N
    order. This ordering is intentional and documented so the published
    percentile intervals are exactly reproducible.

    ``payloads`` must contain, for each pollutant key, ``y26`` (the eligible
    2026 dataframe), ``threshold`` (locked 2025 concentration threshold), and
    ``mass_boundary`` (locked 2025 P99 interval-mass boundary).
    """
    rng = np.random.default_rng(REVISION_METRICS_BOOTSTRAP_SEED)
    out: dict[str, dict] = {}

    for pollutant in ("COD", "TSS", "NH4-N"):
        item = payloads[pollutant]
        dat = item["y26"]
        threshold = float(item["threshold"])
        boundary = float(item["mass_boundary"])

        day_codes, n_days, c, mass, corder, morder, group_end = _bootstrap_weight_context(dat)
        flags = c >= threshold
        fixed_target = mass >= boundary
        both = flags & fixed_target

        rel_tmc = np.empty(B, float)
        fixed_vals = np.empty((B, 5), float)
        empty_target = 0

        for b in range(B):
            # One calendar-day resample supports both the period-relative and
            # fixed-target metrics in this replicate.
            w = _weights_from_day_sample(rng, day_codes, n_days)
            n = int(w.sum())

            # Period-relative top-1% target-mass capture.
            m = int(math.ceil(PRIMARY_LOAD_FRACTION * n))
            target_mult = _allocate_top_multiplicity(morder, w, m)
            rel_target_mass = float(np.dot(target_mult, mass))
            rel_captured_mass = float(np.dot(target_mult[flags], mass[flags]))
            rel_tmc[b] = rel_captured_mass / rel_target_mass if rel_target_mass > 0 else np.nan

            # Fixed historical P99 target on the same resampled days.
            alerts = int(w[flags].sum())
            target_n = int(w[fixed_target].sum())
            tp = int(w[both].sum())
            if target_n == 0:
                empty_target += 1
                recall = np.nan
                # Precision remains defined as 0/alerts when the resample has
                # alerts but no fixed-target interval; recall and mass capture
                # are undefined because their target denominator is zero.
                precision = 0.0 if alerts else np.nan
                capture = np.nan
            else:
                recall = tp / target_n
                precision = tp / alerts if alerts else np.nan
                target_mass = float(np.dot(w[fixed_target], mass[fixed_target]))
                captured_mass = float(np.dot(w[both], mass[both]))
                capture = captured_mass / target_mass if target_mass > 0 else np.nan
            fixed_vals[b] = [target_n, recall, precision, capture, alerts / n]

        out[pollutant] = {
            "B": int(B),
            "period_relative_target_mass_capture_median_ci95": percentile_interval(rel_tmc).tolist(),
            "fixed_target_n_median_ci95": percentile_interval(fixed_vals[:, 0]).tolist(),
            "fixed_recall_median_ci95": percentile_interval(fixed_vals[:, 1]).tolist(),
            "fixed_precision_median_ci95": percentile_interval(fixed_vals[:, 2]).tolist(),
            "fixed_target_mass_capture_median_ci95": percentile_interval(fixed_vals[:, 3]).tolist(),
            "fixed_alert_rate_median_ci95": percentile_interval(fixed_vals[:, 4]).tolist(),
            "fixed_empty_target_replicates": int(empty_target),
        }

    return out


# Fast array implementation used for the nested threshold-estimation bootstrap.
def _derive_threshold_arrays(c, mass, frac=PRIMARY_LOAD_FRACTION, target=PRIMARY_RECALL_TARGET, step=ALERT_GRID_STEP):
    c = np.asarray(c, float)
    mass = np.asarray(mass, float)
    n = len(c)
    m = int(math.ceil(frac * n))
    if n == 0 or m == 0:
        return np.nan

    # Target selection by load rank. The data have sufficient numerical
    # resolution around the target boundary that argpartition is stable for this
    # dataset and substantially reduces the cost of 1,000 nested replicates.
    idx = np.argpartition(mass, n - m)[n - m :]
    h = int(math.ceil(target * m))
    ct = c[idx]
    critical = np.partition(ct, len(ct) - h)[len(ct) - h]
    pos_first = int(np.count_nonzero(c > critical)) + 1
    j = math.floor((pos_first - 1) / (step * n)) + 1
    nominal = j * step
    k = int(math.ceil(nominal * n))
    threshold = np.partition(c, n - k)[n - k]
    return float(threshold)


def _evaluate_arrays(c, mass, threshold, frac=PRIMARY_LOAD_FRACTION):
    c = np.asarray(c, float)
    mass = np.asarray(mass, float)
    n = len(c)
    m = int(math.ceil(frac * n))
    idx = np.argpartition(mass, n - m)[n - m :]
    flags = c >= threshold
    nf = int(np.count_nonzero(flags))
    hit = int(np.count_nonzero(flags[idx]))
    return nf / n, hit / m, hit / nf if nf else np.nan


def _arrays_by_day(dat: pd.DataFrame):
    dd = defaultdict(lambda: [[], []])
    for row in dat[["date", "C", "interval_mass"]].itertuples(index=False):
        dd[row.date][0].append(float(row.C))
        dd[row.date][1].append(float(row.interval_mass))
    return [(d, np.asarray(dd[d][0], float), np.asarray(dd[d][1], float)) for d in sorted(dd)]


def _concat_blocks(blocks, inds):
    return np.concatenate([blocks[i][1] for i in inds]), np.concatenate([blocks[i][2] for i in inds])


def nested_threshold_bootstrap(
    y25: pd.DataFrame,
    y26: pd.DataFrame,
    pollutant: str,
    B: int = PRIMARY_BOOTSTRAP_REPLICATES,
) -> dict:
    """Resample 2025 days, re-derive threshold, and independently resample 2026 days."""
    offset = POLLUTANT_OFFSETS[pollutant]
    d25 = _arrays_by_day(y25)
    d26 = _arrays_by_day(y26)
    rng = np.random.default_rng(NESTED_BASE_SEED + offset)

    metrics = np.empty((B, 3), float)
    thresholds = np.empty(B, float)
    for b in range(B):
        i26 = rng.integers(0, len(d26), size=len(d26))
        c26, m26 = _concat_blocks(d26, i26)
        i25 = rng.integers(0, len(d25), size=len(d25))
        c25, m25 = _concat_blocks(d25, i25)
        thr = _derive_threshold_arrays(c25, m25)
        thresholds[b] = thr
        metrics[b] = _evaluate_arrays(c26, m26, thr)

    q_thr = percentile_interval(thresholds)
    q_alert = percentile_interval(metrics[:, 0])
    q_recall = percentile_interval(metrics[:, 1])
    q_precision = percentile_interval(metrics[:, 2])
    return {
        "B": int(B),
        "threshold_median_ci95": q_thr.tolist(),
        "alert_rate_median_ci95": q_alert.tolist(),
        "recall_median_ci95": q_recall.tolist(),
        "precision_median_ci95": q_precision.tolist(),
    }


def moving_block_holdout(
    y26: pd.DataFrame,
    threshold: float,
    pollutant: str,
    block_lengths=(1, 3, 7),
    B: int = SECONDARY_BOOTSTRAP_REPLICATES,
) -> pd.DataFrame:
    """Circular moving-block sensitivity over ordered evaluation-day blocks."""
    offset = POLLUTANT_OFFSETS[pollutant]
    days = _arrays_by_day(y26)
    n = len(days)
    rows = []
    for block in block_lengths:
        rng = np.random.default_rng(NESTED_BASE_SEED + 100 * block + offset)
        vals = np.empty((B, 3), float)
        nblocks = int(math.ceil(n / block))
        for b in range(B):
            starts = rng.integers(0, n, size=nblocks)
            inds = []
            for s in starts:
                inds.extend(((s + np.arange(block)) % n).tolist())
            inds = inds[:n]
            c, mass = _concat_blocks(days, inds)
            vals[b] = _evaluate_arrays(c, mass, threshold)
        qa = percentile_interval(vals[:, 0])
        qr = percentile_interval(vals[:, 1])
        qp = percentile_interval(vals[:, 2])
        rows.append(
            {
                "Channel": pollutant,
                "Block length (days)": block,
                "Alert median (%)": qa[1] * 100,
                "Alert CI low (%)": qa[0] * 100,
                "Alert CI high (%)": qa[2] * 100,
                "Recall median (%)": qr[1] * 100,
                "Recall CI low (%)": qr[0] * 100,
                "Recall CI high (%)": qr[2] * 100,
                "Precision median (%)": qp[1] * 100,
                "Precision CI low (%)": qp[0] * 100,
                "Precision CI high (%)": qp[2] * 100,
            }
        )
    return pd.DataFrame(rows)
