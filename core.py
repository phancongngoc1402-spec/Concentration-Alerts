from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

from .config import ALERT_GRID_STEP, PRIMARY_LOAD_FRACTION, PRIMARY_RECALL_TARGET


def stable_top_indices(values: np.ndarray, frac: float) -> np.ndarray:
    values = np.asarray(values, float)
    n = len(values)
    k = int(math.ceil(frac * n))
    if k <= 0 or n == 0:
        return np.asarray([], dtype=int)
    return np.argsort(-values, kind="mergesort")[:k]


def top_metrics(dat: pd.DataFrame, frac: float = PRIMARY_LOAD_FRACTION) -> dict:
    n = len(dat)
    k = int(math.ceil(frac * n))
    if n == 0 or k == 0:
        return {"n": n, "k": k, "overlap": np.nan, "jaccard": np.nan}

    c = dat["C"].to_numpy(float)
    mass = dat["interval_mass"].to_numpy(float)
    ic = stable_top_indices(c, frac)
    il = stable_top_indices(mass, frac)
    inter = int(len(np.intersect1d(ic, il)))
    union = 2 * k - inter
    return {"n": n, "k": k, "overlap": inter / k, "jaccard": inter / union if union else np.nan}


def alert_requirement(
    dat: pd.DataFrame,
    load_frac: float = PRIMARY_LOAD_FRACTION,
    recall_target: float = PRIMARY_RECALL_TARGET,
    step: float = ALERT_GRID_STEP,
) -> dict:
    """Minimum nominal concentration-screening grid fraction reaching target recall.

    Sorting is stable. The concentration threshold is the lowest concentration in
    the nominal top-k screening set and all ties at that threshold are flagged.
    Therefore the realized alert fraction can exceed the nominal grid fraction.
    """
    n = len(dat)
    m = int(math.ceil(load_frac * n))
    if n == 0 or m == 0:
        return {
            "nominal_alert_fraction": np.nan,
            "threshold": np.nan,
            "realized_alert_fraction": np.nan,
            "recall": np.nan,
            "precision": np.nan,
        }

    c = dat["C"].to_numpy(float)
    mass = dat["interval_mass"].to_numpy(float)
    corder = np.argsort(-c, kind="mergesort")
    ltop = set(np.argsort(-mass, kind="mergesort")[:m].tolist())

    rank = np.empty(n, dtype=np.int64)
    rank[corder] = np.arange(1, n + 1)
    target_ranks = np.sort(rank[list(ltop)])
    required_hit = int(math.ceil(recall_target * m))
    exact_k = int(target_ranks[required_hit - 1])
    exact_frac = exact_k / n
    grid_frac = math.ceil((exact_frac - 1e-12) / step) * step
    grid_frac = min(1.0, max(step, grid_frac))

    k = int(math.ceil(grid_frac * n))
    selected = corder[:k]
    threshold = float(np.min(c[selected]))
    pred = c >= threshold
    tp = sum(1 for i in ltop if pred[i])
    flagged = int(pred.sum())
    return {
        "nominal_alert_fraction": float(grid_frac),
        "threshold": threshold,
        "realized_alert_fraction": float(flagged / n),
        "recall": float(tp / m),
        "precision": float(tp / flagged) if flagged else np.nan,
        "target_n": int(m),
        "flag_n": flagged,
        "hit_n": int(tp),
    }


def evaluate_threshold(
    dat: pd.DataFrame,
    threshold: float,
    load_frac: float = PRIMARY_LOAD_FRACTION,
) -> dict:
    n = len(dat)
    m = int(math.ceil(load_frac * n))
    if n == 0 or m == 0:
        return {
            "n": n,
            "alert_rate": np.nan,
            "recall": np.nan,
            "precision": np.nan,
            "target_mass_capture": np.nan,
            "alerts": 0,
            "true_positives": 0,
            "target_n": m,
        }

    c = dat["C"].to_numpy(float)
    mass = dat["interval_mass"].to_numpy(float)
    ltop = set(np.argsort(-mass, kind="mergesort")[:m].tolist())
    pred = c >= threshold
    tp = sum(1 for i in ltop if pred[i])
    alerts = int(pred.sum())
    target_mask = np.zeros(n, dtype=bool)
    target_mask[list(ltop)] = True
    target_mass = float(mass[target_mask].sum())
    captured_mass = float(mass[target_mask & pred].sum())
    return {
        "n": int(n),
        "alert_rate": float(alerts / n),
        "recall": float(tp / m),
        "precision": float(tp / alerts) if alerts else np.nan,
        "target_mass_capture": float(captured_mass / target_mass) if target_mass > 0 else np.nan,
        "alerts": alerts,
        "true_positives": int(tp),
        "target_n": int(m),
    }


def evaluate_fixed_mass_target(dat: pd.DataFrame, threshold: float, mass_boundary: float) -> dict:
    c = dat["C"].to_numpy(float)
    mass = dat["interval_mass"].to_numpy(float)
    pred = c >= threshold
    target = mass >= mass_boundary
    tp = int(np.count_nonzero(pred & target))
    alerts = int(np.count_nonzero(pred))
    target_n = int(np.count_nonzero(target))
    target_mass = float(mass[target].sum())
    captured_mass = float(mass[pred & target].sum())
    return {
        "n": int(len(dat)),
        "alert_rate": float(alerts / len(dat)) if len(dat) else np.nan,
        "recall": float(tp / target_n) if target_n else np.nan,
        "precision": float(tp / alerts) if alerts else np.nan,
        "target_mass_capture": float(captured_mass / target_mass) if target_mass > 0 else np.nan,
        "alerts": alerts,
        "true_positives": tp,
        "target_n": target_n,
    }


def p99_mass_boundary(dat: pd.DataFrame, frac: float = PRIMARY_LOAD_FRACTION) -> float:
    mass = dat["interval_mass"].to_numpy(float)
    idx = stable_top_indices(mass, frac)
    if len(idx) == 0:
        return np.nan
    return float(mass[idx[-1]])


def screening_curve(dat: pd.DataFrame, load_frac: float = PRIMARY_LOAD_FRACTION, max_alert: float = 0.35, step: float = 0.005) -> pd.DataFrame:
    n = len(dat)
    m = int(math.ceil(load_frac * n))
    c = dat["C"].to_numpy(float)
    mass = dat["interval_mass"].to_numpy(float)
    ltop = set(np.argsort(-mass, kind="mergesort")[:m].tolist())
    corder = np.argsort(-c, kind="mergesort")
    rows = []
    f = step
    while f <= max_alert + 1e-12:
        k = int(math.ceil(f * n))
        threshold = float(c[corder[k - 1]])
        pred = c >= threshold
        tp = sum(1 for i in ltop if pred[i])
        rows.append(
            {
                "nominal_alert_fraction": f,
                "realized_alert_fraction": float(pred.mean()),
                "recall": float(tp / m),
                "threshold": threshold,
            }
        )
        f += step
    return pd.DataFrame(rows)


def jsd_2d(d1: pd.DataFrame, d2: pd.DataFrame, bins: int = 25) -> float:
    a = np.column_stack([np.log1p(d1["C"]), np.log1p(d1["Q"])])
    b = np.column_stack([np.log1p(d2["C"]), np.log1p(d2["Q"])])
    pooled = np.vstack([a, b])
    qs = np.linspace(0, 1, bins + 1)
    ex = np.unique(np.quantile(pooled[:, 0], qs))
    ey = np.unique(np.quantile(pooled[:, 1], qs))
    if len(ex) < 2 or len(ey) < 2:
        return np.nan
    h1, _, _ = np.histogram2d(a[:, 0], a[:, 1], bins=[ex, ey])
    h2, _, _ = np.histogram2d(b[:, 0], b[:, 1], bins=[ex, ey])
    v1 = h1.ravel().astype(float) + 0.5
    v2 = h2.ravel().astype(float) + 0.5
    v1 /= v1.sum()
    v2 /= v2.sum()
    return float(jensenshannon(v1, v2, base=2.0))


def spearman_cq(dat: pd.DataFrame) -> float:
    if len(dat) < 2:
        return np.nan
    return float(spearmanr(dat["C"], dat["Q"]).statistic)
