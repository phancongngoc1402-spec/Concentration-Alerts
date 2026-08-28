from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import POLLUTANTS, QCOL
from .core import screening_curve
from .data import calendar_coverage, monthly_status_fractions, period_masks, valid_pollutant


def _save(fig, path: Path, dpi: int = 300):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def figure1_recall_alert_curves(df: pd.DataFrame, path: Path):
    masks = period_masks(df)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), sharey=True)
    for ax, pol in zip(axes, POLLUTANTS):
        y25 = valid_pollutant(df, pol, masks.development_2025)
        y26 = valid_pollutant(df, pol, masks.holdout_2026)
        c25 = screening_curve(y25, max_alert=0.35)
        c26 = screening_curve(y26, max_alert=0.35)
        ax.plot(c25["realized_alert_fraction"] * 100, c25["recall"] * 100, label="2025")
        ax.plot(c26["realized_alert_fraction"] * 100, c26["recall"] * 100, linestyle="--", label="2026 Jan-Jul")
        ax.axhline(90, linewidth=1, linestyle=":")
        ax.set_title(pol)
        ax.set_xlabel("Concentration alert fraction (%)")
        ax.set_xlim(0, 35)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Recall of period-relative top-1% load intervals (%)")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle("Recall-alert-burden curves expose temporal instability of concentration-only screening")
    fig.tight_layout()
    _save(fig, path)


def figure2_monthly_transfer(monthly: pd.DataFrame, path: Path):
    months = [f"2026-{m:02d}" for m in range(1, 8)]
    pols = list(POLLUTANTS)
    recall = np.full((len(pols), len(months)), np.nan)
    alert = np.full_like(recall, np.nan)
    for i, pol in enumerate(pols):
        sub = monthly.loc[monthly["Channel"].eq(pol)].set_index("Evaluation month")
        for j, month in enumerate(months):
            if month in sub.index:
                recall[i, j] = sub.loc[month, "Recall"] * 100
                alert[i, j] = sub.loc[month, "Alert"] * 100

    fig, axes = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
    for ax, arr, title, cbar_label in [
        (axes[0], recall, "Recall of each month's top-1% load intervals (%)", "Recall (%)"),
        (axes[1], alert, "Fraction of valid positive-flow observations flagged (%)", "Alert rate (%)"),
    ]:
        im = ax.imshow(arr, aspect="auto")
        ax.set_yticks(range(len(pols)), labels=pols)
        ax.set_title(title)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isfinite(arr[i, j]):
                    ax.text(j, i, f"{arr[i,j]:.1f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.025, pad=0.02)
    axes[1].set_xticks(range(len(months)), labels=[m[-2:] for m in months])
    axes[1].set_xlabel("Month of 2026")
    fig.suptitle("Month-level transfer of locked 2025 thresholds to January-July 2026")
    fig.tight_layout()
    _save(fig, path)


def figure3_joint_regime_shift(df: pd.DataFrame, path: Path):
    masks = period_masks(df)
    fig, axes = plt.subplots(3, 2, figsize=(9.2, 8.2))
    for r, pol in enumerate(POLLUTANTS):
        y25 = valid_pollutant(df, pol, masks.development_2025)
        y26 = valid_pollutant(df, pol, masks.holdout_2026)
        pooled_load = pd.concat([y25["load_rate"], y26["load_rate"]], ignore_index=True).to_numpy(float)
        p95, p99 = np.percentile(pooled_load, [95, 99])
        for c, (dat, label) in enumerate([(y25, "2025 development"), (y26, "Jan-Jul 2026 holdout")]):
            ax = axes[r, c]
            ax.hexbin(dat["C"], dat["Q"], gridsize=45, mincnt=1, xscale="log", yscale="log")
            xmin = max(dat["C"].min(), 1e-6)
            xmax = dat["C"].max()
            xs = np.logspace(np.log10(xmin), np.log10(xmax), 300)
            ax.plot(xs, p95 * 1000.0 / xs, linestyle="--", linewidth=1, label="pooled load P95")
            ax.plot(xs, p99 * 1000.0 / xs, linestyle=":", linewidth=1, label="pooled load P99")
            ax.set_title(f"{pol}: {label}")
            ax.set_xlabel("Concentration")
            ax.set_ylabel("Effluent flow (m3/h)")
            ax.grid(alpha=0.15)
    axes[-1, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Joint concentration-flow regimes shifted between development and holdout periods")
    fig.tight_layout()
    _save(fig, path)


def figure_s1_coverage_status(df: pd.DataFrame, path: Path):
    cov = calendar_coverage(df)["monthly"]
    stat = monthly_status_fractions(df)
    months = cov["month"].tolist()
    x = np.arange(len(months))

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True)
    axes[0].plot(x, cov["coverage"] * 100, marker="o", markersize=3)
    axes[0].set_ylabel("Calendar coverage (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("Monthly five-minute calendar coverage")
    axes[0].grid(alpha=0.25)

    for pol in ["COD", "TSS", "NH4-N"]:
        sub = stat.loc[stat["parameter"].eq(pol)].set_index("month")
        y = [sub.loc[m, "flagged_fraction"] * 100 if m in sub.index else np.nan for m in months]
        axes[1].plot(x, y, marker="o", markersize=2.5, label=pol)
    axes[1].set_ylabel("Flagged records (%)")
    axes[1].set_title("Station-configured exceedance fractions")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    axes[1].set_xticks(x, labels=months, rotation=45, ha="right")
    fig.tight_layout()
    _save(fig, path)


def figure_s2_hourly_error(hourly_table: pd.DataFrame, path: Path):
    pols = hourly_table["Channel"].tolist()
    x = np.arange(len(pols))
    y = hourly_table["P99 |error| (%)"].to_numpy(float)
    low = hourly_table["P99 CI low"].to_numpy(float)
    high = hourly_table["P99 CI high"].to_numpy(float)
    err = np.vstack([y - low, high - y])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(x, y)
    ax.errorbar(x, y, yerr=err, fmt="none", capsize=4)
    ax.set_xticks(x, labels=pols)
    ax.set_ylabel("P99 absolute hourly error (%)")
    ax.set_title("Hourly aggregation error (95% day-block bootstrap CI)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, path)


def figure_s3_event_mass(event_table: pd.DataFrame, path: Path):
    row0 = event_table.loc[event_table["Max zero-flow bridge (min)"].eq(0)].iloc[0]
    row30 = event_table.loc[event_table["Max zero-flow bridge (min)"].eq(30)].iloc[0]
    pols = list(POLLUTANTS)
    x = np.arange(len(pols))
    width = 0.36
    y0 = [row0[f"{p} P90 mass"] for p in pols]
    y30 = [row30[f"{p} P90 mass"] for p in pols]

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.bar(x - width / 2, y0, width=width, label="0-min bridge")
    ax.bar(x + width / 2, y30, width=width, label="30-min bridge")
    ax.set_xticks(x, labels=pols)
    ax.set_ylabel("P90 event mass (reported mass units)")
    ax.set_title("Event mass depends on analytical segmentation")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, path)
