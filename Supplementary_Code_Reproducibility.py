#!/usr/bin/env python3
"""Supplementary Code S1: reproducibility analysis for the manuscript
"Temporal Transferability of Concentration Alerts for Mass-Defined Extremes in
High-Frequency Industrial Effluent Monitoring".

Usage:
  python Supplementary_Code_S1_Q1_Reproducibility.py \
      --input Tong_hop_du_lieu_quan_trac_sach_2025_2026.xlsx \
      --output q1_reproducibility_results.json --bootstrap 1000

The sensor output basis has been confirmed as NH4-N (ammonium as N). The source
workbook retains the legacy field names "NH4+ - Giá trị (mg/L)" and
"NH4+ - Trạng thái"; these names are used only to address the workbook columns
exactly. Archived NH4-N values are interpreted as mg N/L, with derived load rates
in kg N/h and interval/event masses in kg N. No NH4+<->NH4-N conversion is applied.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

SHEET = "Data_Tong_hop"
QCOL = "Flow out 1 - Giá trị (m3/h)"
POLS = {
    "COD": ("COD - Giá trị (mg/L)", "COD - Trạng thái"),
    "TSS": ("TSS - Giá trị (mg/L)", "TSS - Trạng thái"),
    "NH4-N": ("NH4+ - Giá trị (mg/L)", "NH4+ - Trạng thái"),
}
ACCEPTED = {"Hoạt động tổt", "Vượt qui chuẩn"}


def prepare(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=SHEET, engine="openpyxl")
    df["time"] = pd.to_datetime(df["Thời gian ghi nhận"])
    df["date"] = df["time"].dt.floor("D")
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.to_period("M").astype(str)
    df[QCOL] = pd.to_numeric(df[QCOL], errors="coerce")
    return df


def valid(df: pd.DataFrame, pol: str, mask=None, qcut: float = 0.0) -> pd.DataFrame:
    ccol, scol = POLS[pol]
    c = pd.to_numeric(df[ccol], errors="coerce")
    m = df[scol].isin(ACCEPTED) & c.gt(0) & df[QCOL].gt(qcut)
    if mask is not None:
        m &= mask
    out = df.loc[m, ["time", "date", "month", QCOL]].copy()
    out["C"] = c[m].to_numpy(float)
    out["Q"] = df.loc[m, QCOL].to_numpy(float)
    out["load"] = out["C"] * out["Q"] / 1000.0
    return out.reset_index(drop=True)


def top_metrics(dat: pd.DataFrame, frac: float = 0.01) -> dict:
    n = len(dat); k = int(math.ceil(frac * n))
    C = dat["C"].to_numpy(); L = dat["load"].to_numpy()
    ic = np.argsort(-C, kind="mergesort")[:k]
    il = np.argsort(-L, kind="mergesort")[:k]
    inter = len(np.intersect1d(ic, il))
    return {"n": n, "k": k, "overlap": inter / k, "jaccard": inter / (2 * k - inter)}


def alert_requirement(dat: pd.DataFrame, load_frac=0.01, recall_target=0.90, step=0.005) -> dict:
    n = len(dat); m = int(math.ceil(load_frac * n))
    C = dat["C"].to_numpy(); L = dat["load"].to_numpy()
    corder = np.argsort(-C, kind="mergesort")
    ltop = set(np.argsort(-L, kind="mergesort")[:m].tolist())
    rank = np.empty(n, dtype=np.int64); rank[corder] = np.arange(1, n + 1)
    ranks = np.sort(rank[list(ltop)])
    exact_k = int(ranks[int(math.ceil(recall_target * m)) - 1])
    exact_frac = exact_k / n
    grid_frac = min(0.50, math.ceil((exact_frac - 1e-12) / step) * step)
    k = int(math.ceil(grid_frac * n)); selected = corder[:k]
    threshold = float(np.min(C[selected])); pred = C >= threshold
    tp = sum(1 for i in ltop if pred[i])
    return {
        "nominal_alert_fraction": grid_frac,
        "threshold": threshold,
        "realized_alert_fraction": float(pred.mean()),
        "recall": tp / m,
        "precision": tp / int(pred.sum()) if pred.sum() else None,
    }


def evaluate_threshold(dat: pd.DataFrame, threshold: float, load_frac=0.01) -> dict:
    n = len(dat); m = int(math.ceil(load_frac * n))
    C = dat["C"].to_numpy(); L = dat["load"].to_numpy()
    ltop = set(np.argsort(-L, kind="mergesort")[:m].tolist())
    pred = C >= threshold; tp = sum(1 for i in ltop if pred[i])
    return {
        "n": n, "alert_rate": float(pred.mean()), "recall": tp / m,
        "precision": tp / int(pred.sum()) if pred.sum() else None,
        "alerts": int(pred.sum()), "true_positives": int(tp),
    }


def jsd_2d(d1: pd.DataFrame, d2: pd.DataFrame, bins=25) -> float:
    a = np.column_stack([np.log1p(d1["C"]), np.log1p(d1["Q"])])
    b = np.column_stack([np.log1p(d2["C"]), np.log1p(d2["Q"])])
    pooled = np.vstack([a, b]); qs = np.linspace(0, 1, bins + 1)
    ex = np.unique(np.quantile(pooled[:, 0], qs)); ey = np.unique(np.quantile(pooled[:, 1], qs))
    h1, _, _ = np.histogram2d(a[:, 0], a[:, 1], bins=[ex, ey]); h2, _, _ = np.histogram2d(b[:, 0], b[:, 1], bins=[ex, ey])
    v1 = h1.ravel() + 0.5; v2 = h2.ravel() + 0.5; v1 /= v1.sum(); v2 /= v2.sum()
    return float(jensenshannon(v1, v2, base=2.0))


def bootstrap_fixed_holdout(y26: pd.DataFrame, threshold: float, B=1000, seed=20260826) -> dict:
    rng = np.random.default_rng(seed)
    groups = [g.reset_index(drop=True) for _, g in y26.groupby("date", sort=True)]
    vals = []
    for _ in range(B):
        sample = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))], ignore_index=True)
        e = evaluate_threshold(sample, threshold)
        vals.append([e["alert_rate"], e["recall"], np.nan if e["precision"] is None else e["precision"]])
    a = np.asarray(vals, float)
    q = lambda j: [float(x) for x in np.nanpercentile(a[:, j], [2.5, 97.5])]
    return {"B": B, "alert_rate_ci95": q(0), "recall_ci95": q(1), "precision_ci95": q(2)}


def hourly_aggregation(dat: pd.DataFrame) -> dict:
    x = dat.set_index("time").sort_index(); rows = []
    for _, g in x.groupby(pd.Grouper(freq="h")):
        if len(g) != 12: continue
        times = g.index.to_series().sort_values()
        if (times.diff().dropna() != pd.Timedelta(minutes=5)).any(): continue
        direct = (g["C"] * g["Q"] / 1000.0 * (5 / 60)).sum()
        approx = (g["C"].mean() * g["Q"].mean() / 1000.0)
        if direct > 0: rows.append((direct, approx))
    a = np.asarray(rows); rel = (a[:, 1] - a[:, 0]) / a[:, 0] * 100
    return {"complete_hours": len(a), "median_abs_error_pct": float(np.median(np.abs(rel))), "p95_abs_error_pct": float(np.percentile(np.abs(rel),95)), "p99_abs_error_pct": float(np.percentile(np.abs(rel),99)), "cumulative_bias_pct": float((a[:,1].sum()-a[:,0].sum())/a[:,0].sum()*100)}


def event_segmentation(df: pd.DataFrame, pol: str, bridges=(0,5,10,15,30,60)) -> list:
    # Hydraulic event segmentation uses all observed Flow out values; missing timestamps terminate events.
    x=df[["time",QCOL]].copy().sort_values("time").reset_index(drop=True)
    ccol,scol=POLS[pol]; c=pd.to_numeric(df[ccol],errors="coerce"); validchem=df[scol].isin(ACCEPTED)&c.gt(0)
    mass=np.where(validchem & df[QCOL].gt(0), c*df[QCOL]/1000*(5/60), np.nan)
    mass_by_time=pd.Series(mass,index=df["time"]).to_dict()
    # strict positive-flow runs and optional observed-zero bridges
    out=[]
    for bridge in bridges:
        max_slots=int(bridge/5); events=[]; current=[]; zero_buffer=[]; prev=None
        for _,r in x.iterrows():
            t=r['time']; q=r[QCOL]
            contiguous=(prev is not None and t-prev==pd.Timedelta(minutes=5))
            if prev is not None and not contiguous:
                if current: events.append(current); current=[]
                zero_buffer=[]
            if q>0:
                if zero_buffer and len(zero_buffer)<=max_slots and current: current.extend(zero_buffer)
                elif zero_buffer and current: events.append(current); current=[]
                zero_buffer=[]; current.append(t)
            else:
                if current: zero_buffer.append(t)
            prev=t
        if current: events.append(current)
        masses=[]
        for ev in events:
            vals=[mass_by_time.get(t,np.nan) for t in ev]; vals=[v for v in vals if not pd.isna(v)]
            masses.append(sum(vals) if vals else 0.0)
        out.append({"bridge_min":bridge,"events":len(events),"p90_mass_native":float(np.percentile(masses,90))})
    return out


def run(df: pd.DataFrame, B: int, secondary: bool=False) -> dict:
    result={"metadata":{"rows":len(df),"first_timestamp":str(df.time.min()),"last_timestamp":str(df.time.max()),"bootstrap_replicates":B,"NH4N_note":"sensor output confirmed as NH4-N; archived values are mg N/L; raw workbook field name retained for reproducibility"},"pollutants":{}}
    m25_jj=(df.time>=pd.Timestamp('2025-01-01'))&(df.time<pd.Timestamp('2025-08-01'))
    m26_jj=(df.time>=pd.Timestamp('2026-01-01'))&(df.time<pd.Timestamp('2026-08-01'))
    for pol in POLS:
        all_=valid(df,pol); y25=valid(df,pol,df.year.eq(2025)); y26=valid(df,pol,df.year.eq(2026))
        req25=alert_requirement(y25); hold=evaluate_threshold(y26,req25['threshold'])
        season_req=alert_requirement(valid(df,pol,m25_jj)); season_eval=evaluate_threshold(valid(df,pol,m26_jj),season_req['threshold'])
        sensitivity=[]
        for period,mask in [('2025',df.year.eq(2025)),('2026',df.year.eq(2026))]:
            d=valid(df,pol,mask)
            for lf in [0.005,0.01,0.02,0.05]:
                tm=top_metrics(d,lf)
                for rt in [0.80,0.90,0.95]:
                    ar=alert_requirement(d,lf,rt)
                    sensitivity.append({"period":period,"load_target":lf,"recall_target":rt,"overlap":tm['overlap'],"jaccard":tm['jaccard'],"nominal_alert_fraction":ar['nominal_alert_fraction']})
        lowflow=[]
        for qcut in [0.0,0.5,1.0]:
            r=alert_requirement(valid(df,pol,df.year.eq(2025),qcut)); e=evaluate_threshold(valid(df,pol,df.year.eq(2026),qcut),r['threshold']); lowflow.append({"qcut":qcut,"threshold":r['threshold'],**e})
        jsd=jsd_2d(y25,y26); jsd_sm=jsd_2d(valid(df,pol,m25_jj),valid(df,pol,m26_jj))
        result['pollutants'][pol]={
            "sample":{"overall":len(all_),"2025":len(y25),"2026":len(y26)},
            "top1":{"overall":top_metrics(all_),"2025":top_metrics(y25),"2026":top_metrics(y26)},
            "locked_2025":{"threshold":req25['threshold'],"development":req25,"holdout_2026":hold,"bootstrap":bootstrap_fixed_holdout(y26,req25['threshold'],B=B,seed={'COD':20260829,'TSS':20260829,'NH4-N':20260834}[pol])},
            "season_matched":{"threshold":season_req['threshold'],"holdout_2026":season_eval},
            "joint_drift":{"js_distance_2025_vs_2026":jsd,"js_distance_janjul":jsd_sm,"spearman_CQ_2025":float(spearmanr(y25.C,y25.Q).statistic),"spearman_CQ_2026":float(spearmanr(y26.C,y26.Q).statistic)},
            "decision_grid":sensitivity,"low_flow_sensitivity":lowflow
        }
        if secondary:
            result['pollutants'][pol]['hourly_aggregation']=hourly_aggregation(all_)
            result['pollutants'][pol]['event_segmentation']=event_segmentation(df,pol)
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',default='q1_reproducibility_results.json'); ap.add_argument('--bootstrap',type=int,default=1000); ap.add_argument('--secondary',action='store_true',help='also run slower hourly/event secondary analyses')
    args=ap.parse_args(); df=prepare(args.input); res=run(df,args.bootstrap,args.secondary); Path(args.output).write_text(json.dumps(res,indent=2,ensure_ascii=False),encoding='utf-8'); print(f"Wrote {args.output}")

if __name__=='__main__': main()
