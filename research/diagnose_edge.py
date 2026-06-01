#!/usr/bin/env python3
"""
Sanket — Edge diagnostic battery.

Layer-2 confidence showed AUC 0.464 (no edge) on F&O. This localizes WHY before
any fix: it measures where linear edge is or isn't, across features, horizons,
sets, and — critically — across LABEL DEFINITIONS. The current label is
"directional raw forward return > deadband", which is contaminated by market beta.
We also test a cross-sectionally de-meaned label (beat-your-peers, beta removed)
and a payoff-asymmetry label.

It harvests the real panel ONCE via the production path (reusing
validate_intelligence's Streamlit stub + run_timeseries_analysis), caches it to
parquet, and reuses the cache on subsequent runs so the battery iterates cheaply.

Usage:
    python diagnose_edge.py --universe "India Indexes" --index "F&O Stocks"
    python diagnose_edge.py --universe "India Indexes" --index "F&O Stocks" --refresh   # force re-harvest

Reads cache from .sanket_panel_<slug>.parquet next to this file.
"""
from __future__ import annotations

# Path shim: this script lives in research/ but imports the live app modules
# (sanket, priority_engine, intelligence, gate_engine) and sibling research
# scripts from the repo ROOT. Put the parent dir on sys.path so those resolve
# whether run as "python research/foo.py" or from inside research/.
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_here))   # repo root: sanket, priority_engine, ...
_sys.path.insert(0, _here)                       # research/: sibling scripts (validate_intelligence)

import argparse
import datetime
import os
import re
import sys

import numpy as np
import pandas as pd

import validate_intelligence as VH   # reuses the headless Streamlit stub + harvest

_FIRED = ["A: Long", "A: Short", "B: Long", "B: Short", "C: Long", "C: Short"]
_HORIZONS = [2, 3, 5, 8, 13]


def _auc(y, score):
    """ROC AUC via Mann-Whitney U (ties = mid-ranks). NaN if one class absent."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(score, dtype=float)
    ok = ~(np.isnan(y) | np.isnan(s))
    y, s = y[ok], s[ok]
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sr = s[order]
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_") or "na"


def _panel_path(universe, index, timeframe, lookback):
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, f".sanket_panel_{_slug(universe)}_{_slug(index)}_{_slug(timeframe)}_{lookback}.parquet")


def get_panel(args):
    path = _panel_path(args.universe, args.index, args.timeframe, args.lookback_days)
    if os.path.exists(path) and not args.refresh:
        print(f"  Using cached panel: {os.path.basename(path)}")
        return pd.read_parquet(path)

    VH._install_streamlit_stub()
    import sanket as S
    index = args.index or {
        "India Indexes": "NIFTY 50", "US Indexes": "DOW JONES",
        "Crypto": "Digital Assets (Top 20)", "Commodities": "Global Commodities",
    }.get(args.universe)
    end = S._today_ist()
    start = end - datetime.timedelta(days=args.lookback_days)
    print(f"  Harvesting {args.universe} · {index} · {args.timeframe} · ~{args.lookback_days}d (real fetch)…")
    S.run_timeseries_analysis(args.universe, index, start, end,
                              20, 10, 21, (80, 40, -80, -40), args.timeframe,
                              wt2_len=20, wt2_type="ALMA")
    ts = sys.modules["streamlit"].session_state.get("ts_results_df")
    if ts is None or getattr(ts, "empty", True):
        return None
    try:
        ts.to_parquet(path)
        print(f"  Cached panel → {os.path.basename(path)}")
    except Exception as e:
        print(f"  (cache write skipped: {e})")
    return ts


def build_labels(df, dir_sign, fired):
    """Return a dict of {label_name: y_array} — each a 0/1 'good signal' definition.

    raw_h        : directional raw fwd return at horizon h > 0
    raw_mean     : mean directional raw fwd return across horizons > small deadband (≈ production)
    demean_h     : directional CROSS-SECTIONAL-demeaned return at h > 0 (per-date peer-relative; beta removed)
    demean_mean  : mean of the de-meaned directional returns > 0
    payoff_mean  : directional mean return in the TOP TERCILE of |move| (did it pay off big?) — magnitude-aware
    """
    d = np.nan_to_num(dir_sign, nan=0.0)
    dates = df["Date"].to_numpy()
    out = {}

    # Raw per-horizon + mean
    raw = {}
    for h in _HORIZONS:
        col = f"Ret_{h}b"
        if col in df.columns:
            r = df[col].to_numpy(dtype=float)
            raw[h] = d * r
            out[f"raw_{h}b"] = (raw[h] > 0).astype(float)
    if raw:
        rm = np.nanmean(np.column_stack(list(raw.values())), axis=1)
        typ = np.nanmedian(np.abs(rm[fired])) if fired.any() else 0.0
        out["raw_mean"] = (rm > 0.10 * typ).astype(float)
        out["_raw_mean_val"] = rm

    # Cross-sectionally de-meaned (subtract that date's universe mean per horizon) → relative skill
    dem = {}
    for h in _HORIZONS:
        col = f"Ret_{h}b"
        if col not in df.columns:
            continue
        r = pd.Series(df[col].to_numpy(dtype=float), index=df.index)
        date_mean = pd.Series(r.values, index=dates).groupby(level=0).transform("mean").values
        dem[h] = d * (r.values - date_mean)
        out[f"demean_{h}b"] = (dem[h] > 0).astype(float)
    if dem:
        dm = np.nanmean(np.column_stack(list(dem.values())), axis=1)
        out["demean_mean"] = (dm > 0).astype(float)
        out["_demean_mean_val"] = dm

    # Payoff-asymmetry: among fired, label 1 if directional mean return is in the top
    # tercile of the |return| distribution AND positive (big favorable move).
    if "_raw_mean_val" in out:
        rm = out["_raw_mean_val"]
        out["payoff_mean"] = ((rm > 0) & (np.abs(rm) >= np.nanquantile(np.abs(rm[fired]) if fired.any() else np.abs(rm), 0.667))).astype(float)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="India Indexes")
    ap.add_argument("--index", default="F&O Stocks")
    ap.add_argument("--timeframe", default="Daily", choices=["Daily", "Weekly"])
    ap.add_argument("--lookback-days", type=int, default=730)
    ap.add_argument("--refresh", action="store_true", help="force re-harvest, ignore cache")
    args = ap.parse_args()

    print("\n" + "═" * 74)
    print("  SANKET — EDGE DIAGNOSTIC  (where is the signal, and under which label?)")
    print("═" * 74)

    ts = get_panel(args)
    if ts is None:
        print("  ✗ Harvest produced no panel.")
        return 1

    import priority_engine as pe
    X, dir_sign, set_letter, fired = pe.signal_conf_features(ts)
    fired = np.asarray(fired)
    feat_names = list(pe.CONF_FEATURES)
    print(f"\n  Panel: {len(ts)} rows · {ts['Date'].nunique()} dates · "
          f"{ts['Symbol'].nunique()} symbols · {int(fired.sum())} fired signals")

    labels = build_labels(ts, dir_sign, fired)

    # ── A. Label base rates (fired only) ──────────────────────────────────────
    print("\n  ── A. Label base rates (fired signals only) ──")
    for name in ["raw_mean", "demean_mean", "payoff_mean"]:
        if name in labels:
            y = labels[name][fired]
            print(f"    {name:14s} positive rate = {np.nanmean(y):.3f}  (n={int((~np.isnan(y)).sum())})")

    # ── B. Per-feature UNIVARIATE signed AUC vs each label (fired only) ───────
    #     A feature with no linear edge sits at ~0.50. The feature is already signed
    #     by direction, so higher = more bullish-for-the-trade; AUC>0.5 ⇒ predictive.
    print("\n  ── B. Univariate feature AUC (fired only) — raw_mean vs demean_mean labels ──")
    print(f"    {'feature':14s}  {'raw_mean':>9s}  {'demean_mean':>11s}")
    for j, fn in enumerate(feat_names):
        xf = X[fired, j]
        a_raw = _auc(labels["raw_mean"][fired], xf) if "raw_mean" in labels else float("nan")
        a_dem = _auc(labels["demean_mean"][fired], xf) if "demean_mean" in labels else float("nan")
        flag = ""
        for a in (a_raw, a_dem):
            if not np.isnan(a) and abs(a - 0.5) >= 0.04:
                flag = "  ←"
        print(f"    {fn:14s}  {a_raw:9.3f}  {a_dem:11.3f}{flag}")

    # ── C. Multivariate logistic AUC per label (out-of-sample by date) ────────
    #     Reuses the production calibrator by swapping the label in. Tells us the
    #     best a linear model can do under each label definition.
    print("\n  ── C. Out-of-sample logistic AUC by label (chronological 70/30) ──")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    dates_sorted = np.sort(ts["Date"].unique())
    cut = dates_sorted[int(len(dates_sorted) * 0.70)]
    is_tr = (ts["Date"].to_numpy() < cut)
    for name in ["raw_mean", "demean_mean", "payoff_mean"]:
        if name not in labels:
            continue
        y = labels[name]
        m_tr = fired & is_tr & ~np.isnan(y)
        m_va = fired & ~is_tr & ~np.isnan(y)
        if m_tr.sum() < 100 or m_va.sum() < 50:
            print(f"    {name:14s}  (too few rows)")
            continue
        ytr = y[m_tr]
        if ytr.sum() < 5 or (len(ytr) - ytr.sum()) < 5:
            print(f"    {name:14s}  (one class only in train)")
            continue
        sc = StandardScaler().fit(X[m_tr])
        clf = LogisticRegression(max_iter=200, C=1.0).fit(sc.transform(X[m_tr]), ytr)
        p = clf.predict_proba(sc.transform(X[m_va]))[:, 1]
        a = _auc(y[m_va], p)
        verdict = ("USEFUL" if a >= 0.55 else "weak" if a >= 0.52 else "no edge")
        print(f"    {name:14s}  val AUC = {a:.3f}   → {verdict}")

    # ── D. Per-horizon + per-set AUC (raw vs demean), best single-feature proxy ─
    print("\n  ── D. Per-horizon directional-return predictability (logistic, demean label) ──")
    for h in _HORIZONS:
        lab = f"demean_{h}b"
        if lab not in labels:
            continue
        y = labels[lab]
        m_tr = fired & is_tr & ~np.isnan(y); m_va = fired & ~is_tr & ~np.isnan(y)
        if m_tr.sum() < 100 or m_va.sum() < 50 or y[m_tr].sum() < 5:
            continue
        sc = StandardScaler().fit(X[m_tr])
        clf = LogisticRegression(max_iter=200).fit(sc.transform(X[m_tr]), y[m_tr])
        p = clf.predict_proba(sc.transform(X[m_va]))[:, 1]
        print(f"    horizon {h:>2}b   val AUC = {_auc(y[m_va], p):.3f}")

    print("\n  ── E. Per-set AUC (demean_mean label, logistic) ──")
    for s in ("A", "B", "C"):
        sm = (set_letter == s)
        y = labels.get("demean_mean")
        if y is None:
            break
        m_tr = fired & sm & is_tr & ~np.isnan(y); m_va = fired & sm & ~is_tr & ~np.isnan(y)
        if m_tr.sum() < 80 or m_va.sum() < 40 or y[m_tr].sum() < 5 or (len(y[m_tr])-y[m_tr].sum()) < 5:
            print(f"    Set {s}: too few rows ({int(m_tr.sum())} tr / {int(m_va.sum())} va)")
            continue
        sc = StandardScaler().fit(X[m_tr])
        clf = LogisticRegression(max_iter=200).fit(sc.transform(X[m_tr]), y[m_tr])
        p = clf.predict_proba(sc.transform(X[m_va]))[:, 1]
        print(f"    Set {s}:  val AUC = {_auc(y[m_va], p):.3f}   (n_tr={int(m_tr.sum())}, base={np.nanmean(y[m_tr]):.2f})")

    print("\n" + "═" * 74)
    print("  INTERPRETATION")
    print("  • If demean_* AUC >> raw_* AUC: the edge is RELATIVE (beat peers); the production")
    print("    label is beta-contaminated → switch the calibration label to cross-sectional de-mean.")
    print("  • If one horizon/set AUC >0.55 while the blend is ~0.5: stop averaging — calibrate that cell.")
    print("  • If EVERYTHING sits ~0.50 across all labels/horizons/sets: no linear edge in these")
    print("    features — the confidence filter cannot work as-built; keep it Off/advisory honestly.")
    print("═" * 74 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
