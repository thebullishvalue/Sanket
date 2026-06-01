#!/usr/bin/env python3
"""
Sanket — Edge RESEARCH harness (the hard path: crack F&O or prove it's efficient).

The diagnostic battery showed the production confidence features have NO linear edge
on F&O under any label/horizon/set. This goes deeper, composing three probes the
user chose:

  1. NEW orthogonal features — independent of WHY a signal fired (the production
     features are range-restricted by the firing condition). Per-symbol causal
     features (realized-vol percentile, distance-from-MA, return autocorr, volume
     surprise, gap, days-since-signal, RSI-like, Conviction/Pulse deltas, zone
     depth) + cross-sectional features per date (market breadth, sector-free
     relative strength, cross-sectional return rank).
  2. PATH labels — forward MFE/MAE (max favorable / adverse excursion) over a
     horizon, not just the endpoint sign. Captures "tradeable favorable move".
  3. NON-LINEAR model — HistGradientBoosting vs logistic.

It harvests F&O once (reusing the registry cache from a prior run if warm), builds
everything causally, caches an enriched panel to parquet, and reports OOS AUC for
every {feature-set × label × model} cell. Honest verdict at the end.

Usage:
    python research_edge.py --universe "India Indexes" --index "F&O Stocks"
    python research_edge.py ... --refresh        # force re-harvest
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

import validate_intelligence as VH

_FIRED = ["A: Long", "A: Short", "B: Long", "B: Short", "C: Long", "C: Short"]
_SIG_DIR = {"A: Long": 1, "B: Long": 1, "C: Long": 1, "A: Short": -1, "B: Short": -1, "C: Short": -1}
_HZN = 5   # primary path-label horizon (bars)


# ──────────────────────────────────────────────────────────────────────────────
def _auc(y, score):
    y = np.asarray(y, float); s = np.asarray(score, float)
    ok = ~(np.isnan(y) | np.isnan(s)); y, s = y[ok], s[ok]
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); sr = s[order]
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_") or "na"


def _rolling_rank_pct(s: pd.Series, win: int) -> pd.Series:
    """Causal percentile of the last value within its trailing `win` window [0,1]."""
    return s.rolling(win).apply(lambda a: (a[-1] > a[:-1]).mean() if len(a) > 1 else 0.5, raw=True)


# ──────────────────────────────────────────────────────────────────────────────
# NEW per-symbol causal features + path label. Computed on the fully analyzed frame
# (after run_full_analysis + run_regime_analysis) so it can use WT1/Conviction/etc.
# ALL trailing-only (no lookahead) EXCEPT the explicit forward label columns.
# ──────────────────────────────────────────────────────────────────────────────
def add_symbol_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]; high = df["High"]; low = df["Low"]; vol = df["Volume"]
    ret1 = close.pct_change()

    # Realized-vol percentile (is the stock unusually calm/wild vs its own history?)
    rv20 = ret1.rolling(20).std()
    df["nf_rvpct"] = _rolling_rank_pct(rv20, 60)

    # Distance from 50-bar MA in vol-units (stretched / mean-reverting context)
    ma50 = close.rolling(50).mean()
    df["nf_distma"] = ((close - ma50) / (rv20 * close).replace(0, np.nan)).clip(-5, 5)

    # 1-bar return autocorrelation over 20 bars (trending vs choppy microstructure)
    df["nf_autocorr"] = ret1.rolling(20).apply(
        lambda a: np.corrcoef(a[:-1], a[1:])[0, 1] if np.std(a[:-1]) > 0 and np.std(a[1:]) > 0 else 0.0, raw=True)

    # Volume surprise: today's volume vs 20-bar median (z-ish)
    vmed = vol.rolling(20).median()
    df["nf_volsurp"] = ((vol - vmed) / (vol.rolling(20).std().replace(0, np.nan))).clip(-5, 5)

    # Overnight gap (open vs prior close) in vol units
    df["nf_gap"] = ((df["Open"] - close.shift(1)) / (rv20 * close).replace(0, np.nan)).clip(-5, 5)

    # RSI(14) centered to [-1,1]
    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / dn.replace(0, np.nan)
    df["nf_rsi"] = ((100 - 100 / (1 + rs)) / 50.0 - 1.0).clip(-1, 1)

    # Days since the last fired signal of ANY set (signal crowding / freshness)
    any_fired = (df.get("long_cond", False) | df.get("short_cond", False)
                 | df.get("long_cond_comp", False) | df.get("short_cond_comp", False)
                 | df.get("long_cond_wt", False) | df.get("short_cond_wt", False)).astype(bool)
    idx = np.arange(len(df))
    last = np.where(any_fired.to_numpy(), idx, np.nan)
    last = pd.Series(last).ffill().to_numpy()
    df["nf_dayssince"] = np.clip((idx - np.nan_to_num(last, nan=idx)) / 20.0, 0, 5)

    # Zone depth + travel (already computed by the engine) — momentum-of-momentum
    df["nf_recent_travel"] = df.get("Recent_Travel", 0.0)
    df["nf_zscore"] = df.get("ZScore", 0.0)
    df["nf_voltrend"] = df.get("VolTrend", 0.0)
    df["nf_ma_align"] = df.get("MA_Alignment", 0)

    # ── Forward PATH label inputs (the only forward-looking columns) ──
    # MFE/MAE over the next _HZN bars from the close.
    fwd_high = high.shift(-1).rolling(_HZN).max().shift(-(_HZN - 1))
    fwd_low  = low.shift(-1).rolling(_HZN).min().shift(-(_HZN - 1))
    # simpler robust construction: rolling over the forward window
    fh = pd.Series(np.nan, index=df.index); fl = pd.Series(np.nan, index=df.index)
    c = close.to_numpy(); H = high.to_numpy(); L = low.to_numpy(); m = len(df)
    fhv = np.full(m, np.nan); flv = np.full(m, np.nan)
    for t in range(m - _HZN):
        seg_h = H[t + 1:t + 1 + _HZN]; seg_l = L[t + 1:t + 1 + _HZN]
        if c[t] > 0:
            fhv[t] = (seg_h.max() - c[t]) / c[t]
            flv[t] = (seg_l.min() - c[t]) / c[t]
    df["fwd_mfe"] = fhv   # max favorable excursion (+, fraction) over next HZN
    df["fwd_mae"] = flv   # max adverse excursion (-, fraction)
    return df


# ──────────────────────────────────────────────────────────────────────────────
def harvest(args):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f".sanket_research_{_slug(args.universe)}_{_slug(args.index)}_{_slug(args.timeframe)}_{args.lookback_days}.parquet")
    if os.path.exists(path) and not args.refresh:
        print(f"  Using cached research panel: {os.path.basename(path)}")
        return pd.read_parquet(path)

    VH._install_streamlit_stub()
    import sanket as S
    end = S._today_ist(); start = end - datetime.timedelta(days=args.lookback_days)

    # Resolve stock list + data the same way the app does.
    if args.universe == "India Indexes":
        stock_list, _ = S.get_index_stock_list(args.index)
    elif args.universe == "Crypto":
        stock_list, _ = S.get_crypto_symbols(None)
    elif args.universe == "Commodities":
        stock_list, _ = S.get_commodity_symbols(None)
    elif args.universe == "US Indexes":
        stock_list, _ = S.get_us_index_symbols(args.index)
    else:
        stock_list, _ = S.get_index_stock_list(args.index)
    # Fetch DIRECTLY via fetch_batch_data (not get_universe_data, which hard-caps at
    # _MAX_DAYS_BACK=500d ≈ 1.4y). days_back controls the yfinance window; the function
    # already adds +365 internally for indicator warmup. This lets the research harness
    # pull a true multi-year history without touching production constants.
    print(f"  Fetching {len(stock_list)} symbols · ~{args.lookback_days}d window (direct)…")
    data_dict, _ = S.fetch_batch_data(stock_list, end_date=end, days_back=args.lookback_days)
    if not data_dict:
        return None

    rows = []
    print(f"  Analyzing + feature-building {len(data_dict)} symbols…")
    for i, (tk, df) in enumerate(data_dict.items()):
        try:
            if args.timeframe == "Weekly":
                df = S.resample_to_weekly(df)
            df = S.run_full_analysis(df, 20, 10, 21, 80, 40, -80, -40)
            df = S.run_regime_analysis(df)
            df = S.calculate_divergences(df, timeframe=args.timeframe)
            for h in [2, 3, 5, 8, 13]:
                df[f"Ret_{h}b"] = df["Close"].shift(-h) / df["Close"] - 1
            import numpy as _np
            df["SignalType"] = _np.select(
                [df["long_cond_comp"], df["short_cond_comp"], df["long_cond"], df["short_cond"],
                 df["long_cond_wt"], df["short_cond_wt"], df["Condition"] != "Neutral"],
                ["B: Long", "B: Short", "A: Long", "A: Short", "C: Long", "C: Short", df["Condition"]],
                default="-")
            df = add_symbol_features(df)
            df["WT1_5ago"] = df["WT1"].shift(5)
            df["Wave"] = df["WT1"]
            df.index = pd.to_datetime(df.index)
            mask = (df.index.date >= start) & (df.index.date <= end)
            sub = df.loc[mask].copy()
            sub["Symbol"] = tk
            sub["Date"] = sub.index
            rows.append(sub)
        except Exception as e:
            continue
    if not rows:
        return None
    panel = pd.concat(rows, ignore_index=True)

    # ── Cross-sectional features (per date) ──
    g = panel.groupby("Date")
    panel["cs_breadth"] = g["F1_PriceMom"].transform(lambda x: (x > 0).mean())          # market breadth
    panel["cs_ret_rank"] = g["Ret_2b"].transform(lambda x: x.rank(pct=True))            # NOTE: uses fwd — for label only, dropped from features
    panel["cs_mom_rank"] = g["F1_PriceMom"].transform(lambda x: x.rank(pct=True))       # relative strength (causal)
    panel["cs_conv_rank"] = g["Conviction"].transform(lambda x: x.rank(pct=True))       # relative conviction
    panel = panel.drop(columns=["cs_ret_rank"])  # avoid accidental lookahead leakage
    try:
        panel.to_parquet(path)
        print(f"  Cached research panel → {os.path.basename(path)}")
    except Exception as e:
        print(f"  (cache write skipped: {e})")
    return panel


# ──────────────────────────────────────────────────────────────────────────────
NEW_FEATURES = ["nf_rvpct", "nf_distma", "nf_autocorr", "nf_volsurp", "nf_gap", "nf_rsi",
                "nf_dayssince", "nf_recent_travel", "nf_zscore", "nf_voltrend", "nf_ma_align",
                "cs_breadth", "cs_mom_rank", "cs_conv_rank"]


def directionalize(panel, cols):
    """Sign direction-relevant features by the trade direction (like the prod features)."""
    d = panel["SignalType"].map(_SIG_DIR).to_numpy(dtype=float)
    dd = np.nan_to_num(d, nan=0.0)
    M = []
    for c in cols:
        v = panel[c].to_numpy(dtype=float)
        # directional features get multiplied by dir; absolute-context ones stay raw
        if c in ("nf_distma", "nf_rsi", "nf_recent_travel", "nf_zscore", "nf_voltrend",
                 "nf_ma_align", "cs_mom_rank", "cs_conv_rank", "nf_gap", "nf_autocorr"):
            v = dd * v
        M.append(v)
    return np.column_stack(M), d


def build_labels(panel, d):
    dd = np.nan_to_num(d, nan=0.0)
    out = {}
    rets = np.column_stack([panel[f"Ret_{h}b"].to_numpy(float) for h in [2, 3, 5, 8, 13]])
    rm = np.nanmean(rets * dd[:, None], axis=1)
    out["endpoint"] = (rm > 0).astype(float)
    # de-mean per date
    dr = pd.Series(rm, index=panel["Date"].to_numpy())
    out["endpoint_dm"] = (rm - dr.groupby(level=0).transform("mean").to_numpy() > 0).astype(float)
    # PATH: directional MFE vs MAE — a "good" trade reaches favorable excursion
    # bigger than the adverse one it had to sit through (reward >= risk).
    mfe = panel["fwd_mfe"].to_numpy(float); mae = panel["fwd_mae"].to_numpy(float)
    dir_fav = np.where(dd > 0, mfe, -mae)     # favorable excursion in trade direction (+)
    dir_adv = np.where(dd > 0, -mae, mfe)     # adverse excursion in trade direction (+)
    out["path_rr"] = ((dir_fav > 0) & (dir_fav >= dir_adv)).astype(float)   # reward >= risk
    out["path_strong"] = (dir_fav >= 2.0 * dir_adv).astype(float)          # 2:1 favorable
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="India Indexes")
    ap.add_argument("--index", default="F&O Stocks")
    ap.add_argument("--timeframe", default="Daily", choices=["Daily", "Weekly"])
    ap.add_argument("--lookback-days", type=int, default=730)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    print("\n" + "═" * 76)
    print("  SANKET — EDGE RESEARCH  (new features × path labels × non-linear model)")
    print("═" * 76)
    panel = harvest(args)
    if panel is None:
        print("  ✗ No panel."); return 1

    import priority_engine as pe
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier

    Xold, d, set_letter, fired = pe.signal_conf_features(panel)
    fired = np.asarray(fired)
    Xnew, _ = directionalize(panel, NEW_FEATURES)
    Xall = np.column_stack([Xold, Xnew])
    labels = build_labels(panel, d)

    print(f"\n  Panel: {len(panel)} rows · {panel['Date'].nunique()} dates · "
          f"{panel['Symbol'].nunique()} symbols · {int(fired.sum())} fired signals")
    for nm in labels:
        print(f"    label {nm:12s} base rate (fired) = {np.nanmean(labels[nm][fired]):.3f}")

    dates_sorted = np.sort(panel["Date"].unique())
    cut = dates_sorted[int(len(dates_sorted) * 0.70)]
    is_tr = panel["Date"].to_numpy() < cut

    def run_cell(Xmat, label_name, model):
        y = labels[label_name]
        mtr = fired & is_tr & ~np.isnan(y) & ~np.isnan(Xmat).any(axis=1)
        mva = fired & ~is_tr & ~np.isnan(y) & ~np.isnan(Xmat).any(axis=1)
        if mtr.sum() < 150 or mva.sum() < 60 or y[mtr].sum() < 10 or (len(y[mtr]) - y[mtr].sum()) < 10:
            return float("nan")
        if model == "logit":
            sc = StandardScaler().fit(Xmat[mtr])
            clf = LogisticRegression(max_iter=300, C=1.0).fit(sc.transform(Xmat[mtr]), y[mtr])
            p = clf.predict_proba(sc.transform(Xmat[mva]))[:, 1]
        else:
            clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                 max_iter=300, l2_regularization=1.0,
                                                 min_samples_leaf=50, random_state=0)
            clf.fit(Xmat[mtr], y[mtr])
            p = clf.predict_proba(Xmat[mva])[:, 1]
        return _auc(y[mva], p)

    feature_sets = {"old(prod)": Xold, "new": Xnew, "old+new": Xall}
    print("\n  ── OOS val AUC by {feature-set × label × model} (chronological 70/30) ──")
    print(f"    {'features':10s} {'label':12s} {'logit':>7s} {'gbm':>7s}")
    best = ("", "", "", 0.0)
    for fname, Xm in feature_sets.items():
        for lname in ["endpoint", "endpoint_dm", "path_rr", "path_strong"]:
            a_l = run_cell(Xm, lname, "logit")
            a_g = run_cell(Xm, lname, "gbm")
            for tag, a in (("logit", a_l), ("gbm", a_g)):
                if not np.isnan(a) and a > best[3]:
                    best = (fname, lname, tag, a)
            print(f"    {fname:10s} {lname:12s} {a_l:7.3f} {a_g:7.3f}")

    # Univariate AUC of NEW features (path_rr label) to see which, if any, carry signal
    print("\n  ── Univariate AUC of NEW features (path_rr label, fired) ──")
    y = labels["path_rr"]
    for j, fn in enumerate(NEW_FEATURES):
        xf = Xnew[fired, j]
        a = _auc(y[fired], xf)
        flag = "  ←" if not np.isnan(a) and abs(a - 0.5) >= 0.04 else ""
        print(f"    {fn:16s} AUC = {a:.3f}{flag}")

    print("\n" + "═" * 76)
    print(f"  BEST CELL: {best[0]} × {best[1]} × {best[2]}  →  AUC {best[3]:.3f}")
    if best[3] >= 0.55:
        print("  ✓ EDGE FOUND — this feature-set/label/model crosses the usefulness bar on F&O.")
        print("    Next: port these features + label into intelligence.calibrate_signal_confidence.")
    elif best[3] >= 0.52:
        print("  ~ MARGINAL — a whisper of edge; not enough to filter on. Needs more/better features.")
    else:
        print("  ✗ STILL NO EDGE on F&O — even with new features, path labels, and a non-linear model.")
        print("    Honest conclusion: per-signal quality is not predictable from this information set")
        print("    on F&O daily. The confidence layer should stay Off/advisory here; pursue a")
        print("    less-efficient universe (Crypto) or accept ranking-only intelligence.")
    print("═" * 76 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
