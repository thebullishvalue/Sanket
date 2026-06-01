#!/usr/bin/env python3
"""
Sanket — Walk-Forward Gate Validator (the make-or-break test).

The anatomy study learned, on ONE 70/30 split, gates that beat the hand-coded ones.
That can still be one-window luck. This re-learns each signal's gate on ROLLING
train windows and tests on the immediately-following window, across the whole 5y
history. It answers two questions a single split cannot:

  1. EDGE STABILITY — does the learned gate make money out-of-sample in EACH
     forward window, or only on average? (Consistency = real; one big window = luck.)
  2. RECIPE STABILITY — does the gate keep selecting the SAME factors (and the same
     direction) across windows? A gate that picks cs_breadth< every period is a rule;
     one that picks a different factor each period is curve-fit on noise.

Method per window: learn top-k factor half-conditions on the train slice (median
split, keep better-returning side), apply that exact gate to the next test slice,
record the gated mean return, win rate, n, and the chosen factors. Compare against
the naked baseline and the current hand-coded gate IN THE SAME test window.

Gross of costs. Runs on the cached 5y research panel.

Usage: python walk_forward.py [--horizon 5] [--topk 3] [--train-win 252] [--test-win 63]
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
import argparse, glob, os
from collections import Counter
import numpy as np
import pandas as pd

LT, UT = -75, 75
OS2, OB2 = -40, 40

FACTORS = [
    ("Conviction", True), ("Conviction_Delta", True), ("Pulse", True), ("Pulse_Delta", True),
    ("Liquidity_Osc", True), ("Liq_Vel", True), ("F1_PriceMom", True), ("F2_VolQual", True),
    ("AT_Filter", True), ("ZScore", True), ("Recent_Travel", True), ("VolTrend", False),
    ("Regime_Confidence", False), ("cs_breadth", True), ("cs_mom_rank", True), ("cs_conv_rank", True),
    ("nf_rsi", True), ("nf_distma", True), ("nf_rvpct", False), ("nf_gap", True), ("HMM_dir", True),
]
_FDIR = dict(FACTORS)


def _exp(r):
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return (0, np.nan, np.nan)
    return (len(r), r.mean(), (r > 0).mean())


def reconstruct(panel):
    p = panel.sort_values(["Symbol", "Date"]).copy()
    g = p.groupby("Symbol", sort=False)
    wt1, wt2, lo = p["WT1"], p["Signal_Line"], p["LO"]
    w1p, w2p, lop = g["WT1"].shift(1), g["Signal_Line"].shift(1), g["LO"].shift(1)
    bull = (wt1 > wt2) & (w1p <= w2p); bear = (wt1 < wt2) & (w1p >= w2p)
    xol = (lo > LT) & (lop <= LT); xos = (lo < UT) & (lop >= UT)
    masks = {
        "crossover_long": xol, "crossover_short": xos,
        "momentum_long": bull & (~xos), "momentum_short": bear & (~xol),
        "threshold_long": (wt1 < OS2) & (w1p >= OS2) & (wt2 > OS2),
        "threshold_short": (wt1 > OB2) & (w1p <= OB2) & (wt2 < OB2),
    }
    p["HMM_dir"] = p["HMM_Bull"] - p["HMM_Bear"]
    return p, {k: v.fillna(False).to_numpy() for k, v in masks.items()}


def current_gate(p, name):
    cd = p["Conviction_Delta"].to_numpy(); pd_ = p["Pulse_Delta"].to_numpy()
    lo = p["Liquidity_Osc"].to_numpy(); lv = p["Liq_Vel"].to_numpy()
    return {
        "crossover_long": (cd > 0) & (pd_ > 0), "crossover_short": (cd < 0) & (pd_ < 0),
        "momentum_long": (cd > 0) & (pd_ > 0) & (lo > 0), "momentum_short": (cd < 0) & (pd_ < 0) & (lo < 0),
        "threshold_long": (cd > 0) & (pd_ > 0) & (lv > 0), "threshold_short": (cd < 0) & (pd_ < 0) & (lv < 0),
    }[name]


def learn_gate(p, ev_tr, r, sign, topk):
    """On the train slice, rank factors by in-sample gated lift; return the top-k
    (col, cmp, threshold) half-conditions. Mirrors signal_anatomy's learner."""
    base = r[ev_tr]; base = base[~np.isnan(base)]
    if len(base) < 60:
        return []
    base_mu = base.mean()
    scored = []
    for col, directional in FACTORS:
        if col not in p.columns:
            continue
        f = p[col].to_numpy(dtype=float) * (sign if directional else 1.0)
        m = ev_tr & ~np.isnan(f) & ~np.isnan(r)
        if m.sum() < 60:
            continue
        med = np.median(f[m])
        hi = r[m & (f >= med)]; lo = r[m & (f < med)]
        hi = hi[~np.isnan(hi)]; lo = lo[~np.isnan(lo)]
        if len(hi) < 20 or len(lo) < 20:
            continue
        keep_high = hi.mean() >= lo.mean()
        kept = hi if keep_high else lo
        lift = kept.mean() - base_mu
        scored.append((lift, col, "≥" if keep_high else "<", med))
    scored.sort(key=lambda x: -x[0])
    return [(c, cmp, med) for lift, c, cmp, med in scored if lift > 0][:topk]


def apply_gate(p, ev, sign, gate):
    m = ev.copy()
    for col, cmp, med in gate:
        f = p[col].to_numpy(dtype=float) * (sign if _FDIR.get(col, True) else 1.0)
        m = m & ((f >= med) if cmp == "≥" else (f < med))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--train-win", type=int, default=252, help="train window length in trading days")
    ap.add_argument("--test-win", type=int, default=63, help="forward test window length in trading days")
    ap.add_argument("--panel", default=None)
    args = ap.parse_args()
    H = args.horizon

    path = args.panel or sorted(glob.glob(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".sanket_research_*1825*.parquet")) or
        glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".sanket_research_*.parquet")))[-1]
    panel = pd.read_parquet(path)
    p, naked = reconstruct(panel)
    retH = p[f"Ret_{H}b"].to_numpy(dtype=float)
    dates = pd.to_datetime(p["Date"].to_numpy())
    udates = np.sort(np.unique(dates))

    # Build rolling (train, test) date-window pairs
    windows = []
    i = args.train_win
    while i + args.test_win <= len(udates):
        tr_lo, tr_hi = udates[i - args.train_win], udates[i - 1]
        te_lo, te_hi = udates[i], udates[min(i + args.test_win - 1, len(udates) - 1)]
        windows.append((tr_lo, tr_hi, te_lo, te_hi))
        i += args.test_win

    print("\n" + "═" * 86)
    print(f"  SANKET — WALK-FORWARD GATE VALIDATION   (horizon {H}b, learn {args.train_win}d → test {args.test_win}d)")
    print(f"  Panel: {os.path.basename(path)} · {p['Symbol'].nunique()} symbols · {len(udates)} dates · {len(windows)} forward windows")
    print("═" * 86)

    for base in ["crossover", "momentum", "threshold"]:
        for side in ["long", "short"]:
            name = f"{base}_{side}"; sign = +1.0 if side == "long" else -1.0
            ev = naked[name]; r = retH * sign
            cur = current_gate(p, name)

            wins_nk, wins_cur, wins_lg = [], [], []   # per-window (n, mean, win)
            factor_counter = Counter()
            n_pos_windows = 0; n_total = 0
            for tr_lo, tr_hi, te_lo, te_hi in windows:
                tr = (dates >= tr_lo) & (dates <= tr_hi)
                te = (dates >= te_lo) & (dates <= te_hi)
                ev_tr = ev & tr
                if ev_tr.sum() < 60:
                    continue
                gate = learn_gate(p, ev_tr, r, sign, args.topk)
                if not gate:
                    continue
                for col, cmp, _ in gate:
                    factor_counter[f"{col}{cmp}"] += 1
                ev_te = ev & te
                lg_te = apply_gate(p, ev_te, sign, gate)
                nk = _exp(r[ev_te]); cu = _exp(r[ev_te & cur]); lg = _exp(r[lg_te])
                if lg[0] >= 20:
                    wins_nk.append(nk); wins_cur.append(cu); wins_lg.append(lg)
                    n_total += 1
                    if not np.isnan(lg[1]) and lg[1] > 0:
                        n_pos_windows += 1

            if n_total == 0:
                print(f"\n  ▌ {name:16s} — too few events per window; skipping.")
                continue

            def agg(ws):
                ns = np.array([w[0] for w in ws], float)
                mus = np.array([w[1] for w in ws], float)
                wn = np.array([w[2] for w in ws], float)
                # n-weighted mean return across windows
                ok = ns > 0
                wmu = np.nansum(mus[ok] * ns[ok]) / np.nansum(ns[ok]) if ok.any() else np.nan
                return int(np.nansum(ns)), wmu, np.nanmean(wn)

            nk_n, nk_mu, nk_w = agg(wins_nk)
            cu_n, cu_mu, cu_w = agg(wins_cur)
            lg_n, lg_mu, lg_w = agg(wins_lg)
            hit = n_pos_windows / n_total
            top_factors = ", ".join(f"{k}×{v}" for k, v in factor_counter.most_common(4))

            print(f"\n  ▌ {name.upper():16s}   {n_total} forward windows")
            print(f"      naked        : μ {nk_mu*100:+.3f}%  win {nk_w*100:.1f}%  (n {nk_n})")
            print(f"      current code : μ {cu_mu*100:+.3f}%  win {(cu_w or 0)*100:.1f}%  (n {cu_n})")
            print(f"      learned gate : μ {lg_mu*100:+.3f}%  win {(lg_w or 0)*100:.1f}%  (n {lg_n})  "
                  f"·  POSITIVE in {n_pos_windows}/{n_total} windows ({hit*100:.0f}%)")
            print(f"      recipe stability (factor×windows-chosen): {top_factors}")

    print("\n" + "═" * 86)
    print("  HOW TO READ:")
    print("  • learned-gate μ > current-code μ AND positive in ≳65% of forward windows = a")
    print("    STABLE, shippable gate (consistent OOS edge, not one lucky window).")
    print("  • A recipe whose top factor is chosen in MOST windows = a real rule. If every")
    print("    window picks different factors, it's curve-fit — do NOT ship it.")
    print("  • Still gross of costs: a stable +0.3–0.5%/Hbar edge must clear fees+slippage.")
    print("═" * 86 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
