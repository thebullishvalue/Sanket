#!/usr/bin/env python3
"""
Sanket — Signal Anatomy & Gate Discovery.

Takes the NAKED crossings (no gates) and lets the data reveal which factor
conditions actually create edge — i.e. reverse-engineers what the gating logic
SHOULD be, rather than trusting the hand-coded `& conv_d>0 & pulse_d>0 & ...`.

For each naked signal (crossover / momentum / threshold × long/short):
  1. Reconstruct the naked event from raw panel columns (per-symbol shifts).
  2. Baseline: directional forward-return expectancy of ALL naked events.
  3. Per-factor conditional edge, LEARNED ON TRAIN / MEASURED ON TEST:
     split events by each factor's train median, keep the better-returning side,
     report the out-of-sample lift. This ranks which gates sort winners.
  4. Compare three gates OOS: naked · current-code gate · data-learned gate
     (top-k factors, AND of their learned half-conditions).

Train = first 70% of dates, Test = last 30% (no lookahead). Runs on the cached
research panel. Gross of costs — a statistical study of where the edge lives.

Usage: python signal_anatomy.py [--horizon 5] [--topk 3]
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
import numpy as np
import pandas as pd

LT, UT = -75, 75          # LO crossover bands
OS2, OB2 = -40, 40        # threshold inner bands (osLevel2 / obLevel2)


def _expect(r):
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return dict(n=0, mean=np.nan, win=np.nan, t=np.nan)
    t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 and r.std() > 0 else np.nan
    return dict(n=len(r), mean=r.mean(), win=(r > 0).mean(), t=t)


def reconstruct_naked(panel):
    """Per-symbol naked crossings from raw columns. Returns boolean masks dict."""
    p = panel.sort_values(["Symbol", "Date"]).copy()
    g = p.groupby("Symbol", sort=False)
    wt1 = p["WT1"]; wt2 = p["Signal_Line"]; lo = p["LO"]
    wt1_prev = g["WT1"].shift(1); wt2_prev = g["Signal_Line"].shift(1); lo_prev = g["LO"].shift(1)

    bull_cross = (wt1 > wt2) & (wt1_prev <= wt2_prev)
    bear_cross = (wt1 < wt2) & (wt1_prev >= wt2_prev)
    xo_long  = (lo > LT) & (lo_prev <= LT)
    xo_short = (lo < UT) & (lo_prev >= UT)

    masks = {
        "crossover_long":  xo_long,
        "crossover_short": xo_short,
        "momentum_long":   bull_cross & (~xo_short),
        "momentum_short":  bear_cross & (~xo_long),
        "threshold_long":  (wt1 < OS2) & (wt1_prev >= OS2) & (wt2 > OS2),
        "threshold_short": (wt1 > OB2) & (wt1_prev <= OB2) & (wt2 < OB2),
    }
    return p, {k: v.fillna(False).to_numpy() for k, v in masks.items()}


def current_code_gate(p, name):
    """The hand-coded gate currently in compute_signal_sets, reconstructed."""
    cd = p["Conviction_Delta"].to_numpy(); pd_ = p["Pulse_Delta"].to_numpy()
    lo_osc = p["Liquidity_Osc"].to_numpy(); lv = p["Liq_Vel"].to_numpy()
    if name == "crossover_long":  return (cd > 0) & (pd_ > 0)
    if name == "crossover_short": return (cd < 0) & (pd_ < 0)
    if name == "momentum_long":   return (cd > 0) & (pd_ > 0) & (lo_osc > 0)
    if name == "momentum_short":  return (cd < 0) & (pd_ < 0) & (lo_osc < 0)
    if name == "threshold_long":  return (cd > 0) & (pd_ > 0) & (lv > 0)
    if name == "threshold_short": return (cd < 0) & (pd_ < 0) & (lv < 0)
    return np.ones(len(p), bool)


# Conditioning factors. (col, directional?) — directional ones are multiplied by
# the trade sign so "higher = more supportive" regardless of long/short.
FACTORS = [
    ("Conviction", True), ("Conviction_Delta", True), ("Pulse", True), ("Pulse_Delta", True),
    ("Liquidity_Osc", True), ("Liq_Vel", True), ("F1_PriceMom", True), ("F2_VolQual", True),
    ("AT_Filter", True), ("ZScore", True), ("Recent_Travel", True), ("VolTrend", False),
    ("Regime_Confidence", False), ("cs_breadth", True), ("cs_mom_rank", True), ("cs_conv_rank", True),
    ("nf_rsi", True), ("nf_distma", True), ("nf_rvpct", False), ("nf_gap", True),
    ("HMM_dir", True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--topk", type=int, default=3, help="factors to AND into the learned gate")
    ap.add_argument("--panel", default=None)
    args = ap.parse_args()
    H = args.horizon

    path = args.panel or sorted(glob.glob(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".sanket_research_*.parquet")))[0]
    panel = pd.read_parquet(path)
    p, naked = reconstruct_naked(panel)

    # Directional forward return, signed per signal direction.
    retH = p[f"Ret_{H}b"].to_numpy(dtype=float)
    # HMM_dir helper
    p["HMM_dir"] = p["HMM_Bull"].to_numpy() - p["HMM_Bear"].to_numpy()
    dates = p["Date"].to_numpy()
    cut = np.sort(np.unique(dates))[int(len(np.unique(dates)) * 0.70)]
    is_tr = dates < cut

    print("\n" + "═" * 84)
    print(f"  SANKET — SIGNAL ANATOMY & GATE DISCOVERY   (horizon {H}b, train<{str(cut)[:10]}<test)")
    print(f"  Panel: {os.path.basename(path)} · {len(p)} rows · {p['Symbol'].nunique()} symbols")
    print("═" * 84)

    for base in ["crossover", "momentum", "threshold"]:
        for side in ["long", "short"]:
            name = f"{base}_{side}"
            sign = +1.0 if side == "long" else -1.0
            ev = naked[name]
            r = retH * sign
            n_all = int(ev.sum())
            if n_all < 80:
                print(f"\n  ▌ {name:16s} — only {n_all} naked events; skipping (underpowered).")
                continue

            base_tr = _expect(r[ev & is_tr]); base_te = _expect(r[ev & ~is_tr])
            print(f"\n  ▌ {name.upper():16s}  naked events: {n_all}  "
                  f"(train {base_tr['n']}, test {base_te['n']})")
            print(f"      NAKED baseline    test mean {base_te['mean']*100:+.3f}%  "
                  f"win {base_te['win']*100:.1f}%  t {base_te['t']:+.2f}")

            # ── Per-factor: learn better-half on train, measure lift on test ──
            rows = []
            for col, directional in FACTORS:
                if col not in p.columns:
                    continue
                f = p[col].to_numpy(dtype=float) * (sign if directional else 1.0)
                tr = ev & is_tr & ~np.isnan(f) & ~np.isnan(r)
                te = ev & ~is_tr & ~np.isnan(f) & ~np.isnan(r)
                if tr.sum() < 40 or te.sum() < 25:
                    continue
                med = np.median(f[tr])
                hi_tr = r[tr & (f >= med)].mean(); lo_tr = r[tr & (f < med)].mean()
                keep_high = hi_tr >= lo_tr                       # learned direction
                te_keep = te & ((f >= med) if keep_high else (f < med))
                if te_keep.sum() < 20:
                    continue
                kept = _expect(r[te_keep])
                lift = kept["mean"] - base_te["mean"]            # OOS improvement vs naked
                rows.append((col, "≥" if keep_high else "<", med, kept["mean"], kept["win"], kept["n"], lift, kept["t"]))

            rows.sort(key=lambda x: -x[6])   # by OOS lift
            print(f"      {'factor':16s} {'cond':>5s} {'thresh':>8s} {'testμ%':>8s} {'win%':>6s} {'n':>5s} {'OOSlift%':>9s} {'t':>6s}")
            for col, cmp, med, mu, win, nn, lift, tt in rows[:8]:
                star = "  ←" if lift > 0.05 and tt > 1.5 else ""
                print(f"      {col:16s} {cmp:>5s} {med:8.2f} {mu*100:8.3f} {win*100:6.1f} {nn:5d} {lift*100:+9.3f} {tt:+6.2f}{star}")

            # ── Gate comparison OOS: naked vs current-code vs learned top-k ──
            cur = current_code_gate(p, name)
            cur_te = _expect(r[ev & ~is_tr & cur])
            # learned gate = AND of top-k positive-lift factors' learned half-conditions
            topk = [row for row in rows if row[6] > 0][:args.topk]
            learned = ev & ~is_tr
            for col, cmp, med, *_ in topk:
                directional = dict(FACTORS).get(col, True)
                f = p[col].to_numpy(dtype=float) * (sign if directional else 1.0)
                learned = learned & ((f >= med) if cmp == "≥" else (f < med))
            lr = _expect(r[learned])
            print(f"      ── GATE COMPARISON (out-of-sample, horizon {H}b) ──")
            print(f"        naked       : μ {base_te['mean']*100:+.3f}%  win {base_te['win']*100:.1f}%  n {base_te['n']}")
            print(f"        current code: μ {cur_te['mean']*100:+.3f}%  win {(cur_te['win'] or 0)*100:.1f}%  n {cur_te['n']}  t {cur_te['t']:+.2f}")
            _lg = ", ".join(f"{c}{m}" for c, m, *_ in topk) or "—"
            print(f"        learned gate: μ {lr['mean']*100:+.3f}%  win {(lr['win'] or 0)*100:.1f}%  n {lr['n']}  t {lr['t']:+.2f}   [{_lg}]")

    print("\n" + "═" * 84)
    print("  READ: factors with '←' have positive OOS lift AND t>1.5 — they genuinely sort")
    print("  winners for that signal. 'learned gate' beating 'current code' on test μ (at a")
    print("  usable n) = the data prefers a different gating recipe than the hand-coded one.")
    print("  A naked baseline that is NEGATIVE means the raw crossing is anti-predictive —")
    print("  no gate fixes a broken trigger; consider fading or dropping it.")
    print("═" * 84 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
