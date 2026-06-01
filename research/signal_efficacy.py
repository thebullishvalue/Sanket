#!/usr/bin/env python3
"""
Sanket — Signal-Efficacy Event Study (do the triggers themselves have edge?).

Everything else (confidence filter, ranking, Intel) sits on top of the A/B/C
boolean triggers. We never directly tested whether FIRING carries information.
This does: for each signal set × direction, it compares the realised forward
returns against (a) the unconditional baseline (all bars) and (b) a count-matched
random-entry bootstrap, and reports expectancy (win rate, avg win/loss, profit
factor) — not just direction. Broken down by horizon and regime.

Runs on the cached research panel (no re-harvest). Gross of costs — a statistical
first gate; if a set fails THIS, trading costs only make it worse.

Usage:
    python signal_efficacy.py                       # uses the F&O cached panel
    python signal_efficacy.py --panel <parquet>
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
import glob
import os

import numpy as np
import pandas as pd

_HZNS = [2, 3, 5, 8, 13]
_SETS = [("A", "Long"), ("A", "Short"), ("B", "Long"), ("B", "Short"), ("C", "Long"), ("C", "Short")]


def _dir_ret(panel, h, sign):
    """Directional forward return at horizon h: +ret for long, -ret for short."""
    return panel[f"Ret_{h}b"].to_numpy(dtype=float) * sign


def _stats(x):
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return dict(n=0, mean=np.nan, win=np.nan, avg_win=np.nan, avg_loss=np.nan, pf=np.nan, t=np.nan)
    wins = x[x > 0]; losses = x[x < 0]
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    pf = (wins.sum() / -losses.sum()) if losses.sum() < 0 else np.inf
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else np.nan
    return dict(n=len(x), mean=x.mean(), win=(x > 0).mean(), avg_win=avg_win,
                avg_loss=avg_loss, pf=pf, t=t)


def _bootstrap_band(pool, n, sign, reps=2000, seed=0):
    """Distribution of mean directional return from `n` random entries drawn from pool.

    pool is the unconditional directional-return array (already sign-applied).
    Returns (lo2.5, mean, hi97.5)."""
    pool = pool[~np.isnan(pool)]
    if len(pool) == 0 or n == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    for i in range(reps):
        means[i] = pool[rng.integers(0, len(pool), n)].mean()
    return (np.percentile(means, 2.5), means.mean(), np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=None)
    ap.add_argument("--horizon", type=int, default=5, help="primary horizon for expectancy + bootstrap")
    args = ap.parse_args()

    path = args.panel
    if path is None:
        cands = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              ".sanket_research_*.parquet")))
        if not cands:
            print("No cached research panel found. Run research_edge.py first.")
            return 1
        path = cands[0]
    panel = pd.read_parquet(path)
    H = args.horizon

    print("\n" + "═" * 80)
    print("  SANKET — SIGNAL EFFICACY EVENT STUDY")
    print(f"  Panel: {os.path.basename(path)}  ·  {len(panel)} rows · "
          f"{panel['Symbol'].nunique()} symbols · {panel['Date'].nunique()} dates")
    print("═" * 80)

    st = panel["SignalType"].to_numpy()

    # ── 1. Unconditional baseline (all bars), per direction ──────────────────
    print(f"\n  ── Baseline: unconditional forward return over ALL bars (horizon {H}b) ──")
    base_long = _dir_ret(panel, H, +1)            # long baseline = raw fwd ret
    base_short = _dir_ret(panel, H, -1)           # short baseline = -fwd ret
    bl, bs = _stats(base_long), _stats(base_short)
    print(f"    long  baseline mean = {bl['mean']*100:+.3f}%  win {bl['win']*100:.1f}%  (n={bl['n']})")
    print(f"    short baseline mean = {bs['mean']*100:+.3f}%  win {bs['win']*100:.1f}%  (n={bs['n']})")

    # ── 2. Per set/direction: expectancy vs baseline + bootstrap band ────────
    print(f"\n  ── Signal expectancy at horizon {H}b  (directional: +long / inverse-short) ──")
    hdr = f"    {'set':9s} {'n':>5s} {'mean%':>8s} {'win%':>6s} {'avgW%':>7s} {'avgL%':>7s} {'PF':>5s} {'t':>6s} {'vs-rand':>16s}"
    print(hdr)
    verdicts = {}
    for s, side in _SETS:
        sign = +1 if side == "Long" else -1
        key = f"{s}: {side}"
        mask = (st == key)
        if mask.sum() == 0:
            continue
        x = _dir_ret(panel, H, sign)[mask]
        S = _stats(x)
        pool = base_long if side == "Long" else base_short
        lo, mid, hi = _bootstrap_band(pool, S["n"], sign=1, reps=2000)
        # signal mean already directional; compare to random band of same direction/count
        rand_lo, rand_hi = lo, hi
        outside = (S["mean"] > rand_hi) or (S["mean"] < rand_lo)
        beats = S["mean"] > hi
        mark = "✓ beats rand" if beats else ("✗ below rand" if S["mean"] < rand_lo else "~ within band")
        verdicts[key] = (S, (rand_lo, rand_hi), beats)
        pf = S["pf"]; pf_s = "inf" if pf == np.inf else f"{pf:.2f}"
        print(f"    {key:9s} {S['n']:5d} {S['mean']*100:+8.3f} {S['win']*100:6.1f} "
              f"{S['avg_win']*100:7.2f} {S['avg_loss']*100:7.2f} {pf_s:>5s} {S['t']:+6.2f} {mark:>16s}")
        print(f"    {'':9s} {'':5s} random 95% band: [{rand_lo*100:+.3f}%, {rand_hi*100:+.3f}%]")

    # ── 3. Edge decay by horizon (mean directional return per set) ───────────
    print("\n  ── Mean directional return by horizon (does edge live early / decay?) ──")
    print(f"    {'set':9s}" + "".join(f"{str(h)+'b':>9s}" for h in _HZNS))
    for s, side in _SETS:
        sign = +1 if side == "Long" else -1
        key = f"{s}: {side}"; mask = (st == key)
        if mask.sum() == 0:
            continue
        cells = []
        for h in _HZNS:
            x = _dir_ret(panel, h, sign)[mask]; x = x[~np.isnan(x)]
            cells.append(f"{x.mean()*100:+8.3f}" if len(x) else f"{'—':>8s}")
        print(f"    {key:9s}" + "".join(f"{c:>9s}" for c in cells))

    # ── 4. By regime (does any set work only in a regime?) horizon H ─────────
    if "Regime" in panel.columns:
        print(f"\n  ── Mean directional return by HMM regime (horizon {H}b) ──")
        regimes = ["BULL", "WEAK_BULL", "NEUTRAL", "WEAK_BEAR", "BEAR", "TRANSITION"]
        present = [r for r in regimes if (panel["Regime"] == r).any()]
        print(f"    {'set':9s}" + "".join(f"{r[:9]:>11s}" for r in present))
        reg = panel["Regime"].to_numpy()
        for s, side in _SETS:
            sign = +1 if side == "Long" else -1
            key = f"{s}: {side}"; mask = (st == key)
            if mask.sum() == 0:
                continue
            cells = []
            for r in present:
                m = mask & (reg == r)
                x = _dir_ret(panel, H, sign)[m]; x = x[~np.isnan(x)]
                cells.append(f"{x.mean()*100:+.2f}({m.sum()})" if len(x) >= 10 else "·")
            print(f"    {key:9s}" + "".join(f"{c:>11s}" for c in cells))

    # ── 5. Verdict ───────────────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  VERDICT — which triggers beat random entry (mean above the 95% band)?")
    any_edge = False
    for key, (S, band, beats) in verdicts.items():
        if beats and S["n"] >= 50:
            any_edge = True
            print(f"    ✓ {key}: +{S['mean']*100:.3f}% mean, PF {('inf' if S['pf']==np.inf else f'{S['pf']:.2f}')}, "
                  f"win {S['win']*100:.0f}% — statistically above random.")
    if not any_edge:
        print("    ✗ No signal set's mean forward return beats a count-matched random entry at")
        print(f"      horizon {H}b. The triggers do not show standalone statistical edge on this")
        print("      panel — they fire, but the post-fire move is indistinguishable from chance.")
        print("      → The signal LOGIC itself needs rework (this is the foundation), OR this")
        print("        universe/timeframe is too efficient for these crossings.")
    print("═" * 80 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
