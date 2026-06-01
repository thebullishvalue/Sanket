"""
Sanket — Self-Learning Gate Engine.

The research established (and out-of-sample validated on a disjoint universe) that
the RAW A/B/C triggers carry edge only under specific CROSS-SECTIONAL conditions
(breadth, peer-rank), that the edge is long-only, and that the post-hoc confidence
filter (Layer 2) does NOT work (AUC ~0.5). This module turns that research process
into an automated, self-validating part of the daily calibration:

    harvest panel  →  per signal, learn the cross-sectional gate (which factors,
    which direction)  →  WALK-FORWARD validate it  →  ACTIVATE only the gates that
    pass (positive in ≥ activate_hit of forward windows AND beat naked)  →  persist.

Key discipline (why this is self-LEARNING, not self-FOOLING):
  • Gates are activated only on out-of-sample walk-forward evidence — a gate that
    doesn't generalize (e.g. the dead shorts) never activates. Same probation
    philosophy as the gated F7 factor.
  • NO hardcoded thresholds. A gate stores only {factor, direction}. At apply time
    the condition is evaluated as a SELF-CALIBRATING cross-sectional rank within the
    current run ("is this stock in the supportive half of today's universe"), so it
    is universe-relative and never fit to a training period's absolute level.

This replaces the dead Layer-2 confidence model as the meaning of "signal
intelligence": the system learns, per universe, which conditions make each raw
signal profitable, validated, and screens through them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Signal sets, their naked-trigger boolean column in results/harvest, and direction.
SIGNAL_DEFS = {
    "A: Long":  ("long_cond",       +1), "A: Short": ("short_cond",      -1),
    "B: Long":  ("long_cond_comp",  +1), "B: Short": ("short_cond_comp", -1),
    "C: Long":  ("long_cond_wt",    +1), "C: Short": ("short_cond_wt",   -1),
}

# Candidate gating factors. (column, directional?) — directional ones are signed by
# trade direction so "higher = more supportive". These must be derivable BOTH in the
# harvest panel and live at screen time (cross-sectional cs_* are recomputed live).
GATE_FACTORS = [
    ("cs_breadth", True), ("cs_mom_rank", True), ("cs_conv_rank", True),
    ("F1_PriceMom", True), ("Conviction", True), ("Conviction_Delta", True),
    ("Pulse", True), ("Pulse_Delta", True), ("Liquidity_Osc", True), ("Liq_Vel", True),
    ("AT_Filter", True), ("Recent_Travel", True), ("HMM_dir", True), ("Regime_Confidence", False),
]
_FDIR = dict(GATE_FACTORS)


# ──────────────────────────────────────────────────────────────────────────────
# Cross-sectional feature construction (per date). Used identically at LEARN time
# (on the harvest panel) and APPLY time (on the live results_df cross-section).
# ──────────────────────────────────────────────────────────────────────────────
def ensure_cross_sectional(df: pd.DataFrame, date_col=None) -> pd.DataFrame:
    """Add cs_breadth / cs_mom_rank / cs_conv_rank + HMM_dir if absent.

    date_col=None means the whole frame is ONE cross-section (the live screener
    case: results_df is all symbols for a single analysis date). Otherwise group
    by date_col (the harvest-panel case)."""
    df = df.copy()
    if "HMM_dir" not in df.columns and {"HMM_Bull", "HMM_Bear"}.issubset(df.columns):
        df["HMM_dir"] = df["HMM_Bull"].astype(float) - df["HMM_Bear"].astype(float)

    def _add(g):
        f1 = g["F1_PriceMom"].astype(float) if "F1_PriceMom" in g else pd.Series(0.0, index=g.index)
        conv = g["Conviction"].astype(float) if "Conviction" in g else pd.Series(0.0, index=g.index)
        out = pd.DataFrame(index=g.index)
        out["cs_breadth"] = (f1 > 0).mean()                # scalar broadcast: fraction of universe up
        out["cs_mom_rank"] = f1.rank(pct=True) - 0.5       # peer momentum rank, centered
        out["cs_conv_rank"] = conv.rank(pct=True) - 0.5    # peer conviction rank, centered
        return out

    if date_col is None:
        cs = _add(df)
        for c in cs.columns:
            df[c] = cs[c]
    else:
        parts = []
        for _, g in df.groupby(date_col):
            parts.append(_add(g))
        cs = pd.concat(parts).reindex(df.index)
        for c in cs.columns:
            df[c] = cs[c]
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Gate learning + walk-forward validation (mirrors walk_forward.py, packaged).
# ──────────────────────────────────────────────────────────────────────────────
def _gate_lift(panel, ev_tr, r, sign, topk):
    """On a train slice, rank factors by in-sample gated lift; return top-k
    [(factor, '≥'|'<')] half-conditions (direction only — NO thresholds kept)."""
    base = r[ev_tr]; base = base[~np.isnan(base)]
    if len(base) < 60:
        return []
    base_mu = base.mean(); scored = []
    for col, directional in GATE_FACTORS:
        if col not in panel.columns:
            continue
        f = panel[col].to_numpy(dtype=float) * (sign if directional else 1.0)
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
        scored.append((kept.mean() - base_mu, col, "≥" if keep_high else "<"))
    scored.sort(key=lambda x: -x[0])
    return [(c, cmp) for lift, c, cmp in scored if lift > 0][:topk]


def _apply_rank_gate(panel, ev, sign, conds):
    """Apply direction-only conditions via SELF-CALIBRATING cross-sectional median
    of the (directional) factor over the CURRENT event set — no stored thresholds."""
    m = ev.copy()
    for col, cmp in conds:
        if col not in panel.columns:
            return np.zeros(len(panel), bool)  # required factor missing → no fire
        f = panel[col].to_numpy(dtype=float) * (sign if _FDIR.get(col, True) else 1.0)
        thr = np.nanmedian(f[ev]) if ev.any() else 0.0
        m = m & ((f >= thr) if cmp == "≥" else (f < thr))
    return m


def learn_gates(panel: pd.DataFrame, horizon: int = 5, topk: int = 3,
                train_win: int = 252, test_win: int = 63,
                activate_hit: float = 0.65, min_windows: int = 4) -> dict:
    """Learn + walk-forward-validate a gate per signal. Returns a GateModel dict:

        {horizon, topk, signals: {sigtype: {conds:[(factor,cmp)...], active:bool,
         hit:float, wf_mean:float, naked_mean:float, n_windows:int}}}

    A gate is ACTIVE only if positive in ≥ activate_hit of forward windows AND its
    walk-forward mean beats the naked baseline. The conds stored are the MODAL
    (most-frequently-selected) half-conditions across windows — the stable recipe.
    """
    p = panel.copy()
    if "Date" not in p.columns:
        return {"horizon": horizon, "topk": topk, "signals": {}}
    p = ensure_cross_sectional(p, date_col="Date")
    rcol = f"Ret_{horizon}b"
    if rcol not in p.columns:
        return {"horizon": horizon, "topk": topk, "signals": {}}
    retH = p[rcol].to_numpy(dtype=float)
    dates = pd.to_datetime(p["Date"].to_numpy())
    udates = np.sort(np.unique(dates))

    windows = []
    i = train_win
    while i + test_win <= len(udates):
        windows.append((udates[i - train_win], udates[i - 1], udates[i],
                        udates[min(i + test_win - 1, len(udates) - 1)]))
        i += test_win

    out = {"horizon": horizon, "topk": topk, "train_win": train_win,
           "test_win": test_win, "activate_hit": activate_hit, "signals": {}}

    from collections import Counter
    for sig, (col, sign) in SIGNAL_DEFS.items():
        if col not in p.columns:
            continue
        ev = p[col].fillna(False).to_numpy().astype(bool)
        r = retH * sign
        cond_counter = Counter()
        wf_rets, naked_rets, npos, ntot = [], [], 0, 0
        for tr_lo, tr_hi, te_lo, te_hi in windows:
            tr = (dates >= tr_lo) & (dates <= tr_hi)
            te = (dates >= te_lo) & (dates <= te_hi)
            ev_tr = ev & tr
            if ev_tr.sum() < 60:
                continue
            conds = _gate_lift(p, ev_tr, r, sign, topk)
            if not conds:
                continue
            for c in conds:
                cond_counter[c] += 1
            gated = _apply_rank_gate(p, ev & te, sign, conds)
            gr = r[gated]; gr = gr[~np.isnan(gr)]
            nk = r[ev & te]; nk = nk[~np.isnan(nk)]
            if len(gr) >= 20:
                wf_rets.append(gr.mean()); naked_rets.append(nk.mean() if len(nk) else np.nan)
                ntot += 1
                if gr.mean() > 0:
                    npos += 1
        if ntot < min_windows:
            out["signals"][sig] = dict(conds=[], active=False, hit=0.0, wf_mean=np.nan,
                                       naked_mean=np.nan, n_windows=ntot, reason="too few windows")
            continue
        hit = npos / ntot
        wf_mean = float(np.nanmean(wf_rets)); naked_mean = float(np.nanmean(naked_rets))
        modal = [c for c, _ in cond_counter.most_common(topk)]
        active = (hit >= activate_hit) and (wf_mean > naked_mean) and (wf_mean > 0)
        out["signals"][sig] = dict(
            conds=[list(c) for c in modal], active=bool(active), hit=round(hit, 3),
            wf_mean=round(wf_mean, 6), naked_mean=round(naked_mean, 6),
            n_windows=ntot, reason="" if active else "failed walk-forward",
        )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Apply at screen time. results_df = all symbols for ONE date (a cross-section).
# Adds Signal_Grade (0–1, self-calibrating) + Gate_Pass (bool) + Gate_Active per row.
# ──────────────────────────────────────────────────────────────────────────────
def apply_gates(results_df: pd.DataFrame, model: dict) -> pd.DataFrame:
    df = results_df.copy()
    if df.empty or not model or not model.get("signals"):
        df["Signal_Grade"] = np.nan
        df["Gate_Pass"] = False
        df["Gate_Active"] = False
        return df
    df = ensure_cross_sectional(df, date_col=None)  # whole frame = one cross-section
    sig_col = df["SignalType"] if "SignalType" in df.columns else pd.Series("-", index=df.index)

    grade = np.full(len(df), np.nan)
    gate_pass = np.zeros(len(df), bool)
    gate_active = np.zeros(len(df), bool)

    for sig, (col, sign) in SIGNAL_DEFS.items():
        m = (sig_col == sig).to_numpy()
        if not m.any():
            continue
        spec = model["signals"].get(sig)
        if not spec:
            continue
        gate_active[m] = bool(spec.get("active"))
        conds = [tuple(c) for c in spec.get("conds", [])]
        if not conds:
            # No usable gate → grade is neutral, doesn't pass an (inactive) gate.
            grade[m] = 0.5
            continue
        # Per-condition support score over THIS cross-section's fired-signal members
        ev = m  # the signal's members are its own cross-section reference
        passes = np.ones(m.sum(), bool)
        sub_idx = np.where(m)[0]
        score = np.zeros(m.sum())
        for col_f, cmp in conds:
            if col_f not in df.columns:
                passes[:] = False
                continue
            f = df[col_f].to_numpy(dtype=float) * (sign if _FDIR.get(col_f, True) else 1.0)
            thr = np.nanmedian(f[m]) if m.any() else 0.0
            cond_ok = (f[sub_idx] >= thr) if cmp == "≥" else (f[sub_idx] < thr)
            passes &= cond_ok
            score += cond_ok.astype(float)
        grade[sub_idx] = score / max(len(conds), 1)   # fraction of conditions met [0,1]
        gate_pass[sub_idx] = passes

    df["Signal_Grade"] = grade
    df["Gate_Pass"] = gate_pass
    df["Gate_Active"] = gate_active
    return df
