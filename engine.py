"""
Sanket Ranking Engine — Cross-Sectional Momentum (long tilt) with a reversion entry overlay.

Why this replaced the reversion core
-------------------------------------
The prior engine ranked on short-horizon cross-sectional REVERSION. Rebuilt from a clean
slate with a reproducible, cost-aware harness (``research.py``), that thesis does not hold
up as something you would trade: the reversion rank-IC is real (+0.03, t~7) but it lives at
a 1–2 day horizon where turnover is ~80%, so after realistic costs the long/short book is
net NEGATIVE (−23%/yr at 25bps). It predicts; you cannot harvest it.

The same harness, searching a battery of signals on 100 NIFTY-100 names (2016–2026,
adjusted, non-overlapping monthly, cost-charged), found the edge that SURVIVES costs:

    **12-1 cross-sectional momentum, long-only, monthly rebalance.**
    Long-only top quintile: ~+6%/yr EXCESS over the equal-weight universe, excess Sharpe
    ~0.6, turnover only ~21%, cost-robust (0.55 even at 25bps), worst excess year 2018 −2%.

Momentum's IC GROWS with horizon (+0.03 @21d, +0.047 @63d), so a monthly book turns slowly
and the gross edge clears costs — the mirror image of reversion. The old engine deleted
momentum for "anti-predicting", but it measured momentum at SHORT horizons where reversion
dominates; at the correct 12-1 monthly horizon momentum is the survivor.

Honest limits (attached to every claim this engine makes)
---------------------------------------------------------
* BETA, not a money-printer: the top-quintile's ~30%/yr absolute is mostly market beta
  (absolute Sharpe 1.34 barely beats the eq-wt benchmark's 1.29). The real skill is the
  ~+6% EXCESS. This engine reports the tilt, never the 30%.
* DECAY is real: momentum IC went negative 2024–2026 (momentum crashes cluster in high-vol
  turning points). The alpha-health monitor exists precisely to stand the book down then,
  and the vol-regime factor damps momentum in HIGH/EXTREME vol.
* SURVIVORSHIP: validated on current constituents → the true point-in-time number is lower.

What this engine is
-------------------
A daily, LONG-tilt decision-support ranker for a trader who holds for weeks: momentum ranks
WHAT to hold; a short-horizon reversion overlay refines WHEN to enter (favour names that
have pulled back inside their uptrend). Bottom-momentum names are surfaced as
underweight/avoid ("Short" in the retained column names) — on NSE cash they are not
executable single-name shorts, only F&O/underweight.

Output column contract (unchanged, so the UI renders without edits)
-------------------------------------------------------------------
``compute_ranking(df)`` adds:
  Rev_Score (now the momentum alpha score), Rev_Rank_Pct, Conviction, Side, Entry_Timing,
  Priority_Long, Priority_Short, Priority_Long_pct, Priority_Short_pct,
  Intel_Confidence, Intel_Source, Intel_Stars, Meta_Score, Meta_Tier, Meta_Source, Meta_Reason
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Momentum (primary alpha) ─────────────────────────────────────────────────────────────
MOM_FORM = 252     # 12-month formation window
MOM_SKIP = 21      # skip the most recent ~month (dodge short-term reversal contamination)
MOM_FORM_2 = 126   # 6-month formation (coverage fallback for shorter histories)

# ── Reversion overlay (entry timing only — NOT the ranking edge) ──────────────────────────
# Real at 1–2 days; used to prefer momentum names that have pulled back, never to pick the
# universe. Kept intentionally weak (a conviction nudge, not a rank driver).
REV_RETURN_LAGS = (2, 5)
REV_MA_WINDOWS  = (5, 10)
REV_RANGE_WIN   = 10

# Vol-regime suitability for MOMENTUM. Flipped vs the old reversion map: momentum crashes in
# high-vol turning points, so damp it there; it earns cleanest in LOW/NORMAL tape. A prior
# from the 2020 & 2024–26 momentum decays + the general momentum-crash literature, not a fit.
VOL_REGIME_MOM = {'LOW': 1.10, 'NORMAL': 1.05, 'HIGH': 0.85, 'EXTREME': 0.55}

# Surface a name as an actionable tail only past this conviction; below it, context only.
SIDE_CONVICTION_MIN = 0.55

# Forward horizons used by the harvest / alpha-health monitor.
HOLD_HORIZONS = [5, 10, 21, 42, 63]


# ════════════════════════════════════════════════════════════════════════════════════════
# PER-SYMBOL FEATURES  (time-series; run once per name before cross-sectional ranking)
# ════════════════════════════════════════════════════════════════════════════════════════
def add_alpha_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the primary momentum features plus the reversion entry-overlay features to a
    single symbol's frame. Requires Open/High/Low/Close; ATR14 added if missing. Safe on
    short frames (momentum → NaN until ~MOM_FORM+MOM_SKIP bars exist)."""
    df = df.copy()
    close = df['Close']
    if 'ATR14' not in df.columns:
        h, l, c = df['High'], df['Low'], close
        pc = c.shift(1)
        tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        df['ATR14'] = tr.rolling(14).mean()
    atr = df['ATR14'].clip(lower=1e-9)

    # PRIMARY — 12-1 (and 6-1 coverage) momentum, price-return based (skip last month).
    df['_mom_12_1'] = close.shift(MOM_SKIP) / close.shift(MOM_FORM) - 1.0
    df['_mom_6_1'] = close.shift(MOM_SKIP) / close.shift(MOM_FORM_2) - 1.0

    # OVERLAY — reversion entry-timing features (used weakly by conviction, never the rank).
    for k in REV_RETURN_LAGS:
        df[f'_rev_ret{k}'] = (close - close.shift(k)) / atr
    for w in REV_MA_WINDOWS:
        df[f'_rev_dist{w}'] = (close - close.rolling(w).mean()) / atr
    hi = df['High'].rolling(REV_RANGE_WIN).max()
    lo = df['Low'].rolling(REV_RANGE_WIN).min()
    df['_rev_rngpos'] = (close - lo) / (hi - lo).replace(0, np.nan)

    df['ATR_Pct'] = (df['ATR14'] / close.clip(lower=1e-9)).clip(0, 1)
    return df


# Back-compat alias: sanket.py's call site still says add_reversion_features.
add_reversion_features = add_alpha_features


def _mom_cols() -> list[str]:
    return ['_mom_12_1', '_mom_6_1']


# ════════════════════════════════════════════════════════════════════════════════════════
# ALPHA-HEALTH MONITOR  (is the momentum edge currently working?)
# ════════════════════════════════════════════════════════════════════════════════════════
# The system measures its OWN realized edge: the trailing mean daily cross-sectional IC of the
# momentum score vs forward return. Momentum decays (it went negative 2024–2026); conviction
# scales by this so the book stands down when the factor is off. Thresholds sit around
# momentum's short-horizon IC scale (~+0.02 healthy, ≤0 dead).
ALPHA_HEALTH_FLOOR = 0.35
_ALPHA_IC_GOOD = 0.020
_ALPHA_IC_DEAD = -0.005


def alpha_health(trailing_ic: float | None) -> float:
    """Map a trailing realized-IC reading to a conviction multiplier in [FLOOR, 1]."""
    if trailing_ic is None or not np.isfinite(trailing_ic):
        return 0.75   # cold start → mild discount, not full confidence
    x = (trailing_ic - _ALPHA_IC_DEAD) / (_ALPHA_IC_GOOD - _ALPHA_IC_DEAD)
    return float(np.clip(ALPHA_HEALTH_FLOOR + (1 - ALPHA_HEALTH_FLOOR) * x, ALPHA_HEALTH_FLOOR, 1.0))


def cross_sectional_ic(score: pd.Series, fwd_ret: pd.Series, min_n: int = 15) -> float:
    """Spearman rank-IC of score vs forward return over one cross-section. NaN if too thin."""
    s = pd.concat([score, fwd_ret], axis=1).dropna()
    if len(s) < min_n or s.iloc[:, 0].nunique() < 3:
        return np.nan
    return float(s.iloc[:, 0].corr(s.iloc[:, 1], method='spearman'))


# ════════════════════════════════════════════════════════════════════════════════════════
# CROSS-SECTIONAL RANKING  (the headline computation; one date's universe)
# ════════════════════════════════════════════════════════════════════════════════════════
def _robust_z(s: pd.Series) -> pd.Series:
    """Within-cross-section robust z (median / MAD), clipped. Resistant to single outliers."""
    med = s.median()
    mad = (s - med).abs().median()
    return ((s - med) / (1.4826 * mad + 1e-9)).clip(-4, 4)


def compute_ranking(df: pd.DataFrame, alpha_health_mult: float = 1.0) -> pd.DataFrame:
    """Rank one date's cross-section by the validated 12-1 momentum score (long tilt).

    df: one row per symbol carrying _mom_12_1/_mom_6_1 (+ optional _rev_* overlay,
        Vol_Regime, Regime_Confidence). alpha_health_mult: live edge-health in [0,1].

    Adds the full output contract and returns the frame sorted by Priority_Long desc.
    Pure & deterministic.
    """
    df = df.copy()
    n = len(df)
    contract = ('Rev_Score', 'Rev_Rank_Pct', 'Conviction', 'Side', 'Entry_Timing',
                'Priority_Long', 'Priority_Short', 'Priority_Long_pct', 'Priority_Short_pct',
                'Intel_Confidence', 'Intel_Source', 'Intel_Stars',
                'Meta_Score', 'Meta_Tier', 'Meta_Source', 'Meta_Reason')
    if n == 0:
        for c in contract:
            df[c] = pd.Series(dtype=float)
        return df

    # ── 1. Momentum score: robust-z of 12-1 (fall back to 6-1 where history is short) ──
    mom = df['_mom_12_1'].astype(float) if '_mom_12_1' in df.columns else pd.Series(np.nan, index=df.index)
    if '_mom_6_1' in df.columns:
        mom = mom.fillna(df['_mom_6_1'].astype(float))
    mom_z = _robust_z(mom)
    df['Rev_Score'] = mom_z.fillna(0.0)          # retained column name; now the MOMENTUM alpha score

    # ── 2. Cross-sectional rank percentile [0,100] ──
    rank_pct = df['Rev_Score'].rank(pct=True) if n >= 2 else pd.Series(0.5, index=df.index)
    df['Rev_Rank_Pct'] = (rank_pct * 100).round(2)

    # ── 3. Reversion entry-timing overlay [0,1] (high = pulled back inside the trend) ──
    if '_rev_ret2' in df.columns:
        entry = (-_robust_z(df['_rev_ret2'].astype(float))).rank(pct=True).fillna(0.5)
    else:
        entry = pd.Series(0.5, index=df.index)
    df['Entry_Timing'] = entry.round(3)

    # ── 4. Regime factor (momentum vol-regime suitability) + confidence temper ──
    regime_f = df['Vol_Regime'].map(VOL_REGIME_MOM).fillna(1.0) if 'Vol_Regime' in df.columns \
        else pd.Series(1.0, index=df.index)
    reg_conf = df['Regime_Confidence'].astype(float).fillna(0.6) if 'Regime_Confidence' in df.columns \
        else pd.Series(0.7, index=df.index)

    # ── 5. Side: top tail = Long, bottom tail = Short/underweight, by rank ──
    tail_cut = 0.5 + 0.5 * SIDE_CONVICTION_MIN
    side = np.where(rank_pct >= tail_cut, 'Long',
            np.where(rank_pct <= 1 - tail_cut, 'Short', '—'))
    df['Side'] = side

    # ── 6. Conviction [0,1]: tail strength × alpha-health × regime × confidence × entry nudge ──
    # Entry nudge is side-aware: for Longs a deeper pullback (high Entry_Timing) is a better
    # entry; for Shorts the sign flips. Kept small ([0.9,1.1]) — timing, not thesis.
    tail_strength = (2.0 * (rank_pct - 0.5)).abs().clip(0, 1)
    health = float(np.clip(alpha_health_mult, 0.0, 1.0))
    entry_long = 0.9 + 0.2 * entry
    entry_short = 0.9 + 0.2 * (1.0 - entry)
    entry_nudge = pd.Series(np.where(side == 'Long', entry_long,
                            np.where(side == 'Short', entry_short, 1.0)), index=df.index)
    conviction = (tail_strength
                  * health
                  * (regime_f.clip(0.4, 1.3) / 1.3)
                  * (0.7 + 0.3 * reg_conf.clip(0, 1))
                  * entry_nudge)
    df['Conviction'] = conviction.clip(0, 1).fillna(0.0)

    # ── 7. UI contract mapping (retained names, momentum semantics) ──
    scale = 100.0
    df['Priority_Long'] = (df['Rev_Score'] * scale).fillna(0.0)
    df['Priority_Short'] = (-df['Rev_Score'] * scale).fillna(0.0)
    df['Priority_Long_pct'] = (rank_pct * 100)
    df['Priority_Short_pct'] = ((1 - rank_pct) * 100)

    fired = df['Side'] != '—'
    df['Intel_Confidence'] = np.where(fired, df['Conviction'], np.nan)
    df['Intel_Source'] = np.where(fired, 'momentum', '')
    bands = np.array([0.20, 0.35, 0.50, 0.65])
    stars = 1 + np.digitize(np.where(fired, df['Conviction'], -1.0), bands)
    df['Intel_Stars'] = np.where(fired, stars, 0).astype(int)

    meta = (df['Conviction'] * (0.5 + 0.5 * tail_strength)).clip(0, 1)
    df['Meta_Score'] = np.where(fired, meta, np.nan)
    df['Meta_Tier'] = np.where(fired, np.clip(np.digitize(meta, [0.30, 0.50, 0.70]), 0, 3), 0).astype(int)
    df['Meta_Source'] = np.where(fired, 'fused', '')
    df['Meta_Reason'] = [
        (f"{s} · mom rank {rp:.0f}%ile · conv {cv:.2f} · entry {et:.0%}"
         + (" (underweight/F&O only)" if s == 'Short' else "")
         if s != '—' else "context only")
        for s, rp, cv, et in zip(df['Side'], df['Rev_Rank_Pct'], df['Conviction'], df['Entry_Timing'])
    ]

    return df.sort_values('Priority_Long', ascending=False, kind='stable')
