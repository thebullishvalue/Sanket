"""
Sanket Ranking Engine — Cross-Sectional Reversion.

This module replaces the legacy ``priority_engine`` + ``intelligence`` stack. Its design and
every constant in it are justified in ``ARCHITECTURE.md`` and were validated on real NSE F&O
data (147 names, 5y, ~170k symbol-days) with walk-forward, cost-aware testing.

Thesis (what edge we exploit)
-----------------------------
Short-horizon **cross-sectional reversion**: names that have moved most relative to their own
volatility under-perform peers over the next 1–5 days, and vice-versa. Validated rank-IC of the
reversion composite vs 1–5d forward returns is +0.025…+0.031 (t up to +8.6), positive every
year 2021–2025, strongest in HIGH-vol regimes. The old momentum factor stack *anti*-predicted
(IC −0.023) and is removed.

What this engine is (and is not)
--------------------------------
It is a **decision-support ranker**: a daily ranked long/short shortlist with a conviction and
risk read for a human who holds for days. After realistic costs the raw signal only survives at
multi-day holding periods, so we never present it as a costless high-turnover strategy. The
engine is honest about *when its own edge is working* via a live alpha-health monitor.

Output column contract (kept identical to the old engine so the UI is unchanged)
--------------------------------------------------------------------------------
``compute_ranking(df)`` (cross-sectional, one date's universe) adds:
  Rev_Score, Rev_Rank_Pct, Conviction, Side,
  Priority_Long, Priority_Short, Priority_Long_pct, Priority_Short_pct,
  Intel_Confidence, Intel_Source, Intel_Stars,
  Meta_Score, Meta_Tier, Meta_Source, Meta_Reason

``Priority_*`` / ``Intel_*`` / ``Meta_*`` are retained names mapped onto the new semantics so the
existing tables/cards render without change; their *meaning* is documented below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Reversion feature set (per-symbol, computed on each name's own series) ──────────────
# Validated individually (all reversion-IC positive, t up to +8.6). Equal-weighted because
# they are collinear and a fitted weight vector did not beat the equal blend out of sample.
REV_RETURN_LAGS = (2, 5)        # k-bar return / ATR  (recent move to fade)
REV_MA_WINDOWS  = (5, 10)       # distance from SMA_w / ATR (overextension)
REV_RANGE_WIN   = 10            # position within rolling high/low range

# Conviction blends three orthogonal pieces, each in [0,1]:
#   rank strength  — how far into the tail this name sits today (cross-sectional)
#   alpha health   — is the reversion edge currently working? (live, self-measured)
#   regime factor  — vol-regime suitability for reversion (best HIGH, damped EXTREME)
VOL_REGIME_REV = {'LOW': 0.90, 'NORMAL': 1.00, 'HIGH': 1.15, 'EXTREME': 0.55}

# A name is surfaced as a Long/Short candidate only past this conviction (keeps the shortlist
# tight; below it the row is context-only). 0.55 ≈ top/bottom ~third on a healthy day.
SIDE_CONVICTION_MIN = 0.55

# Forward horizons used by the alpha-health monitor / harvest. 3 bars matched the strongest IC.
HOLD_HORIZONS = [2, 3, 5, 8, 13]


# ════════════════════════════════════════════════════════════════════════════════════════
# PER-SYMBOL FEATURES  (time-series; run once per name before cross-sectional ranking)
# ════════════════════════════════════════════════════════════════════════════════════════
def add_reversion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the raw (pre-cross-section) reversion features to a single symbol's frame.

    Requires Open/High/Low/Close + an ATR14 column (added if missing). Safe on short frames.
    """
    df = df.copy()
    close = df['Close']
    if 'ATR14' not in df.columns:
        h, l, c = df['High'], df['Low'], close
        pc = c.shift(1)
        tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        df['ATR14'] = tr.rolling(14).mean()
    atr = df['ATR14'].clip(lower=1e-9)

    for k in REV_RETURN_LAGS:
        df[f'_rev_ret{k}'] = (close - close.shift(k)) / atr
    for w in REV_MA_WINDOWS:
        df[f'_rev_dist{w}'] = (close - close.rolling(w).mean()) / atr
    hi = df['High'].rolling(REV_RANGE_WIN).max()
    lo = df['Low'].rolling(REV_RANGE_WIN).min()
    df['_rev_rngpos'] = (close - lo) / (hi - lo).replace(0, np.nan)

    # ATR% (per-name risk context, used by conviction + UI)
    df['ATR_Pct'] = (df['ATR14'] / close.clip(lower=1e-9)).clip(0, 1)
    return df


def _rev_feature_cols() -> list[str]:
    cols = [f'_rev_ret{k}' for k in REV_RETURN_LAGS]
    cols += [f'_rev_dist{w}' for w in REV_MA_WINDOWS]
    cols += ['_rev_rngpos']
    return cols


# ════════════════════════════════════════════════════════════════════════════════════════
# ALPHA-HEALTH MONITOR  (is the reversion edge currently working?)
# ════════════════════════════════════════════════════════════════════════════════════════
# The system measures its OWN realized edge: the trailing mean of daily cross-sectional IC of
# the reversion score vs realized forward return. On "healthy" days (~78% of history) forward
# IC ≈ +0.036; on dormant days ≈ +0.012 (noise). Conviction scales by this so the system stands
# down when its edge is off (e.g. the 2026 dormancy). Stored in session by sanket.py; here we
# expose a pure helper that maps a trailing-IC reading to a [0,1] health multiplier.
ALPHA_HEALTH_FLOOR = 0.35   # never fully zero — keep a dim screen, just low-conviction
_ALPHA_IC_GOOD = 0.020      # trailing IC at/above this → full health
_ALPHA_IC_DEAD = -0.005     # trailing IC at/below this → floor


def alpha_health(trailing_ic: float | None) -> float:
    """Map a trailing realized-IC reading to a conviction multiplier in [FLOOR, 1]."""
    if trailing_ic is None or not np.isfinite(trailing_ic):
        return 0.75   # unknown (cold start) → mild discount, not full confidence
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
    """Rank one date's cross-section by the validated reversion composite.

    df: one row per symbol, carrying the raw _rev_* features (from add_reversion_features),
        plus optional Vol_Regime / Regime_Confidence for the conviction/risk read.
    alpha_health_mult: [0,1] live edge-health from the monitor (1.0 = cold-start neutral).

    Adds the full output contract (see module docstring) and returns a frame sorted by
    Priority_Long descending. Pure & deterministic.
    """
    df = df.copy()
    n = len(df)
    if n == 0:
        for c in ('Rev_Score', 'Rev_Rank_Pct', 'Conviction', 'Side', 'Priority_Long',
                  'Priority_Short', 'Priority_Long_pct', 'Priority_Short_pct',
                  'Intel_Confidence', 'Intel_Source', 'Intel_Stars',
                  'Meta_Score', 'Meta_Tier', 'Meta_Source', 'Meta_Reason'):
            df[c] = pd.Series(dtype=float)
        return df

    # ── 1. Reversion score: equal blend of robustly-z'd, sign-flipped reversion features ──
    feats = [c for c in _rev_feature_cols() if c in df.columns]
    if feats:
        zsum = pd.Series(0.0, index=df.index)
        for c in feats:
            zsum = zsum - _robust_z(df[c].astype(float))   # minus = fade the move (reversion)
        rev = zsum / len(feats)
    else:
        rev = pd.Series(0.0, index=df.index)
    df['Rev_Score'] = rev.fillna(0.0)

    # ── 2. Cross-sectional rank percentile [0,100] (the universe-relative standing) ──
    if n >= 2:
        rank_pct = df['Rev_Score'].rank(pct=True)
    else:
        rank_pct = pd.Series(0.5, index=df.index)
    df['Rev_Rank_Pct'] = (rank_pct * 100).round(2)

    # ── 3. Regime factor (per-name vol-regime suitability for reversion) ──
    if 'Vol_Regime' in df.columns:
        regime_f = df['Vol_Regime'].map(VOL_REGIME_REV).fillna(1.0)
    else:
        regime_f = pd.Series(1.0, index=df.index)
    # regime confidence lightly tempers extreme convictions (low confidence → toward neutral)
    reg_conf = df['Regime_Confidence'].astype(float).fillna(0.6) if 'Regime_Confidence' in df.columns \
        else pd.Series(0.7, index=df.index)

    # ── 4. Conviction [0,1]: tail strength × alpha-health × regime suitability ──
    # tail_strength: how far from the 50th pct (0 at median, 1 at the extremes). The diffuse
    # reversion edge means we DON'T over-concentrate — conviction rises smoothly toward the tails.
    # `alpha_health_mult` is the FINAL health multiplier from the monitor (already in [FLOOR,1]);
    # we apply it directly here — no second floor — so dormant-edge days dampen conviction once.
    tail_strength = (2.0 * (rank_pct - 0.5)).abs().clip(0, 1)
    health = float(np.clip(alpha_health_mult, 0.0, 1.0))
    conviction = (tail_strength
                  * health
                  * (regime_f.clip(0.4, 1.3) / 1.3)
                  * (0.7 + 0.3 * reg_conf.clip(0, 1)))
    df['Conviction'] = conviction.clip(0, 1).fillna(0.0)

    # ── 5. Side: top tail = Long, bottom tail = Short by RANK (independent of conviction) ──
    # The shortlist is defined by cross-sectional standing, so it is never empty on a dormant
    # day — instead every surfaced name simply carries LOW conviction. Only the clear tails get
    # a side; the muddy middle stays context-only. This keeps the screen honest, not blank.
    tail_cut = 0.5 + 0.5 * SIDE_CONVICTION_MIN   # rank percentile defining the actionable tails
    side = np.where(rank_pct >= tail_cut, 'Long',
            np.where(rank_pct <= 1 - tail_cut, 'Short', '—'))
    df['Side'] = side

    # ── 6. UI contract mapping (retained names, new semantics) ──────────────────────────
    # Priority_Long  = reversion score in long orientation (higher = better long candidate)
    # Priority_Short = same, short orientation (= -score). Magnitude scaled to ~bps-equivalent
    # for the existing displays (the old engine emitted bp-equivalents).
    scale = 100.0
    df['Priority_Long'] = (df['Rev_Score'] * scale).fillna(0.0)
    df['Priority_Short'] = (-df['Rev_Score'] * scale).fillna(0.0)
    df['Priority_Long_pct'] = (rank_pct * 100)
    df['Priority_Short_pct'] = ((1 - rank_pct) * 100)

    # Intel_Confidence = the headline Conviction (the table's Intel column). Direction-aware:
    # it is the confidence in the row's *assigned side*; context rows get NaN like before.
    fired = df['Side'] != '—'
    df['Intel_Confidence'] = np.where(fired, df['Conviction'], np.nan)
    df['Intel_Source'] = np.where(fired, 'reversion', '')
    bands = np.array([0.20, 0.35, 0.50, 0.65])
    stars = 1 + np.digitize(np.where(fired, df['Conviction'], -1.0), bands)
    df['Intel_Stars'] = np.where(fired, stars, 0).astype(int)

    # Meta_Score = conviction × universe-rank standing (fused single number for the Meta column).
    # Tier 0–3 on fixed bands so a tier means the same across runs.
    meta = (df['Conviction'] * (0.5 + 0.5 * tail_strength)).clip(0, 1)
    df['Meta_Score'] = np.where(fired, meta, np.nan)
    df['Meta_Tier'] = np.where(fired, np.clip(np.digitize(meta, [0.30, 0.50, 0.70]), 0, 3), 0).astype(int)
    df['Meta_Source'] = np.where(fired, 'fused', '')
    df['Meta_Reason'] = [
        (f"{s} · rev rank {rp:.0f}%ile · conv {cv:.2f}" if s != '—' else "context only")
        for s, rp, cv in zip(df['Side'], df['Rev_Rank_Pct'], df['Conviction'])
    ]

    return df.sort_values('Priority_Long', ascending=False, kind='stable')
