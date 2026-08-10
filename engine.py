"""
Sanket Signal Engine — SB v8 Close-Location Reversal (port of ``sb_v8.pine``).

What this is
------------
One signal, measured on one variable::

    clv = ((close - low) - (high - close)) / (high - low)

i.e. **where price closes inside its own daily range** (-1 = closed on the low,
+1 = closed on the high). Its cross-time z-score over a trailing window is the entire
engine. The sign is the finding: a **strong close predicts WEAKNESS**, so the fade of a
weak close is the buy.

Signals (exactly the two the Pine plots)
----------------------------------------
* **BUY — green triangle** (``buy_cond``): ``z < -thr``. A weak close → expect mean
  reversion up. Holdout-confirmed on both eras.
* **SELL — yellow diamond** (``sell_cond``): ``z > +thr``. A strong close → expect
  weakness. Note the Pine labels this side CAUTION rather than a short entry: its
  drift-free holdout was +0.0094 with a CI of [-0.030, +0.052], i.e. it did **not**
  confirm out of sample. It is surfaced here as a sell signal; that caveat travels with
  it into the UI reference card.

Why the event form (and not a continuous position)
--------------------------------------------------
Holding a continuous position on this signal turns over daily, costs ~12%/yr at 3bp, and
nets Sharpe -0.48. Firing only on ``|z| > 1.5`` (~9.3% of days) and holding ~10 days cuts
turnover ~35x, which is what makes it tradeable at all: discovery NET +0.124 / holdout NET
+0.132 pooled at 3bp, and holdout NET +0.430 on US equity indices and sectors. Turnover,
not signal strength, was the binding constraint.

Scope is not universal
----------------------
Drift-free and holdout-confirmed **only** on US equity indices and US sectors. India
indices are positive but the CI includes zero (n=239). Commodities, FX, rates, credit and
international equity did not survive. ``instrument_class`` wires the system's universe
selector to that measured expectancy so the UI can state which case the user is in —
honesty wiring, not a tunable parameter.

The edge is small and decaying
------------------------------
1993-99 IC -0.113 → 2000-06 -0.065 → 2007-13 -0.056 → holdout 2014-19 -0.025,
2020-26 -0.026. Roughly a quarter of its 1990s strength, now stable there. This is an
overlay, not a system.

Horizon
-------
5-10 trading days. There is **no intraday edge** here; none was found and none is claimed.
Entry is the next session's open after the signal bar closes.

Bar convention (one deliberate difference from the Pine)
-------------------------------------------------------
The Pine reads ``z[1]`` inside ``request.security(..., "D", ...)`` so that an *intraday*
chart cannot repaint a daily signal. Sanket evaluates completed daily (or weekly) bars
directly, so that shift is unnecessary: a signal fires on the bar whose close produced it,
and entry is the next session's open — the same trade the Pine backtested (EXEC-B). The one
carry-over: a signal on a session that has not closed yet is provisional until it does.

Output column contract (``compute_ranking``)
--------------------------------------------
  SB_Score, SB_Rank_Pct, Fade_Score, Conviction, Side,
  Priority_Long, Priority_Short, Priority_Long_pct, Priority_Short_pct,
  Signal_Reason
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Measured plateaus from the Pine's inputs — defaults, not fitted values ────────────────
# Z-score lookback: net Sharpe stays positive in BOTH eras across 63-504 daily bars; 252 is
# mid-plateau. Weekly has no measured plateau (the study was daily) — 52 bars is one year,
# the closest structural analogue, and is flagged as an extrapolation in the UI.
SB_Z_LOOK_DAILY  = 252
SB_Z_LOOK_WEEKLY = 52

# Threshold: 1.0 fires 44% of days and loses to costs; 2.0 fires 0.1% and failed holdout;
# 1.5 fires 9.3% and is net-positive in both eras.
SB_THRESHOLD = 1.5

# Hold horizon: the edge lives at 5-10 days. Below 5, turnover cost exceeds the gross edge.
SB_HORIZON = 10

# Round-trip cost assumption for the cost gate. Measured breakeven ~7bp pooled; on the two
# established classes the event form stays net-positive past 10bp.
SB_COST_BPS = 3.0

# Forward horizons the Historical Range harvest attaches as Ret_*b labels. Centred on the
# 5-10 day window where this edge actually lives, with 1d and 21d as decay bookends.
HOLD_HORIZONS = [1, 5, 10, 21]


# ════════════════════════════════════════════════════════════════════════════════════════
# INSTRUMENT CLASS  (the Pine's "Instrument class" input, wired to Sanket's universe)
# ════════════════════════════════════════════════════════════════════════════════════════
# Holdout 2014-2026, 10-day horizon, entry next open, each instrument's own mean forward
# return removed within era (so "equities went up" cannot contribute). `established` = the
# block-bootstrap CI excluded zero. These numbers are measured; none is asserted.
INSTRUMENT_CLASSES = [
    "US index / ETF", "US sector ETF", "India index", "International equity",
    "Commodity", "FX", "Rates / Credit", "Other / unknown",
]

CLASS_EDGE = {
    "US index / ETF":        0.121,
    "US sector ETF":         0.068,
    "India index":           0.089,
    "International equity":  0.003,
    "Commodity":             0.028,
    "FX":                    0.035,
    "Rates / Credit":       -0.003,
    "Other / unknown":       0.000,
}

CLASS_HIT = {
    "US index / ETF":       57.5,
    "US sector ETF":        54.9,
    "India index":          51.9,
    "International equity": 53.2,
    "Commodity":            51.0,
    "FX":                   51.9,
    "Rates / Credit":       51.1,
    "Other / unknown":      50.0,
}

# Only these two had a bootstrap CI excluding zero after drift removal.
ESTABLISHED_CLASSES = ("US index / ETF", "US sector ETF")

# Sanket universe → the Pine's instrument class. This is what "wire the indicator's asset
# class to the system's asset selection" means: the user picks a universe, the measured
# out-of-sample expectancy for that asset class follows automatically.
#
# Caveat carried into the UI: the study measured index- and ETF-level instruments. On a
# constituent universe (individual stocks inside an index) the class expectancy is
# indicative of the asset class, not a measurement on those single names.
UNIVERSE_CLASS_MAP = {
    "US Indexes":     "US index / ETF",
    "India Indexes":  "India index",
    "Global Indexes": "International equity",
    "ETF Index":      "India index",        # NSE ETFs track Indian indices / sectors
    "Commodities":    "Commodity",
    "Currency":       "FX",
    "Global Macro":   "Rates / Credit",
    "Crypto":         "Other / unknown",    # the study covered no digital assets
}


def instrument_class(universe: str, selected_index: str | None = None) -> str:
    """Resolve Sanket's universe selection to the Pine's instrument class.

    ``selected_index`` is accepted so a future sub-universe split (e.g. India sectoral vs
    benchmark) can refine the answer without changing call sites.
    """
    return UNIVERSE_CLASS_MAP.get(universe, "Other / unknown")


def class_edge(iclass: str) -> float:
    """Measured drift-free holdout expectancy (in vol units) for an instrument class."""
    return CLASS_EDGE.get(iclass, 0.0)


def class_hit(iclass: str) -> float:
    """Measured holdout hit rate (%) for an instrument class."""
    return CLASS_HIT.get(iclass, 50.0)


def is_established(iclass: str) -> bool:
    """True only where the bootstrap CI excluded zero after drift removal."""
    return iclass in ESTABLISHED_CLASSES


def cost_ok(cost_bps: float, iclass: str) -> bool:
    """Is the event form still net-positive at this round-trip cost?

    Measured breakeven is ~7bp pooled across all 39 instruments; on the two established
    classes it stays positive past 10bp.
    """
    try:
        c = float(cost_bps)
    except (TypeError, ValueError):
        return False
    return c <= (10.0 if is_established(iclass) else 7.0)


def z_look_for(timeframe: str) -> int:
    """Z-score lookback for a Sanket timeframe (Daily 252 bars / Weekly 52 bars)."""
    return SB_Z_LOOK_WEEKLY if str(timeframe) == "Weekly" else SB_Z_LOOK_DAILY


def min_bars_for(z_look: int) -> int:
    """Bars a symbol needs before it can carry a signal (the Pine's ``zLook + 2`` warmup)."""
    return int(z_look) + 2


# ════════════════════════════════════════════════════════════════════════════════════════
# PER-SYMBOL FEATURES  (time-series; run once per name before cross-sectional ranking)
# ════════════════════════════════════════════════════════════════════════════════════════
def add_sb_features(df: pd.DataFrame,
                    z_look: int = SB_Z_LOOK_DAILY,
                    thr: float = SB_THRESHOLD,
                    horizon: int = SB_HORIZON) -> pd.DataFrame:
    """Attach the SB v8 close-location signal to one symbol's OHLC frame.

    Columns written:
      ``SB_CLV``       close location in [-1, +1] (Pine ``f_clv``)
      ``SB_Z``         z-score of SB_CLV over ``z_look`` bars; NaN until warm
      ``Fade_Score``   ``-SB_Z`` — positive = bullish. The sign flip IS the finding.
      ``buy_cond``     green triangle: ``SB_Z < -thr`` (weak close → fade long)
      ``sell_cond``    yellow diamond: ``SB_Z > +thr`` (strong close → sell)
      ``SB_Hold_Dir``  +1 inside a buy window, -1 inside a sell window, 0 outside
      ``SB_Hold_Age``  bars since that window opened (0 = fired on this bar)
      ``SB_State``     WARMING UP / BUY / SELL / NEUTRAL for the bar

    ``ta.stdev`` in Pine is the population standard deviation, so ``ddof=0`` here — using
    the sample stdev would shift every z by ~0.2% and drift the fire rate off the measured
    9.3%. Safe on short frames: SB_Z is NaN until ``z_look`` bars exist and nothing fires.
    """
    df = df.copy()
    high, low, close = df['High'], df['Low'], df['Close']
    z_look = max(int(z_look), 2)

    # ── Core measurement: where the bar closed inside its own range ──
    rng = (high - low)
    clv = ((close - low) - (high - close)) / rng.where(rng > 0)
    clv = clv.fillna(0.0)

    m = clv.rolling(z_look).mean()
    s = clv.rolling(z_look).std(ddof=0)
    z = (clv - m) / s.where(s > 0)

    df['SB_CLV']     = clv
    df['SB_Z']       = z
    df['Fade_Score'] = -z

    # ── The two plotted events ──
    buy_cond  = (z < -float(thr)).fillna(False)
    sell_cond = (z > +float(thr)).fillna(False)
    df['buy_cond']  = buy_cond.to_numpy(dtype=bool)
    df['sell_cond'] = sell_cond.to_numpy(dtype=bool)

    # ── Hold window (Pine's sinceSig / sigDir / active), vectorised ──
    # A fire opens a window in its direction; a later fire re-opens it. Outside `horizon`
    # bars the window has expired — the measured edge does not extend past it.
    n = len(df)
    pos = np.arange(n, dtype=float)
    fires = (buy_cond | sell_cond).to_numpy(dtype=bool)
    fire_dir = np.where(buy_cond.to_numpy(dtype=bool), 1.0,
                        np.where(sell_cond.to_numpy(dtype=bool), -1.0, np.nan))

    last_fire = pd.Series(np.where(fires, pos, np.nan), index=df.index).ffill()
    held_dir  = pd.Series(fire_dir, index=df.index).ffill()
    age       = pos - last_fire.to_numpy(dtype=float)
    in_window = np.isfinite(age) & (age <= int(horizon))

    df['SB_Hold_Dir'] = np.where(in_window, held_dir.fillna(0.0).to_numpy(dtype=float), 0.0).astype(int)
    df['SB_Hold_Age'] = np.where(in_window, age, np.nan)

    df['SB_State'] = np.select(
        [~np.isfinite(z.to_numpy(dtype=float)), buy_cond, sell_cond],
        ['WARMING UP', 'BUY', 'SELL'],
        default='NEUTRAL',
    )
    return df


# ════════════════════════════════════════════════════════════════════════════════════════
# CROSS-SECTIONAL RANKING  (one date's universe, ordered by the fade score)
# ════════════════════════════════════════════════════════════════════════════════════════
def compute_ranking(df: pd.DataFrame,
                    iclass: str = "Other / unknown",
                    cost_bps: float = SB_COST_BPS,
                    thr: float = SB_THRESHOLD) -> pd.DataFrame:
    """Rank one date's cross-section by the SB v8 fade score.

    df: one row per symbol carrying ``SB_Z`` (and optionally ``Fade_Score``).
    iclass / cost_bps: the measured-expectancy and cost gates that scale conviction.

    Adds the output contract and returns the frame sorted by ``Priority_Long`` desc
    (warming-up rows, whose score is NaN, sort last). Pure & deterministic.
    """
    df = df.copy()
    contract = ('SB_Score', 'SB_Rank_Pct', 'Fade_Score', 'Conviction', 'Side',
                'Priority_Long', 'Priority_Short', 'Priority_Long_pct', 'Priority_Short_pct',
                'Signal_Reason')
    if len(df) == 0:
        for c in contract:
            df[c] = pd.Series(dtype=float)
        return df

    thr = float(thr)
    z = df['SB_Z'].astype(float) if 'SB_Z' in df.columns else pd.Series(np.nan, index=df.index)

    # ── 1. Score = fade score = -z. Positive = bullish (a weak close is the buy) ──
    fade = -z
    df['SB_Score']   = fade
    df['Fade_Score'] = fade

    # ── 2. Cross-sectional rank percentile [0,100] (NaN where still warming up) ──
    rank_pct = fade.rank(pct=True) if len(df) >= 2 else pd.Series(0.5, index=df.index)
    df['SB_Rank_Pct'] = (rank_pct * 100).round(2)

    # ── 3. Side: only a fired event is actionable; everything else is context ──
    #    Green triangle → Buy, yellow diamond → Sell. Sub-threshold rows are '—': the
    #    measured edge is in the EVENT, not in the continuous score.
    df['Side'] = np.where(z < -thr, 'Buy', np.where(z > thr, 'Sell', '—'))

    # ── 4. Conviction [0,1] = |z| magnitude × class expectancy × cost gate ──
    # Magnitude: |z|/3 — the threshold (1.5σ) lands at 0.50 and a 3σ close-location
    # extreme at 1.00. Class factor grades how well the asset class held up out of
    # sample; the cost gate halves conviction once the assumed round-trip cost passes
    # the measured breakeven, because past it the event form is net negative.
    mag = (z.abs() / 3.0).clip(0.0, 1.0)
    base = 0.30 + 0.70 * mag
    edge = class_edge(iclass)
    if is_established(iclass):
        class_f = 1.00          # CI excluded zero after drift removal
    elif edge >= 0.05:
        class_f = 0.75          # positive but the CI includes zero (India index)
    elif edge > 0.0:
        class_f = 0.55          # nominally positive, did not survive
    else:
        class_f = 0.40          # zero or negative expectancy for this class
    cost_f = 1.0 if cost_ok(cost_bps, iclass) else 0.5
    df['Conviction'] = (base * class_f * cost_f).clip(0.0, 1.0).fillna(0.0)

    # ── 5. UI contract mapping ──
    scale = 100.0
    df['Priority_Long']      = fade * scale
    df['Priority_Short']     = -fade * scale
    df['Priority_Long_pct']  = rank_pct * 100
    df['Priority_Short_pct'] = (1 - rank_pct) * 100

    est_note = "" if is_established(iclass) else f" · {iclass} not established OOS"
    df['Signal_Reason'] = [
        ("warming up — needs a full z-score lookback" if not np.isfinite(zz) else
         f"BUY · weak close z {zz:+.2f} · fade up, hold 5-10d, enter next open{est_note}"
         if sd == 'Buy' else
         f"SELL · strong close z {zz:+.2f} · short side failed holdout confirmation{est_note}"
         if sd == 'Sell' else
         f"context only · z {zz:+.2f} inside ±{thr:.1f}σ")
        for zz, sd in zip(z, df['Side'])
    ]

    return df.sort_values('Priority_Long', ascending=False, kind='stable',
                          na_position='last')
