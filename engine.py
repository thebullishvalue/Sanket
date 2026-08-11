"""
Sanket Signal Engine — CLOSE-LOCATION REVERSAL (CLR).

Ported from ``sb_v8.pine``. That file's title carries two halves — a legacy family tag and a
descriptive name — and only the descriptive half means anything: the tag belonged to a lineage
of session-breadth indicators whose premise this engine actually refutes (their core measure
tested flat, and the surviving variable has the opposite sign). So the engine is named for what
it measures, and the legacy tag is retained nowhere but the source filename.

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

Scope is not universal — and it is MEASURED, not asserted
---------------------------------------------------------
In the source study the edge was drift-free and holdout-confirmed only on US equity indices
and US sectors; India indices were positive with a CI including zero; commodities, FX, rates,
credit and international equity did not survive.

Those are *that study's* 39 instruments. This module does not apply them. Expectancy for the
universe on screen is measured by ``edge.py`` — event study, drift removed within era,
vol-normalised, block-bootstrapped over dates, with the effective sample size and minimum
detectable effect stated. The per-class numbers below survive only as a reference row to
compare a measurement against. Nothing here reads them to compute a signal or a conviction.

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
  CLR_Score, CLR_Rank_Pct, Fade_Score, Conviction, Side,
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
CLR_Z_LOOK_DAILY  = 252
CLR_Z_LOOK_WEEKLY = 52

# Threshold: 1.0 fires 44% of days and loses to costs; 2.0 fires 0.1% and failed holdout;
# 1.5 fires 9.3% and is net-positive in both eras.
CLR_THRESHOLD = 1.5

# Hold horizon: the edge lives at 5-10 days. Below 5, turnover cost exceeds the gross edge.
CLR_HORIZON = 10

# Round-trip cost assumption for the cost gate. Measured breakeven ~7bp pooled; on the two
# established classes the event form stays net-positive past 10bp.
CLR_COST_BPS = 3.0

# Forward horizons the Historical Range harvest attaches as Ret_*b labels. Centred on the
# 5-10 day window where this edge actually lives, with 1d and 21d as decay bookends.
HOLD_HORIZONS = [1, 5, 10, 21]


# ════════════════════════════════════════════════════════════════════════════════════════
# INSTRUMENT CLASS  (the Pine's "Instrument class" input, wired to Sanket's universe)
# ════════════════════════════════════════════════════════════════════════════════════════
# ⚠ REFERENCE PRIOR ONLY — NOT USED TO COMPUTE ANYTHING.
#
# These are the source study's published per-class results (holdout 2014-2026, 10-day
# horizon, entry next open, each instrument's own mean forward return removed within era;
# `established` = the block-bootstrap CI excluded zero). They were once wired into
# conviction as a hardcoded lookup. That was indefensible: eight frozen constants from
# someone else's 39 instruments cannot cover a universe the study never touched, cannot
# apply an asset-class claim to instrument-level decisions, and cannot report that the edge
# has decayed — while the study's own headline is that it fell 4x since the 1990s.
#
# Expectancy is now MEASURED per universe by `edge.py`, from the user's own symbols, at
# these same pre-declared parameters. What remains here is a labelled comparison line: "the
# source study measured this class at +0.121; here is what we measure on your universe."
# `compute_ranking` does not read it, and `instrument_class` exists only to pick which
# reference row to display.
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

# Pooled cost breakeven across all 39 instruments in the source study — the fallback the cost
# gate uses before a study has measured this universe's actual trading cost. See `cost_ok`.
POOLED_BREAKEVEN_BPS = 7.0

# The largest drift-free effect the source study found on any asset class (+0.121, US equity
# indices). Used as a CEILING, not a forecast: it is the most this signal has ever been worth
# anywhere, so a trading cost exceeding it cannot be survived by any plausible version of the
# edge. That makes it the right yardstick for a cost gate — and, in `edge.py`, for deciding
# when a test is too underpowered to say anything.
LARGEST_KNOWN_EFFECT = 0.121

# Sanket universe → the Pine's instrument class. Used ONLY to select which reference row to
# display beside the measured result, so the reader can compare their universe against the
# source study's published number for the nearest asset class.
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
    """Which reference row to show beside the measured result. Display only.

    ``selected_index`` is accepted so a future sub-universe split can refine the label
    without changing call sites.
    """
    return UNIVERSE_CLASS_MAP.get(universe, "Other / unknown")


def class_edge(iclass: str) -> float:
    """The SOURCE STUDY's published expectancy for a class — a reference prior, not ours."""
    return CLASS_EDGE.get(iclass, 0.0)


def class_hit(iclass: str) -> float:
    """The SOURCE STUDY's published hit rate for a class — a reference prior, not ours."""
    return CLASS_HIT.get(iclass, 50.0)


def is_established(iclass: str) -> bool:
    """Whether the SOURCE STUDY established this class. Reference prior, not our verdict."""
    return iclass in ESTABLISHED_CLASSES


def cost_ok(cost_bps: float, study=None) -> bool:
    """Is this round-trip cost survivable on this universe? A question about COST, not edge.

    With a measured :class:`edge.EdgeStudy`, the study knows what trading this universe
    actually costs in the units the edge is measured in: ``cost_bps/1e4 / sigma_h``, averaged
    over the instruments that fired. The gate asks whether that charge is smaller than
    :data:`LARGEST_KNOWN_EFFECT` — the most this signal has ever been worth on any asset
    class. If the cost exceeds that ceiling, no plausible version of the edge survives it.

    It deliberately does NOT compare the cost against the *measured* edge. Doing so would fail
    the gate on any universe that measures no edge, halving its conviction — which would make
    the measurement a hidden multiplier on the signal, the exact thing this design refuses to
    do. Expectancy is reported; only cost gates conviction.

    Without a study there is no per-universe cost charge, so it falls back to the source
    study's pooled breakeven in bps. :func:`cost_basis` reports which basis was used.
    """
    try:
        c = float(cost_bps)
    except (TypeError, ValueError):
        return False
    charge = _measured_cost_charge(study)
    if charge is not None:
        return bool(charge < LARGEST_KNOWN_EFFECT)
    return c <= POOLED_BREAKEVEN_BPS


def _measured_cost_charge(study) -> float | None:
    """This universe's measured trading cost in vol units, or None if not measured."""
    if study is None:
        return None
    r = study.get("buy", "holdout") or study.get("buy", "full")
    if r is None:
        return None
    ch = getattr(r, "cost_charge", float("nan"))
    return float(ch) if np.isfinite(ch) else None


def cost_basis(study=None) -> str:
    """'measured' when this universe's own cost charge backs the gate, else the pooled prior."""
    if _measured_cost_charge(study) is not None:
        return "measured"
    return f"pooled prior (~{POOLED_BREAKEVEN_BPS:.0f}bp)"


def cost_in_vol_units(cost_bps: float, sigma_h: float) -> float:
    """Convert a round-trip cost in bps to the vol units the edge is reported in.

    ``sigma_h`` is the h-bar forward-return sigma of the instrument (or the universe
    median). This conversion is why the edge dies on low-volatility instruments: 3bp against
    a 4% 10-day sigma costs 0.008 vol units, but against a 1% sigma it costs 0.030 — a real
    drag on an edge of ~0.05. A per-class cost table cannot express that; this can.
    """
    try:
        s = float(sigma_h)
        return float((float(cost_bps) / 1e4) / s) if s > 0 else float("nan")
    except (TypeError, ValueError, ZeroDivisionError):
        return float("nan")


def z_look_for(timeframe: str) -> int:
    """Z-score lookback for a Sanket timeframe (Daily 252 bars / Weekly 52 bars)."""
    return CLR_Z_LOOK_WEEKLY if str(timeframe) == "Weekly" else CLR_Z_LOOK_DAILY


def min_bars_for(z_look: int) -> int:
    """Bars a symbol needs before it can carry a signal (the Pine's ``zLook + 2`` warmup)."""
    return int(z_look) + 2


# ════════════════════════════════════════════════════════════════════════════════════════
# PER-SYMBOL FEATURES  (time-series; run once per name before cross-sectional ranking)
# ════════════════════════════════════════════════════════════════════════════════════════
def add_clr_features(df: pd.DataFrame,
                    z_look: int = CLR_Z_LOOK_DAILY,
                    thr: float = CLR_THRESHOLD,
                    horizon: int = CLR_HORIZON) -> pd.DataFrame:
    """Attach the CLR close-location signal to one symbol's OHLC frame.

    Columns written:
      ``CLR_CLV``       close location in [-1, +1] (Pine ``f_clv``)
      ``CLR_Z``         z-score of CLR_CLV over ``z_look`` bars; NaN until warm
      ``Fade_Score``   ``-CLR_Z`` — positive = bullish. The sign flip IS the finding.
      ``buy_cond``     green triangle: ``CLR_Z < -thr`` (weak close → fade long)
      ``sell_cond``    yellow diamond: ``CLR_Z > +thr`` (strong close → sell)
      ``CLR_Hold_Dir``  +1 inside a buy window, -1 inside a sell window, 0 outside
      ``CLR_Hold_Age``  bars since that window opened (0 = fired on this bar)
      ``CLR_State``     WARMING UP / BUY / SELL / NEUTRAL for the bar

    ``ta.stdev`` in Pine is the population standard deviation, so ``ddof=0`` here — using
    the sample stdev would shift every z by ~0.2% and drift the fire rate off the measured
    9.3%. Safe on short frames: CLR_Z is NaN until ``z_look`` bars exist and nothing fires.
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

    df['CLR_CLV']     = clv
    df['CLR_Z']       = z
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

    df['CLR_Hold_Dir'] = np.where(in_window, held_dir.fillna(0.0).to_numpy(dtype=float), 0.0).astype(int)
    df['CLR_Hold_Age'] = np.where(in_window, age, np.nan)

    df['CLR_State'] = np.select(
        [~np.isfinite(z.to_numpy(dtype=float)), buy_cond, sell_cond],
        ['WARMING UP', 'BUY', 'SELL'],
        default='NEUTRAL',
    )
    return df


# ════════════════════════════════════════════════════════════════════════════════════════
# CROSS-SECTIONAL RANKING  (one date's universe, ordered by the fade score)
# ════════════════════════════════════════════════════════════════════════════════════════
def compute_ranking(df: pd.DataFrame,
                    cost_bps: float = CLR_COST_BPS,
                    thr: float = CLR_THRESHOLD,
                    study=None) -> pd.DataFrame:
    """Rank one date's cross-section by the CLR fade score.

    df: one row per symbol carrying ``CLR_Z`` (and optionally ``Fade_Score``).
    cost_bps / study: the cost gate (see :func:`cost_ok`). ``study`` is an optional
    :class:`edge.EdgeStudy` measured on this universe; when present the gate is answered
    from its measured net edge rather than the pooled prior.

    Conviction is |z| magnitude x the cost gate, and nothing else. Note what is
    deliberately absent: no per-class expectancy lookup (that was a hardcoded table and is
    now measured separately by ``edge.py``, for reporting), no per-name vol factor, no
    regime factor, no live-IC scaling. The measured expectancy is REPORTED, never applied —
    a universe that measures no edge still fires at full conviction, and says so.

    Adds the output contract and returns the frame sorted by ``Priority_Long`` desc
    (warming-up rows, whose score is NaN, sort last). Pure & deterministic.
    """
    df = df.copy()
    contract = ('CLR_Score', 'CLR_Rank_Pct', 'Fade_Score', 'Conviction', 'Side',
                'Priority_Long', 'Priority_Short', 'Priority_Long_pct', 'Priority_Short_pct',
                'Signal_Reason')
    if len(df) == 0:
        for c in contract:
            df[c] = pd.Series(dtype=float)
        return df

    thr = float(thr)
    z = df['CLR_Z'].astype(float) if 'CLR_Z' in df.columns else pd.Series(np.nan, index=df.index)

    # ── 1. Score = fade score = -z. Positive = bullish (a weak close is the buy) ──
    fade = -z
    df['CLR_Score']   = fade
    df['Fade_Score'] = fade

    # ── 2. Cross-sectional rank percentile [0,100] (NaN where still warming up) ──
    rank_pct = fade.rank(pct=True) if len(df) >= 2 else pd.Series(0.5, index=df.index)
    df['CLR_Rank_Pct'] = (rank_pct * 100).round(2)

    # ── 3. Side: only a fired event is actionable; everything else is context ──
    #    Green triangle → Buy, yellow diamond → Sell. Sub-threshold rows are '—': the
    #    measured edge is in the EVENT, not in the continuous score.
    df['Side'] = np.where(z < -thr, 'Buy', np.where(z > thr, 'Sell', '—'))

    # ── 4. Conviction [0,1] = |z| magnitude × cost gate ──
    # Magnitude: |z|/3 — the threshold (1.5σ) lands at 0.50 and a 3σ close-location extreme
    # at 1.00. The cost gate halves conviction when the assumed round-trip cost sinks the
    # event form, because past that point the strategy is net negative no matter how
    # extreme the close was. There is no expectancy term: that is measured per universe by
    # `edge.py` and reported, not folded silently into a number the reader cannot audit.
    mag = (z.abs() / 3.0).clip(0.0, 1.0)
    base = 0.30 + 0.70 * mag
    cost_f = 1.0 if cost_ok(cost_bps, study) else 0.5
    df['Conviction'] = (base * cost_f).clip(0.0, 1.0).fillna(0.0)

    # ── 5. UI contract mapping ──
    scale = 100.0
    df['Priority_Long']      = fade * scale
    df['Priority_Short']     = -fade * scale
    df['Priority_Long_pct']  = rank_pct * 100
    df['Priority_Short_pct'] = (1 - rank_pct) * 100

    # A per-row note on the measured state of THIS universe, when a study exists. Never a
    # class label — that would be the hardcoded claim this design removed.
    def _verdict_note(side_key: str) -> str:
        if study is None:
            return " · expectancy not yet measured on this universe"
        lbl, _kind, _detail = study.verdict(side_key)
        return f" · measured on this universe: {lbl}"

    _buy_note, _sell_note = _verdict_note('buy'), _verdict_note('sell')
    df['Signal_Reason'] = [
        ("warming up — needs a full z-score lookback" if not np.isfinite(zz) else
         f"BUY · weak close z {zz:+.2f} · fade up, hold 5-10d, enter next open{_buy_note}"
         if sd == 'Buy' else
         f"SELL · strong close z {zz:+.2f}{_sell_note}"
         if sd == 'Sell' else
         f"context only · z {zz:+.2f} inside ±{thr:.1f}σ")
        for zz, sd in zip(z, df['Side'])
    ]

    return df.sort_values('Priority_Long', ascending=False, kind='stable',
                          na_position='last')
