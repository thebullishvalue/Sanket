"""
Sanket Signal Engine — SIDDHI CONVICTION OSCILLATOR.

Ported from ``siddhi.pine`` (Siddhi · Conviction Oscillator v3). It replaces the
close-location reversal (CLR) engine that shipped through v6.x: that engine measured
*where a bar closed inside its own range* and faded it. This one measures *how much of
the market's effort converted into displacement*, and trades the turn in that quantity.

What this is
------------
A bounded oscillator and its own signal line. The screening condition is the sign of the
gap between them.

    conviction     c = (close - close[1]) / TR                bounded -1 … +1
    participation  w = min(volume / EMA(volume, vn), cap)     range fallback where no volume
    raw            = 100 · SMA(c·w, len) / SMA(|c|·w, len)
    scaled         = 100 · tanh( raw / 3σ(raw, norm) )        adaptive self-normalisation
    osc            = EMA(scaled, smooth)
    sig            = EMA(osc, signal)
    hist           = osc - sig

``conv`` is signed by direction and normalised by true range, so gaps count and a wide
violent bar that closes where it opened scores zero — nothing was accomplished. ``w``
weights each bar by how much of the market showed up for it, capped so one expiry print
cannot own the window. The ratio is therefore *the participation-weighted share of effort
that went somewhere*, and the adaptive rescaling exists because that raw share cannot
reach its own bounds (it lives inside roughly ±15 intraday), which makes fixed thresholds
against ±100 thresholds that never fire.

Signals (the two events the screener fires)
-------------------------------------------
* **BUY — green triangle** (``buy_cond``): ``hist`` crosses **above** zero. The
  oscillator has pulled above its own signal line: conviction is turning up.
* **SELL — yellow diamond** (``sell_cond``): ``hist`` crosses **below** zero. The
  oscillator has dropped under its signal line: conviction is turning down.

That is the entire screening condition. It is a *state change*, not a level: nothing
fires while the histogram merely sits on one side of zero, and both events are symmetric
— unlike the CLR engine this replaces, where the two sides meant different things and
only one of them survived its own holdout.

The magnitude gate, and why it defaults to off
----------------------------------------------
:data:`SID_K` scales an optional magnitude requirement — the histogram must cross
``± k·σ(hist)`` rather than ``± 0``. **The default is 0.0, which is exactly the plain
zero-crossing described above.** The knob exists because the source indicator's own
measurement is that a bare crossover is its weakest configuration (it fires ~113 times
per 1000 bars for +0.0205R on the instruments it was fitted to and +0.0015R, t=0.2, on
held-out ones). Raising ``k`` trades signal count for separation. The plumbing carries it
so the parameter can be measured by ``edge.py`` on a real universe rather than argued
about; nothing in the shipped default path uses a non-zero value.

Why the event form (and not a continuous position)
--------------------------------------------------
Same reason as every version of this app: a continuous position on an oscillator turns
over every time the two lines touch, and near zero they touch constantly. Firing on the
crossing and holding :data:`SID_HORIZON` bars is what makes the rule costable at all.
``edge.py`` measures whether that survives on the universe actually on screen.

What is NOT claimed
-------------------
The source indicator publishes its own measurements and they are modest and honest:
nothing in it reaches statistical significance once overlapping forward windows are
accounted for (best case t = 1.9 across 48 horizon/bracket cells), the correlation between
a configuration's fitted edge and its edge on unseen instruments is approximately zero,
and on 5-minute bars the round-trip cost is roughly double anything the indicator has been
shown to produce. Read every published number as a ranking, not a promise — and read the
number ``edge.py`` measures on the user's own universe as the one that applies here.

Horizon
-------
Bar-scale, following the lookback: a 20-bar window is a swing instrument on daily bars.
Entry is the next session's open after the signal bar closes. There is no intraday claim.

Bar convention (one deliberate difference from the Pine)
-------------------------------------------------------
The Pine gates every discrete object on ``barstate.isconfirmed`` so nothing is drawn on a
forming bar and then withdrawn. Sanket evaluates completed daily (or weekly) bars
directly, so that gate is structural here rather than explicit: a signal fires on the bar
whose close produced it, and entry is the next session's open. The one carry-over: a
signal on a session that has not closed yet is provisional until it does.

Numerical fidelity to the Pine
------------------------------
``ta.ema`` seeds on its first value and uses ``alpha = 2/(n+1)`` — that is
``ewm(span=n, adjust=False)``. ``ta.stdev`` is the POPULATION standard deviation, so
``ddof=0`` throughout. ``ta.tr(true)`` includes the gap and falls back to ``high-low`` on
the first bar. The hollow-bar volume carry (``volLast``) is reproduced: a holiday or thin
overnight print must not kill the participation baseline for a whole averaging window.

Output column contract (``compute_ranking``)
--------------------------------------------
  SID_Score, SID_Rank_Pct, Signal_Score, Conviction, Side,
  Priority_Long, Priority_Short, Priority_Long_pct, Priority_Short_pct,
  Signal_Reason
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── The source indicator's inputs, as shipped defaults ───────────────────────────────────
# Every value below is siddhi.pine's own default. They are NOT fitted here and must not be
# tuned to a backtest: the source's 900-configuration search found the correlation between
# a setting's fitted edge and its edge on unseen instruments to be approximately zero
# (-0.07 to +0.11). The only two preferences that held across every architecture tested
# were a lookback at or above 20 and an inner zone at or below 40 — both respected here.

# Lookback: how many bars of effort the oscillator accounts for. This is the horizon of the
# whole instrument, and the same bar count on either timeframe — 20 daily bars is a swing
# read, 20 weekly bars is a positional one.
SID_LENGTH = 20

# Final EMA on the oscillator. Its real job is to keep the series from being jagged; 1
# disables it, above ~5 the turns arrive materially late.
SID_SMOOTH = 3

# The signal line the histogram is measured against.
SID_SIGNAL = 9

# Sample used for the adaptive scaling σ and for the (default-off) magnitude threshold.
# Longer is stabler and slower to acknowledge that the instrument has changed character.
SID_NORM = 200

# Averaging length for the volume (or true-range) participation baseline.
SID_VOL_N = 20

# Ceiling on the participation weight. Expiry days, index rebalances and block prints
# generate volume many multiples of normal; uncapped, one print would dominate the whole
# window for its full length and the oscillator would be reporting that single bar.
SID_CAP = 3.0

# What weights each bar's conviction. "Auto" uses volume where the symbol has it and
# relative true range where it does not, so index spot works without configuration.
SID_PARTICIPATION = "Auto"
PARTICIPATION_MODES = ("Auto", "Volume", "True range", "Off")

# "Adaptive (self-normalized)" maps the raw share through 100·tanh(raw / 3σ). "Raw share"
# plots the untransformed quantity, which is honest about absolute conviction and nearly
# useless for thresholds.
SID_SCALING = "Adaptive (self-normalized)"
SCALING_MODES = ("Adaptive (self-normalized)", "Raw share")

# Magnitude gate in σ of the histogram's own distribution. 0.0 == the plain zero-crossing,
# which is the shipped screening condition. See the module docstring.
SID_K = 0.0

# Hold horizon in bars. Entry next open, exit `horizon` bars later.
SID_HORIZON = 10

# Zones, in the oscillator's own scaled units. Under adaptive scaling the inner zone is
# occupied roughly a third of the time and the outer about 4% (measured 3.2-4.3%). These
# are DISPLAY context — they classify how one-sided the window is. Nothing gates on them.
SID_ZONE_INNER = 30.0
SID_ZONE_OUTER = 60.0

# Degenerate-window guard. When an instrument's raw conviction share stops varying — an
# illiquid or range-collapsed stretch — the adaptive σ collapses and the tanh rescaling
# amplifies arithmetic noise into a full-scale reading. The Pine handles this by emitting
# zero; here the bar is marked DEGENERATE and suppressed so it cannot fire or rank, which
# is the same posture the previous engine took toward a collapsed CLV sigma.
SID_MIN_RAW_SIGMA = 1e-6

# Round-trip cost assumption for the cost gate.
SID_COST_BPS = 3.0

# Forward horizons the Historical Range harvest attaches as Ret_*b labels.
HOLD_HORIZONS = [1, 5, 10, 21]


# ════════════════════════════════════════════════════════════════════════════════════════
# INSTRUMENT CLASS  (wired to Sanket's universe)
# ════════════════════════════════════════════════════════════════════════════════════════
# ⚠ REFERENCE PRIOR ONLY — NOT USED TO COMPUTE ANYTHING.
#
# These are the SOURCE INDICATOR's published per-class results, converted from its R units
# (2 ATR target / 1 ATR stop, 20 bars max, entry next open, measured against a matched
# random-entry baseline over the same bracket) into the vol units this app reports in.
# `established` is false everywhere, because the source's own headline is that nothing it
# measured reaches significance once overlapping forward windows are accounted for.
#
# They were never wired into conviction and must not be: frozen constants from someone
# else's eleven instruments cannot cover a universe the study never touched. Expectancy is
# MEASURED per universe by `edge.py`, from the user's own symbols, at these same
# pre-declared parameters. What remains here is a labelled comparison line.
INSTRUMENT_CLASSES = [
    "US index / ETF", "US sector ETF", "India index", "International equity",
    "Commodity", "FX", "Rates / Credit", "Other / unknown",
]

# Source-indicator R-edge by the nearest instrument group it actually tested. Gold, silver
# and crude were the primary futures it was calibrated on (+0.036R on the continuation
# architecture); the eight held-out instruments — ES, SPY, QQQ, TLT, 6E, HG, NG, BTC — came
# in at -0.016R, i.e. the sign flips off the primaries. Both are reported at face value.
CLASS_EDGE = {
    "US index / ETF":       -0.016,
    "US sector ETF":        -0.016,
    "India index":           0.000,   # never tested by the source
    "International equity":  0.000,   # never tested by the source
    "Commodity":             0.036,
    "FX":                   -0.016,
    "Rates / Credit":       -0.016,
    "Other / unknown":       0.000,
}

CLASS_HIT = {
    "US index / ETF":       50.0,
    "US sector ETF":        50.0,
    "India index":          50.0,
    "International equity": 50.0,
    "Commodity":            52.0,
    "FX":                   50.0,
    "Rates / Credit":       50.0,
    "Other / unknown":      50.0,
}

# The source establishes NO class: its best t-statistic across 48 horizon/bracket cells was
# 1.9, and it says so itself. Kept as an explicit empty tuple rather than deleted so the
# reference row can still say "not established by the source" instead of saying nothing.
ESTABLISHED_CLASSES: tuple[str, ...] = ()

# Cost breakeven fallback, in bps, before a study has measured this universe's actual cost.
# The source quotes ~0.02R of round-trip cost on daily bars at one ATR of stop against a
# best-case measured edge near 0.05R; 7bp is the pooled equity-universe analogue this app
# has always used and remains the conservative fallback.
POOLED_BREAKEVEN_BPS = 7.0

# The largest edge the source indicator found on any instrument group (+0.036R, the primary
# futures on the continuation architecture). Used as a CEILING, not a forecast: it is the
# most this construction has ever been worth anywhere, so a trading cost exceeding it
# cannot be survived by any plausible version of the edge. That makes it the right yardstick
# for a cost gate — and, in `edge.py`, for deciding when a test is too underpowered to say
# anything.
LARGEST_KNOWN_EFFECT = 0.036

# Sanket universe → instrument class. Used ONLY to select which reference row to display.
UNIVERSE_CLASS_MAP = {
    "US Indexes":     "US index / ETF",
    "India Indexes":  "India index",
    "Global Indexes": "International equity",
    "ETF Index":      "India index",        # NSE ETFs track Indian indices / sectors
    "Commodities":    "Commodity",
    "Currency":       "FX",
    "Global Macro":   "Rates / Credit",
    "Crypto":         "Other / unknown",    # the source covered BTC only, on one architecture
}


def instrument_class(universe: str, selected_index: str | None = None) -> str:
    """Which reference row to show beside the measured result. Display only.

    ``selected_index`` is accepted so a future sub-universe split can refine the label
    without changing call sites.
    """
    return UNIVERSE_CLASS_MAP.get(universe, "Other / unknown")


def class_edge(iclass: str) -> float:
    """The SOURCE INDICATOR's published edge for a class — a reference prior, not ours."""
    return CLASS_EDGE.get(iclass, 0.0)


def class_hit(iclass: str) -> float:
    """The SOURCE INDICATOR's published hit rate for a class — a reference prior, not ours."""
    return CLASS_HIT.get(iclass, 50.0)


def is_established(iclass: str) -> bool:
    """Whether the SOURCE established this class. It established none; see the constant."""
    return iclass in ESTABLISHED_CLASSES


def cost_ok(cost_bps: float, study=None) -> bool:
    """Is this round-trip cost survivable on this universe? A question about COST, not edge.

    With a measured :class:`edge.EdgeStudy`, the study knows what trading this universe
    actually costs in the units the edge is measured in: ``cost_bps/1e4 / sigma_h``, averaged
    over the instruments that fired. The gate asks whether that charge is smaller than
    :data:`LARGEST_KNOWN_EFFECT` — the most this signal has ever been worth anywhere. If the
    cost exceeds that ceiling, no plausible version of the edge survives it.

    It deliberately does NOT compare the cost against the *measured* edge. Doing so would fail
    the gate on any universe that measures no edge, halving its conviction — which would make
    the measurement a hidden multiplier on the signal, the exact thing this design refuses to
    do. Expectancy is reported; only cost gates conviction.

    Without a study there is no per-universe cost charge, so it falls back to the pooled
    breakeven in bps. :func:`cost_basis` reports which basis was used.
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
    drag on an edge of ~0.03. A per-class cost table cannot express that; this can.
    """
    try:
        s = float(sigma_h)
        return float((float(cost_bps) / 1e4) / s) if s > 0 else float("nan")
    except (TypeError, ValueError, ZeroDivisionError):
        return float("nan")


def length_for(timeframe: str) -> int:
    """Oscillator lookback for a Sanket timeframe.

    The same bar count on both: the lookback IS the horizon, so 20 weekly bars is the
    positional read and 20 daily bars the swing one. Kept as a function because every call
    site passes a timeframe and a future split should not have to change them.
    """
    return SID_LENGTH


def warmup_bars(length: int = SID_LENGTH, norm: int = SID_NORM,
                vol_n: int = SID_VOL_N, smooth: int = SID_SMOOTH) -> int:
    """Bars a symbol needs before it can carry a signal (the Pine's ``ready`` gate).

    The dependency is ADDITIVE, not a maximum, and that is deliberate: ``rawSd`` needs
    ``norm`` bars of valid ``raw``, ``raw`` is itself pinned to 0 for its first ``length``
    bars while its SMA warms, the participation baseline needs ``vol_n``, and the final EMA
    needs ``smooth``. Taking a maximum would let the first σ be computed across zero-filled
    bars, biasing it low and so inflating the adaptive scaling exactly where the series
    begins.
    """
    return int(norm) + int(length) + int(vol_n) + int(smooth) + 2


# ════════════════════════════════════════════════════════════════════════════════════════
# THE OSCILLATOR  (pure numeric core — no DataFrame, so it is trivially testable)
# ════════════════════════════════════════════════════════════════════════════════════════
def _ema(s: pd.Series, span: int) -> pd.Series:
    """Pine ``ta.ema``: alpha = 2/(span+1), seeded on the first value."""
    return s.ewm(span=max(int(span), 1), adjust=False).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Pine ``ta.tr(true)`` — gap-inclusive, falling back to ``high-low`` on the first bar."""
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev).abs(),
                    (low - prev).abs()], axis=1).max(axis=1)
    return tr.where(prev.notna(), (high - low).abs())


def siddhi_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                      volume: pd.Series | None = None,
                      length: int = SID_LENGTH,
                      smooth: int = SID_SMOOTH,
                      signal: int = SID_SIGNAL,
                      norm: int = SID_NORM,
                      vol_n: int = SID_VOL_N,
                      cap: float = SID_CAP,
                      participation: str = SID_PARTICIPATION,
                      scaling: str = SID_SCALING) -> dict[str, pd.Series]:
    """The Siddhi conviction oscillator, its signal line and their histogram.

    Returns a dict of aligned Series: ``raw``, ``osc``, ``sig``, ``hist``, ``hist_sd``,
    ``conv``, ``w``. A direct transcription of ``siddhi.pine`` sections 1-3; see the module
    docstring for the numerical-fidelity notes that make it a transcription rather than an
    approximation.
    """
    length = max(int(length), 3)
    norm   = max(int(norm), 30)
    vol_n  = max(int(vol_n), 5)

    # ── 1 · CONVICTION AND PARTICIPATION ──
    tr = true_range(high, low, close)
    disp = close - close.shift(1).fillna(close)
    conv = (disp / tr.where(tr > 1e-12)).fillna(0.0)      # bounded -1 … +1

    if volume is None:
        vol = pd.Series(np.nan, index=close.index, dtype=float)
    else:
        vol = pd.to_numeric(volume, errors="coerce").astype(float)
    vol_ok = vol.notna() & (vol > 0)

    # A hollow bar (holiday, half session, thin overnight print) would otherwise feed NaN
    # into this EMA and kill the participation baseline for a whole averaging window,
    # silently dropping the series back to range weighting. Carrying the last good print
    # keeps it alive; a genuinely volume-less symbol still leaves the baseline NaN and
    # still falls through to true range, as documented.
    vol_last = vol.where(vol_ok).ffill()
    vol_avg  = _ema(vol_last, vol_n)
    tr_avg   = _ema(tr, vol_n)

    w_vol = (vol / vol_avg.where(vol_avg > 1e-12)).where(vol_ok)
    w_rng = (tr / tr_avg.where(tr_avg > 1e-12)).fillna(1.0)

    mode = str(participation)
    if mode == "Off":
        w_raw = pd.Series(1.0, index=close.index)
    elif mode == "True range":
        w_raw = w_rng
    elif mode == "Volume":
        w_raw = w_vol.fillna(1.0)
    else:                                                  # "Auto"
        w_raw = w_vol.fillna(w_rng)
    w = w_raw.fillna(1.0).clip(lower=0.0, upper=float(cap))

    # ── 2 · THE OSCILLATOR ──
    # A ratio of SUMS, so a dead bar contributes nothing to either side rather than an
    # undefined ratio. Then scaled against its own dispersion, because the raw ratio cannot
    # reach its own bounds and a threshold against a range the series never visits is a
    # threshold that never fires.
    num = (conv * w).rolling(length).mean()
    den = (conv.abs() * w).rolling(length).mean()
    raw = (100.0 * num / den.where(den > 1e-12))

    raw_sd = raw.rolling(norm).std(ddof=0)
    if str(scaling) == "Raw share":
        scaled = raw
    else:
        ok = raw_sd.notna() & (raw_sd >= SID_MIN_RAW_SIGMA)
        # tanh is clamped at ±10 in the Pine to keep exp() finite; np.tanh saturates
        # gracefully on its own, so the clamp is only kept for exact parity of intent.
        scaled = pd.Series(
            np.where(ok, 100.0 * np.tanh(np.clip(raw / (3.0 * raw_sd), -10.0, 10.0)), np.nan),
            index=close.index, dtype=float)
        # Bars whose raw is warm but whose σ is not yet available stay NaN rather than
        # collapsing to 0 — a fabricated zero would fire a crossing that never happened.
        scaled = scaled.where(raw.notna())

    osc = _ema(scaled, smooth) if int(smooth) > 1 else scaled
    # `_ema` propagates the leading NaNs of `scaled` rather than seeding on them, so the
    # oscillator stays undefined until the whole chain is warm.
    sig = _ema(osc, signal)
    hist = osc - sig
    hist_sd = hist.rolling(norm).std(ddof=0)

    return {"raw": raw, "osc": osc, "sig": sig, "hist": hist,
            "hist_sd": hist_sd, "conv": conv, "w": w}


# ════════════════════════════════════════════════════════════════════════════════════════
# PER-SYMBOL FEATURES  (time-series; run once per name before cross-sectional ranking)
# ════════════════════════════════════════════════════════════════════════════════════════
def add_siddhi_features(df: pd.DataFrame,
                        length: int = SID_LENGTH,
                        smooth: int = SID_SMOOTH,
                        signal: int = SID_SIGNAL,
                        norm: int = SID_NORM,
                        vol_n: int = SID_VOL_N,
                        cap: float = SID_CAP,
                        participation: str = SID_PARTICIPATION,
                        scaling: str = SID_SCALING,
                        k: float = SID_K,
                        horizon: int = SID_HORIZON) -> pd.DataFrame:
    """Attach the Siddhi conviction signal to one symbol's OHLCV frame.

    Columns written:
      ``SID_Raw``       raw participation-weighted share of effort, 100·Σcw/Σ|c|w
      ``SID_Osc``       the scaled, smoothed oscillator (adaptive: bounded ±100)
      ``SID_Sig``       its signal-line EMA
      ``SID_Hist``      ``SID_Osc - SID_Sig`` — the histogram. THIS is the screening variable.
      ``SID_Hist_Sd``   σ of the histogram over the normalization window
      ``SID_Hist_Z``    ``SID_Hist / SID_Hist_Sd`` — the histogram in its own σ units, which
                        is what makes it comparable ACROSS symbols and therefore rankable
      ``SID_Impulse``   ``(hist - hist[1]) / SID_Hist_Sd`` — the size of the crossing step
      ``SID_Thr``       the magnitude gate, ``k · SID_Hist_Sd`` (0 at the shipped default)
      ``Signal_Score``  ``SID_Hist_Z`` — positive = bullish. The score the universe ranks on.
      ``buy_cond``      green triangle: histogram crossed UP through the gate
      ``sell_cond``     yellow diamond: histogram crossed DOWN through the gate
      ``SID_Hold_Dir``  +1 inside a buy window, -1 inside a sell window, 0 outside
      ``SID_Hold_Age``  bars since that window opened (0 = fired on this bar)
      ``SID_Zone``      where the oscillator sits: Extreme Bull / Bull / Neutral / Bear /
                        Extreme Bear, against the inner and outer zones. Context, not a gate.
      ``SID_State``     WARMING UP / DEGENERATE / BUY / SELL / NEUTRAL for the bar

    A crossing is ``ta.crossover(hist, thr)`` — ``hist > thr and hist[1] <= thr[1]`` — so at
    the default ``k = 0`` it is precisely "the histogram crossed zero". Nothing fires before
    :func:`warmup_bars`; a frame shorter than that produces no events at all rather than
    events computed from a half-warm oscillator.
    """
    df = df.copy()
    high, low, close = df['High'], df['Low'], df['Close']
    vol = df['Volume'] if 'Volume' in df.columns else None

    o = siddhi_oscillator(high, low, close, vol,
                          length=length, smooth=smooth, signal=signal, norm=norm,
                          vol_n=vol_n, cap=cap, participation=participation, scaling=scaling)

    hist, hist_sd = o['hist'], o['hist_sd']
    thr = float(k) * hist_sd.fillna(0.0)

    # In σ units, so two instruments on different scales are directly comparable. This is
    # the cross-sectional score; the raw histogram is not comparable across symbols.
    hist_z  = hist / hist_sd.where(hist_sd > 1e-12)
    impulse = hist.diff() / hist_sd.where(hist_sd > 1e-12)

    df['SID_Raw']     = o['raw']
    df['SID_Osc']     = o['osc']
    df['SID_Sig']     = o['sig']
    df['SID_Hist']    = hist
    df['SID_Hist_Sd'] = hist_sd
    df['SID_Hist_Z']  = hist_z
    df['SID_Impulse'] = impulse
    df['SID_Thr']     = thr
    df['Signal_Score'] = hist_z

    # ── Warmup and degeneracy ──
    n = len(df)
    pos = np.arange(n)
    warm = pos >= warmup_bars(length, norm, vol_n, smooth)
    # A collapsed histogram σ means the two lines have stopped separating at all; the
    # z-score and the impulse are then divisions by ~0 and describe arithmetic, not the
    # market. Those bars must not fire and must not rank.
    degenerate = hist_sd.notna() & (hist_sd <= 1e-12)
    valid = warm & hist.notna() & hist.shift(1).notna() & ~degenerate.to_numpy(dtype=bool)

    # ── The two plotted events: Pine ta.crossover / ta.crossunder against ±thr ──
    buy_cond  = ((hist > thr) & (hist.shift(1) <= thr.shift(1))).fillna(False).to_numpy(dtype=bool) & valid
    sell_cond = ((hist < -thr) & (hist.shift(1) >= -thr.shift(1))).fillna(False).to_numpy(dtype=bool) & valid
    df['buy_cond']  = buy_cond
    df['sell_cond'] = sell_cond

    # ── Hold window, vectorised ──
    # A fire opens a window in its direction; a later fire re-opens it. Outside `horizon`
    # bars the window has expired.
    fpos = pos.astype(float)
    fires = buy_cond | sell_cond
    fire_dir = np.where(buy_cond, 1.0, np.where(sell_cond, -1.0, np.nan))

    last_fire = pd.Series(np.where(fires, fpos, np.nan), index=df.index).ffill()
    held_dir  = pd.Series(fire_dir, index=df.index).ffill()
    age       = fpos - last_fire.to_numpy(dtype=float)
    in_window = np.isfinite(age) & (age <= int(horizon))

    df['SID_Hold_Dir'] = np.where(in_window, held_dir.fillna(0.0).to_numpy(dtype=float), 0.0).astype(int)
    df['SID_Hold_Age'] = np.where(in_window, age, np.nan)

    # ── Zone: how one-sided the window's effort is. Display context only. ──
    oscv = o['osc']
    df['SID_Zone'] = np.select(
        [oscv >=  SID_ZONE_OUTER, oscv >=  SID_ZONE_INNER,
         oscv <= -SID_ZONE_OUTER, oscv <= -SID_ZONE_INNER],
        ['Extreme Bull', 'Bull', 'Extreme Bear', 'Bear'],
        default='Neutral',
    )
    df.loc[oscv.isna(), 'SID_Zone'] = '—'

    df['SID_State'] = np.select(
        [degenerate.to_numpy(dtype=bool) & warm,
         ~valid,
         buy_cond, sell_cond],
        ['DEGENERATE', 'WARMING UP', 'BUY', 'SELL'],
        default='NEUTRAL',
    )
    return df


# ════════════════════════════════════════════════════════════════════════════════════════
# CROSS-SECTIONAL RANKING  (one date's universe, ordered by the conviction histogram)
# ════════════════════════════════════════════════════════════════════════════════════════
def compute_ranking(df: pd.DataFrame,
                    cost_bps: float = SID_COST_BPS,
                    k: float = SID_K,
                    horizon: int = SID_HORIZON,
                    study=None) -> pd.DataFrame:
    """Rank one date's cross-section by the conviction histogram.

    df: one row per symbol carrying ``SID_Hist_Z`` (and optionally ``SID_Impulse``,
    ``buy_cond`` / ``sell_cond``, ``SID_Hold_*``). cost_bps / study: the cost gate (see
    :func:`cost_ok`). ``horizon`` must match the one the per-symbol pass used, or the hold
    band decays against the wrong denominator.

    The score is ``SID_Hist_Z`` — the histogram in its own σ units, positive = the
    oscillator is above its signal line. Ranking on the *level* while firing on the
    *crossing* is deliberate: the level says who is currently in control, the crossing says
    when that changed, and the measured edge is claimed only for the crossing.

    Conviction is the CROSSING STEP, not the level
    ----------------------------------------------
    At the moment a histogram crosses zero it is, by construction, approximately zero — so
    scaling conviction off ``|hist|`` would score every fresh signal at zero and every stale
    one high, which is backwards. What actually distinguishes crossings is how forcefully
    the gap opened: ``SID_Impulse``, the one-bar change in the histogram measured in σ of
    its own distribution. A crossing that snaps open half a sigma in one bar is a different
    event from one that drifts across, and this is the only quantity available at fire time
    that separates them.

    It remains a DESCRIPTION. Nothing has established that a larger crossing step predicts a
    better outcome; it does not gate anything, and no measured expectancy enters it — that
    is ``edge.py``'s job, reported rather than applied. A universe that measures no edge
    still fires at full conviction, and says so.

    Adds the output contract and returns the frame sorted by ``Priority_Long`` desc
    (warming-up rows, whose score is NaN, sort last). Pure & deterministic.
    """
    df = df.copy()
    contract = ('SID_Score', 'SID_Rank_Pct', 'Signal_Score', 'Conviction', 'Side',
                'Priority_Long', 'Priority_Short', 'Priority_Long_pct', 'Priority_Short_pct',
                'Signal_Reason')
    if len(df) == 0:
        for c in contract:
            df[c] = pd.Series(dtype=float)
        return df

    k = float(k)
    idx = df.index

    def _col(name: str) -> pd.Series:
        return df[name].astype(float) if name in df.columns else pd.Series(np.nan, index=idx)

    hz  = _col('SID_Hist_Z')
    imp = _col('SID_Impulse')

    # ── 1. Score = the histogram in σ units. Positive = bullish. ──
    df['SID_Score']    = hz
    df['Signal_Score'] = hz

    # ── 2. Cross-sectional rank percentile [0,100] (NaN where still warming up) ──
    rank_pct = hz.rank(pct=True) if len(df) >= 2 else pd.Series(0.5, index=idx)
    df['SID_Rank_Pct'] = (rank_pct * 100).round(2)

    # ── 3. Side: only a fired CROSSING is actionable; a level is context ──
    #    The events come from the per-symbol pass, which is the only place that can see the
    #    previous bar. Falling back to the sign of the histogram would turn "who is in
    #    control" into a trade signal, which is exactly the claim this engine does not make.
    if 'buy_cond' in df.columns and 'sell_cond' in df.columns:
        buy  = df['buy_cond'].fillna(False).to_numpy(dtype=bool)
        sell = df['sell_cond'].fillna(False).to_numpy(dtype=bool)
    else:
        state = df['SID_State'].astype(str) if 'SID_State' in df.columns else pd.Series('', index=idx)
        buy  = (state == 'BUY').to_numpy(dtype=bool)
        sell = (state == 'SELL').to_numpy(dtype=bool)
    df['Side'] = np.where(buy, 'Buy', np.where(sell, 'Sell', '—'))

    # ── 4. Conviction [0,1] = crossing force × cost gate ──
    # tanh of |impulse| in σ units: bounded, smooth, monotone, and free of a fitted
    # constant — one sigma of the histogram's own distribution is the natural unit, so
    # a one-sigma step reads 0.76 and a two-sigma step 0.96. See the docstring above for
    # why the LEVEL cannot be used here.
    mag = pd.Series(np.tanh(imp.abs().clip(upper=10.0)), index=idx)
    base = 0.30 + 0.70 * mag
    cost_f = 1.0 if cost_ok(cost_bps, study) else 0.5
    df['Conviction'] = (base * cost_f).clip(0.0, 1.0).fillna(0.0)

    # ── 5. Priority: BANDED, because a crossing sits at zero ──
    #
    # The previous engine could sort the universe on its raw score, because there the
    # extreme readings WERE the fired signals — a buy was the most negative z on the board.
    # A zero-crossing is the opposite: at the instant it fires the histogram is, by
    # construction, ~0, so sorting the universe on the level would bury every fresh signal
    # in the middle of the list. The sort has to encode what the engine actually claims.
    #
    # Three bands, highest first:
    #   FIRED TODAY      2 + conviction         a crossing on this bar, strongest first
    #   IN HOLD WINDOW   1 + remaining fraction a crossing still inside its horizon
    #   CONTEXT          tanh(hist σ) ∈ (−1,1)  no crossing; just who is currently in control
    #
    # The bands cannot overlap, so an actionable row always outranks a merely bullish one —
    # which is the same statement the `Side` column and `Signal_Reason` already make.
    conv = df['Conviction'].astype(float).fillna(0.0)
    age  = _col('SID_Hold_Age')
    hdir = _col('SID_Hold_Dir').fillna(0.0)
    hrz  = float(max(int(horizon), 1))
    remaining = (1.0 - (age / hrz)).clip(0.0, 1.0).fillna(0.0)
    context = pd.Series(np.tanh(hz.fillna(0.0)), index=idx)

    def _priority(fired: np.ndarray, side_sign: float) -> pd.Series:
        held = (hdir == side_sign) & age.notna() & ~pd.Series(fired, index=idx)
        return pd.Series(
            np.where(fired, 2.0 + conv,
                     np.where(held, 1.0 + remaining, side_sign * context)),
            index=idx, dtype=float)

    # Warming-up rows carry no opinion at all and must sort last, not at the middle of the
    # context band — NaN plus na_position='last' is how the caller expects that expressed.
    warm = hz.notna()
    p_long  = _priority(buy, +1.0).where(warm)
    p_short = _priority(sell, -1.0).where(warm)

    scale = 100.0
    df['Priority_Long']      = p_long * scale
    df['Priority_Short']     = p_short * scale
    df['Priority_Long_pct']  = p_long.rank(pct=True) * 100
    df['Priority_Short_pct'] = p_short.rank(pct=True) * 100

    # A per-row note on the measured state of THIS universe, when a study exists. Never a
    # class label — that would be the hardcoded claim this design refuses to make.
    def _verdict_note(side_key: str) -> str:
        if study is None:
            return " · expectancy not yet measured on this universe"
        lbl, _kind, _detail = study.verdict(side_key)
        return f" · measured on this universe: {lbl}"

    _buy_note, _sell_note = _verdict_note('buy'), _verdict_note('sell')
    _gate = "zero" if k <= 0 else f"+{k:g}σ"
    _gate_dn = "zero" if k <= 0 else f"−{k:g}σ"
    df['Signal_Reason'] = [
        ("warming up — the oscillator has no normalization window yet" if not np.isfinite(zz) else
         f"BUY · histogram crossed above {_gate} · conviction turning up, hold the horizon, "
         f"enter next open{_buy_note}"
         if sd == 'Buy' else
         f"SELL · histogram crossed below {_gate_dn} · conviction turning down{_sell_note}"
         if sd == 'Sell' else
         f"context only · histogram {zz:+.2f}σ, no crossing on this bar")
        for zz, sd in zip(hz, df['Side'])
    ]

    return df.sort_values('Priority_Long', ascending=False, kind='stable',
                          na_position='last')
