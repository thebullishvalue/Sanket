"""
Sanket Signal Engine — PRAGATI · Conviction × Value.

The screener's logical stack, rebuilt on pragati.pine v9.2 and on the inference
Pragyam already carries for it. It replaces the Siddhi zero-crossing engine that
shipped through v7.x: Siddhi fired whenever conviction crossed its own signal
line — ~113 times per 1000 bars, its source's weakest tested configuration
(+0.0015R, t = 0.2, held out). Pragati keeps Siddhi's measurement at its root
(c = ΔC / TR, participation-weighted) and asks the other half of the question
on the same bar: progress, and the PRICE it was made at.

THE STACK, bottom to top
────────────────────────
    samanvaya.py   VALUE — rich or cheap against what the macro drivers
                   explain (Samanvaya's unified z), on the chart and its parent
    pragati.py     CONVICTION — who controls, how firmly, on the chart and its
                   ladder; the TRACE (conviction × value) and its HISTOGRAM;
                   divergence evidence; the ◆'s condition, and signals() —
                   the ▲▼ read from the grid, with the ◆ and the declaration
    cvgrid.py      THE GRID — the 3 × 3 state the two tapes place a name in,
                   named as an action with its measured units
    engine.py      THIS FILE — the Sanket-facing layer: settings, per-symbol
                   features on an OHLCV frame, the snapshot row, and the
                   cross-sectional ranking
    edge.py        the measured expectancy of the signal set on YOUR universe

THREE KINDS OF OUTPUT, AND WHAT EACH CLAIMS
───────────────────────────────────────────
    EVENTS    ▲ CAPITULATION — the grid's Buy · capitulation turning: sellers in
              control across the ladder at a price cheap past θ, and value
              momentum already reverting. ▼ DISTRIBUTION — sellers taking control
              of a price rich past θ. v9's events, read from the grid (the Pine's
              section 8b); the last one stands as the DECLARATION. ◆ RESUME is off
              by default.
    WATCH     a name in capitulation whose value is still cheapening — the ▲
              comes when it turns.
    STATE     the grid cell — Buy / Accumulate / Hold / Wait / Trim / Exit —
              where the name stands between events. Every name has one.

RANKING IS BY STRETCH, read as REVERSION — measured, not inherited:

    long side                              short side
    5 + s   ▲ on this bar                   5 + s'  ▼ on this bar
    s       everything else                 s'      everything else

s = −trace / 200 ∈ (−½, ½): the name stretched furthest DOWN (sellers in control,
priced cheap) leads the long side; s' = −s, so the most stretched UP leads the
short side. A ▲▼ on this bar — v9: a capitulation turning, or distribution —
stays on top of its side.

Why not the grid's weight (v8.0.0 ranked by it, banded TURN > RESUME > hold >
open window > grid state): trace_study.py measured it on five NSE universes over
~15 years and the ranking ran BACKWARDS — long-minus-short −0.022σ before 2021 and
−0.043σ after, clearly negative in 4 of 8 runs. The grid reads buyers-firm as
Buy / Add; over 5–40 bars those names LAGGED the cross-section and sellers-firm,
cheap names led — the stretch reverts, and every ingredient (conviction, value, the
trace, the push, the grid weight) carried the same negative sign. Designed on the
pre-2021 era only: stretch +0.058σ, TURN kept on top +0.059σ; RESUME (continuation)
and the hold / open-window bands, which pin stale names to the top, cost edge
(+0.056σ, +0.032σ) and no longer order the list. On the sealed 2021–2026 holdout the
stretch ranking beat the grid's on average (daily −0.007σ vs −0.043σ, weekly
+0.087σ vs −0.040σ). Stated plainly: once the name's own 20-bar return is removed,
the trace carries ~0 information — on NSE equities this ranking IS short-term
reversal, read through the indicator.

The grid, RESUME, the hold and the watchlist are still computed and shown — they
describe a name; they do not order the list. (That study measured v8's TURN on top;
v9's ▲ is the capitulation turn, measured separately in studies/pragati_v9_audit.md.)

WHAT IS NOT CLAIMED
───────────────────
The source indicator's own evidence section, stated as it states it: regular
divergence RANKED first (not established); participation weighting earns its
place; the scaling is calibrated. The chart-only reversal trigger had no edge
(+0.0003R); hidden divergence none; continuation changed sign off its primaries
(+0.036R / −0.016R). NOTHING reaches significance once overlapping windows are
accounted for (best t = 1.9 of 48 cells). The stack itself was measured by the
v8 and v9 audits (studies/): the readings carry almost no timing information of
their own; the capitulation state and its turn carry the one robust edge (+0.05 to
+0.08σ over 10-20 bars in each era outside crypto). So the number that applies to
YOUR screen is still the one edge.py measures on your symbols — reported, never
applied.

Bar convention: signals are committed on the close of the bar that produced
them (the Pine's barstate.isconfirmed); entry is the next session's open. A
signal on a session that has not closed yet is provisional until it does.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

import cvgrid as cg
import intraday as idm
import pragati as pg
import samanvaya as sv

# ── Declared parameters ──────────────────────────────────────────────────────────────────
# Every indicator input is pragati.pine's own default (pragati.Params). None is fitted here
# and none is exposed in the UI: the source's 900-configuration search found fitted and
# out-of-sample edge uncorrelated. Two numbers are Sanket's, and both are disclosed:

# WEEKLY NORMALIZATION — Sanket's number, not the source's. Warm-up costs two
# normalization windows; at 200 a weekly symbol needs 8.7 years before its histogram is
# calibrated. 60 weeks is ~14 months of context, the closest wall-clock analogue to what
# 200 daily bars give the daily screen, and well above the source's minimum of 30.
PRG_NORM = pg.DEFAULT.norm
PRG_NORM_WEEKLY = 60

# Hold horizon in bars — how long an event stays "open" in the tables, and the forward
# window the edge study measures. The Pine's own evidence used a 20-bar bracket; Sanket
# has always declared 10, and a declared horizon is not re-chosen after the fact.
HORIZON = 10

# Round-trip cost assumption for the cost gate.
COST_BPS = 3.0

# Forward horizons the Historical Range harvest attaches as Ret_*b labels.
HOLD_HORIZONS = [1, 5, 10, 21]

# The largest edge the source family has found on any instrument group (+0.036R, the
# primary futures on the continuation architecture). Used as a CEILING, not a forecast: a
# cost above it cannot be survived by any plausible version of the edge (the cost gate),
# and a test whose minimum detectable effect exceeds it cannot resolve anything (edge.py).
LARGEST_KNOWN_EFFECT = 0.036

# Cost-gate fallback in bps before a study has measured this universe's own cost charge.
POOLED_BREAKEVEN_BPS = 7.0

# Instrument class — a DISPLAY LABEL only, naming the kind of universe on screen. The
# Siddhi-era per-class expectancy priors are gone: the Pragati signal set is unmeasured in
# its source, so there is no prior to quote.
INSTRUMENT_CLASSES = [
    "US index / ETF", "US sector ETF", "India index", "International equity",
    "Commodity", "FX", "Rates / Credit", "Other / unknown",
]
UNIVERSE_CLASS_MAP = {
    "US Indexes":     "US index / ETF",
    "India Indexes":  "India index",
    "Global Indexes": "International equity",
    "ETF Index":      "India index",
    "Commodities":    "Commodity",
    "Currency":       "FX",
    "Global Macro":   "Rates / Credit",
    "Crypto":         "Other / unknown",
}


def instrument_class(universe: str, selected_index: str | None = None) -> str:
    """The display label for a universe. Nothing computes from it."""
    return UNIVERSE_CLASS_MAP.get(universe, "Other / unknown")


def chart_of(timeframe: str) -> str:
    return "W" if str(timeframe) == "Weekly" else "D"


# ════════════════════════════════════════════════════════════════════════════════════════
# SETTINGS  (one run's configuration)
# ════════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class EngineSettings:
    """One run's Pragati configuration: the indicator's inputs plus Sanket's own three."""
    timeframe: str
    params: pg.Params = field(default_factory=pg.Params)
    horizon: int = HORIZON
    cost_bps: float = COST_BPS
    iclass: str = "Other / unknown"

    @property
    def chart(self) -> str:
        return chart_of(self.timeframe)

    @property
    def norm(self) -> int:
        return int(self.params.norm)

    @property
    def params_sig(self) -> tuple:
        """Everything that changes a per-symbol analysed frame."""
        p = self.params
        return (self.chart, p.length, p.smooth, p.norm, p.participation, p.denominator,
                p.vol_n, float(p.cap), float(p.z1), float(p.z2), float(p.mix), p.signal,
                p.turn, p.resume, float(p.k), p.pull, p.effort, p.cool,
                p.pl, p.pr, p.gap_min, p.gap_max, p.zone_gate, p.quiet, p.parent_norm,
                int(self.horizon), sv.DEFAULT_BASKET)

    @property
    def study_sig(self) -> tuple:
        return self.params_sig

    @property
    def norm_is_adapted(self) -> bool:
        """True when the normalization window is Sanket's, not the indicator's."""
        return int(self.params.norm) != PRG_NORM

    @property
    def ladder_label(self) -> str:
        if self.params.ladder != "down":
            return "D inside · W" if self.chart == "W" else "W · D"
        return "1h·4h·D inside · W" if self.chart == "W" else "1m…4h inside · D (↺ W·D before intraday history)"

    @property
    def value_ladder_label(self) -> str:
        return "M · W" if self.chart == "W" else "W · D"

    @property
    def trigger_label(self) -> str:
        """How the signal set reads in prose — the one place it is worded."""
        return "▲ capitulation · ▼ distribution · ◆ RESUME"

    @property
    def trigger_short(self) -> str:
        return f"▲▼ · ◆ · {self.horizon}b"

    @property
    def min_bars(self) -> int:
        """Bars before the histogram is calibrated — below this a symbol cannot signal."""
        return pg.warmup_bars(self.params)

    @property
    def theta(self) -> float:
        return self.params.theta

    def cost_ok(self, study=None) -> bool:
        return cost_ok(self.cost_bps, study)

    def cost_basis(self, study=None) -> str:
        return cost_basis(study)


def settings_for(universe, selected_index, timeframe, overrides: dict | None = None) -> EngineSettings:
    """Resolve the active settings for a (universe, timeframe) selection.

    Every indicator input is the Pine's default except the normalization window on
    Weekly (see PRG_NORM_WEEKLY). ``overrides`` may carry any Params field or
    horizon / cost_bps — for research, never from the UI.
    """
    o = dict(overrides or {})
    pfields = {k: o.pop(k) for k in list(o) if k in pg.Params.__dataclass_fields__}
    p = pg.Params(norm=PRG_NORM_WEEKLY if str(timeframe) == "Weekly" else PRG_NORM)
    if pfields:
        p = replace(p, **pfields)
    return EngineSettings(timeframe=str(timeframe), params=p,
                          horizon=int(o.get("horizon", HORIZON)),
                          cost_bps=float(o.get("cost_bps", COST_BPS)),
                          iclass=instrument_class(universe, selected_index))


# ════════════════════════════════════════════════════════════════════════════════════════
# COST GATE  (a question about cost, never about edge)
# ════════════════════════════════════════════════════════════════════════════════════════
def _measured_cost_charge(study) -> float | None:
    if study is None:
        return None
    r = study.get("buy", "holdout") or study.get("buy", "full")
    if r is None:
        return None
    ch = getattr(r, "cost_charge", float("nan"))
    return float(ch) if np.isfinite(ch) else None


def cost_ok(cost_bps: float, study=None) -> bool:
    """Is this round-trip cost survivable here? Compared against LARGEST_KNOWN_EFFECT.

    Never against the MEASURED edge: that would fail the gate on every universe that
    measures no edge, smuggling the expectancy back in as a hidden multiplier.
    """
    try:
        c = float(cost_bps)
    except (TypeError, ValueError):
        return False
    charge = _measured_cost_charge(study)
    if charge is not None:
        return bool(charge < LARGEST_KNOWN_EFFECT)
    return c <= POOLED_BREAKEVEN_BPS


def cost_basis(study=None) -> str:
    if _measured_cost_charge(study) is not None:
        return "measured"
    return f"pooled prior (~{POOLED_BREAKEVEN_BPS:.0f}bp)"


def cost_in_vol_units(cost_bps: float, sigma_h: float) -> float:
    try:
        s = float(sigma_h)
        return float((float(cost_bps) / 1e4) / s) if s > 0 else float("nan")
    except (TypeError, ValueError, ZeroDivisionError):
        return float("nan")


# ════════════════════════════════════════════════════════════════════════════════════════
# PER-SYMBOL FEATURES  (time-series; one name, every bar)
# ════════════════════════════════════════════════════════════════════════════════════════
KIND_GLYPH = {"turn": "▲", "resume": "◆"}          # per side: ▲/▼ TURN, ◆ RESUME
EVENT_LABEL = {("turn", 1): "▲ CAPITULATION", ("turn", -1): "▼ DISTRIBUTION",
               ("resume", 1): "◆ RESUME ↑", ("resume", -1): "◆ RESUME ↓"}


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    """Sanket frames are Title-case (yfinance); the ports read lower-case."""
    out = pd.DataFrame({
        "open": pd.to_numeric(df["Open"], errors="coerce"),
        "high": pd.to_numeric(df["High"], errors="coerce"),
        "low": pd.to_numeric(df["Low"], errors="coerce"),
        "close": pd.to_numeric(df["Close"], errors="coerce"),
        "volume": (pd.to_numeric(df["Volume"], errors="coerce") if "Volume" in df.columns
                   else pd.Series(np.nan, index=df.index)),
    }, index=pd.DatetimeIndex(df.index))
    if out.index.tz is not None:
        out.index = out.index.tz_convert(None)
    return out


def _chart_bars(df: pd.DataFrame) -> pd.DataFrame:
    lo = _lower(df)
    # A non-positive close has no log return: the value engine drops it, so the conviction
    # engine must too or the two run on different bars (CL=F, 20 April 2020).
    return lo[lo["close"].notna() & lo["high"].notna() & lo["low"].notna() & (lo["close"] > 0)]


def value_frame(df: pd.DataFrame, drivers: pd.DataFrame | None, symbol: str,
                settings: EngineSettings) -> pd.DataFrame:
    """Samanvaya's value engine for one name — the part of the stack the trace setting
    never touches, so a study comparing trace settings computes it ONCE and passes it to
    :func:`compute_frame` / :func:`add_pragati_features` as ``value=``."""
    return sv.compute_value(_chart_bars(df), drivers, symbol, chart=settings.chart)


def compute_frame(df: pd.DataFrame, drivers: pd.DataFrame | None, symbol: str,
                  settings: EngineSettings, daily: pd.DataFrame | None = None,
                  value: pd.DataFrame | None = None, intraday: dict | None = None) -> pd.DataFrame:
    """The whole stack for one name, as a frame on the chart's own index.

    The conviction ladder reads DOWN (v9.1): ``intraday`` is {frame: bars} from
    intraday.py; when None it is taken from intraday.py's session cache (fetched on demand).

    ``df`` is the chart's OHLCV (Title-case, ascending). ``drivers`` the macro closes
    (samanvaya.prepare_drivers for the chart). ``daily`` the daily bars behind a weekly
    chart, for its Ladder-down conviction rung. Returns pragati's columns, the value
    engine's, and the grid's, prefixed — see add_pragati_features for the contract.
    """
    lo = _chart_bars(df)
    chart = settings.chart
    val = sv.compute_value(lo, drivers, symbol, chart=chart) if value is None else value
    dl = _lower(daily) if daily is not None and len(daily) else None
    if intraday is None and settings.params.ladder == "down":
        intraday = idm.frames(symbol, idm.DAILY_FRAMES if chart == "D" else idm.WEEKLY_FRAMES)
    out = pg.compute(lo, val, settings.params, chart=chart, daily=dl, intraday=intraday)
    # The grid (v8, 3 × 3): conviction's own histogram runs its rows, so it reads chart
    # conviction and its calibration gate beside the tapes.
    p = settings.params
    ch = pg.chart_conviction(lo, p)
    cv_ready = np.cumsum(ch["sd_ok"].to_numpy(bool)) > p.norm + p.smooth + p.signal
    grid = cg.classify(out, ch["osc"], ch["raw_sd"], cv_ready, p)
    # v9: the ▲▼ are read from the grid (the Pine's section 8b) — the capitulation
    # turn and distribution — with the ◆, the declaration and the watch.
    out = pg.signals(out, grid, p)
    extra = val[["rv_z", "breadth_z", "legs_split", "hedge", "drivers", "n_obs", "enough",
                 "basket_warm", "bars_since_rot"]]
    return pd.concat([out, grid, extra], axis=1)


def add_pragati_features(df: pd.DataFrame, drivers: pd.DataFrame | None = None,
                         symbol: str = "", settings: EngineSettings | None = None,
                         daily: pd.DataFrame | None = None,
                         value: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach the Pragati stack to one symbol's OHLCV frame. Columns written:

    READINGS
      PRG_Conv / PRG_Conv_Z      chart conviction, ±100 and in σ — the flow ingredient
      PRG_Value / PRG_Value_Z    Samanvaya's value, ±100 (+ rich) and in σ
      PRG_Trace                  conviction × value, ±100 — how far the move is stretched
      PRG_Hist / PRG_Hist_Z      the trace's push (trace − EMA9), native and in its own σ
      PRG_Push / PRG_Push_Tier   the push in five levels (−2 … +2) and its drawn tier
      PRG_CTape / PRG_VTape      the MTF conviction and value tapes (the grid's axes)
      PRG_Ladder                 the conviction ladder read on this bar: down, or up↺ where no
                                 intraday history exists yet
      PRG_Raw / PRG_Raw_Sd       the raw participation-weighted share and its σ
      PRG_Eff_Pct / PRG_Absorbed effort → result percentile; absorbed = bottom fifth
      PRG_Hedge / PRG_Drivers    the macro hedge applied and the drivers in use
      PRG_RV_Z / PRG_Breadth_Z   the value ingredient's two legs
      PRG_Split / PRG_Quiet / PRG_Settling / PRG_Legs_Split   the gold qualifiers
      PRG_Stack_OK / PRG_Why     can the signal set judge this bar, and if not why
    EVIDENCE
      PRG_Bull_Div / PRG_Bear_Div       a qualified divergence confirmed on this bar
      PRG_Div_Seen_Bull / _Bear, PRG_Abs_Seen   in the last 20 bars (context only)
    EVENTS
      turn_buy / turn_sell / resume_long / resume_short     the four signals
      long_cond / short_cond                                either kind, per side
      PRG_Event                  the label of the event on this bar, else ''
      PRG_Armed / PRG_Armed_Age  in capitulation, value still cheapening (+1) and bars in the cell
      PRG_Decl / PRG_Decl_Age    the standing declaration (+1 ▲ / −1 ▼) and its age
      PRG_Hold_Dir / PRG_Hold_Age / PRG_Hold_Kind   the latest event inside its horizon
    STATE
      CVG_Cell / CVG_Action / CVG_Why / CVG_Units / CVG_Side / CVG_Held / CVG_Bars /
      CVG_From / CVG_Chart / CVG_Lead
      PRG_State                  WARMING UP / PAUSED / ▲ … / WATCH ▲ … / NEUTRAL
      Signal_Score               the trace — what the level views sort on
    """
    settings = settings or settings_for(None, None, "Daily")
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    f = compute_frame(df, drivers, symbol, settings, daily, value=value).reindex(df.index)
    T = len(df)
    horizon = int(settings.horizon)

    b = lambda c: f[c].fillna(False).astype(bool).to_numpy()     # noqa: E731
    tb, ts, rl, rs = b("turn_buy"), b("turn_sell"), b("resume_long"), b("resume_short")
    long_c, short_c = tb | rl, ts | rs

    df["PRG_Conv"] = f["conv"].where(f["conv_ready"].fillna(False).astype(bool))
    df["PRG_Conv_Z"] = f["conv_z"]
    df["PRG_Value"] = f["value"].where(f["value_built"].fillna(False).astype(bool))
    df["PRG_Value_Z"] = f["value_z"]
    df["PRG_Trace"] = f["trace"]
    df["PRG_Hist"] = f["hist"].where(f["hist_ready"].fillna(False).astype(bool))
    df["PRG_Hist_Z"] = f["hist_z"].where(f["hist_ready"].fillna(False).astype(bool))
    df["PRG_Thr"] = f["thr"]
    df["PRG_Push"] = f["push"].fillna(0).astype(int)
    df["PRG_Push_Tier"] = f["push_tier"]
    df["PRG_CTape"] = f["c_tape"].where(f["c_ready"].fillna(False).astype(bool))
    df["PRG_VTape"] = f["v_tape"].where(f["v_ready"].fillna(False).astype(bool))
    df["PRG_Ladder"] = f["c_ladder"].fillna("")        # down · up↺ (no intraday yet) · '' warming
    df["PRG_Raw"] = f["raw"]
    df["PRG_Raw_Sd"] = f["raw_sd"]
    df["PRG_Eff_Pct"] = f["eff_pct"]
    df["PRG_Absorbed"] = b("eff_abs")
    df["PRG_Hedge"] = f["hedge"]
    df["PRG_Drivers"] = f["drivers"].fillna("")
    df["PRG_RV_Z"] = f["rv_z"]
    df["PRG_Breadth_Z"] = f["breadth_z"]
    df["PRG_Split"] = b("split")
    df["PRG_Quiet"] = b("quiet")
    df["PRG_Settling"] = b("settling")
    df["PRG_Legs_Split"] = b("legs_split")
    df["PRG_Stack_OK"] = b("stack_ok")
    df["PRG_Why"] = f["stack_why"].fillna("chart warming")
    df["PRG_Rec_Err"] = f["rec_err"]
    df["PRG_Bull_Div"] = b("bull_div")
    df["PRG_Bear_Div"] = b("bear_div")
    df["PRG_Div_Seen_Bull"] = b("bull_div_seen")
    df["PRG_Div_Seen_Bear"] = b("bear_div_seen")
    df["PRG_Abs_Seen"] = b("abs_seen")
    for c in ("div_x1", "div_p1", "div_x2", "div_p2"):
        df["PRG_" + c.title().replace("_", "")] = f[c]

    df["turn_buy"], df["turn_sell"] = tb, ts
    df["resume_long"], df["resume_short"] = rl, rs
    df["long_cond"], df["short_cond"] = long_c, short_c
    df["PRG_Event"] = np.select([tb, ts, rl, rs],
                                [EVENT_LABEL[("turn", 1)], EVENT_LABEL[("turn", -1)],
                                 EVENT_LABEL[("resume", 1)], EVENT_LABEL[("resume", -1)]], "")

    armed = f["armed"].fillna(0).astype(int).to_numpy()
    df["PRG_Armed"] = armed
    df["PRG_Armed_Age"] = np.where(armed != 0, f["armed_age"].fillna(0).astype(int), 0)
    decl = f["decl"].fillna(0).astype(int).to_numpy()
    since = f["decl_since"].fillna(-1).astype(int).to_numpy()
    pos = np.arange(T)
    df["PRG_Decl"] = decl
    df["PRG_Decl_Age"] = np.where(decl != 0, pos - since, np.nan)

    # ── the hold window: the latest event, while it is inside the declared horizon ──
    fires = long_c | short_c
    fire_dir = np.where(long_c, 1.0, np.where(short_c, -1.0, np.nan))
    fire_kind = np.where(tb, "▲ capitulation", np.where(ts, "▼ distribution", np.where(rl | rs, "◆ RESUME", None)))
    last_fire = pd.Series(np.where(fires, pos.astype(float), np.nan)).ffill().to_numpy()
    held_dir = pd.Series(fire_dir).ffill().to_numpy()
    held_kind = pd.Series(fire_kind, dtype=object).ffill().to_numpy()
    age = pos - last_fire
    in_win = np.isfinite(age) & (age <= horizon)
    df["PRG_Hold_Dir"] = np.where(in_win, np.nan_to_num(held_dir), 0.0).astype(int)
    df["PRG_Hold_Age"] = np.where(in_win, age, np.nan)
    df["PRG_Hold_Kind"] = np.where(in_win, held_kind, "")

    # ── the grid ──
    cell = f["cvg_cell"].fillna(cg.UNREAD).astype(int).to_numpy()
    df["CVG_Cell"] = cell
    df["CVG_Action"] = [cg.action(c) for c in cell]
    df["CVG_Why"] = [cg.reason(c) for c in cell]
    # graded: the cell's units read at the name's shaded position (Pragyam's map)
    df["CVG_Units"] = f["cvg_units"].where(f["cvg_cell"].fillna(cg.UNREAD) != cg.UNREAD,
                                           cg.UNITS[cg.UNREAD]).astype(float).round(3)
    df["CVG_Side"] = [cg.SIDES[c] for c in cell]
    df["CVG_Held"] = b("cvg_held")
    df["CVG_Bars"] = pos - f["cvg_since"].fillna(0).astype(int).to_numpy() + 1
    df["CVG_From"] = f["cvg_from"].fillna(cg.UNREAD).astype(int)
    chart_cell = f["cvg_chart"].fillna(cg.UNREAD).astype(int).to_numpy()
    df["CVG_Chart"] = chart_cell
    df["CVG_Chart_Action"] = [cg.NAMES[c] for c in chart_cell]
    df["CVG_Lead"] = f["cvg_lead"].fillna(0).astype(int)

    stack = b("stack_ok")
    hist_ready = b("hist_ready")
    df["PRG_State"] = np.select(
        [~hist_ready, ~stack, tb, ts, rl, rs, armed > 0, armed < 0],
        ["WARMING UP", "PAUSED", "CAPITULATION ▲", "DISTRIBUTION ▼", "RESUME ◆↑", "RESUME ◆↓", "WATCH ▲", "WATCH ▼"],
        default="NEUTRAL")
    df["Signal_Score"] = df["PRG_Trace"]
    return df


# ════════════════════════════════════════════════════════════════════════════════════════
# THE SNAPSHOT ROW  (one symbol, one date — what the screener tables carry)
# ════════════════════════════════════════════════════════════════════════════════════════
SNAPSHOT_COLUMNS = (
    "PRG_Conv", "PRG_Conv_Z", "PRG_Value", "PRG_Value_Z", "PRG_Trace", "PRG_Hist", "PRG_Hist_Z",
    "PRG_Push", "PRG_Push_Tier", "PRG_CTape", "PRG_VTape", "PRG_Raw", "PRG_Raw_Sd",
    "PRG_Eff_Pct", "PRG_Absorbed", "PRG_Hedge", "PRG_Drivers", "PRG_RV_Z", "PRG_Breadth_Z",
    "PRG_Split", "PRG_Quiet", "PRG_Settling", "PRG_Legs_Split", "PRG_Stack_OK", "PRG_Why",
    "PRG_Div_Seen_Bull", "PRG_Div_Seen_Bear", "PRG_Abs_Seen", "PRG_Event",
    "turn_buy", "turn_sell", "resume_long", "resume_short", "long_cond", "short_cond",
    "PRG_Armed", "PRG_Armed_Age", "PRG_Decl", "PRG_Decl_Age",
    "PRG_Hold_Dir", "PRG_Hold_Age", "PRG_Hold_Kind",
    "CVG_Cell", "CVG_Action", "CVG_Why", "CVG_Units", "CVG_Side", "CVG_Held", "CVG_Bars",
    "CVG_From", "CVG_Chart", "CVG_Chart_Action", "CVG_Lead", "PRG_State", "Signal_Score",
)


def _clean(v):
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def snapshot(df: pd.DataFrame, idx_pos: int, settings: EngineSettings) -> dict:
    """The engine's fields for one bar, plus the per-age event history the tables bucket.

    ``BUY_Today … BUY_5d`` carry the long event's glyph at that age (▲ capitulation, ◆ RESUME)
    or '—'; ``SELL_*`` the short side's. ``Kind_Hist`` / ``Trace_Hist`` / ``Close_Hist``
    are [today, 1 back, …, 4 back] so an aged row can report the bar that fired it.
    """
    row = df.iloc[idx_pos]
    out = {c: _clean(row.get(c)) for c in SNAPSHOT_COLUMNS if c in df.columns}
    lo = max(0, idx_pos - 4)
    win = df.iloc[lo: idx_pos + 1]

    def _glyph(r, side):
        if side > 0:
            return "▲" if r["turn_buy"] else "◆" if r["resume_long"] else "—"
        return "▼" if r["turn_sell"] else "◆" if r["resume_short"] else "—"

    longs = [_glyph(r, 1) for _, r in win.iterrows()][::-1]
    shorts = [_glyph(r, -1) for _, r in win.iterrows()][::-1]
    longs += ["—"] * (5 - len(longs))
    shorts += ["—"] * (5 - len(shorts))
    for i, lab in enumerate(("Today", "1d", "2d", "3d")):
        out[f"BUY_{lab}"] = longs[i]
        out[f"SELL_{lab}"] = shorts[i]
    out["BUY_5d"] = next((g for g in longs if g != "—"), "—")
    out["SELL_5d"] = next((g for g in shorts if g != "—"), "—")

    def _hist(col):
        v = [float(x) if pd.notna(x) else float("nan") for x in win[col].tolist()][::-1]
        return v + [float("nan")] * (5 - len(v))

    out["Trace_Hist"] = _hist("PRG_Trace")
    out["Close_Hist"] = _hist("Close")
    out["Units_Hist"] = _hist("CVG_Units")
    out["PRG_Horizon"] = int(settings.horizon)
    return out


# ════════════════════════════════════════════════════════════════════════════════════════
# CROSS-SECTIONAL RANKING  (one date's universe)
# ════════════════════════════════════════════════════════════════════════════════════════
def grid_weight(units, side: int = 1):
    """The grid's weight in [0, 1): (units − ¼)/2¾ for the long side, mirrored short."""
    u = np.asarray(units, dtype=float)
    g = (u - 0.25) / 2.75 if side > 0 else (3.0 - u) / 2.75
    return np.clip(g, 0.0, 0.999)


def priorities(tb, ts, trace, ready) -> tuple:
    """(long, short) priority on plain arrays of any shape (÷100 of the column).

    ▲ / ▼ on this bar ranks first on its side (5 + s); every other name by its
    stretch, s = −trace/200 for the long side and +trace/200 for the short side. See
    the module docstring for the measurement behind this. Shared by
    :func:`compute_ranking` and trace_study.py, so the study ranks exactly as the screen.
    """
    tr = np.asarray(trace, dtype=float)
    s = np.clip(-tr / 200.0, -0.5, 0.5)
    pl = np.where(tb, 5.0 + s, s)
    ps = np.where(ts, 5.0 - s, -s)
    return np.where(ready, pl, np.nan), np.where(ready, ps, np.nan)


RANK_CONTRACT = ("Side", "Signal_Kind", "Priority_Long", "Priority_Short",
                 "Priority_Long_pct", "Priority_Short_pct", "Trace_Rank_Pct",
                 "Grid_Weight", "Cost_OK", "Signal_Reason")


def compute_ranking(df: pd.DataFrame, settings: EngineSettings | None = None,
                    study=None) -> pd.DataFrame:
    """Rank one date's cross-section. Pure and deterministic; see the module docstring.

    ``df`` has one row per symbol carrying the snapshot fields. Adds RANK_CONTRACT and
    returns the frame sorted by Priority_Long (warming rows last).
    """
    settings = settings or settings_for(None, None, "Daily")
    df = df.copy()
    if len(df) == 0:
        for c in RANK_CONTRACT:
            df[c] = pd.Series(dtype=float)
        return df
    idx = df.index
    horizon = float(max(int(settings.horizon), 1))

    def _b(c):
        return (df[c].fillna(False).astype(bool).to_numpy() if c in df.columns
                else np.zeros(len(df), dtype=bool))

    def _f(c, default=np.nan):
        return (pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) if c in df.columns
                else np.full(len(df), default))

    tb, ts, rl, rs = _b("turn_buy"), _b("turn_sell"), _b("resume_long"), _b("resume_short")
    units = _f("CVG_Units", 1.0)
    units = np.where(np.isfinite(units), units, 1.0)
    ready = np.isfinite(_f("PRG_Trace"))
    armed = _f("PRG_Armed", 0.0)
    a_age = _f("PRG_Armed_Age", 0.0)
    h_dir = _f("PRG_Hold_Dir", 0.0)
    h_age = _f("PRG_Hold_Age")

    df["Side"] = np.where(tb | rl, "Buy", np.where(ts | rs, "Sell", "—"))
    df["Signal_Kind"] = np.where(tb | ts, "TURN", np.where(rl | rs, "RESUME", ""))
    p_long, p_short = priorities(tb, ts, _f("PRG_Trace"), ready)
    p_long = pd.Series(p_long, index=idx)
    p_short = pd.Series(p_short, index=idx)
    df["Priority_Long"] = p_long * 100.0
    df["Priority_Short"] = p_short * 100.0
    df["Priority_Long_pct"] = p_long.rank(pct=True) * 100
    df["Priority_Short_pct"] = p_short.rank(pct=True) * 100
    tr = pd.Series(_f("PRG_Trace"), index=idx)
    df["Trace_Rank_Pct"] = (tr.rank(pct=True) * 100).round(2) if len(df) >= 2 else 50.0
    df["Grid_Weight"] = units / 3.0
    gate = cost_ok(settings.cost_bps, study)
    df["Cost_OK"] = gate

    def _note(side_key: str) -> str:
        if study is None:
            return "expectancy not yet measured on this universe"
        lbl, _k, _d = study.verdict(side_key)
        return f"measured on this universe: {lbl}"

    notes = {"buy": _note("buy"), "sell": _note("sell")}
    cost_txt = "" if gate else f" · cost gate fails at {settings.cost_bps:g}bp"
    trace = _f("PRG_Trace")
    theta = float(settings.params.theta)

    def _stretch(t: float) -> str:
        # The ranking's own reason: how far the trace is stretched, read as reversion.
        if t <= -theta:
            return f"stretched down {t:+.0f} — reversion candidate, leads the long side"
        if t >= theta:
            return f"stretched up {t:+.0f} — reversion candidate, leads the short side"
        return f"trace {t:+.0f}, inside ±{theta:.0f} — little stretch to revert"

    reasons = []
    for i, r in enumerate(df.itertuples(index=False)):
        rd = r._asdict()
        grid = f"grid {rd.get('CVG_Action', '—')} · {rd.get('CVG_Why', '')} ({units[i]:g}u)"
        if not ready[i]:
            reasons.append(f"warming up — {rd.get('PRG_Why') or 'the histogram is not calibrated yet'}")
            continue
        if not bool(rd.get("PRG_Stack_OK", False)):
            reasons.append(f"signals paused — {rd.get('PRG_Why') or 'a tape is warming'} · {grid}")
            continue
        ev = ""
        if tb[i] or rl[i] or ts[i] or rs[i]:
            side_key = "buy" if (tb[i] or rl[i]) else "sell"
            kind = "turn" if (tb[i] or ts[i]) else "resume"
            lab = EVENT_LABEL[(kind, 1 if side_key == "buy" else -1)]
            if kind == "turn":
                ev = (f"{lab} · " + ("sellers in control of a cheap price, and value has turned "
                                     "back toward fair" if side_key == "buy" else
                                     "sellers have taken control of a rich price")
                      + f" · {notes[side_key]}")
            else:
                ev = (f"{lab} · a trend resuming from inside the zone, control held · "
                      f"{notes[side_key]}")
            reasons.append(f"{ev} · {grid}{cost_txt}")
            continue
        a = int(armed[i]) if np.isfinite(armed[i]) else 0
        if a != 0:
            reasons.append(f"{_stretch(trace[i])} · in capitulation {int(a_age[i])} bars, value still "
                           f"cheapening — the ▲ comes when it turns · {grid}")
            continue
        hd = int(h_dir[i]) if np.isfinite(h_dir[i]) else 0
        if hd != 0 and np.isfinite(h_age[i]):
            k = rd.get("PRG_Hold_Kind") or "event"
            reasons.append(f"{_stretch(trace[i])} · {'long' if hd > 0 else 'short'} {k} "
                           f"{int(h_age[i])} bars ago, inside its {int(horizon)}-bar hold · {grid}")
            continue
        reasons.append(f"{_stretch(trace[i])} · {grid}")
    df["Signal_Reason"] = reasons
    return df.sort_values("Priority_Long", ascending=False, kind="stable", na_position="last")


__all__ = [
    "COST_BPS", "EVENT_LABEL", "EngineSettings", "HOLD_HORIZONS", "HORIZON", "KIND_GLYPH",
    "LARGEST_KNOWN_EFFECT", "POOLED_BREAKEVEN_BPS", "PRG_NORM", "PRG_NORM_WEEKLY",
    "RANK_CONTRACT", "SNAPSHOT_COLUMNS", "add_pragati_features", "chart_of", "compute_frame",
    "compute_ranking", "cost_basis", "cost_in_vol_units", "cost_ok", "grid_weight",
    "instrument_class", "settings_for", "snapshot",
]
