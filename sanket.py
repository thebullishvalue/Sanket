"""
Sanket - Market Signal Screener | A Pragyam Product Family Member
Pragati · Conviction × Value — Quantitative Signal Screener Terminal

Engine: PRAGATI (pragati.pine v9.2), the indicator Pragyam's Conviction-Value Grid
reads. One trace — conviction × value, how far a move is stretched in one-sided
effort and in price against fair value — its histogram (the trace's push), and
its two ingredients read across horizons on two tapes. One state: the 3 × 3
conviction-value grid, named as an action with its measured units. Two events,
read from the grid (v9): ▲ CAPITULATION (sellers in control of a cheap price, value
turning back toward fair) and ▼ DISTRIBUTION (sellers taking control of a rich
price); ◆ RESUME is off by default. `edge.py` measures the signal set on the symbols
actually on screen. See engine.py, pragati.py, samanvaya.py, cvgrid.py and
ARCHITECTURE.md.
"""

import os

# ── BLAS thread pinning (MUST run before numpy import) ────────────────────────
# The screener runs the regime engine + a rolling volume profile across a ~500-
# symbol universe. On Streamlit Community Cloud the container is throttled to ~1
# shared vCPU but the host reports many logical CPUs, so OpenBLAS/MKL spawn one
# thread per reported core and thrash. One thread per process is strictly faster
# here. os.environ.setdefault → respects any explicit override from the env.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import html
import re
import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import numpy as np
import plotly.graph_objects as go
import requests
import io
import urllib3
import engine as eng
import edge
import intraday as idm
import samanvaya as sv
import cvgrid as cg
import charts
import warnings
import logging
import time
from dataclasses import dataclass
from typing import Optional
from nsepython import nse_get_advances_declines
from logger import console

# UI — Obsidian Quant Terminal System
from ui.theme import (inject_css, apply_chart_theme, progress_bar, chart_color,
                      chart_rgba, grid_rgba)
import ui.components as ui

# ── SVG ICON SYSTEM ────────────────────────────────────────────────────────
SVGS = {
    "CHECK": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>',
    "LONG": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m5 12 7-7 7 7"/><path d="M12 19V5"/></svg>',
    "SHORT": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14"/><path d="m19 12-7 7-7-7"/></svg>',
    "DOT": '<svg width="8" height="8" viewBox="0 0 24 24" fill="currentColor" style="display: inline-block; vertical-align: middle; margin-right: 4px;"><circle cx="12" cy="12" r="10"/></svg>',
    "UP": '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 4px;"><path d="m5 12 7-7 7 7"/><path d="M12 19V5"/></svg>',
    "DOWN": '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 4px;"><path d="M12 5v14"/><path d="m19 12-7 7-7-7"/></svg>',
    "ZAP": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m13 2-2 10h3L11 22l2-10h-3l2-10z"/></svg>',
    "CHART": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>',
    "STRENGTH": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z"/><path d="M8 8V4h8v4"/><path d="M16 16v4H8v-4"/></svg>',
    "SETTINGS": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.1a2 2 0 0 1-1-1.72v-.51a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>'
}

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Silence noisy warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
np.seterr(divide="ignore", invalid="ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SANKET | Market Signal Screener",
    page_icon="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRDN0RGMCIgc3Ryb2tlLXdpZHRoPSIyIi8+PHBhdGggZD0iTTggMTRsMy01IDIgMyAzLTQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRDN0RGMCIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz48L3N2Zz4=",
    layout="wide",
    initial_sidebar_state="expanded",
)

VERSION = "v9.2.0"

# ── Engine identity ───────────────────────────────────────────────────────────
# Named for what it measures: progress (प्रगति), and the price it was made at. Defined here
# so the name appears in exactly one place.
ENGINE_NAME = "Pragati · Conviction × Value"
ENGINE_CODE = "PRAGATI"

# Bumped whenever the engine's OUTPUT changes for inputs that are otherwise identical.
# It feeds both cache identities — the analysed-frame signature and the edge-study key —
# because a parameter tuple is not sufficient on its own.
#   sid1  v7.0.0  initial Siddhi port
#   sid2  v7.0.1  counted warmup (452 bars, was 245); "Raw share" scaling removed
#   prg1  v8.0.0  Pragati v6: trace, tapes, TURN / RESUME, the 4 × 4 grid
#   prg2  v9.0.0  Pragati v9: the ▲▼ read from the grid; the Edge Study scored causally
#   prg3  v9.1.0  the conviction ladder reads DOWN from yfinance intraday (↺ W·D before it)
ENGINE_SIG = "prg3"

# IST timezone offset — used wherever "today" matters for data or display
_IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

def _today_ist() -> datetime.date:
    """Return the current calendar date in IST (UTC+5:30)."""
    return datetime.datetime.now(_IST).date()


# ══════════════════════════════════════════════════════════════════════════════
# SESSION-STATE DATA REGISTRY
#
# Unified OHLCV pool per session.  Instead of re-fetching the same universe
# on every mode switch, all analysis paths share one in-memory store keyed by
# (frozenset(stock_list), days_back).  The registry is populated with the timeframe's
# full depth (_max_days_back) so every mode (screener, range harvest, correlation) can
# slice what it needs without an extra round-trip.  The depth is IN THE KEY because
# Daily and Weekly need different amounts of history and one must not be served the
# other's pool.
#
# Two-tier caching:
#   L1 — session-state registry (per-user, sub-millisecond lookup)
#   L2 — @st.cache_data on fetch_batch_data (cross-user, process-level, 5 min TTL)
#   L3 — yfinance network fetch (slow path, only on true misses)
# ══════════════════════════════════════════════════════════════════════════════

_REGISTRY_KEY  = "data_registry"

# Fetch depth, in calendar days, per timeframe. Fetched once per (universe, depth); all
# modes then slice what they need. `fetch_batch_data` pads a further 365 calendar days on
# top of whichever value is used.
_MAX_DAYS_BACK        = 1300   # Daily  → ~1150 bars, ~480 of them signal-bearing
_MAX_DAYS_BACK_WEEKLY = 2600   # Weekly → ~425 bars, ~165 of them signal-bearing
# THE BINDING WARM-UP IS THE LADDER, not the chart. The histogram is calibrated at
# length + vol_n + 2·norm + signal = 469 daily bars, but every signal also needs both tapes:
# the conviction tape's weekly rung calibrates after 92 weeks (lookback + baseline + its
# 52-week window) and then holds a normalization window more, so the stack first judges
# near bar ~670. 1300 + 365 calendar days leaves ~480 signal-bearing daily dates — the live
# cross-section plus a Historical Range over the same pool.
#
# WEEKLY IS BOUND BY THE VALUE LADDER: its parent rung is the MONTH, and the RV ensemble's
# slowest member needs 55 settled months after the model builds — about five years. 2600 +
# 365 days (~8 years) leaves ~3 years of weekly bars on which the whole stack can judge.
# The pool is raw OHLCV (5 float columns), ~50 MB on a 500-symbol universe.
# Bound the L1 registry so cycling through indices (or stock_list variations from
# transient fetch failures) can't accumulate stale 500-day universe DataFrames in
# session_state until the tab closes. Keep only the N most-recently-used universes;
# each entry is one universe's worth of OHLCV (~a few hundred rows × N symbols).
_REGISTRY_MAX_ENTRIES = 6


def _registry_ttl_seconds() -> int:
    """15 min during NSE market hours (Mon–Fri 09:15–15:30 IST), 90 min outside."""
    now = datetime.datetime.now(_IST)
    mo  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    mc  = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if now.weekday() < 5 and mo <= now <= mc:
        return 15 * 60
    return 90 * 60


def _max_days_back(timeframe: str = "Daily") -> int:
    """Calendar days of history to fetch for a timeframe. See the constants above."""
    return _MAX_DAYS_BACK_WEEKLY if str(timeframe) == "Weekly" else _MAX_DAYS_BACK


def _registry_get(stock_list: list, end_date: datetime.date, days_back: int):
    """Return cached data_dict if still fresh for this universe+date+depth, else None.

    ``days_back`` is part of the identity, not a hint: a Daily run stores a ~900-day pool
    and a Weekly run needs ~1900, so serving one from the other would silently hand the
    weekly screen a frame too short to warm its oscillator.

    On a hit, the key is moved to the most-recently-used position so the LRU
    eviction in _registry_put drops genuinely-cold universes, not just oldest-stored.
    """
    reg   = st.session_state.get(_REGISTRY_KEY, {})
    key   = (frozenset(stock_list), int(days_back))
    entry = reg.get(key)
    if entry is None or entry["end_date"] != end_date:
        return None
    age = (datetime.datetime.now(_IST) - entry["fetched_at"]).total_seconds()
    if age > _registry_ttl_seconds():
        return None
    # Mark as recently used (dict preserves insertion order → re-insert = move to end).
    reg[key] = reg.pop(key)
    return entry["data"]


def _registry_put(stock_list: list, end_date: datetime.date, data_dict: dict,
                  days_back: int):
    """Store data_dict in the session-state registry under (frozenset(stock_list), depth).

    DataFrames are stored as copies so downstream mutation (adding indicator
    columns) never corrupts the cached source data. Bounded LRU: when the registry
    exceeds _REGISTRY_MAX_ENTRIES, the least-recently-used universes are evicted so
    memory can't grow without limit across index switches / re-fetches.
    """
    if _REGISTRY_KEY not in st.session_state:
        st.session_state[_REGISTRY_KEY] = {}
    reg = st.session_state[_REGISTRY_KEY]
    key = (frozenset(stock_list), int(days_back))
    reg.pop(key, None)            # ensure re-insert lands at the most-recent end
    reg[key] = {
        "data":       {k: v.copy() for k, v in data_dict.items()},
        "end_date":   end_date,
        "fetched_at": datetime.datetime.now(_IST),
    }
    # Evict least-recently-used (front of the insertion-ordered dict) past the cap.
    while len(reg) > _REGISTRY_MAX_ENTRIES:
        reg.pop(next(iter(reg)))


# ──────────────────────────────────────────────────────────────────────────────
# Analyzed-frame cache (L1.5) — avoid re-running the per-stock analysis pipeline
# (run_full_analysis + run_regime_analysis + calculate_divergences) twice when a
# forced/missing-profile screener run first harvests the timeseries and then
# re-screens the same universe in the same rerun.
#
# Safe because the analysis is causal: every bar's values depend only on trailing
# data, so a frame ending at `analysis_date` (harvest) and one extending to today
# (screener) share identical values on the overlapping bars. The cache key
# therefore encodes `end_date` — a backdated screener (different date basis, needs
# post-analysis-date bars) gets a different key and correctly bypasses the cache.
#
# Frames are stored post-analysis; consumers must not mutate them in place (the
# screener copies before adding its own columns). Scoped per screener run: the
# harvest writes it, the screener reads it, then it is cleared.
# ──────────────────────────────────────────────────────────────────────────────
_ANALYZED_CACHE_KEY = "analyzed_frame_cache"


def _analysis_params_sig(timeframe, reg_len, wt_n1, wt_n2, levels,
                         wt2_len, wt2_type, end_date, sb_params=None) -> tuple:
    """Identity of an analyzed frame — everything that changes its computed values.

    The engine tag invalidates frames cached under a previous signal/feature engine.
    History: 'rev1'–'rev6' = the retired reversion-ranker + delta-divergence/clamp-cross
    signal sets; 'mom1'/'mom2' (v5.0/v5.1) = the 12-1 momentum rank with the Set A/Set B
    entry screeners; 'sbv8'/'clr1' (v6.0/v6.1) = close-location reversal; 'sid1'/'sid2'
    (v7.0.x) = the Siddhi conviction oscillator; 'prg1' (v8.0) = Pragati, conviction ×
    value. The live tag is :data:`ENGINE_SIG`.

    ``sb_params`` = :attr:`eng.EngineSettings.params_sig`. These are baked into the frame
    (the events, the grid state and the hold window all depend on them), so a change must
    miss the cache rather than serve stale conditions. The engine tag covers the case the
    parameters cannot: a fix that changes what identical parameters produce.
    """
    return (ENGINE_SIG, str(timeframe), int(reg_len), int(wt_n1), int(wt_n2),
            tuple(levels), int(wt2_len), str(wt2_type), end_date,
            tuple(sb_params) if sb_params else None)


def _analyzed_cache_reset(params_sig: tuple):
    """Start a fresh analyzed-frame cache for one screener run under params_sig."""
    st.session_state[_ANALYZED_CACHE_KEY] = {"sig": params_sig, "frames": {}}


def _analyzed_cache_put(ticker: str, df: pd.DataFrame, params_sig: tuple):
    """Store an analyzed frame if the active cache matches params_sig."""
    cache = st.session_state.get(_ANALYZED_CACHE_KEY)
    if cache is None or cache.get("sig") != params_sig:
        return
    cache["frames"][ticker] = df


def _analyzed_cache_get(ticker: str, params_sig: tuple):
    """Return a cached analyzed frame for (ticker, params_sig), or None on miss."""
    cache = st.session_state.get(_ANALYZED_CACHE_KEY)
    if cache is None or cache.get("sig") != params_sig:
        return None
    return cache["frames"].get(ticker)


def _analyzed_cache_clear():
    st.session_state.pop(_ANALYZED_CACHE_KEY, None)


def get_universe_data(stock_list: list, end_date: datetime.date = None,
                      timeframe: str = "Daily"):
    """Fetch OHLCV data for a universe, checking the session-state registry first.

    Fetches :func:`_max_days_back` days for the timeframe so the screener, the range
    harvest and correlation can all slice from the same pool without re-fetching.
    Correlation callers should pass only the universe symbols here, then supplement the
    returned dict with a single-ticker fetch for the target asset if it is missing.

    ``timeframe`` selects the depth, and the depth is part of the cache identity — a
    Weekly run must not be served the Daily run's shallower pool.

    Returns: (data_dict, message_str) — same contract as fetch_batch_data.
    """
    if end_date is None:
        end_date = _today_ist()
    days_back = _max_days_back(timeframe)

    cached = _registry_get(stock_list, end_date, days_back)
    if cached is not None:
        console.detail(
            f"Data registry HIT — {len(cached)} symbols available "
            f"(requested {len(stock_list)}, end_date={end_date})"
        )
        return cached, f"✓ {len(cached)} symbols (session registry)"

    console.detail(
        f"Data registry MISS — fetching {len(stock_list)} symbols "
        f"from yfinance (end_date={end_date}, days_back={days_back}, {timeframe})"
    )
    data_dict, msg = fetch_batch_data(
        stock_list, end_date=end_date, days_back=days_back
    )
    if data_dict:
        _registry_put(stock_list, end_date, data_dict, days_back)
    return data_dict, msg


def _ladder_frames(timeframe: str = "Daily") -> tuple:
    """The intraday rungs of a timeframe's Ladder down: 1m…4h under a daily bar, 1h · 4h
    (beside the daily rung) under a weekly one."""
    return idm.WEEKLY_FRAMES if str(timeframe) == "Weekly" else idm.DAILY_FRAMES


def _prefetch_intraday(symbols: list, progress=None, timeframe: str = "Daily") -> None:
    """The conviction ladder reads DOWN (v9.1): batch-fetch every intraday frame yfinance
    carries for the universe once, so each name's engine call reads from the cache.

    ``progress(i, n, frame)`` drives the caller's progress bar, one step per frame. Logs the
    coverage per frame. On a daily chart a name with no intraday frame reads Ladder up (↺
    W · D) throughout; on a weekly chart it still reads down, on the daily rung alone."""
    weekly = str(timeframe) == "Weekly"
    fallback = "their weekly ladder reads the daily rung only" if weekly else "their conviction reads W · D (↺)"
    wanted = _ladder_frames(timeframe)
    try:
        t0 = time.time()
        cov = idm.prefetch(symbols, frames=idm.sources(wanted), on_frame=progress, report=wanted)
        n = len(symbols)
        if cov:
            console.item("Intraday ladder", " · ".join(f"{f} {cov.get(f, 0)}" for f in wanted)
                         + (f" of {n} symbols [{_dt:.1f}s]" if (_dt := time.time() - t0) >= 0.1 else f" of {n} symbols [cached]"))
            none = n - max(cov.values())
            if none:
                console.warning(f"{none} symbol(s) have no intraday history — {fallback}")
        else:
            console.item("Intraday ladder", f"disabled — {fallback}")
    except Exception as e:
        console.warning(f"Intraday prefetch failed ({type(e).__name__}: {e}) — {fallback}")

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

if "results_df" not in st.session_state:
    st.session_state["results_df"] = None
if "run_screener_flag" not in st.session_state:
    st.session_state["run_screener_flag"] = False
if "timeseries_done" not in st.session_state:
    st.session_state["timeseries_done"] = False
if "ts_results_df" not in st.session_state:
    st.session_state["ts_results_df"] = None
if "ts_meta" not in st.session_state:
    st.session_state["ts_meta"] = None
if "run_error" not in st.session_state:
    st.session_state["run_error"] = None
if "corr_data" not in st.session_state:
    st.session_state["corr_data"] = None
if "screener_meta" not in st.session_state:
    st.session_state["screener_meta"] = None
if _REGISTRY_KEY not in st.session_state:
    st.session_state[_REGISTRY_KEY] = {}

# ──────────────────────────────────────────────────────────────────────────────
# Engine settings — ONE definition, in engine.py (EngineSettings), shared by the screener,
# the range harvest, correlation and the edge study. Every indicator input is pragati.pine's
# own default, NOT a value fitted here: its 900-configuration search found fitted and
# out-of-sample edge uncorrelated. `iclass` is a DISPLAY LABEL only.
# ──────────────────────────────────────────────────────────────────────────────
def _engine_settings(universe, selected_index, timeframe, overrides=None) -> eng.EngineSettings:
    """Resolve the active settings for a (universe, timeframe) selection."""
    return eng.settings_for(universe, selected_index, timeframe, overrides)


def _active_engine_settings() -> eng.EngineSettings:
    """The settings the last run resolved, for renderers that don't take them as args."""
    es = st.session_state.get("engine_settings")
    if isinstance(es, eng.EngineSettings):
        return es
    return _engine_settings(None, None, "Daily")


# ══════════════════════════════════════════════════════════════════════════════
# EDGE STUDY — measured expectancy for the universe on screen
#
# Replaces what used to be a hardcoded eight-row lookup of the source study's per-class
# results. See edge.py for the method and why each step exists.
#
# Everything here is shaped by the deployment target: Streamlit Community Cloud, ~1 GB RAM
# on a shared vCPU. The naive implementation — fetch 15 years for 500 symbols, run the full
# analysis pipeline, hold the panel — is ~500 MB and OOMs. Three choices avoid that:
#
#   1. LEAN     the study computes the conviction oscillator and forward returns ONLY. No
#               volume profile (a Python double loop, the app's slowest path), no regime
#               engine, no order flow. The study does not need them.
#   2. STREAMING symbols are fetched and reduced in chunks; each chunk's frames are released
#               before the next is fetched. What accumulates is event tuples at a ~11% fire
#               rate — a few MB, not a panel.
#   3. SAMPLED  large universes are sampled. This costs almost nothing statistically because
#               the participation ratio saturates: 500 correlated NSE equities carry ~15-20
#               independent observations per date, not 500. Sampling is reported, not hidden.
# ══════════════════════════════════════════════════════════════════════════════

# ~15 years. The power arithmetic: resolving an effect of e needs n_eff ~ (1.96/e)^2, and
# n_eff = (n_dates / horizon) x participation_ratio. At 15y daily with a 10-bar horizon and
# a typical equity participation ratio of ~10, n_eff ~ 3500 → resolves ~0.033. The screener's
# own 900-day window would give n_eff ~260 → resolves only ~0.12, i.e. nothing but the single
# largest effect the source study ever found. Hence a separate, deeper fetch.
_STUDY_YEARS = 15
_STUDY_SYMBOL_CAP = 80      # sampling cap; the participation ratio saturates well below this
_STUDY_CHUNK = 20           # symbols per yfinance request — bounds the download memory spike
_STUDY_CORR_BARS = 1000     # bars used for the participation-ratio correlation matrix
_STUDY_MIN_SYMBOLS = 5      # below this the cross-section is too thin to study at all
# Share of one run's progress bar the study takes when it actually measures. The analysis that
# follows renders into the remainder of the SAME bar, so a run shows one continuous bar.
_STUDY_PROGRESS_SHARE = 35

_EDGE_KEY = "edge_studies"          # session cache: {key: EdgeStudy}
_EDGE_DISK_DIR = ".sanket_cache"    # ephemeral on Streamlit Cloud; treated as best-effort


def _edge_key(universe, selected_index, timeframe, sid: eng.EngineSettings) -> str:
    """Cache identity for a study: engine + universe + timeframe + measured-at parameters.

    :data:`ENGINE_SIG` is in the key because the disk cache outlives a deploy: a study
    measured against a previous engine must never be served as a measurement of this one.
    """
    parts = [ENGINE_SIG, str(universe), str(selected_index), str(timeframe),
             "p" + _slug("_".join(str(x) for x in sid.study_sig))]
    return _slug("__".join(parts))


def _study_is_fresh(study) -> bool:
    """Is this study still current?

    The study reads completed bars and needs forward returns, so it excludes the forming bar:
    two runs on the same calendar day measure identical data and must produce a bit-identical
    answer. Re-measuring within a day is therefore provably redundant work — a 15-year fetch
    for a result we already have. A study is fresh for the IST day it was measured on, and
    goes stale when the date rolls, which is exactly when new bars can change the answer.
    """
    stamp = str(getattr(study, "measured_at", "") or "")[:10]
    return stamp == _today_ist().strftime("%Y-%m-%d")


def _edge_cache_get(key: str, require_fresh: bool = False):
    """Session cache first, then the best-effort disk cache. None on a miss.

    ``require_fresh`` is used by :func:`ensure_edge_study` to decide whether to re-measure.
    Renderers leave it False: showing yesterday's measurement is far better than showing
    nothing, and the card carries the measurement timestamp.
    """
    mem = st.session_state.setdefault(_EDGE_KEY, {})
    hit = mem.get(key)
    if hit is not None:
        return None if (require_fresh and not _study_is_fresh(hit)) else hit
    # Disk is a courtesy: on Streamlit Cloud the container filesystem is wiped on restart,
    # so a miss here is normal and never an error.
    try:
        import json
        path = os.path.join(_EDGE_DISK_DIR, f"{key}.json")
        if os.path.exists(path):
            with open(path, "r") as fh:
                study = edge.EdgeStudy.from_dict(json.load(fh))
            mem[key] = study
            console.detail(f"Edge study: loaded from disk cache ({key})")
            return None if (require_fresh and not _study_is_fresh(study)) else study
    except Exception as e:
        console.detail(f"Edge study: disk cache read skipped ({type(e).__name__}: {e})")
    return None


def _edge_cache_put(key: str, study) -> None:
    st.session_state.setdefault(_EDGE_KEY, {})[key] = study
    try:
        import json
        os.makedirs(_EDGE_DISK_DIR, exist_ok=True)
        with open(os.path.join(_EDGE_DISK_DIR, f"{key}.json"), "w") as fh:
            json.dump(study.to_dict(), fh)
    except Exception as e:
        console.detail(f"Edge study: disk cache write skipped ({type(e).__name__}: {e})")


def _active_edge_study(universe=None, selected_index=None, timeframe=None, sid=None):
    """The study matching the current selection, or None if it has not been measured."""
    if sid is None:
        sid = _active_engine_settings()
    if universe is None:
        meta = st.session_state.get("screener_meta") or {}
        universe = meta.get("universe")
        selected_index = meta.get("selected_index")
        timeframe = meta.get("timeframe", "Daily")
    return _edge_cache_get(_edge_key(universe, selected_index, timeframe, sid))


# CSS kind per verdict rung, so a verdict can never read "success" in one place and
# "danger" in another. edge.VERDICTS is the source of truth.
def _verdict_kind(label: str) -> str:
    return (edge.VERDICTS.get(label) or ("neutral", ""))[0]


def _study_state(study, side: str = "buy") -> tuple:
    """(label, css_kind, detail) for the active universe — MEASURED, or 'not measured'.

    This replaces the old ``_scope_state``, which read a hardcoded per-class constant and
    announced a verdict the app had never actually tested. When no study exists the honest
    answer is that we do not know, not that the asset class is unproven.
    """
    if study is None:
        return ("NOT MEASURED", "neutral",
                "expectancy has not been measured on this universe yet — "
                "it is measured on every run, so this resolves as soon as one completes")
    return study.verdict(side)


def _study_summary_line(study, side: str = "buy") -> str:
    """One-line measured read for a side: 'edge [CI] · n_eff · MDE', or a not-measured note."""
    if study is None:
        return "not measured on this universe"
    r = study.get(side, "holdout") or study.get(side, "full")
    if r is None:
        return "no events measured for this side"
    _ci = (f"[{r.ci_lo:+.3f},{r.ci_hi:+.3f}]" if np.isfinite(r.ci_lo) and np.isfinite(r.ci_hi)
           else "[CI n/a — too few dates]")
    return (f"{r.edge:+.3f} {_ci} vol · {r.hit:.1f}% hit · "
            f"n_eff {r.n_eff:.0f} · resolves ≥{r.mde:.3f}")


def _render_edge_study_panel(sid: eng.EngineSettings, study) -> None:
    """Full Edge Study readout — the numbers behind the verdict, per slice and per era."""
    ui.render_section_header(
        "Edge Study",
        "Measured out-of-sample expectancy of ▲ capitulation / ▼ distribution on the symbols on screen · "
        "re-measured daily",
        icon="activity", accent="violet",
    )
    if study is None:
        ui.render_interpretation_card(
            "Not measured on this universe",
            "The signal set is a fixed, pre-declared rule; whether it carries an edge on THESE "
            f"symbols is a separate empirical question, and the app measures it on every run — an "
            f"event study over ~{_STUDY_YEARS} years through the same engine call the screener "
            "makes, each instrument's own trailing drift removed (causally), vol-normalised, "
            "block-bootstrapped over dates. This one did not complete: either the cross-section "
            "was too thin to measure, or the deep history request came back short (yfinance "
            "rate-limits deep requests from shared cloud IPs). It is retried on the next session.",
            "neutral",
        )
        return

    rows = []
    for key in edge.SLICES:
        for era in ("discovery", "holdout", "full"):
            r = study.get(key, era)
            if r is None:
                continue
            rows.append({
                "Slice": edge.SLICE_LABEL[key], "Era": era.title(),
                "Edge (vol)": r.edge, "CI low": r.ci_lo, "CI high": r.ci_hi,
                "Net": r.net, "Hit %": r.hit,
                "Events": r.n_events, "Dates": r.n_dates,
                "n_eff": r.n_eff, "Resolves ≥": r.mde,
                "Significant": "yes" if r.significant else ("ANTI" if r.anti else "no"),
            })
    if not rows:
        ui_info("The study ran but no event fired in the measured history — every layer must "
                "agree on one bar, so a thin or short universe can go years without a ▲ or ▼.")
        return

    v_buy, v_sell = study.verdict("buy"), study.verdict("sell")
    n = study.counts()
    m1, m2, m3, m4 = st.columns(4)
    with m1: ui.render_metric_card("Long · ▲ + ◆", v_buy[0], _study_summary_line(study, "buy"),
                                   _verdict_kind(v_buy[0]))
    with m2: ui.render_metric_card("Short · ▼ + ◆", v_sell[0], _study_summary_line(study, "sell"),
                                   _verdict_kind(v_sell[0]))
    with m3: ui.render_metric_card("Events", f"{n.get('turn_buy', 0)} ▲ · {n.get('turn_sell', 0)} ▼",
                                   f"{n.get('resume_long', 0) + n.get('resume_short', 0)} RESUME · "
                                   f"{study.fire_rate*1000:.1f} per 1000 bars", "info")
    with m4: ui.render_metric_card("Independence", f"{study.part_ratio:.1f}",
                                   f"of {study.n_symbols_studied} names studied", "info")

    ui.render_table_panel(
        pd.DataFrame(rows), key="edge-results",
        context=f"{study.n_symbols_studied} symbols · {study.start} to {study.end}",
        show_index=False, label_col="Slice", max_height=560,
        col_precision={"Edge (vol)": 4, "CI low": 4, "CI high": 4, "Net": 4,
                       "Hit %": 1, "n_eff": 0, "Resolves \u2265": 4},
        footer=_glossary({
            "Slice": "Long · all and Short · all are the screen's two sides (▲▼ and ◆ "
                     "pooled). The kind rows say which situation carries the pooled number.",
            "Edge (vol)": "Mean drift-free, vol-normalised return following an event. GROSS.",
            "CI low / high": "Block-bootstrap 95% bounds. An edge is claimed only when the low is > 0.",
            "Net": f"Edge minus the cost charge at {sid.cost_bps:.1f} bp, in the same vol units "
                   f"via each instrument's own h-bar sigma.",
            "Hit %": "Share of events where the signal beat that symbol's own drift. 50% is the no-edge line.",
            "n_eff": "Independent observations = (dates / horizon) × participation ratio. Not the "
                     "event count — overlapping returns and a correlated cross-section both reduce it.",
            "Resolves \u2265": "Minimum detectable effect at this power (1.96·\u03c3/\u221an_eff). A "
                          "'no edge' verdict only means anything when this is smaller than the "
                          "effect you would care about.",
        }),
    )

    st.markdown(
        f'<div style="font-family:var(--data); font-size:var(--fs-xs); color:var(--ink-tertiary); '
        f'padding:0.7rem 0 0.1rem 0; line-height:1.6;">'
        f'<b style="color:var(--ink-secondary);">Method.</b> Event study of the pre-declared '
        f'signal set ({html.escape(study.trigger)}, {study.length}-bar conviction lookback, '
        f'{study.horizon}-bar hold, entry the bar after the signal), through the same engine call '
        f'the screener makes — warm-up, the stack gate, the basket gate and the cooldowns all apply. '
        f'Each instrument\'s own mean forward return over the 500 returns already <i>realised</i> '
        f'before the event is removed, so a rising market cannot read as edge and nothing from the '
        f'future leaks in; the residual is divided by that instrument\'s own trailing sigma so asset '
        f'classes are comparable. Confidence intervals come from a block bootstrap over '
        f'<i>dates</i>. Parameters are never tuned here: this measures a fixed rule, it does not '
        f'search for a better one.<br><br>'
        f'<b style="color:var(--ink-secondary);">What the source measured.</b> pragati.pine\'s '
        f'evidence covers conviction\'s components on eleven instruments across four timeframes: '
        f'regular divergence ranked first (not established); participation weighting earns its '
        f'place; the chart-only reversal trigger had no edge (+0.0003R). Nothing reached '
        f'significance once overlapping windows were accounted for (best t = 1.9 of 48 cells). '
        f'The v9 audit (studies/pragati_v9_audit.md) measured the ▲ across 380 instruments: about '
        f'+0.05σ over 10-20 bars in every era outside crypto. This study is the number for YOUR symbols.'
        f'</div>',
        unsafe_allow_html=True,
    )


def ensure_edge_study(universe, selected_index, timeframe, sid,
                      progress_slot=None, progress_offset=0, progress_scale=100):
    """Guarantee a current edge measurement for this selection. Runs on EVERY run.

    Not opt-in. The expectancy of the rule on the universe in front of you is not an optional
    extra — it is the thing that tells you whether to believe the signals — so the app measures
    it as part of every run rather than hiding it behind a checkbox.

    Reuses a same-day measurement (see :func:`_study_is_fresh`: within one calendar day the
    study reads identical data and must return a bit-identical answer, so re-measuring is a
    15-year fetch for a result we already hold). Re-measures automatically once the date rolls.

    A failure is never fatal and is not retried on every click: if the study cannot complete —
    too few usable symbols, or yfinance rate-limiting a deep request from a shared cloud IP —
    the attempt is recorded for the day and the run proceeds with the last measurement if there
    is one, or "not measured" if there is not.
    """
    key = _edge_key(universe, selected_index, timeframe, sid)
    fresh = _edge_cache_get(key, require_fresh=True)
    if fresh is not None:
        console.detail(f"Edge study: reusing today's measurement · "
                       f"{fresh.verdict('buy')[0]} ({fresh.n_symbols_studied} symbols)")
        return fresh

    # Don't re-attempt a failed study on every click within a session.
    failed = st.session_state.setdefault("_edge_failed", {})
    today = _today_ist().strftime("%Y-%m-%d")
    if failed.get(key) == today:
        console.detail("Edge study: already failed today for this selection — not retrying")
        return _edge_cache_get(key)

    study = run_edge_study(universe, selected_index, timeframe, sid,
                           progress_slot=progress_slot,
                           progress_offset=progress_offset, progress_scale=progress_scale)
    if study is None:
        failed[key] = today
        return _edge_cache_get(key)      # fall back to a stale measurement if one exists
    _edge_cache_put(key, study)
    return study


def _study_sample(symbols: list, cap: int = _STUDY_SYMBOL_CAP) -> list:
    """Deterministically sample a large universe down to `cap` symbols.

    Fixed-seed random sampling rather than "first N": taking the head of an NSE constituent
    list would bias the study toward one alphabetical slice (and, since those lists are often
    sector-ordered, toward one sector). Seeded from the symbol set so the same universe always
    yields the same sample — a study that changed answer on every run would be worthless.
    """
    if len(symbols) <= cap:
        return list(symbols)
    seed = abs(hash(frozenset(symbols))) % (2 ** 32)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(symbols), size=cap, replace=False)
    return [symbols[i] for i in sorted(idx)]


def _fetch_study_chunk(symbols: list, start, end, timeframe: str = "Daily"):
    """Deep-history OHLCV for one chunk of symbols. Returns {ticker: frame}.

    Separate from ``fetch_batch_data`` because that path is tuned for the screener: it caps
    history, appends a live intraday bar, and is memoised for 5 minutes. The study wants the
    opposite — long history, completed bars only, no live append (a forming bar has no
    forward return anyway).
    """
    try:
        raw = yf.download(symbols, start=start, end=end, progress=False,
                          auto_adjust=True, group_by="ticker", threads=True)
    except Exception as e:
        console.detail(f"Edge study: chunk fetch failed ({type(e).__name__}: {e})")
        return {}
    if raw is None or (hasattr(raw, "empty") and raw.empty):
        return {}
    try:   # the ladder's intraday rungs, batched and silent (the screen already logged them)
        wanted = _ladder_frames(timeframe)
        idm.prefetch(list(symbols), frames=idm.sources(wanted), report=wanted)
    except Exception:
        pass
    out = {}
    if isinstance(raw, pd.DataFrame) and isinstance(raw.columns, pd.MultiIndex):
        for t in symbols:
            try:
                f = raw.xs(t, level=0, axis=1)
            except KeyError:
                continue
            if f.empty or f["Close"].isnull().all():
                continue
            f = f.dropna(subset=["Close"])
            f.index = pd.to_datetime(f.index)
            if f.index.tz is not None:
                f.index = f.index.tz_convert(None)
            out[t] = f
    elif isinstance(raw, pd.DataFrame) and len(symbols) == 1:
        f = raw.dropna(subset=["Close"])
        f.index = pd.to_datetime(f.index)
        if f.index.tz is not None:
            f.index = f.index.tz_convert(None)
        out[symbols[0]] = f
    return out


def run_edge_study(universe, selected_index, timeframe, sid: eng.EngineSettings,
                   progress_slot=None, progress_offset=0, progress_scale=100):
    """Measure the signal set's out-of-sample expectancy on this universe. Returns an EdgeStudy.

    Streams chunk-by-chunk so peak memory stays a few MB regardless of universe size (see
    the section header). Partial coverage is reported rather than fatal: if a chunk fails to
    fetch — yfinance rate-limits shared cloud IPs — the study proceeds on what arrived and
    flags itself ``partial``.
    """
    def _p(pct, label, sub):
        if progress_slot is not None:
            progress_bar(progress_slot, int(progress_offset + pct * progress_scale / 100),
                         label, sub)

    console.start_phase("EDGE STUDY", 1, 1)
    console.section("Measuring expectancy on this universe")

    all_symbols = _universe_symbols(universe, selected_index)
    if not all_symbols:
        console.error("Edge study: could not resolve the universe")
        return None
    symbols = _study_sample(all_symbols)
    sampled = len(symbols) < len(all_symbols)

    end = _today_ist() + datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=int(_STUDY_YEARS * 365.25))
    console.item("Universe", f"{len(all_symbols)} symbols"
                             + (f" → sampled {len(symbols)}" if sampled else ""))
    console.item("History", f"{start} to {end} (~{_STUDY_YEARS}y)")
    console.item("Parameters", f"lookback {sid.params.length} · norm {sid.norm} · θ {sid.theta:.1f} · "
                               f"{sid.trigger_label} · hold {sid.horizon} · {sid.cost_bps:.1f}bp")

    _p(3, "Measuring Edge", f"{len(symbols)} symbols · ~{_STUDY_YEARS}y")
    # The macro drivers behind the value ingredient — one deep batch for the whole study.
    _study_days = int(_STUDY_YEARS * 365.25)
    drivers = _drivers_for(end, timeframe, days_back=_study_days)
    console.item("Macro drivers", "fetched" if drivers is not None else "unavailable — value runs unhedged")

    events, baselines, ret_cols, bar_counts = [], {}, {}, []
    n_failed_chunks = 0
    chunks = [symbols[i:i + _STUDY_CHUNK] for i in range(0, len(symbols), _STUDY_CHUNK)]

    for ci, chunk in enumerate(chunks):
        _p(3 + (ci / max(len(chunks), 1)) * 82, "Measuring Edge",
           f"chunk {ci + 1}/{len(chunks)} · {len(baselines)} symbols reduced")
        data = _fetch_study_chunk(chunk, start, end, timeframe)
        if not data:
            n_failed_chunks += 1
            continue
        for tkr, f in data.items():
            try:
                daily = f
                if timeframe == "Weekly":
                    f = resample_to_weekly(f)
                if len(f) < sid.min_bars + sid.horizon + 2:
                    continue
                # The exact engine call the screener makes — so the study measures the rule
                # the screen fires, guards and all.
                ev = edge.symbol_events(f, drivers, tkr, sid,
                                        daily=daily if timeframe == "Weekly" else None)
                base = edge.symbol_baseline(f["Close"], sid.horizon)
                if base.empty:
                    continue
                baselines[tkr] = base
                bar_counts.append(len(f))
                ret_cols[tkr] = f["Close"].pct_change().tail(_STUDY_CORR_BARS)
                if not ev.empty:
                    events.append(ev.assign(symbol=tkr))
            except Exception as e:
                console.detail(f"Edge study: {tkr} reduced with error ({type(e).__name__}: {e})")
                continue
        # Release the chunk's frames before fetching the next one — this is what keeps peak
        # memory flat instead of growing with the universe.
        del data

    if len(baselines) < _STUDY_MIN_SYMBOLS:
        console.warning(f"Edge study: only {len(baselines)} symbols usable — "
                        f"cross-section too thin to measure")
        console.end_phase("EDGE STUDY")
        return None

    _p(88, "Measuring Edge", "bootstrapping confidence intervals")
    ev_all = pd.concat(events, ignore_index=True) if events else pd.DataFrame(
        columns=["date", "side", "fwd", "symbol"])
    ret_matrix = pd.DataFrame(ret_cols)

    study = edge.measure(
        ev_all, baselines, ret_matrix,
        universe=universe, selected_index=selected_index, timeframe=timeframe,
        iclass=sid.iclass, length=sid.params.length, trigger=sid.trigger_label,
        horizon=sid.horizon, cost_bps=sid.cost_bps,
        n_symbols_universe=len(all_symbols),
        n_bars_median=int(np.median(bar_counts)) if bar_counts else 0,
        partial=bool(n_failed_chunks),
        measured_at=datetime.datetime.now(_IST).strftime("%Y-%m-%d %H:%M"),
    )
    if sampled:
        study.note = (f"measured on a fixed-seed random sample of {len(baselines)} of "
                      f"{len(all_symbols)} symbols").strip()
    if n_failed_chunks:
        study.note = (study.note + " · " if study.note else "") + \
                     f"{n_failed_chunks} of {len(chunks)} fetch chunks failed"

    for side in edge.SLICES:
        lbl, _kind, detail = study.verdict(side)
        console.item(f"{edge.SLICE_LABEL[side]} verdict", f"{lbl} — {detail}")
    console.item("Coverage", f"{study.n_symbols_studied} symbols · {study.start} to "
                             f"{study.end} · participation ratio {study.part_ratio:.1f}")
    console.item("Fire rate", f"{study.fire_rate*1000:.1f} events per 1000 bars "
                              f"(the Siddhi zero-cross fired ~113)")
    console.end_phase("EDGE STUDY")
    _p(100, "Edge Measured", f"{study.n_symbols_studied} symbols")
    return study


# ══════════════════════════════════════════════════════════════════════════════
# APPEARANCE
# ══════════════════════════════════════════════════════════════════════════════
#: The DURABLE record of the appearance choice — a plain session key, never a
#: widget key.
#:
#: This distinction is the whole fix for a theme that flips back on its own.
#: Streamlit garbage-collects the state of any widget that was NOT instantiated
#: during a run. The appearance control lives at the foot of the rail, so every
#: run that returns or reruns before reaching it — clicking RUN (which executes
#: the analysis inline and re-renders), switching mode, a failed fetch — would
#: discard a widget key entirely, and the next run would fall back to the
#: default. That is the failure mode where a theme survives idle reruns but
#: dies on exactly the actions a user takes, which reads as "entirely buggy"
#: rather than simply broken.
#:
#: A plain key is never collected, so it survives every one of those paths.
_THEME_CHOICE = "theme_choice"

#: The two appearances. Both are reading surfaces — Slate is the dark one you
#: work on, Paper the light one you read a result on and print from.
#:
#: SLATE LEADS, and the order is the default: `theme_choice()` falls back to
#: APPEARANCES[0] for any unset or unrecognised value, so first-in-tuple IS
#: first-run. Kept as one fact rather than a separate DEFAULT_ constant, so the
#: toggle's left-to-right order and the default can never disagree.
#:
#: Slate leads here where Paper leads in Tattva, and .streamlit/config.toml is
#: set to match: a screener is a working surface, and the run console beside it
#: is dark. `base` in that file must agree with whatever leads here, or the
#: FIRST load — before any choice exists — renders Streamlit's own natives for
#: one theme on the other theme's ground.
APPEARANCES = ("Slate", "Paper")


def theme_choice() -> str:
    """The appearance the user last chose, always one of ``APPEARANCES``.

    A value that is not in the list is treated as unset. That matters across a
    rename: a session opened before this list changed still holds the old
    string in the durable key, and handing an unknown option to the segmented
    control as its default is an error rather than a fallback.
    """
    choice = st.session_state.get(_THEME_CHOICE)
    return choice if choice in APPEARANCES else APPEARANCES[0]


# ─── Resolve the theme BEFORE anything is styled ──────────────────────────
# This runs at module scope, which in Streamlit IS the top of the script, and
# therefore before render_sidebar() emits the control that sets it.
#
# The bug it prevents: if `theme` were written by the appearance control down
# in the rail, then on the rerun following a click inject_css() would still see
# the PREVIOUS theme while every chart — which resolves its palette at render
# time, further down the script — already saw the new one. The page renders as
# a mix of both: chrome in one theme, plots in the other, which is exactly
# "some elements show up, some do not". Reading the DURABLE choice here, first,
# makes the whole script agree on one value for the whole run.
st.session_state["theme"] = "light" if theme_choice() == "Paper" else "dark"
inject_css(theme=st.session_state["theme"])


def _render_appearance_control() -> None:
    """The theme switch — LAST control in the rail, deliberately.

    Anywhere higher gives the least consequential switch in the application the
    most valuable position in it.
    """
    with st.container(key="appearance"):
        st.markdown('<div class="sidebar-title">Appearance</div>', unsafe_allow_html=True)
        _mode = st.segmented_control(
            "Appearance", list(APPEARANCES), key="theme_mode",
            default=theme_choice(), label_visibility="collapsed",
            help="Slate — dark, for working. Paper — light, for reading and print.",
        )
        # Mirror the widget into the DURABLE key, and rerun so the stylesheet at
        # the top of the script is re-injected with the new value. Without the
        # rerun the change would land half-way down the page and the run would
        # render as a mix of both themes.
        if _mode is not None and _mode != theme_choice():
            st.session_state[_THEME_CHOICE] = _mode
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & UNIVERSE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

INDEX_LIST = [
    "F&O Stocks",
    # Broad market
    "NIFTY 50", "NIFTY NEXT 50", "NIFTY 100", "NIFTY 200", "NIFTY 500",
    # Midcap
    "NIFTY MIDCAP 50", "NIFTY MIDCAP 100", "NIFTY MIDCAP 150", "NIFTY MID SELECT",
    # Smallcap
    "NIFTY SMLCAP 50", "NIFTY SMLCAP 100", "NIFTY SMLCAP 250",
    # Sectoral
    "NIFTY BANK", "NIFTY PRIVATE BANK", "NIFTY PSU BANK",
    "NIFTY FIN SERVICE",
    "NIFTY IT", "NIFTY AUTO", "NIFTY FMCG", "NIFTY PHARMA",
    "NIFTY METAL", "NIFTY ENERGY", "NIFTY INFRA", "NIFTY REALTY",
    "NIFTY MEDIA",
    # All indexes as instruments
    "Benchmark Indexes",
]

# Broad-market + sectoral index instruments (traded as tickers, not constituents)
BENCHMARK_INDEXES_LIST = [
    # Broad market — NSE
    "^NSEI",           # Nifty 50
    "^NSMIDCP",        # Nifty Next 50
    "NIFTY_100.NS",    # Nifty 100
    "NIFTY_200.NS",    # Nifty 200
    "NIFTY_500.NS",    # Nifty 500
    "^NSEMDCP50",      # Nifty Midcap 50
    "NIFTY_MIDCAP_100.NS",    # Nifty Midcap 100
    "NIFTY_MIDCAP_150.NS",    # Nifty Midcap 150
    "NIFTY_MID_SELECT.NS",    # Nifty Midcap Select
    "NIFTYSMLCAP50.NS",       # Nifty Smallcap 50
    "NIFTY_SMALLCAP_100.NS",  # Nifty Smallcap 100
    "NIFTY_SMALLCAP_250.NS",  # Nifty Smallcap 250
    # Volatility
    "^INDIAVIX",       # India VIX
    # Broad market — BSE
    "^BSESN",          # S&P BSE Sensex
    "BSE-100.BO",      # BSE 100
    "BSE-200.BO",      # BSE 200
    "BSE-500.BO",      # BSE 500
    # Sectoral — NSE
    "^NSEBANK",        # Nifty Bank
    "^CNXFIN",         # Nifty Financial Services
    "^CNXIT",          # Nifty IT
    "^CNXAUTO",        # Nifty Auto
    "^CNXFMCG",        # Nifty FMCG
    "^CNXPHARMA",      # Nifty Pharma
    "^CNXMETAL",       # Nifty Metal
    "^CNXREALTY",      # Nifty Realty
    "^CNXENERGY",      # Nifty Energy
    "^CNXINFRA",       # Nifty Infrastructure
    "^CNXPSUBANK",     # Nifty PSU Bank
    "NIFTY_PRIVATE_BANK.NS",  # Nifty Private Bank
    "^CNXMEDIA",       # Nifty Media
]

BASE_URL = "https://archives.nseindia.com/content/indices/"
INDEX_URL_MAP = {
    "NIFTY 50": f"{BASE_URL}ind_nifty50list.csv",
    "NIFTY NEXT 50": f"{BASE_URL}ind_niftynext50list.csv",
    "NIFTY 100": f"{BASE_URL}ind_nifty100list.csv",
    "NIFTY 200": f"{BASE_URL}ind_nifty200list.csv",
    "NIFTY 500": f"{BASE_URL}ind_nifty500list.csv",
    "NIFTY MIDCAP 50": f"{BASE_URL}ind_niftymidcap50list.csv",
    "NIFTY MIDCAP 100": f"{BASE_URL}ind_niftymidcap100list.csv",
    "NIFTY MIDCAP 150": f"{BASE_URL}ind_niftymidcap150list.csv",
    "NIFTY MID SELECT": f"{BASE_URL}ind_niftymidcapselectlist.csv",
    "NIFTY SMLCAP 50":  f"{BASE_URL}ind_niftysmallcap50list.csv",
    "NIFTY SMLCAP 100": f"{BASE_URL}ind_niftysmallcap100list.csv",
    "NIFTY SMLCAP 250": f"{BASE_URL}ind_niftysmallcap250list.csv",
    "NIFTY BANK": f"{BASE_URL}ind_niftybanklist.csv",
    "NIFTY PRIVATE BANK": f"{BASE_URL}ind_niftypvtbanklist.csv",
    "NIFTY PSU BANK": f"{BASE_URL}ind_niftypsubanklist.csv",
    "NIFTY AUTO": f"{BASE_URL}ind_niftyautolist.csv",
    "NIFTY FIN SERVICE": f"{BASE_URL}ind_niftyfinancelist.csv",
    "NIFTY FMCG": f"{BASE_URL}ind_niftyfmcglist.csv",
    "NIFTY IT": f"{BASE_URL}ind_niftyitlist.csv",
    "NIFTY PHARMA": f"{BASE_URL}ind_niftypharmalist.csv",
    "NIFTY METAL": f"{BASE_URL}ind_niftymetallist.csv",
    "NIFTY ENERGY": f"{BASE_URL}ind_niftyenergylist.csv",
    "NIFTY INFRA": f"{BASE_URL}ind_niftyinfrastructurelist.csv",
    "NIFTY REALTY": f"{BASE_URL}ind_niftyrealtylist.csv",
    "NIFTY MEDIA": f"{BASE_URL}ind_niftymedialist.csv",
}

WIKI_URL_MAP = {
    "NIFTY 50": "https://en.wikipedia.org/wiki/NIFTY_50",
    "NIFTY NEXT 50": "https://en.wikipedia.org/wiki/NIFTY_Next_50",
    "NIFTY BANK": "https://en.wikipedia.org/wiki/NIFTY_Bank",
    "NIFTY IT": "https://en.wikipedia.org/wiki/NIFTY_IT",
    "NIFTY FIN SERVICE": "https://en.wikipedia.org/wiki/Nifty_Financial_Services_Index",
}

UNIVERSE_OPTIONS = ["India Indexes", "Global Indexes", "US Indexes", "ETF Index", "Commodities", "Currency", "Crypto", "Global Macro"]
TIMEFRAME_OPTIONS = ["Daily", "Weekly"]

# ETF Universe (from Pragyam)
ETF_LIST = [
    "CHEMICAL.NS", "NIFTYIETF.NS", "MON100.NS", "MAKEINDIA.NS", "SILVERIETF.NS",
    "HEALTHIETF.NS", "CONSUMIETF.NS", "GOLDIETF.NS", "INFRAIETF.NS", "CPSEETF.NS",
    "TNIDETF.NS", "COMMOIETF.NS", "MODEFENCE.NS", "MOREALTY.NS", "PSUBNKIETF.NS",
    "MASPTOP50.NS", "FMCGIETF.NS", "GROWWPOWER.NS", "ITIETF.NS", "EVINDIA.NS",
    "MNC.NS", "FINIETF.NS", "AUTOIETF.NS", "PVTBANIETF.NS", "MONIFTY500.NS",
    "ECAPINSURE.NS", "MIDCAPIETF.NS", "MOSMALL250.NS", "OILIETF.NS", "METALIETF.NS"
]

# US Index list
US_INDEX_LIST = ["S&P 500", "DOW JONES", "NASDAQ 100"]

# Hardcoded DOW 30 fallback (as of late 2024 — used only when Wikipedia is unreachable)
_DOW30_FALLBACK = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA",  "CAT", "CRM", "CSCO", "CVX", "DIS",
    "DOW",  "GS",   "HD",   "HON", "IBM",  "JNJ", "JPM", "KO",   "MCD", "MRK",
    "MSFT", "NKE",  "NVDA", "PG",  "SHW",  "TRV", "UNH", "V",    "VZ",  "WMT",
]

# Commodities list (Yahoo Finance) — Expanded from Pragyam
COMMODITY_MAP = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Platinum": "PL=F",
    "Palladium": "PA=F",
    "Copper": "HG=F",
    "Crude Oil WTI": "CL=F",
    "Brent Crude": "BZ=F",
    "Natural Gas": "NG=F",
    "Gasoline RBOB": "RB=F",
    "Heating Oil": "HO=F",
    "Corn": "ZC=F",
    "Wheat": "ZW=F",
    "Soybeans": "ZS=F",
    "Soybean Meal": "ZM=F",
    "Soybean Oil": "ZL=F",
    "Cotton": "CT=F",
    "Coffee": "KC=F",
    "Sugar": "SB=F",
    "Cocoa": "CC=F",
    "Orange Juice": "OJ=F",
    "Lumber": "LBS=F",
    "Live Cattle": "LE=F",
    "Lean Hogs": "HE=F",
    "Feeder Cattle": "GF=F",
}
COMMODITY_LIST = list(COMMODITY_MAP.keys())

# Currency pairs (Yahoo Finance) — Expanded from Pragyam
CURRENCY_MAP = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "USD/CHF": "USDCHF=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X",
    "NZD/USD": "NZDUSD=X",
    "USD/INR": "USDINR=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "AUD/JPY": "AUDJPY=X",
    "EUR/CHF": "EURCHF=X",
    "EUR/AUD": "EURAUD=X",
    "GBP/CHF": "GBPCHF=X",
    "GBP/AUD": "GBPAUD=X",
    "USD/SGD": "USDSGD=X",
    "USD/HKD": "USDHKD=X",
    "USD/CNH": "USDCNH=X",
    "USD/ZAR": "USDZAR=X",
    "USD/MXN": "USDMXN=X",
    "USD/TRY": "USDTRY=X",
    "USD/BRL": "USDBRL=X",
    "USD/KRW": "USDKRW=X",
}
CURRENCY_LIST = list(CURRENCY_MAP.keys())

# Crypto universe (Yahoo Finance)
CRYPTO_MAP = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
    "Binance Coin": "BNB-USD",
    "Ripple (XRP)": "XRP-USD",
    "Cardano": "ADA-USD",
    "Dogecoin": "DOGE-USD",
    "Tron": "TRX-USD",
    "Chainlink": "LINK-USD",
    "Polkadot": "DOT-USD",
    "Polygon (POL)": "POL-USD",
    "Litecoin": "LTC-USD",
    "Bitcoin Cash": "BCH-USD",
    "Shiba Inu": "SHIB-USD",
    "Avalanche": "AVAX-USD",
    "Near Protocol": "NEAR-USD",
    "Uniswap": "UNI-USD",
    "Stellar": "XLM-USD",
    "Ethereum Classic": "ETC-USD",
    "Monero": "XMR-USD",
    "Cosmos": "ATOM-USD"
}
CRYPTO_LIST = list(CRYPTO_MAP.keys())

# Global Macro Bond ETF Universe — proxy for global yield dynamics via yfinance-available instruments
GLOBAL_MACRO_MAP = {
    # ── US Treasuries (Full Curve) ─────────────────────────────────────────────
    "US Treasury 1-3 Month":             "BIL",
    "US Treasury Ultra-Short (0-1Y)":    "SHV",
    "US Treasury 0-3 Month (SGOV)":      "SGOV",
    "US Treasury Short (1-3Y)":          "SHY",
    "US Treasury Short (1-3Y) Vanguard": "VGSH",
    "US Treasury Intermediate (3-7Y)":   "IEI",
    "US Treasury Intermediate (7-10Y)":  "IEF",
    "US Treasury Intermediate Vanguard": "VGIT",
    "US Treasury Long (10-20Y)":         "TLH",
    "US Treasury Long (20Y+)":           "TLT",
    "US Treasury Long Vanguard":         "VGLT",
    "US Treasury Total Market":          "GOVT",
    # ── Direct Yield Indices (Raw %) ──────────────────────────────────────────
    "US 13-Week T-Bill Yield":           "^IRX",
    "US 5-Year Treasury Yield":          "^FVX",
    "US 10-Year Treasury Yield":         "^TNX",
    "US 30-Year Treasury Yield":         "^TYX",
    # ── Inflation-Protected (TIPS) ─────────────────────────────────────────────
    "US TIPS Broad Market":              "TIP",
    "US TIPS Short-Term":                "VTIP",
    "International Govt Inflation-Linked": "WIP",
    # ── Aggregate / Multi-Sector ───────────────────────────────────────────────
    "US Core Aggregate Bond":            "AGG",
    "US Total Bond Market":              "BND",
    "US Floating Rate Notes":            "FLOT",
    "Global Aggregate Bond (Hedged)":    "BNDW",
    "Total International Bond (ex-US)":  "BNDX",
    # ── US Corporate: Investment Grade ────────────────────────────────────────
    "US Corporate Investment Grade":     "LQD",
    "US Corporate Short-Term (1-5Y)":    "VCSH",
    "US Corporate Intermediate":         "VCIT",
    "US Corporate Long-Term":            "VCLT",
    # ── High Yield & Alternative Credit ───────────────────────────────────────
    "US High Yield Corporate":           "HYG",
    "US High Yield Corporate SPDR":      "JNK",
    "Global High Yield Bond":            "GHYG",
    "Global Green Bond":                 "BGRN",
    "Preferred Stock (Hybrid)":          "PFF",
    "Convertible Bonds":                 "CWB",
    "Fallen Angels (Recent HY)":         "FALN",
    # ── Structured & Asset-Backed ─────────────────────────────────────────────
    "US Mortgage-Backed Securities":     "MBB",
    "US Mortgage-Backed Vanguard":       "VMBS",
    "US Senior Loan (Floating Rate)":    "BKLN",
    # ── Municipal Bonds ───────────────────────────────────────────────────────
    "US Municipal National":             "MUB",
    "US Municipal Tax-Exempt Vanguard":  "VTEB",
    # ── Developed Markets Sovereign (Europe) ─────────────────────────────────
    "International Treasury (ex-US)":    "IGOV",
    "International Treasury SPDR":       "BWX",
    "International Corporate Bonds":     "IBND",
    "Eurozone Government Bond":          "IEGA.L",
    "Eurozone Corporate Bond (IG)":      "IEAC.L",
    "Germany Govt Bonds (Bunds/Long)":   "BUNL.L",
    "Germany Short-Term (Schatz)":       "SDEU.L",
    "UK Gilts":                          "IGLT.L",
    "UK Gilts (Inflation-Linked)":       "INXG.L",
    "UK Corporate Bonds":                "SLXX.L",
    # ── Developed Markets Sovereign (Asia-Pacific) ────────────────────────────
    "Japan Government Bonds (Broad)":    "JGBL.L",
    "Australia Government Bonds":        "VGB.AX",
    "Canada Broad Aggregate Bond":       "XBB.TO",
    # ── India Fixed Income ────────────────────────────────────────────────────
    "India Gov Bonds (LSE Proxy)":       "IIND.L",
    "India 8-13Y G-Sec":                 "LTGILTBEES.NS",
    "India 5Y G-Sec":                    "GILT5YBEES.NS",
    "India AAA PSU Bond (Bharat 2030)":  "EBBETF0430.NS",
    "India Overnight Rate (Liquid)":     "LIQUIDBEES.NS",
    # ── Emerging Markets ──────────────────────────────────────────────────────
    "EM Sovereign Debt (USD)":           "EMB",
    "EM Sovereign Debt USD Invesco":     "PCY",
    "EM Sovereign (Local Currency)":     "EMLC",
    "EM High Yield Corporate":           "EMHY",
    "China Government Bonds":            "CBON",
    "China CNY Local Bonds":             "CNYB.L",
    # ── Broad Duration Proxies ────────────────────────────────────────────────
    "Short-Term Broad Bond":             "BSV",
    "Long-Term Broad Bond":              "BLV",
}

# Global Benchmark Indexes Universe — primary national equity index per country.
# Futures proxies used where the cash index is not available on Yahoo Finance.
GLOBAL_INDEXES_MAP = {
    # ── North America ──────────────────────────────────────────────────────────
    "S&P 500 (USA)":                     "^GSPC",
    "Dow Jones (USA)":                   "^DJI",
    "NASDAQ 100 (USA)":                  "^NDX",
    "Russell 2000 (USA)":                "^RUT",
    "TSX Composite (Canada)":            "^GSPTSE",
    "IPC (Mexico)":                      "^MXX",
    "Bovespa (Brazil)":                  "^BVSP",
    "Merval (Argentina)":                "^MERV",
    "IPSA (Chile)":                      "^IPSA",
    "COLCAP (Colombia)":                 "^COLCAP",
    # ── Europe ─────────────────────────────────────────────────────────────────
    "FTSE 100 (UK)":                     "^FTSE",
    "DAX (Germany)":                     "^GDAXI",
    "CAC 40 (France)":                   "^FCHI",
    "IBEX 35 (Spain)":                   "^IBEX",
    "FTSE MIB (Italy)":                  "FTSEMIB.MI",
    "AEX (Netherlands)":                 "^AEX",
    "SMI (Switzerland)":                 "^SSMI",
    "OMX Stockholm 30 (Sweden)":         "^OMXS30",
    "Oslo Bors All-Share (Norway)":      "^OSEAX",
    "OMX Copenhagen 25 (Denmark)":       "^OMXC25",
    "ATX (Austria)":                     "^ATX",
    "BEL 20 (Belgium)":                  "^BFX",
    "WIG 20 (Poland)":                   "^WIG20",
    "BIST 100 (Turkey)":                 "XU100.IS",
    "PSI 20 (Portugal)":                 "^PSI20",
    "ASE General (Greece)":              "^ATG",
    "OMX Helsinki 25 (Finland)":         "^OMXH25",
    "PX (Czech Republic)":               "^PX",
    "BUX (Hungary)":                     "^BUX",
    "MOEX (Russia)":                     "IMOEX.ME",
    # ── Asia-Pacific ───────────────────────────────────────────────────────────
    "Nikkei 225 (Japan)":                "^N225",
    "TOPIX (Japan)":                     "^TOPX",
    "Shanghai Composite (China)":        "000001.SS",
    "CSI 300 (China)":                   "000300.SS",
    "Hang Seng (Hong Kong)":             "^HSI",
    "KOSPI (South Korea)":               "^KS11",
    "KOSDAQ (South Korea)":              "^KQ11",
    "TAIEX (Taiwan)":                    "^TWII",
    "Nifty 50 (India)":                  "^NSEI",
    "Sensex (India)":                    "^BSESN",
    "ASX 200 (Australia)":               "^AXJO",
    "All Ordinaries (Australia)":        "^AORD",
    "STI (Singapore)":                   "^STI",
    "KLCI (Malaysia)":                   "^KLSE",
    "SET Composite (Thailand)":          "^SET",
    "Jakarta Composite (Indonesia)":     "^JKSE",
    "PSEi (Philippines)":                "PSEi.PS",
    "NZX 50 (New Zealand)":              "^NZ50",
    "VN-Index (Vietnam)":                "^VNINDEX",
    "KSE 100 (Pakistan)":                "^KSE",
    # ── Middle East & Africa ───────────────────────────────────────────────────
    "TA-125 (Israel)":                   "^TA125.TA",
    "Tadawul (Saudi Arabia)":            "^TASI.SR",
    "DFM General (UAE)":                 "^DFMGI",
    "QE Index (Qatar)":                  "^QSI",
    "JSE All-Share (South Africa)":      "J203.JO",
    "EGX 30 (Egypt)":                    "^CASE",
}

# Asset Name Lookup for friendly display (Reverse map tickers to names)
ASSET_NAME_LOOKUP = {v: k for k, v in {**COMMODITY_MAP, **CURRENCY_MAP, **CRYPTO_MAP, **GLOBAL_MACRO_MAP, **GLOBAL_INDEXES_MAP}.items()}

# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _dedupe_preserve_order(items):
    """Return items with duplicates removed, keeping first-seen order."""
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def get_fno_stock_list():
    """Fetch F&O eligible stocks from NSE with multiple fallback sources."""
    # ── Source 0: NseKit (preferred) ──────────────────────────────────────────
    # Uses NSE's official "underlying-information" API (the authoritative F&O
    # underlyings master), not the equity-stockIndices index view. No index
    # aggregate header row, and NseKit handles NSE's cookie/session warmup itself,
    # which tends to survive datacenter-IP blocking better. Lazy-imported so a
    # missing/broken package simply falls through to the legacy sources below.
    try:
        from NseKit import NseKit
        symbols = NseKit.Nse().nse_eom_fno_full_list(list_only=True)
        if symbols:
            symbols_ns = _dedupe_preserve_order(
                [str(s).strip() + ".NS" for s in symbols if s and str(s).strip()]
            )
            if symbols_ns:
                return symbols_ns, f"✓ Fetched {len(symbols_ns)} F&O securities (NseKit)"
    except Exception as e:
        console.detail(f"F&O source 0 (NseKit) failed: {type(e).__name__}: {e}")

    try:
        url = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nseindia.com/market-data/live-equity-market?symbol=NIFTY%20FIN%20SERVICE',
        }

        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)

        response = session.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                symbols = [item['symbol'] for item in data['data'] if 'symbol' in item]
                # Skip the first entry — equity-stockIndices always returns the index
                # aggregate row as data[0], not a constituent (same as get_index_stock_list).
                symbols = [s for s in symbols[1:] if s and str(s).strip()]
                if symbols:
                    symbols_ns = _dedupe_preserve_order([str(s) + ".NS" for s in symbols])
                    return symbols_ns, f"✓ Fetched {len(symbols_ns)} F&O securities"
    except Exception as e:
        console.detail(f"F&O source 1 (NSE JSON) failed: {type(e).__name__}: {e}")

    try:
        # NOTE: nse_get_advances_declines() hits the SAME "SECURITIES IN F&O" endpoint
        # as source 1 (the name is misleading); it's a redundant retry via nsepython's
        # session handling, and its data[0] is likewise the index aggregate row.
        stock_data = nse_get_advances_declines()
        if isinstance(stock_data, pd.DataFrame) and not stock_data.empty:
            symbols = None
            if 'SYMBOL' in stock_data.columns:
                symbols = stock_data['SYMBOL'].tolist()
            elif 'symbol' in stock_data.columns:
                symbols = stock_data['symbol'].tolist()
            elif len(stock_data.index) > 0 and not isinstance(stock_data.index, pd.RangeIndex):
                symbols = stock_data.index.tolist()

            if symbols:
                # Drop the leading index aggregate row, same as source 1.
                symbols = [s for s in symbols[1:] if s and str(s).strip()]
                symbols_ns = _dedupe_preserve_order([str(s) + ".NS" for s in symbols])
                if symbols_ns:
                    return symbols_ns, f"✓ Fetched {len(symbols_ns)} F&O securities"
    except Exception as e:
        console.detail(f"F&O source 2 (advances/declines) failed: {type(e).__name__}: {e}")

    try:
        # Last-resort fallback. NOTE: NIFTY 500 is a DIFFERENT, ~2.5x larger universe
        # than the ~220 F&O securities (it is a superset that contains them). Surfaced
        # with an explicit ⚠ so the user knows the screened universe is not pure F&O.
        url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, verify=False, timeout=10)
        if response.status_code == 200:
            csv_file = io.StringIO(response.text)
            stock_df = pd.read_csv(csv_file)
            symbol_col = next((c for c in stock_df.columns if str(c).strip().lower() == 'symbol'), None)
            if symbol_col:
                symbols = stock_df[symbol_col].tolist()
                symbols_ns = _dedupe_preserve_order(
                    [str(s) + ".NS" for s in symbols if s and str(s).strip()]
                )
                return symbols_ns, (f"⚠ F&O endpoint unavailable — using NIFTY 500 superset "
                                    f"({len(symbols_ns)} stocks, not pure F&O)")
    except Exception as e:
        console.detail(f"F&O source 3 (NSE archive CSV) failed: {type(e).__name__}: {e}")

    return None, "Failed to fetch F&O list from all sources"


def get_index_stock_list(index):
    if index == "F&O Stocks":
        return get_fno_stock_list()

    if index == "Benchmark Indexes":
        return BENCHMARK_INDEXES_LIST, f"✓ Loaded {len(BENCHMARK_INDEXES_LIST)} benchmark index instruments"

    # --- Source 1: NSE JSON API (most reliable, same endpoint as F&O) ---
    try:
        import urllib.parse
        api_url = f"https://www.nseindia.com/api/equity-stockIndices?index={urllib.parse.quote(index)}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nseindia.com/market-data/live-equity-market',
        }
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        response = session.get(api_url, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                symbols = [item['symbol'] for item in data['data'] if 'symbol' in item]
                # Skip the first entry — it's always the index itself, not a constituent
                symbols = [s for s in symbols[1:] if s and str(s).strip()]
                if symbols:
                    symbols_ns = [str(s) + ".NS" for s in symbols]
                    return symbols_ns, f"✓ Fetched {len(symbols_ns)} constituents (NSE API)"
    except Exception as e:
        console.detail(f"Index source 1 (NSE JSON API) failed for '{index}': {type(e).__name__}: {e}")

    # --- Source 2: NSE archives CSV ---
    # NSE is migrating its archive host from archives.nseindia.com to the newer
    # nsearchives.nseindia.com. Try both so the fallback keeps working if either
    # host is retired or blocked; the static-file hosts are rarely IP-blocked.
    url = INDEX_URL_MAP.get(index)
    if url:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        }
        for host in ("archives.nseindia.com", "nsearchives.nseindia.com"):
            candidate_url = re.sub(r"https://[^/]+", f"https://{host}", url)
            try:
                session = requests.Session()
                session.get(f"https://{host}", headers=headers, verify=False, timeout=10)
                response = session.get(candidate_url, headers=headers, verify=False, timeout=15)
                if response.status_code == 403:
                    # Some egress paths are refused on the full browser string and accepted
                    # on the bare token — retry once with it before moving on.
                    response = session.get(candidate_url, headers={**headers, 'User-Agent': 'Mozilla/5.0'},
                                           verify=False, timeout=15)
                response.raise_for_status()
                stock_df = pd.read_csv(io.StringIO(response.text))
                symbol_col = next((c for c in stock_df.columns if c.lower() == 'symbol'), None)
                if symbol_col:
                    symbols = stock_df[symbol_col].tolist()
                    symbols_ns = _dedupe_preserve_order(
                        [str(s) + ".NS" for s in symbols if s and str(s).strip()]
                    )
                    if symbols_ns:
                        return symbols_ns, f"✓ Fetched {len(symbols_ns)} constituents (NSE archive · {host})"
            except Exception as e:
                console.detail(f"Index source 2 (NSE archive CSV · {host}) failed for '{index}': {type(e).__name__}: {e}")

    # --- Source 3: Wikipedia fallback ---
    wiki_result = _fetch_index_from_wikipedia(index)
    if wiki_result[0]:
        return wiki_result

    return None, f"Could not fetch constituents for '{index}'. NSE API, archive CSV, and Wikipedia all failed."


def _fetch_index_from_wikipedia(index):
    wiki_url = WIKI_URL_MAP.get(index)
    if not wiki_url:
        return None, f"No Wikipedia fallback for {index}"
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(wiki_url, headers=headers, timeout=15)
        response.raise_for_status()
        tables = pd.read_html(io.StringIO(response.text))
        for table in tables:
            cols_lower = [str(c).lower() for c in table.columns]
            symbol_col = None
            for candidate in ('symbol', 'ticker', 'nse code', 'code'):
                for i, c in enumerate(cols_lower):
                    if candidate in c:
                        symbol_col = table.columns[i]
                        break
                if symbol_col is not None:
                    break
            if symbol_col is None:
                continue
            symbols = [str(s).strip() for s in table[symbol_col].dropna().tolist()]
            symbols_ns = [s + ".NS" for s in symbols if s and s.lower() != 'nan']
            if symbols_ns:
                return symbols_ns, f"✓ Fetched {len(symbols_ns)} constituents (Wikipedia fallback)"
        return None, "No symbol table found on Wikipedia page"
    except Exception as e:
        return None, f"Wikipedia fallback error: {e}"


def _fetch_us_index_from_wikipedia(index_name):
    """Scrape constituent tickers for a US index from Wikipedia."""
    wiki_urls = {
        "S&P 500":    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "NASDAQ 100": "https://en.wikipedia.org/wiki/Nasdaq-100",
        "DOW JONES":  "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
    }
    url = wiki_urls.get(index_name)
    if not url:
        return None, f"No Wikipedia URL configured for {index_name}"
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        tables = pd.read_html(io.StringIO(response.text))
        for table in tables:
            cols_lower = [str(c).lower() for c in table.columns]
            symbol_col = None
            for candidate in ('symbol', 'ticker'):
                for i, c in enumerate(cols_lower):
                    if candidate in c:
                        symbol_col = table.columns[i]
                        break
                if symbol_col is not None:
                    break
            if symbol_col is None:
                continue
            raw = [str(s).strip() for s in table[symbol_col].dropna().tolist()]
            # Normalise BRK.B → BRK-B style; drop header echoes and junk rows
            symbols = []
            for s in raw:
                s = s.replace('.', '-')
                if s and s.lower() not in ('symbol', 'ticker', 'nan') and 1 <= len(s) <= 6:
                    symbols.append(s)
            if len(symbols) >= 10:
                return symbols, f"✓ Fetched {len(symbols)} constituents (Wikipedia)"
        return None, "No valid symbol table found on Wikipedia page"
    except Exception as e:
        return None, f"Wikipedia fetch error: {e}"


def get_us_index_symbols(index_name):
    """Get constituent stock tickers for a US index.

    Primary source: Wikipedia scrape. Fallback: hardcoded list for DOW JONES.
    Returns plain NYSE/NASDAQ tickers (no exchange suffix).
    """
    symbols, msg = _fetch_us_index_from_wikipedia(index_name)
    if symbols:
        return symbols, msg
    if index_name == "DOW JONES":
        return _DOW30_FALLBACK.copy(), f"✓ Loaded {len(_DOW30_FALLBACK)} DOW constituents (hardcoded fallback)"
    return None, f"Could not fetch constituents for '{index_name}': {msg}"


def get_global_macro_symbols():
    """Return the Global Macro bond ETF universe."""
    symbols = list(GLOBAL_MACRO_MAP.values())
    return symbols, f"✓ Loaded {len(symbols)} Global Macro instruments"


def get_global_index_symbols():
    """Return the Global Indexes universe — one benchmark index per country."""
    symbols = list(GLOBAL_INDEXES_MAP.values())
    return symbols, f"✓ Loaded {len(symbols)} global benchmark indexes"


def get_commodity_symbols(commodity_type=None):
    """Get commodity futures symbols."""
    if commodity_type is None:
        return list(COMMODITY_MAP.values()), f"✓ Fetched {len(COMMODITY_MAP)} commodities"
    symbol = COMMODITY_MAP.get(commodity_type)
    if symbol:
        return [symbol], f"✓ Fetched {commodity_type}"
    return None, f"Unknown commodity: {commodity_type}"


def get_currency_symbols(currency_pair=None):
    """Get currency pair symbols."""
    if currency_pair is None:
        return list(CURRENCY_MAP.values()), f"✓ Fetched {len(CURRENCY_MAP)} currency pairs"
    symbol = CURRENCY_MAP.get(currency_pair)
    if symbol:
        return [symbol], f"✓ Fetched {currency_pair}"
    return None, f"Unknown currency pair: {currency_pair}"


def get_crypto_symbols(crypto_name=None):
    """Get cryptocurrency symbols."""
    if crypto_name is None:
        return list(CRYPTO_MAP.values()), f"✓ Fetched {len(CRYPTO_MAP)} digital assets"
    symbol = CRYPTO_MAP.get(crypto_name)
    if symbol:
        return [symbol], f"✓ Fetched {crypto_name}"
    return None, f"Unknown crypto asset: {crypto_name}"


def get_etf_symbols():
    """Return the fixed ETF universe for analysis"""
    return ETF_LIST, f"✓ Loaded {len(ETF_LIST)} ETFs"


def resolve_universe(universe, selected_index):
    """Universe selection → (symbols, message). Single dispatch for every analysis path.

    The screener, the range harvest, correlation and the edge study must all study the SAME
    symbols for a given selection, or a measured expectancy would describe a different set
    than the one on screen.
    """
    if universe == "India Indexes":
        return get_index_stock_list(selected_index)
    if universe == "Global Indexes":
        return get_global_index_symbols()
    if universe == "US Indexes":
        return get_us_index_symbols(selected_index)
    if universe == "Commodities":
        return get_commodity_symbols(None)
    if universe == "Currency":
        return get_currency_symbols(None)
    if universe == "Crypto":
        return get_crypto_symbols(None)
    if universe == "ETF Index":
        return get_etf_symbols()
    if universe == "Global Macro":
        return get_global_macro_symbols()
    return None, f"Unknown universe: {universe}"


def _universe_symbols(universe, selected_index):
    """Just the symbol list (no message), or None. Used by the edge study."""
    syms, _msg = resolve_universe(universe, selected_index)
    return list(syms) if syms else None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_batch_data(stock_list, end_date=None, days_back=300, include_live=True):
    if end_date is None:
        end_date = _today_ist()

    download_end = end_date + datetime.timedelta(days=5)
    start_date = end_date - datetime.timedelta(days=days_back + 365)

    try:
        all_data = yf.download(
            stock_list,
            start=start_date,
            end=download_end,
            progress=False,
            auto_adjust=True,
            group_by='ticker',
            threads=True,
        )
        
        if all_data.empty:
            return None, "No data returned"
            
        _ohlc_cols = ['Open', 'High', 'Low', 'Close']

        def _clean_ticker_df(tdf):
            """Drop rows where all core OHLC columns are NaN; keep rows with partial data."""
            core = [c for c in _ohlc_cols if c in tdf.columns]
            if core:
                tdf = tdf.dropna(subset=core, how='all')
            return tdf

        if isinstance(all_data, pd.DataFrame) and isinstance(all_data.columns, pd.MultiIndex):
            data_dict = {}
            for ticker in stock_list:
                try:
                    ticker_df = all_data.xs(ticker, level=0, axis=1)
                    if not ticker_df.empty and not ticker_df['Close'].isnull().all():
                        data_dict[ticker] = _clean_ticker_df(ticker_df.copy())
                except KeyError:
                    pass
        elif isinstance(all_data, dict):
            data_dict = {t: _clean_ticker_df(df.copy()) for t, df in all_data.items()
                         if not df.empty and not df['Close'].isnull().all()}
        else:
             return None, "Unexpected data structure"

        if include_live and end_date == _today_ist() and data_dict:
            sample_df = list(data_dict.values())[0]
            sample_df.index = pd.to_datetime(sample_df.index)
            if sample_df.index.tz is not None:
                sample_df.index = sample_df.index.tz_convert(None)

            _ist_today = _today_ist()
            # NOTE: `sample_df` is only the first ticker — used as a cheap hint for
            # whether a live append is worth attempting. The actual today-already-present
            # check is done PER TICKER below, by calendar date, because (a) tickers can be
            # heterogeneous (some already have today's bar, some not) and (b) yfinance live
            # 1d bars are stamped with an intraday time while historical daily bars are
            # stamped 00:00:00 — an exact-timestamp .difference() would therefore append a
            # SECOND "today" row next to the 00:00 one, double-counting today and shifting
            # every rolling window (including the oscillator) by a bar.
            _hint_has_today = any(idx.date() == _ist_today for idx in sample_df.index)
            if not _hint_has_today:
                try:
                    live_data = yf.download(list(data_dict.keys()), period="1d", progress=False, auto_adjust=True, group_by='ticker')
                    if not live_data.empty:
                        for ticker in data_dict.keys():
                            try:
                                live_ticker = live_data.xs(ticker, level=0, axis=1)
                                if not live_ticker.empty and not live_ticker['Close'].isnull().all():
                                    hist_df = data_dict[ticker]
                                    hist_df.index = pd.to_datetime(hist_df.index)
                                    if hist_df.index.tz is not None: hist_df.index = hist_df.index.tz_convert(None)
                                    live_ticker.index = pd.to_datetime(live_ticker.index)
                                    if live_ticker.index.tz is not None: live_ticker.index = live_ticker.index.tz_convert(None)
                                    # Normalize the live bar to midnight and keep only calendar
                                    # dates not already present in history — date-based, so an
                                    # intraday-stamped live bar can't duplicate a 00:00 daily bar.
                                    live_norm = live_ticker.copy()
                                    live_norm.index = live_norm.index.normalize()
                                    hist_dates = set(hist_df.index.normalize())
                                    keep = live_norm[~live_norm.index.isin(hist_dates)]
                                    if len(keep) > 0:
                                        data_dict[ticker] = pd.concat([hist_df, keep]).sort_index()
                            except KeyError:
                                pass
                except Exception as e:
                    console.detail(f"Live-data append failed: {type(e).__name__}: {e}")
        return data_dict, f"✓ Downloaded {len(data_dict)} tickers"
    except Exception as e:
        return None, f"Download error: {e}"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_macro_drivers(end_date, days_back):
    """Daily closes of the value engine's macro drivers — one batch per (date, depth).

    Samanvaya's hedge basket (samanvaya.DRIVERS): US yields, bond-ETF proxies for the
    other 10-year yields, the dollar index, energy, metals, the INR crosses and the home
    equity indices. Shared by every name in the universe; each name aligns them to its
    own calendar and close time.

    Returns None rather than raising. The value engine runs without drivers — the RV leg
    becomes the name's own path, the Pine's "Macro hedge: Off" — so a failed fetch
    degrades the value reading, it does not end the run.
    """
    start = end_date - datetime.timedelta(days=int(days_back) + 365)
    try:
        raw = yf.download(sv.DRIVER_TICKERS, start=start, end=end_date + datetime.timedelta(days=5),
                          progress=False, auto_adjust=True, threads=True)
    except Exception as e:
        console.detail(f"Macro drivers unavailable ({type(e).__name__}: {e}) — value runs unhedged")
        return None
    if raw is None or getattr(raw, "empty", True):
        console.detail("Macro drivers returned nothing — value runs unhedged")
        return None
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = pd.DataFrame(close).dropna(how="all", axis=1)
    close.index = pd.to_datetime(close.index)
    if close.index.tz is not None:
        close.index = close.index.tz_convert(None)
    missing = [t for t in sv.DRIVER_TICKERS if t not in close.columns]
    console.detail(f"Macro drivers · {close.shape[1]} of {len(sv.DRIVER_TICKERS)} series · "
                   f"{len(close)} bars" + (f" · missing {', '.join(missing)}" if missing else ""))
    return close if not close.empty else None


def _drivers_for(end_date, timeframe, days_back=None):
    """The macro drivers prepared on the chart's calendar, or None when unavailable."""
    closes = fetch_macro_drivers(end_date, int(days_back or _max_days_back(timeframe)))
    if closes is None:
        return None
    return sv.prepare_drivers(closes, eng.chart_of(timeframe))


def resample_to_weekly(df):
    if df is None or df.empty:
        return df
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    weekly_raw = df.resample('W-MON', closed='left', label='left').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    weekly = weekly_raw.dropna()
    dropped = len(weekly_raw) - len(weekly)
    if dropped > 0:
        console.detail(f"resample_to_weekly: dropped {dropped} incomplete week(s) with NaN OHLCV")
    return weekly


def _slug(value) -> str:
    """Sanitize a string for use in a filename. Returns 'na' for empty/None inputs."""
    if value is None:
        return "na"
    s = str(value).strip().lower()
    if not s:
        return "na"
    # Collapse non-[A-Za-z0-9_-] runs into a single underscore.
    s = re.sub(r"[^a-z0-9_-]+", "_", s).strip("_")
    return s or "na"


def _date_slug(value) -> str:
    """Date or datetime → YYYYMMDD. Pass-through for already-formatted strings."""
    if value is None:
        return "na"
    if hasattr(value, "strftime"):
        return value.strftime("%Y%m%d")
    s = str(value).replace("-", "").replace("/", "")[:8]
    return s if s.isdigit() else _slug(value)


def build_download_filename(context: str, *,
                            universe=None, selected_index=None,
                            dates=None, ext: str = "xlsx") -> str:
    """Standardized download filename.

    Format: ``sanket_<context>_<universe>[_<index>]_<dates>.<ext>``

    Args:
        context: short label identifying the export (e.g. ``"snapshot"``,
            ``"bullish"``, ``"range"``, ``"profile"``, ``"correlation"``).
        universe: sidebar universe (e.g. ``"India Indexes"``).
        selected_index: optional sub-selection (e.g. ``"NIFTY 50"``).
        dates: a single date, a (start, end) tuple, or a pre-formatted string.
        ext: file extension without the dot.

    Examples:
        sanket_snapshot_india_indexes_nifty_50_20260507.xlsx
        sanket_range_us_indexes_dow_jones_20240101-20260507.xlsx
        sanket_profile_crypto_digital_assets_top_20_20260507.json
    """
    parts = ["sanket", _slug(context)]
    if universe:
        uni = _slug(universe)
        if selected_index:
            uni = f"{uni}_{_slug(selected_index)}"
        parts.append(uni)
    if dates is not None:
        if isinstance(dates, (tuple, list)) and len(dates) == 2:
            parts.append(f"{_date_slug(dates[0])}-{_date_slug(dates[1])}")
        else:
            parts.append(_date_slug(dates))
    return "_".join(parts) + "." + ext.lstrip(".")


def to_excel(df):
    """Convert DataFrame to Excel bytes for download with a Legend sheet.

    Per-bar history columns (Trace_Hist / Close_Hist / Units_Hist) hold Python lists — they
    exist so the UI can report the bar a signal fired on, and would serialise as list-reprs.
    Dropped from the export; the per-age BUY_*/SELL_* columns carry the events legibly.
    """
    output = io.BytesIO()
    _drop = [c for c in ('Trace_Hist', 'Units_Hist', 'Close_Hist') if c in df.columns]
    if _drop:
        df = df.drop(columns=_drop)
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sanket_Quant_Data')
        
        # Add Legend for user clarity. THE SIGNAL block comes first, then everything
        # that is descriptive context — the distinction matters more than the ordering.
        legend = [
            ("— THE SIGNAL SET (PRAGATI · conviction × value) —", ""),
            ("turn_buy / turn_sell", "v9: ▲ CAPITULATION — the first bar the grid stands in Buy · capitulation (sellers in control across the ladder, value cheap past θ) with value momentum turning back toward fair; ▼ DISTRIBUTION — the first bar sellers hold control of a price rich past θ. Read from the grid, 10-bar cooldown per side. The last one stands as the declaration."),
            ("resume_long / resume_short", "◆ RESUME — OFF by default (negative in the v8 and v9 audits): the histogram dipped to the wrong side inside 6 bars and now crosses its k·σ gate; chart conviction on its side; conviction tape past the inner zone; value tape short of θ; effort not absorbed."),
            ("BUY_* / SELL_*", "The long / short event by age (Today … Within 5): ▲ capitulation / ▼ distribution, ◆ a RESUME, — none."),
            ("Side / Signal_Kind", "Buy / Sell / — : an event fired on THIS bar, and its kind — TURN (the ▲▼) or RESUME (◆)."),
            ("PRG_Event / PRG_State", "The event label on this bar; the bar's state — WARMING UP / PAUSED / CAPITULATION / DISTRIBUTION / RESUME / WATCH (in capitulation, value still cheapening) / NEUTRAL."),
            ("PRG_Armed / PRG_Armed_Age", "The watchlist: +1 while a name sits in capitulation with value still cheapening, and bars in the cell. The ▲ fires when value turns."),
            ("PRG_Decl / PRG_Decl_Age", "The standing declaration (+1 after a ▲, −1 after a ▼) and bars since. No exit; as a held position it carried no edge."),
            ("PRG_Hold_Dir / PRG_Hold_Age / PRG_Hold_Kind", "The latest event while inside the declared hold horizon."),
            ("PRG_Trace / Signal", "The trace, ±100: conviction and value in σ, blended with their measured correlation, bounded once on Samanvaya's scale. θ = ±42.9. How far the move is stretched."),
            ("PRG_Hist / PRG_Hist_Z", "The trace's push — trace minus its 9-bar EMA — native, and in σ of its own distribution."),
            ("PRG_Push / PRG_Push_Tier", "The push in five levels (+2 impulse ↑ … −2 impulse ↓, 0 = pale / quiet column) and its drawn tier."),
            ("PRG_CTape", "MTF conviction tape, ±100 — who controls across the ladder (Ladder down: Daily, the intraday frames inside each day · D, ↺ W·D before intraday history; Weekly, 1h·4h·daily bars inside the week · W)."),
            ("PRG_VTape", "MTF value tape, ±100 — rich (+) or cheap (−) across the ladder (Daily: W·D; Weekly: M·W)."),
            ("PRG_Conv / PRG_Conv_Z", "The chart's conviction (Nishchaya v3 exactly), ±100 and in σ — the trace's flow ingredient."),
            ("PRG_Value / PRG_Value_Z", "Samanvaya's value on this chart, ±100 and in σ — the trace's position ingredient: the macro-hedged relative-value spread blended with seven price-only breadth views."),
            ("PRG_Hedge / PRG_Drivers", "How much of the macro hedge the value leg applies (its own out-of-sample skill) and the drivers selected."),
            ("PRG_Absorbed / PRG_Eff_Pct", "Effort → result: the share of participation that became displacement, as a percentile of its own history. Absorbed = bottom fifth."),
            ("PRG_Div_Seen_Bull / _Bear", "A regular divergence on conviction's own pivots, zone-gated, at a price value called stretched, in the last 20 bars — context only; no signal reads it."),
            ("PRG_Split / PRG_Quiet / PRG_Settling", "Read with caution: the trace's ingredients disagree; the regime is quiet (conviction amplifying a small imbalance); the value basket is settling after a rotation."),
            ("PRG_Stack_OK / PRG_Why", "Whether the signal set can judge this bar, and if not which layer is warming."),
            ("— THE STATE (CONVICTION-VALUE GRID · 3 × 3, v8) —", ""),
            ("CVG_Action / CVG_Why / CVG_Units", "The grid cell as an action and its reason, with its GRADED units — the cell's units read at the name's shaded position, as Pragyam sizes. v9.1's measured cell units: Buy · capitulation 4 · Accumulate · washout 1.5 · Exit · distribution 0.25 · Accumulate · basing 1.5 · Wait · idle 1 · Trim · stalling 0.75 · Buy · turned 3 · Hold · building 1.5 · Trim · paid 0.75. A weight, not a forecast."),
            ("CVG_Held", "The conviction tape has moved to another row but the push has not confirmed it, so the row is held."),
            ("CVG_Bars / CVG_From", "Bars in the current cell, and the cell before it."),
            ("CVG_Chart_Action / CVG_Lead", "Where the chart's own conviction and value would place the name, and whether that cell carries more (+1) or fewer (−1) units than the state."),
            ("Priority_Long / Priority_Short", "Ranking keys, by STRETCH read as reversion: a ▲/▼ on this bar first, then every name by how far its trace is stretched against the side (long: stretched down first; short: stretched up first). Measured by trace_study.py — the grid-weight ranking it replaced ran backwards."),
            ("Signal_Reason", "Plain-language read of the row, with the measured verdict for this universe."),
            ("— CONTEXT ONLY (never a signal input; none predicts outcome out of sample) —", ""),
            ("Zone / Condition", "Where cumulative delta sits vs its 20-bar mean: Accumulation(+) / Distribution(+) / Neutral."),
            ("Bar_Delta / CVD / CVD_Slope", "Inferred per-bar volume delta (close-location proxy), its running sum and 3-bar change."),
            ("Delta_Z", "Signed z-score of Bar_Delta vs its 20-bar distribution. A close-location proxy, unrelated to Pragati's conviction."),
            ("Abs_Strength / Buy_Share / Absorption_Score", "Absorption magnitude, rolling inferred buy share, and a [0,1] absorption context score."),
            ("Regime / Regime_Confidence / Vol_Regime / Change_Point", "HMM regime, GARCH volatility regime and CUSUM change points. Per-name RISK CONTEXT."),
            ("Ret_1b / Ret_5b / Ret_10b / Ret_21b", "Forward returns (Historical Range only). LABELS for evaluation — never inputs."),
        ]
        legend_data = {"Column Identifier": [k for k, _ in legend],
                       "Metric Description": [v for _, v in legend]}
        pd.DataFrame(legend_data).to_excel(writer, index=False, sheet_name='Legend')
        
    return output.getvalue()

# ══════════════════════════════════════════════════════════════════════════════
# SHARED MATH HELPERS  (SMA + True Range — the only primitives the engine needs)
# ──────────────────────────────────────────────────────────────────────────────
#  The WRCI-era MA library (EMA/HMA/WMA/VWMA/ALMA/RMA, f_smooth, linreg, RSI) and
#  the Ehlers AutoTune filter were removed with the WRCI engine. Pragati needs neither:
#  it carries its own EMA/SMA/stdev chain inside pragati.py and samanvaya.py, transcribed
#  from the Pine. What remains here serves the descriptive order-flow context only.
# ══════════════════════════════════════════════════════════════════════════════

def calculate_sma(series, length):
    if length <= 1:
        return series
    return series.rolling(window=length).mean()


def calculate_true_range(df):
    """Standard True Range calculation (ATR base)."""
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _rolling_volume_profile(high, low, vol, win=20, bins=24, va_pct=0.70):
    """Rolling volume-by-price profile over `win` bars → (POC, VAH, VAL) Series.

    Mirrors the `Order Flow.pine` profile builder: each bar's volume is distributed
    across the price bins its range spans, POC = highest-volume bin, and the value area
    expands from the POC to `va_pct` of total volume. POC/VAH/VAL are the structural
    backbone of the validated signal sets (fair value + acceptance edges).
    """
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    v = vol.to_numpy(dtype=float)
    n = len(h)
    poc = np.full(n, np.nan); vah = np.full(n, np.nan); val = np.full(n, np.nan)
    for i in range(win - 1, n):
        s = slice(i - win + 1, i + 1)
        wh, wl, wv = h[s], l[s], v[s]
        lo, hi = wl.min(), wh.max()
        if not (hi > lo):
            continue
        step = (hi - lo) / bins
        bucket = np.zeros(bins)
        for j in range(win):
            b0 = int(min(bins - 1, max(0, (wl[j] - lo) // step)))
            b1 = int(min(bins - 1, max(0, (wh[j] - lo) // step)))
            bucket[b0:b1 + 1] += wv[j] / (b1 - b0 + 1)
        pidx = int(bucket.argmax())
        poc[i] = lo + (pidx + 0.5) * step
        tot = bucket.sum(); tgt = tot * va_pct
        acc = bucket[pidx]; a = b = pidx
        while acc < tgt and (a > 0 or b < bins - 1):
            up = bucket[b + 1] if b < bins - 1 else -1.0
            dn = bucket[a - 1] if a > 0 else -1.0
            if up >= dn:
                b += 1; acc += bucket[b]
            else:
                a -= 1; acc += bucket[a]
        vah[i] = lo + (b + 1) * step
        val[i] = lo + a * step
    idx = high.index
    return (pd.Series(poc, index=idx), pd.Series(vah, index=idx), pd.Series(val, index=idx))


def run_full_analysis(df, reg_len=20, n1=10, n2=21, obLevel1=80, obLevel2=40, osLevel1=-80, osLevel2=-40,
                      wt2_len=20, wt2_type="ALMA",
                      hci_thres=0.25, hci_look=102, hci_sig_len=53, hci_sig_type="SMA", hci_roc_len=15,
                      sid=None, drivers=None, symbol="", daily=None):
    """Per-symbol feature engine — the Pragati stack plus order-flow context.

    The SIGNALS are Pragati's (see engine.py / pragati.py): the trace (conviction × value),
    its histogram, the two tapes, the ▲▼ (read from the grid) and ◆, and the 3 × 3 grid state. They
    are attached here via ``eng.add_pragati_features``; the cross-section is ranked later
    by ``eng.compute_ranking``. ``drivers`` are the macro closes behind the value
    ingredient (prepared for the chart), ``symbol`` names the instrument (its close time
    and home market decide driver timing), and ``daily`` carries the daily bars behind a
    weekly chart for its Ladder-down conviction rung.

    Everything else written here is DESCRIPTIVE CONTEXT, never a signal input: the inferred
    delta / CVD / volume profile (OHLC proxies, validated to add no cross-sectional edge),
    the MA alignment count, and the F1/F2 features the regime engine consumes.

    ``sid`` is the run's :class:`eng.EngineSettings`; ``None`` falls back to the defaults.
    The unused WRCI-era params (n1/n2/obLevel*/osLevel*/wt2_*/hci_*) are retained in the
    signature only so existing call sites keep working; ``reg_len`` still drives the ATR
    window. ``_analysis_params_sig`` carries an engine tag plus the SB parameters, so frames
    cached by the old engine — or under a different threshold — are invalidated.
    """
    reg_len = max(reg_len, 2)

    high, low, close = df['High'], df['Low'], df['Close']
    vol = df['Volume']

    # Institutional Volume Fallback: many index symbols report zero volume. Without it
    # the inferred delta degenerates to 0 everywhere; with it the close-location model
    # still yields a price-shape proxy (divergence/absorption are weak on such symbols).
    if vol.sum() == 0:
        vol = pd.Series(1.0, index=df.index)

    # ── INFERRED BAR DELTA (OHLC proxy · Order Flow.pine f_proxy_delta) ─────────
    # close-location value in [-1, 1]: +1 = close on the high (max inferred buying),
    # -1 = close on the low (max inferred selling).
    tr_range = (high - low).clip(lower=1e-4)
    clv      = ((close - low) - (high - close)) / tr_range
    buy_vol  = vol * (clv + 1.0) / 2.0
    sell_vol = vol - buy_vol
    bar_delta = (buy_vol - sell_vol).fillna(0.0)

    # ── CUMULATIVE VOLUME DELTA + slope + trend EMA ────────────────────────────
    cvd        = bar_delta.cumsum()
    cvd_slope  = cvd.diff(3).fillna(0.0)          # 3-bar flow build/drain
    cvd_ma     = cvd.rolling(20).mean()
    cvd_ema    = cvd.ewm(span=20, adjust=False).mean()   # CVD flow-trend (UI context)

    # ── ATR(14) for absorption range-normalization ─────────────────────────────
    # Pine's ta.atr is RMA-smoothed (Wilder), not SMA — matched exactly so Rel_Range
    # reproduces inferred_delta.pine's absorption geometry.
    tr    = calculate_true_range(df)
    atr14 = pd.Series(_rma(tr.to_numpy(dtype=float), 14), index=df.index)
    mintick = (close.abs().clip(lower=1e-6) * 1e-4)   # proxy for syminfo.mintick

    # ── Signal strengths surfaced to UI + priority factors ─────────────────────
    abs_delta      = bar_delta.abs()
    abs_delta_sma  = calculate_sma(abs_delta, 20).clip(lower=1e-9)
    rel_delta      = (abs_delta / abs_delta_sma).fillna(0.0)              # absorption magnitude
    rel_range      = ((high - low) / np.maximum(atr14, mintick)).fillna(0.0)
    # Normalized delta z-score (signed) — generic "how one-sided is this bar" strength.
    delta_mean = bar_delta.rolling(20).mean()
    delta_std  = bar_delta.rolling(20).std(ddof=0).clip(lower=1e-9)
    delta_z    = ((bar_delta - delta_mean) / delta_std).clip(-5, 5).fillna(0.0)

    # ── Participation (RVOL) — measured participation gauge (UI context) ──
    rvol = (vol / vol.rolling(20).mean().clip(lower=1e-9)).fillna(1.0)

    # ── Rolling buy share (inferred_delta.pine winBuy/winSell · L372-373) ───────
    # Bar_Delta is signed VOLUME and CVD is a cumsum from the first fetched bar, so
    # across the universe neither is comparable: Bar_Delta scales with the symbol's
    # absolute volume, and CVD's level is an artifact of how much history was pulled.
    # This windowed buy share is the cross-sectionally-safe read the Pine already
    # carries (dashboard "Rolling 20-bar inferred buy share", L1590) but the port
    # dropped — the volume-weighted fraction of inferred buying over 20 bars, bounded
    # [0,1] (0.5 = balanced), baseline-invariant (windowed, not cumulative). Verified:
    # identical for two symbols of identical bar shape at 1× vs 100× volume, where
    # Bar_Delta/CVD differ 100×. Smoother than the per-bar Delta_Z. Context only.
    win_buy   = buy_vol.rolling(20).sum()
    win_vol   = vol.rolling(20).sum().clip(lower=1e-12)
    buy_share = (win_buy / win_vol).clip(0.0, 1.0).fillna(0.5)

    # ── Absorption score — smooth [0,1] fusion of rel_delta × rel_range ─────────
    # inferred_delta.pine flags rawAbsorb = relDelta > 1.8 AND relRange < 0.6 (large
    # delta soaked by a small range = passive limit absorption). The port kept only
    # the two magnitudes as separate columns, so an absorbed bar can't be sorted for
    # without cross-referencing both. This fuses them via a logistic gate on each
    # Pine threshold; the score's 0.25 iso-contour reproduces the Pine boundary
    # (verified: 96% grid agreement off the thin boundary band), 1 = deep absorption.
    # Context only — NOT a ranking input (order-flow signals add no cross-sectional
    # edge here, validated), surfaced as flow colour.
    g_delta  = 1.0 / (1.0 + np.exp(-3.0 * (rel_delta - 1.8)))
    g_range  = 1.0 / (1.0 + np.exp(-8.0 * (0.6 - rel_range)))
    absorption_score = (g_delta * g_range).fillna(0.0)

    # ── Rolling volume profile — POC (fair value) + value-area edges (VAH/VAL) ──
    poc, vah, val = _rolling_volume_profile(high, low, vol, win=20, va_pct=0.70)
    # Position within the value area: 0 = at VAL (cheap), 1 = at VAH (rich). UI context.
    va_pos = ((close - val) / (vah - val).replace(0, np.nan))

    # ── F1 · PRICE MOMENTUM (orthogonal, retained from prior engine) ───────────
    close_lag5 = close.shift(5).fillna(close)
    log_ret_5  = np.log(close / close_lag5)
    atr_pct_v4 = (tr.rolling(14).mean() / close).clip(lower=1e-6)
    F1_PriceMom = (log_ret_5 / atr_pct_v4).clip(-5, 5).fillna(0)

    # ── F2 · VOLUME QUALITY (signed, smoothed; retained) ───────────────────────
    vol_mean   = df['Volume'].rolling(20).mean()
    vol_std    = df['Volume'].rolling(20).std(ddof=0).clip(lower=1e-6)
    vol_z_raw  = (df['Volume'] - vol_mean) / vol_std
    price_dir_5 = np.sign(close - close_lag5)
    F2_VolQual = (vol_z_raw * price_dir_5).rolling(5).mean().clip(-5, 5).fillna(0)

    # ── WRITE ORDER-FLOW COLUMNS ───────────────────────────────────────────────
    df['F1_PriceMom']  = F1_PriceMom
    df['F2_VolQual']   = F2_VolQual
    df['Bar_Delta']    = bar_delta
    df['Buy_Vol']      = buy_vol
    df['Sell_Vol']     = sell_vol
    df['CVD']          = cvd
    df['CVD_Slope']    = cvd_slope
    df['CVD_EMA']      = cvd_ema
    df['Delta_Z']      = delta_z
    df['Abs_Strength'] = rel_delta
    df['Rel_Range']    = rel_range
    df['Buy_Share']        = buy_share          # rolling 20-bar inferred buy fraction ∈ [0,1]
    df['Absorption_Score'] = absorption_score   # smooth [0,1] absorption context
    df['RVOL']         = rvol
    df['POC']          = poc
    df['VAH']          = vah
    df['VAL']          = val
    df['VA_Pos']       = va_pos

    # ── MA ALIGNMENT (retained display metric) ─────────────────────────────────
    ma_counts = pd.Series(0, index=df.index)
    for ma in [8, 21, 50, 100, 200]:
        ema = close.ewm(span=ma, adjust=False).mean()
        ma_counts += (close > ema).astype(int)
    df['MA_Alignment'] = ma_counts

    # ── FLOW CONDITION (context only) ──────────────────────────────────────────
    # Where cumulative delta sits vs its 20-bar mean → accumulation / distribution.
    # Consumed by the Correlation setup classifier and the range-mode breadth charts;
    # it is not part of the signal.
    cvd_dev = (cvd - cvd_ma)
    band    = cvd_dev.abs().rolling(20).mean().clip(lower=1e-9)
    df['Condition'] = np.select(
        [cvd_dev >  2 * band, cvd_dev >  band, cvd_dev < -2 * band, cvd_dev < -band],
        ['Accumulation+', 'Accumulation', 'Distribution+', 'Distribution'],
        default='Neutral',
    )

    # ── THE SIGNAL STACK — Pragati (engine.py) ──────────────────────────────────
    # Writes the PRG_* readings, the four events (turn_buy / turn_sell / resume_long /
    # resume_short), the watch and hold windows, and the CVG_* grid state. Cross-sectional
    # ranking happens later, once the universe is assembled.
    #
    # Note this reads Volume directly from `df`, NOT the zero-volume-substituted `vol`
    # built above for the order-flow proxies: participation weighting is supposed to fall
    # through to true range on a volume-less symbol, and feeding it a synthetic 1.0 would
    # silently make every bar equally weighted instead.
    _sb = sid if sid is not None else _engine_settings(None, None, "Daily")
    df = eng.add_pragati_features(df, drivers=drivers, symbol=symbol, settings=_sb, daily=daily)

    return df


def _rma(x: np.ndarray, length: int) -> np.ndarray:
    """Port of Pine ``ta.rma`` (Wilder smoothing, used by ``ta.atr``).

    Seeded exactly like Pine: the first output is the SMA of the first ``length``
    finite values, then the ``alpha = 1/length`` recursion. NaN inputs hold the
    previous value; output is NaN until the seed exists.
    """
    n = x.shape[0]
    out = np.full(n, np.nan)
    csum = 0.0
    cnt = 0
    start = -1
    for i in range(n):
        if np.isfinite(x[i]):
            cnt += 1
            csum += x[i]
            if cnt == length:
                out[i] = csum / length
                start = i
                break
    if start == -1:
        return out
    alpha = 1.0 / length
    for i in range(start + 1, n):
        xi = x[i]
        out[i] = out[i - 1] if not np.isfinite(xi) else out[i - 1] + alpha * (xi - out[i - 1])
    return out


# ══════════════════════════════════════════════════════════════════════════════
# REGIME ENGINE (per-name risk context — never a signal input)
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveHMM:
    """Hidden Markov Model for regime state discovery over the joint feature observation."""
    
    def __init__(self):
        self.n_states = 3
        self.transition_matrix = np.array([
            [0.85, 0.10, 0.05],
            [0.10, 0.80, 0.10],
            [0.05, 0.10, 0.85]
        ])
        self.emission_means = np.array([1.5, 0.0, -1.5])
        self.emission_stds = np.array([1.2, 0.8, 1.2])
        self.state_probabilities = np.array([0.33, 0.34, 0.33])
        self.observation_history = []
        self.state_history = []
    
    def _gaussian_pdf(self, x, mean, std):
        if std < 1e-8:
            return 1.0 if abs(x - mean) < 1e-8 else 0.0
        return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    
    def update(self, observation):
        self.observation_history.append(observation)
        predicted = self.transition_matrix.T @ self.state_probabilities
        emissions = np.array([self._gaussian_pdf(observation, self.emission_means[s], self.emission_stds[s]) for s in range(3)])
        updated = emissions * predicted
        total = updated.sum()
        if total > 1e-10:
            updated /= total
        else:
            # Carry forward prior state rather than resetting to uniform —
            # preserves regime belief when all emissions are numerically tiny.
            updated = self.state_probabilities.copy()
        self.state_probabilities = updated
        most_likely = np.argmax(updated)
        self.state_history.append(most_likely)
        
        if len(self.observation_history) >= 10:
            recent_obs = np.array(self.observation_history[-50:])
            recent_states = self.state_history[-len(recent_obs):]
            for state in range(3):
                mask = np.array(recent_states) == state
                if mask.sum() >= 2:
                    state_obs = recent_obs[mask]
                    self.emission_means[state] = 0.9 * self.emission_means[state] + 0.1 * np.mean(state_obs)
                    self.emission_stds[state] = 0.9 * self.emission_stds[state] + 0.1 * max(np.std(state_obs), 0.1)

            # Identifiability constraint: online adaptation of unconstrained Gaussian
            # means is subject to LABEL SWITCHING — the "BULL" state's mean can drift
            # below "BEAR"'s, after which every regime label is semantically flipped.
            # Enforce mean(BULL) >= mean(NEUTRAL) >= mean(BEAR) by re-sorting the
            # states whenever the ordering breaks, permuting every piece of per-state
            # state (means, stds, beliefs, transition matrix, recorded state labels)
            # consistently so the model is unchanged up to relabeling.
            order = np.argsort(-self.emission_means)
            if not np.array_equal(order, [0, 1, 2]):
                self.emission_means = self.emission_means[order]
                self.emission_stds = self.emission_stds[order]
                self.state_probabilities = self.state_probabilities[order]
                self.transition_matrix = self.transition_matrix[np.ix_(order, order)]
                remap = np.empty(3, dtype=int)
                remap[order] = np.arange(3)
                self.state_history = [int(remap[s]) for s in self.state_history]
                updated = self.state_probabilities

        return {"BULL": updated[0], "NEUTRAL": updated[1], "BEAR": updated[2]}


class GARCHDetector:
    """GARCH-inspired volatility regime detection on the joint-observation shocks."""
    
    def __init__(self):
        self.current_variance = 0.04
        self.omega = 0.0001
        self.alpha = 0.1
        self.beta = 0.85
        self.long_term_mean = 0.04
        self.shock_history = []
    
    def update(self, shock):
        self.shock_history.append(shock)
        shock_sq = shock ** 2
        new_var = self.omega + self.alpha * shock_sq + self.beta * self.current_variance
        # Numerical guard ONLY — the ceiling must never bind on realistic shocks.
        # The old cap of 1.0 was routinely hit (joint-obs shocks have variance ~1-6):
        # current_variance pinned at 1.0 while long_term_mean tracked the UNCLIPPED
        # realized variance, so the current/long-term ratio collapsed below 0.6 and
        # sustained HIGH volatility was reported as "LOW" — inverting the regime read
        # that scales conviction. 25.0 (σ=5 on a ±5-clipped observation scale) is
        # unreachable in normal operation.
        self.current_variance = np.clip(new_var, 1e-4, 25.0)
        
        if len(self.shock_history) >= 10:
            realized = np.var(self.shock_history[-min(50, len(self.shock_history)):])
            self.long_term_mean = 0.95 * self.long_term_mean + 0.05 * realized
        
        return np.sqrt(self.current_variance)
    
    def get_regime(self):
        current_vol = np.sqrt(self.current_variance)
        long_term_vol = np.sqrt(self.long_term_mean)
        ratio = current_vol / long_term_vol if long_term_vol > 0 else 1.0
        
        if ratio < 0.6:
            return "LOW", 1.3
        elif ratio < 0.9:
            return "NORMAL", 1.0
        elif ratio < 1.4:
            return "HIGH", 0.8
        else:
            return "EXTREME", 0.6


class CUSUMDetector:
    """CUSUM change-point detection for regime shifts in the joint observation."""
    
    def __init__(self, threshold=4.0, drift=0.5):
        self.threshold = threshold
        self.drift = drift
        self.positive_cusum = 0.0
        self.negative_cusum = 0.0
        self.value_history = []
        self.running_mean = 0.0
        self.running_std = 1.0
    
    def update(self, value):
        self.value_history.append(value)
        
        if len(self.value_history) >= 3:
            recent = self.value_history[-min(20, len(self.value_history)):]
            self.running_mean = np.mean(recent)
            self.running_std = max(np.std(recent), 0.1)
        
        z = (value - self.running_mean) / self.running_std
        # 0.99 decay prevents unreleased drift from accumulating during quiet periods
        self.positive_cusum = max(0, self.positive_cusum * 0.99 + z - self.drift)
        self.negative_cusum = max(0, self.negative_cusum * 0.99 - z - self.drift)
        
        change_detected = self.positive_cusum > self.threshold or self.negative_cusum > self.threshold
        
        if change_detected:
            self.positive_cusum = 0
            self.negative_cusum = 0

        return change_detected


class AdaptiveKalmanFilter:
    """Kalman filter smoothing of the joint observation before HMM/CUSUM."""
    
    def __init__(self, process_var=0.01, measurement_var=0.1):
        self.estimate = 0.0
        self.error_covariance = 1.0
        self.process_variance = process_var
        self.measurement_variance = measurement_var
        self.innovation_history = []
    
    def update(self, measurement):
        predicted_estimate = self.estimate
        predicted_covariance = self.error_covariance + self.process_variance
        innovation = measurement - predicted_estimate
        self.innovation_history.append(innovation)
        if len(self.innovation_history) > 50:
            self.innovation_history.pop(0)
        innovation_cov = predicted_covariance + self.measurement_variance
        kalman_gain = predicted_covariance / innovation_cov
        self.estimate = predicted_estimate + kalman_gain * innovation
        self.error_covariance = (1 - kalman_gain) * predicted_covariance
        
        if len(self.innovation_history) >= 5:
            innovation_var = np.var(self.innovation_history[-min(20, len(self.innovation_history)):])
            self.measurement_variance = 0.9 * self.measurement_variance + 0.1 * innovation_var

        return self.estimate


def run_regime_analysis(df):
    """
    Apply joint-state regime classification over (F1_PriceMom, F2_VolQual, CVD flow).
    The three input dimensions are roughly orthogonal (price momentum, volume
    quality, cumulative-delta flow), so HMM's classification reflects true market
    state.

    Pure per-name RISK CONTEXT. Its Regime / Vol_Regime / Change_Point outputs are
    displayed alongside the signal and aggregated in the range-mode Regime tab; they do
    NOT enter the Pragati stack — its trace, tapes, signals or grid state — which reads
    OHLCV and the macro drivers and nothing else.
    """
    hmm    = AdaptiveHMM()
    garch  = GARCHDetector()
    cusum  = CUSUMDetector()
    kalman = AdaptiveKalmanFilter()

    regimes, hmm_bulls, hmm_bears, vol_regimes = [], [], [], []
    change_points, confidences, signal_history = [], [], []

    f1_vals = df['F1_PriceMom'].values
    f2_vals = df['F2_VolQual'].values
    # Third orthogonal view: cumulative-delta flow build/drain, squashed to ~[-5, +5].
    cv_vals = (np.tanh(df['CVD_Slope'].values / 1.0e6) * 5.0)

    # Warmup pass: prime detectors on first bars so that bar-0 output isn't
    # determined purely by uninformed priors. State is carried forward into the
    # main recording loop; the warmup output is discarded.
    _warmup = min(20, len(df) // 4)
    for _wi in range(_warmup):
        _f1 = 0.0 if np.isnan(f1_vals[_wi]) else f1_vals[_wi]
        _f2 = 0.0 if np.isnan(f2_vals[_wi]) else f2_vals[_wi]
        _cv = 0.0 if np.isnan(cv_vals[_wi]) else cv_vals[_wi]
        _obs = 0.40 * _f1 + 0.25 * _f2 + 0.35 * _cv
        _filt = kalman.update(_obs)
        _shock = _obs - (signal_history[-1] if signal_history else 0.0)
        garch.update(_shock)
        hmm.update(_filt)
        cusum.update(_filt)
        signal_history.append(_obs)
    # End of warmup. Keep the *adapted scalar estimates* (emission means/stds,
    # current/long-term variance, Kalman estimate, CUSUM accumulators, running
    # mean/std) — that is the priming benefit — but clear the raw rolling-history
    # LISTS. Otherwise the main loop below re-feeds bars 0..warmup-1, recording
    # them a SECOND time into these windows and creating an "echo" that skews the
    # rolling variance/emission baselines. Clearing lets the windows rebuild
    # naturally from bar 0 while starting from the warmed estimates.
    signal_history.clear()
    hmm.observation_history.clear()
    hmm.state_history.clear()
    garch.shock_history.clear()
    cusum.value_history.clear()
    kalman.innovation_history.clear()

    for i in range(len(df)):
        # Joint observation: weighted mean of orthogonal views
        f1 = 0.0 if np.isnan(f1_vals[i]) else f1_vals[i]
        f2 = 0.0 if np.isnan(f2_vals[i]) else f2_vals[i]
        cv = 0.0 if np.isnan(cv_vals[i]) else cv_vals[i]
        joint_obs = (0.40 * f1 + 0.25 * f2 + 0.35 * cv)

        filtered = kalman.update(joint_obs)
        shock    = joint_obs - signal_history[-1] if signal_history else 0.0
        garch.update(shock)
        vol_regime, _ = garch.get_regime()

        hmm_probs = hmm.update(filtered)
        change    = cusum.update(filtered)

        bull_p = hmm_probs['BULL']
        bear_p = hmm_probs['BEAR']
        if change:
            regime = "TRANSITION"
        elif bull_p > 0.6:    regime = "BULL"
        elif bear_p > 0.6:    regime = "BEAR"
        elif bull_p > 0.4:    regime = "WEAK_BULL"
        elif bear_p > 0.4:    regime = "WEAK_BEAR"
        else:                 regime = "NEUTRAL"

        regimes.append(regime); hmm_bulls.append(bull_p); hmm_bears.append(bear_p)
        vol_regimes.append(vol_regime); change_points.append(change)
        confidences.append(max(bull_p, bear_p, hmm_probs['NEUTRAL']))
        signal_history.append(joint_obs)

    df['Regime']            = regimes
    df['HMM_Bull']          = hmm_bulls
    df['HMM_Bear']          = hmm_bears
    df['Vol_Regime']        = vol_regimes
    df['Change_Point']      = change_points
    df['Regime_Confidence'] = confidences
    return df


def _classify_signal_type(row) -> str:
    """Return the signal type for a single bar row (pandas Series).

    A fired event wins (▲ capitulation / ▼ distribution / ◆ RESUME ↑ / ◆ RESUME ↓); otherwise the row
    falls back to its flow zone (context only). Matches the vectorised np.select in the
    harvest path.
    """
    ev = row.get('PRG_Event')
    if isinstance(ev, str) and ev:
        return ev
    cond = row.get('Condition', 'Neutral')
    return cond if cond != 'Neutral' else '-'

# ══════════════════════════════════════════════════════════════════════════════
# DATA HANDLING & UTILITIES
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── Column glossaries ─────────────────────────────────────────────────────
# Streamlit's grid puts per-column explanations in a hover tooltip, which is
# the only place it HAS to put them. Moving off that grid would have thrown the
# text away, so it lands here instead: a definition list under the table, in
# the same key/value grammar the rail readout and the landing specs use. More
# discoverable than a tooltip nobody hovers, and it survives a screenshot.

def _glossary(defs: "dict[str, str]") -> str:
    """Render ``{column: explanation}`` as the panel-footer definition list."""
    if not defs:
        return ""
    return ('<div class="panel-specs">' + "".join(
        f'<div class="lookback-row"><span class="lbl">{html.escape(str(k))}</span>'
        f'<span class="val">{html.escape(str(v))}</span></div>'
        for k, v in defs.items()) + "</div>")


# ── The context line every chart panel carries ────────────────────────────
# A panel header does not restate the section header above it — that would be
# the same title four pixels lower. What the section header cannot say is which
# universe and timeframe the plot is actually drawn on, so that is what the
# panel carries. Built from the same session keys the command bar reads, which
# is both less plumbing than threading it through every call site and strictly
# more correct: a context built from those keys cannot disagree with the bar.

def _chart_ctx(units: str = "") -> str:
    """``UNIVERSE · Timeframe [· units]`` for a chart panel header."""
    parts = [
        str(st.session_state.get("active_universe", "") or "").upper(),
        str(st.session_state.get("active_timeframe", "") or ""),
    ]
    if units:
        parts.append(units)
    return " · ".join(x for x in parts if x)


# ── Notices, in the app's own vocabulary ──────────────────────────────────
# Streamlit's st.error / st.warning / st.info each bring their own typeface,
# icon, radius and ink, none of which the stylesheet reaches — three of them on
# a page read as three different products' alert systems. These are drop-in
# replacements taking the same single string, so a call site does not have to
# know which component it lands in.
#
# The split is by SEVERITY, not by colour: an error is something that failed, a
# warning is something the reader must not miss, an info is context. Amber
# appears only in the warning tier, because amber is caution in this system and
# nothing else.

def ui_error(msg, *_a, **_kw) -> None:
    """Something failed. Rendered as the app's warning box, titled."""
    ui.render_warning_box("Error", str(msg))


def ui_warning(msg, *_a, **_kw) -> None:
    """Something the reader must not miss, but the page still works."""
    ui.render_warning_box("Warning", str(msg))


def ui_info(msg, *_a, **_kw) -> None:
    """Context — the empty/degraded state, not an alert.

    ``render_empty_state`` rather than an info box because almost every
    ``st.info`` in this app was a "nothing to show, and here is why" message,
    which is what an empty state IS. An info box around that copy reads as an
    interruption of content that is not there.
    """
    ui.render_empty_state("Nothing to show", str(msg))


def render_footer():
    """Render app footer with copyright and version info."""
    ist = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
    st.markdown(f"""
    <div class="app-footer">
        <div class="content">
            © {ist.year} <strong>Sanket</strong> &nbsp;·&nbsp; @thebullishvalue &nbsp;·&nbsp; {VERSION} &nbsp;·&nbsp; {ist.strftime("%Y-%m-%d %H:%M:%S IST")}
        </div>
    </div>
    """, unsafe_allow_html=True)


#: The three parts of the system, as the cold-start screen describes them.
#: Data, not markup — the landing page renders them through one template, so
#: the three panels cannot drift apart in structure the way three hand-written
#: HTML blocks did.
_SYSTEM_PANELS = (
    ("engine", "Pragati · Conviction × Value", "One trace, its push, two tapes",
     "How far a move is stretched — in one-sided effort, and in price against what the "
     "macro drivers explain — on one trace, with its own push beneath it. The two "
     "ingredients are read apart, across horizons, on two tapes: who controls, and "
     "where price stands.",
     (("Conviction", "Σ(c·w) / Σ(|c|·w), c = ΔC / TR"),
      ("Value", "Samanvaya: hedged RV ⊕ breadth"),
      ("Trace", "blended in σ, bounded once · θ ±43"),
      ("Ladders", "conviction ↓ 1m…4h·D (↺ W·D) · value W·D / M·W"))),
    ("events", "Two Signals, One State", "Events, and where a name stands",
     "The 3 × 3 grid names each name's state as an action, and v9 reads the events from it: "
     "▲ CAPITULATION — sellers in control of a cheap price, value turning back toward fair, "
     "the one event measured positive in every era; ▼ DISTRIBUTION — sellers taking control "
     "of a rich price. ◆ RESUME is off by default (measured negative).",
     (("▲", "capitulation, value turning"),
      ("▼", "sellers take a rich price"),
      ("Grid", "Buy 4 … Exit 0.25 (measured units)"),
      ("Ranking", "▲▼ today, then stretch"))),
    ("measured", "Measured, Not Inherited", "Expectancy on your symbols",
     "The v9 audit measured the stack on 380 instruments; the Edge Study measures it on "
     "YOUR universe, through the same engine call the screen makes. Nothing about "
     "expectancy is hardcoded; until you measure, the app says \u201cnot measured\u201d.",
     (("Method", "Event study, ~15y"),
      ("Drift", "Removed causally (trailing)"),
      ("Intervals", "Block bootstrap over dates"),
      ("Power", "n_eff and MDE stated"))),
)


def render_landing_page():
    """Cold start — a description of the product, built from the product's own parts.

    Every block here uses the components the analysis pages use: a section
    header for each division, ``render_kpi_strip`` for the coverage numbers,
    and ``panel()`` for each part. The previous version was built from
    compositions that existed nowhere else in the app — a bespoke
    ``.system-card`` with a coloured top bar where every other container is a
    hairline panel, and a ``.landing-prompt`` with its own heading scale — so
    the landing page was the only page not on the section-rhythm contract. It
    read as a different product's marketing page bolted to the front of this
    one.

    The claim leads, because a reader who has not run anything needs to know
    what the thing IS before they are shown what it covers.
    """
    # ── The proposition ───────────────────────────────────────────────────
    st.markdown(
        """<div class="lede">
  <div class="lede-claim">Is the push paid for — and at what price? One indicator
    across a universe: where a stretch releases or a trend resumes, where every
    name stands between those events, and whether any of it has been worth
    anything on the symbols you are actually looking at.</div>
  <div class="lede-cta">Pick a universe and a mode in the rail, then
    <strong>Run</strong>.</div>
</div>""",
        unsafe_allow_html=True,
    )

    # ── Coverage — the app's own KPI grammar, not a bespoke number row ─────
    ui.render_section_header("Coverage", icon="layers")
    ui.render_kpi_strip(
        [
            {"label": "Universe Groups", "value": str(len(UNIVERSE_OPTIONS)),
             "subtext": "India and US indices, global benchmarks, NSE ETFs, "
                        "commodities, FX, crypto and macro"},
            {"label": "Analysis Modes", "value": "4",
             "subtext": "Single Date · Pulse Narrative · Historical Range · "
                        "Correlation"},
            {"label": "History Per Run", "value": "~4.5y",
             "subtext": "Daily; Weekly fetches deeper. Every reading is causal — "
                        "a bar depends only on bars before it"},
        ],
        max_cols=3,
        key="landing-coverage",
    )

    # ── The three parts, as panels ────────────────────────────────────────
    ui.render_section_header("System", icon="cpu")
    cols = st.columns(3, gap="small")
    for col, (cls, name, kicker, body, specs) in zip(cols, _SYSTEM_PANELS):
        with col:
            with ui.panel(f"landing-{cls}", name, context=kicker):
                st.markdown(
                    f'<div class="panel-copy">{body}</div>'
                    '<div class="panel-specs">'
                    + "".join(
                        f'<div class="lookback-row"><span class="lbl">{html.escape(k)}</span>'
                        f'<span class="val">{html.escape(v)}</span></div>'
                        for k, v in specs
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )

    # ── What a run returns ────────────────────────────────────────────────
    ui.render_section_header("What a run returns", icon="target")
    _out = (
        ("The fired events", "Every ▲ capitulation, ▼ distribution and ◆ in the last five bars, bucketed by "
                             "age, with the grid state, push, tapes and evidence behind each."),
        ("The grid and the watchlist", "Where every name stands — Buy to Exit — and the names "
                                       "in capitulation whose value has not yet turned."),
        ("A measured edge", "An event study on your own symbols: trailing drift removed, "
                            "vol-normalised, with the interval and the power stated."),
        ("The evidence", "Every computed column, with a legend that says which are signal "
                         "and which are context. Exportable."),
    )
    # ONE markdown block, not four panels. These four cards are static text, so
    # they gain nothing from a Streamlit container and lose something real to
    # it: on 1.52 the anonymous row Streamlit wraps markdown in sizes to 31px
    # around 47px of copy and will not grow, so each panel comes out ~15px
    # short and clips its own last line at `overflow: hidden` — by a different
    # amount per card, which is what makes such a grid ragged. A single grid of
    # plain divs has no wrapper to collapse.
    st.markdown(
        '<div class="outcome-grid">'
        + "".join(
            f'<div class="outcome"><div class="o-t">{html.escape(t)}</div>'
            f'<div class="o-d">{html.escape(d)}</div></div>'
            for t, d in _out
        )
        + "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS & SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SidebarState:
    """Inputs collected from the sidebar for one render frame.

    Returned by render_sidebar(). Fields are named (not positional) so adding
    or reordering inputs no longer requires updating a 16-element unpack.
    """
    universe: str
    selected_index: Optional[str]
    analysis_date: datetime.date
    reg_len: int
    wt_n1: int
    wt_n2: int
    wt2_len: int     # WT2 signal-line smoothing length (wrci.pine: "Signal Line Length")
    wt2_type: str    # WT2 signal-line MA type (wrci.pine: "Signal Line Type", ALMA default)
    levels: tuple  # (obLevel1, obLevel2, osLevel1, osLevel2)
    timeframe: str
    mode: str
    start_date: Optional[datetime.date]
    end_date: Optional[datetime.date]
    run_clicked: bool
    corr_target_ticker: Optional[str]
    corr_lookback: int
    corr_method: str
    sid: "eng.EngineSettings"     # resolved engine config for this run


def render_sidebar() -> SidebarState:
    with st.sidebar:
        # The mark, left-aligned and split so the second half carries the
        # accent. It was centred over a left-aligned column — the single most
        # common tell of a template — and drawn in the retired amber, which now
        # means caution and nothing else.
        ui.render_nav_brand("SANKET", "संकेत · Conviction × Value")

        # Analysis Depth
        st.markdown('<div class="sidebar-title">Analysis Depth</div>', unsafe_allow_html=True)
        timeframe = st.selectbox("Timeframe", TIMEFRAME_OPTIONS, key="sb_timeframe", label_visibility="collapsed")


        # Universe Selection
        st.markdown('<div class="sidebar-title">Universe Selection</div>', unsafe_allow_html=True)
        universe = st.selectbox("Universe", UNIVERSE_OPTIONS, key="sb_universe", label_visibility="collapsed")
        selected_index = None

        if universe == "India Indexes":
            selected_index = st.selectbox("Index", INDEX_LIST, index=INDEX_LIST.index("Benchmark Indexes"), key="sb_india_index", label_visibility="collapsed")
        elif universe == "Global Indexes":
            selected_index = "Global Benchmark Indexes"
        elif universe == "US Indexes":
            selected_index = st.selectbox("Index", US_INDEX_LIST, index=US_INDEX_LIST.index("DOW JONES"), key="sb_us_index", label_visibility="collapsed")
        elif universe == "ETF Index":
            selected_index = "NSE ETF Universe"
        elif universe == "Commodities":
            selected_index = "Global Commodities"
        elif universe == "Currency":
            selected_index = "Major FX Pairs"
        elif universe == "Crypto":
            selected_index = "Digital Assets (Top 20)"
        elif universe == "Global Macro":
            selected_index = "Global Macro Bonds"


        # Analysis Mode
        st.markdown('<div class="sidebar-title">Analysis Mode</div>', unsafe_allow_html=True)
        analysis_mode = st.selectbox(
            "Mode",
            ["Single Date", "Historical Range", "Correlation Analysis", "Pulse Narrative"],
            key="sb_mode",
            label_visibility="collapsed",
        )

        if analysis_mode in ["Single Date", "Pulse Narrative"]:
            st.markdown('<div class="sidebar-title">Analysis Date</div>', unsafe_allow_html=True)
            analysis_date = st.date_input("Date", _today_ist(), max_value=_today_ist(), key="sb_analysis_date", label_visibility="collapsed")
            start_date_hist, end_date_hist = None, None
            corr_target_ticker, corr_lookback, corr_method = None, 90, "Pearson"
        elif analysis_mode == "Historical Range":
            st.markdown('<div class="sidebar-title">Analysis Range</div>', unsafe_allow_html=True)
            analysis_date = _today_ist()
            today = _today_ist()
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                start_date_hist = st.date_input(
                    "Start", today - datetime.timedelta(days=300),
                    max_value=today, key="sb_start_date", label_visibility="collapsed",
                )
            with col_date2:
                end_date_hist = st.date_input(
                    "End", today, max_value=today, key="sb_end_date", label_visibility="collapsed",
                )
            corr_target_ticker, corr_lookback, corr_method = None, 90, "Pearson"
        else:  # Correlation Analysis mode
            st.markdown('<div class="sidebar-title">Analysis Date</div>', unsafe_allow_html=True)
            analysis_date = st.date_input("Analysis Date", _today_ist(), max_value=_today_ist(), key="sb_corr_date", label_visibility="collapsed")
            start_date_hist, end_date_hist = None, None

            # Target Asset Panel
            st.markdown('<div class="sidebar-title">Target Asset</div>', unsafe_allow_html=True)
            target_class = st.selectbox("Asset Class", ["Commodities", "Currency", "Crypto", "Global Indexes"], key="sb_target_class", label_visibility="collapsed")

            # Build target asset options from maps
            if target_class == "Commodities":
                target_map = COMMODITY_MAP
                target_display_names = list(COMMODITY_MAP.keys())
            elif target_class == "Currency":
                target_map = CURRENCY_MAP
                target_display_names = list(CURRENCY_MAP.keys())
            elif target_class == "Crypto":
                target_map = CRYPTO_MAP
                target_display_names = list(CRYPTO_MAP.keys())
            else:  # Global Indexes
                target_map = GLOBAL_INDEXES_MAP
                target_display_names = list(GLOBAL_INDEXES_MAP.keys())

            target_selected = st.selectbox("Asset", target_display_names, key="sb_target_asset", label_visibility="collapsed")
            corr_target_ticker = target_map.get(target_selected, target_selected)

            # Correlation params
            st.markdown('<div class="sidebar-title">Analysis Params</div>', unsafe_allow_html=True)
            corr_lookback_str = st.selectbox("Lookback", ["30D", "60D", "90D", "180D"], key="sb_corr_lookback", label_visibility="collapsed")
            corr_lookback = int(corr_lookback_str.replace("D", ""))
            corr_method = st.selectbox("Method", ["Pearson", "Spearman"], key="sb_corr_method", label_visibility="collapsed")

        # Legacy engine parameters — retained ONLY because run_full_analysis /
        # the analyzed-frame cache signature still thread them (reg_len drives the
        # ATR window; the rest are inert). Do not expose as user knobs.
        reg_len, wt_n1, wt_n2 = 20, 10, 21
        wt2_len, wt2_type = 20, "ALMA"
        obLevel1, obLevel2, osLevel1, osLevel2 = 80, 40, -80, -40

        # ── Date-range validation (Historical Range only) ──
        date_range_valid = True
        if analysis_mode == "Historical Range":
            if start_date_hist and end_date_hist and start_date_hist >= end_date_hist:
                date_range_valid = False
                st.markdown(
                    '<div style="font-family:var(--data); font-size:var(--fs-2xs); '
                    'color:var(--rose); padding:0.4rem 0 0.2rem 0; line-height:1.4;">'
                    '⚠ End date must be after start date.</div>',
                    unsafe_allow_html=True,
                )

        # Mode-specific RUN button label so users know what they're triggering.
        _RUN_LABELS = {
            "Single Date":                "◈ RUN SCREENER",
            "Pulse Narrative":            "◈ RUN PULSE",
            "Historical Range":           "◈ RUN HARVEST",
            "Correlation Analysis":       "◈ RUN CORRELATION",
        }
        run_clicked = st.button(
            _RUN_LABELS.get(analysis_mode, "◈ RUN ANALYSIS"),
            type="primary", width='stretch',
            disabled=not date_range_valid,
        )

        # Engine panel — rendered in every mode: the measured verdict for this
        # universe, the number behind it, what fires and what it costs. Returns
        # the resolved eng.EngineSettings for this run.
        #
        # It is the LAST readout in the rail. A session readout used to follow it
        # — version, universe, timeframe, mode, class — and every one of those is
        # already on the page: the command bar carries universe, timeframe and
        # as-of across the top of every loaded page, the mode is the control the
        # reader set three inches above, the class is a display label nothing
        # computes from, and the version is in the footer. A rail that repeats
        # the command bar is a second caption for the same facts.
        sid = _render_engine_status_sidebar(universe, selected_index, timeframe)

        # Appearance is the LAST control in the rail. See _render_appearance_control.
        _render_appearance_control()

        return SidebarState(
            universe=universe,
            selected_index=selected_index,
            analysis_date=analysis_date,
            reg_len=reg_len,
            wt_n1=wt_n1,
            wt_n2=wt_n2,
            wt2_len=wt2_len,
            wt2_type=wt2_type,
            levels=(obLevel1, obLevel2, osLevel1, osLevel2),
            timeframe=timeframe,
            mode=analysis_mode,
            start_date=start_date_hist,
            end_date=end_date_hist,
            run_clicked=run_clicked,
            corr_target_ticker=corr_target_ticker,
            corr_lookback=corr_lookback,
            corr_method=corr_method,
            sid=sid,
        )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SCREENER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_screener_analysis(universe, selected_index, analysis_date, reg_len, wt_n1, wt_n2, levels, timeframe, show_progress=True, external_progress_slot=None, progress_offset=0, progress_scale=100, wt2_len=20, wt2_type="ALMA", sid=None, study=None):
    """Execute the Pragati screen and return the ranked cross-section.

    Fetches market data and the macro drivers for the universe, builds each symbol's
    Pragati stack — trace, histogram, tapes, the ▲▼ and ◆, grid state — plus order-flow /
    regime context, then ranks the whole cross-section (engine.compute_ranking): a ▲▼
    fired on this bar first, then every name by stretch read as reversion.

    Args:
        sid: the run's :class:`eng.EngineSettings`; ``None`` resolves defaults for this universe.
        study: an optional :class:`edge.EdgeStudy` measured on this universe. Used for the
            cost gate and the per-row read; never to filter or scale a signal.
        external_progress_slot: Optional Streamlit container for external progress tracking (e.g., from correlation analysis)
        progress_offset: Starting percentage for external progress tracking (default 0)
        progress_scale: Scale factor for progress percentage within external slot (default 100 = full)

    Returns: DataFrame with signals ranked by priority, or None on error.
    """
    obLevel1, obLevel2, osLevel1, osLevel2 = levels
    if sid is None:
        sid = _engine_settings(universe, selected_index, timeframe)
    progress_slot = external_progress_slot if external_progress_slot is not None else (st.empty() if show_progress else None)

    if show_progress or external_progress_slot is not None:
        pct_val = progress_offset + (5 * progress_scale / 100)
        progress_bar(progress_slot, pct_val, "Initializing Engine", f"Universe: {universe}")

    console.start_phase("DATA ACQUISITION", 1, 2)
    console.section("Universe Configuration")
    console.item("Universe", universe)
    console.item("Selected Index", selected_index)
    console.item("Timeframe", timeframe)

    stock_list, msg = resolve_universe(universe, selected_index)

    if not stock_list:
        console.error(msg)
        ui_error(msg)
        return None

    console.success(f"Fetched {len(stock_list)} symbols for {selected_index}")
    console.section("Market Data Fetch")
    if show_progress or external_progress_slot is not None:
        pct_val = progress_offset + (15 * progress_scale / 100)
        progress_bar(progress_slot, pct_val, "Fetching Market Data", f"{len(stock_list)} Stocks")
    # Anchor the fetch at analysis_date (not today): the screener snaps to the
    # analysis_date bar for all signal/ranking reads (the only post-date read is the
    # display-only "% Chng Since" column, which uses the few buffer days after it).
    # For the common analysis_date == today run this is identical to before.
    end_date = analysis_date if isinstance(analysis_date, datetime.date) else _today_ist()
    data_dict, fetch_msg = get_universe_data(stock_list, end_date=end_date,
                                             timeframe=timeframe)

    if not data_dict:
        console.error(fetch_msg)
        ui_error(fetch_msg)
        return None

    console.success(f"Successfully downloaded data for {len(data_dict)} stocks")

    # The conviction ladder's intraday frames — one batch per frame, its own progress step.
    def _ip(i, n, frame):
        if show_progress or external_progress_slot is not None:
            progress_bar(progress_slot, progress_offset + (15 + 5 * i / max(n, 1)) * progress_scale / 100,
                         "Fetching Intraday Ladder", f"{frame} bars · {i + 1} / {n} frames")
    _prefetch_intraday(list(data_dict), progress=_ip, timeframe=timeframe)

    # The macro drivers behind the value ingredient — one batch for the whole universe.
    drivers = _drivers_for(end_date, timeframe)
    if drivers is None:
        console.warning("Macro drivers unavailable — the value ingredient runs unhedged")

    console.end_phase("DATA ACQUISITION")

    console.start_phase("SIGNAL SCREEN", 2, 2)

    console.section("Engine Parameters")
    console.item("Engine", f"{ENGINE_NAME} ({ENGINE_CODE})")
    console.item("Timeframe", timeframe)
    _p = sid.params
    console.item("Conviction", f"lookback {_p.length} · smooth {_p.smooth} · norm {_p.norm} · "
                               f"{_p.participation} participation (cap {_p.cap:g}x) · {_p.denominator}")
    console.item("Trace", f"conviction × value · signal EMA {_p.signal} · θ ±{sid.theta:.1f} "
                          f"(histogram calibrated after {sid.min_bars} bars)")
    console.item("Ladders", f"conviction {sid.ladder_label} · value {sid.value_ladder_label}")
    console.item("Signals", f"{sid.trigger_label} · read from the grid · cooldown {_p.cool} · "
                            f"hold {sid.horizon} bars · entry next open")
    console.item("Macro drivers", "prepared" if drivers is not None else "unavailable (unhedged)")
    _vl, _vk, _vd = _study_state(study, "buy")
    console.item("Measured edge (buy)", f"{_vl} — {_study_summary_line(study, 'buy')}")
    console.item("Measured edge (sell)", f"{_study_state(study, 'sell')[0]} — "
                                         f"{_study_summary_line(study, 'sell')}")
    console.item("Universe class", f"{sid.iclass} (display label)")
    console.item("Cost gate", f"{sid.cost_bps:.1f} bp · "
                              + ("net positive" if sid.cost_ok(study) else "NET NEGATIVE")
                              + f" · basis {sid.cost_basis(study)}")
    console.item("Instruments", f"{len(data_dict)} of {len(stock_list)} fetched successfully")
    if show_progress or external_progress_slot is not None:
        pct_val = progress_offset + (20 * progress_scale / 100)
        progress_bar(progress_slot, pct_val, "Ranking Cross-Section", f"{len(data_dict)} Stocks")

    results = []
    _failed_symbols = []
    _warmup_skipped = 0

    # If a range harvest just ran for this exact universe + params + date, its analyzed
    # frames are cached — reuse them instead of recomputing the whole per-stock pipeline.
    _cache_sig = _analysis_params_sig(timeframe, reg_len, wt_n1, wt_n2, levels,
                                      wt2_len, wt2_type, end_date, sid.params_sig)
    _cache_hits = 0

    _tf_label = "weekly" if timeframe == "Weekly" else "daily"
    _ladders = {}
    console.section(f"Signal Analysis — {len(data_dict)} {_tf_label} instruments")

    for i, (ticker, df) in enumerate(data_dict.items()):
        try:
            pct = int(progress_offset + (20 + (i + 1) / len(data_dict) * 75) * progress_scale / 100)
            if show_progress or external_progress_slot is not None:
                progress_bar(progress_slot, pct, "Analyzing Instruments", f"{i + 1} / {len(data_dict)} Stocks")

            _cached = _analyzed_cache_get(ticker, _cache_sig)
            if _cached is not None:
                # Copy so the screener's own column additions never mutate the cache.
                # Analysis adds columns, not rows, so the cached frame's length equals
                # the resampled input — the insufficient-data guard below still applies.
                df = _cached.copy()
                _cache_hits += 1
            else:
                _daily = df
                if timeframe == "Weekly":
                    df = resample_to_weekly(df)

            # Warmup guard — a symbol cannot carry a signal until the histogram is
            # calibrated (the Pine's `ready` gate: lookback + participation baseline + two
            # normalization windows + the signal EMA). Applied on both cache hit and miss
            # so a short frame cached by the (unguarded) harvest can't slip through.
            _min_bars = max(reg_len + 30, sid.min_bars)
            if len(df) < _min_bars:
                console.detail(f"{ticker}: Skipped (warming up: {len(df)} of {_min_bars} bars needed)")
                _warmup_skipped += 1
                continue

            if _cached is None:
                df = run_full_analysis(df, reg_len, wt_n1, wt_n2, obLevel1, obLevel2, osLevel1, osLevel2,
                                       wt2_len=wt2_len, wt2_type=wt2_type, sid=sid,
                                       drivers=drivers, symbol=ticker,
                                       daily=_daily if timeframe == "Weekly" else None)
                df = run_regime_analysis(df)        # adds HMM_Bull/Bear, Vol_Regime, Change_Point, Regime_Confidence

            # Sample at analysis_date — snap to the correct historical bar.
            # Weekly resampling re-labels bars to week-start Mondays, so an exact
            # match on a non-Monday selection would fail; 'pad' snaps any date back
            # to the most recent bar at-or-before it (the bar the date falls within).
            # This is what makes historical weekly snapshots work — without it a miss
            # silently fell through to len(df)-1, i.e. the live/current bar.
            df.index = pd.to_datetime(df.index)
            target_dt = pd.to_datetime(analysis_date)

            _pos = df.index.get_indexer([target_dt], method='pad')[0]
            if _pos == -1:
                # Requested date precedes all available history — nothing to snap to.
                console.detail(f"{ticker}: analysis_date {analysis_date} precedes available history — skipped")
                continue
            idx_pos = int(_pos)
            if df.index[idx_pos] != target_dt:
                console.detail(f"{ticker}: snapped {analysis_date} → bar {df.index[idx_pos].date()}")

            if idx_pos < 5:
                continue

            last_row = df.iloc[idx_pos]

            # Recent return volatility — the asset-agnostic σ scale used to report how far
            # price has run since an aged signal fired.
            try:
                _retvol20 = float(df['Close'].pct_change().rolling(20).std().iloc[idx_pos])
            except Exception:
                _retvol20 = float('nan')

            signal_type = _classify_signal_type(last_row)

            # Clean display names
            simple_name = ticker.replace(".NS", "").lstrip("^")
            friendly_name = ASSET_NAME_LOOKUP.get(ticker)
            if friendly_name:
                display_name = f"{ticker} ({friendly_name})"
            else:
                display_name = simple_name

            # Calculate % change from previous close (day-over-day)
            prev_close = df.iloc[idx_pos - 1]['Close'] if idx_pos > 0 else last_row['Close']
            pct_change = ((last_row['Close'] - prev_close) / prev_close * 100) if prev_close > 0 else 0.0

            # Calculate % change since analysis date if it's in the past relative to latest bar.
            # Use None sentinel for missing data so downstream display can show "—" rather than 0.0.
            pct_chng_since = None
            if idx_pos < len(df) - 1:
                analysis_price = last_row['Close']
                latest_price = df.iloc[-1]['Close']
                if pd.notna(analysis_price) and pd.notna(latest_price) and analysis_price > 0:
                    pct_chng_since = round((latest_price - analysis_price) / analysis_price * 100, 2)

            def _num(col, default=0.0, nd=None):
                v = last_row.get(col, default)
                if v is None or pd.isna(v):
                    return default
                return round(float(v), nd) if nd is not None else float(v)

            row = {
                "% Chng Since": pct_chng_since,  # None when data unavailable — displays as NaN / "—"
                "Symbol": ticker,
                "DisplayName": display_name,
                "SimpleName": simple_name,
                # Signal == the trace, ±100: how far the move is stretched, in one-sided
                # effort and in price against fair value. The level; the EVENTS are below.
                "Signal": _num('PRG_Trace', np.nan, 2),
                "Bar_Delta": _num('Bar_Delta', 0.0, 2),
                "CVD": _num('CVD', 0.0, 2),
                "CVD_Slope": _num('CVD_Slope', 0.0, 2),
                "Delta_Z": _num('Delta_Z', 0.0, 2),
                "Abs_Strength": _num('Abs_Strength', 0.0, 2),
                "Buy_Share": _num('Buy_Share', 0.5, 3),
                "Absorption_Score": _num('Absorption_Score', 0.0, 3),
                "Zone": last_row['Condition'],
                "SignalType": signal_type,
                "Price": round(last_row['Close'], 2),
                "PctChange": round(pct_change, 2),
                "RetVol20":      _retvol20,
                "HMM_Bull":      _num('HMM_Bull', 0.33),
                "HMM_Bear":      _num('HMM_Bear', 0.33),
                "Vol_Regime":    str(last_row.get('Vol_Regime', 'NORMAL')),
                "Change_Point":  bool(last_row.get('Change_Point', False)),
                "Regime_Confidence": _num('Regime_Confidence', 0.0),
                "F1_PriceMom":   _num('F1_PriceMom', 0.0),
                "F2_VolQual":    _num('F2_VolQual', 0.0),
                "ATR_Pct":       last_row.get('ATR_Pct'),
                "MA_Alignment": int(last_row.get('MA_Alignment', 0)),
            }
            # The engine's own fields for this bar — readings, events by age, grid state.
            row.update(eng.snapshot(df, idx_pos, sid))
            results.append(row)

            console.detail(f"[{i+1}/{len(data_dict)}] {ticker}: trace={_num('PRG_Trace', float('nan')):+.1f}  "
                           f"state={last_row.get('PRG_State', '—')}  grid={last_row.get('CVG_Action', '—')}  "
                           f"C{ {'down': '↓', 'up↺': '↺'}.get(str(last_row.get('PRG_Ladder', '')), '')}={_num('PRG_CTape', float('nan')):+.0f} "
                           f"V={_num('PRG_VTape', float('nan')):+.0f}")
            _ladders[str(last_row.get('PRG_Ladder', '') or 'warming')] = _ladders.get(str(last_row.get('PRG_Ladder', '') or 'warming'), 0) + 1

        except Exception as e:
            console.failure(f"Analysis Failed: {ticker}", str(e))
            _failed_symbols.append(ticker)
            continue

    if _ladders:
        console.item("Conviction ladder", " · ".join(f"{k} {v}" for k, v in sorted(_ladders.items())))
    console.end_phase("SIGNAL SCREEN")
    if _cache_hits:
        console.detail(f"Analyzed-frame cache: reused {_cache_hits}/{len(data_dict)} frames from the range harvest (skipped re-analysis)")
    if _warmup_skipped:
        console.detail(f"Warmup: {_warmup_skipped} symbol(s) skipped — fewer than {sid.min_bars} bars, so the histogram is not calibrated")
    # One-shot cache — release the harvested frames now that the screener has consumed them.
    _analyzed_cache_clear()

    _fail_count = len(_failed_symbols)
    console.summary("RUN SUMMARY", {
        "Universe": universe,
        "Universe Index": selected_index,
        "Instrument Class": sid.iclass,
        "Total Symbols": len(stock_list),
        "Data Success": len(data_dict),
        "Analyzed Stocks": len(results),
        "Warming Up": _warmup_skipped,
        "Failed Symbols": f"{_fail_count} ({', '.join(_failed_symbols[:5])}{'…' if _fail_count > 5 else ''})" if _fail_count else "0",
        "Analysis Date": analysis_date,
        "Status": "COMPLETE",
    })
    # Surface run stats so body renders can show "47 / 50 symbols · Daily · 2025-01-15"
    st.session_state["screener_run_stats"] = {
        "total_in_universe": len(stock_list),
        "data_fetched":      len(data_dict),
        "analyzed":          len(results),
        "failed":            _fail_count,
        "warming_up":        _warmup_skipped,
        "drivers":           drivers is not None,
        # analysed (past warm-up) but a tape still calibrating — no signal can fire on them
        "paused":            sum(1 for r in results if not r.get("PRG_Stack_OK", False)),
    }
    console.line('═', 70)

    if show_progress or external_progress_slot is not None:
        # If this run owns the tail of the bar (offset+scale reaches 100), show a
        # clean 100%; otherwise cap at 95% of the slice for a following phase.
        _tail = (progress_offset + progress_scale) >= 100
        pct_val = 100 if (external_progress_slot is None or _tail) else int(progress_offset + 95 * progress_scale / 100)
        progress_bar(progress_slot, pct_val, "Analysis Complete", f"{len(results)} Stocks Analyzed")
        if show_progress and external_progress_slot is None:
            progress_slot.empty()

    if not results:
        _n_fetched = len(data_dict)
        _n_total   = len(stock_list)
        if _n_fetched == 0:
            ui_warning(
                f"**No market data retrieved** for {selected_index} as of {analysis_date}. "
                "The exchange may have been closed, or yfinance may be rate-limiting. "
                "Try refreshing or selecting a recent trading day."
            )
        elif _warmup_skipped >= _n_fetched:
            ui_warning(
                f"**Every symbol is still warming up.** The trace's histogram needs {sid.min_bars} "
                f"{'weekly' if timeframe == 'Weekly' else 'daily'} bars before its lookback, "
                f"participation baseline and both normalization windows are calibrated, and none "
                f"of the {_n_fetched} symbols in {selected_index} has that much history as of "
                f"{analysis_date}. Try the Daily timeframe, or a universe with longer-listed "
                "instruments."
            )
        else:
            ui_info(
                f"**Nothing to show** — {_n_fetched} of {_n_total} symbols had data for {analysis_date}, "
                "but none produced a usable Pragati reading. "
                "Try an adjacent trading date, or check that the selected date is a market session."
            )
        # Return empty DataFrame with expected columns to prevent downstream KeyErrors
        expected_cols = (["Symbol", "DisplayName", "SimpleName", "Signal",
                          "Bar_Delta", "CVD", "CVD_Slope", "Delta_Z", "Buy_Share",
                          "Absorption_Score", "Zone", "SignalType", "Price", "PctChange",
                          "BUY_Today", "BUY_1d", "BUY_2d", "BUY_3d", "BUY_5d",
                          "SELL_Today", "SELL_1d", "SELL_2d", "SELL_3d", "SELL_5d",
                          "MA_Alignment"]
                         + list(eng.SNAPSHOT_COLUMNS) + list(eng.RANK_CONTRACT))
        return pd.DataFrame(columns=list(dict.fromkeys(expected_cols)))

    results_df = pd.DataFrame(results)

    # Cross-sectional ranking (engine.py). A ▲▼ fired on this bar ranks first on its side,
    # then every name by stretch read as reversion (−trace for the long side, +trace for the
    # short) — measured by trace_study.py, which found the grid-weight ranking ran backwards
    # on NSE universes. The measured
    # expectancy (`study`) informs the cost gate and the per-row read; it never scales or
    # filters a signal. One call emits the whole UI contract.
    if not results_df.empty:
        results_df = eng.compute_ranking(results_df, sid, study=study)

    return results_df


def run_timeseries_analysis(universe, selected_index, start_date, end_date, reg_len, wt_n1, wt_n2, levels, timeframe, wt2_len=20, wt2_type="ALMA",
                            external_progress_slot=None, progress_offset=0, progress_scale=100,
                            sid=None, study=None):
    """Compute the per-(date, symbol) Pragati frame for a date range.

    Pure compute path: fetches history, runs the full / regime analyses on every symbol,
    builds the per-(date, symbol) row set with forward-return labels, and stores
    ts_results_df + ts_meta in ``st.session_state``. **Does not render UI.** The dashboard
    is rendered separately by ``render_timeseries_dashboard()`` so it survives sidebar
    interactions / reruns.

    ``external_progress_slot`` lets a caller share one progress bar over
    [offset, offset+scale] instead of stacking a second bar.
    """
    if sid is None:
        sid = _engine_settings(universe, selected_index, timeframe)
    _own_slot = external_progress_slot is None
    progress_slot = st.empty() if _own_slot else external_progress_slot
    def _p(pct, label, sub):
        progress_bar(progress_slot, int(progress_offset + pct * progress_scale / 100), label, sub)
    _p(5, "Fetching Historical Depth", f"{start_date} to {end_date}")

    console.start_phase("HISTORICAL ACQUISITION", 1, 2)
    console.section("Range Configuration")
    console.item("Universe", universe)
    console.item("Selected Index", selected_index)
    console.item("Start Date", start_date)
    console.item("End Date", end_date)
    console.item("Timeframe", timeframe)

    stock_list, _ = resolve_universe(universe, selected_index)

    if not stock_list:
        console.error("Failed to retrieve stock list")
        ui_error("Failed to retrieve stock list")
        return

    console.success(f"Fetched {len(stock_list)} symbols for {selected_index}")
    console.section("Mass Historical Download")
    # Registry-first: if the same universe was fetched recently it won't hit yfinance again
    data_dict, msg = get_universe_data(stock_list, end_date=end_date,
                                       timeframe=timeframe)

    if not data_dict:
        console.error("No historical data available")
        ui_error("No historical data available for selected range.")
        return

    console.success(f"Downloaded depth for {len(data_dict)} entities")
    _prefetch_intraday(list(data_dict),
                       progress=lambda i, n, f: _p(5 + 10 * i / max(n, 1), "Fetching Intraday Ladder",
                                                   f"{f} bars · {i + 1} / {n} frames"),
                       timeframe=timeframe)
    drivers = _drivers_for(end_date, timeframe)
    if drivers is None:
        console.warning("Macro drivers unavailable — the value ingredient runs unhedged")

    # Start Unified Harvesting Phase
    console.start_phase("SIGNAL HARVEST", 2, 2)
    start_harvest = time.time()

    _p(15, "Harvesting Signals", f"{len(data_dict)} Stocks")
    all_results = []

    # Analyzed-frame cache for this run — lets a screener that follows skip
    # re-running the identical per-stock analysis pipeline (see helper comment).
    _cache_sig = _analysis_params_sig(timeframe, reg_len, wt_n1, wt_n2, levels,
                                      wt2_len, wt2_type, end_date, sid.params_sig)
    _analyzed_cache_reset(_cache_sig)

    for i, (ticker, df) in enumerate(data_dict.items()):
        try:
            elapsed = time.time() - start_harvest
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (len(data_dict) - (i + 1))
            eta_str = time.strftime("%M:%S", time.gmtime(remaining))

            # Local 15% -> 85% band for the per-symbol harvest loop.
            pct = 15 + (i + 1) / len(data_dict) * 70
            _p(pct, "Harvesting Signals", f"{i + 1} / {len(data_dict)} Symbols · ETA {eta_str}")
            _daily = df
            if timeframe == "Weekly":
                df = resample_to_weekly(df)
            df = run_full_analysis(df, reg_len, wt_n1, wt_n2, *levels,
                                   wt2_len=wt2_len, wt2_type=wt2_type, sid=sid,
                                   drivers=drivers, symbol=ticker,
                                   daily=_daily if timeframe == "Weekly" else None)
            df = run_regime_analysis(df)
            # Cache the analyzed frame so run_screener_analysis can reuse it instead
            # of recomputing. Stored by reference — the harvest-only columns appended
            # below (Ret_*, SignalType) are harmless extras; the screener copies on read.
            _analyzed_cache_put(ticker, df, _cache_sig)

            # Forward-return labels at the declared horizons (10 bars is the hold; 1, 5
            # and 21 bracket it). Labels only — never signal inputs.
            for h in eng.HOLD_HORIZONS:
                df[f'Ret_{h}b'] = df['Close'].shift(-h) / df['Close'] - 1

            # Vectorized SignalType per bar — a fired event wins, else the flow zone.
            df['SignalType'] = np.where(df['PRG_Event'] != '', df['PRG_Event'],
                                        np.where(df['Condition'] != 'Neutral', df['Condition'], '-'))

            mask = (df.index.date >= start_date) & (df.index.date <= end_date)
            range_df = df.loc[mask]

            for date, row in range_df.iterrows():
                all_results.append({
                    'Date': date,
                    'Symbol': ticker,
                    # Signal == the trace (±100). The EVENTS are below.
                    'Signal': row.get('PRG_Trace'),
                    'Hist_Z': row.get('PRG_Hist_Z'),
                    'CTape': row.get('PRG_CTape'),
                    'VTape': row.get('PRG_VTape'),
                    'Push': row.get('PRG_Push', 0),
                    'Stack_OK': bool(row.get('PRG_Stack_OK', False)),
                    'Action': row.get('CVG_Action', 'Unread'),
                    'Cell': row.get('CVG_Cell', cg.UNREAD),
                    'Units': row.get('CVG_Units', 1.0),
                    'CVG_Side': row.get('CVG_Side', 0),
                    'Held': bool(row.get('CVG_Held', False)),
                    'Armed': row.get('PRG_Armed', 0),
                    'Event': row.get('PRG_Event', ''),
                    'Bar_Delta': row['Bar_Delta'],
                    'CVD': row['CVD'],
                    'CVD_Slope': row['CVD_Slope'],
                    'Delta_Z': row['Delta_Z'],
                    'Zone': row['Condition'],
                    # BuySignal / SellSignal are the aggregation-facing names the range
                    # dashboard counts per day: every long / short event.
                    'BuySignal': bool(row['long_cond']),
                    'SellSignal': bool(row['short_cond']),
                    'TurnBuy': bool(row['turn_buy']),
                    'TurnSell': bool(row['turn_sell']),
                    'ResumeLong': bool(row['resume_long']),
                    'ResumeShort': bool(row['resume_short']),
                    'SignalType': row['SignalType'],
                    # Regime risk context (never a signal input)
                    'Regime': row.get('Regime', 'NEUTRAL'),
                    'HMM_Bull': row.get('HMM_Bull', 0),
                    'HMM_Bear': row.get('HMM_Bear', 0),
                    'Vol_Regime': row.get('Vol_Regime', 'NORMAL'),
                    'Change_Point': row.get('Change_Point', False),
                    'Regime_Confidence': row.get('Regime_Confidence', 0),
                    # Forward returns (labels; horizons = engine.HOLD_HORIZONS)
                    **{f'Ret_{h}b': row.get(f'Ret_{h}b') for h in eng.HOLD_HORIZONS},
                    'F1_PriceMom': row.get('F1_PriceMom', 0),
                    'F2_VolQual': row.get('F2_VolQual', 0),
                    'ATR_Pct':     row.get('ATR_Pct'),
                    'Close':       row.get('Close'),
                })

        except Exception as e:
            console.failure(f"Range Analysis Failed: {ticker}", str(e))
            continue

    console.success(f"Successfully processed {len(data_dict)} symbols for historical depth")
    console.end_phase("SIGNAL HARVEST")

    if not all_results:
        if _own_slot:
            progress_slot.empty()
        ui_error("No results generated for the selected timeframe.")
        return

    ts_df = pd.DataFrame(all_results)
    ts_df['Date'] = pd.to_datetime(ts_df['Date'])
    ts_df = ts_df.sort_values('Date')

    daily_agg, summary = _aggregate_timeseries(ts_df)

    console.summary("HISTORICAL RANGE SUMMARY", {
        "Universe": universe,
        "Universe Index": selected_index,
        "Instrument Class": sid.iclass,
        "Historical Range": f"{start_date} to {end_date}",
        "Total Signals Fired": summary['total_signals'],
        "Long / Short": f"{summary['total_buys']} / {summary['total_sells']}",
        "▲▼ / ◆": f"{summary['total_turns']} / {summary['total_resumes']}",
        "Avg Trace": round(summary['avg_signal'], 2),
        "Long:Short Ratio": round(summary['overall_ratio'], 2),
        "Dominant Zone": summary['most_common_zone'],
        "HMM Regime": summary['dominant_regime'],
        "Status": "HARVEST COMPLETE"
    })
    console.line('═', 70)

    st.session_state["timeseries_done"] = True
    st.session_state["ts_results_df"] = ts_df
    st.session_state["ts_meta"] = {
        "universe":       universe,
        "selected_index": selected_index,
        "start_date":     start_date,
        "end_date":       end_date,
        "timeframe":      timeframe,
        "iclass":         sid.iclass,
        "length":         sid.params.length,
        "trigger":        sid.trigger_label,
    }

    # Only clear our OWN bar. When sharing the Single-Date bar, the screener that
    # follows keeps rendering into it (the 40→100% phase).
    if _own_slot:
        progress_slot.empty()


# ══════════════════════════════════════════════════════════════════════════════
# TIMESERIES — AGGREGATION + DASHBOARD RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def _aggregate_timeseries(ts_df):
    """Aggregate the per-(date, symbol) Pragati frame into daily metrics + summary stats.

    Pure function — used by both ``run_timeseries_analysis`` (for the console
    summary on harvest) and ``render_timeseries_dashboard`` (re-rendered on every
    Streamlit run from session state, so sidebar interactions don't lose the view).
    """
    ts_df = ts_df.copy()
    for c in ('TurnBuy', 'TurnSell', 'ResumeLong', 'ResumeShort', 'BuySignal', 'SellSignal'):
        if c not in ts_df.columns:
            ts_df[c] = False
        ts_df[c] = ts_df[c].fillna(False).astype(bool)
    if 'Action' not in ts_df.columns:
        ts_df['Action'] = 'Unread'
    if 'CVG_Side' not in ts_df.columns:
        ts_df['CVG_Side'] = 0
    ts_df['_read'] = ts_df['Action'] != 'Unread'
    ts_df['_build'] = ts_df['_read'] & (pd.to_numeric(ts_df['CVG_Side'], errors='coerce') > 0)
    ts_df['_cut'] = ts_df['_read'] & (pd.to_numeric(ts_df['CVG_Side'], errors='coerce') < 0)
    ts_df['_armed_up'] = pd.to_numeric(ts_df.get('Armed', 0), errors='coerce') > 0
    ts_df['_armed_dn'] = pd.to_numeric(ts_df.get('Armed', 0), errors='coerce') < 0

    g = ts_df.groupby('Date')
    daily_agg = g.agg({
        'Signal': 'mean',
        'CVD': 'mean',
        'CVD_Slope': 'mean',
        'BuySignal': 'sum',
        'SellSignal': 'sum',
        'TurnBuy': 'sum',
        'TurnSell': 'sum',
        'ResumeLong': 'sum',
        'ResumeShort': 'sum',
        '_read': 'sum',
        '_build': 'sum',
        '_cut': 'sum',
        '_armed_up': 'sum',
        '_armed_dn': 'sum',
        'Zone': lambda x: x.value_counts().idxmax() if len(x) > 0 else 'Neutral',
        'Regime': lambda x: x.value_counts().idxmax() if len(x) > 0 else 'NEUTRAL',
        'HMM_Bull': 'mean',
        'HMM_Bear': 'mean',
        'Vol_Regime': lambda x: x.value_counts().idxmax() if len(x) > 0 else 'NORMAL',
        'Change_Point': 'sum',
        'Regime_Confidence': 'mean',
    })
    for c in ('CTape', 'VTape'):
        daily_agg[f'Avg_{c}'] = g[c].mean() if c in ts_df.columns else np.nan

    daily_agg['TotalSignals'] = daily_agg['BuySignal'] + daily_agg['SellSignal']
    daily_agg['B_S_Ratio']    = np.where(
        daily_agg['SellSignal'] == 0,
        np.nan,                          # undefined (all long, no short) — NaN in charts
        daily_agg['BuySignal'] / daily_agg['SellSignal'],
    )

    # Signal breadth: % of the universe firing each side on a given day.
    total_per_day = g.size()
    daily_agg['Buy_Breadth_Pct']  = (daily_agg['BuySignal']  / total_per_day * 100).fillna(0)
    daily_agg['Sell_Breadth_Pct'] = (daily_agg['SellSignal'] / total_per_day * 100).fillna(0)

    # Grid breadth: of the names the grid can read, the share whose cell builds the position
    # (Buy / Add / Accumulate) and the share whose cell cuts it (Trim / Reduce / Exit). The
    # remainder holds (Hold / Wait).
    _read = daily_agg['_read'].where(daily_agg['_read'] > 0)
    daily_agg['Build_Pct'] = (daily_agg['_build'] / _read * 100).fillna(0)
    daily_agg['Cut_Pct']   = (daily_agg['_cut'] / _read * 100).fillna(0)
    daily_agg['Armed_Up']  = daily_agg['_armed_up']
    # Tone share — the grid's five tones (Pragyam's CVG_TONE) as a share of names read.
    if 'Cell' in ts_df.columns:
        _cells = pd.to_numeric(ts_df['Cell'], errors='coerce').fillna(cg.UNREAD).astype(int)
        _tone = _cells.map(lambda k: cg.TONES[k]).where(_cells != cg.UNREAD)
        _share = (pd.crosstab(ts_df['Date'], _tone).reindex(daily_agg.index).fillna(0))
        _den = _share.sum(axis=1).where(lambda x: x > 0)
        for _t in charts.TONE_ORDER:
            daily_agg[f'Tone_{_t}'] = ((_share[_t] if _t in _share.columns else 0) / _den * 100).fillna(0)
    daily_agg['Armed_Dn']  = daily_agg['_armed_dn']

    # Flow-zone breadth: % of names in accumulation vs distribution each day.
    acc_counts  = g['Zone'].apply(lambda x: (x.isin(['Accumulation+', 'Accumulation'])).sum())
    dist_counts = g['Zone'].apply(lambda x: (x.isin(['Distribution+', 'Distribution'])).sum())
    daily_agg['Oversold_Pct']   = (dist_counts / total_per_day * 100).fillna(0)
    daily_agg['Overbought_Pct'] = (acc_counts  / total_per_day * 100).fillna(0)

    regime_bull  = g['Regime'].apply(lambda x: x.str.contains('BULL', na=False).sum())
    regime_bear  = g['Regime'].apply(lambda x: x.str.contains('BEAR', na=False).sum())
    regime_trans = g['Regime'].apply(lambda x: (x == 'TRANSITION').sum())
    daily_agg['Regime_Bull_Pct']       = (regime_bull  / total_per_day * 100).fillna(0)
    daily_agg['Regime_Bear_Pct']       = (regime_bear  / total_per_day * 100).fillna(0)
    daily_agg['Regime_Transition_Pct'] = (regime_trans / total_per_day * 100).fillna(0)

    # Mean grid units of the names that fired — did the day's events land on names the grid
    # already favoured (Buy · turn, Add) or on names it was cutting?
    _fired = ts_df[ts_df['BuySignal'] | ts_df['SellSignal']]
    if len(_fired) and 'Units' in _fired.columns:
        daily_agg['Avg_Fired_Units'] = _fired.groupby('Date')['Units'].mean()
    else:
        daily_agg['Avg_Fired_Units'] = np.nan
    daily_agg = daily_agg.drop(columns=['_read', '_build', '_cut', '_armed_up', '_armed_dn'])

    _n_buys  = int(daily_agg['BuySignal'].sum())
    _n_sells = int(daily_agg['SellSignal'].sum())
    summary = {
        'total_signals':       int(daily_agg['TotalSignals'].sum()),
        'total_buys':          _n_buys,
        'total_sells':         _n_sells,
        'total_turns':         int(daily_agg['TurnBuy'].sum() + daily_agg['TurnSell'].sum()),
        'total_resumes':       int(daily_agg['ResumeLong'].sum() + daily_agg['ResumeShort'].sum()),
        'avg_signal':          float(daily_agg['Signal'].mean()) if daily_agg['Signal'].notna().any() else float('nan'),
        'overall_ratio':       float(_n_buys / max(_n_sells, 1)),
        'avg_buy_breadth':     float(daily_agg['Buy_Breadth_Pct'].mean()),
        'avg_sell_breadth':    float(daily_agg['Sell_Breadth_Pct'].mean()),
        'avg_build_pct':       float(daily_agg['Build_Pct'].mean()),
        'avg_cut_pct':         float(daily_agg['Cut_Pct'].mean()),
        'avg_fired_units':     float(daily_agg['Avg_Fired_Units'].mean()) if daily_agg['Avg_Fired_Units'].notna().any() else float('nan'),
        'most_common_zone':    ts_df['Zone'].mode()[0]   if len(ts_df['Zone'].mode())   > 0 else 'Neutral',
        'dominant_regime':     ts_df['Regime'].mode()[0] if len(ts_df['Regime'].mode()) > 0 else 'NEUTRAL',
        'avg_oversold':        float(daily_agg['Oversold_Pct'].mean()),
        'avg_overbought':      float(daily_agg['Overbought_Pct'].mean()),
        'avg_bull_regime':     float(daily_agg['Regime_Bull_Pct'].mean()),
        'avg_bear_regime':     float(daily_agg['Regime_Bear_Pct'].mean()),
        'total_change_points': int(daily_agg['Change_Point'].sum()),
    }
    return daily_agg, summary


def render_timeseries_dashboard():
    """Render the bulk-range dashboard from ``ts_results_df`` + ``ts_meta`` in session state.

    Called from ``main()`` whenever ``timeseries_done`` is True and the active
    mode wants the dashboard. Re-renders on every Streamlit run, so sidebar
    interactions don't blank the view.
    """
    ts_df = st.session_state.get("ts_results_df")
    meta  = st.session_state.get("ts_meta") or {}
    if ts_df is None or ts_df.empty:
        return

    start_date = meta.get('start_date')
    end_date   = meta.get('end_date')
    timeframe  = meta.get('timeframe', 'Daily')

    daily_agg, summary = _aggregate_timeseries(ts_df)
    timeframe_label    = "Weekly Average" if timeframe == 'Weekly' else "Daily Average"

    range_label = (f"{start_date} to {end_date}"
                   if start_date and end_date
                   else f"{len(daily_agg)} periods")
    _iclass  = meta.get('iclass', '—')
    _trigger = meta.get('trigger') or "▲ capitulation · ▼ distribution · ◆ RESUME"
    ui.render_section_header(
        f"Historical Range ({range_label})",
        f"{ENGINE_NAME} · {_trigger} · {_iclass}",
        icon="history", accent="violet",
    )

    # ── Summary metric row (6 cards, mirrors single-date / pulse cadence) ──
    _avg_u = summary.get('avg_fired_units', float('nan'))
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        ui.render_metric_card("Signals Fired", str(summary['total_signals']),
                              f"{summary['total_buys']} long · {summary['total_sells']} short", "info")
    with c2:
        ui.render_metric_card("▲▼ / ◆", f"{summary['total_turns']} / {summary['total_resumes']}",
                              "capitulation · distribution / continuations", "violet")
    with c3:
        ui.render_metric_card("Avg Build-Side", f"{summary['avg_build_pct']:.0f}%",
                              f"{timeframe_label} · grid Buy / Add / Accumulate", "success")
    with c4:
        ui.render_metric_card("Avg Cut-Side", f"{summary['avg_cut_pct']:.0f}%",
                              f"{timeframe_label} · grid Trim / Reduce / Exit", "danger")
    with c5:
        ui.render_metric_card("Units at Fire", f"{_avg_u:.2f}u" if np.isfinite(_avg_u) else "—",
                              "mean grid weight of the names that fired", "warning")
    with c6:
        ui.render_metric_card("Trading Days", str(len(daily_agg)), "Analyzed", "neutral")


    tab1, tab2, tab3, tab4 = st.tabs([
        "Signal Dashboard",
        "Grid Dynamics",
        "Regime Analysis",
        "Data Terminal",
    ])

    # ── TAB 1 · Signal Dashboard ───────────────────────────────────────────
    with tab1:
        ui.render_section_header("Signal Breadth",
                                 "% of universe firing a long (▲ capitulation / ◆ ↑) or short "
                                 "(▼ distribution / ◆ ↓) event",
                                 icon="activity", accent="cyan")
        fig_breadth = go.Figure()
        fig_breadth.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Buy_Breadth_Pct'],
                                         mode='lines', name='Long %',
                                         fill='tozeroy', fillcolor=chart_rgba('emerald', 0.12),
                                         line=dict(color=chart_color('emerald'), width=2)))
        fig_breadth.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Sell_Breadth_Pct'],
                                         mode='lines', name='Short %',
                                         fill='tozeroy', fillcolor=chart_rgba('rose', 0.12),
                                         line=dict(color=chart_color('rose'), width=2)))
        _pct_raw = max(daily_agg['Buy_Breadth_Pct'].max(), daily_agg['Sell_Breadth_Pct'].max())
        _pct_raw = float(_pct_raw) if pd.notna(_pct_raw) and np.isfinite(_pct_raw) else 0.0
        ymax = max(_pct_raw * 1.15, 5.0)   # floor at 5 so axis always renders sensibly
        fig_breadth.update_layout(title='', height=350, hovermode='x unified',
                                  yaxis=dict(range=[0, ymax], title='% of Universe'))
        apply_chart_theme(fig_breadth)
        ui.render_chart_panel(fig_breadth, key='breadth', context=_chart_ctx())

        st.markdown("<br>", unsafe_allow_html=True)
        ui.render_section_header("Events by Kind", "▲ capitulation and ▼ distribution, ◆ continuations "
                                 "per session, long up and short down",
                                 icon="bar-chart", accent="info")
        fig_counts = go.Figure()
        for col, name, color, sign in (('TurnBuy', '▲ capitulation', 'emerald', 1), ('ResumeLong', '◆ RESUME ↑', 'cyan', 1),
                                       ('TurnSell', '▼ distribution', 'rose', -1), ('ResumeShort', '◆ RESUME ↓', 'amber', -1)):
            fig_counts.add_trace(go.Bar(x=daily_agg.index, y=sign * daily_agg[col], name=name,
                                        marker=dict(color=chart_color(color))))
        fig_counts.update_layout(title='', height=300, hovermode='x unified', barmode='relative')
        apply_chart_theme(fig_counts)
        ui.render_chart_panel(fig_counts, key='signal_counts', context=_chart_ctx())

        st.markdown("<br>", unsafe_allow_html=True)
        ui.render_section_header("Watchlist Size", "Names in capitulation whose value is still "
                                 "cheapening — the ▲ comes when it turns",
                                 icon="eye", accent="violet")
        fig_arm = go.Figure()
        fig_arm.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Armed_Up'], mode='lines',
                                     name='▲ watch (capitulation)', line=dict(color=chart_color('emerald'), width=2)))
        fig_arm.update_layout(title='', height=260, hovermode='x unified', yaxis=dict(title='# Symbols'))
        apply_chart_theme(fig_arm)
        ui.render_chart_panel(fig_arm, key='armed', context=_chart_ctx())

    # ── TAB 2 · Grid Dynamics ──────────────────────────────────────────────
    with tab2:
        ui.render_section_header("Grid Breadth",
                                 "Share of the readable universe whose grid cell builds (Buy / Add / "
                                 "Accumulate) vs cuts (Trim / Reduce / Exit)",
                                 icon="grid", accent="emerald")
        _tone_cols = [f'Tone_{t}' for t in charts.TONE_ORDER]
        if all(c in daily_agg.columns for c in _tone_cols):
            _ts = daily_agg[_tone_cols].rename(columns=lambda c: c[5:])
            ui.render_chart_panel(charts.create_tone_history(_ts), key='grid_breadth',
                                  context=_chart_ctx('% of names read'))
            ui.render_note("Stacked in the order a book reads the grid — **build** (emerald), "
                           "**watched** (cyan), **no edge** (grey), **caution** (amber), **cut** "
                           "(rose). A band thickening toward the top is the universe climbing "
                           "the grid; toward the bottom, sliding down it.")
        else:
            fig_grid = go.Figure()
            fig_grid.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Build_Pct'], mode='lines',
                                          name='Build-side %', line=dict(color=chart_color('emerald'), width=2)))
            fig_grid.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Cut_Pct'], mode='lines',
                                          name='Cut-side %', line=dict(color=chart_color('rose'), width=2)))
            fig_grid.update_layout(title='', height=320, hovermode='x unified',
                                   yaxis=dict(range=[0, 100], title='% of readable universe'))
            apply_chart_theme(fig_grid)
            ui.render_chart_panel(fig_grid, key='grid_breadth', context=_chart_ctx())

        st.markdown("<br>", unsafe_allow_html=True)
        ui.render_section_header("The Two Tapes, Universe Mean",
                                 "Mean MTF conviction (who controls) and MTF value (+ rich / − cheap)",
                                 icon="activity", accent="violet")
        fig_tapes = go.Figure()
        fig_tapes.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Avg_CTape'], mode='lines',
                                       name='Conviction tape', line=dict(color=chart_color('accent'), width=2)))
        fig_tapes.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Avg_VTape'], mode='lines',
                                       name='Value tape', line=dict(color=chart_color('violet'), width=2)))
        fig_tapes.add_hline(y=0, line=dict(color=grid_rgba(0.30), width=1))
        fig_tapes.update_layout(title='', height=280, hovermode='x unified')
        apply_chart_theme(fig_tapes)
        ui.render_chart_panel(fig_tapes, key='tapes', context=_chart_ctx())

        st.markdown("<br>", unsafe_allow_html=True)
        ui.render_section_header("Flow-Zone Breadth",
                                 "Accumulation vs Distribution over time — context, not signal",
                                 icon="trending-up", accent="amber")
        fig_div = go.Figure()
        fig_div.add_trace(go.Bar(x=daily_agg.index, y=daily_agg['Overbought_Pct'],
                                 name='Accumulation %',
                                 marker=dict(color=chart_color('amber'), line=dict(color=chart_color('amber'), width=1))))
        fig_div.add_trace(go.Bar(x=daily_agg.index, y=-daily_agg['Oversold_Pct'],
                                 name='Distribution %',
                                 marker=dict(color=chart_color('cyan'), line=dict(color=chart_color('cyan'), width=1))))
        fig_div.update_layout(title='', height=300, hovermode='x unified', barmode='relative')
        apply_chart_theme(fig_div)
        ui.render_chart_panel(fig_div, key='divergence', context=_chart_ctx())

    # ── TAB 3 · Regime Analysis ────────────────────────────────────────────
    with tab3:
        ui.render_section_header("Aggregate Stretch",
                                 "Universe-mean trace (±100) over time — how stretched the tape is",
                                 icon="activity", accent="rose")
        # Signal = the per-name trace; its cross-sectional MEAN concentrates well inside a
        # single name's range, so the bands sit at ±15 — a third of θ — rather than at θ.
        # Green (positive) = the universe is, on balance, stretched up.
        _sig_band = 15.0
        colors = [chart_color('emerald') if v > _sig_band else chart_color('amber') if v < -_sig_band else chart_color('slate')
                  for v in daily_agg['Signal']]
        fig_avg = go.Figure()
        fig_avg.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Signal'].clip(lower=0),
                                     fill='tozeroy', fillcolor=chart_rgba('emerald', 0.05),
                                     line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig_avg.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Signal'].clip(upper=0),
                                     fill='tozeroy', fillcolor=chart_rgba('amber', 0.05),
                                     line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig_avg.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Signal'],
                                     mode='lines+markers', name='Avg trace',
                                     line=dict(color=chart_color('amber'), width=2),
                                     marker=dict(size=6, color=colors)))
        fig_avg.add_hline(y=_sig_band,  line=dict(color=chart_rgba('emerald', 0.5), width=1, dash='dash'))
        fig_avg.add_hline(y=-_sig_band, line=dict(color=chart_rgba('amber', 0.5), width=1, dash='dash'))
        fig_avg.add_hline(y=0,   line=dict(color=grid_rgba(0.30), width=1))
        _sig_span = float(np.nanmax(np.abs(daily_agg['Signal']))) if len(daily_agg) else 0.0
        _sig_span = _sig_span if np.isfinite(_sig_span) else 0.0
        _sig_lim = max(_sig_span * 1.2, _sig_band * 1.6)
        fig_avg.update_layout(title='', height=300, hovermode='x unified',
                              yaxis=dict(range=[-_sig_lim, _sig_lim]))
        apply_chart_theme(fig_avg)
        ui.render_chart_panel(fig_avg, key='avg_signal', context=_chart_ctx())

        st.markdown("<br>", unsafe_allow_html=True)
        ui.render_section_header("HMM Regime Distribution Over Time",
                                 "Percentage of symbols in each HMM regime daily",
                                 icon="activity", accent="cyan")
        fig_regime = go.Figure()
        fig_regime.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Regime_Bull_Pct'],
                                        mode='lines', name='Bull Regime %',
                                        fill='tozeroy', fillcolor=chart_rgba('emerald', 0.12),
                                        line=dict(color=chart_color('emerald'), width=2)))
        fig_regime.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Regime_Bear_Pct'],
                                        mode='lines', name='Bear Regime %',
                                        fill='tozeroy', fillcolor=chart_rgba('rose', 0.12),
                                        line=dict(color=chart_color('rose'), width=2)))
        fig_regime.update_layout(title='', height=300, hovermode='x unified',
                                 yaxis=dict(range=[0, 100], title='% of Universe'))
        apply_chart_theme(fig_regime)
        ui.render_chart_panel(fig_regime, key='regime', context=_chart_ctx())

        st.markdown("<br>", unsafe_allow_html=True)
        ui.render_section_header("Volatility Dynamics",
                                 "Volatility Regime & Change Points Over Time",
                                 icon="shield", accent="amber")
        vol_high = ts_df.groupby('Date')['Vol_Regime'].apply(
            lambda x: (x.isin(['HIGH', 'EXTREME'])).sum() / len(x) * 100)
        # Two measures on two scales → two charts on one shared time axis, never a
        # second y-axis (a dual axis lets the reader's eye pair any two levels).
        v1, v2 = st.columns(2)
        with v1:
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(x=daily_agg.index, y=vol_high.reindex(daily_agg.index).fillna(0),
                                         mode='lines', name='High / extreme vol %',
                                         fill='tozeroy', fillcolor=chart_rgba('amber', 0.12),
                                         line=dict(color=chart_color('amber'), width=2)))
            fig_vol.update_layout(title='', height=250, hovermode='x unified', showlegend=False,
                                  yaxis=dict(title='% in high / extreme vol', range=[0, 100]))
            apply_chart_theme(fig_vol)
            ui.render_chart_panel(fig_vol, key='volatility', context=_chart_ctx('% of universe'))
        with v2:
            fig_cp = go.Figure()
            fig_cp.add_trace(go.Bar(x=daily_agg.index, y=daily_agg['Change_Point'],
                                    name='Regime change points',
                                    marker=dict(color=chart_color('violet'), line=dict(width=0))))
            fig_cp.update_layout(title='', height=250, hovermode='x unified', showlegend=False,
                                 yaxis=dict(title='# symbols changing regime'))
            apply_chart_theme(fig_cp)
            ui.render_chart_panel(fig_cp, key='change_points', context=_chart_ctx('# symbols'))

        st.markdown("<br>", unsafe_allow_html=True)
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            ui.render_section_header("State Transition Metrics", "HMM Regime Statistics",
                                     icon="bar-chart", accent="emerald")
            regime_stats = {
                "Metric": ["Avg Bull Regime %", "Avg Bear Regime %", "Total Change Points", "Avg High Vol %"],
                "Value": [f"{summary['avg_bull_regime']:.1f}%",
                          f"{summary['avg_bear_regime']:.1f}%",
                          f"{summary['total_change_points']}",
                          f"{vol_high.mean():.1f}%"],
            }
            ui.render_data_table(pd.DataFrame(regime_stats), show_index=False,
                                 label_col="Metric", max_height=260)
        with col_r2:
            ui.render_section_header("Trace Distribution",
                                     "Universe-mean trace statistics",
                                     icon="database", accent="rose")
            signal_stats = {
                "Metric": ["Mean", "Median", "Min", "Max", "Std Dev"],
                "Value": [f"{daily_agg['Signal'].mean():+.2f}",
                          f"{daily_agg['Signal'].median():+.2f}",
                          f"{daily_agg['Signal'].min():+.2f}",
                          f"{daily_agg['Signal'].max():+.2f}",
                          f"{daily_agg['Signal'].std():.2f}"],
            }
            ui.render_data_table(pd.DataFrame(signal_stats), show_index=False,
                                 label_col="Metric", max_height=260)

    # ── TAB 4 · Data Terminal ──────────────────────────────────────────────
    with tab4:
        timeframe_label = "Weekly Time Series" if timeframe == 'Weekly' else "Daily Time Series"
        ui.render_section_header("Analytical Data",
                                 f"{timeframe_label} ({len(daily_agg)} periods)",
                                 icon="list", accent="cyan")
        display_ts = daily_agg.copy()
        display_ts.index = display_ts.index.strftime('%Y-%m-%d')
        display_ts = display_ts.reset_index().rename(columns={'Date': 'Date'})
        display_cols = ['Date', 'TurnBuy', 'ResumeLong', 'TurnSell', 'ResumeShort', 'Signal',
                        'Build_Pct', 'Cut_Pct', 'Avg_Fired_Units',
                        'Regime_Bull_Pct', 'Regime_Bear_Pct', 'Change_Point']
        display_ts = display_ts[display_cols]
        display_ts.columns = ['Date', '▲', '◆ ↑', '▼', '◆ ↓', 'Avg Trace',
                              'Build %', 'Cut %', 'Units at Fire',
                              'Bull Regime %', 'Bear Regime %', 'Change Pts']
        ui.render_table_panel(
            display_ts, key="range-terminal",
            context=f"{len(daily_agg)} periods",
            show_index=False, label_col="Date", max_height=560,
            col_precision={"Avg Trace": 1, "Units at Fire": 2, "Build %": 0, "Cut %": 0},
            footer=_glossary({
                "▲ / ▼": "Symbols firing a ▲ capitulation (sellers in control of a cheap price, "
                         "value turning back toward fair) or a ▼ distribution on this bar.",
                "◆ ↑ / ↓": "Symbols firing a RESUME — a trend resuming from inside the zone.",
                "Avg Trace": "Cross-sectional mean trace (±100) — how stretched the universe is, "
                             "on balance. θ is ±42.9 for a single name.",
                "Build / Cut %": "Share of the readable universe whose grid cell builds (Buy / Add "
                                 "/ Accumulate) or cuts (Trim / Reduce / Exit).",
                "Units at Fire": "Mean grid weight of the names that fired — did events land on "
                                 "names the grid already favoured?",
                "Bull / Bear Regime %": "Percent of universe with an HMM label containing BULL or "
                                        "BEAR. Risk context, never a signal input.",
                "Change Pts": "Count of symbols with a regime-state transition on this day.",
            }),
        )

        st.download_button(
            label="↓ Download Full Report (Excel)",
            data=to_excel(ts_df),
            file_name=build_download_filename(
                "range",
                universe=meta.get("universe"),
                selected_index=meta.get("selected_index"),
                dates=(start_date, end_date) if (start_date and end_date) else None,
                ext="xlsx",
            ),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )



# ══════════════════════════════════════════════════════════════════════════════
# CORRELATION MODE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_correlation_analysis(universe, selected_index, target_ticker, lookback, method, timeframe, analysis_date=None, sid=None, study=None):
    """Execute correlation analysis between universe constituents and a target asset.

    Returns a dict with correlation data, rolling correlations, prices, and returns,
    plus a confluence score (|correlation| × normalised Pragati signal strength).
    """
    if analysis_date is None:
        analysis_date = _today_ist()
    if sid is None:
        sid = _engine_settings(universe, selected_index, timeframe)
    progress_slot = st.empty()
    progress_bar(progress_slot, 5, "Initializing Correlation Engine", "Fetching Market Data")

    try:
        # Fetch universe symbols
        stock_list, msg = resolve_universe(universe, selected_index)

        if not stock_list:
            ui_error(f"Failed to fetch universe symbols: {msg}")
            return None

        console.item("Symbols fetched", len(stock_list))

        progress_bar(progress_slot, 15, "Fetching OHLCV Data", f"Symbols: {len(stock_list)}")

        # ── Universe data from registry (shared pool with the screener) ──
        # Passing only the universe symbols so the registry key is consistent with
        # the screener and timeseries paths.  The target ticker is supplemented
        # below with a single small fetch if it is not already in the pool.
        data_dict, fetch_msg = get_universe_data(stock_list, end_date=analysis_date,
                                                 timeframe=timeframe)
        if data_dict is None:
            ui_error(f"Data fetch failed: {fetch_msg}")
            console.item("Data fetch error", fetch_msg)
            return None

        # ── Supplement with target ticker if not already in the universe pool ──
        if target_ticker not in data_dict:
            console.detail(
                f"Target ticker '{target_ticker}' not in registry — fetching individually"
            )
            # Registry-first single-ticker fetch: get_universe_data checks the
            # session registry (15-min TTL) before yfinance and stores the result,
            # so repeated correlation runs on the same target reuse the cache instead
            # of re-hitting the network with identical requests.
            target_raw, _ = get_universe_data([target_ticker], end_date=analysis_date,
                                              timeframe=timeframe)
            if target_raw and target_ticker in target_raw:
                # Merge into a new dict so we don't mutate the registry entry
                data_dict = {**data_dict, target_ticker: target_raw[target_ticker]}
                console.detail(f"Target ticker '{target_ticker}' merged into data pool")
            else:
                ui_error(f"Could not fetch target asset '{target_ticker}'")
                return None
        else:
            console.detail(f"Target ticker '{target_ticker}' already in registry pool")

        console.item("Data available for symbols", len(data_dict))

        progress_bar(progress_slot, 25, "Building Price Matrix", "Pivoting Close Prices")

        # Build Close price matrix — handle MultiIndex columns from yfinance
        close_dict = {}
        for ticker, data in data_dict.items():
            if len(data) > 0:
                if 'Close' in data.columns:
                    close_dict[ticker] = data['Close']
                else:
                    # Handle MultiIndex case
                    try:
                        close_dict[ticker] = data[data.columns[data.columns.get_level_values(-1) == 'Close'][0]]
                    except (IndexError, KeyError):
                        console.item(f"Skipping {ticker}", "No Close column found")

        if not close_dict:
            ui_error("No valid price data found for universe")
            console.item("Error", "No Close prices extracted")
            return None

        console.item("Close prices extracted for", len(close_dict))

        close_df = pd.DataFrame(close_dict)
        close_df = close_df.dropna(axis=1, how='all')

        console.item("Close DataFrame shape", f"{close_df.shape}")

        if len(close_df) < lookback + 10:
            ui_error(f"Insufficient historical data for correlation analysis (only {len(close_df)} rows, need {lookback + 10})")
            console.item("Error", f"Only {len(close_df)} rows, need {lookback + 10}")
            return None

        # Resample to weekly if needed
        if timeframe == "Weekly":
            # A frame of closes, one column per name — not OHLCV, so the last close of each
            # week (the same W-MON bucketing resample_to_weekly uses).
            close_df = close_df.copy()
            close_df.index = pd.to_datetime(close_df.index)
            close_df = (close_df.resample('W-MON', closed='left', label='left').last()
                        .dropna(how='all'))

        progress_bar(progress_slot, 40, "Computing Returns", f"Method: {method}")

        # Compute log returns — drop rows only where all values are NaN
        returns_df = np.log(close_df / close_df.shift(1)).dropna(how='all')

        if target_ticker not in returns_df.columns:
            ui_error(f"Target asset '{target_ticker}' not in data")
            console.item("Error", f"Target {target_ticker} not in returns columns")
            return None

        target_returns = returns_df[target_ticker].dropna()
        console.item("Target returns available", len(target_returns))

        # Filter to common dates with target
        common_idx = returns_df.index.intersection(target_returns.index)
        if len(common_idx) < lookback + 10:
            ui_error(f"Insufficient overlapping data (only {len(common_idx)} days). Try a shorter lookback period.")
            console.item("Error", f"Only {len(common_idx)} common dates, need {lookback + 10}")
            return None

        returns_df = returns_df.loc[common_idx]
        target_returns = target_returns.loc[common_idx]
        universe_returns = returns_df.drop(columns=[target_ticker])

        console.item("Universe returns shape", f"{universe_returns.shape}")
        console.item("Target returns shape", target_returns.shape)

        progress_bar(progress_slot, 60, "Computing Rolling Correlation", f"Lookback: {lookback} bars")

        # Compute rolling correlation — use vectorized rolling correlation
        console.item("Computing rolling correlations", f"method={method}, lookback={lookback}, cols={len(universe_returns.columns)}")

        try:
            # Vectorized rolling correlation of every universe column against the
            # target in one C-level pass — replaces a per-column Python loop that
            # built a temp DataFrame and called .rolling().corr() per symbol. Output
            # is byte-identical (verified): same NaN handling (universe cols filled
            # with 0.0, target raw, as before), same Pearson rolling window, same
            # warmup NaNs. RangeIndex preserved to match the prior positional frame.
            _uni = universe_returns.fillna(0.0).reset_index(drop=True)
            _tgt = pd.Series(target_returns.values)             # positional align
            rolling_corr_df = _uni.rolling(window=lookback).corr(_tgt)

            console.item("Rolling corr dict entries", rolling_corr_df.shape[1])

            if rolling_corr_df.shape[1] == 0:
                ui_error("Could not compute rolling correlations for any column")
                return None

            console.item("Rolling corr DataFrame shape", rolling_corr_df.shape)
        except Exception as e:
            ui_error(f"Error in rolling correlation: {str(e)}")
            console.item("Rolling corr computation error", str(e)[:100])
            return None

        if rolling_corr_df.empty or len(rolling_corr_df) == 0:
            ui_error("Could not compute rolling correlations. Check data availability.")
            console.item("Error", "Rolling correlation DataFrame is empty")
            return None

        # Get current and average correlations
        current_corr = rolling_corr_df.iloc[-1]
        avg_corr = rolling_corr_df.mean()
        corr_trend = current_corr - avg_corr

        # Per-symbol return σ over the SAME lookback window as the correlation.
        # The correlation-implied expected move is a regression BETA, not the bare
        # correlation: E[r_sym | r_tgt] = corr × (σ_sym / σ_tgt) × r_tgt. Using corr
        # alone silently assumed every symbol has the target's volatility — inflating
        # implied moves (and thus "Divergence") for low-vol names by the vol ratio.
        _win_sigma = returns_df.tail(lookback).std()
        _tgt_sigma = float(_win_sigma.get(target_ticker, np.nan))

        # Compute tiers
        def get_corr_tier(corr):
            if pd.isna(corr):
                return "Neutral"
            abs_corr = abs(corr)
            if corr > 0:
                if abs_corr >= 0.6: return "Strong+"
                elif abs_corr >= 0.4: return "Moderate+"
                elif abs_corr >= 0.2: return "Weak+"
                else: return "Neutral"
            else:
                if abs_corr >= 0.6: return "Strong-"
                elif abs_corr >= 0.4: return "Moderate-"
                elif abs_corr >= 0.2: return "Weak-"
                else: return "Neutral"

        # Reuse screener results from session state when they match the current run —
        # avoids a full re-fetch+re-analysis just to enrich the correlation output.
        _smeta = st.session_state.get("screener_meta")
        _sdf   = st.session_state.get("results_df")
        _can_reuse = (
            _smeta is not None and _sdf is not None and not _sdf.empty
            and _smeta.get("universe")       == universe
            and _smeta.get("selected_index") == selected_index
            and _smeta.get("analysis_date")  == analysis_date
            and _smeta.get("timeframe")      == timeframe
        )
        if _can_reuse:
            console.detail("Correlation: reusing cached screener results from session state")
            sid_results = _sdf
        else:
            _corr_reg_len, _corr_n1, _corr_n2 = 20, 10, 21
            _corr_levels = (80, 40, -80, -40)
            _corr_wt2_len, _corr_wt2_type = 20, "ALMA"
            sid_results = run_screener_analysis(
                universe, selected_index, analysis_date,
                _corr_reg_len, _corr_n1, _corr_n2, _corr_levels, timeframe,
                show_progress=False, external_progress_slot=progress_slot,
                progress_offset=60, progress_scale=30,
                wt2_len=_corr_wt2_len, wt2_type=_corr_wt2_type, sid=sid, study=study,
            )

        progress_bar(progress_slot, 90, "Building Results DataFrame", "Computing Divergence Metrics")

        # Build correlation results dataframe
        corr_data_list = []
        for symbol in universe_returns.columns:
            if symbol not in close_df.columns or symbol not in current_corr.index:
                continue

            # Get current data — aligned to the last bar where BOTH the symbol and
            # the target have data. Independent iloc[-1] on each column would compare
            # mismatched sessions when exchanges run on different calendars/timezones
            # (e.g. NSE universe vs a US target during the Asian session: the symbol's
            # last row is Tuesday, the target's last valid bar is Monday). dropna()
            # over the pair yields the most recent common session, so the divergence
            # math compares the same trading day for both legs.
            _pair = close_df[[symbol, target_ticker]].dropna()
            if len(_pair) >= 2:
                current_price  = _pair[symbol].iloc[-1]
                price_change   = _pair[symbol].pct_change().iloc[-1] * 100
                target_price   = _pair[target_ticker].iloc[-1]
                target_change  = _pair[target_ticker].pct_change().iloc[-1] * 100
            else:
                # Not enough overlapping history to compute a same-session move.
                current_price = _pair[symbol].iloc[-1] if len(_pair) else np.nan
                price_change = np.nan
                target_price = _pair[target_ticker].iloc[-1] if len(_pair) else np.nan
                target_change = np.nan

            # Pull this symbol's Pragati read from the screener output already computed
            # above, so the confluence ranking carries the live signal state.
            _pr = {}
            sid_zone, sid_signal_type = "—", "Neutral"
            if sid_results is not None and len(sid_results) > 0:
                sid_row = sid_results[sid_results['SimpleName'] == symbol.replace('.NS', '').replace('^', '')]
                if len(sid_row) > 0:
                    _r0 = sid_row.iloc[0]
                    sid_zone = _r0.get('Zone', '—')
                    sid_signal_type = _r0.get('SignalType', 'Neutral')
                    for _c in ('Signal_Score', 'PRG_Hist_Z', 'Side', 'Signal_Kind', 'PRG_State',
                               'PRG_Armed', 'PRG_Armed_Age', 'PRG_Decl', 'PRG_Decl_Age',
                               'PRG_Push', 'PRG_CTape', 'PRG_VTape', 'CVG_Cell', 'CVG_Held',
                               'CVG_Bars', 'CVG_From', 'CVG_Chart', 'CVG_Lead', 'CVG_Units',
                               'CVG_Action', 'PRG_Why', 'Priority_Long', 'Priority_Short'):
                        _pr[_c] = _r0.get(_c)

            # Correlation-implied expected move = beta × target move, where
            # beta = corr × σ_sym/σ_tgt over the same lookback window (see above).
            _sym_sigma = float(_win_sigma.get(symbol, np.nan))
            if np.isfinite(_sym_sigma) and np.isfinite(_tgt_sigma) and _tgt_sigma > 0:
                _beta = current_corr[symbol] * (_sym_sigma / _tgt_sigma)
            else:
                _beta = current_corr[symbol]   # degraded fallback: assume equal vols
            expected_change = _beta * target_change
            divergence = price_change - expected_change

            corr_data_list.append({
                'Symbol': symbol,
                'DisplayName': symbol,
                'SimpleName': symbol.replace('.NS', '').replace('^', ''),
                'Corr_Current': current_corr[symbol],
                'Corr_Avg': avg_corr[symbol],
                'Corr_Trend': corr_trend[symbol],
                'Corr_Tier': get_corr_tier(current_corr[symbol]),
                'Price': current_price,
                'PctChange': price_change,
                'Target_Pct': target_change,
                'Expected_Change': expected_change,
                'Divergence': divergence,
                'Regime_Zone': sid_zone,
                'SignalType': sid_signal_type,
                **_pr,
            })

        corr_df = pd.DataFrame(corr_data_list)
        if len(corr_df) == 0:
            ui_error("No correlation data could be computed")
            console.item("Error", "Empty correlation DataFrame")
            return None

        corr_df = corr_df.sort_values('Corr_Current', key=abs, ascending=False)

        # ── Confluence score ────────────────────────────────────────────
        # |Corr| × normalised signal strength, i.e. how loud this symbol's own Pragati read
        # is relative to the rest of the universe. Normalising by the observed max keeps the
        # score in [0,1] across universes whose readings spread differently.
        #
        # Strength is DIRECTION-FREE: max(Priority_Long, Priority_Short) — a ▲▼ on this bar,
        # else how far the name is stretched either way (|trace| / 200), whichever side it is
        # on. That is what the confluence question asks: how loud is this name's own read.
        _pri_cols = [c for c in ('Priority_Long', 'Priority_Short') if c in corr_df.columns]
        if _pri_cols and corr_df[_pri_cols].notna().any().any():
            abs_pri = corr_df[_pri_cols].max(axis=1).fillna(0).clip(lower=0)
            pri_norm = abs_pri / max(abs_pri.max(), 1e-6)        # [0, 1]
            corr_df['Priority_Strength'] = pri_norm
            corr_df['Confluence_Score'] = (corr_df['Corr_Current'].abs() * pri_norm).clip(0.0, 1.0)
            _n_fired = int((corr_df['Side'].isin(['Buy', 'Sell'])).sum()) if 'Side' in corr_df.columns else 0
            console.item("Confluence formula", "|Corr| × normalised signal strength")
            console.item("Fired signals in universe",
                         f"{_n_fired} symbol(s) fired an event on this bar")
        else:
            # Defensive fallback — no screener output to join against, so rank on the
            # correlation alone rather than inventing a signal strength.
            corr_df['Priority_Strength'] = 0.0
            corr_df['Confluence_Score'] = corr_df['Corr_Current'].abs().clip(0.0, 1.0)
            console.item("Confluence formula", "|Corr| only (fallback — no screener data)")

        # Get target name from maps (maps are display_name -> ticker, so reverse lookup)
        target_name = target_ticker
        for map_dict in [COMMODITY_MAP, CURRENCY_MAP, CRYPTO_MAP, GLOBAL_INDEXES_MAP]:
            if target_ticker in map_dict.values():
                target_name = [k for k, v in map_dict.items() if v == target_ticker][0]
                break
            elif target_ticker in map_dict.keys():
                target_name = target_ticker
                break

        progress_bar(progress_slot, 100, "Analysis Complete", "Ready to display")
        time.sleep(0.3)
        progress_slot.empty()

        return {
            "corr_df": corr_df,
            "rolling_corr": rolling_corr_df,
            "target_ticker": target_ticker,
            "target_name": target_name,
            "prices": close_df,
            "returns": returns_df,
            "lookback": lookback,
            "method": method,
            "timeframe": timeframe,
            "trigger": sid.trigger_label,
            "iclass": sid.iclass,
        }

    except Exception as e:
        ui_error(f"Correlation analysis error: {str(e)}")
        console.item("Exception", str(e))
        import traceback
        console.item("Traceback", traceback.format_exc()[:2000])
        return None


# ══════════════════════════════════════════════════════════════════════════════
# CORRELATION MODE — HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── Shared HTML-builder palette helpers ──────────────────────────────────────
# Used by every bespoke table below. Keep these in sync — changing one colour here
# propagates to every signal table.
#
# PRAGATI'S COLOUR LANGUAGE, carried from the pane: GREEN AND RED ARE DIRECTION —
# up-stretch, buyers in control, rich price and a long event are green; their
# opposites red. GOLD (the app's amber) means one thing: READ WITH CAUTION — a held
# grid row, split ingredients, a quiet regime, a settling basket, a capitulation still
# cheapening (the watchlist). It never means up or down. (Siddhi drew its SELL as a yellow
# diamond; that is gone, because here yellow is a qualifier, not a side.)
#
# THESE ARE FUNCTIONS, NOT CONSTANTS. A module-level colour binds once at import, when
# there is no session to read an appearance from — which is how a UI ends up with its
# chrome in one theme and its cells in the other. Every one resolves from
# ui.table_tokens() at render time instead. They are literals inside the iframe because
# an iframe cannot see the app's CSS variables; they are not literals in this module.


def _long_c() -> str:
    """Long / up / rich — green."""
    return ui.table_tokens()["emerald"]


def _short_c() -> str:
    """Short / down / cheap — red."""
    return ui.table_tokens()["rose"]


def _gold_c() -> str:
    """Read with caution. Never a direction."""
    return ui.table_tokens()["amber"]


def _neut_c() -> str:
    """No claim. The muted ink, not a colour."""
    return ui.table_tokens()["ink_tertiary"]


def _dim() -> str:
    """The 'nothing here' ink for an em-dash cell."""
    return ui.table_tokens()["ink_quaternary"]

# 'buy'/'sell' are the canonical side keys. 'long'/'short' are accepted so any
# lingering caller keeps working rather than silently getting the sell palette.
_BUY_SIDES = ('buy', 'long')


def _is_buy_side(side: str) -> bool:
    return str(side).lower() in _BUY_SIDES


def _priority_pct_col(side: str) -> str:
    """The cross-sectional percentile column for a side."""
    return 'Priority_Long_pct' if _is_buy_side(side) else 'Priority_Short_pct'


def _side_palette(side: str) -> dict:
    """Side-keyed accents — long green ▲, short red ▼. Resolved per call."""
    t = ui.table_tokens()
    if _is_buy_side(side):
        return {"accent_light": t["emerald"], "border_color": t["border"],
                "header_bg": t["header_a"], "mark": "▲", "label": "LONG"}
    return {"accent_light": t["rose"], "border_color": t["border"],
            "header_bg": t["header_a"], "mark": "▼", "label": "SHORT"}


def _signed_color(value: float, pos: str = "", neg: str = "") -> str:
    """Green for non-negative, red for negative (or supplied overrides)."""
    t = ui.table_tokens()
    return (pos or t["emerald"]) if value >= 0 else (neg or t["rose"])


def _delta_arrow(value: float) -> str:
    """Up arrow for non-negative deltas, down arrow for negative."""
    return "↑" if value >= 0 else "↓"


def _human_vol(value: float, signed: bool = True) -> str:
    """Compact K/M/B/T formatting for large volume-unit numbers (Bar Δ, CVD, CVD Slope)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(v):
        return "—"
    sign = "-" if v < 0 else ("+" if signed else "")
    a = abs(v)
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{sign}{a / div:.2f}{suf}"
    return f"{sign}{a:.0f}"


def _fmt_num(v, fmt="{:+.2f}", dash="—"):
    """Format a possibly-NaN/None number, falling back to an em dash."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return dash
    return dash if not np.isfinite(f) else fmt.format(f)


def _dash_cell() -> str:
    return f'<td class="numeric" style="color:{_dim()};">—</td>'


# ── The cells. Each reads one engine field and says what it means in its title. ──
def _trace_cell(v) -> str:
    """The trace, ±100 — green stretched up, red stretched down; bold past θ."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return _dash_cell()
    if not np.isfinite(f):
        return _dash_cell()
    th = sv.THETA_OSC
    col = _long_c() if f > 0 else _short_c() if f < 0 else _neut_c()
    past = abs(f) >= th
    note = (f"STRETCHED {'up' if f > 0 else 'down'} past θ (±{th:.0f})" if past else
            f"{'leaning up' if f > 0 else 'leaning down' if f < 0 else 'balanced'}, inside θ")
    title = (f"trace {f:+.1f} · {note}. The trace is conviction × value on this chart: how far "
             f"the move is stretched, in one-sided effort and in price against fair value.")
    return (f'<td class="numeric" style="color:{col}; font-weight:{700 if past else 500};" '
            f'title="{html.escape(title)}">{f:+.0f}</td>')


def _tape_cell(v, kind: str) -> str:
    """A tape reading — conviction (who controls) or value (+ rich / − cheap)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return _dash_cell()
    if not np.isfinite(f):
        return _dash_cell()
    knee = 30.0 if kind == "conv" else sv.THETA_OSC
    col = _long_c() if f > 0 else _short_c() if f < 0 else _neut_c()
    if kind == "conv":
        word = ("buyers firm" if f >= knee else "buyers edge" if f >= 0 else
                "sellers edge" if f > -knee else "sellers firm")
        title = f"MTF conviction {f:+.0f} · {word} — who controls, across the ladder"
    else:
        word = ("rich" if f >= knee else "above fair" if f >= 0 else
                "below fair" if f > -knee else "cheap")
        title = f"MTF value {f:+.0f} · {word} — where price stands, across the ladder"
    return (f'<td class="numeric" style="color:{col}; font-weight:{700 if abs(f) >= knee else 500};" '
            f'title="{html.escape(title)}">{f:+.0f}</td>')


def _push_cell(push, held: bool = False, tier: str = "") -> str:
    """The histogram's push in five levels — ↑↑ impulse … ↓↓ impulse, · none."""
    try:
        p = int(push)
    except (TypeError, ValueError):
        p = 0
    glyph = cg.PUSH_GLYPH.get(p, "·")
    col = _long_c() if p > 0 else _short_c() if p < 0 else _dim()
    title = (f"{cg.PUSH_TEXT.get(p, 'no push')}" + (f" · column {tier}" if tier else "")
             + ". The trace's push — a push moves the grid's row one step toward the tape, "
               "an impulse all the way, no push holds it.")
    if held:
        title += " HELD: the row stands against its tape for want of a push."
        glyph += " held"
        col = _gold_c()
    return (f'<td class="numeric" style="color:{col}; font-weight:700; font-size:{ui.FS["2xs"]};" '
            f'title="{html.escape(title)}">{html.escape(glyph)}</td>')


#: The glyph each tone carries in a table — the same shape it plots as on the map,
#: so a state is never told by colour alone (charts.TONE_SYMBOL).
_TONE_GLYPH = {"emerald": "▲", "cyan": "●", "amber": "■", "rose": "▼", "slate": "◆"}


def _tone_ink(tone: str, t: "dict | None" = None) -> str:
    """A grid tone as table ink; slate reads as secondary ink, not as a colour."""
    t = t or ui.table_tokens()
    return t["ink_secondary"] if tone == "slate" else t[tone]


def _grid_cell(row) -> str:
    """The grid state — the action, its reason and its units, in the cell's TONE (Pragyam's CVG_TONE)."""
    try:
        cell = int(row.get('CVG_Cell', cg.UNREAD))
    except (TypeError, ValueError):
        cell = cg.UNREAD
    if cell == cg.UNREAD:
        return (f'<td class="numeric" style="color:{_dim()}; font-size:{ui.FS["2xs"]};" '
                f'title="{html.escape(cg.MEANING[cg.UNREAD])}">unread</td>')
    tone = cg.TONES[cell]
    held = bool(row.get('CVG_Held', False))
    col = _tone_ink(tone)
    lead = int(row.get('CVG_Lead', 0) or 0)
    tip = cg.tooltip(cell, cg.UNITS[cell], int(row.get('CVG_Bars', 1) or 1),
                     int(row.get('CVG_From', cg.UNREAD) or cg.UNREAD),
                     int(row.get('CVG_Chart', cg.UNREAD) or cg.UNREAD), lead, held)
    lead_g = (f' <span style="color:{_long_c() if lead > 0 else _short_c()};">'
              f'{"↑" if lead > 0 else "↓"}</span>') if lead else ""
    return (f'<td style="color:{col}; font-weight:600; font-size:{ui.FS["2xs"]}; white-space:nowrap;" '
            f'title="{html.escape(tip)}"><span style="font-size:{ui.FS["3xs"]};">{_TONE_GLYPH[tone]}</span> '
            f'{html.escape(cg.action(cell))}'
            f'<span style="color:{_neut_c()}; font-weight:400;"> · {html.escape(cg.reason(cell))} · '
            f'{cg.UNITS[cell]:g}u</span>{lead_g}</td>')


_EVENT_TITLES = {
    "▲ CAPITULATION": "v9's ▲: the grid's Buy · capitulation turning — sellers in control across the "
                      "ladder at a price cheap past θ, and value momentum has already turned back toward "
                      "fair. The one event the v9 audit found positive in all three eras (2006-13, "
                      "2014-19, 2020-26), daily and weekly: a lean of about +0.05σ over 10-20 bars, not a "
                      "trade, and not on crypto. It stands as the declaration until a ▼.",
    "▼ DISTRIBUTION": "v9's ▼: sellers have taken control across the ladder of a price rich past θ — "
                      "the grid's Exit cell. As a state it was followed by underperformance in every era "
                      "of the v9 audit; the entry itself is too rare to measure alone.",
    "◆ RESUME ↑": "A trend resuming from inside the zone: the push dipped and returned past its "
                  "impulse gate, buyers still firmly in control, room left on the value tape.",
    "◆ RESUME ↓": "A trend resuming from inside the zone: the push bounced and returned past its "
                  "impulse gate, sellers still firmly in control, room left on the value tape.",
}


def _event_cell(label) -> str:
    s = str(label or "")
    if not s:
        return f'<td class="numeric" style="color:{_dim()}; font-size:{ui.FS["xs"]};">—</td>'
    col = _long_c() if ("▲" in s or "↑" in s) else _short_c()
    weight = 700 if ("CAPITULATION" in s or "DISTRIBUTION" in s) else 600
    return (f'<td class="numeric" style="color:{col}; font-weight:{weight}; font-size:{ui.FS["xs"]}; '
            f'white-space:nowrap;" title="{html.escape(_EVENT_TITLES.get(s, s))}">{html.escape(s)}</td>')


def _state_cell(row) -> str:
    """PRG_State — the bar's standing: an event, an open window, paused, or neutral."""
    s = str(row.get('PRG_State', '') or '')
    if s.startswith(("CAPITULATION", "DISTRIBUTION", "RESUME")):
        lab = {"CAPITULATION ▲": "▲ CAPITULATION", "DISTRIBUTION ▼": "▼ DISTRIBUTION",
               "RESUME ◆↑": "◆ RESUME ↑", "RESUME ◆↓": "◆ RESUME ↓"}.get(s, s)
        return _event_cell(lab)
    if s.startswith("WATCH"):
        n = int(row.get('PRG_Armed_Age', 0) or 0)
        title = (f"In capitulation for {n} bars — sellers in control of a cheap price — but value is "
                 "still cheapening. The ▲ fires when value momentum turns back toward fair.")
        return (f'<td class="numeric" style="color:{_gold_c()}; font-weight:600; font-size:{ui.FS["2xs"]}; '
                f'white-space:nowrap;" title="{html.escape(title)}">watch ▲ {n}b</td>')
    if s in ("PAUSED", "WARMING UP"):
        why = str(row.get('PRG_Why', '') or '')
        return (f'<td class="numeric" style="color:{_gold_c()}; font-size:{ui.FS["2xs"]};" '
                f'title="{html.escape("signals paused · " + why)}">{s.lower()}</td>')
    d = int(row.get('PRG_Decl', 0) or 0)
    if d:
        age = row.get('PRG_Decl_Age')
        age_t = f" {int(age)}b" if age is not None and pd.notna(age) else ""
        title = ("The standing declaration: the last event was " + ("▲ capitulation" if d > 0 else "▼ distribution")
                 + f"{age_t} ago. It stands until the opposite one and has no exit — as a held position "
                 "it carried no edge in the v9 audit; read the grid for where the name stands now.")
        return (f'<td class="numeric" style="color:{_long_c() if d > 0 else _short_c()}; '
                f'font-size:{ui.FS["2xs"]};" title="{html.escape(title)}">'
                f'{"▲" if d > 0 else "▼"} decl{age_t}</td>')
    return f'<td class="numeric" style="color:{_dim()}; font-size:{ui.FS["2xs"]};">—</td>'


def _evidence_cell(row, side: str) -> str:
    """What flow says about the push: a qualified divergence, and effort absorbed."""
    buy = _is_buy_side(side)
    parts, tips = [], []
    if bool(row.get('PRG_Div_Seen_Bull' if buy else 'PRG_Div_Seen_Bear', False)):
        parts.append("div")
        tips.append(("bullish" if buy else "bearish") + " divergence at a stretched price, "
                    "in the last 20 bars")
    if bool(row.get('PRG_Abs_Seen', False)):
        parts.append("abs")
        tips.append("effort absorbed — heavy participation, little result")
    if bool(row.get('PRG_Split', False)):
        parts.append("split")
        tips.append("the trace's ingredients point opposite ways — read with caution")
    if not parts:
        return f'<td class="numeric" style="color:{_dim()}; font-size:{ui.FS["2xs"]};">—</td>'
    col = _gold_c() if parts == ["split"] else (_long_c() if buy else _short_c())
    return (f'<td class="numeric" style="color:{col}; font-size:{ui.FS["2xs"]};" '
            f'title="{html.escape(" · ".join(tips))}">{" · ".join(parts)}</td>')


def _hold_cell(age, horizon, direction) -> str:
    """Render the hold window as "day N/H" — how far into the declared horizon this event is."""
    try:
        a = float(age)
        h = int(horizon)
    except (TypeError, ValueError):
        return _dash_cell()
    if not np.isfinite(a) or h <= 0:
        return _dash_cell()
    n = int(a)
    d = int(direction or 0)
    col = _long_c() if d > 0 else _short_c() if d < 0 else _neut_c()
    if n > h:
        return (f'<td class="numeric" style="color:{_neut_c()}; font-size:{ui.FS["2xs"]};" '
                f'title="window expired — the declared horizon is {h} bars">expired</td>')
    frac = 1.0 - (n / max(h, 1))
    title = (f'day {n} of {h} in the hold window · {frac*100:.0f}% of the declared horizon left. '
             f'Entry was the open after the signal bar.')
    return (f'<td class="numeric" style="color:{col}; font-weight:600; font-size:{ui.FS["2xs"]};" '
            f'title="{html.escape(title)}">{n}/{h}</td>')


def _entry_status(row, offset: int, side: str = 'buy'):
    """Has price already run since the event fired — i.e. is the entry now late?

    Directional move from the fire bar to the snapshot bar, normalised by the symbol's
    own recent return volatility x sqrt(bars elapsed) so the bands are asset-agnostic
    (sigma units). Returns (label, color, title).
    """
    if offset == 0:
        return ('Now', _neut_c(), 'fresh — fired on the snapshot bar')
    closes = row.get('Close_Hist')
    if not isinstance(closes, (list, tuple)) or offset >= len(closes):
        return ('—', _dim(), '')
    fire_close, now_close = closes[offset], closes[0]
    if not (pd.notna(fire_close) and pd.notna(now_close) and float(fire_close) > 0):
        return ('—', _dim(), '')
    side_sign = 1.0 if _is_buy_side(side) else -1.0
    dm = (float(now_close) - float(fire_close)) / float(fire_close) * side_sign
    good = _long_c() if _is_buy_side(side) else _short_c()
    rv = row.get('RetVol20')
    scale = (float(rv) * (offset ** 0.5)) if (rv is not None and pd.notna(rv) and float(rv) > 0) else None
    if scale and scale > 0:
        sig = dm / scale
        title = f'{dm*100:+.1f}% since the fire bar, in the signal\'s direction ({sig:+.1f} sigma)'
        if sig <= -1.0: return ('Adverse', _neut_c(), title)
        if sig >= 1.5:  return ('Extended', _gold_c(), title)
        if sig >= 0.5:  return ('Running', good, title)
        return ('Open', good, title)
    title = f'{dm*100:+.1f}% since the fire bar, in the signal\'s direction'
    if dm <= -0.03: return ('Adverse', _neut_c(), title)
    if dm >= 0.06:  return ('Extended', _gold_c(), title)
    if dm >= 0.02:  return ('Running', good, title)
    return ('Open', good, title)


def _status_cell(status) -> str:
    """Render a (label, color, title) status tuple as a small table cell."""
    label, color, title = (status if isinstance(status, (tuple, list)) and len(status) == 3
                           else ('—', _dim(), ''))
    _t = html.escape(str(title)) if title else ''
    return (f'<td class="numeric" style="color:{color}; font-weight:700; font-size:{ui.FS["2xs"]};" '
            f'title="{_t}">{html.escape(str(label))}</td>')


def _html_doc(head_cells: str, rows: list, max_h: int) -> str:
    """The iframe document every bespoke table uses — one shell, one set of tokens."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>{ui.table_shell_css(max_height=max_h)}</style>
    </head>
    <body>
    <div class="tt-scroll">
        <table>
            <thead><tr>{head_cells}</tr></thead>
            <tbody>{"".join(rows)}</tbody>
        </table>
    </div>
    </body>
    </html>
    """


def _th(label: str, title: str = "", numeric: bool = True, center: bool = False) -> str:
    cls = ' class="numeric"' if numeric and not center else ""
    sty = ' style="text-align:center;"' if center else ""
    t = f' title="{html.escape(title)}"' if title else ""
    return f"<th{cls}{sty}{t}>{html.escape(label)}</th>"


_TH_TRACE = ("The trace, ±100: conviction × value on this chart — how far the move is stretched. "
             "θ is ±43. Green stretched up, red stretched down; bold past θ.")
_TH_PUSH = ("The trace's push, from its histogram: ↑↑ impulse · ↑ push · · none · ↓ push · ↓↓ "
            "impulse. 'held' (gold): the grid row stands against its tape for want of a push.")
_TH_GRID = ("The conviction-value grid (3 × 3, v8): where the two tapes place this name, as an "
            "action with its graded units (Buy 4 … Exit 0.25). ↑/↓: the chart's own cell carries "
            "more / fewer units. Hover for the full reading.")
_TH_C = "MTF conviction tape — who controls across the ladder. ±30 is the knee."
_TH_V = "MTF value tape — rich (+) or cheap (−) across the ladder. ±43 (θ) is the knee."


def _build_confluence_table_html(df: pd.DataFrame) -> str:
    """Ranked HTML table for confluence setups: correlation × the live Pragati reading."""
    _MAXH = 560
    rows = []
    if df.empty:
        rows.append('<tr><td class="empty" colspan="11">— no setups —</td></tr>')
    else:
        _t = ui.table_tokens()
        for idx, (_, row) in enumerate(df.iterrows(), 1):
            symbol = html.escape(str(row.get('SimpleName', '')))
            corr = float(row.get('Corr_Current', 0))
            zone = html.escape(str(row.get('Regime_Zone', 'Neutral')))
            actual = float(row.get('PctChange', 0) or 0)
            expected = float(row.get('Expected_Change', 0) or 0)
            divergence = float(row.get('Divergence', 0) or 0)
            confluence = float(row.get('Confluence_Score', 0) or 0)
            corr_color = _t["emerald"] if corr > 0 else _t["rose"]
            div_color = _t["emerald"] if divergence > 0 else _t["rose"]
            rows.append(f"""
            <tr>
                <td class="numeric" style="color:{_t["accent"]}; font-weight:700;">{idx:02d}</td>
                <td class="symbol">{symbol}</td>
                <td class="numeric" style="color:{corr_color}; font-weight:600;">{corr:+.3f}</td>
                {_trace_cell(row.get('Signal_Score'))}
                {_state_cell(row)}
                {_grid_cell(row)}
                <td class="numeric" style="color:{_neut_c()}; font-size:{ui.FS["2xs"]};">{zone}</td>
                <td class="numeric" style="color:{_neut_c()};">{actual:+.2f}%</td>
                <td class="numeric" style="color:{_neut_c()};">{expected:+.2f}%</td>
                <td class="numeric" style="color:{div_color}; font-weight:600;">{divergence:+.2f}%</td>
                <td class="numeric" style="color:{_t["violet"]}; font-weight:600;">{confluence:.2f}</td>
            </tr>""")
    head = (_th("Rank") + _th("Symbol", numeric=False) + _th("Corr") + _th("Trace", _TH_TRACE)
            + _th("State", "An event on this bar, a capitulation still cheapening (watch), or the standing declaration")
            + _th("Grid", _TH_GRID, numeric=False)
            + _th("Zone", "Cumulative-delta flow zone — context only")
            + _th("Actual %", "Symbol's price change on the analysis date")
            + _th("Expected %", "Target return × beta (rolling correlation × vol ratio)")
            + _th("Div %", "Actual − Expected (positive = outperforming expectation)")
            + _th("Confluence", "|Correlation| × normalised signal strength (a ▲▼ today, else "
                                "how far the trace is stretched)"))
    return _html_doc(head, rows, _MAXH)


def _age_labels(timeframe: str) -> list:
    if timeframe == 'Weekly':
        return ["This Week", "1 Week Ago", "2 Weeks Ago", "3 Weeks Ago", "Within 5 Weeks"]
    return ["Today", "1 Day Ago", "2 Days Ago", "3 Days Ago", "Within 5 Days"]


def _bucket_signals_by_age(results_df: pd.DataFrame, side: str = 'buy', timeframe: str = 'Daily') -> tuple:
    """Bucket fired events by age (Today, 1d, 2d, 3d, within 5d) for the timeline.

    side: 'buy' (long events, BUY_* columns) or 'sell' (short events, SELL_*). Each cell
    holds the event's glyph at that age — ▲/▼ for capitulation / distribution, ◆ for a RESUME — or '—'.

    A symbol appears in the NEWEST bucket it fired in and nowhere else, and each row
    carries the readings from the bar that fired it (the trace and the grid units from
    their per-age histories) plus an entry-exhaustion read.
    """
    prefix = 'BUY' if _is_buy_side(side) else 'SELL'
    age_labels = _age_labels(timeframe)
    buckets = {label: [] for label in age_labels}
    col_map = dict(zip(age_labels, [f"{prefix}_Today", f"{prefix}_1d", f"{prefix}_2d",
                                    f"{prefix}_3d", f"{prefix}_5d"]))
    seen = set()
    for _offset, age in enumerate(age_labels):
        col = col_map[age]
        if col not in results_df.columns:
            continue
        subset = results_df[(results_df[col].astype(str) != "—") & (~results_df['Symbol'].isin(seen))]
        for _, r in subset.iterrows():
            r = r.copy()
            # Walked newest-first with `seen` blocking re-listing, so a symbol reaching
            # the last bucket did NOT fire at offsets 0-3; the *_5d column holds the newest
            # glyph in the window, which is then offset 4's.
            glyph = str(r.get(col, "—"))
            kind = "TURN" if glyph in ("▲", "▼") else "RESUME"

            def _at(hist_col, fallback_col):
                v = r.get(hist_col)
                if isinstance(v, (list, tuple)) and _offset < len(v):
                    x = v[_offset]
                    if pd.notna(x) and np.isfinite(float(x)):
                        return x
                return r.get(fallback_col, float('nan'))

            r['_kind'] = kind
            r['_event'] = (("▲ CAPITULATION" if glyph == "▲" else "▼ DISTRIBUTION") if kind == "TURN"
                           else f"◆ RESUME {'↑' if _is_buy_side(side) else '↓'}")
            r['_fire_trace'] = _at('Trace_Hist', 'PRG_Trace')
            r['_fire_units'] = _at('Units_Hist', 'CVG_Units')
            r['_age_offset'] = _offset
            r['_entry'] = _entry_status(r, _offset, side)
            buckets[age].append(r)
            seen.add(r['Symbol'])

    stats = {}
    for age, rows in buckets.items():
        units = [float(r['_fire_units']) for r in rows
                 if pd.notna(r.get('_fire_units')) and np.isfinite(float(r['_fire_units']))]
        stats[age] = {
            'count': len(rows),
            'turns': sum(1 for r in rows if r['_kind'] == "TURN"),
            'avg_units': float(np.mean(units)) if units else float('nan'),
            'avg_pct_change': float(np.mean([r.get('PctChange', 0) or 0 for r in rows])) if rows else 0.0,
            'rows': rows,
        }
    n_turn = sum(s['turns'] for s in stats.values())
    n_all = sum(s['count'] for s in stats.values())
    trend = f"{n_turn} {'▲' if _is_buy_side(side) else '▼'} · {n_all - n_turn} ◆ in the last 5 bars"
    trend_color = _side_palette(side)["accent_light"] if n_all else _neut_c()
    return buckets, stats, trend, trend_color


def _build_signal_table_html(stats: dict, side: str = 'buy', timeframe: str = 'Daily') -> str:
    """Build the age-grouped HTML table of fired events, with section headers."""
    _pal = _side_palette(side)
    accent_light = _pal["accent_light"]
    _MAXH = 760
    _NCOLS = 12
    rows = []
    for age in _age_labels(timeframe):
        if stats[age]['count'] == 0:
            continue
        s = stats[age]
        u = s['avg_units']
        rows.append(f"""
        <tr>
            <td class="sect" colspan="{_NCOLS}" style="color: {accent_light};">
                {_pal["mark"]} {age} · {s['count']} {_pal["label"]} event{'s' if s['count'] != 1 else ''}
                · {s['turns']} {'▲' if _pal["mark"] == '▲' else '▼'} · grid {'' if not np.isfinite(u) else f'{u:.2f}u avg'} · Avg %: {s['avg_pct_change']:+.1f}
            </td>
        </tr>""")
        for row in s['rows']:
            symbol = html.escape(str(row.get('DisplayName', row.get('Symbol', ''))))
            price = float(row.get('Price', 0) or 0)
            pct_change = float(row.get('PctChange', 0) or 0)
            rows.append(f"""
            <tr>
                <td class="symbol">{symbol}</td>
                <td class="numeric currency">{price:,.2f}</td>
                <td class="numeric" style="color: {_signed_color(pct_change)}; font-weight: 600;">{pct_change:+.2f}%</td>
                {_event_cell(row.get('_event'))}
                {_grid_cell(row)}
                {_push_cell(row.get('PRG_Push'), bool(row.get('CVG_Held', False)), str(row.get('PRG_Push_Tier', '') or ''))}
                {_trace_cell(row.get('_fire_trace'))}
                {_tape_cell(row.get('PRG_CTape'), 'conv')}
                {_tape_cell(row.get('PRG_VTape'), 'value')}
                {_evidence_cell(row, side)}
                {_hold_cell(row.get('PRG_Hold_Age'), row.get('PRG_Horizon', eng.HORIZON), row.get('PRG_Hold_Dir'))}
                {_status_cell(row.get('_entry', ('—', _dim(), '')))}
            </tr>""")
    if not rows:
        rows.append(f'<tr><td class="empty" colspan="{_NCOLS}">— no {_pal["label"]} events in the last 5 bars —</td></tr>')
    head = (_th("Symbol", numeric=False) + _th("Price") + _th("% Change")
            + _th("Event", "▲ CAPITULATION — sellers in control of a cheap price, value turning back toward fair · ▼ DISTRIBUTION — sellers taking control of a rich price · ◆ RESUME — a trend resuming")
            + _th("Grid", _TH_GRID, numeric=False) + _th("Push", _TH_PUSH)
            + _th("Trace", "The trace at the bar that FIRED this event. " + _TH_TRACE)
            + _th("C", _TH_C) + _th("V", _TH_V)
            + _th("Evidence", "div: a qualified divergence · abs: effort absorbed · split: the "
                              "ingredients disagree (gold)")
            + _th("Hold", "Bars into the declared hold window (entry was the open after the signal bar)")
            + _th("Entry", "Has price already run in the signal's direction since it fired (σ units)?"))
    return _html_doc(head, rows, _MAXH)


def _build_narrative_table_html(df: pd.DataFrame, side: str = 'buy') -> str:
    """Full-universe HTML table: every symbol's state, grid and readings."""
    _MAXH = 1200
    _NCOLS = 12
    rows = []
    if df.empty:
        rows.append(f'<tr><td class="empty" colspan="{_NCOLS}">— no data available —</td></tr>')
    else:
        _t = ui.table_tokens()
        for _, row in df.iterrows():
            symbol = html.escape(str(row.get('DisplayName', row.get('Symbol', ''))))
            price = float(row.get('Price', 0) or 0)
            pct_change = float(row.get('PctChange', 0) or 0)
            bar_delta = float(row.get('Bar_Delta', 0) or 0)
            abs_strength = float(row.get('Abs_Strength', 0) or 0)
            abs_color = _signed_color(abs_strength - 1.0, pos=_t["amber"], neg=_t["cyan"])
            rows.append(f"""
            <tr>
                <td class="symbol" style="color: {_t["ink_primary"]};">{symbol}</td>
                <td class="numeric currency">{price:,.2f}</td>
                <td class="numeric" style="color: {_signed_color(pct_change)}; font-weight: 600;">{pct_change:+.2f}%</td>
                {_state_cell(row)}
                {_grid_cell(row)}
                {_push_cell(row.get('PRG_Push'), bool(row.get('CVG_Held', False)), str(row.get('PRG_Push_Tier', '') or ''))}
                {_trace_cell(row.get('Signal_Score'))}
                {_tape_cell(row.get('PRG_CTape'), 'conv')}
                {_tape_cell(row.get('PRG_VTape'), 'value')}
                {_hold_cell(row.get('PRG_Hold_Age'), row.get('PRG_Horizon', eng.HORIZON), row.get('PRG_Hold_Dir'))}
                <td class="numeric" style="color: {_t["ink_secondary"]}; font-weight: 600;">{_human_vol(bar_delta)}</td>
                <td class="numeric" style="color: {abs_color}; font-weight: 600;">{abs_strength:.2f}×</td>
            </tr>""")
    head = (_th("Symbol", numeric=False) + _th("Price") + _th("% Change")
            + _th("State", "An event on this bar, a capitulation still cheapening (watch, gold), the standing declaration, or —")
            + _th("Grid", _TH_GRID, numeric=False) + _th("Push", _TH_PUSH) + _th("Trace", _TH_TRACE)
            + _th("C", _TH_C) + _th("V", _TH_V)
            + _th("Hold", "Bars into the declared hold window of the latest event")
            + _th("Bar Δ", "Inferred bar delta — context only")
            + _th("Absorp", "Absorption strength — context only"))
    return _html_doc(head, rows, _MAXH)


def _build_signal_strength_table_html(df: pd.DataFrame, side: str = 'buy') -> str:
    """Ranked HTML table for one side, by the side's priority (a ▲▼ on this bar, then stretch)."""
    _pct_col = _priority_pct_col(side)
    _is_buy = _is_buy_side(side)
    _MAXH = 900
    _NCOLS = 13
    rows = []
    if df.empty:
        rows.append(f'<tr><td class="empty" colspan="{_NCOLS}">— no symbols to rank —</td></tr>')
    else:
        _t = ui.table_tokens()
        for idx, (_, row) in enumerate(df.iterrows(), 1):
            symbol = html.escape(str(row.get('DisplayName', row.get('Symbol', ''))))
            price = float(row.get('Price', 0) or 0)
            pct_change = float(row.get('PctChange', 0) or 0)
            pct_rank = float(row.get(_pct_col, 0) or 0)
            hmm_bull = float(row.get('HMM_Bull', 0.5) or 0.5)
            hmm_bear = float(row.get('HMM_Bear', 0.5) or 0.5)
            vol_reg = str(row.get('Vol_Regime', 'NORMAL'))
            regime_tag, regime_color = "NEUTRAL", _neut_c()
            if _is_buy:
                if hmm_bull > 0.7: regime_tag, regime_color = "BULL", _t["emerald"]
                elif hmm_bull < 0.3: regime_tag, regime_color = "BEAR", _t["rose"]
            else:
                if hmm_bear > 0.7: regime_tag, regime_color = "BEAR", _t["rose"]
                elif hmm_bear < 0.3: regime_tag, regime_color = "BULL", _t["emerald"]
            vol_color = {"LOW": _t["accent"], "NORMAL": _t["ink_tertiary"],
                         "HIGH": _t["amber"], "EXTREME": _t["rose"]}.get(vol_reg, _t["ink_tertiary"])
            rows.append(f"""
            <tr>
                <td class="numeric" style="color: {_t["accent"]}; font-weight: 700;">{idx:02d}</td>
                <td class="symbol">{symbol}</td>
                <td class="numeric" style="color: {_t["accent"]}; font-weight: 700;">TOP {min(100.0, 101-pct_rank):,.1f}%</td>
                <td class="numeric currency">{price:,.2f}</td>
                <td class="numeric" style="color: {_signed_color(pct_change)}; font-weight: 600;">{pct_change:+.2f}%</td>
                {_state_cell(row)}
                {_grid_cell(row)}
                {_push_cell(row.get('PRG_Push'), bool(row.get('CVG_Held', False)), str(row.get('PRG_Push_Tier', '') or ''))}
                {_trace_cell(row.get('Signal_Score'))}
                {_tape_cell(row.get('PRG_CTape'), 'conv')}
                {_tape_cell(row.get('PRG_VTape'), 'value')}
                <td class="numeric" style="color: {regime_color}; font-weight: 700; font-size: {ui.FS["2xs"]};">{regime_tag}</td>
                <td class="numeric" style="color: {vol_color}; font-weight: 700; font-size: {ui.FS["2xs"]};">{vol_reg}</td>
            </tr>""")
    head = (_th("Rank") + _th("Symbol", numeric=False)
            + _th("Percentile", "This side's priority percentile. A ▲▼ on this bar first, then "
                                "by stretch: the long side leads with the names stretched furthest "
                                "down, the short side with the names stretched furthest up.")
            + _th("Price") + _th("% Change")
            + _th("State", "An event on this bar, a capitulation still cheapening (watch, gold), the standing declaration, or —")
            + _th("Grid", _TH_GRID, numeric=False) + _th("Push", _TH_PUSH) + _th("Trace", _TH_TRACE)
            + _th("C", _TH_C) + _th("V", _TH_V)
            + _th("Regime", "HMM regime — risk context, not a signal input")
            + _th("Vol", "GARCH volatility regime — risk context, not a signal input"))
    return _html_doc(head, rows, _MAXH)


_CENSUS_CELL_H = 66   # three stacked lines of text + padding, measured in the rendered iframe


def _census_iframe_height(n_unread: int) -> int:
    """The census is four tall rows, not four table rows — size its iframe for that."""
    return ui.TABLE_HEADER_H + cg.N_ROWS * _CENSUS_CELL_H + (ui.TABLE_ROW_H + 8 if n_unread else 0) + 6


def _build_grid_census_html(df: pd.DataFrame) -> str:
    """The 3 × 3 as the pane's grid: rows are who controls, columns where price stands.

    Each cell names its action and counts the names in it; held rows are counted apart,
    in gold, because a held row is a claim the push has not yet backed.
    """
    _t = ui.table_tokens()
    cells = pd.to_numeric(df.get('CVG_Cell', pd.Series(dtype=float)), errors='coerce').fillna(cg.UNREAD).astype(int)
    held = df.get('CVG_Held', pd.Series(False, index=df.index)).fillna(False).astype(bool)
    n_read = int((cells != cg.UNREAD).sum())
    rows = []
    for r in reversed(range(cg.N_ROWS)):
        tds = [f'<td class="symbol" style="color:{_t["ink_secondary"]}; white-space:nowrap;">'
               f'{html.escape(cg.ROW_LABELS[r])}</td>']
        for c in range(cg.N_COLS):
            cell = r * cg.N_COLS + c
            n = int((cells == cell).sum())
            nh = int(((cells == cell) & held).sum())
            tone = cg.TONES[cell]
            col = _tone_ink(tone, _t)
            tint = "" if tone == "slate" or not n else f" background:{charts.chart_rgba(tone, 0.08)};"
            share = f"{n / n_read * 100:.0f}%" if n_read else "—"
            held_t = (f' <span style="color:{_t["amber"]};" title="rows held against their tape">'
                      f'· {nh} held</span>') if nh else ""
            tip = f"{cg.NAMES[cell]} ({cg.FAMILY[cell]}) - {cg.MEANING[cell]}. {cg.UNITS[cell]:g} units."
            tds.append(
                f'<td style="text-align:center; padding:0.45rem 0.35rem; opacity:{1.0 if n else 0.45};{tint}" '
                f'title="{html.escape(tip)}">'
                f'<div style="color:{col}; font-weight:700; font-size:{ui.FS["xs"]};">'
                f'<span style="font-size:{ui.FS["3xs"]};">{_TONE_GLYPH[tone]}</span> {html.escape(cg.action(cell))}</div>'
                f'<div style="color:{_t["ink_tertiary"]}; font-size:{ui.FS["2xs"]};">{html.escape(cg.reason(cell))} · {cg.UNITS[cell]:g}u</div>'
                f'<div style="color:{_t["ink_primary"]}; font-weight:700; font-size:{ui.FS["sm"]}; margin-top:2px;">'
                f'{n}<span style="color:{_t["ink_tertiary"]}; font-weight:400;"> · {share}</span>{held_t}</div></td>')
        rows.append("<tr>" + "".join(tds) + "</tr>")
    n_un = int((cells == cg.UNREAD).sum())
    if n_un:
        rows.append(f'<tr><td class="empty" colspan="{cg.N_COLS + 1}">{n_un} name{"s" if n_un != 1 else ""} unread — '
                    f'a tape not yet calibrated</td></tr>')
    head = (_th("conviction ↓ · value →", numeric=False)
            + "".join(_th(cg.COL_LABELS[c], "value ≤ −θ" if c == 0 else "−θ < value < θ" if c == 1
                          else "value ≥ θ", center=True) for c in range(cg.N_COLS)))
    return _html_doc(head, rows, 420)


# ══════════════════════════════════════════════════════════════════════════════
# CORRELATION MODE — RESULTS RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def render_correlation_results(corr_data: dict) -> None:
    """Render Correlation mode 4-tab results interface."""
    corr_df = corr_data["corr_df"]
    rolling_corr_df = corr_data["rolling_corr"]
    target_ticker = corr_data["target_ticker"]
    target_name = corr_data["target_name"]
    lookback = corr_data["lookback"]
    method = corr_data["method"]
    trigger = corr_data.get("trigger") or "▲ capitulation · ▼ distribution · ◆ RESUME"
    iclass = corr_data.get("iclass", "—")

    tab1, tab2, tab3 = st.tabs([
        "Correlation Dashboard",
        "Confluence Setups",
        "Heatmap Matrix"
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1: CORRELATION DASHBOARD
    # ═══════════════════════════════════════════════════════════════════════════
    with tab1:
        ui.render_section_header(
            "Correlation Dashboard",
            f"Target: {target_name} ({target_ticker}) | {lookback}D Rolling {method}",
            icon="crosshair",
            accent="violet"
        )

        # Summary metrics
        strong_corr_count = len(corr_df[corr_df['Corr_Current'] >= 0.6])
        strong_inv_count = len(corr_df[corr_df['Corr_Current'] <= -0.6])
        avg_abs_corr = abs(corr_df['Corr_Current']).mean()
        target_change = corr_df['Target_Pct'].iloc[0] if len(corr_df) > 0 else 0

        metrics = [
            {"label": "Target Performance", "value": f"{target_change:+.2f}%", "kind": "success" if target_change >= 0 else "danger"},
            {"label": "Highly Correlated", "value": str(strong_corr_count), "kind": "info"},
            {"label": "Highly Inverse", "value": str(strong_inv_count), "kind": "warning"},
            {"label": "Avg |Correlation|", "value": f"{avg_abs_corr:.2f}", "kind": "neutral"},
            {"label": "Correlation Signal", "value": "CONCENTRATED" if strong_corr_count > len(corr_df) * 0.3 else "DIVERSIFIED", "kind": "violet"},
        ]

        cols = st.columns(len(metrics))
        for i, m in enumerate(metrics):
            with cols[i]:
                ui.render_metric_card(m["label"], m["value"], color_class=m["kind"])

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

        # Ranked lists
        def _corr_rows(frame, tone: str) -> str:
            """One correlation list as panel rows, in the app's own grammar."""
            out = []
            for _, r in frame.iterrows():
                cv = float(r['Corr_Current'])
                trend = ("\u2191" if r['Corr_Trend'] > 0.05
                         else "\u2193" if r['Corr_Trend'] < -0.05 else "\u2192")
                # Fill is a PERCENTAGE of the track. It used to be `|corr|*50`
                # px against a track with no declared width, so how long a bar
                # looked depended on the column it landed in rather than on the
                # correlation it was drawing.
                pct = min(abs(cv), 1.0) * 100.0
                out.append(
                    f'<div class="lookback-row">'
                    f'<span class="lbl">{html.escape(str(r["SimpleName"]))}'
                    f'<span class="sub"> {r["PctChange"]:+.2f}% · exp '
                    f'{r["Expected_Change"]:+.2f}%</span></span>'
                    f'<span class="val {tone}">'
                    f'<span class="conviction-bar" style="display:inline-block;'
                    f'width:54px;vertical-align:middle;margin-right:8px;">'
                    f'<span class="conviction-bar-fill fill-{"buy" if tone == "long" else "caution"}" '
                    f'style="display:block;width:{pct:.0f}%;"></span></span>'
                    f'{cv:+.3f} {trend}</span></div>'
                )
            return "".join(out)

        col_pos, col_neg = st.columns(2)

        with col_pos:
            ui.render_section_header("Top Positively Correlated", icon="trending", accent="emerald")
            pos_corr = corr_df[corr_df['Corr_Current'] > 0].head(7)
            with ui.panel("corr-pos", context=f"{len(pos_corr)} of "
                          f"{int((corr_df['Corr_Current'] > 0).sum())} positive"):
                st.markdown(f'<div class="panel-specs">{_corr_rows(pos_corr, "long")}</div>',
                            unsafe_allow_html=True)

        with col_neg:
            ui.render_section_header("Top Inversely Correlated", icon="trending", accent="rose")
            neg_corr = corr_df[corr_df['Corr_Current'] < 0].head(7)
            with ui.panel("corr-neg", context=f"{len(neg_corr)} of "
                          f"{int((corr_df['Corr_Current'] < 0).sum())} inverse"):
                st.markdown(f'<div class="panel-specs">{_corr_rows(neg_corr, "short")}</div>',
                            unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2: TRADE INTELLIGENCE
    # ═══════════════════════════════════════════════════════════════════════════
    with tab2:
        ui.render_section_header(
            "Confluence Setups",
            f"Confluence: Correlation × Pragati signal strength · {trigger} · {iclass}",
            icon="zap",
            accent="cyan"
        )

        # How to read this tab. Was a hand-built box tinted in the retired
        # cyan — and a PLAIN string carrying {…} placeholders, so those colours
        # rendered as literal braces and never applied. The component does not
        # have that failure mode because it takes text, not markup.
        ui.render_info_box(
            "How to read",
            "Each setup type is ranked by Confluence Score (0-1) = |Correlation| \u00d7 "
            "normalised signal strength. Highest rank = strongest overlap between the "
            "correlation relationship and a live Pragati reading. Look for a score above "
            "0.7, a divergence past \u00b13%, and an event (\u25b2 capitulation / \u25bc distribution) "
            "or a watch in State rather than a dash — and read the Grid column "
            "for where the name already stands.",
            color="cyan",
        )

        # Trade setup classification.
        # Thresholds: corr ±0.4 = meaningful directional relationship;
        # div ±2 = at least 2% price divergence from the target asset;
        # zone conditions ensure the flow read agrees with the setup direction.
        # Zone vocabulary = the flow Condition column (Accumulation*/Distribution*):
        # Distribution zones (net selling flow) play the oversold side, Accumulation
        # zones (net buying flow) the overbought side — the same mapping the
        # historical dashboard uses. (The old OB/OS names died with the WRCI engine;
        # matching on them made LAGGARD/RUNAWAY/CONTRA unreachable.)
        _CORR_THRESH = 0.4   # minimum |correlation| to consider a relationship directional
        _DIV_THRESH  = 2.0   # minimum % divergence to flag a laggard / runaway
        _ZONES_SOLD   = ('Distribution', 'Distribution+')   # net selling flow
        _ZONES_BOUGHT = ('Accumulation', 'Accumulation+')   # net buying flow
        def classify_setup(row):
            corr = row['Corr_Current']
            div = row['Divergence']
            zone = row['Regime_Zone']

            # Div = Actual − Expected: NEGATIVE = the name underperformed what the
            # correlation implied (a laggard), POSITIVE = it outran the implication.
            # (The pre-rewrite version had these signs inverted vs its own rationale.)
            if corr > _CORR_THRESH and div < -_DIV_THRESH and zone in _ZONES_SOLD:
                return "LAGGARD"
            elif corr > _CORR_THRESH and div > _DIV_THRESH and zone in _ZONES_BOUGHT:
                return "RUNAWAY"
            elif abs(corr) < 0.2:
                return "CONVERGING"
            elif corr < -_CORR_THRESH and div > _DIV_THRESH and zone in _ZONES_BOUGHT:
                return "CONTRA"
            else:
                return "NEUTRAL"

        corr_df['Setup'] = corr_df.apply(classify_setup, axis=1)

        # Summary metrics
        laggard_count = len(corr_df[corr_df['Setup'] == 'LAGGARD'])
        runaway_count = len(corr_df[corr_df['Setup'] == 'RUNAWAY'])
        converging_count = len(corr_df[corr_df['Setup'] == 'CONVERGING'])
        contra_count = len(corr_df[corr_df['Setup'] == 'CONTRA'])
        avg_confluence = corr_df[corr_df['Setup'] != 'NEUTRAL']['Confluence_Score'].mean()

        metrics = [
            {"label": "Laggard Setups", "value": str(laggard_count), "kind": "success"},
            {"label": "Runaway Setups", "value": str(runaway_count), "kind": "danger"},
            {"label": "Converging", "value": str(converging_count), "kind": "warning"},
            {"label": "Contra Setups", "value": str(contra_count), "kind": "info"},
            {"label": "Avg Confluence", "value": f"{avg_confluence:.2f}", "kind": "neutral"},
        ]

        cols = st.columns(len(metrics))
        for i, m in enumerate(metrics):
            with cols[i]:
                ui.render_metric_card(m["label"], m["value"], color_class=m["kind"])


        # Render each setup type as a section
        setup_configs = [
            {
                "name": "LAGGARD",
                "title": "Laggard Setups",
                "description": "High corr + oversold + underperforming — expect catch-up rally",
                "color": _long_c(),
                "bg_color": "var(--long-fill)",
                "border_color": "var(--long-edge)"
            },
            {
                "name": "RUNAWAY",
                "title": "Runaway Setups",
                "description": "High corr + overbought + overextended — expect pullback",
                "color": ui.table_tokens()["rose"],
                "bg_color": "var(--short-fill)",
                "border_color": "var(--short-edge)"
            },
            {
                "name": "CONVERGING",
                "title": "Converging Setups",
                "description": "Low corr or normalizing — expect tightening after divergence",
                "color": ui.table_tokens()["amber"],
                "bg_color": "var(--caution-fill)",
                "border_color": "var(--caution-edge)"
            },
            {
                "name": "CONTRA",
                "title": "Contra Setups",
                "description": "Strong negative corr + overbought — expect rally vs target decline",
                "color": ui.table_tokens()["violet"],
                "bg_color": "var(--violet-fill)",
                "border_color": "var(--violet-edge)"
            }
        ]

        # Setup interpretation guide
        setup_interpretation = {
            "LAGGARD": {
                "action": "BUY",
                "rationale": "Stock lagging its correlation-implied move, with selling flow already absorbed — expect catch-up toward the target's pace",
                "validate": "Check that Zone is Distribution/Distribution+ and Div % is negative & large (<-3%)",
                "risk": "Correlation may break; stock continues lagging instead of catching up"
            },
            "RUNAWAY": {
                "action": "SHORT",
                "rationale": "Stock outran its correlation-implied move on buying flow — expect pullback to fair value",
                "validate": "Check that Zone is Accumulation/Accumulation+ and Div % is positive & large (>3%)",
                "risk": "Stock may continue running; wait for flow to weaken before shorting"
            },
            "CONVERGING": {
                "action": "DE-RISK",
                "rationale": "Correlation collapsing — pair-trade falling apart, avoid new entries",
                "validate": "Corr close to 0 or unstable; watch for re-correlation before re-entering",
                "risk": "Old positions may unwind suddenly; previous divergence trades may fail"
            },
            "CONTRA": {
                "action": "LONG (vs target)",
                "rationale": "Strong inverse mover beating its inverse-implied move on buying flow — relative-strength long against target weakness",
                "validate": "Check Corr is strongly negative (<-0.4), Div % positive & large, Zone Accumulation/Accumulation+",
                "risk": "Negative correlations are unstable; requires conviction and risk management"
            }
        }

        for config in setup_configs:
            setup_data = corr_df[corr_df['Setup'] == config['name']].nlargest(10, 'Confluence_Score')

            if len(setup_data) > 0:
                st.markdown(f"""
                <div style="display:flex; align-items:baseline; gap:0.65rem; margin:1.75rem 0 0.9rem 0;
                             padding-bottom:0.6rem; border-bottom:1px solid {config['border_color']};">
                    <span style="font-family:var(--display); font-size:var(--fs-2xs); font-weight:700;
                                 letter-spacing:0.12em; text-transform:uppercase; color:{config['color']};
                                 padding:0.18rem 0.5rem; background:{config['bg_color']};
                                 border:1px solid {config['border_color']}; border-radius:4px;">
                        {config['name']}</span>
                    <span style="font-family:var(--display); font-size:var(--fs-lg); font-weight:700;
                                 color:{ui.table_tokens()["ink_primary"]}; letter-spacing:0.04em;">{config['title']}</span>
                    <span style="font-family:var(--data); font-size:var(--fs-sm); color:{_dim()};">
                        {config['description']}</span>
                    <span style="margin-left:auto; font-family:var(--data); font-size:var(--fs-sm);
                                 color:{config['color']};">→ {len(setup_data)}</span>
                </div>
                """, unsafe_allow_html=True)

                # Interpretation card
                interp = setup_interpretation[config['name']]
                st.markdown(f"""
                <div style="background:{config['bg_color']}; border:1px solid {config['border_color']};
                            border-radius:8px; padding:0.75rem 1rem; margin-bottom:1rem; font-family:var(--data); font-size:var(--fs-sm);">
                    <div style="display:grid; grid-template-columns:auto 1fr; gap:0.5rem 1rem; color:{ui.table_tokens()["ink_primary"]};">
                        <span style="color:{config['color']}; font-weight:700; text-transform:uppercase;">Action</span>
                        <span>{interp['action']}</span>
                        <span style="color:{config['color']}; font-weight:700; text-transform:uppercase;">Rationale</span>
                        <span>{interp['rationale']}</span>
                        <span style="color:{config['color']}; font-weight:700; text-transform:uppercase;">Validate</span>
                        <span>{interp['validate']}</span>
                        <span style="color:{ui.table_tokens()["rose"]}; font-weight:700; text-transform:uppercase;">⚠ Risk</span>
                        <span style="color:{ui.table_tokens()["rose"]};">{interp['risk']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Display as two-column table
                col_left, col_right = st.columns(2)
                with col_left:
                    st.markdown(f"""<p style="font-family:var(--data); font-size:var(--fs-2xs); font-weight:600;
                                   text-transform:uppercase; letter-spacing:0.1em; color:{config['color']};
                                   margin:0 0 0.4rem 0; display:flex; align-items:center; gap:0.35rem;">
                        Top Confluence</p>""", unsafe_allow_html=True)
                    top_half = setup_data.head(5)
                    if len(top_half) > 0:
                        with ui.html_panel(f"conf-{config['name'].lower()}-a",
                                           context=f"{config['name']} · top half"):
                            st.components.v1.html(
                                _build_confluence_table_html(top_half),
                                height=ui.table_iframe_height(len(top_half), max_height=560))
                with col_right:
                    st.markdown(f"""<p style="font-family:var(--data); font-size:var(--fs-2xs); font-weight:600;
                                   text-transform:uppercase; letter-spacing:0.1em; color:{config['color']};
                                   margin:0 0 0.4rem 0; display:flex; align-items:center; gap:0.35rem;">
                        Also Considered</p>""", unsafe_allow_html=True)
                    bottom_half = setup_data.iloc[5:10]
                    if len(bottom_half) > 0:
                        with ui.html_panel(f"conf-{config['name'].lower()}-b",
                                           context=f"{config['name']} · bottom half"):
                            st.components.v1.html(
                                _build_confluence_table_html(bottom_half),
                                height=ui.table_iframe_height(len(bottom_half), max_height=560))
                    else:
                        ui_info("No additional setups")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3: HEATMAP MATRIX
    # ═══════════════════════════════════════════════════════════════════════════
    with tab3:
        ui.render_section_header("Correlation Matrix", "Top constituents by |correlation|", icon="grid", accent="violet")

        # Build heatmap data using Symbol (original ticker) to match rolling_corr_df columns
        top_by_corr = corr_df.copy()
        top_by_corr['AbsCorr'] = abs(top_by_corr['Corr_Current'])
        top_rows = top_by_corr.nlargest(30, 'AbsCorr')
        top_symbols = top_rows['Symbol'].tolist()
        valid_symbols = [s for s in top_symbols if s in rolling_corr_df.columns]
        heatmap_data = rolling_corr_df[valid_symbols].iloc[-1:].T if valid_symbols else pd.DataFrame()

        if len(heatmap_data) > 0:
            # Filter to only the top symbols that exist in rolling_corr_df
            heatmap_rows = corr_df[corr_df['Symbol'].isin(valid_symbols)].copy()
            heatmap_rows = heatmap_rows.sort_values('Corr_Current', ascending=True)
            fig = charts.create_correlation_heatmap(heatmap_rows['SimpleName'].values,
                                                    heatmap_rows['Corr_Current'].values)
            ui.render_chart_panel(fig, key='corr_0', context=_chart_ctx())
        else:
            ui_info("No correlation data available for heatmap")


_SIGNAL_TYPE_REFERENCE = [
    ("▲ CAPITULATION · ▼ DISTRIBUTION — read from the grid", "emerald",
     "v9's events. ▲ fires on the first closed bar the grid stands in Buy · capitulation — the "
     "conviction tape past −30 and held there by conviction's own histogram (sellers in "
     "control across the ladder), the value tape cheap past −θ — with value momentum "
     "REVERTING: the fast end of value has already turned back toward fair. ▼ fires on the "
     "first bar sellers hold control of a price rich past +θ (Exit · distribution). A 10-bar "
     "cooldown per side; the last one stands as the declaration, with no exit. Measured in "
     "the v9 audit (studies/pragati_v9_audit.md; 380 instruments, six classes, three eras, "
     "scored without look-ahead): the ▲ was positive in 2006-13, 2014-19 and 2020-26, on daily "
     "and weekly bars — about +0.05σ over 10-20 bars, a lean rather than a trade, and not on "
     "crypto. The ▼'s state was followed by underperformance in every era; the entry itself "
     "is too rare to measure alone. v8's TURN (a stretch crossing back through θ, confirmed on "
     "its tapes) faded to nothing after 2020 and read negative on weekly bars; v9.2 removed "
     "it."),
    ("◆ RESUME · a trend resuming", "cyan",
     "OFF BY DEFAULT since v8. Measured across 380 instruments it was negative outside crypto "
     "in the v8 audit and negative or mixed again in v9, so the screen does not fire it. The rule, when switched on (engine settings): chart conviction "
     "on the ◆'s side; the histogram dipped to the wrong side inside 6 bars and now crosses its "
     "impulse gate (k·σ); the conviction tape past its inner zone on the ◆'s side; the value "
     "tape short of θ; effort not absorbed on the bar. A ▲▼ takes precedence, and the two "
     "share one 10-bar cooldown per direction."),
    ("The grid · where a name stands between signals", "violet",
     "The conviction tape (who controls) is the row — UP past +30, DOWN past −30, FAINT "
     "between — and the value tape (where price stands) the column — cheap past −θ, rich past "
     "+θ, fair between: Pragyam's 3 × 3, the grid pragati.pine v9.2 draws. Its units were MEASURED "
     "(studies/): Buy · capitulation 4 and Accumulate · washout 1½ where sellers "
     "hold a cheap or fair price, Buy · turned 3, Hold · building 1½ and Trim · paid ¾ where "
     "buyers do. Conviction's own histogram decides when a row may change — a confirmed push, "
     "else the row is HELD (gold). The units are graded by where the name sits inside its "
     "cell. A weight, not a forecast — and the source of v9's ▲▼."),
    ("Scope · measured on YOUR universe", "amber",
     "The v9 audit measured the stack on 380 instruments (the readings describe; the "
     "capitulation carries the one robust edge). The app still measures it on YOUR symbols: an event study over ~15 years "
     "through the same engine call the screener makes, each instrument's own drift removed "
     "causally (its trailing drift), vol-normalised, block-bootstrapped over dates, with the power stated. "
     "Parameters are never tuned to your data. A 'no edge' or 'underpowered' verdict is "
     "reported, never applied: the signals still fire. The conviction ladder reads DOWN — "
     "the intraday frames yfinance carries inside each bar — and falls back to W · D (↺) "
     "on bars older than that history; PRG_Ladder says which."),
]


def _render_system_data_tab(results_df, analysis_date, universe=None, selected_index=None,
                            sid=None, study=None):
    """System Data tab — exports, raw frame, the signal reference and the Edge Study."""
    if sid is None:
        sid = _active_engine_settings()
    ui.render_section_header(
        "System Data",
        "Exports, raw signal frame, and reference legends",
        icon="database", accent="cyan",
    )

    # ── Downloads — split on the FIRED events, not on the sign of a level ──
    _side = results_df['Side'] if 'Side' in results_df.columns else None
    buy_df = results_df[_side == 'Buy'] if _side is not None else results_df.iloc[0:0]
    sell_df = results_df[_side == 'Sell'] if _side is not None else results_df.iloc[0:0]

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            "↓ Full Report (Excel)",
            data=to_excel(results_df),
            file_name=build_download_filename("snapshot", universe=universe,
                                              selected_index=selected_index,
                                              dates=analysis_date, ext="xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width='stretch', key="sysdata_dl_full",
            help=(f"All {len(results_df)} symbols with every computed column, and a Legend "
                  "sheet defining each one: the signal set, the readings, the grid state, "
                  "the descriptive order-flow context and the regime columns."),
        )
    with dl2:
        st.download_button(
            "▲ Long Events (Excel)",
            data=to_excel(buy_df),
            file_name=build_download_filename("long", universe=universe,
                                              selected_index=selected_index,
                                              dates=analysis_date, ext="xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width='stretch', key="sysdata_dl_buy", disabled=len(buy_df) == 0,
            help=f"{len(buy_df)} symbols firing ▲ capitulation or ◆ RESUME ↑ on this bar.",
        )
    with dl3:
        st.download_button(
            "▼ Short Events (Excel)",
            data=to_excel(sell_df),
            file_name=build_download_filename("short", universe=universe,
                                              selected_index=selected_index,
                                              dates=analysis_date, ext="xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width='stretch', key="sysdata_dl_sell", disabled=len(sell_df) == 0,
            help=f"{len(sell_df)} symbols firing ▼ distribution or ◆ RESUME ↓ on this bar.",
        )

    # ── Raw Data Table ────────────────────────────────────────────────────
    ui.render_section_header(
        "Raw Signal Frame",
        f"{len(results_df)} symbols · sorted by the trace (most stretched up first)",
        icon="list", accent="emerald",
    )
    cols = ["DisplayName", "Price", "Signal_Score", "PRG_Hist_Z", "PRG_Push", "PRG_CTape",
            "PRG_VTape", "Side", "Signal_Kind", "PRG_State", "CVG_Action", "CVG_Why",
            "CVG_Units", "CVG_Held", "PRG_Hold_Age", "PRG_Conv", "PRG_Value", "PRG_Hedge",
            "PRG_Drivers", "Trace_Rank_Pct"]
    if "% Chng Since" in results_df.columns and results_df["% Chng Since"].notna().any():
        cols.insert(2, "% Chng Since")
    cols += ["Zone", "Bar_Delta", "CVD_Slope", "Delta_Z", "Abs_Strength",
             "Vol_Regime", "Regime_Confidence"]
    cols += [c for c in ("BUY_Today", "BUY_1d", "BUY_2d", "BUY_3d", "BUY_5d",
                         "SELL_Today", "SELL_1d", "SELL_2d", "SELL_3d", "SELL_5d")
             if c in results_df.columns]
    cols += [c for c in ("Signal_Reason",) if c in results_df.columns]
    cols = [c for c in cols if c in results_df.columns]
    _col_display_names = {
        "DisplayName": "Symbol", "Signal_Score": "Trace", "PRG_Hist_Z": "Push σ",
        "PRG_Push": "Push", "PRG_CTape": "C tape", "PRG_VTape": "V tape",
        "Signal_Kind": "Kind", "PRG_State": "State", "CVG_Action": "Grid",
        "CVG_Why": "Grid why", "CVG_Units": "Units", "CVG_Held": "Held",
        "PRG_Hold_Age": "Hold Age", "PRG_Conv": "Conviction", "PRG_Value": "Value",
        "PRG_Hedge": "Hedge", "PRG_Drivers": "Drivers", "Trace_Rank_Pct": "Trace %ile",
        "Bar_Delta": "Bar Δ", "CVD_Slope": "CVD Slope", "Delta_Z": "Δ-Z",
        "Abs_Strength": "Absorption ×", "Signal_Reason": "Read",
    }
    _sort = "Signal_Score" if "Signal_Score" in results_df.columns else cols[0]
    display_frame = (results_df[cols]
                     .sort_values(_sort, ascending=False, na_position='last')
                     .rename(columns=_col_display_names))
    ui.render_table_panel(
        display_frame, key="sysdata-frame",
        context=f"{len(display_frame)} symbols",
        show_index=False, label_col="Symbol", max_height=560,
        col_precision={"Trace": 1, "Push σ": 2, "C tape": 0, "V tape": 0, "Units": 2,
                       "Hold Age": 0, "Conviction": 1, "Value": 1, "Hedge": 2, "Trace %ile": 1},
        sign_color_cols={"Trace", "Push σ", "C tape", "V tape", "Conviction", "Value"},
        footer=_glossary({
            "Trace": "Conviction × value on this chart, ±100 — how far the move is stretched. "
                     "θ = ±42.9. A level, not a signal.",
            "Push σ / Push": "The trace minus its 9-bar EMA, in σ of its own distribution, "
                             "and the same push in five levels (+2 impulse … −2 impulse).",
            "C tape / V tape": "The two tapes: who controls (conviction) and where price stands "
                               "(value, + rich / − cheap), across the ladder. The grid's axes.",
            "Kind / State": "The event fired on this bar (TURN = the ▲▼, RESUME = ◆) and the bar's "
                            "state — an event, WATCH (in capitulation, value still cheapening), PAUSED, or NEUTRAL.",
            "Grid / Units": "The conviction-value grid's action and its measured units — a weight, "
                            "not a forecast.",
            "Conviction / Value": "The trace's two ingredients on this chart, each on its ±100 scale.",
            "Hedge / Drivers": "How much of the macro hedge the value leg applies (its own "
                               "out-of-sample skill) and the drivers selected.",
        }),
    )

    # ── Run configuration — Pragyam's key/value readout ───────────────────
    ui.render_section_header(
        "Run Configuration",
        "The settings this frame was computed under",
        icon="cpu", accent="violet",
    )
    _p = sid.params
    ui.render_kv_table({
        "Universe": f"{universe or '—'}" + (f" · {selected_index}" if selected_index else ""),
        "Analysis date": str(analysis_date or "—"),
        "Timeframe · chart": f"{sid.timeframe} · {sid.chart}",
        "Conviction ladder": sid.ladder_label,
        "Value ladder": sid.value_ladder_label,
        "Length · smooth · norm": f"{_p.length} · {_p.smooth} · {_p.norm}"
                                  + (" (adapted)" if sid.norm_is_adapted else ""),
        "Signal EMA": str(_p.signal),
        "θ (trace stretch)": f"±{sv.THETA_OSC:.1f}",
        "▲▼ · ◆": f"{'on' if _p.turn else 'off'} · {'on' if _p.resume else 'off'}",
        "Impulse k · cooldown": f"{float(_p.k):g}σ · {_p.cool} bars",
        "Hold horizon": f"{sid.horizon} bars",
        "Round-trip cost": f"{sid.cost_bps:g} bps",
        "Instrument class": sid.iclass,
    }, header_left="Setting", header_right="Value")

    # ── Signal Reference ──────────────────────────────────────────────────
    ui.render_section_header(
        "Signal Reference",
        "The two signals, the one state, and where they hold",
        icon="info", accent="amber",
    )
    ref_cols = st.columns(2)
    for i, (title, accent_key, body) in enumerate(_SIGNAL_TYPE_REFERENCE):
        with ref_cols[i % 2]:
            with ui.panel(f"sigref-{accent_key}", title):
                st.markdown(f'<div class="panel-copy">{body}</div>', unsafe_allow_html=True)

    # ── Edge Study ────────────────────────────────────────────────────────
    _render_edge_study_panel(sid, study)


def _render_grid_tab(results_df, sid, key: str = "grid") -> None:
    """The conviction-value grid across the universe: the census, the watchlist, the lists."""
    cells = pd.to_numeric(results_df.get('CVG_Cell', pd.Series(dtype=float)), errors='coerce').fillna(cg.UNREAD)
    read = results_df[cells != cg.UNREAD]
    sides = pd.to_numeric(read.get('CVG_Side', pd.Series(dtype=float)), errors='coerce')
    n_read = max(len(read), 1)
    n_build, n_cut = int((sides > 0).sum()), int((sides < 0).sum())
    n_held = int(read.get('CVG_Held', pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    armed = pd.to_numeric(results_df.get('PRG_Armed', pd.Series(dtype=float)), errors='coerce').fillna(0)

    ui.render_section_header(
        "Conviction-Value Grid",
        f"{len(read)} of {len(results_df)} names read · the 3 × 3 the two tapes place each "
        f"name in, named as an action",
        icon="grid", accent="violet",
    )
    # ── census by TONE — Pragyam's reading order: build → watched → no edge → caution → cut ──
    tones = cells.astype(int).map(lambda k: cg.TONES[k])[cells != cg.UNREAD]
    actions = read.get('CVG_Action', pd.Series("", index=read.index)).astype(str)
    kpis = []
    for tone in charts.TONE_ORDER:
        sel = (tones == tone).to_numpy()
        n = int(sel.sum())
        mix = actions[sel].value_counts()
        kpis.append({"label": f"{_TONE_GLYPH[tone]} {charts.TONE_LABEL[tone].split(' — ')[0]}",
                     "value": f"{n / n_read * 100:.0f}%",
                     "subtext": " · ".join(f"{k} {v}" for k, v in mix.items()) or "none on this bar",
                     "color_class": cg.TONE_CHIP[tone],
                     "tooltip": charts.TONE_LABEL[tone]})
    ui.render_kpi_strip(kpis, max_cols=5, key=f"{key}-tone-strip")
    ui.render_note(f"**{n_build}** names sit on the build side and **{n_cut}** on the cut side. "
                   f"**{n_held}** rows are *held* — the tape moved, the push has not confirmed — "
                   f"and **{int((armed != 0).sum())}** names sit in capitulation with value still "
                   f"cheapening — the watchlist.")

    # ── the plane: every name at its two tapes ──
    ui.render_sub_header("Conviction-value map")
    ui.render_chart_panel(charts.create_conviction_value_map(results_df), f"{key}-map",
                          units="conviction tape × value tape")
    ui.render_note("Each name sits where its two tapes put it; its **shape and colour are its grid "
                   "state**. Dotted lines are the knees (±30 conviction, ±θ value). A *hollow* "
                   "point is a held row — the tape has moved into a new cell but the push has not "
                   "yet backed it. An accent ring marks a ▲▼ or ◆ on this bar.")

    ui.render_sub_header("Census")
    n_un = int((cells == cg.UNREAD).sum())
    with ui.html_panel(f"{key}-census", context=_chart_ctx("cell counts · share of names read")):
        st.components.v1.html(_build_grid_census_html(results_df),
                              height=_census_iframe_height(n_un))
    ui.render_note("Rows are who controls (the conviction tape: UP past +30, DOWN past −30); "
                   "columns are where price stands (the value tape: cheap past −θ, rich past +θ). "
                   "A cell wears its tone: **emerald** builds — capitulation or a turn; **cyan** "
                   "accumulates a washout or a base; **amber** holds or trims — building, stalling, "
                   "paid; **rose** exits distribution; grey waits. " + cg.READ_THE_PUSH)
    ui.render_note("**Measured (v9 audit, scored without look-ahead):** across 380 instruments in "
                   "six asset classes and three eras (2006-13, 2014-19, 2020-26), *Buy · capitulation* "
                   "— sellers in control of a cheap price — was followed by gains in every era, daily "
                   "and weekly (+0.06 to +0.08σ over 10-20 bars), and kept working after 2020 when "
                   "plain oversold stopped; *Exit · distribution* was followed by underperformance in "
                   "every era; the other cells are near zero. The v8 units held and beat the seed in "
                   "Pragyam's own allocator. Crypto trends and is the stated exception.")

    # ── the watchlist: capitulations whose value has not turned ──
    wl = results_df[armed != 0].copy()
    ui.render_sub_header(f"Watchlist · capitulation, value still cheapening ({len(wl)})")
    ui.render_note("Sellers in control across the ladder at a price cheap past θ — *Buy · "
                   "capitulation* — but the fast end of value is still cheapening. The ▲ fires on "
                   "the bar value momentum turns back toward fair. Sorted by bars in the cell.")
    if len(wl):
        wl['_w'] = pd.to_numeric(wl.get('PRG_Armed_Age'), errors='coerce')
        wl = wl.sort_values(['PRG_Armed', '_w'], ascending=[False, True])
        with ui.html_panel(f"{key}-watch", context=_chart_ctx(f"{len(wl)} in capitulation")):
            st.components.v1.html(_build_narrative_table_html(wl, side='buy'),
                                  height=ui.table_iframe_height(len(wl), max_height=560))
    else:
        ui_info("No name sits in capitulation with value still cheapening on this bar.")

    # ── the lists, by what the grid says to do ──
    ui.render_sub_header("Names by action")
    t_build, t_hold, t_cut = st.tabs(["Build · Buy / Accumulate",
                                      "Hold · Hold / Wait",
                                      "Cut · Trim / Exit"])
    for tab, sel, side_key, pcol, name in ((t_build, sides > 0, 'buy', 'Priority_Long', 'build'),
                                           (t_hold, sides == 0, 'buy', 'Priority_Long', 'hold'),
                                           (t_cut, sides < 0, 'sell', 'Priority_Short', 'cut')):
        with tab:
            part = read[sel.to_numpy()] if len(read) else read
            if pcol in part.columns:
                part = part.sort_values(pcol, ascending=False, na_position='last')
            if len(part):
                with ui.html_panel(f"{key}-{name}", context=_chart_ctx(f"{len(part)} names")):
                    st.components.v1.html(_build_narrative_table_html(part, side=side_key),
                                          height=ui.table_iframe_height(len(part), max_height=900))
            else:
                ui_info("No names in these cells on this bar.")


def _render_ranking_tab(results_df, sid, study, _mv_label, _mv_kind, key: str = "rank") -> None:
    """Signal Strength — the whole cross-section, by priority (a ▲▼ today, then stretch)."""
    ui.render_section_header(
        "Signal Strength",
        "Full universe by priority — a ▲▼ on this bar first, then by stretch: the long side "
        "leads with names stretched furthest down, the short side with names stretched furthest up",
        icon="zap", accent="amber",
    )
    _n = max(len(results_df), 1)
    side = results_df['Side'] if 'Side' in results_df.columns else pd.Series(dtype=str)
    kind = results_df['Signal_Kind'] if 'Signal_Kind' in results_df.columns else pd.Series(dtype=str)
    n_long, n_short = int((side == 'Buy').sum()), int((side == 'Sell').sum())
    n_turn = int((kind == 'TURN').sum())
    stretched = pd.to_numeric(results_df.get('Signal_Score', pd.Series(dtype=float)), errors='coerce').abs() >= sv.THETA_OSC
    s1, s2, s3, s4 = st.columns(4)
    with s1: ui.render_metric_card("Events Today", f"{n_long + n_short}",
                                   f"{n_long} long · {n_short} short · {n_turn} ▲▼", "info")
    with s2: ui.render_metric_card("Stretched", f"{int(stretched.sum())}",
                                   f"{stretched.sum()/_n*100:.0f}% of names with the trace past θ", "neutral")
    with s3:
        units = pd.to_numeric(results_df.get('CVG_Units', pd.Series(dtype=float)), errors='coerce')
        ui.render_metric_card("Mean Grid Weight", _fmt_num(units.mean(), "{:.2f}u"),
                              "1u is Wait · the universe's average stance", "neutral")
    with s4:
        _r4 = (study.get("buy", "holdout") or study.get("buy", "full")) if study else None
        ui.render_metric_card("Measured Edge · Long",
                              f"{_r4.edge:+.3f}" if _r4 is not None else "—",
                              (f"{_r4.hit:.1f}% hit · {_mv_label}" if _r4 is not None
                               else "not measured on this universe"), _mv_kind)

    def _col_label(side_label, side_key):
        _p = _side_palette(side_key)
        return (f'<p style="font-family:var(--data); font-size:var(--fs-2xs); font-weight:600; '
                f'text-transform:uppercase; letter-spacing:0.1em; color:{_p["accent_light"]}; '
                f'margin:0 0 0.4rem 0;">{_p["mark"]} {side_label}</p>')

    ui.render_sub_header("Top 10 Each Side")
    ui.render_note("Highest-priority rows in the universe. Ranking reads the trace as **reversion**: "
                   "across ~15 years of five NSE universes, names stretched down (sellers in control, "
                   "priced cheap) led the next 10 bars and names stretched up lagged — so the long "
                   "side starts with the most stretched-down names. A ▲▼ on this bar ranks first.")
    top_buys = results_df.sort_values('Priority_Long', ascending=False, na_position='last').head(10)
    top_sells = results_df.sort_values('Priority_Short', ascending=False, na_position='last').head(10)
    _l, _s = st.columns(2)
    with _l:
        st.markdown(_col_label("Top 10 Long Side", "buy"), unsafe_allow_html=True)
        st.components.v1.html(_build_signal_strength_table_html(top_buys, side='buy'),
                              height=ui.table_iframe_height(len(top_buys), max_height=900))
    with _s:
        st.markdown(_col_label("Top 10 Short Side", "sell"), unsafe_allow_html=True)
        st.components.v1.html(_build_signal_strength_table_html(top_sells, side='sell'),
                              height=ui.table_iframe_height(len(top_sells), max_height=900))

    ui.render_note("Full universe ranked by long-side priority. The ranking is continuous, but "
                   "the claim is in the EVENTS: a ▲▼ fired on this bar is a signal; "
                   "the grid state beneath it is where the name stands between signals, not a "
                   "forecast.")
    _all = results_df.sort_values('Priority_Long', ascending=False, na_position='last')
    st.components.v1.html(_build_signal_strength_table_html(_all, side='buy'),
                          height=ui.table_iframe_height(len(_all), max_height=900))


def main():
    """Main app entry point with state-based flow."""
    # ── Animation-on-first-render gate ────────────────────────────────────
    # Streamlit re-mounts DOM on every rerun, which causes our entrance
    # animations (.metric-card stagger, .system-card fade, .system-spec slide)
    # to replay on every interaction — visible flicker. The CSS animations
    # are great on the FIRST encounter; we suppress them on subsequent reruns
    # so interactions feel instant. No design change — first impression is
    # preserved exactly as designed.
    is_first_render = not st.session_state.get("_first_render_done")
    if not is_first_render:
        st.markdown(
            "<style>"
            ".metric-card, .system-card, .system-spec { animation: none !important; }"
            "</style>",
            unsafe_allow_html=True,
        )
    st.session_state["_first_render_done"] = True

    # ── Session-start log (once per browser session) ──────────────────────
    # Banner-style header anchors the terminal output for grep-by-session.
    if is_first_render:
        console.header("SANKET TERMINAL — Session Start", VERSION)
        console.item("Started", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        console.item("Signal engine", f"{ENGINE_CODE} — {ENGINE_NAME} (pragati.pine v9.2)")

    # Render sidebar and get parameters + run button state
    sbs = render_sidebar()
    # Local aliases keep the main() body readable; the data flow from the sidebar is
    # name-keyed (sbs.field) rather than a positional tuple unpack.
    universe           = sbs.universe
    selected_index     = sbs.selected_index
    analysis_date      = sbs.analysis_date
    reg_len            = sbs.reg_len
    wt_n1              = sbs.wt_n1
    wt_n2              = sbs.wt_n2
    wt2_len            = sbs.wt2_len
    wt2_type           = sbs.wt2_type
    levels             = sbs.levels
    timeframe          = sbs.timeframe
    mode               = sbs.mode
    start_date         = sbs.start_date
    end_date           = sbs.end_date
    run_clicked        = sbs.run_clicked
    corr_target_ticker = sbs.corr_target_ticker
    corr_lookback      = sbs.corr_lookback
    corr_method        = sbs.corr_method
    sid                = sbs.sid          # resolved engine settings for this run

    # ── Run button click — single-pass execution ─────────────────────────
    # Previously: click → set flag → st.rerun() → run analysis → st.rerun() → render body.
    # That's THREE script executions per click, with two visible flashes between them.
    # New pattern: run analysis directly in this script run, then continue to the
    # body render below — ONE execution, ONE render frame, no inter-rerun flicker.
    if run_clicked:
        # Reset any stale display state from a prior run
        st.session_state["timeseries_done"] = False
        st.session_state["results_df"] = None
        st.session_state["corr_data"] = None
        st.session_state["run_error"] = None
        st.session_state["run_screener_flag"] = False  # legacy guard, kept for safety

        # ── Edge study — runs on EVERY run ───────────────────────────────
        # First, because the screener's cost gate and every reported verdict read from it.
        # Reuses a same-day measurement (identical inputs ⇒ identical answer), re-measures
        # once the date rolls, and never blocks the run if it fails.
        _study_slot = st.empty()
        _had_study = _edge_cache_get(_edge_key(universe, selected_index, timeframe, sid)) is not None
        study = ensure_edge_study(universe, selected_index, timeframe, sid,
                                  progress_slot=_study_slot, progress_offset=0,
                                  progress_scale=_STUDY_PROGRESS_SHARE)
        # The sidebar card was painted before this ran — repaint it so a study measured on
        # THIS click shows immediately instead of one interaction later.
        _refresh_engine_card()
        if study is None and not _had_study:
            ui_warning(
                "**Edge study could not complete.** Not enough history came back to measure "
                "expectancy on this universe (a common cause is yfinance rate-limiting a deep "
                "request from a shared cloud IP). The screen below still runs; the expectancy "
                "simply reads as not measured, and the study is retried on the next session."
            )
        # ONE progress bar for the whole click. When the study actually measured it owns the
        # head of the bar, so the analysis renders into the tail of the SAME bar; when the
        # study was reused from cache it painted nothing and the analysis owns all of it.
        _measured_now = study is not None and not _had_study
        _an_offset = _STUDY_PROGRESS_SHARE if _measured_now else 0
        _an_scale = (100 - _STUDY_PROGRESS_SHARE) if _measured_now else 100

        if mode in ("Single Date", "Pulse Narrative"):
            header_text = "Pragati Signal Screener" if mode == "Single Date" else "Pulse Narrative Analysis"
            console.header(f"SANKET TERMINAL — {header_text}", VERSION)
            console.main_header("ANALYSIS RUN START", {
                "Universe": universe, "Index": selected_index, "Timeframe": timeframe,
                "Target Date": analysis_date, "Mode": mode,
                "Measured edge": _study_state(study, "buy")[0],
            })
            results_df = run_screener_analysis(
                universe, selected_index, analysis_date,
                reg_len, wt_n1, wt_n2, levels, timeframe,
                wt2_len=wt2_len, wt2_type=wt2_type,
                external_progress_slot=_study_slot,
                progress_offset=_an_offset, progress_scale=_an_scale,
                sid=sid, study=study,
            )
            _study_slot.empty()
            if results_df is None:
                st.session_state["run_error"] = f"Failed to fetch constituents for '{selected_index}'."
            st.session_state["results_df"] = results_df
            # Store metadata so correlation analysis can reuse these results
            st.session_state["screener_meta"] = {
                "universe":      universe,
                "selected_index": selected_index,
                "analysis_date": analysis_date,
                "timeframe":     timeframe,
            }

        elif mode == "Historical Range":
            console.header("SANKET TERMINAL — Historical Signal Harvest", VERSION)
            run_timeseries_analysis(
                universe, selected_index, start_date, end_date,
                reg_len, wt_n1, wt_n2, levels, timeframe,
                wt2_len=wt2_len, wt2_type=wt2_type, sid=sid, study=study,
                external_progress_slot=_study_slot,
                progress_offset=_an_offset, progress_scale=_an_scale,
            )
            _study_slot.empty()
            # Standalone harvest — no screener follows to consume the analyzed-frame
            # cache the harvest just populated, so release it here. (In the Single-Date
            # / Correlation flows the screener consumes then clears it itself.)
            _analyzed_cache_clear()

        elif mode == "Correlation Analysis":
            # Correlation drives its own multi-phase bar internally, so hand it a clean slate
            # rather than trying to nest two offset schemes.
            _study_slot.empty()
            corr_data = run_correlation_analysis(
                universe, selected_index, corr_target_ticker,
                corr_lookback, corr_method, timeframe, analysis_date, sid=sid, study=study,
            )
            st.session_state["corr_data"] = corr_data

    # ── Mode-change cleanup ──────────────────────────────────────────────
    last_mode = st.session_state.get("_last_mode")
    if last_mode != mode:
        st.session_state["run_error"] = None
        st.session_state["_last_mode"] = mode

    # ── Landing-page gate ────────────────────────────────────────────────
    show_landing = False
    if mode in ("Single Date", "Pulse Narrative") and st.session_state["results_df"] is None:
        show_landing = True
    elif mode == "Correlation Analysis" and st.session_state.get("corr_data") is None:
        show_landing = True
    elif mode == "Historical Range" and not st.session_state.get("timeseries_done"):
        show_landing = True

    # The measured study for the current selection, if one exists (a prior run may have
    # measured it, or the disk cache may have survived). Renderers read this instead of a
    # hardcoded class constant.
    study = _edge_cache_get(_edge_key(universe, selected_index, timeframe, sid))
    _mv_label, _mv_kind, _mv_detail = _study_state(study, "buy")

    # The universe as the command bar names it — same resolution the rail
    # readout uses, so the two cannot disagree about what is being analysed.
    _cb_universe = selected_index or universe
    if universe == "ETF Index":
        _cb_universe = "NSE ETFs"
    elif universe == "Global Macro":
        _cb_universe = "Global Macro Bonds"
    elif universe == "Global Indexes" and not selected_index:
        _cb_universe = "Global Benchmark Indexes"

    # Publish what the command bar resolved, so every panel header downstream
    # names the same universe and timeframe without being handed them.
    st.session_state["active_universe"] = _cb_universe
    st.session_state["active_timeframe"] = timeframe

    if show_landing:
        # The masthead is the cold-start screen's job and only that: it is the
        # one thing on an empty page that says what the application is. Once a
        # session is loaded the command bar takes over — it carries the same
        # mark plus what the masthead cannot, namely what is being analysed and
        # how fresh it is. Two persistent headers stacked on every page is one
        # more than the screen can justify.
        ui.render_header("Sanket", f"Market Signal Screener · {ENGINE_NAME}")
        if st.session_state.get("run_error"):
            ui.render_warning_box("Run failed", str(st.session_state["run_error"]))
        render_landing_page()
        render_footer()
    else:
        # ── The command bar — the first element on every loaded page ──────
        # Reading order left to right is identity → state → trust: which
        # universe, what the screen found, and whether the expectancy behind it
        # has been measured. NOTHING renders above this bar; data-quality
        # notices hang BELOW it in the notice rail, so the thing being analysed
        # is always the first thing on screen rather than an apology about it.
        _run_stats = st.session_state.get("screener_run_stats", {}) or {}
        _cb_meta = [
            ("Timeframe", timeframe),
            ("As of", analysis_date.strftime("%d %b %Y")
             if hasattr(analysis_date, "strftime") else str(analysis_date)),
        ]
        if _run_stats.get("analyzed"):
            _cb_meta.insert(0, ("Symbols", f"{_run_stats['analyzed']} / "
                                           f"{_run_stats.get('total_in_universe', '—')}"))
        ui.render_top_bar(
            target=_cb_universe,
            status_label=_mv_label,
            status_tone=_mv_kind,
            meta_items=_cb_meta,
        )

        # ── Notice rail — everything that qualifies the reading above it ──
        _notices = []
        if st.session_state.get("run_error"):
            _notices.append({"kind": "warning", "title": "Run failed",
                             "body": html.escape(str(st.session_state["run_error"]))})
        if _run_stats.get("warming_up"):
            _notices.append({
                "kind": "info", "title": "Warming up",
                "body": f"{_run_stats['warming_up']} symbol(s) excluded — fewer than "
                        f"{sid.min_bars} bars, so the trace's histogram is not calibrated.",
            })
        if _run_stats.get("failed"):
            _notices.append({
                "kind": "warning", "title": "Incomplete fetch",
                "body": f"{_run_stats['failed']} symbol(s) failed to fetch or analyse; "
                        f"the cross-section below is the remainder.",
            })
        if study is None:
            _notices.append({
                "kind": "info", "title": "Expectancy not measured",
                "body": "No edge study exists for this selection yet. Signals still fire — "
                        "the measurement is reported, never applied.",
            })
        if _run_stats.get("drivers") is False:
            _notices.append({
                "kind": "warning", "title": "Value runs unhedged",
                "body": "The macro drivers could not be fetched, so the value ingredient is the "
                        "name's own path (the Pine's 'Macro hedge: Off') — breadth and the RV "
                        "leg without its hedge. Signals still fire; value means less.",
            })
        _paused = int(_run_stats.get("paused", 0) or 0)
        if _paused:
            _notices.append({
                "kind": "info", "title": "Signals paused on some names",
                "body": f"{_paused} symbol(s) have a calibrated histogram but a tape still warming "
                        f"— the signal set needs both tapes, so they carry a grid state (or read "
                        f"unread) but cannot fire. The deepest wait is the value ladder's parent "
                        f"rung ({sid.value_ladder_label}).",
            })
        if sid.norm_is_adapted:
            # The one indicator input on screen that is Sanket's rather than the Pine's. It
            # belongs here and not in the rail: it is a fact about THIS run, it only applies
            # on Weekly, and the notice rail is where facts about this run go.
            _notices.append({
                "kind": "info", "title": "Adapted normalization window",
                "body": f"Weekly runs a {sid.norm}-bar window, not the indicator's "
                        f"{eng.PRG_NORM}. Calibration costs two of them, so at {eng.PRG_NORM} a "
                        f"weekly symbol would need "
                        f"{eng.pg.warmup_bars(eng.pg.Params(norm=eng.PRG_NORM))} weekly bars before "
                        f"its histogram was calibrated.",
            })
        ui.render_notice_rail(_notices)

        # Body renders directly from session-state — analysis (when triggered)
        # already populated session state above in the run_clicked block.

        # Display single-date results
        if mode in ["Single Date", "Pulse Narrative"] and st.session_state["results_df"] is not None:
            results_df = st.session_state["results_df"]

            # Safety: Ensure required columns exist
            if 'SimpleName' not in results_df.columns and not results_df.empty:
                results_df['SimpleName'] = results_df['Symbol'].str.replace(".NS", "", regex=False).str.lstrip("^")
            for _col in ['BUY_Today', 'BUY_1d', 'BUY_2d', 'BUY_3d', 'BUY_5d',
                         'SELL_Today', 'SELL_1d', 'SELL_2d', 'SELL_3d', 'SELL_5d']:
                if _col not in results_df.columns:
                    results_df[_col] = "—"

            _run_stats = st.session_state.get("screener_run_stats", {})
            _n_analyzed = _run_stats.get("analyzed", len(results_df))
            _n_universe = _run_stats.get("total_in_universe", _n_analyzed)
            _n_warming  = _run_stats.get("warming_up", 0)
            _date_str   = analysis_date.strftime("%d %b %Y") if hasattr(analysis_date, "strftime") else str(analysis_date)

            if mode == "Pulse Narrative":
                tab_narrative, tab_grid, tab_strength, tab_raw = st.tabs(
                    ["Pulse Narrative Dashboard", "Grid", "Signal Strength", "System Data"])
                with tab_narrative:
                    ui.render_section_header(
                        f"Pulse Narrative — {timeframe} Universe State",
                        f"{_n_analyzed} / {_n_universe} symbols · {_date_str} · {sid.iclass} · "
                        f"full universe, every name's state",
                        icon="zap", accent="amber"
                    )
                    _n = max(len(results_df), 1)
                    _tr = pd.to_numeric(results_df.get('Signal_Score', pd.Series(dtype=float)), errors='coerce')
                    n_buy  = int((results_df['Side'] == 'Buy').sum())  if 'Side' in results_df.columns else 0
                    n_sell = int((results_df['Side'] == 'Sell').sum()) if 'Side' in results_df.columns else 0
                    _sides = pd.to_numeric(results_df.get('CVG_Side', pd.Series(dtype=float)), errors='coerce')
                    _read = (pd.to_numeric(results_df.get('CVG_Cell', pd.Series(dtype=float)), errors='coerce') != cg.UNREAD)
                    _build = (_sides[_read] > 0).mean() * 100 if _read.any() else float('nan')
                    m1, m2, m3, m4 = st.columns(4)
                    with m1: ui.render_metric_card("Universe Stretch", _fmt_num(_tr.mean(), "{:+.1f}"),
                                                   "mean trace · + stretched up, − down (θ = ±43)", "neutral")
                    with m2: ui.render_metric_card("▲ Long Events", str(n_buy),
                                                   f"{n_buy/_n*100:.0f}% of universe · ▲ capitulation or ◆ ↑",
                                                   "success" if n_buy else "neutral")
                    with m3: ui.render_metric_card("▼ Short Events", str(n_sell),
                                                   f"{n_sell/_n*100:.0f}% of universe · ▼ distribution or ◆ ↓",
                                                   "danger" if n_sell else "neutral")
                    with m4: ui.render_metric_card("Build-Side", _fmt_num(_build, "{:.0f}%"),
                                                   "names the grid places in Buy / Accumulate",
                                                   "success" if (np.isfinite(_build) and _build > 50) else "neutral")
                    buy_narr_tab, sell_narr_tab = st.tabs(["Long side", "Short side"])
                    with buy_narr_tab:
                        buy_rank_df = results_df.sort_values('Priority_Long', ascending=False, na_position='last')
                        with ui.html_panel("pn-narr-buy", context=_chart_ctx("long side")):
                            st.components.v1.html(
                                _build_narrative_table_html(buy_rank_df, side='buy'),
                                height=ui.table_iframe_height(len(buy_rank_df), max_height=1200))
                    with sell_narr_tab:
                        sell_rank_df = results_df.sort_values('Priority_Short', ascending=False, na_position='last')
                        with ui.html_panel("pn-narr-sell", context=_chart_ctx("short side")):
                            st.components.v1.html(
                                _build_narrative_table_html(sell_rank_df, side='sell'),
                                height=ui.table_iframe_height(len(sell_rank_df), max_height=1200))
                    if _n_warming:
                        ui.render_note(f"{_n_warming} symbol(s) excluded — fewer than {sid.min_bars} bars, "
                                       "so the trace's histogram is not calibrated yet.")
                with tab_grid:
                    _render_grid_tab(results_df, sid, key="pn-grid")
                with tab_strength:
                    _render_ranking_tab(results_df, sid, study, _mv_label, _mv_kind, key="pn-rank")
                with tab_raw:
                    _render_system_data_tab(results_df, analysis_date,
                                            universe=universe, selected_index=selected_index,
                                            sid=sid, study=study)
            else:
                tab_signals, tab_grid, tab_strength, tab_raw = st.tabs(
                    ["Action Dashboard", "Grid", "Signal Strength", "System Data"])
                with tab_signals:
                    timeframe_label = "This Week's" if timeframe == 'Weekly' else "Today's"
                    ui.render_section_header(
                        f"{timeframe_label} Signals",
                        f"{_n_analyzed} / {_n_universe} symbols · {timeframe} · {_date_str} · "
                        f"{ENGINE_NAME} · {sid.trigger_label} · measured: {_mv_label}",
                        icon="zap",
                        accent="amber"
                    )

                    # The events, bucketed by how long ago they fired.
                    buys_df  = results_df[results_df['BUY_5d'].astype(str)  != "—"].copy().sort_values('Priority_Long',  ascending=False, na_position='last')
                    sells_df = results_df[results_df['SELL_5d'].astype(str) != "—"].copy().sort_values('Priority_Short', ascending=False, na_position='last')
                    _age_order = _age_labels(timeframe)
                    _armed = pd.to_numeric(results_df.get('PRG_Armed', pd.Series(dtype=float)), errors='coerce').fillna(0)
                    _when = 'this week' if timeframe == 'Weekly' else 'today'

                    _fired_today_buy  = int((results_df['BUY_Today'].astype(str)  != "—").sum())
                    _fired_today_sell = int((results_df['SELL_Today'].astype(str) != "—").sum())
                    _turns_today = int((results_df.get('Signal_Kind', pd.Series(dtype=str)) == 'TURN').sum())
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    with mc1:
                        ui.render_metric_card("▲ Long Events", str(len(buys_df)),
                                              f"{_fired_today_buy} fired {_when} · last 5 bars", "success")
                    with mc2:
                        ui.render_metric_card("▼ Short Events", str(len(sells_df)),
                                              f"{_fired_today_sell} fired {_when} · last 5 bars", "danger")
                    with mc3:
                        ui.render_metric_card("▲▼ " + _when.title(), str(_turns_today),
                                              "capitulation turns · distribution", "violet")
                    with mc4:
                        ui.render_metric_card("Watchlist", str(int((_armed != 0).sum())),
                                              "in capitulation, value still cheapening", "warning")

                    if not (buys_df.empty and sells_df.empty):
                        buy_tab, sell_tab = st.tabs(["▲ Long Events by Timing", "▼ Short Events by Timing"])

                        def _render_age_table(df_, side_key):
                            _, _stats, _trend, _tcol = _bucket_signals_by_age(
                                df_, side=side_key, timeframe=timeframe)
                            _html = _build_signal_table_html(_stats, side=side_key, timeframe=timeframe)
                            _g = sum(1 for a in _age_order if _stats[a]['count'] > 0)
                            _r = sum(_stats[a]['count'] for a in _age_order)
                            st.markdown(
                                f'<div style="font-family:var(--data); font-size:var(--fs-xs); '
                                f'color:{_tcol}; padding:0.2rem 0 0.5rem 0;">{_trend}.</div>',
                                unsafe_allow_html=True,
                            )
                            st.components.v1.html(_html, height=ui.table_iframe_height(_r, extra_rows=_g * 2, max_height=760),
                                                  scrolling=True)

                        with buy_tab:
                            st.markdown(
                                '<div style="font-family:var(--data); font-size:var(--fs-xs); color:var(--ink-tertiary); '
                                'padding:0.2rem 0 0.5rem 0;"><b>▲ CAPITULATION</b> — sellers in control across the ladder '
                                'at a price cheap past θ, and value has turned back toward fair: the v9 audit\'s one event '
                                'positive in every era (about +0.05σ over 10-20 bars — a lean, not a trade; not on crypto). '
                                f'Entry is the next session\'s open; the declared hold is {sid.horizon} '
                                'bars. The Grid column says where each name already stands.</div>',
                                unsafe_allow_html=True,
                            )
                            _render_age_table(buys_df, 'buy')
                        with sell_tab:
                            st.markdown(
                                '<div style="font-family:var(--data); font-size:var(--fs-xs); color:var(--ink-tertiary); '
                                'padding:0.2rem 0 0.5rem 0;"><b>▼ DISTRIBUTION</b> — sellers have taken control across the '
                                'ladder of a price rich past θ. As a state it was followed by underperformance in every era '
                                'of the v9 audit; the entry itself is too rare to measure alone. The Edge Study in System '
                                'Data measures both on this universe.</div>',
                                unsafe_allow_html=True,
                            )
                            _render_age_table(sells_df, 'sell')
                    else:
                        ui_info(
                            f"**No events fired** for {selected_index} on {analysis_date} ({timeframe}). "
                            f"All {_n_analyzed} symbols were analyzed and none fired a ▲ or ▼ in "
                            "the last 5 bars — capitulations are rare, so quiet stretches are normal. "
                            "The Grid tab shows where every name stands, and its watchlist the names in "
                            "capitulation whose value has not yet turned."
                        )
                    if _n_warming:
                        ui.render_note(f"{_n_warming} symbol(s) excluded — fewer than {sid.min_bars} bars of history.")

                with tab_grid:
                    _render_grid_tab(results_df, sid, key="sd-grid")
                with tab_strength:
                    _render_ranking_tab(results_df, sid, study, _mv_label, _mv_kind, key="sd-rank")
                with tab_raw:
                    _render_system_data_tab(results_df, analysis_date,
                                            universe=universe, selected_index=selected_index,
                                            sid=sid, study=study)

        # ── Bulk-range dashboard (Historical Range only) ──
        # Re-renders on every Streamlit run from session-state ts_results_df,
        # so sidebar interactions don't blank the view.
        if st.session_state.get("timeseries_done") and mode == "Historical Range":
            render_timeseries_dashboard()

        # ── Correlation results ───────────────────────────────────────────
        if mode == "Correlation Analysis" and st.session_state.get("corr_data") is not None:
            render_correlation_results(st.session_state["corr_data"])

        # Always render footer
        render_footer()

#: Verdict kind (from `_verdict_kind`) -> the tone class `.rail-readout .v`
#: understands. The readout speaks long/short/caution/accent; the study speaks
#: success/danger/warning/neutral. One mapping, stated once, rather than a
#: conditional at each call site.
_VERDICT_TONE = {"success": "long", "danger": "short",
                 "warning": "caution", "neutral": ""}


def _render_engine_status_body(sid, study) -> None:
    """Paint the engine's state into the current container.

    Four rows at most, each one fact, in the rail's own readout grammar — the
    same component the session readout below it uses, so the sidebar is made of
    one kind of thing instead of a card pretending to be a rail.

    Everything this deliberately does NOT show — the per-side confidence
    intervals, the hit rate, the minimum detectable effect, the participation
    ratio, the studied date range — is in System Data ▸ Edge Study, in full,
    per era, with a glossary. A 145px column clipped to twelve characters was
    not reporting those numbers, it was hinting at them.

    Nor does it carry the engine's caveats any more. The bare-crossing warning
    is stated twice in the body already — on the Signal Reference card and in
    the SELL tab's own description — and a rail is not where a reader goes to
    be argued with. The one disclosure with nowhere else to live, the adapted
    weekly normalization window, moved to the notice rail, which is the
    component for "something about THIS run you should know".
    """
    label, kind, _detail = _study_state(study, "buy")
    tone = _VERDICT_TONE.get(kind, "")
    gate_ok = sid.cost_ok(study)

    rows = [("Verdict", label, tone)]

    # The number behind the verdict, when there is one. A verdict without its
    # magnitude is an opinion; the magnitude without the verdict is a number
    # nobody can act on. They belong adjacent.
    _r = (study.get("buy", "holdout") or study.get("buy", "full")) if study else None
    if _r is not None:
        rows.append(("Edge", f"{_r.edge:+.3f} vol", tone))

    rows.append(("Signals", sid.trigger_short, ""))
    rows.append(("Cost", f"{sid.cost_bps:.0f}bp · " + ("net +" if gate_ok else "NET NEG"),
                 "long" if gate_ok else "short"))

    ui.render_rail_readout(rows)



def _refresh_engine_card() -> None:
    """Repaint the Engine Status card after a run measured a study.

    The sidebar is rendered before the analysis executes, so a study measured during this
    click would otherwise not show until the next interaction.
    """
    slot = st.session_state.get("_engine_card_slot")
    args = st.session_state.get("_engine_card_args")
    if slot is None or args is None:
        return
    try:
        sid = _active_engine_settings()
        # `.container()` because the body is now several components rather than
        # one HTML string; writing into the slot replaces whatever it held.
        with slot.container():
            _render_engine_status_body(sid, _edge_cache_get(_edge_key(*args, sid)))
    except Exception:
        pass


def _render_engine_status_sidebar(current_universe: str, current_index,
                                  current_timeframe) -> tuple:
    """Sidebar Engine panel — visible in every mode.

    A status line, and now actually one: the verdict for the universe on screen (or an honest
    "not measured yet"), the number behind it, what fires, and what it costs. Four rows in the
    rail's own readout grammar. The per-era breakdown, both sides' intervals, the hit rate and
    the power behind them live in System Data ▸ Edge Study, which is where a reader who
    wants them is already going.

    There are no controls here. Parameters are all the indicator's own defaults, so
    a slider would only invite fitting them to whatever universe is on screen — the exact thing
    that would destroy the credibility of the measurement above it. And the edge study is not
    opt-in: it runs on every run (see :func:`ensure_edge_study`), so there is nothing to tick.

    Caller must be inside a ``with st.sidebar:`` context. Returns the resolved
    :class:`eng.EngineSettings` and stashes it in session state so renderers that do not take it as
    an argument can read it back.
    """
    st.markdown('<div class="sidebar-title">Engine</div>', unsafe_allow_html=True)

    sid = _engine_settings(current_universe, current_index, current_timeframe)
    st.session_state["engine_settings"] = sid

    study = _edge_cache_get(_edge_key(current_universe, current_index, current_timeframe, sid))
    # Painted into a placeholder so it can be repainted after a run measures a study — the
    # sidebar renders BEFORE the analysis executes (single-pass render), so without this the
    # card would show "not measured" for one extra interaction after you measured.
    _slot = st.empty()
    with _slot.container():
        _render_engine_status_body(sid, study)
    st.session_state["_engine_card_slot"] = _slot
    st.session_state["_engine_card_args"] = (current_universe, current_index, current_timeframe)

    return sid


if __name__ == "__main__":
    main()
