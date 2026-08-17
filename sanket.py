"""
Sanket - Market Signal Screener | A Pragyam Product Family Member
Close-Location Reversal (CLR) · Quantitative Signal Screener Terminal

Engine: CLOSE-LOCATION REVERSAL (CLR), ported from sb_v8.pine — the z-score of where price
closes inside its own bar range.
ONE screening condition, two events: a weak close (green triangle) is the BUY, a strong
close (yellow diamond) is the SELL. The system's universe selector drives the indicator's
instrument-class input, so the measured out-of-sample expectancy shown always matches the
asset class on screen. See engine.py, sb_v8.pine, and ARCHITECTURE.md.
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
import warnings
import logging
import time
from dataclasses import dataclass
from typing import Optional
from nsepython import nse_get_advances_declines
from logger import console

# UI — Obsidian Quant Terminal System
from ui.theme import inject_css, apply_chart_theme, progress_bar
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
    page_icon="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iI0Q0QTg1MyIgc3Ryb2tlLXdpZHRoPSIyIi8+PHBhdGggZD0iTTggMTRsMy01IDIgMyAzLTQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iI0Q0QTg1MyIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz48L3N2Zz4=",
    layout="wide",
    initial_sidebar_state="expanded",
)

VERSION = "v6.4.0"

# ── Engine identity ───────────────────────────────────────────────────────────
# Named for what it measures. The source indicator (sb_v8.pine) titled itself
# "SB v8 — CLOSE-LOCATION REVERSAL"; the "SB v8" half was a family tag from a lineage of
# session-breadth indicators whose premise this engine refutes, so only the descriptive half
# carries over. Defined here so the name appears in exactly one place.
ENGINE_NAME = "Close-Location Reversal"
ENGINE_CODE = "CLR"

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
# frozenset(stock_list).  The registry is always populated with _MAX_DAYS_BACK
# days of history so every mode (screener, intelligence, correlation) can slice
# what it needs without an extra round-trip.
#
# Two-tier caching:
#   L1 — session-state registry (per-user, sub-millisecond lookup)
#   L2 — @st.cache_data on fetch_batch_data (cross-user, process-level, 5 min TTL)
#   L3 — yfinance network fetch (slow path, only on true misses)
# ══════════════════════════════════════════════════════════════════════════════

_REGISTRY_KEY  = "data_registry"
_MAX_DAYS_BACK = 900  # fetch the maximum once; all modes slice what they need.
# 900 calendar days ≈ ~620 trading days, and fetch_batch_data pads a further 365 calendar
# days on top (≈ 870 trading bars). The CLR z-score needs a full 252-bar lookback before
# it can signal (engine.min_bars_for), so this leaves ~600 signal-bearing daily dates —
# enough for the live cross-section and for a Historical Range harvest over the same pool.
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


def _registry_get(stock_list: list, end_date: datetime.date):
    """Return cached data_dict if still fresh for this universe+date, else None.

    On a hit, the key is moved to the most-recently-used position so the LRU
    eviction in _registry_put drops genuinely-cold universes, not just oldest-stored.
    """
    reg   = st.session_state.get(_REGISTRY_KEY, {})
    key   = frozenset(stock_list)
    entry = reg.get(key)
    if entry is None or entry["end_date"] != end_date:
        return None
    age = (datetime.datetime.now(_IST) - entry["fetched_at"]).total_seconds()
    if age > _registry_ttl_seconds():
        return None
    # Mark as recently used (dict preserves insertion order → re-insert = move to end).
    reg[key] = reg.pop(key)
    return entry["data"]


def _registry_put(stock_list: list, end_date: datetime.date, data_dict: dict):
    """Store data_dict in the session-state registry under frozenset(stock_list).

    DataFrames are stored as copies so downstream mutation (adding indicator
    columns) never corrupts the cached source data. Bounded LRU: when the registry
    exceeds _REGISTRY_MAX_ENTRIES, the least-recently-used universes are evicted so
    memory can't grow without limit across index switches / re-fetches.
    """
    if _REGISTRY_KEY not in st.session_state:
        st.session_state[_REGISTRY_KEY] = {}
    reg = st.session_state[_REGISTRY_KEY]
    key = frozenset(stock_list)
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
    entry screeners; 'sbv8'/'clr1' (v6.0/v6.1) = close-location reversal, the only screening
    condition.

    ``sb_params`` = (z_look, thr, horizon). These are baked into the frame (buy_cond /
    sell_cond / the hold window all depend on them), so a threshold change in the sidebar
    must miss the cache rather than serve stale conditions.
    """
    return ("clr1", str(timeframe), int(reg_len), int(wt_n1), int(wt_n2),
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


def get_universe_data(stock_list: list, end_date: datetime.date = None):
    """Fetch OHLCV data for a universe, checking the session-state registry first.

    Always fetches _MAX_DAYS_BACK days so the screener, the range harvest, and
    correlation can all slice from the same pool without re-fetching.  Correlation callers
    should pass only the universe symbols here, then supplement the returned dict
    with a single-ticker fetch for the target asset if it is missing.

    Returns: (data_dict, message_str) — same contract as fetch_batch_data.
    """
    if end_date is None:
        end_date = _today_ist()

    cached = _registry_get(stock_list, end_date)
    if cached is not None:
        console.detail(
            f"Data registry HIT — {len(cached)} symbols available "
            f"(requested {len(stock_list)}, end_date={end_date})"
        )
        return cached, f"✓ {len(cached)} symbols (session registry)"

    console.detail(
        f"Data registry MISS — fetching {len(stock_list)} symbols "
        f"from yfinance (end_date={end_date}, days_back={_MAX_DAYS_BACK})"
    )
    data_dict, msg = fetch_batch_data(
        stock_list, end_date=end_date, days_back=_MAX_DAYS_BACK
    )
    if data_dict:
        _registry_put(stock_list, end_date, data_dict)
    return data_dict, msg

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
# Engine parameter resolution — the settings the source indicator exposes as inputs.
# Every default is a measured plateau (see engine.py), not a fitted value. The z-score
# lookback follows the timeframe. `iclass` is a DISPLAY LABEL only: it selects which of the
# source study's published rows to show as a comparison beside the measurement that
# `edge.py` makes on the user's own universe. Nothing computes from it.
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CLRSettings:
    """One run's CLR configuration."""
    z_look:   int
    thr:      float
    horizon:  int
    cost_bps: float
    iclass:   str        # reference-row label, not an input to anything

    @property
    def params_sig(self) -> tuple:
        """The subset that changes a per-symbol analyzed frame (see _analysis_params_sig)."""
        return (int(self.z_look), float(self.thr), int(self.horizon))

    @property
    def study_sig(self) -> tuple:
        """Identity of an edge study: the parameters it was measured at."""
        return (int(self.z_look), float(self.thr), int(self.horizon))

    # ── The SOURCE STUDY's published numbers for the nearest class. Reference only. ──
    @property
    def prior_edge(self) -> float:
        return eng.class_edge(self.iclass)

    @property
    def prior_hit(self) -> float:
        return eng.class_hit(self.iclass)

    @property
    def prior_established(self) -> bool:
        return eng.is_established(self.iclass)

    @property
    def min_bars(self) -> int:
        return eng.min_bars_for(self.z_look)

    def cost_ok(self, study=None) -> bool:
        """Cost gate — measured from `study` when one exists, else the pooled prior."""
        return eng.cost_ok(self.cost_bps, study)

    def cost_basis(self, study=None) -> str:
        return eng.cost_basis(study)


def _clr_settings(universe, selected_index, timeframe, overrides=None) -> CLRSettings:
    """Resolve the active CLR settings for a (universe, timeframe) selection.

    ``overrides`` is the sidebar dict ({thr, horizon, cost_bps}); anything absent falls
    back to the measured default.
    """
    o = overrides or {}
    return CLRSettings(
        z_look   = eng.z_look_for(timeframe),
        thr      = float(o.get("thr", eng.CLR_THRESHOLD)),
        horizon  = int(o.get("horizon", eng.CLR_HORIZON)),
        cost_bps = float(o.get("cost_bps", eng.CLR_COST_BPS)),
        iclass   = eng.instrument_class(universe, selected_index),
    )


def _active_clr_settings() -> CLRSettings:
    """The settings the last run resolved, for renderers that don't take them as args."""
    clr = st.session_state.get("clr_settings")
    if isinstance(clr, CLRSettings):
        return clr
    return _clr_settings(None, None, "Daily")


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
#   1. LEAN     the study computes the close-location z and forward returns ONLY. No volume
#               profile (a Python double loop, the app's slowest path), no regime engine, no
#               order flow. The study does not need them.
#   2. STREAMING symbols are fetched and reduced in chunks; each chunk's frames are released
#               before the next is fetched. What accumulates is event tuples at a ~9% fire
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


def _edge_key(universe, selected_index, timeframe, clr: CLRSettings) -> str:
    """Cache identity for a study: universe + timeframe + the parameters it was measured at."""
    parts = [str(universe), str(selected_index), str(timeframe),
             f"z{clr.z_look}", f"t{clr.thr:g}", f"h{clr.horizon}"]
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


def _active_edge_study(universe=None, selected_index=None, timeframe=None, clr=None):
    """The study matching the current selection, or None if it has not been measured."""
    if clr is None:
        clr = _active_clr_settings()
    if universe is None:
        meta = st.session_state.get("screener_meta") or {}
        universe = meta.get("universe")
        selected_index = meta.get("selected_index")
        timeframe = meta.get("timeframe", "Daily")
    return _edge_cache_get(_edge_key(universe, selected_index, timeframe, clr))


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
                "tick “Measure edge” in the sidebar")
    return study.verdict(side)


def _study_summary_line(study, side: str = "buy") -> str:
    """One-line measured read for a side: 'edge [CI] · n_eff · MDE', or a not-measured note."""
    if study is None:
        return "not measured on this universe"
    r = study.get(side, "holdout") or study.get(side, "full")
    if r is None:
        return "no events measured for this side"
    return (f"{r.edge:+.3f} [{r.ci_lo:+.3f},{r.ci_hi:+.3f}] vol · {r.hit:.1f}% hit · "
            f"n_eff {r.n_eff:.0f} · resolves ≥{r.mde:.3f}")


def _render_edge_study_panel(clr: CLRSettings, study) -> None:
    """Full Edge Study readout — the numbers behind the verdict, per side and per era."""
    ui.render_section_header(
        "Edge Study",
        "Measured out-of-sample expectancy for the symbols on screen · re-measured daily",
        icon="activity", accent="violet",
    )
    if study is None:
        ui.render_interpretation_card(
            "Not measured on this universe",
            "The engine is a fixed, pre-declared rule; whether it carries an edge on THESE symbols "
            f"is a separate empirical question, and the app measures it on every run — an event "
            f"study over ~{_STUDY_YEARS} years with each instrument's own drift removed within "
            "era, vol-normalised, block-bootstrapped over dates. This one did not complete: "
            "either the cross-section was too thin to measure, or the deep history request came "
            "back short (yfinance rate-limits deep requests from shared cloud IPs). It is retried "
            "on the next session.",
            "neutral",
        )
        return

    rows = []
    for side, mark in (("buy", "▲ BUY"), ("sell", "◆ SELL")):
        for era in ("discovery", "holdout", "full"):
            r = study.get(side, era)
            if r is None:
                continue
            rows.append({
                "Side": mark, "Era": era.title(),
                "Edge (vol)": r.edge, "CI low": r.ci_lo, "CI high": r.ci_hi,
                "Net": r.net, "Hit %": r.hit,
                "Events": r.n_events, "Dates": r.n_dates,
                "n_eff": r.n_eff, "Resolves ≥": r.mde,
                "Significant": "yes" if r.significant else ("ANTI" if r.anti else "no"),
            })
    if not rows:
        st.info("The study ran but no events fired in the measured history.")
        return

    v_buy, v_sell = study.verdict("buy"), study.verdict("sell")
    m1, m2, m3, m4 = st.columns(4)
    with m1: ui.render_metric_card("▲ BUY", v_buy[0], _study_summary_line(study, "buy"),
                                   _verdict_kind(v_buy[0]))
    with m2: ui.render_metric_card("◆ SELL", v_sell[0], _study_summary_line(study, "sell"),
                                   _verdict_kind(v_sell[0]))
    with m3: ui.render_metric_card("Independence", f"{study.part_ratio:.1f}",
                                   f"of {study.n_symbols_studied} names studied", "info")
    with m4: ui.render_metric_card("Fire Rate", f"{study.fire_rate*100:.2f}%",
                                   "of bars · source study ~9.3%", "info")

    st.dataframe(
        pd.DataFrame(rows), width='stretch', hide_index=True,
        column_config={
            "Edge (vol)": st.column_config.NumberColumn(
                help="Mean drift-free, vol-normalised return following an event. GROSS.",
                format="%+.4f"),
            "CI low": st.column_config.NumberColumn(
                help="Block-bootstrap 95% lower bound. An edge is claimed only when this is > 0.",
                format="%+.4f"),
            "CI high": st.column_config.NumberColumn(format="%+.4f"),
            "Net": st.column_config.NumberColumn(
                help=f"Edge minus the cost charge at {clr.cost_bps:.1f} bp, converted into the "
                     "same vol units using each instrument's own h-bar sigma.",
                format="%+.4f"),
            "Hit %": st.column_config.NumberColumn(
                help="Share of events where the signal beat that symbol's own drift.",
                format="%.1f"),
            "n_eff": st.column_config.NumberColumn(
                help="Independent observations = (dates / horizon) × participation ratio. "
                     "Not the event count — overlapping returns and a correlated cross-section "
                     "both reduce it.",
                format="%.0f"),
            "Resolves ≥": st.column_config.NumberColumn(
                help="Minimum detectable effect at this power (1.96·σ/√n_eff). A 'no edge' "
                     "verdict only means anything when this is smaller than the effect you "
                     "would care about.",
                format="%.4f"),
        },
    )

    _pe, _ph, _pest = study.prior()
    st.markdown(
        f'<div style="font-family:var(--data); font-size:0.66rem; color:var(--ink-tertiary); '
        f'padding:0.7rem 0 0.1rem 0; line-height:1.6;">'
        f'<b style="color:var(--ink-secondary);">Method.</b> Event study at the pre-declared '
        f'parameters (±{study.thr:.1f}σ, {study.z_look}-bar lookback, {study.horizon}-bar hold, '
        f'entry the bar after the signal). Each instrument\'s own mean forward return is removed '
        f'<i>within era</i>, so a rising market cannot read as edge; the residual is divided by '
        f'that instrument\'s own sigma so asset classes are comparable. Confidence intervals come '
        f'from a block bootstrap over <i>dates</i> — blocks absorb the overlap between '
        f'{study.horizon}-bar forward returns, whole dates absorb the cross-sectional '
        f'correlation. Parameters are never tuned here: this measures a fixed rule, it does not '
        f'search for a better one.<br><br>'
        f'<b style="color:var(--ink-secondary);">Reference prior.</b> The source study measured '
        f'<b>{html.escape(study.iclass)}</b> — the nearest asset class it covered — at '
        f'<b>{_pe:+.3f}</b> vol, {_ph:.1f}% hit'
        f'{" (established)" if _pest else " (not established)"}, on its own 39 instruments over '
        f'1993–2026. Shown only so the two can be compared. Nothing in this app computes from it.'
        f'</div>',
        unsafe_allow_html=True,
    )


def ensure_edge_study(universe, selected_index, timeframe, clr,
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
    key = _edge_key(universe, selected_index, timeframe, clr)
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

    study = run_edge_study(universe, selected_index, timeframe, clr,
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


def _fetch_study_chunk(symbols: list, start, end):
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


def run_edge_study(universe, selected_index, timeframe, clr: CLRSettings,
                   progress_slot=None, progress_offset=0, progress_scale=100):
    """Measure CLR's out-of-sample expectancy on this universe. Returns an EdgeStudy.

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
    console.item("Parameters", f"z_look {clr.z_look} · ±{clr.thr:.1f}σ · hold {clr.horizon} · "
                               f"{clr.cost_bps:.1f}bp")

    _p(3, "Measuring Edge", f"{len(symbols)} symbols · ~{_STUDY_YEARS}y")

    events, baselines, ret_cols, bar_counts = [], {}, {}, []
    n_failed_chunks = 0
    chunks = [symbols[i:i + _STUDY_CHUNK] for i in range(0, len(symbols), _STUDY_CHUNK)]

    for ci, chunk in enumerate(chunks):
        _p(3 + (ci / max(len(chunks), 1)) * 82, "Measuring Edge",
           f"chunk {ci + 1}/{len(chunks)} · {len(baselines)} symbols reduced")
        data = _fetch_study_chunk(chunk, start, end)
        if not data:
            n_failed_chunks += 1
            continue
        for tkr, f in data.items():
            try:
                if timeframe == "Weekly":
                    f = resample_to_weekly(f)
                if len(f) < clr.min_bars + clr.horizon + 2:
                    continue
                ev = edge.symbol_events(f["Close"], f["High"], f["Low"],
                                        clr.z_look, clr.thr, clr.horizon)
                base = edge.symbol_baseline(f["Close"], clr.horizon)
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
        iclass=clr.iclass, z_look=clr.z_look, thr=clr.thr, horizon=clr.horizon,
        cost_bps=clr.cost_bps,
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

    for side in ("buy", "sell"):
        lbl, _kind, detail = study.verdict(side)
        console.item(f"{side.upper()} verdict", f"{lbl} — {detail}")
    console.item("Coverage", f"{study.n_symbols_studied} symbols · {study.start} to "
                             f"{study.end} · participation ratio {study.part_ratio:.1f}")
    console.item("Fire rate", f"{study.fire_rate*100:.2f}% of bars "
                              f"(source study measured ~9.3%)")
    console.end_phase("EDGE STUDY")
    _p(100, "Edge Measured", f"{study.n_symbols_studied} symbols")
    return study


# ══════════════════════════════════════════════════════════════════════════════
# INITIALIZE UI
# ══════════════════════════════════════════════════════════════════════════════
inject_css()
# (The old theme-toggle component was removed: it rendered inside a 0-height
# component iframe — invisible, unstyled, and its JS set data-theme on the
# IFRAME's document, not the app's, so it never actually switched themes.
# The app is dark-theme-only; theme.css's [data-theme="light"] rules are
# retained but currently unreachable.)

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
            # every rolling window (including the close-location z) by a bar.
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

    Per-bar history columns (Z_Hist / Close_Hist) hold Python lists — they exist so the UI
    can report the z at the bar a signal fired, and would serialise as list-reprs. Dropped
    from the export; the per-age BUY_*/SELL_* columns carry the same information legibly.
    """
    output = io.BytesIO()
    _drop = [c for c in ('Z_Hist', 'Close_Hist') if c in df.columns]
    if _drop:
        df = df.drop(columns=_drop)
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sanket_Quant_Data')
        
        # Add Legend for user clarity. THE SIGNAL block comes first, then everything
        # that is descriptive context — the distinction matters more than the ordering.
        legend_data = {
            "Column Identifier": [
                "— THE SIGNAL (CLR) —",
                "CLR_CLV",
                "CLR_Z",
                "CLR_Z_Cap",
                "Signal / Fade_Score / CLR_Score",
                "buy_cond / BUY_*",
                "sell_cond / SELL_*",
                "Side",
                "Conviction",
                "CLR_State",
                "CLR_Hold_Dir / CLR_Hold_Age",
                "CLR_Rank_Pct",
                "Priority_Long / Priority_Short",
                "Signal_Reason",
                "— CONTEXT ONLY (never a signal input; none predicts outcome out of sample) —",
                "Zone / Condition",
                "Bar_Delta",
                "CVD",
                "CVD_Slope",
                "Delta_Z",
                "Abs_Strength",
                "Buy_Share",
                "Absorption_Score",
                "Regime / Regime_Confidence",
                "Vol_Regime",
                "Change_Point",
                "Ret_1b / Ret_5b / Ret_10b / Ret_21b",
            ],
            "Metric Description": [
                "",
                "Close location in [-1, +1]: ((C-L) - (H-C)) / (H-L). -1 = closed on the low, +1 = on the high.",
                "THE MEASURE. Z-score of CLR_CLV over the trailing lookback (252 daily / 52 weekly bars). Population stdev, matching Pine ta.stdev. Suppressed when the window's CLV sigma collapses below 0.30 — those bars produce meaningless 15-sigma readings, not extreme closes.",
                "Largest |z| this bar's window could arithmetically produce, (1 -/+ mean)/sigma. Because CLR_CLV is bounded in [-1,+1] the ceiling is ~1.9, so |z| never approaches 3. Conviction is scaled against this.",
                "Fade score = -CLR_Z. Positive = bullish. The sign flip IS the finding: a strong close predicts weakness.",
                "BUY event (green triangle): CLR_Z below -threshold, a weak close to fade up. BUY_Today/_1d/_2d/_3d/_5d mark the signal's age.",
                "SELL event (yellow diamond): CLR_Z above +threshold. NOTE: this side did NOT confirm out of sample (holdout +0.0094, CI [-0.030, +0.052]).",
                "Buy / Sell / '-' — only a fired event is actionable; sub-threshold rows are context.",
                "How far |z| sits between the firing threshold and CLR_Z_Cap, x the cost gate, in [0,1]. Cap-relative, so two bars at the same |z| can differ (Spearman vs |z| = 0.93; under the old |z|/3 scale it was exactly 1.00, i.e. pure duplication). A DESCRIPTION of the close, not a validated forecast. Not a probability.",
                "WARMING UP (no full lookback yet) / DEGENERATE (CLV sigma collapsed, z suppressed) / BUY / SELL / NEUTRAL for this bar.",
                "Hold window: direction (+1 buy, -1 sell, 0 none) and bars elapsed since it opened. Edge lives at 5-10 bars. Measured on Nifty 50: a day-0 signal did NOT outperform a day-3 one out of sample — age is not a quality grade.",
                "Cross-sectional fade-score percentile within the universe on this date.",
                "Fade score x 100 and its negation — the ranking keys the UI tables sort on.",
                "Plain-language read of the row: which event (if any), the z that produced it, and any scope caveat.",
                "",
                "Where cumulative delta sits vs its 20-bar mean: Accumulation(+) / Distribution(+) / Neutral. Measured: no out-of-sample expectancy — its one holdout-significant result reversed the sign it had in discovery.",
                "Inferred per-bar buy-sell volume delta (OHLC close-location proxy).",
                "Cumulative volume delta (running sum of Bar_Delta).",
                "3-bar change in CVD — flow building (+) or draining (-). Scales with the symbol's absolute volume, so it is NOT comparable across names; z-score it within symbol first. Measured: no out-of-sample expectancy.",
                "Signed z-score of Bar_Delta vs its 20-bar distribution. Volume-weighted, so distinct from CLR_Z.",
                "Absorption strength: |Bar_Delta| / its 20-bar average.",
                "Rolling 20-bar inferred buy share in [0,1] (0.5 = balanced) — volume-normalized, cross-sectionally comparable.",
                "Absorption context in [0,1]: high delta soaked by a small range; >0.25 approximates inferred_delta.pine rawAbsorb. Only 1.6% of fires reach 0.25 (median 0.003), so it reads ~0 for almost every signal. Measured: no out-of-sample expectancy.",
                "HMM regime label and the probability of the detected state. Per-name RISK CONTEXT.",
                "Volatility regime (LOW/NORMAL/HIGH/EXTREME) via GARCH. Risk context.",
                "Structural change point (CUSUM) identifying regime shifts. Risk context.",
                "Forward returns at the CLR horizons (Historical Range mode only). LABELS for evaluation — never inputs.",
            ]
        }
        pd.DataFrame(legend_data).to_excel(writer, index=False, sheet_name='Legend')
        
    return output.getvalue()

# ══════════════════════════════════════════════════════════════════════════════
# SHARED MATH HELPERS  (SMA + True Range — the only primitives the engine needs)
# ──────────────────────────────────────────────────────────────────────────────
#  The WRCI-era MA library (EMA/HMA/WMA/VWMA/ALMA/RMA, f_smooth, linreg, RSI) and
#  the Ehlers AutoTune filter were removed with the WRCI engine. CLR needs neither:
#  the signal is a rolling mean/stdev of the close location (engine.add_sb_features).
#  What remains here serves the descriptive order-flow context only.
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
                      clr=None):
    """Per-symbol feature engine — the CLR close-location signal plus order-flow context.

    The SIGNAL is CLR (see engine.py / sb_v8.pine): the z-score of where price closes
    inside its own bar range. It is attached here via ``eng.add_clr_features`` — one
    screening condition producing two events, a BUY on a weak close (``buy_cond``, the
    green triangle) and a SELL on a strong close (``sell_cond``, the yellow diamond).
    The cross-section is ranked later by ``eng.compute_ranking`` on the fade score.

    Everything else written here is DESCRIPTIVE CONTEXT, never a signal input: the inferred
    delta / CVD / volume profile (OHLC proxies, validated to add no cross-sectional edge),
    the MA alignment count, and the F1/F2 features the regime engine consumes.

    ``clr`` is the run's :class:`CLRSettings`; ``None`` falls back to the measured defaults.
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

    # ── THE SCREENING CONDITION — CLR close-location reversal (engine.py) ────
    # The only signal in the system. Writes CLR_CLV / CLR_Z / Fade_Score plus the two
    # plotted events (buy_cond = green triangle, sell_cond = yellow diamond) and the
    # hold window. Cross-sectional ranking happens later, once the universe is assembled.
    _sb = clr if clr is not None else _clr_settings(None, None, "Daily")
    df = eng.add_clr_features(df, z_look=_sb.z_look, thr=_sb.thr, horizon=_sb.horizon)

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
    NOT enter the CLR signal or its conviction, which is a function of the
    close-location z-score, the instrument class's measured expectancy, and the cost gate.
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
    """Return the CLR signal type for a single bar row (pandas Series).

    A fired event wins; otherwise the row falls back to its flow zone (context only).
    Matches the vectorised np.select in the harvest path.
    """
    if row.get('buy_cond'):   return "BUY"
    if row.get('sell_cond'):  return "SELL"
    cond = row.get('Condition', 'Neutral')
    return cond if cond != 'Neutral' else '-'

# ══════════════════════════════════════════════════════════════════════════════
# DATA HANDLING & UTILITIES
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

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


def render_landing_page():
    """Render landing page with system overview."""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='system-card portfolio'>
            <h3>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                CLOSE-LOCATION REVERSAL
            </h3>
            <p>One screening condition: where price closes inside its own bar range, z-scored over a trailing year. The sign is the finding — a <strong>strong close predicts weakness</strong>, so the fade of a weak close is the buy.</p>
            <div class='spec'>
                <span>Measure:</span> ((C−L) − (H−C)) / (H−L), z over 252 bars<br>
                <span>Discovery:</span> IC −0.0634, z −7.96, p_bonf 1.2e-13<br>
                <span>Holdout:</span> confirmed at h=1 and h=5 (2014–2026)<br>
                <span>Sorting:</span> Rank by fade score (−z)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='system-card regime'>
            <h3>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>
                TWO EVENTS
            </h3>
            <p><span style="color:#00E676;">▲ BUY</span> on a weak close (z &lt; −1.5σ) — holdout-confirmed in both eras. <span style="color:#FFA726;">◆ SELL</span> on a strong close (z &gt; +1.5σ) — the Pine calls this side CAUTION: it did not confirm out of sample.</p>
            <div class='spec'>
                <span>Horizon:</span> 5–10 trading days · no intraday edge<br>
                <span>Entry:</span> next session's open after the signal bar<br>
                <span>Why events:</span> a continuous position costs 12%/yr and nets −0.48 Sharpe<br>
                <span>Conviction:</span> |z| within its attainable range × cost gate
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='system-card strategies'>
            <h3>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
                MEASURED, NOT INHERITED
            </h3>
            <p>Whether the rule carries an edge is a question about <strong>your symbols</strong>, so the Edge Study measures it on them — nothing about expectancy is hardcoded. Until you measure, the app says "not measured" rather than quoting a class average.</p>
            <div class='spec'>
                <span>Method:</span> event study · drift removed within era · vol-normalised<br>
                <span>Intervals:</span> block bootstrap over dates (overlap + correlation)<br>
                <span>Power:</span> effective sample size + minimum detectable effect stated<br>
                <span>Reported, not applied:</span> a "no edge" verdict filters nothing
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class='landing-prompt'>
        <h4>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>
            AWAITING ANALYSIS PARAMETERS
        </h4>
        <p>Configure via the <strong>Sidebar</strong>: select <strong>Universe</strong>, <strong>Timeframe</strong>, <strong>Analysis Mode</strong>, and any mode-specific settings.<br>
           Click the <strong>RUN</strong> button — its label adapts to the active mode (Screener · Pulse · Harvest · Correlation).<br>
           <span style="color:var(--ink-secondary); font-size:0.85em; margin-top:0.5rem; display:inline-block;">System will z-score each symbol's close location · fire BUY / SELL past ±1.5σ · rank the cross-section by fade score · and report what the edge measures on your universe</span></p>
    </div>
    """, unsafe_allow_html=True)


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
    clr: "CLRSettings"     # resolved engine config for this run


def render_sidebar() -> SidebarState:
    with st.sidebar:
        # Centered Masthead
        st.markdown("""
        <div style="text-align:center; padding:0.75rem 0 1.5rem 0;">
            <div style="font-family:var(--display); font-size:1.5rem; font-weight:800; color:var(--amber); letter-spacing:-0.02em;">SANKET</div>
            <div style="font-family:var(--data); color:var(--ink-tertiary); font-size:0.65rem; margin-top:0.2rem; letter-spacing:0.08em; text-transform:uppercase;">संकेत | Signal Screener</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Analysis Depth
        st.markdown('<div class="sidebar-title">Analysis Depth</div>', unsafe_allow_html=True)
        timeframe = st.selectbox("Timeframe", TIMEFRAME_OPTIONS, key="sb_timeframe", label_visibility="collapsed")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

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

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

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
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
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
                    '<div style="font-family:var(--data); font-size:0.65rem; '
                    'color:var(--rose); padding:0.4rem 0 0.2rem 0; line-height:1.4;">'
                    '⚠ End date must be after start date.</div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

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

        # Engine Status panel — rendered in every mode. Surfaces the CLR engine, the
        # instrument class derived from the universe above, that class's measured
        # out-of-sample expectancy, and the four Pine parameters. Returns the resolved
        # CLRSettings for this run.
        clr = _render_engine_status_sidebar(universe, selected_index, timeframe)

        # System Spec Card — always rendered as the LAST block in the sidebar.
        try:
            if universe == "India Indexes" and selected_index:
                universe_display = selected_index
            elif universe == "Global Indexes":
                universe_display = "Global Benchmark Indexes"
            elif universe == "US Indexes" and selected_index:
                universe_display = selected_index
            elif universe == "Commodities" and selected_index:
                universe_display = selected_index
            elif universe == "Currency" and selected_index:
                universe_display = selected_index
            elif universe == "ETF Index":
                universe_display = "NSE ETFs"
            elif universe == "Global Macro":
                universe_display = "Global Macro Bonds"
            else:
                universe_display = universe
        except Exception:
            universe_display = universe

        spec_html = f"""
        <div class="system-spec">
            <div class="spec-row"><span class="spec-label">Version</span><span class="spec-value">{VERSION}</span></div>
            <div class="spec-row"><span class="spec-label">Universe</span><span class="spec-value" style="font-size:0.7rem;">{universe_display}</span></div>
            <div class="spec-row"><span class="spec-label">Timeframe</span><span class="spec-value">{timeframe}</span></div>
            <div class="spec-row"><span class="spec-label">Mode</span><span class="spec-value" style="font-size:0.7rem;">{analysis_mode}</span></div>
            <div class="spec-row"><span class="spec-label">Asset Class</span><span class="spec-value" style="font-size:0.7rem;">{clr.iclass}</span></div>
        """
        if analysis_mode == "Correlation Analysis":
            spec_html += f'<div class="spec-row"><span class="spec-label">Target</span><span class="spec-value" style="font-size:0.7rem;">{target_selected}</span></div>'
        spec_html += "</div>"

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(spec_html, unsafe_allow_html=True)

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
            clr=clr,
        )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SCREENER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_screener_analysis(universe, selected_index, analysis_date, reg_len, wt_n1, wt_n2, levels, timeframe, show_progress=True, external_progress_slot=None, progress_offset=0, progress_scale=100, wt2_len=20, wt2_type="ALMA", clr=None, study=None):
    """Execute the CLR screen and return the ranked cross-section.

    Fetches market data for the universe, computes the per-symbol close-location z-score
    (plus order-flow / regime context), then ranks the whole cross-section by the fade
    score (engine.compute_ranking).

    Args:
        clr: the run's :class:`CLRSettings`; ``None`` resolves defaults for this universe.
        study: an optional :class:`edge.EdgeStudy` measured on this universe. Used for the
            cost gate and the per-row read; never to filter or scale a signal.
        external_progress_slot: Optional Streamlit container for external progress tracking (e.g., from correlation analysis)
        progress_offset: Starting percentage for external progress tracking (default 0)
        progress_scale: Scale factor for progress percentage within external slot (default 100 = full)

    Returns: DataFrame with signals ranked by fade score, or None on error.
    """
    obLevel1, obLevel2, osLevel1, osLevel2 = levels
    if clr is None:
        clr = _clr_settings(universe, selected_index, timeframe)
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
        st.error(msg)
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
    data_dict, fetch_msg = get_universe_data(stock_list, end_date=end_date)

    if not data_dict:
        console.error(fetch_msg)
        st.error(fetch_msg)
        return None

    console.success(f"Successfully downloaded data for {len(data_dict)} stocks")

    console.end_phase("DATA ACQUISITION")

    console.start_phase("SIGNAL SCREEN", 2, 2)

    console.section("Engine Parameters")
    console.item("Engine", f"{ENGINE_NAME} ({ENGINE_CODE})")
    console.item("Timeframe", timeframe)
    console.item("Z-score lookback", f"{clr.z_look} bars (needs {clr.min_bars} to signal)")
    console.item("Trigger", f"±{clr.thr:.1f}σ · hold {clr.horizon} bars · entry next open")
    _vl, _vk, _vd = _study_state(study, "buy")
    console.item("Measured edge (buy)", f"{_vl} — {_study_summary_line(study, 'buy')}")
    console.item("Measured edge (sell)", f"{_study_state(study, 'sell')[0]} — "
                                         f"{_study_summary_line(study, 'sell')}")
    console.item("Reference class", f"{clr.iclass} · source study {clr.prior_edge:+.3f} vol · "
                                    f"{clr.prior_hit:.1f}% hit (prior, not applied)")
    console.item("Cost gate", f"{clr.cost_bps:.1f} bp · "
                              + ("net positive" if clr.cost_ok(study) else "NET NEGATIVE")
                              + f" · basis {clr.cost_basis(study)}")
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
                                      wt2_len, wt2_type, end_date, clr.params_sig)
    _cache_hits = 0

    _tf_label = "weekly" if timeframe == "Weekly" else "daily"
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
                if timeframe == "Weekly":
                    df = resample_to_weekly(df)

            # Warmup guard — a symbol cannot carry a CLR signal until it has a full
            # z-score lookback (the Pine's `dBars < zLook + 2` refusal). Applied on both
            # cache hit and miss so a short frame cached by the (unguarded) harvest can't
            # slip a symbol whose z-score would be NaN.
            _min_bars = max(reg_len + 30, clr.min_bars)
            if len(df) < _min_bars:
                console.detail(f"{ticker}: Skipped (warming up: {len(df)} of {_min_bars} bars needed)")
                _warmup_skipped += 1
                continue

            if _cached is None:
                df = run_full_analysis(df, reg_len, wt_n1, wt_n2, obLevel1, obLevel2, osLevel1, osLevel2,
                                       wt2_len=wt2_len, wt2_type=wt2_type, clr=clr)
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

            # Get historical signals for tracking (Today, 1d, 2d, 3d, Within 5d)
            sample_range = df.iloc[max(0, idx_pos - 5) : idx_pos + 1]

            last_row = df.iloc[idx_pos]

            # Recent return volatility — the asset-agnostic σ scale used to report how far
            # price has run since an aged signal fired.
            try:
                _retvol20 = float(df['Close'].pct_change().rolling(20).std().iloc[idx_pos])
            except Exception:
                _retvol20 = float('nan')

            signal_type = _classify_signal_type(last_row)

            # The z-score at each of the last 5 bars, so an aged signal can report the z
            # that fired it (offset 0 = the snapshot bar … 4 = five bars back).
            _z_win = df['CLR_Z'].iloc[max(0, idx_pos - 4): idx_pos + 1].tolist()
            _z_hist = list(reversed(_z_win))            # [today, 1 back, 2 back, …]
            _z_hist += [float('nan')] * (5 - len(_z_hist))
            _close_win = df['Close'].iloc[max(0, idx_pos - 4): idx_pos + 1].tolist()
            _close_hist = list(reversed(_close_win))
            _close_hist += [float('nan')] * (5 - len(_close_hist))

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

            results.append({
                "% Chng Since": pct_chng_since,  # None when data unavailable — displays as NaN / "—"
                "Symbol": ticker,
                "DisplayName": display_name,
                "SimpleName": simple_name,
                # Signal == the CLR fade score (-z). Positive = bullish (weak close).
                "Signal": round(float(last_row['Fade_Score']), 3) if pd.notna(last_row['Fade_Score']) else np.nan,
                "CLR_Z": float(last_row['CLR_Z']) if pd.notna(last_row['CLR_Z']) else np.nan,
                # Arithmetic ceiling for this bar's window — conviction is scaled against it
                # because a z-score of a [-1,+1] bounded variable cannot reach 3. See engine.py.
                "CLR_Z_Cap": (float(last_row['CLR_Z_Cap'])
                              if pd.notna(last_row.get('CLR_Z_Cap')) else np.nan),
                "CLR_CLV": float(last_row['CLR_CLV']) if pd.notna(last_row['CLR_CLV']) else np.nan,
                "CLR_State": str(last_row.get('CLR_State', 'NEUTRAL')),
                "CLR_Hold_Dir": int(last_row.get('CLR_Hold_Dir', 0) or 0),
                "CLR_Hold_Age": (float(last_row['CLR_Hold_Age'])
                                if pd.notna(last_row.get('CLR_Hold_Age')) else np.nan),
                "CLR_Horizon": int(clr.horizon),
                "Bar_Delta": round(last_row['Bar_Delta'], 2) if not pd.isna(last_row['Bar_Delta']) else 0.0,
                "CVD": round(last_row['CVD'], 2) if not pd.isna(last_row['CVD']) else 0.0,
                "CVD_Slope": round(last_row['CVD_Slope'], 2) if not pd.isna(last_row['CVD_Slope']) else 0.0,
                "Delta_Z": round(last_row['Delta_Z'], 2) if not pd.isna(last_row['Delta_Z']) else 0.0,
                "Abs_Strength": round(last_row.get('Abs_Strength', 0), 2) if not pd.isna(last_row.get('Abs_Strength', 0)) else 0.0,
                "Buy_Share": round(last_row.get('Buy_Share', 0.5), 3) if not pd.isna(last_row.get('Buy_Share', 0.5)) else 0.5,
                "Absorption_Score": round(last_row.get('Absorption_Score', 0.0), 3) if not pd.isna(last_row.get('Absorption_Score', 0.0)) else 0.0,
                "Zone": last_row['Condition'],
                "SignalType": signal_type,
                "Price": round(last_row['Close'], 2),
                "PctChange": round(pct_change, 2),
                # v3 Metrics for Engine 2.0
                "RetVol20":      _retvol20,
                "HMM_Bull":      float(last_row.get('HMM_Bull', 0.33)),
                "HMM_Bear":      float(last_row.get('HMM_Bear', 0.33)),
                "Vol_Regime":    str(last_row.get('Vol_Regime', 'NORMAL')),
                "Change_Point":  bool(last_row.get('Change_Point', False)),
                "Regime_Confidence": float(last_row.get('Regime_Confidence', 0.0)),
                "F1_PriceMom":   float(last_row.get('F1_PriceMom', 0)),
                "F2_VolQual":    float(last_row.get('F2_VolQual', 0)),
                "ATR_Pct":       last_row.get('ATR_Pct'),
                # ── BUY_* — green triangle (weak close, z < -thr), by signal age ──
                "BUY_Today": "●" if sample_range.iloc[-1]['buy_cond'] else "—",
                "BUY_1d": "●" if sample_range.iloc[-2]['buy_cond'] else "—",
                "BUY_2d": "●" if sample_range.iloc[-3]['buy_cond'] else "—",
                "BUY_3d": "●" if sample_range.iloc[-4]['buy_cond'] else "—",
                "BUY_5d": "●" if sample_range.tail(5)['buy_cond'].any() else "—",
                # ── SELL_* — yellow diamond (strong close, z > +thr), by signal age ──
                "SELL_Today": "●" if sample_range.iloc[-1]['sell_cond'] else "—",
                "SELL_1d": "●" if sample_range.iloc[-2]['sell_cond'] else "—",
                "SELL_2d": "●" if sample_range.iloc[-3]['sell_cond'] else "—",
                "SELL_3d": "●" if sample_range.iloc[-4]['sell_cond'] else "—",
                "SELL_5d": "●" if sample_range.tail(5)['sell_cond'].any() else "—",
                # Per-age z-score / close so an aged row reports the bar that fired it.
                "Z_Hist":     _z_hist,
                "Close_Hist": _close_hist,
                # Additional fields for detail cards
                "Osc_Value": round(last_row.get('Delta_Z', 0), 2),
                "MA_Alignment": int(last_row.get('MA_Alignment', 0)),
                "ZScore_Value": round(last_row.get('Delta_Z', 0), 2),
            })

            _z_disp = float(last_row['CLR_Z']) if pd.notna(last_row['CLR_Z']) else float('nan')
            console.detail(f"[{i+1}/{len(data_dict)}] {ticker}: z={_z_disp:+.2f}  "
                           f"state={last_row.get('CLR_State', '—')}  zone={last_row['Condition']}")

        except Exception as e:
            console.failure(f"Analysis Failed: {ticker}", str(e))
            _failed_symbols.append(ticker)
            continue

    console.end_phase("SIGNAL SCREEN")
    if _cache_hits:
        console.detail(f"Analyzed-frame cache: reused {_cache_hits}/{len(data_dict)} frames from the range harvest (skipped re-analysis)")
    if _warmup_skipped:
        console.detail(f"Warmup: {_warmup_skipped} symbol(s) skipped — fewer than {clr.min_bars} bars, so the z-score has no lookback")
    # One-shot cache — release the harvested frames now that the screener has consumed them.
    _analyzed_cache_clear()

    _fail_count = len(_failed_symbols)
    console.summary("RUN SUMMARY", {
        "Universe": universe,
        "Universe Index": selected_index,
        "Instrument Class": clr.iclass,
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
            st.warning(
                f"**No market data retrieved** for {selected_index} as of {analysis_date}. "
                "The exchange may have been closed, or yfinance may be rate-limiting. "
                "Try refreshing or selecting a recent trading day."
            )
        elif _warmup_skipped >= _n_fetched:
            st.warning(
                f"**Every symbol is still warming up.** CLR needs {clr.min_bars} "
                f"{'weekly' if timeframe == 'Weekly' else 'daily'} bars before the close-location "
                f"z-score has a lookback, and none of the {_n_fetched} symbols in {selected_index} "
                f"has that much history as of {analysis_date}. Try the Daily timeframe, or a "
                "universe with longer-listed instruments."
            )
        else:
            st.info(
                f"**Nothing to show** — {_n_fetched} of {_n_total} symbols had data for {analysis_date}, "
                "but none produced a usable close-location reading. "
                "Try an adjacent trading date, or check that the selected date is a market session."
            )
        # Return empty DataFrame with expected columns to prevent downstream KeyErrors
        expected_cols = [
            "Symbol", "DisplayName", "SimpleName", "Signal", "CLR_Z", "CLR_Z_Cap", "CLR_CLV",
            "CLR_State", "CLR_Hold_Dir", "CLR_Hold_Age", "CLR_Horizon",
            "Bar_Delta", "CVD", "CVD_Slope", "Delta_Z", "Buy_Share", "Absorption_Score",
            "Zone", "SignalType", "Price", "PctChange",
            "BUY_Today", "BUY_1d", "BUY_2d", "BUY_3d", "BUY_5d",
            "SELL_Today", "SELL_1d", "SELL_2d", "SELL_3d", "SELL_5d",
            "Osc_Value", "MA_Alignment", "ZScore_Value",
        ]
        return pd.DataFrame(columns=expected_cols)

    results_df = pd.DataFrame(results)

    # Cross-sectional ranking (engine.py): order by the fade score (-z), assign Side from
    # which side of ±thr the close location landed, and set conviction from |z|'s position
    # inside its attainable range × the cost gate. The measured expectancy (`study`) informs
    # the cost gate and the per-row read; it never scales or filters a signal. One call emits
    # the whole UI contract.
    if not results_df.empty:
        results_df = eng.compute_ranking(results_df, cost_bps=clr.cost_bps,
                                         thr=clr.thr, study=study)

    return results_df


def run_timeseries_analysis(universe, selected_index, start_date, end_date, reg_len, wt_n1, wt_n2, levels, timeframe, wt2_len=20, wt2_type="ALMA",
                            external_progress_slot=None, progress_offset=0, progress_scale=100,
                            clr=None, study=None):
    """Compute the per-(date, symbol) CLR frame for a date range.

    Pure compute path: fetches history, runs the full / regime analyses on every symbol,
    builds the per-(date, symbol) row set with forward-return labels, and stores
    ts_results_df + ts_meta in ``st.session_state``. **Does not render UI.** The dashboard
    is rendered separately by ``render_timeseries_dashboard()`` so it survives sidebar
    interactions / reruns.

    ``external_progress_slot`` lets a caller share one progress bar over
    [offset, offset+scale] instead of stacking a second bar.
    """
    if clr is None:
        clr = _clr_settings(universe, selected_index, timeframe)
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
        st.error("Failed to retrieve stock list")
        return

    console.success(f"Fetched {len(stock_list)} symbols for {selected_index}")
    console.section("Mass Historical Download")
    # Registry-first: if the same universe was fetched recently it won't hit yfinance again
    data_dict, msg = get_universe_data(stock_list, end_date=end_date)

    if not data_dict:
        console.error("No historical data available")
        st.error("No historical data available for selected range.")
        return

    console.success(f"Downloaded depth for {len(data_dict)} entities")

    # Start Unified Harvesting Phase
    console.start_phase("SIGNAL HARVEST", 2, 2)
    start_harvest = time.time()

    _p(15, "Harvesting Signals", f"{len(data_dict)} Stocks")
    all_results = []

    # Analyzed-frame cache for this run — lets a screener that follows skip
    # re-running the identical per-stock analysis pipeline (see helper comment).
    _cache_sig = _analysis_params_sig(timeframe, reg_len, wt_n1, wt_n2, levels,
                                      wt2_len, wt2_type, end_date, clr.params_sig)
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
            if timeframe == "Weekly":
                df = resample_to_weekly(df)
            df = run_full_analysis(df, reg_len, wt_n1, wt_n2, *levels,
                                   wt2_len=wt2_len, wt2_type=wt2_type, clr=clr)
            df = run_regime_analysis(df)
            # Cache the analyzed frame so run_screener_analysis can reuse it instead
            # of recomputing. Stored by reference — the harvest-only columns appended
            # below (Ret_*, SignalType) are harmless extras; the screener copies on read.
            _analyzed_cache_put(ticker, df, _cache_sig)

            # Forward-return labels at the CLR horizons (5-10 bars is where the edge
            # lives; 1 and 21 bracket its decay). Labels only — never signal inputs.
            for h in eng.HOLD_HORIZONS:
                df[f'Ret_{h}b'] = df['Close'].shift(-h) / df['Close'] - 1

            # Vectorized SignalType per bar — a fired event wins, else the flow zone.
            df['SignalType'] = np.select(
                [df['buy_cond'], df['sell_cond'], df['Condition'] != 'Neutral'],
                ['BUY', 'SELL', df['Condition']],
                default='-',
            )

            mask = (df.index.date >= start_date) & (df.index.date <= end_date)
            range_df = df.loc[mask]

            for date, row in range_df.iterrows():
                all_results.append({
                    'Date': date,
                    'Symbol': ticker,
                    # Signal == the CLR fade score (-z). Positive = bullish (weak close).
                    'Signal': row['Fade_Score'],
                    'CLR_Z': row['CLR_Z'],
                    'CLR_CLV': row['CLR_CLV'],
                    'CLR_State': row.get('CLR_State', 'NEUTRAL'),
                    'CLR_Hold_Dir': row.get('CLR_Hold_Dir', 0),
                    'CLR_Hold_Age': row.get('CLR_Hold_Age'),
                    'Bar_Delta': row['Bar_Delta'],
                    'CVD': row['CVD'],
                    'CVD_Slope': row['CVD_Slope'],
                    'Delta_Z': row['Delta_Z'],
                    'Zone': row['Condition'],
                    # BuySignal / SellSignal are the aggregation-facing names the range
                    # dashboard counts per day; buy_cond / sell_cond are the raw booleans.
                    'BuySignal': row['buy_cond'],
                    'SellSignal': row['sell_cond'],
                    'buy_cond': row['buy_cond'],
                    'sell_cond': row['sell_cond'],
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
        st.error("No results generated for the selected timeframe.")
        return

    ts_df = pd.DataFrame(all_results)
    ts_df['Date'] = pd.to_datetime(ts_df['Date'])
    ts_df = ts_df.sort_values('Date')

    daily_agg, summary = _aggregate_timeseries(ts_df)

    console.summary("HISTORICAL RANGE SUMMARY", {
        "Universe": universe,
        "Universe Index": selected_index,
        "Instrument Class": clr.iclass,
        "Historical Range": f"{start_date} to {end_date}",
        "Total Signals Fired": summary['total_signals'],
        "Buy / Sell": f"{summary['total_buys']} / {summary['total_sells']}",
        "Avg Fade Score": round(summary['avg_signal'], 3),
        "Buy:Sell Ratio": round(summary['overall_ratio'], 2),
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
        "iclass":         clr.iclass,
        "thr":            clr.thr,
        "z_look":         clr.z_look,
    }

    # Only clear our OWN bar. When sharing the Single-Date bar, the screener that
    # follows keeps rendering into it (the 40→100% phase).
    if _own_slot:
        progress_slot.empty()


# ══════════════════════════════════════════════════════════════════════════════
# TIMESERIES — AGGREGATION + DASHBOARD RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def _aggregate_timeseries(ts_df):
    """Aggregate the per-(date, symbol) CLR frame into daily metrics + summary stats.

    Pure function — used by both ``run_timeseries_analysis`` (for the console
    summary on harvest) and ``render_timeseries_dashboard`` (re-rendered on every
    Streamlit run from session state, so sidebar interactions don't lose the view).
    """
    daily_agg = ts_df.groupby('Date').agg({
        'Signal': 'mean',
        'CVD': 'mean',
        'CVD_Slope': 'mean',
        'BuySignal': 'sum',
        'SellSignal': 'sum',
        'Zone': lambda x: x.value_counts().idxmax() if len(x) > 0 else 'Neutral',
        'Regime': lambda x: x.value_counts().idxmax() if len(x) > 0 else 'NEUTRAL',
        'HMM_Bull': 'mean',
        'HMM_Bear': 'mean',
        'Vol_Regime': lambda x: x.value_counts().idxmax() if len(x) > 0 else 'NORMAL',
        'Change_Point': 'sum',
        'Regime_Confidence': 'mean',
    })

    daily_agg['TotalSignals'] = daily_agg['BuySignal'] + daily_agg['SellSignal']
    daily_agg['B_S_Ratio']    = np.where(
        daily_agg['SellSignal'] == 0,
        np.nan,                          # undefined (all buys, no sells) — NaN in charts
        daily_agg['BuySignal'] / daily_agg['SellSignal'],
    )
    daily_agg['Flow_Strength'] = daily_agg['Signal'].abs()

    # Signal breadth: % of the universe firing each event on a given day. This is the
    # read that matters for CLR — a day where 30% of names close weak is a very
    # different tape from one where 3% do.
    total_per_day_all = ts_df.groupby('Date').size()
    daily_agg['Buy_Breadth_Pct']  = (daily_agg['BuySignal']  / total_per_day_all * 100).fillna(0)
    daily_agg['Sell_Breadth_Pct'] = (daily_agg['SellSignal'] / total_per_day_all * 100).fillna(0)

    # Flow-zone breadth: % of names in accumulation vs distribution each day.
    acc_counts    = ts_df.groupby('Date')['Zone'].apply(lambda x: (x.isin(['Accumulation+', 'Accumulation'])).sum())
    dist_counts   = ts_df.groupby('Date')['Zone'].apply(lambda x: (x.isin(['Distribution+', 'Distribution'])).sum())
    total_per_day = ts_df.groupby('Date').size()
    daily_agg['Oversold_Pct']   = (dist_counts / total_per_day * 100).fillna(0)
    daily_agg['Overbought_Pct'] = (acc_counts  / total_per_day * 100).fillna(0)

    regime_bull  = ts_df.groupby('Date')['Regime'].apply(lambda x: x.str.contains('BULL', na=False).sum())
    regime_bear  = ts_df.groupby('Date')['Regime'].apply(lambda x: x.str.contains('BEAR', na=False).sum())
    regime_trans = ts_df.groupby('Date')['Regime'].apply(lambda x: (x == 'TRANSITION').sum())
    daily_agg['Regime_Bull_Pct']       = (regime_bull  / total_per_day * 100).fillna(0)
    daily_agg['Regime_Bear_Pct']       = (regime_bear  / total_per_day * 100).fillna(0)
    daily_agg['Regime_Transition_Pct'] = (regime_trans / total_per_day * 100).fillna(0)

    # Mean |z| of the day's fired signals — how far past the threshold the tape actually
    # went, not just how many names crossed it.
    _fired = ts_df[ts_df['BuySignal'] | ts_df['SellSignal']] if 'BuySignal' in ts_df.columns else ts_df.iloc[0:0]
    if len(_fired) and 'CLR_Z' in _fired.columns:
        daily_agg['Avg_Fired_Z'] = _fired.groupby('Date')['CLR_Z'].apply(lambda s: s.abs().mean())
    else:
        daily_agg['Avg_Fired_Z'] = np.nan

    _n_buys  = int(daily_agg['BuySignal'].sum())
    _n_sells = int(daily_agg['SellSignal'].sum())
    summary = {
        'total_signals':       int(daily_agg['TotalSignals'].sum()),
        'total_buys':          _n_buys,
        'total_sells':         _n_sells,
        'avg_signal':          float(daily_agg['Signal'].mean()),
        'overall_ratio':       float(_n_buys / max(_n_sells, 1)),
        'avg_buy_breadth':     float(daily_agg['Buy_Breadth_Pct'].mean()),
        'avg_sell_breadth':    float(daily_agg['Sell_Breadth_Pct'].mean()),
        'avg_fired_z':         float(daily_agg['Avg_Fired_Z'].mean()) if daily_agg['Avg_Fired_Z'].notna().any() else float('nan'),
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
    _iclass = meta.get('iclass', '—')
    _thr    = float(meta.get('thr', eng.CLR_THRESHOLD))
    ui.render_section_header(
        f"Historical Range ({range_label})",
        f"{ENGINE_NAME} · ±{_thr:.1f}σ · {_iclass}",
        icon="history", accent="violet",
    )

    # ── Summary metric row (6 cards, mirrors single-date / pulse cadence) ──
    _avg_z = summary.get('avg_fired_z', float('nan'))
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        ui.render_metric_card("Signals Fired", str(summary['total_signals']),
                              f"{summary['total_buys']} buy · {summary['total_sells']} sell", "info")
    with c2:
        ui.render_metric_card("Avg Buy Breadth", f"{summary['avg_buy_breadth']:.1f}%",
                              f"{timeframe_label} · weak closes", "success")
    with c3:
        ui.render_metric_card("Avg Sell Breadth", f"{summary['avg_sell_breadth']:.1f}%",
                              f"{timeframe_label} · strong closes", "danger")
    with c4:
        ui.render_metric_card("Avg Fired |z|", f"{_avg_z:.2f}" if np.isfinite(_avg_z) else "—",
                              f"vs ±{_thr:.1f}σ trigger", "warning")
    with c5:
        ui.render_metric_card("Buy:Sell Ratio", f"{summary['overall_ratio']:.2f}",
                              f"{'Buy' if summary['overall_ratio'] > 1 else 'Sell'}-skewed tape", "info")
    with c6:
        ui.render_metric_card("Trading Days", str(len(daily_agg)), "Analyzed", "neutral")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Signal Dashboard",
        "Transaction Dynamics",
        "Regime Analysis",
        "Data Terminal",
    ])

    # ── TAB 1 · Signal Dashboard ───────────────────────────────────────────
    with tab1:
        ui.render_section_header("Signal Breadth",
                                 f"% of universe firing BUY / SELL past ±{_thr:.1f}σ",
                                 icon="activity", accent="cyan")
        fig_breadth = go.Figure()
        fig_breadth.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Buy_Breadth_Pct'],
                                         mode='lines', name='Buy % (weak closes)',
                                         fill='tozeroy', fillcolor='rgba(0,230,118,0.12)',
                                         line=dict(color='#00E676', width=2)))
        fig_breadth.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Sell_Breadth_Pct'],
                                         mode='lines', name='Sell % (strong closes)',
                                         fill='tozeroy', fillcolor='rgba(255,167,38,0.12)',
                                         line=dict(color='#FFA726', width=2)))
        _pct_raw = max(daily_agg['Buy_Breadth_Pct'].max(), daily_agg['Sell_Breadth_Pct'].max())
        _pct_raw = float(_pct_raw) if pd.notna(_pct_raw) and np.isfinite(_pct_raw) else 0.0
        ymax = max(_pct_raw * 1.15, 5.0)   # floor at 5 so axis always renders sensibly
        fig_breadth.update_layout(title='', height=350, hovermode='x unified',
                                  yaxis=dict(range=[0, ymax], title='% of Universe'))
        apply_chart_theme(fig_breadth)
        st.plotly_chart(fig_breadth, width='stretch', key='chart_breadth')

        st.markdown("<br>", unsafe_allow_html=True)
        ui.render_section_header("Signal Count by Date", "BUY vs SELL fires per session",
                                 icon="bar-chart", accent="info")
        fig_counts = go.Figure()
        fig_counts.add_trace(go.Bar(x=daily_agg.index, y=daily_agg['BuySignal'],
                                    name='Buy Signals',
                                    marker=dict(color='#00E676', line=dict(color='#00E676', width=1))))
        fig_counts.add_trace(go.Bar(x=daily_agg.index, y=daily_agg['SellSignal'],
                                    name='Sell Signals',
                                    marker=dict(color='#FFA726', line=dict(color='#FFA726', width=1))))
        fig_counts.update_layout(title='', height=300, hovermode='x unified', barmode='group')
        apply_chart_theme(fig_counts)
        st.plotly_chart(fig_counts, width='stretch', key='chart_signal_counts')

    # ── TAB 2 · Transaction Dynamics ───────────────────────────────────────
    with tab2:
        ui.render_section_header("Signal Trends",
                                 "BUY / SELL fire counts over time",
                                 icon="zap", accent="emerald")
        fig_signals = go.Figure()
        fig_signals.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['BuySignal'],
                                         mode='lines+markers', name='Buy Signals',
                                         line=dict(color='#00E676', width=2),
                                         marker=dict(size=6, color='#00E676')))
        fig_signals.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['SellSignal'],
                                         mode='lines+markers', name='Sell Signals',
                                         line=dict(color='#FFA726', width=2),
                                         marker=dict(size=6, color='#FFA726')))
        fig_signals.update_layout(title='', height=300, hovermode='x unified')
        apply_chart_theme(fig_signals)
        st.plotly_chart(fig_signals, width='stretch', key='chart_signals_overtime')

        st.markdown("<br>", unsafe_allow_html=True)
        ui.render_section_header("Flow-Zone Breadth",
                                 "Accumulation vs Distribution over time — context, not signal",
                                 icon="trending-up", accent="amber")
        fig_div = go.Figure()
        fig_div.add_trace(go.Bar(x=daily_agg.index, y=daily_agg['Overbought_Pct'],
                                 name='Accumulation %',
                                 marker=dict(color='#D4A853', line=dict(color='#D4A853', width=1))))
        fig_div.add_trace(go.Bar(x=daily_agg.index, y=-daily_agg['Oversold_Pct'],
                                 name='Distribution %',
                                 marker=dict(color='#06B6D4', line=dict(color='#06B6D4', width=1))))
        fig_div.update_layout(title='', height=300, hovermode='x unified', barmode='relative')
        apply_chart_theme(fig_div)
        st.plotly_chart(fig_div, width='stretch', key='chart_divergence')

    # ── TAB 3 · Regime Analysis ────────────────────────────────────────────
    with tab3:
        ui.render_section_header("Aggregate Close-Location Skew",
                                 "Universe-mean fade score (−z) over time",
                                 icon="activity", accent="rose")
        # Signal = per-name fade score (−z, σ≈1); its daily cross-sectional MEAN
        # concentrates near zero (σ ≈ 1/√N ≈ 0.08 for ~150 names), so the extreme
        # bands sit at ±0.25 (~3σ of that mean) rather than at the ±1.5σ per-name
        # trigger. Green (positive) = the universe closed weak = bullish for CLR.
        _sig_band = 0.25
        colors = ['#00E676' if v > _sig_band else '#FFA726' if v < -_sig_band else '#64748B'
                  for v in daily_agg['Signal']]
        fig_avg = go.Figure()
        fig_avg.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Signal'].clip(lower=0),
                                     fill='tozeroy', fillcolor='rgba(0,230,118,0.05)',
                                     line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig_avg.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Signal'].clip(upper=0),
                                     fill='tozeroy', fillcolor='rgba(255,167,38,0.05)',
                                     line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig_avg.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Signal'],
                                     mode='lines+markers', name='Avg fade score',
                                     line=dict(color='#D4A853', width=2),
                                     marker=dict(size=6, color=colors)))
        fig_avg.add_hline(y=_sig_band,  line=dict(color='rgba(0,230,118,0.5)', width=1, dash='dash'))
        fig_avg.add_hline(y=-_sig_band, line=dict(color='rgba(255,167,38,0.5)', width=1, dash='dash'))
        fig_avg.add_hline(y=0,   line=dict(color='rgba(255,255,255,0.3)', width=1))
        _sig_span = float(np.nanmax(np.abs(daily_agg['Signal']))) if len(daily_agg) else 0.0
        _sig_span = _sig_span if np.isfinite(_sig_span) else 0.0
        _sig_lim = max(_sig_span * 1.2, _sig_band * 1.6)
        fig_avg.update_layout(title='', height=300, hovermode='x unified',
                              yaxis=dict(range=[-_sig_lim, _sig_lim]))
        apply_chart_theme(fig_avg)
        st.plotly_chart(fig_avg, width='stretch', key='chart_avg_signal')

        st.markdown("<br>", unsafe_allow_html=True)
        ui.render_section_header("HMM Regime Distribution Over Time",
                                 "Percentage of symbols in each HMM regime daily",
                                 icon="activity", accent="cyan")
        fig_regime = go.Figure()
        fig_regime.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Regime_Bull_Pct'],
                                        mode='lines', name='Bull Regime %',
                                        fill='tozeroy', fillcolor='rgba(52,211,153,0.12)',
                                        line=dict(color='#2DD4A8', width=2)))
        fig_regime.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Regime_Bear_Pct'],
                                        mode='lines', name='Bear Regime %',
                                        fill='tozeroy', fillcolor='rgba(232,85,90,0.12)',
                                        line=dict(color='#E8555A', width=2)))
        fig_regime.update_layout(title='', height=300, hovermode='x unified',
                                 yaxis=dict(range=[0, 100], title='% of Universe'))
        apply_chart_theme(fig_regime)
        st.plotly_chart(fig_regime, width='stretch', key='chart_regime')

        st.markdown("<br>", unsafe_allow_html=True)
        ui.render_section_header("Volatility Dynamics",
                                 "Volatility Regime & Change Points Over Time",
                                 icon="shield", accent="amber")
        vol_high = ts_df.groupby('Date')['Vol_Regime'].apply(
            lambda x: (x.isin(['HIGH', 'EXTREME'])).sum() / len(x) * 100)
        fig_vol = go.Figure()
        # High-Vol % belongs on the RIGHT axis (yaxis2) — without the explicit
        # assignment both series shared y1 and the labeled right axis sat empty,
        # letting the 0-100% line crush the per-day change-point counts.
        fig_vol.add_trace(go.Scatter(x=daily_agg.index, y=vol_high.fillna(0),
                                     mode='lines+markers', name='High Vol %',
                                     yaxis='y2',
                                     line=dict(color='#D4A853', width=2),
                                     marker=dict(size=5)))
        fig_vol.add_trace(go.Bar(x=daily_agg.index, y=daily_agg['Change_Point'],
                                 name='Symbols with Regime Change',
                                 marker=dict(color='#A855F7', opacity=0.7)))
        fig_vol.update_layout(
            title='', height=250, hovermode='x unified',
            yaxis=dict(title='# Symbols'),
            yaxis2=dict(title='High-Vol %', overlaying='y', side='right'),
        )
        apply_chart_theme(fig_vol)
        st.plotly_chart(fig_vol, width='stretch', key='chart_volatility')

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
            st.dataframe(pd.DataFrame(regime_stats), width='stretch', hide_index=True)
        with col_r2:
            ui.render_section_header("Fade-Score Distribution",
                                     "Universe-mean fade score (−z) statistics",
                                     icon="database", accent="rose")
            signal_stats = {
                "Metric": ["Mean", "Median", "Min", "Max", "Std Dev"],
                "Value": [f"{daily_agg['Signal'].mean():+.3f}",
                          f"{daily_agg['Signal'].median():+.3f}",
                          f"{daily_agg['Signal'].min():+.3f}",
                          f"{daily_agg['Signal'].max():+.3f}",
                          f"{daily_agg['Signal'].std():.3f}"],
            }
            st.dataframe(pd.DataFrame(signal_stats), width='stretch', hide_index=True)

    # ── TAB 4 · Data Terminal ──────────────────────────────────────────────
    with tab4:
        timeframe_label = "Weekly Time Series" if timeframe == 'Weekly' else "Daily Time Series"
        ui.render_section_header("Analytical Data",
                                 f"{timeframe_label} ({len(daily_agg)} periods)",
                                 icon="list", accent="cyan")
        display_ts = daily_agg.copy()
        display_ts.index = display_ts.index.strftime('%Y-%m-%d')
        display_ts = display_ts.reset_index().rename(columns={'Date': 'Date'})
        display_cols = ['Date', 'BuySignal', 'SellSignal', 'Signal', 'Avg_Fired_Z',
                        'Buy_Breadth_Pct', 'Sell_Breadth_Pct',
                        'Regime_Bull_Pct', 'Regime_Bear_Pct', 'Change_Point']
        display_ts = display_ts[display_cols]
        display_ts.columns = ['Date', 'Buy Sig', 'Sell Sig', 'Avg Fade', 'Avg Fired |z|',
                              'Buy Breadth %', 'Sell Breadth %',
                              'Bull Regime %', 'Bear Regime %', 'Change Pts']
        st.dataframe(
            display_ts, width='stretch', hide_index=True,
            column_config={
                'Date':          st.column_config.TextColumn(help="Trading day (YYYY-MM-DD)."),
                'Buy Sig':       st.column_config.NumberColumn(help=f"Symbols firing the CLR BUY (green triangle) — close-location z below −{_thr:.1f}σ, a weak close to fade up."),
                'Sell Sig':      st.column_config.NumberColumn(help=f"Symbols firing the CLR SELL (yellow diamond) — close-location z above +{_thr:.1f}σ. Note: this side did not confirm out of sample."),
                'Avg Fade':      st.column_config.NumberColumn(help="Cross-sectional mean fade score (−z) on this day. The daily mean concentrates near 0; ±0.25 is already a strongly one-sided tape.", format="%.3f"),
                'Avg Fired |z|': st.column_config.NumberColumn(help=f"Mean |z| of the symbols that actually fired — how far past the ±{_thr:.1f}σ trigger the tape went, blank on days with no fires.", format="%.2f"),
                'Buy Breadth %': st.column_config.NumberColumn(help="Percent of the universe firing BUY on this day.", format="%.1f"),
                'Sell Breadth %':st.column_config.NumberColumn(help="Percent of the universe firing SELL on this day.", format="%.1f"),
                'Bull Regime %': st.column_config.NumberColumn(help="Percent of universe with HMM regime label containing 'BULL' (risk context, not a signal input)."),
                'Bear Regime %': st.column_config.NumberColumn(help="Percent of universe with HMM regime label containing 'BEAR' (risk context, not a signal input)."),
                'Change Pts':    st.column_config.NumberColumn(help="Sum of Change_Point flags — count of symbols with a regime-state transition on this day."),
            },
        )

        st.markdown("<br>", unsafe_allow_html=True)
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

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CORRELATION MODE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_correlation_analysis(universe, selected_index, target_ticker, lookback, method, timeframe, analysis_date=None, clr=None, study=None):
    """Execute correlation analysis between universe constituents and a target asset.

    Returns a dict with correlation data, rolling correlations, prices, and returns,
    plus a confluence score (|correlation| × normalised CLR fade-score strength).
    """
    if analysis_date is None:
        analysis_date = _today_ist()
    if clr is None:
        clr = _clr_settings(universe, selected_index, timeframe)
    progress_slot = st.empty()
    progress_bar(progress_slot, 5, "Initializing Correlation Engine", "Fetching Market Data")

    try:
        # Fetch universe symbols
        stock_list, msg = resolve_universe(universe, selected_index)

        if not stock_list:
            st.error(f"Failed to fetch universe symbols: {msg}")
            return None

        console.item("Symbols fetched", len(stock_list))

        progress_bar(progress_slot, 15, "Fetching OHLCV Data", f"Symbols: {len(stock_list)}")

        # ── Universe data from registry (shared pool with the screener) ──
        # Passing only the universe symbols so the registry key is consistent with
        # the screener and timeseries paths.  The target ticker is supplemented
        # below with a single small fetch if it is not already in the pool.
        data_dict, fetch_msg = get_universe_data(stock_list, end_date=analysis_date)
        if data_dict is None:
            st.error(f"Data fetch failed: {fetch_msg}")
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
            target_raw, _ = get_universe_data([target_ticker], end_date=analysis_date)
            if target_raw and target_ticker in target_raw:
                # Merge into a new dict so we don't mutate the registry entry
                data_dict = {**data_dict, target_ticker: target_raw[target_ticker]}
                console.detail(f"Target ticker '{target_ticker}' merged into data pool")
            else:
                st.error(f"Could not fetch target asset '{target_ticker}'")
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
            st.error("No valid price data found for universe")
            console.item("Error", "No Close prices extracted")
            return None

        console.item("Close prices extracted for", len(close_dict))

        close_df = pd.DataFrame(close_dict)
        close_df = close_df.dropna(axis=1, how='all')

        console.item("Close DataFrame shape", f"{close_df.shape}")

        if len(close_df) < lookback + 10:
            st.error(f"Insufficient historical data for correlation analysis (only {len(close_df)} rows, need {lookback + 10})")
            console.item("Error", f"Only {len(close_df)} rows, need {lookback + 10}")
            return None

        # Resample to weekly if needed
        if timeframe == "Weekly":
            close_df = resample_to_weekly(close_df)

        progress_bar(progress_slot, 40, "Computing Returns", f"Method: {method}")

        # Compute log returns — drop rows only where all values are NaN
        returns_df = np.log(close_df / close_df.shift(1)).dropna(how='all')

        if target_ticker not in returns_df.columns:
            st.error(f"Target asset '{target_ticker}' not in data")
            console.item("Error", f"Target {target_ticker} not in returns columns")
            return None

        target_returns = returns_df[target_ticker].dropna()
        console.item("Target returns available", len(target_returns))

        # Filter to common dates with target
        common_idx = returns_df.index.intersection(target_returns.index)
        if len(common_idx) < lookback + 10:
            st.error(f"Insufficient overlapping data (only {len(common_idx)} days). Try a shorter lookback period.")
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
                st.error("Could not compute rolling correlations for any column")
                return None

            console.item("Rolling corr DataFrame shape", rolling_corr_df.shape)
        except Exception as e:
            st.error(f"Error in rolling correlation: {str(e)}")
            console.item("Rolling corr computation error", str(e)[:100])
            return None

        if rolling_corr_df.empty or len(rolling_corr_df) == 0:
            st.error("Could not compute rolling correlations. Check data availability.")
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
            clr_results = _sdf
        else:
            _corr_reg_len, _corr_n1, _corr_n2 = 20, 10, 21
            _corr_levels = (80, 40, -80, -40)
            _corr_wt2_len, _corr_wt2_type = 20, "ALMA"
            clr_results = run_screener_analysis(
                universe, selected_index, analysis_date,
                _corr_reg_len, _corr_n1, _corr_n2, _corr_levels, timeframe,
                show_progress=False, external_progress_slot=progress_slot,
                progress_offset=60, progress_scale=30,
                wt2_len=_corr_wt2_len, wt2_type=_corr_wt2_type, clr=clr, study=study,
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

            # Pull this symbol's CLR read from the screener output already computed
            # above, so the confluence ranking carries the live signal state.
            clr_signal = np.nan            # fade score (-z)
            clr_zone = "—"
            clr_signal_type = "Neutral"
            clr_z = np.nan
            clr_side = "—"
            clr_conv = np.nan
            priority_long = np.nan
            priority_short = np.nan
            if clr_results is not None and len(clr_results) > 0:
                clr_row = clr_results[clr_results['SimpleName'] == symbol.replace('.NS', '').replace('^', '')]
                if len(clr_row) > 0:
                    clr_signal = clr_row['Signal'].values[0]
                    clr_zone = clr_row['Zone'].values[0]
                    clr_signal_type = clr_row['SignalType'].values[0]
                    if 'CLR_Z' in clr_row.columns:
                        clr_z = clr_row['CLR_Z'].values[0]
                    if 'Side' in clr_row.columns:
                        clr_side = clr_row['Side'].values[0]
                    if 'Conviction' in clr_row.columns:
                        clr_conv = clr_row['Conviction'].values[0]
                    if 'Priority_Long' in clr_row.columns:
                        priority_long = clr_row['Priority_Long'].values[0]
                    if 'Priority_Short' in clr_row.columns:
                        priority_short = clr_row['Priority_Short'].values[0]

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
                'CLR_Signal': clr_signal,          # fade score (-z)
                'CLR_Z': clr_z,
                'CLR_Zone': clr_zone,
                'CLR_Signal_Type': clr_signal_type,
                'Side': clr_side,
                'Conviction': clr_conv,
                'Priority_Long':  priority_long,
                'Priority_Short': priority_short,
            })

        corr_df = pd.DataFrame(corr_data_list)
        if len(corr_df) == 0:
            st.error("No correlation data could be computed")
            console.item("Error", "Empty correlation DataFrame")
            return None

        corr_df = corr_df.sort_values('Corr_Current', key=abs, ascending=False)

        # ── Confluence score ──────────────────────────────────────────────
        # |Corr| × normalised |fade score|, i.e. how strong this symbol's own CLR
        # close-location reading is relative to the rest of the universe. Normalising by
        # the observed max keeps the score in [0,1] across universes whose |z| spreads
        # differ.
        #
        # A conviction weight (0.5 + 0.5·Conviction) used to be applied on top. It has been
        # removed: conviction is a monotone function of |z| times a universe-level constant,
        # and `pri_norm` is already the normalised |fade score| — i.e. |z|. Multiplying the
        # two counted the same variable twice, compressing the ranking toward whatever the
        # |z| term already said while presenting itself as a second, independent check. The
        # Nifty 50 study confirmed the redundancy directly: standardised, conviction and
        # |z| produce identical regression slopes to four decimals. Conviction remains in
        # the table as a description of the close; it no longer scales the ranking.
        if 'Priority_Long' in corr_df.columns and corr_df['Priority_Long'].notna().any():
            abs_pri = corr_df['Priority_Long'].abs().fillna(0)
            pri_norm = abs_pri / max(abs_pri.max(), 1e-6)        # [0, 1]
            corr_df['Priority_Strength'] = pri_norm
            corr_df['Confluence_Score'] = (corr_df['Corr_Current'].abs() * pri_norm).clip(0.0, 1.0)
            _n_fired = int((corr_df['Side'].isin(['Buy', 'Sell'])).sum()) if 'Side' in corr_df.columns else 0
            console.item("Confluence formula", "|Corr| × normalised |fade score|")
            console.item("Fired signals in universe", f"{_n_fired} symbol(s) past ±{clr.thr:.1f}σ")
        else:
            # Defensive fallback — no screener output to join against, so rank on the
            # correlation alone rather than inventing a signal strength.
            corr_df['Priority_Strength'] = 0.0
            corr_df['Confluence_Score'] = corr_df['Corr_Current'].abs().clip(0.0, 1.0)
            console.item("Confluence formula", "|Corr| only (fallback — no CLR screener data)")

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
            "thr": clr.thr,
            "iclass": clr.iclass,
        }

    except Exception as e:
        st.error(f"Correlation analysis error: {str(e)}")
        console.item("Exception", str(e))
        import traceback
        console.item("Traceback", traceback.format_exc()[:2000])
        return None


# ══════════════════════════════════════════════════════════════════════════════
# CORRELATION MODE — HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── Shared HTML-builder palette helpers ──────────────────────────────────────
# Used by _build_confluence_table_html, _build_signal_table_html,
# _build_narrative_table_html, _build_signal_strength_table_html. Keep these
# in sync — changing one color here propagates to every signal table.
_GREEN  = "#34D399"
_RED    = "#FB7185"

# The indicator's own marker colours, so the app and the TradingView chart read the
# same: green triangle = BUY, yellow/amber diamond = SELL. (sb_v8.pine colorBull /
# colorWarn / colorNeut.)
_CLR_BUY  = "#00E676"
_CLR_SELL = "#FFA726"
_CLR_NEUT = "#787B86"

# 'buy'/'sell' are the canonical side keys. 'long'/'short' are accepted so any
# lingering caller keeps working rather than silently getting the sell palette.
_BUY_SIDES = ('buy', 'long')


def _is_buy_side(side: str) -> bool:
    return str(side).lower() in _BUY_SIDES


def _priority_pct_col(side: str) -> str:
    """The cross-sectional percentile column for a side."""
    return 'Priority_Long_pct' if _is_buy_side(side) else 'Priority_Short_pct'


def _side_palette(side: str) -> dict:
    """Side-keyed accent colors — BUY green triangle / SELL amber diamond."""
    if _is_buy_side(side):
        return {
            "accent_light": _CLR_BUY,
            "border_color": "rgba(0, 230, 118, 0.3)",
            "header_bg":    "rgba(0, 230, 118, 0.13)",
            "mark":         "▲",
            "label":        "BUY",
        }
    return {
        "accent_light": _CLR_SELL,
        "border_color": "rgba(255, 167, 38, 0.3)",
        "header_bg":    "rgba(255, 167, 38, 0.13)",
        "mark":         "◆",
        "label":        "SELL",
    }

def _signed_color(value: float, pos: str = _GREEN, neg: str = _RED) -> str:
    """Green for non-negative, red for negative (or supplied overrides)."""
    return pos if value >= 0 else neg

def _delta_arrow(value: float) -> str:
    """Up arrow for non-negative deltas, down arrow for negative."""
    return "↑" if value >= 0 else "↓"


def _human_vol(value: float, signed: bool = True) -> str:
    """Compact K/M/B/T formatting for large volume-unit numbers (Bar Δ, CVD, CVD Slope).

    1_234_567 → "1.23M", -45_000 → "-45.0K". `signed` prepends an explicit '+' on
    positives so direction reads at a glance in the signed flow columns.
    """
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


def _build_confluence_table_html(df: pd.DataFrame, thr: float = None) -> str:
    """Build the ranked HTML table for confluence setups.

    Displays symbol, correlation, the CLR close-location z + side, flow zone,
    actual/expected/divergence, conviction, and the confluence score. ``thr`` is the run's
    active trigger, so the z cell's colouring agrees with the Side the engine assigned.

    Returns: Complete HTML document string ready for st.components.v1.html().
    """
    thr = eng.CLR_THRESHOLD if thr is None else float(thr)
    table_rows = []
    if df.empty:
        table_rows.append(f"""
        <tr>
            <td colspan="11" style="
                text-align: center;
                color: #374151;
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.72rem;
                letter-spacing: 0.06em;
                padding: 2.25rem 1rem;
            ">— no setups —</td>
        </tr>
        """)
    else:
        for idx, (_, row) in enumerate(df.iterrows(), 1):
            symbol = html.escape(str(row.get('SimpleName', '')))
            corr = float(row.get('Corr_Current', 0))
            zone = html.escape(str(row.get('CLR_Zone', 'Neutral')))
            actual = float(row.get('PctChange', 0))
            expected = float(row.get('Expected_Change', 0))
            divergence = float(row.get('Divergence', 0))
            confluence = float(row.get('Confluence_Score', 0))

            z_cell    = _z_cell(row.get('CLR_Z'), thr)
            side_cell = _side_cell(row.get('Side'))
            conv_cell = _conv_cell(row.get('Conviction'))

            # Note: confluence uses strict > 0 (not >=), so zero is "red" here.
            corr_color = _GREEN if corr > 0 else _RED
            div_color  = _GREEN if divergence > 0 else _RED
            conf_color = "#A78BFA"

            rank_str = f"{idx:02d}"

            table_rows.append(f"""
            <tr>
                <td class="numeric" style="color: #D4A853; font-weight: 700;">{rank_str}</td>
                <td class="symbol">{symbol}</td>
                <td class="numeric" style="color: {corr_color}; font-weight: 600;">{corr:+.3f}</td>
                {z_cell}
                {side_cell}
                <td class="numeric" style="color: #94A3B8; font-size:0.65rem;">{zone}</td>
                <td class="numeric" style="color: #94A3B8;">{actual:+.2f}%</td>
                <td class="numeric" style="color: #94A3B8;">{expected:+.2f}%</td>
                <td class="numeric" style="color: {div_color}; font-weight: 600;">{divergence:+.2f}%</td>
                {conv_cell}
                <td class="numeric" style="color:{conf_color}; font-weight:600;">{confluence:.2f}</td>
            </tr>
            """)

    # Build full HTML
    table_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'IBM Plex Mono', monospace;
            background: transparent;
            color: #F1F5F9;
            padding: 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        thead th {{
            background: transparent;
            color: #4B5563;
            font-size: 0.62rem !important;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            padding: 0.5rem 0.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            text-align: left;
        }}
        thead th.numeric {{ text-align: right; }}
        tbody tr {{
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
        }}
        tbody tr:hover {{ background: rgba(139, 92, 246, 0.05); }}
        tbody td {{
            padding: 0.5rem 0.5rem;
            color: #F1F5F9;
            font-size: 0.72rem !important;
        }}
        tbody td.symbol {{
            font-weight: 700;
            font-size: 0.75rem;
            letter-spacing: 0.02em;
        }}
        tbody td.numeric {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
    </style>
    </head>
    <body>
    <table>
        <thead>
            <tr>
                <th class="numeric">Rank</th>
                <th>Symbol</th>
                <th class="numeric">Corr</th>
                <th class="numeric" title="Close-location z — the CLR core measure">Close-Loc z</th>
                <th class="numeric" title="▲ BUY (weak close) · ◆ SELL (strong close) · — inside the band">Side</th>
                <th class="numeric" title="Cumulative-delta flow zone — context only">Zone</th>
                <th class="numeric" title="Symbol's price change on the analysis date">Actual %</th>
                <th class="numeric" title="Expected move = target return × beta (rolling correlation × vol ratio over the lookback)">Expected %</th>
                <th class="numeric" title="Divergence = Actual − Expected (positive = outperforming expectation)">Div %</th>
                <th class="numeric" title="How far |z| sits between the firing threshold and the largest reading this bar's own window could produce, × the cost gate. Cap-relative, so two bars at the same z can differ. A description, not a validated forecast. Not a probability.">Conv</th>
                <th class="numeric" title="Confluence = |Correlation| × normalised |fade score|">Confluence</th>
            </tr>
        </thead>
        <tbody>
            {"".join(table_rows)}
        </tbody>
    </table>
    </body>
    </html>
    """
    return table_html


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
    thr = float(corr_data.get("thr", eng.CLR_THRESHOLD))
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
        col_pos, col_neg = st.columns(2)

        with col_pos:
            ui.render_section_header("Top Positively Correlated", icon="trending", accent="emerald")
            pos_corr = corr_df[corr_df['Corr_Current'] > 0].head(7)
            for _, row in pos_corr.iterrows():
                trend_arrow = "↑" if row['Corr_Trend'] > 0.05 else "↓" if row['Corr_Trend'] < -0.05 else "→"
                corr_val = row['Corr_Current']
                tier_class = row['Corr_Tier'].lower().replace("+", "-pos").replace("-", "-neg")

                st.markdown(f"""
                <div class="corr-row">
                    <div>
                        <div class="name">{row['SimpleName']}</div>
                        <div class="sub">{row['PctChange']:+.2f}% | Expected: {row['Expected_Change']:+.2f}%</div>
                    </div>
                    <div style="display:flex; gap:8px; align-items:center;">
                        <span class="corr-tier {tier_class}">{corr_val:.3f}</span>
                        <div class="corr-bar-track">
                            <div class="corr-bar-center"></div>
                            <div class="corr-bar-fill pos" style="width:{abs(corr_val)*50}px;"></div>
                        </div>
                        <span style="font-size:0.75rem; color:var(--ink-secondary);">{trend_arrow}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_neg:
            ui.render_section_header("Top Inversely Correlated", icon="trending", accent="rose")
            neg_corr = corr_df[corr_df['Corr_Current'] < 0].head(7)
            for _, row in neg_corr.iterrows():
                trend_arrow = "↑" if row['Corr_Trend'] > 0.05 else "↓" if row['Corr_Trend'] < -0.05 else "→"
                corr_val = row['Corr_Current']
                tier_class = row['Corr_Tier'].lower().replace("+", "-pos").replace("-", "-neg")

                st.markdown(f"""
                <div class="corr-row">
                    <div>
                        <div class="name">{row['SimpleName']}</div>
                        <div class="sub">{row['PctChange']:+.2f}% | Expected: {row['Expected_Change']:+.2f}%</div>
                    </div>
                    <div style="display:flex; gap:8px; align-items:center;">
                        <span class="corr-tier {tier_class}">{corr_val:.3f}</span>
                        <div class="corr-bar-track">
                            <div class="corr-bar-center"></div>
                            <div class="corr-bar-fill neg" style="width:{abs(corr_val)*50}px;"></div>
                        </div>
                        <span style="font-size:0.75rem; color:var(--ink-secondary);">{trend_arrow}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2: TRADE INTELLIGENCE
    # ═══════════════════════════════════════════════════════════════════════════
    with tab2:
        ui.render_section_header(
            "Confluence Setups",
            f"Confluence: Correlation × CLR close-location strength · ±{thr:.1f}σ · {iclass}",
            icon="zap",
            accent="cyan"
        )

        # How to read this tab - styled as interpretation card
        st.markdown("""
        <div style="background:rgba(56,189,248,0.08); border:1px solid rgba(56,189,248,0.2);
                    border-radius:8px; padding:1rem; margin:1.5rem 0; font-family:var(--data); font-size:0.75rem;">
            <div style="color:#38BDF8; font-weight:700; text-transform:uppercase; margin-bottom:0.75rem; letter-spacing:0.06em;">
                How to Read
            </div>
            <div style="color:#F1F5F9; line-height:1.6;">
                Each setup type is ranked by <span style="color:#38BDF8; font-weight:600;">Confluence Score</span> (0-1)
                = |Correlation| × normalised |fade score|. Highest rank = strongest
                overlap between the correlation relationship and a live CLR reading. Look for:
                <span style="font-weight:600;">(1) Score &gt;0.7</span>,
                <span style="font-weight:600;">(2) |Div %| &gt;3%</span>,
                <span style="font-weight:600;">(3) a fired Side (▲ / ◆), not a blank one</span>
            </div>
            <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:0.5rem; margin-top:0.75rem;">
                <div style="font-family:var(--data); font-size:0.65rem; color:var(--ink-secondary);">
                    <span style="color:#38BDF8; font-weight:600;">Corr</span> — Correlation strength
                </div>
                <div style="font-family:var(--data); font-size:0.65rem; color:var(--ink-secondary);">
                    <span style="color:#38BDF8; font-weight:600;">Close-Loc z</span> — where the bar closed in its range
                </div>
                <div style="font-family:var(--data); font-size:0.65rem; color:var(--ink-secondary);">
                    <span style="color:#38BDF8; font-weight:600;">Side</span> — ▲ BUY / ◆ SELL / — no fire
                </div>
                <div style="font-family:var(--data); font-size:0.65rem; color:var(--ink-secondary);">
                    <span style="color:#38BDF8; font-weight:600;">Div %</span> — Actual vs Expected
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

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
            zone = row['CLR_Zone']

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

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Render each setup type as a section
        setup_configs = [
            {
                "name": "LAGGARD",
                "title": "Laggard Setups",
                "description": "High corr + oversold + underperforming — expect catch-up rally",
                "color": "#34D399",
                "bg_color": "rgba(45, 212, 168, 0.1)",
                "border_color": "rgba(45, 212, 168, 0.25)"
            },
            {
                "name": "RUNAWAY",
                "title": "Runaway Setups",
                "description": "High corr + overbought + overextended — expect pullback",
                "color": "#FB7185",
                "bg_color": "rgba(232, 85, 90, 0.1)",
                "border_color": "rgba(232, 85, 90, 0.25)"
            },
            {
                "name": "CONVERGING",
                "title": "Converging Setups",
                "description": "Low corr or normalizing — expect tightening after divergence",
                "color": "#D4A853",
                "bg_color": "rgba(212, 168, 83, 0.1)",
                "border_color": "rgba(212, 168, 83, 0.25)"
            },
            {
                "name": "CONTRA",
                "title": "Contra Setups",
                "description": "Strong negative corr + overbought — expect rally vs target decline",
                "color": "#A78BFA",
                "bg_color": "rgba(139, 92, 246, 0.1)",
                "border_color": "rgba(139, 92, 246, 0.25)"
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
                    <span style="font-family:var(--display); font-size:0.62rem; font-weight:700;
                                 letter-spacing:0.12em; text-transform:uppercase; color:{config['color']};
                                 padding:0.18rem 0.5rem; background:{config['bg_color']};
                                 border:1px solid {config['border_color']}; border-radius:4px;">
                        {config['name']}</span>
                    <span style="font-family:var(--display); font-size:1rem; font-weight:700;
                                 color:#F1F5F9; letter-spacing:0.04em;">{config['title']}</span>
                    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#6B7280;">
                        {config['description']}</span>
                    <span style="margin-left:auto; font-family:'IBM Plex Mono',monospace; font-size:0.72rem;
                                 color:{config['color']};">→ {len(setup_data)}</span>
                </div>
                """, unsafe_allow_html=True)

                # Interpretation card
                interp = setup_interpretation[config['name']]
                st.markdown(f"""
                <div style="background:{config['bg_color']}; border:1px solid {config['border_color']};
                            border-radius:8px; padding:0.75rem 1rem; margin-bottom:1rem; font-family:var(--data); font-size:0.75rem;">
                    <div style="display:grid; grid-template-columns:auto 1fr; gap:0.5rem 1rem; color:#F1F5F9;">
                        <span style="color:{config['color']}; font-weight:700; text-transform:uppercase;">Action</span>
                        <span>{interp['action']}</span>
                        <span style="color:{config['color']}; font-weight:700; text-transform:uppercase;">Rationale</span>
                        <span>{interp['rationale']}</span>
                        <span style="color:{config['color']}; font-weight:700; text-transform:uppercase;">Validate</span>
                        <span>{interp['validate']}</span>
                        <span style="color:#FB7185; font-weight:700; text-transform:uppercase;">⚠ Risk</span>
                        <span style="color:#FB7185;">{interp['risk']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Display as two-column table
                col_left, col_right = st.columns(2)
                with col_left:
                    st.markdown(f"""<p style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem; font-weight:600;
                                   text-transform:uppercase; letter-spacing:0.1em; color:{config['color']};
                                   margin:0 0 0.4rem 0; display:flex; align-items:center; gap:0.35rem;">
                        Top Confluence</p>""", unsafe_allow_html=True)
                    top_half = setup_data.head(5)
                    if len(top_half) > 0:
                        st.components.v1.html(_build_confluence_table_html(top_half, thr=thr), height=100 + len(top_half) * 48)
                with col_right:
                    st.markdown(f"""<p style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem; font-weight:600;
                                   text-transform:uppercase; letter-spacing:0.1em; color:{config['color']};
                                   margin:0 0 0.4rem 0; display:flex; align-items:center; gap:0.35rem;">
                        Also Considered</p>""", unsafe_allow_html=True)
                    bottom_half = setup_data.iloc[5:10]
                    if len(bottom_half) > 0:
                        st.components.v1.html(_build_confluence_table_html(bottom_half, thr=thr), height=100 + len(bottom_half) * 48)
                    else:
                        st.info("No additional setups")

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
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_rows['Corr_Current'].values.reshape(-1, 1),
                x=["Correlation"],
                y=heatmap_rows['SimpleName'].values,
                colorscale=[[0, "#E8555A"], [0.5, "#1a2133"], [1, "#2DD4A8"]],
                zmid=0,
                zmin=-1,
                zmax=1,
                text=heatmap_rows['Corr_Current'].values.reshape(-1, 1),
                texttemplate='%{text:.2f}',
                textfont={"size": 8, "color": "#94A3B8"},
                colorbar=dict(title="Corr", thickness=15, len=0.7)
            ))
            apply_chart_theme(fig)
            fig.update_layout(height=600, margin=dict(l=150, r=50, t=50, b=50))
            st.plotly_chart(fig, width='stretch', key='chart_corr_0')
        else:
            st.info("No correlation data available for heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS FOR TAB RENDERING
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_num(v, fmt="{:+.2f}", dash="—"):
    """Format a possibly-NaN/None number, falling back to an em dash."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return dash
    return dash if not np.isfinite(f) else fmt.format(f)


# Close-location z bands. Colour follows the CLR semantics, NOT price direction:
# a deeply NEGATIVE z (weak close) is the bullish read, so it renders green.
def _z_cell(z, thr: float = None) -> str:
    """Render a close-location z-score cell, coloured by CLR meaning."""
    thr = eng.CLR_THRESHOLD if thr is None else float(thr)
    try:
        f = float(z)
    except (TypeError, ValueError):
        f = float('nan')
    if not np.isfinite(f):
        return '<td class="numeric" style="color:#4B5563;">—</td>'
    if f <= -thr:
        col, note = '#00E676', 'weak close — BUY (fade up)'
    elif f >= thr:
        col, note = '#FFA726', 'strong close — SELL (holdout-unconfirmed side)'
    else:
        col, note = '#787B86', f'inside ±{thr:.1f}σ — context only'
    title = f'close-location z {f:+.2f} · {note}'
    return (f'<td class="numeric" style="color:{col}; font-weight:700;" '
            f'title="{html.escape(title)}">{f:+.2f}</td>')


def _conv_cell(conv) -> str:
    """Render the conviction cell — |z|'s position in its attainable range × the cost gate."""
    try:
        c = float(conv)
    except (TypeError, ValueError):
        c = float('nan')
    if not np.isfinite(c):
        return '<td class="numeric" style="color:#4B5563;">—</td>'
    if   c >= 0.70: col = '#2DD4A8'
    elif c >= 0.55: col = '#A3E635'
    elif c >= 0.40: col = '#D4A853'
    else:           col = '#FB923C'
    title = (f'Conviction {c*100:.0f}% — how far |z| sits between the firing threshold and the '
             f'largest reading this bar\'s own window could produce, x the cost gate. '
             f'Cap-relative position, so two bars at the same z can differ. A description of the close, '
             f'not a validated forecast, and not a probability.')
    return (f'<td class="numeric" style="color:{col}; font-weight:700;" '
            f'title="{html.escape(title)}">{c*100:.0f}%</td>')


def _side_cell(side) -> str:
    """Render the Side cell with the indicator's own marks (▲ buy / ◆ sell)."""
    s = str(side or '—')
    if s == 'Buy':
        return ('<td class="numeric" style="color:#00E676; font-weight:700; font-size:0.68rem;" '
                'title="green triangle — weak close, fade long">▲ BUY</td>')
    if s == 'Sell':
        return ('<td class="numeric" style="color:#FFA726; font-weight:700; font-size:0.68rem;" '
                'title="yellow diamond — strong close. The Pine labels this side CAUTION: it did '
                'not confirm out of sample.">◆ SELL</td>')
    return '<td class="numeric" style="color:#4B5563; font-size:0.68rem;">—</td>'


def _hold_cell(age, horizon, direction) -> str:
    """Render the hold window as "day N/H" — how far into the measured 5-10 bar window."""
    try:
        a = float(age)
        h = int(horizon)
    except (TypeError, ValueError):
        return '<td class="numeric" style="color:#4B5563;">—</td>'
    if not np.isfinite(a) or h <= 0:
        return '<td class="numeric" style="color:#4B5563;">—</td>'
    n = int(a)
    d = int(direction or 0)
    col = '#00E676' if d > 0 else '#FFA726' if d < 0 else '#787B86'
    if n > h:
        return ('<td class="numeric" style="color:#787B86; font-size:0.65rem;" '
                f'title="window expired — the measured edge does not extend past {h} bars">expired</td>')
    frac = 1.0 - (n / max(h, 1))
    title = (f'day {n} of {h} in the hold window · {frac*100:.0f}% of the measured horizon left. '
             f'Entry was the open after the signal bar.')
    return (f'<td class="numeric" style="color:{col}; font-weight:600; font-size:0.65rem;" '
            f'title="{html.escape(title)}">{n}/{h}</td>')


def _entry_status(row, offset: int):
    """Has price already run since the signal fired — i.e. is the entry now late?

    Directional move from the fire bar to the snapshot bar, normalised by the symbol's
    own recent return volatility x sqrt(bars elapsed) so the bands are asset-agnostic
    (sigma units). Returns (label, color, title).
    """
    if offset == 0:
        return ('Now', '#94a3b8', 'fresh — fired on the snapshot bar')
    closes = row.get('Close_Hist')
    if not isinstance(closes, (list, tuple)) or offset >= len(closes):
        return ('—', '#4B5563', '')
    fire_close, now_close = closes[offset], closes[0]
    if not (pd.notna(fire_close) and pd.notna(now_close) and float(fire_close) > 0):
        return ('—', '#4B5563', '')
    # Direction of the trade the signal implied, read from the z that fired it.
    zs = row.get('Z_Hist')
    fire_z = zs[offset] if isinstance(zs, (list, tuple)) and offset < len(zs) else float('nan')
    if not (pd.notna(fire_z) and np.isfinite(float(fire_z))):
        return ('—', '#4B5563', '')
    side_sign = 1.0 if float(fire_z) < 0 else -1.0      # weak close → long, strong → sell
    dm = (float(now_close) - float(fire_close)) / float(fire_close) * side_sign
    rv = row.get('RetVol20')
    scale = (float(rv) * (offset ** 0.5)) if (rv is not None and pd.notna(rv) and float(rv) > 0) else None
    if scale and scale > 0:
        sig = dm / scale
        title = f'{dm*100:+.1f}% since the fire bar, in the signal\'s direction ({sig:+.1f} sigma)'
        if sig <= -1.0: return ('Adverse', '#E8555A', title)
        if sig >= 1.5:  return ('Extended', '#FB923C', title)
        if sig >= 0.5:  return ('Running', '#5EBFA8', title)
        return ('Open', '#2DD4A8', title)
    title = f'{dm*100:+.1f}% since the fire bar, in the signal\'s direction'
    if dm <= -0.03: return ('Adverse', '#E8555A', title)
    if dm >= 0.06:  return ('Extended', '#FB923C', title)
    if dm >= 0.02:  return ('Running', '#5EBFA8', title)
    return ('Open', '#2DD4A8', title)


def _status_cell(status) -> str:
    """Render a (label, color, title) status tuple as a small table cell."""
    label, color, title = (status if isinstance(status, (tuple, list)) and len(status) == 3
                           else ('—', '#4B5563', ''))
    _t = html.escape(str(title)) if title else ''
    return (f'<td class="numeric" style="color:{color}; font-weight:700; font-size:0.62rem;" '
            f'title="{_t}">{html.escape(str(label))}</td>')


def _bucket_signals_by_age(results_df: pd.DataFrame, side: str = 'buy', timeframe: str = 'Daily') -> tuple:
    """Bucket fired CLR signals by age (Today, 1d, 2d, 3d, within 5d) for the timeline.

    side: 'buy' (green triangle, BUY_* columns) or 'sell' (yellow diamond, SELL_*).
    timeframe: 'Daily' or 'Weekly' — determines the age label names.

    A symbol appears in the NEWEST bucket it fired in and nowhere else. Each row carries
    the z that actually fired it (``_fire_z``, read from Z_Hist at that offset) plus an
    entry-exhaustion read, so an aged signal reports its own bar rather than today's.
    """
    prefix = 'BUY' if side == 'buy' else 'SELL'
    target_indicator = "●"

    if timeframe == 'Weekly':
        age_labels = ["This Week", "1 Week Ago", "2 Weeks Ago", "3 Weeks Ago", "Within 5 Weeks"]
    else:
        age_labels = ["Today", "1 Day Ago", "2 Days Ago", "3 Days Ago", "Within 5 Days"]

    buckets = {label: [] for label in age_labels}
    col_map = {
        age_labels[0]: f"{prefix}_Today",
        age_labels[1]: f"{prefix}_1d",
        age_labels[2]: f"{prefix}_2d",
        age_labels[3]: f"{prefix}_3d",
        age_labels[4]: f"{prefix}_5d",
    }
    seen = set()

    for _offset, age in enumerate(buckets.keys()):
        col = col_map[age]
        if col not in results_df.columns:
            continue
        subset = results_df[(results_df[col] == target_indicator) & (~results_df['Symbol'].isin(seen))]
        for _, r in subset.iterrows():
            sym = r['Symbol']
            r = r.copy()
            # The z at the bar that fired, not at the snapshot bar. The buckets are walked
            # newest-first and `seen` blocks re-listing, so a symbol reaching the last
            # bucket did NOT fire at offsets 0-3 — and since the *_5d column is an .any()
            # over offsets 0-4, offset 4 is then the fire bar exactly. Every offset is
            # therefore knowable; the snapshot z is only a fallback for a missing window.
            _zs = r.get('Z_Hist')
            _fz = float('nan')
            if isinstance(_zs, (list, tuple)) and _offset < len(_zs):
                _fz = _zs[_offset]
            if not (pd.notna(_fz) and np.isfinite(float(_fz))):
                _fz = r.get('CLR_Z', float('nan'))
            r['_fire_z'] = _fz
            r['_age_offset'] = _offset
            r['_entry'] = _entry_status(r, _offset)
            buckets[age].append(r)
            seen.add(sym)

    # Per-bucket stats
    stats = {}
    for age, rows in buckets.items():
        if rows:
            fire_zs = [float(r['_fire_z']) for r in rows
                       if pd.notna(r['_fire_z']) and np.isfinite(float(r['_fire_z']))]
            stats[age] = {
                'count': len(rows),
                'avg_signal': float(np.mean([-z for z in fire_zs])) if fire_zs else 0.0,
                'avg_abs_z': float(np.mean([abs(z) for z in fire_zs])) if fire_zs else 0.0,
                'avg_pct_change': float(np.mean([r.get('PctChange', 0) or 0 for r in rows])),
                'rows': rows,
            }
        else:
            stats[age] = {'count': 0, 'avg_signal': 0.0, 'avg_abs_z': 0.0,
                          'avg_pct_change': 0.0, 'rows': []}

    # Trend: are the newest fires more extreme than the older ones? |z| is the honest
    # scale here — a fresh batch at 2.4 sigma is a stronger tape than one at 1.6.
    newest_label = age_labels[0]
    older_labels = age_labels[1:]
    newest_avg = stats[newest_label]['avg_abs_z'] if stats[newest_label]['count'] > 0 else 0.0
    _older = [stats[a]['avg_abs_z'] for a in older_labels if stats[a]['count'] > 0]
    older_avg = float(np.mean(_older)) if _older else 0.0

    # |z| bucket means live on a ~1.5-3.0 scale, so 0.25 sigma is a meaningful shift.
    _TREND_EPS = 0.25
    if newest_avg > older_avg + _TREND_EPS:
        trend = f"{SVGS['UP']} Strengthening"
        trend_color = "#2DD4A8"
    elif newest_avg < older_avg - _TREND_EPS:
        trend = f"{SVGS['DOWN']} Weakening"
        trend_color = "#E8555A"
    else:
        trend = "— Stable"
        trend_color = "#D4A853"

    return buckets, stats, trend, trend_color


def _build_signal_table_html(stats: dict, side: str = 'buy', timeframe: str = 'Daily',
                             thr: float = None) -> str:
    """Build the age-grouped HTML table of fired CLR signals, with section headers."""
    _pal = _side_palette(side)
    accent_light = _pal["accent_light"]
    border_color = _pal["border_color"]
    header_bg    = _pal["header_bg"]
    _mark, _label = _pal["mark"], _pal["label"]
    thr = eng.CLR_THRESHOLD if thr is None else float(thr)
    _NCOLS = 11

    table_rows = []
    if timeframe == 'Weekly':
        age_order = ["This Week", "1 Week Ago", "2 Weeks Ago", "3 Weeks Ago", "Within 5 Weeks"]
    else:
        age_order = ["Today", "1 Day Ago", "2 Days Ago", "3 Days Ago", "Within 5 Days"]

    for age in age_order:
        if stats[age]['count'] == 0:
            continue

        # Section header for this age group
        avg_abs_z = stats[age].get('avg_abs_z', 0)
        avg_pct   = stats[age].get('avg_pct_change', 0)
        count     = stats[age]['count']
        table_rows.append(f"""
        <tr style="background: {header_bg}; border-bottom: 2px solid {border_color};">
            <td colspan="{_NCOLS}" style="padding: 0.75rem 1rem; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.8rem !important; font-weight: 700; color: {accent_light}; text-transform: uppercase; letter-spacing: 0.05em;">
                {_mark} {age} · {count} {_label} signal{'s' if count != 1 else ''} · Avg |z|: {avg_abs_z:.2f}σ · Avg %: {avg_pct:+.1f}
            </td>
        </tr>
        """)

        # Data rows for this age group
        for row in stats[age]['rows']:
            symbol = html.escape(str(row.get('DisplayName', row.get('Symbol', ''))))
            price = float(row.get('Price', 0) or 0)
            pct_change = float(row.get('PctChange', 0) or 0)
            cvd_slope = float(row.get('CVD_Slope', 0) or 0)
            abs_strength = float(row.get('Abs_Strength', 0) or 0)
            abs_color = _signed_color(abs_strength - 1.0, pos="#fbbf24", neg="#38bdf8")  # >1× = amber
            zone = html.escape(str(row.get('Zone', '—')))

            pct_color        = _signed_color(pct_change)
            cvd_slope_color  = _signed_color(cvd_slope, pos="#4a9eff", neg="#D4A853")
            cvd_slope_arrow  = _delta_arrow(cvd_slope)

            # The z that FIRED this signal (its own bar), and the fade score it implies.
            _fz = row.get('_fire_z', row.get('CLR_Z'))
            z_cell = _z_cell(_fz, thr)
            _fade = (-float(_fz) if (pd.notna(_fz) and np.isfinite(float(_fz))) else float('nan'))
            fade_txt = _fmt_num(_fade)

            conv_cell  = _conv_cell(row.get('Conviction'))
            hold_cell  = _hold_cell(row.get('CLR_Hold_Age'), row.get('CLR_Horizon', eng.CLR_HORIZON),
                                    row.get('CLR_Hold_Dir'))
            entry_cell = _status_cell(row.get('_entry', ('—', '#4B5563', '')))

            table_rows.append(f"""
            <tr>
                <td class="symbol">{symbol}</td>
                <td class="numeric currency">{price:,.2f}</td>
                <td class="numeric" style="color: {pct_color}; font-weight: 600;">{pct_change:+.2f}%</td>
                <td class="numeric" style="color: {accent_light}; font-weight: 600;">{fade_txt}</td>
                {z_cell}
                {conv_cell}
                {hold_cell}
                {entry_cell}
                <td class="numeric" style="color: #94A3B8; font-size: 0.65rem;">{zone}</td>
                <td class="numeric" style="color: {cvd_slope_color}; font-size: 0.65rem; font-weight: 600;">{cvd_slope_arrow}{_human_vol(abs(cvd_slope), signed=False)}</td>
                <td class="numeric" style="color: {abs_color}; font-weight: 600;">{abs_strength:.2f}×</td>
            </tr>
            """)

    if not table_rows:
        table_rows.append(f"""
        <tr>
            <td colspan="{_NCOLS}" style="text-align:center; color:#374151; font-family:'IBM Plex Mono',monospace;
                font-size:0.72rem; letter-spacing:0.06em; padding:2.25rem 1rem;">
                — no {_label} signals in the last 5 bars —
            </td>
        </tr>""")

    table_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        * {{
            -webkit-text-size-adjust: 100%;
            -moz-text-size-adjust: 100%;
            text-size-adjust: 100%;
        }}
        body {{
            font-family: 'IBM Plex Mono', monospace;
            background: transparent;
            color: #F1F5F9;
            padding: 0.5rem 0.5rem 1.5rem 0.5rem;
            font-size: 16px !important;
        }}
        @media (max-width: 768px) {{
            body {{
                font-size: 16px !important;
            }}
        }}
        .portfolio-table {{
            width: 100%;
            border-radius: 10px;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            border: 1px solid rgba(255, 255, 255, 0.05);
            background: linear-gradient(145deg, rgba(17, 24, 39, 0.45) 0%, rgba(17, 24, 39, 0.4) 100%);
        }}
        .portfolio-table table {{
            width: 100%;
            min-width: 480px;
            border-collapse: collapse;
        }}
        .portfolio-table thead th {{
            background: linear-gradient(180deg, rgba(10, 14, 23, 0.95) 0%, rgba(10, 14, 23, 0.85) 100%);
            color: #4B5563;
            font-size: 0.62rem !important;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            padding: 0.75rem 0.75rem;
            border-bottom: 2px solid {border_color};
            text-align: left;
        }}
        .portfolio-table thead th.numeric {{ text-align: right; }}
        .portfolio-table tbody tr {{
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
            transition: background 0.2s ease;
        }}
        .portfolio-table tbody tr:nth-child(odd) {{ background: rgba(255, 255, 255, 0.01); }}
        .portfolio-table tbody tr:nth-child(even) {{ background: rgba(255, 255, 255, 0.005); }}
        .portfolio-table tbody tr:hover {{ background: {border_color}; }}
        .portfolio-table tbody td {{
            padding: 0.75rem 0.75rem;
            color: #F1F5F9;
            vertical-align: middle;
            font-size: 0.75rem !important;
        }}
        .portfolio-table tbody td.symbol {{
            font-weight: 700;
            font-size: 0.78rem;
            letter-spacing: 0.02em;
            font-family: 'Space Grotesk', sans-serif;
        }}
        .portfolio-table tbody td.numeric {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
    </style>
    </head>
    <body>
    <div class="portfolio-table">
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th class="numeric">Price</th>
                    <th class="numeric">% Change</th>
                    <th class="numeric" title="Fade score = −z at the bar that fired. Positive = bullish.">Fade</th>
                    <th class="numeric" title="Close-location z at the bar that fired (CLR core measure)">Close-Loc z</th>
                    <th class="numeric" title="How far |z| sits between the firing threshold and the largest reading this bar's own window could produce, × the cost gate. Cap-relative, so two bars at the same z can differ. A description, not a validated forecast. Not a probability.">Conv</th>
                    <th class="numeric" title="Bars into the measured hold window (entry was the open after the signal bar)">Hold</th>
                    <th class="numeric" title="Has price already run in the signal's direction since it fired (σ units)?">Entry</th>
                    <th class="numeric" title="Cumulative-delta flow zone — context only, not a signal input">Zone</th>
                    <th class="numeric">CVD Slope</th>
                    <th class="numeric">Absorp</th>
                </tr>
            </thead>
            <tbody>
                {"".join(table_rows)}
            </tbody>
        </table>
    </div>
    </body>
    </html>
    """
    return table_html

def _build_narrative_table_html(df: pd.DataFrame, side: str = 'buy', thr: float = None) -> str:
    """Build the full-universe HTML table for Pulse Narrative mode (every symbol)."""
    _pal = _side_palette(side)
    border_color = _pal["border_color"]
    thr = eng.CLR_THRESHOLD if thr is None else float(thr)
    _NCOLS = 11

    table_rows = []
    if df.empty:
        table_rows.append(f"""
        <tr>
            <td colspan="{_NCOLS}" style="text-align:center; color:#374151; font-family:'IBM Plex Mono',monospace;
                font-size:0.72rem; letter-spacing:0.06em; padding:2.25rem 1rem;">
                — no data available —
            </td>
        </tr>""")
    else:
        for _, row in df.iterrows():
            symbol = html.escape(str(row.get('DisplayName', row.get('Symbol', ''))))
            price = float(row.get('Price', 0) or 0)
            pct_change = float(row.get('PctChange', 0) or 0)
            bar_delta = float(row.get('Bar_Delta', 0) or 0)
            cvd_slope = float(row.get('CVD_Slope', 0) or 0)
            abs_strength = float(row.get('Abs_Strength', 0) or 0)
            abs_color = _signed_color(abs_strength - 1.0, pos="#fbbf24", neg="#38bdf8")

            pct_color       = _signed_color(pct_change)
            cvd_slope_color = _signed_color(cvd_slope, pos="#4a9eff", neg="#D4A853")
            cvd_slope_arrow = _delta_arrow(cvd_slope)

            fade_txt   = _fmt_num(row.get('Signal'))
            z_cell     = _z_cell(row.get('CLR_Z'), thr)
            side_cell  = _side_cell(row.get('Side'))
            conv_cell  = _conv_cell(row.get('Conviction'))
            hold_cell  = _hold_cell(row.get('CLR_Hold_Age'), row.get('CLR_Horizon', eng.CLR_HORIZON),
                                    row.get('CLR_Hold_Dir'))

            table_rows.append(f"""
            <tr>
                <td class="symbol" style="color: #F1F5F9;">{symbol}</td>
                <td class="numeric currency">{price:,.2f}</td>
                <td class="numeric" style="color: {pct_color}; font-weight: 600;">{pct_change:+.2f}%</td>
                <td class="numeric" style="color: #60A5FA; font-weight: 600;">{fade_txt}</td>
                {z_cell}
                {side_cell}
                {conv_cell}
                {hold_cell}
                <td class="numeric" style="color: #D4A853; font-weight: 600;">{_human_vol(bar_delta)}</td>
                <td class="numeric" style="color: {cvd_slope_color}; font-size: 0.65rem; font-weight: 600;">{cvd_slope_arrow}{_human_vol(abs(cvd_slope), signed=False)}</td>
                <td class="numeric" style="color: {abs_color}; font-weight: 600;">{abs_strength:.2f}×</td>
            </tr>
            """)

    table_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'IBM Plex Mono', monospace;
            background: transparent;
            color: #F1F5F9;
            padding: 0.5rem;
            font-size: 14px;
        }}
        .portfolio-table {{
            width: 100%;
            border-radius: 8px;
            overflow-x: auto;
            border: 1px solid rgba(255, 255, 255, 0.05);
            background: rgba(10, 14, 23, 0.4);
        }}
        .portfolio-table table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .portfolio-table thead th {{
            background: rgba(15, 23, 42, 0.9);
            color: #94A3B8;
            font-size: 0.65rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            padding: 0.75rem;
            border-bottom: 2px solid {border_color};
            text-align: left;
        }}
        .portfolio-table thead th.numeric {{ text-align: right; }}
        .portfolio-table tbody tr {{
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
        }}
        .portfolio-table tbody tr:hover {{ background: rgba(255, 255, 255, 0.04); }}
        .portfolio-table tbody td {{
            padding: 0.85rem 0.75rem;
            vertical-align: middle;
            font-size: 0.75rem;
            white-space: nowrap;
        }}
        .portfolio-table tbody td.symbol {{
            font-weight: 700;
            font-family: 'Space Grotesk', sans-serif;
        }}
        .portfolio-table tbody td.numeric {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
    </style>
    </head>
    <body>
    <div class="portfolio-table">
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th class="numeric">Price</th>
                    <th class="numeric">% Change</th>
                    <th class="numeric" title="Fade score = −z. Positive = bullish (a weak close).">Fade</th>
                    <th class="numeric" title="Close-location z — the CLR core measure">Close-Loc z</th>
                    <th class="numeric" title="▲ BUY past −thr · ◆ SELL past +thr · — inside the band (context only)">Side</th>
                    <th class="numeric" title="How far |z| sits between the firing threshold and the largest reading this bar's own window could produce, × the cost gate. Cap-relative, so two bars at the same z can differ. A description, not a validated forecast. Not a probability.">Conv</th>
                    <th class="numeric" title="Bars into the measured hold window">Hold</th>
                    <th class="numeric">Bar Δ</th>
                    <th class="numeric">CVD Slope</th>
                    <th class="numeric">Absorp</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
    </div>
    </body>
    </html>
    """
    return table_html



def _build_signal_strength_table_html(df: pd.DataFrame, side: str = 'buy', thr: float = None) -> str:
    """Build the ranked HTML table of CLR candidates for one side.

    Ranks by the cross-sectional fade-score percentile: for 'buy' the weakest closes in
    the universe come first, for 'sell' the strongest. Rows below the ±thr trigger still
    appear — the table is the full ordering — but their Side reads '—' (context only).

    Returns: Complete HTML document string ready for st.components.v1.html().
    """
    _pal = _side_palette(side)
    accent_light = _pal["accent_light"]
    border_color = _pal["border_color"]
    _is_buy = _is_buy_side(side)
    _pct_col = _priority_pct_col(side)
    thr = eng.CLR_THRESHOLD if thr is None else float(thr)
    _NCOLS = 13

    table_rows = []
    if df.empty:
        table_rows.append(f"""
        <tr>
            <td colspan="{_NCOLS}" style="
                text-align: center;
                color: #374151;
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.72rem;
                letter-spacing: 0.06em;
                padding: 2.25rem 1rem;
            ">— no symbols to rank —</td>
        </tr>
        """)
    else:
        for idx, (_, row) in enumerate(df.iterrows(), 1):
            symbol = html.escape(str(row.get('DisplayName', row.get('Symbol', ''))))
            price = float(row.get('Price', 0) or 0)
            pct_change = float(row.get('PctChange', 0) or 0)
            bar_delta = float(row.get('Bar_Delta', 0) or 0)
            cvd_slope = float(row.get('CVD_Slope', 0) or 0)

            rank_str = f"{idx:02d}"
            pct_color       = _signed_color(pct_change)
            cvd_slope_color = _signed_color(cvd_slope)
            cvd_slope_arrow = _delta_arrow(cvd_slope)

            pct_rank = float(row.get(_pct_col, 0) or 0)
            hmm_bull = float(row.get('HMM_Bull', 0.5) or 0.5)
            hmm_bear = float(row.get('HMM_Bear', 0.5) or 0.5)
            vol_reg  = str(row.get('Vol_Regime', 'NORMAL'))

            # Regime risk context — displayed beside the signal, never inside it.
            regime_tag = "NEUTRAL"
            regime_color = "#94a3b8"
            if _is_buy:
                if hmm_bull > 0.7: regime_tag, regime_color = "BULL", _GREEN
                elif hmm_bull < 0.3: regime_tag, regime_color = "BEAR", _RED
            else:
                if hmm_bear > 0.7: regime_tag, regime_color = "BEAR", _RED
                elif hmm_bear < 0.3: regime_tag, regime_color = "BULL", _GREEN

            vol_color = {"LOW": "#60a5fa", "NORMAL": "#94a3b8", "HIGH": "#fbbf24", "EXTREME": "#f87171"}.get(vol_reg, "#94a3b8")

            fade_txt  = _fmt_num(row.get('Signal'))
            z_cell    = _z_cell(row.get('CLR_Z'), thr)
            side_cell = _side_cell(row.get('Side'))
            conv_cell = _conv_cell(row.get('Conviction'))

            table_rows.append(f"""
            <tr>
                <td class="numeric" style="color: #D4A853; font-weight: 700;">{rank_str}</td>
                <td class="symbol">{symbol}</td>
                <td class="numeric" style="color: #4a9eff; font-weight: 700;">TOP {min(100.0, 101-pct_rank):,.1f}%</td>
                <td class="numeric currency">{price:,.2f}</td>
                <td class="numeric" style="color: {pct_color}; font-weight: 600;">{pct_change:+.2f}%</td>
                <td class="numeric" style="color: {accent_light}; font-weight: 600;">{fade_txt}</td>
                {z_cell}
                {side_cell}
                {conv_cell}
                <td class="numeric" style="color: #D4A853; font-weight: 600;">{_human_vol(bar_delta)}</td>
                <td class="numeric" style="color: {cvd_slope_color}; font-size: 0.65rem; font-weight: 600;">{cvd_slope_arrow}{_human_vol(abs(cvd_slope), signed=False)}</td>
                <td class="numeric" style="color: {regime_color}; font-weight: 700; font-size: 0.65rem;">{regime_tag}</td>
                <td class="numeric" style="color: {vol_color}; font-weight: 700; font-size: 0.65rem;">{vol_reg}</td>
            </tr>
            """)

    table_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'IBM Plex Mono', monospace;
            background: transparent;
            color: #F1F5F9;
            padding: 0.5rem;
        }}
        .portfolio-table {{
            width: 100%;
            border-radius: 10px;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            border: 1px solid rgba(255, 255, 255, 0.05);
            background: linear-gradient(145deg, rgba(17, 24, 39, 0.45) 0%, rgba(17, 24, 39, 0.4) 100%);
        }}
        .portfolio-table table {{
            width: 100%;
            min-width: 480px;
            border-collapse: collapse;
        }}
        .portfolio-table thead th {{
            background: linear-gradient(180deg, rgba(10, 14, 23, 0.95) 0%, rgba(10, 14, 23, 0.85) 100%);
            color: #4B5563;
            font-size: 0.62rem !important;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            padding: 0.75rem 0.75rem;
            border-bottom: 2px solid {border_color};
            text-align: left;
        }}
        .portfolio-table thead th.numeric {{ text-align: right; }}
        .portfolio-table tbody tr {{
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
            transition: background 0.2s ease;
        }}
        .portfolio-table tbody tr:nth-child(odd) {{ background: rgba(255, 255, 255, 0.01); }}
        .portfolio-table tbody tr:nth-child(even) {{ background: rgba(255, 255, 255, 0.005); }}
        .portfolio-table tbody tr:hover {{ background: {border_color}; }}
        .portfolio-table tbody td {{
            padding: 0.85rem 0.75rem;
            color: #F1F5F9;
            vertical-align: middle;
            font-size: 0.75rem !important;
            white-space: nowrap;
        }}
        .portfolio-table tbody td.symbol {{
            font-weight: 700;
            font-size: 0.78rem;
            letter-spacing: 0.02em;
            font-family: 'Space Grotesk', sans-serif;
        }}
        .portfolio-table tbody td.numeric {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
    </style>
    </head>
    <body>
    <div class="portfolio-table">
        <table>
            <thead>
                <tr>
                    <th class="numeric">Rank</th>
                    <th>Symbol</th>
                    <th class="numeric" title="Cross-sectional fade-score percentile within this universe">Percentile</th>
                    <th class="numeric">Price</th>
                    <th class="numeric">% Change</th>
                    <th class="numeric" title="Fade score = −z. Positive = bullish (a weak close).">Fade</th>
                    <th class="numeric" title="Close-location z — the CLR core measure">Close-Loc z</th>
                    <th class="numeric" title="▲ BUY past −thr · ◆ SELL past +thr · — inside the band (context only)">Side</th>
                    <th class="numeric" title="How far |z| sits between the firing threshold and the largest reading this bar's own window could produce, × the cost gate. Cap-relative, so two bars at the same z can differ. A description, not a validated forecast. Not a probability.">Conv</th>
                    <th class="numeric">Bar Δ</th>
                    <th class="numeric">CVD Slope</th>
                    <th class="numeric" title="HMM regime — risk context, not a signal input">Regime</th>
                    <th class="numeric" title="GARCH volatility regime — risk context, not a signal input">Vol</th>
                </tr>
            </thead>
            <tbody>
                {"".join(table_rows)}
            </tbody>
        </table>
    </div>
    </body>
    </html>
    """
    return table_html




_SIGNAL_TYPE_REFERENCE = [
    ("▲ BUY · Weak Close (green triangle)", "emerald",
     "The screening condition. Fires when the close-location z-score drops below −1.5σ — the bar "
     "closed unusually near its low relative to its own trailing year. A weak close predicts "
     "STRENGTH, so this is the buy: fade it, enter at the next session's open, hold 5-10 bars. "
     "This is the side that survived: drift-free discovery +0.0546 and holdout +0.0534, confirmed "
     "in both eras. The 1.5σ trigger fires on ~9.3% of days, which is what cuts turnover ~35x and "
     "makes the signal clear costs at all."),
    ("◆ SELL · Strong Close (yellow diamond)", "amber",
     "Fires when the close-location z rises above +1.5σ — the bar closed unusually near its high. "
     "The same mean-reversion logic says expect weakness. Read the caveat, though: the source "
     "indicator labels this side CAUTION rather than a short entry, because its drift-free holdout "
     "was +0.0094 with a CI of [−0.030, +0.052] — it did NOT confirm out of sample. Sanket surfaces "
     "it as a sell signal as configured; treat it as stronger evidence for trimming longs than for "
     "initiating shorts."),
    ("Scope · measured on YOUR universe, not inherited", "violet",
     "Whether this rule carries an edge is a question about your symbols, so the app measures it "
     "on them rather than quoting a class average. The Edge Study below runs an event study over "
     "~15 years: each instrument's own drift removed within era (so a rising market cannot read "
     "as edge), vol-normalised, confidence intervals from a block bootstrap over dates (so "
     "overlapping returns and a correlated cross-section cannot fake significance), with the "
     "effective sample size and minimum detectable effect stated. Parameters are never tuned to "
     "your data — that would fit noise. If the interval straddles zero, the app says so and still "
     "fires the signals: it is a measurement, not a filter. There is NO intraday edge here; none "
     "is claimed."),
]


def _render_system_data_tab(results_df, analysis_date, universe=None, selected_index=None,
                            clr=None, study=None):
    """System Data tab — exports, raw factor frame, and the signal-type legend.

    Used by both Single Date and Pulse Narrative modes (their tab_raw share content).
    Universe context is threaded through so download filenames stay self-describing; ``clr``
    and ``study`` drive the Edge Study readout at the bottom.
    """
    if clr is None:
        clr = _active_clr_settings()
    ui.render_section_header(
        "System Data",
        "Exports, raw factor frame, and reference legends",
        icon="database", accent="cyan",
    )

    # ── Downloads ─────────────────────────────────────────────────────────
    # Split on the FIRED events, not on the sign of the score: a positive fade score
    # below the trigger is context, not a buy candidate.
    _side = results_df['Side'] if 'Side' in results_df.columns else None
    buy_df  = results_df[_side == 'Buy']  if _side is not None else results_df.iloc[0:0]
    sell_df = results_df[_side == 'Sell'] if _side is not None else results_df.iloc[0:0]

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            "↓ Full Report (Excel)",
            data=to_excel(results_df),
            file_name=build_download_filename(
                "snapshot", universe=universe, selected_index=selected_index,
                dates=analysis_date, ext="xlsx",
            ),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width='stretch',
            key="sysdata_dl_full",
            help=(
                f"All {len(results_df)} symbols with every computed column. "
                "Includes a Legend sheet defining each one: the CLR signal (close location, z, "
                "fade score, buy/sell events, hold window, conviction), the descriptive order-flow "
                "context, and the regime columns."
            ),
        )
    with dl2:
        st.download_button(
            "▲ BUY Signals (Excel)",
            data=to_excel(buy_df),
            file_name=build_download_filename(
                "buy", universe=universe, selected_index=selected_index,
                dates=analysis_date, ext="xlsx",
            ),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width='stretch',
            key="sysdata_dl_buy",
            disabled=len(buy_df) == 0,
            help=f"{len(buy_df)} symbols firing the green triangle (weak close, z below the trigger).",
        )
    with dl3:
        st.download_button(
            "◆ SELL Signals (Excel)",
            data=to_excel(sell_df),
            file_name=build_download_filename(
                "sell", universe=universe, selected_index=selected_index,
                dates=analysis_date, ext="xlsx",
            ),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width='stretch',
            key="sysdata_dl_sell",
            disabled=len(sell_df) == 0,
            help=f"{len(sell_df)} symbols firing the yellow diamond (strong close, z above the trigger).",
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Raw Data Table ────────────────────────────────────────────────────
    ui.render_section_header(
        "Raw Signal Frame",
        f"{len(results_df)} symbols · sorted by fade score (weakest closes first)",
        icon="list", accent="emerald",
    )
    cols = ["DisplayName", "Price", "CLR_Z", "Signal", "Side", "Conviction",
            "CLR_State", "CLR_Hold_Age", "SignalType", "CLR_CLV", "CLR_Rank_Pct"]
    if "% Chng Since" in results_df.columns and results_df["% Chng Since"].notna().any():
        cols.insert(2, "% Chng Since")
    cols += ["Zone", "Bar_Delta", "CVD_Slope", "Delta_Z", "Abs_Strength",
             "Vol_Regime", "Regime_Confidence"]
    # The per-age event columns, so the frame carries the same signal history the
    # Action Dashboard buckets by.
    cols += [c for c in ("BUY_Today", "BUY_1d", "BUY_2d", "BUY_3d", "BUY_5d",
                         "SELL_Today", "SELL_1d", "SELL_2d", "SELL_3d", "SELL_5d")
             if c in results_df.columns]
    cols += [c for c in ("Signal_Reason",) if c in results_df.columns]
    cols = [c for c in cols if c in results_df.columns]
    # Rename internal column names to domain-readable labels for display
    _col_display_names = {
        "DisplayName":  "Symbol",
        "CLR_Z":         "Close-Loc z",
        "Signal":       "Fade Score",
        "CLR_State":     "State",
        "CLR_Hold_Age":  "Hold Age",
        "SignalType":   "Type",
        "CLR_CLV":       "Close Location",
        "CLR_Rank_Pct":  "Fade %ile",
        "Bar_Delta":    "Bar Δ",
        "CVD_Slope":    "CVD Slope",
        "Delta_Z":      "Δ-Z",
        "Abs_Strength": "Absorption ×",
        "Signal_Reason": "Read",
    }
    display_frame = (results_df[cols]
                     .sort_values("Signal", ascending=False, na_position='last')
                     .rename(columns=_col_display_names))
    _sysdata_colcfg = {
        "Close-Loc z": st.column_config.NumberColumn(
            help=("The signal. Z-score of where the bar closed inside its own range, over the "
                  "trailing lookback. Below −1.5σ fires BUY, above +1.5σ fires SELL."),
            format="%+.2f",
        ),
        "Fade Score": st.column_config.NumberColumn(
            help="−z. Positive = bullish (a weak close). This is what the tables rank on.",
            format="%+.2f",
        ),
        "Close Location": st.column_config.NumberColumn(
            help="((C−L) − (H−C)) / (H−L) in [−1, +1]. −1 = closed on the low, +1 = on the high.",
            format="%+.3f",
        ),
        "Conviction": st.column_config.ProgressColumn(
            help=("|z| magnitude × the instrument class's measured out-of-sample expectancy × the "
                  "cost gate, in [0,1]. Not a probability — a relative weighting."),
            format="%.2f", min_value=0.0, max_value=1.0,
        ),
        "Hold Age": st.column_config.NumberColumn(
            help="Bars since the current hold window opened. Blank = no window open.",
            format="%.0f",
        ),
    }
    st.dataframe(display_frame, width='stretch', height=500, column_config=_sysdata_colcfg)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Signal Type Reference ─────────────────────────────────────────────
    ui.render_section_header(
        "Signal Reference",
        "The one screening condition, its two events, and where it holds",
        icon="info", accent="amber",
    )
    # One column per reference card so the three cards widen equally and fill the
    # row — a fixed 4-column grid would leave an empty slot / dead space on the right.
    ref_cols = st.columns(len(_SIGNAL_TYPE_REFERENCE))
    accent_var_map = {
        "amber":   "var(--amber)",
        "violet":  "var(--violet)",
        "cyan":    "var(--cyan)",
        "rose":    "var(--rose)",
        "emerald": _CLR_BUY,
    }
    # min-height + flex layout keeps all cards visually equal regardless of body text
    # length. Without it cards stretch to their own content because Streamlit's columns
    # don't enforce a shared height.
    SIG_CARD_MIN_H = "14rem"
    for slot, (title, accent_key, body) in zip(ref_cols, _SIGNAL_TYPE_REFERENCE):
        with slot:
            color = accent_var_map.get(accent_key, "var(--ink-secondary)")
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.015);
                        border:1px solid var(--border);
                        border-left:3px solid {color};
                        border-radius:var(--r-sm);
                        padding:0.85rem 1rem;
                        min-height:{SIG_CARD_MIN_H};
                        display:flex; flex-direction:column;
                        box-sizing:border-box;">
                <div style="font-family:var(--display); font-size:0.78rem; font-weight:700;
                            color:{color}; letter-spacing:0.04em; margin-bottom:0.5rem;">
                    {title}
                </div>
                <div style="font-family:var(--data); font-size:0.7rem; color:var(--ink-secondary);
                            line-height:1.55; flex:1;">
                    {body}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Edge Study ────────────────────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    _render_edge_study_panel(clr, study)


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
        console.item("Signal engine", "CLR — Close-Location Reversal (sb_v8.pine)")

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
    clr                = sbs.clr          # resolved engine settings for this run

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
        _had_study = _edge_cache_get(_edge_key(universe, selected_index, timeframe, clr)) is not None
        study = ensure_edge_study(universe, selected_index, timeframe, clr,
                                  progress_slot=_study_slot, progress_offset=0,
                                  progress_scale=_STUDY_PROGRESS_SHARE)
        # The sidebar card was painted before this ran — repaint it so a study measured on
        # THIS click shows immediately instead of one interaction later.
        _refresh_engine_card()
        if study is None and not _had_study:
            st.warning(
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
            header_text = "CLR Signal Screener" if mode == "Single Date" else "Pulse Narrative Analysis"
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
                clr=clr, study=study,
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
                wt2_len=wt2_len, wt2_type=wt2_type, clr=clr, study=study,
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
                corr_lookback, corr_method, timeframe, analysis_date, clr=clr, study=study,
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
    study = _edge_cache_get(_edge_key(universe, selected_index, timeframe, clr))
    _mv_label, _mv_kind, _mv_detail = _study_state(study, "buy")

    if show_landing:
        ui.render_header("Sanket", f"Market Signal Screener · {ENGINE_NAME}")
        if st.session_state.get("run_error"):
            st.error(st.session_state["run_error"])
        render_landing_page()
        render_footer()
    else:
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

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            if mode == "Pulse Narrative":
                tab_narrative, tab_strength, tab_raw = st.tabs(["Pulse Narrative Dashboard", "Signal Strength", "System Data"])
                with tab_narrative:
                    _pn_stats   = st.session_state.get("screener_run_stats", {})
                    _pn_n       = _pn_stats.get("analyzed", len(results_df))
                    _pn_total   = _pn_stats.get("total_in_universe", _pn_n)
                    _pn_date    = analysis_date.strftime("%d %b %Y") if hasattr(analysis_date, "strftime") else str(analysis_date)
                    ui.render_section_header(
                        f"Pulse Narrative — {timeframe} Universe State",
                        f"{_pn_n} / {_pn_total} symbols · {_pn_date} · {clr.iclass} · "
                        f"full universe ranked by close-location fade score",
                        icon="zap", accent="amber"
                    )
                    _n = max(len(results_df), 1)
                    avg_fade  = results_df['Signal'].mean()
                    n_buy     = int((results_df['Side'] == 'Buy').sum())  if 'Side' in results_df.columns else 0
                    n_sell    = int((results_df['Side'] == 'Sell').sum()) if 'Side' in results_df.columns else 0
                    weak_bias = (results_df['Signal'] > 0).sum() / _n * 100
                    m1, m2, m3, m4 = st.columns(4)
                    with m1: ui.render_metric_card("Universe Fade", _fmt_num(avg_fade, "{:+.3f}"),
                                                   "Mean −z · >0 = closing weak", "neutral")
                    with m2: ui.render_metric_card("▲ BUY Fires", str(n_buy),
                                                   f"{n_buy/_n*100:.0f}% of universe past −{clr.thr:.1f}σ",
                                                   "success" if n_buy else "neutral")
                    with m3: ui.render_metric_card("◆ SELL Fires", str(n_sell),
                                                   f"{n_sell/_n*100:.0f}% of universe past +{clr.thr:.1f}σ",
                                                   "warning" if n_sell else "neutral")
                    with m4: ui.render_metric_card("Weak-Close Breadth", f"{weak_bias:.0f}%",
                                                   "symbols closing below their own mean",
                                                   "success" if weak_bias > 50 else "danger")
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    buy_narr_tab, sell_narr_tab = st.tabs(["Weakest Closes (buy side)", "Strongest Closes (sell side)"])
                    with buy_narr_tab:
                        buy_rank_df = results_df.sort_values('Priority_Long', ascending=False, na_position='last')
                        st.components.v1.html(_build_narrative_table_html(buy_rank_df, side='buy', thr=clr.thr),
                                              height=min(1200, 150 + len(buy_rank_df) * 52), scrolling=True)
                    with sell_narr_tab:
                        sell_rank_df = results_df.sort_values('Priority_Short', ascending=False, na_position='last')
                        st.components.v1.html(_build_narrative_table_html(sell_rank_df, side='sell', thr=clr.thr),
                                              height=min(1200, 150 + len(sell_rank_df) * 52), scrolling=True)

                # ════ Pulse Narrative · TAB 2: SIGNAL STRENGTH ═════════════════════════════
                with tab_strength:
                    ui.render_section_header(
                        "Close-Location Extremes",
                        "Top 10 each side by |z| — the most stretched closes in the universe",
                        icon="zap", accent="amber",
                    )
                    pn_top_buys  = results_df.sort_values('Priority_Long',  ascending=False, na_position='last').head(10)
                    pn_top_sells = results_df.sort_values('Priority_Short', ascending=False, na_position='last').head(10)

                    _n = max(len(results_df), 1)
                    _absz         = results_df['CLR_Z'].abs() if 'CLR_Z' in results_df.columns else pd.Series(dtype=float)
                    pn_avg_absz   = _absz.mean()
                    pn_max_absz   = _absz.max()
                    pn_past_thr   = int((_absz > clr.thr).sum())
                    pn_warming    = st.session_state.get("screener_run_stats", {}).get("warming_up", 0)

                    s1, s2, s3, s4 = st.columns(4)
                    with s1: ui.render_metric_card("Avg |z|", _fmt_num(pn_avg_absz, "{:.2f}"),
                                                   f"vs ±{clr.thr:.1f}σ trigger", "neutral")
                    with s2: ui.render_metric_card("Max |z|", _fmt_num(pn_max_absz, "{:.2f}"),
                                                   "most stretched close today", "info")
                    with s3: ui.render_metric_card("Past Trigger", str(pn_past_thr),
                                                   f"{pn_past_thr/_n*100:.0f}% of universe · ~9.3% is typical", "info")
                    with s4:
                        _r = (study.get("buy", "holdout") or study.get("buy", "full")) if study else None
                        ui.render_metric_card(
                            "Measured Edge",
                            f"{_r.edge:+.3f}" if _r is not None else "—",
                            (f"{_r.hit:.1f}% hit · {_mv_label}" if _r is not None
                             else "not measured on this universe"),
                            _mv_kind)
                    if pn_warming:
                        st.caption(f"{pn_warming} symbol(s) excluded — fewer than {clr.min_bars} bars, "
                                   "so the close-location z-score has no lookback yet.")

                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    pn_l, pn_s = st.columns(2)
                    with pn_l:
                        st.markdown(
                            f'<p style="font-family:\'IBM Plex Mono\',monospace; font-size:0.62rem; '
                            f'font-weight:600; text-transform:uppercase; letter-spacing:0.1em; '
                            f'color:{_CLR_BUY}; margin:0 0 0.4rem 0;">▲ Top 10 Weakest Closes</p>',
                            unsafe_allow_html=True,
                        )
                        st.components.v1.html(
                            _build_signal_strength_table_html(pn_top_buys, side='buy', thr=clr.thr),
                            height=150 + len(pn_top_buys) * 55,
                        )
                    with pn_s:
                        st.markdown(
                            f'<p style="font-family:\'IBM Plex Mono\',monospace; font-size:0.62rem; '
                            f'font-weight:600; text-transform:uppercase; letter-spacing:0.1em; '
                            f'color:{_CLR_SELL}; margin:0 0 0.4rem 0;">◆ Top 10 Strongest Closes</p>',
                            unsafe_allow_html=True,
                        )
                        st.components.v1.html(
                            _build_signal_strength_table_html(pn_top_sells, side='sell', thr=clr.thr),
                            height=150 + len(pn_top_sells) * 55,
                        )

                # ════ Pulse Narrative · TAB 3: SYSTEM DATA ════════════════════════════════
                with tab_raw:
                    _render_system_data_tab(results_df, analysis_date,
                                            universe=universe, selected_index=selected_index,
                                            clr=clr, study=study)
            else:
                tab_signals, tab_strength, tab_raw = st.tabs(["Action Dashboard", "Signal Strength", "System Data"])
                with tab_signals:
                    timeframe_label = "This Week's" if timeframe == 'Weekly' else "Today's"
                    _run_stats = st.session_state.get("screener_run_stats", {})
                    _n_analyzed = _run_stats.get("analyzed", len(results_df))
                    _n_universe = _run_stats.get("total_in_universe", _n_analyzed)
                    _n_warming  = _run_stats.get("warming_up", 0)
                    _date_str   = analysis_date.strftime("%d %b %Y") if hasattr(analysis_date, "strftime") else str(analysis_date)
                    ui.render_section_header(
                        f"{timeframe_label} Signals",
                        f"{_n_analyzed} / {_n_universe} symbols · {timeframe} · {_date_str} · "
                        f"{ENGINE_NAME} ±{clr.thr:.1f}σ · measured: {_mv_label}",
                        icon="zap",
                        accent="amber"
                    )

                    # The two events, bucketed by how long ago they fired.
                    buys_df  = results_df[results_df['BUY_5d']  != "—"].copy().sort_values('Priority_Long',  ascending=False, na_position='last')
                    sells_df = results_df[results_df['SELL_5d'] != "—"].copy().sort_values('Priority_Short', ascending=False, na_position='last')

                    if timeframe == 'Weekly':
                        _age_order = ["This Week", "1 Week Ago", "2 Weeks Ago", "3 Weeks Ago", "Within 5 Weeks"]
                    else:
                        _age_order = ["Today", "1 Day Ago", "2 Days Ago", "3 Days Ago", "Within 5 Days"]

                    has_signals = not (buys_df.empty and sells_df.empty)

                    if has_signals:
                        _fired_today_buy  = int((results_df['BUY_Today']  != "—").sum())
                        _fired_today_sell = int((results_df['SELL_Today'] != "—").sum())

                        mc1, mc2, mc3, mc4 = st.columns(4)
                        with mc1:
                            ui.render_metric_card("▲ BUY Signals", str(len(buys_df)),
                                                  f"{_fired_today_buy} fired {'this week' if timeframe == 'Weekly' else 'today'}",
                                                  "success")
                        with mc2:
                            ui.render_metric_card("◆ SELL Signals", str(len(sells_df)),
                                                  f"{_fired_today_sell} fired {'this week' if timeframe == 'Weekly' else 'today'}",
                                                  "warning")
                        with mc3:
                            _sb_top = buys_df.iloc[0] if not buys_df.empty else None
                            ui.render_metric_card(
                                "Weakest Close",
                                _sb_top['SimpleName'] if _sb_top is not None else "—",
                                (f"z {float(_sb_top['CLR_Z']):+.2f}σ" if _sb_top is not None
                                 and pd.notna(_sb_top.get('CLR_Z')) else "no BUY signals"),
                                "info")
                        with mc4:
                            _ss_top = sells_df.iloc[0] if not sells_df.empty else None
                            ui.render_metric_card(
                                "Strongest Close",
                                _ss_top['SimpleName'] if _ss_top is not None else "—",
                                (f"z {float(_ss_top['CLR_Z']):+.2f}σ" if _ss_top is not None
                                 and pd.notna(_ss_top.get('CLR_Z')) else "no SELL signals"),
                                "info")

                        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                        buy_tab, sell_tab = st.tabs(["▲ BUY Signals by Timing", "◆ SELL Signals by Timing"])

                        def _render_age_table(df_, side_key):
                            _, _stats, _trend, _tcol = _bucket_signals_by_age(
                                df_, side=side_key, timeframe=timeframe)
                            _html = _build_signal_table_html(_stats, side=side_key,
                                                             timeframe=timeframe, thr=clr.thr)
                            _g = sum(1 for a in _age_order if _stats[a]['count'] > 0)
                            _r = sum(_stats[a]['count'] for a in _age_order)
                            st.markdown(
                                f'<div style="font-family:var(--data); font-size:0.66rem; '
                                f'color:{_tcol}; padding:0.2rem 0 0.5rem 0;">{_trend} — newest fires vs older, by |z|.'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            st.components.v1.html(_html, height=max(120 + _g * 60 + _r * 56, 160),
                                                  scrolling=True)

                        with buy_tab:
                            st.markdown(
                                f'<div style="font-family:var(--data); font-size:0.66rem; color:var(--ink-tertiary); '
                                f'padding:0.2rem 0 0.5rem 0;">Close-location z below <b>−{clr.thr:.1f}σ</b> — a weak close '
                                f'to fade up. Entry is the next session\'s open; the measured horizon is 5–10 bars. '
                                f'This is the holdout-confirmed side.</div>',
                                unsafe_allow_html=True,
                            )
                            _render_age_table(buys_df, 'buy')
                        with sell_tab:
                            st.markdown(
                                f'<div style="font-family:var(--data); font-size:0.66rem; color:var(--ink-tertiary); '
                                f'padding:0.2rem 0 0.5rem 0;">Close-location z above <b>+{clr.thr:.1f}σ</b> — a strong close. '
                                f'The source indicator labels this side <b>CAUTION</b> rather than a short entry: its '
                                f'drift-free holdout was +0.0094 with a CI of [−0.030, +0.052], so it did not confirm '
                                f'out of sample.</div>',
                                unsafe_allow_html=True,
                            )
                            _render_age_table(sells_df, 'sell')
                    else:
                        st.info(
                            f"**No signals fired** for {selected_index} on {analysis_date} ({timeframe}). "
                            f"All {_n_analyzed} symbols were analyzed but none closed past ±{clr.thr:.1f}σ in the last "
                            f"5 bars — at the measured threshold that is normal (~9.3% of days fire). "
                            "Try an adjacent trading date, a broader universe, or the Signal Strength tab for the "
                            "full ranking."
                        )
                        if _n_warming:
                            st.caption(f"{_n_warming} symbol(s) excluded — fewer than {clr.min_bars} bars of history.")

                # Action Dashboard's own Signal Strength + System Data tabs.
                # Pulse Narrative has its own equivalents inside the `if` branch above
                # (different framing — universe extremes rather than fired-signal filter),
                # so these blocks must NOT escape the `else:` indentation level — that would
                # cause Pulse Narrative to register the same widget keys twice.

                # ════ Action Dashboard · TAB 2: SIGNAL STRENGTH ═══════════════════════
                with tab_strength:
                    ui.render_section_header(
                        "Close-Location Ranking",
                        f"Full universe ordered by fade score — the ±{clr.thr:.1f}σ trigger marks where it becomes actionable",
                        icon="zap",
                        accent="amber"
                    )

                    _n = max(len(results_df), 1)
                    _absz = results_df['CLR_Z'].abs() if 'CLR_Z' in results_df.columns else pd.Series(dtype=float)
                    avg_absz    = _absz.mean()
                    past_thr    = int((_absz > clr.thr).sum())
                    n_buy_all   = int((results_df['Side'] == 'Buy').sum())  if 'Side' in results_df.columns else 0
                    n_sell_all  = int((results_df['Side'] == 'Sell').sum()) if 'Side' in results_df.columns else 0

                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                    with col_s1: ui.render_metric_card("Avg |z|", _fmt_num(avg_absz, "{:.2f}"),
                                                       f"vs ±{clr.thr:.1f}σ trigger", "neutral")
                    with col_s2: ui.render_metric_card("Past Trigger", str(past_thr),
                                                       f"{past_thr/_n*100:.0f}% of universe · ~9.3% typical", "info")
                    with col_s3: ui.render_metric_card("▲ / ◆ Split", f"{n_buy_all} / {n_sell_all}",
                                                       "weak closes vs strong closes", "info")
                    with col_s4:
                        _r4 = (study.get("buy", "holdout") or study.get("buy", "full")) if study else None
                        ui.render_metric_card(
                            "Measured Edge",
                            f"{_r4.edge:+.3f}" if _r4 is not None else "—",
                            (f"{_r4.hit:.1f}% hit · {_mv_label}" if _r4 is not None
                             else "not measured on this universe"),
                            _mv_kind)

                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

                    # ── column label renderer ──
                    def _col_label(side_label, side_key):
                        _p = _side_palette(side_key)
                        return f"""
                        <p style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem; font-weight:600;
                                   text-transform:uppercase; letter-spacing:0.1em; color:{_p['accent_light']};
                                   margin:0 0 0.4rem 0; display:flex; align-items:center; gap:0.35rem;">
                            {_p['mark']} {side_label}
                        </p>"""

                    st.markdown(f"""
                    <div style="display:flex; align-items:baseline; gap:0.65rem; margin:1.75rem 0 0.9rem 0;
                                 padding-bottom:0.6rem; border-bottom:1px solid rgba(212,168,83,0.2);">
                        <span style="font-family:var(--display); font-size:0.62rem; font-weight:700;
                                     letter-spacing:0.12em; text-transform:uppercase; color:#D4A853;
                                     padding:0.18rem 0.5rem; background:rgba(212,168,83,0.1);
                                     border:1px solid rgba(212,168,83,0.3); border-radius:4px;">{ENGINE_CODE} ENGINE</span>
                        <span style="font-family:var(--display); font-size:1rem; font-weight:700;
                                     color:#F1F5F9; letter-spacing:0.04em;">Top 10 Each Side</span>
                        <span style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#6B7280;">
                            most stretched closes in the universe · a blank Side means it has not crossed the trigger</span>
                    </div>
                    """, unsafe_allow_html=True)

                    top_buys  = results_df.sort_values('Priority_Long',  ascending=False, na_position='last').head(10)
                    top_sells = results_df.sort_values('Priority_Short', ascending=False, na_position='last').head(10)

                    _col_l, _col_s = st.columns(2)
                    with _col_l:
                        st.markdown(_col_label("Top 10 Weakest Closes", "buy"), unsafe_allow_html=True)
                        st.components.v1.html(
                            _build_signal_strength_table_html(top_buys, side='buy', thr=clr.thr),
                            height=150 + len(top_buys) * 55)
                    with _col_s:
                        st.markdown(_col_label("Top 10 Strongest Closes", "sell"), unsafe_allow_html=True)
                        st.components.v1.html(
                            _build_signal_strength_table_html(top_sells, side='sell', thr=clr.thr),
                            height=150 + len(top_sells) * 55)

                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div style="font-family:var(--data); font-size:0.66rem; color:var(--ink-tertiary); '
                        f'padding:0.2rem 0 0.6rem 0; line-height:1.55;">Full universe ranked by fade score '
                        f'(−z). The ranking is continuous, but the measured edge is in the <b>event</b>: holding '
                        f'a continuous position on this signal turns over daily, costs ~12%/yr at 3bp, and nets '
                        f'−0.48 Sharpe. Only rows whose Side shows ▲ or ◆ have crossed the ±{clr.thr:.1f}σ trigger.</div>',
                        unsafe_allow_html=True,
                    )
                    _all_ranked = results_df.sort_values('Priority_Long', ascending=False, na_position='last')
                    st.components.v1.html(
                        _build_signal_strength_table_html(_all_ranked, side='buy', thr=clr.thr),
                        height=min(150 + len(_all_ranked) * 55, 900), scrolling=True)

                # ════ Action Dashboard · TAB 3: SYSTEM DATA ═══════════════════════════
                with tab_raw:
                    _render_system_data_tab(results_df, analysis_date,
                                            universe=universe, selected_index=selected_index,
                                            clr=clr, study=study)

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

def _engine_card_html(clr, study) -> str:
    """The Engine Status card as HTML. Pure, so it can be repainted after a study runs.

    Content is a 2x4 grid — eight cells, one fact each. The sidebar gives each column roughly
    145px, so values are kept to ~12 characters and everything verbose (confidence intervals,
    n_eff, the studied date range, the cost basis) lives in the cell's tooltip. The full
    per-era breakdown is in System Data ▸ Edge Study; this card is a status line, not a report.

    The eight were chosen to answer, in order: is there an edge on each side, how often was the
    signal right and could this test even have found an edge that small, on how broad a sample,
    and at what settings and cost.
    """
    buy_label = _study_state(study, "buy")[0]
    cost_gate_ok = clr.cost_ok(study)
    card_class = _verdict_kind(buy_label) if study is not None else "neutral"
    if not cost_gate_ok:
        card_class = "danger"

    DIM = "var(--ink-tertiary)"

    def _cell(label, value, color="var(--ink-secondary)", title=""):
        t = f' title="{html.escape(title)}"' if title else ""
        return (
            f'<div{t} style="min-width:0;">'
            f'<div style="font-family:var(--data); font-size:0.52rem; color:var(--ink-tertiary); '
            f'text-transform:uppercase; letter-spacing:0.09em; line-height:1.2; '
            f'white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{label}</div>'
            f'<div style="font-family:var(--data); font-size:0.72rem; font-weight:600; '
            f'color:{color}; line-height:1.35; white-space:nowrap; overflow:hidden; '
            f'text-overflow:ellipsis;">{value}</div></div>'
        )

    def _side_cells(side, mark):
        r = (study.get(side, "holdout") or study.get(side, "full")) if study else None
        if r is None:
            return _cell(f"{mark} {side}", "—", DIM,
                         "not measured on this universe yet")
        col = ("var(--emerald)" if r.significant
               else "var(--rose)" if r.anti else "var(--ink-secondary)")
        tip = (f"{side} side, {r.era}: drift-free edge {r.edge:+.4f} vol units, "
               f"95% CI [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] from a block bootstrap over dates. "
               f"{r.n_events} events on {r.n_dates} dates. "
               f"An edge is claimed only when the interval excludes zero.")
        return _cell(f"{mark} {side}", f"{r.edge:+.3f}", col, tip)

    _rb = (study.get("buy", "holdout") or study.get("buy", "full")) if study else None

    if _rb is not None:
        hit_cell = _cell("HIT", f"{_rb.hit:.1f}%", "var(--ink-secondary)",
                         f"Share of buy events where the signal beat that symbol's OWN mean "
                         f"forward return — not a raw win rate. 50% is the no-edge line.")
        mde_cell = _cell("RESOLVES", f"≥{_rb.mde:.3f}", "var(--ink-secondary)",
                         f"Minimum detectable effect at this power: n_eff {_rb.n_eff:.0f} "
                         f"independent observations, so the interval could only separate "
                         f"effects of {_rb.mde:.3f} vol units or larger from zero. A 'no edge' "
                         f"verdict means nothing unless this is smaller than the effect you "
                         f"would care about.")
    else:
        hit_cell = _cell("HIT", "—", DIM, "not measured on this universe yet")
        mde_cell = _cell("RESOLVES", "—", DIM, "not measured on this universe yet")

    if study is not None:
        sample_cell = _cell("SAMPLE", f"{study.n_symbols_studied} syms", "var(--ink-secondary)",
                            f"{study.n_symbols_studied} of {study.n_symbols_universe} symbols, "
                            f"{study.start} to {study.end}, holdout from {study.split_date}."
                            + (f" {study.note}." if study.note else ""))
        indep_cell = _cell("INDEP", f"{study.part_ratio:.1f}", "var(--ink-secondary)",
                           f"Effective independent names in the cross-section "
                           f"({study.part_ratio:.1f} of {study.n_symbols_studied} studied), from "
                           f"the eigenvalues of their correlation matrix. Correlated symbols do "
                           f"not each contribute a fresh observation, which is why power is "
                           f"computed from this and not the symbol count.")
    else:
        sample_cell = _cell("SAMPLE", "—", DIM, "not measured on this universe yet")
        indep_cell = _cell("INDEP", "—", DIM, "not measured on this universe yet")

    # Weekly is an unvalidated extrapolation of a daily study. That caveat must stay VISIBLE —
    # burying it in a tooltip would quietly upgrade an extrapolation to a measured setting.
    _extrap = clr.z_look == eng.CLR_Z_LOOK_WEEKLY
    trigger_cell = _cell("TRIGGER ⚠ EXTRAP" if _extrap else "TRIGGER",
                         f"±{clr.thr:.1f}σ · {clr.horizon}b",
                         "var(--amber)" if _extrap else "var(--ink-secondary)",
                         f"Fires past ±{clr.thr:.1f}σ, holds {clr.horizon} bars, entry the next "
                         f"session's open. Z-score over a {clr.z_look}-bar lookback. Every value "
                         f"is a measured plateau, which is why none is adjustable."
                         + (" EXTRAPOLATED: the source study was daily, so the 52-bar weekly "
                            "lookback is a structural analogue, not a measured plateau."
                            if _extrap else ""))
    cost_cell = _cell("COST", f"{clr.cost_bps:.0f}bp " + ("net +" if cost_gate_ok else "NET NEG"),
                      "var(--emerald)" if cost_gate_ok else "var(--rose)",
                      f"{clr.cost_bps:.1f} bp round-trip. The gate asks whether that cost, in the "
                      f"vol units the edge is measured in, stays under the largest effect this "
                      f"signal has ever shown ({eng.LARGEST_KNOWN_EFFECT:.3f}). It never compares "
                      f"cost against the MEASURED edge — that would let a no-edge verdict halve "
                      f"conviction. Basis: {clr.cost_basis(study)}.")

    cells = (_side_cells("buy", "▲") + _side_cells("sell", "◆")
             + hit_cell + mde_cell
             + sample_cell + indep_cell
             + trigger_cell + cost_cell)

    return f"""
        <div class="metric-card {card_class}" style="
                min-height:auto; padding:0.8rem 0.9rem; margin-bottom:0.7rem; animation:none;">
            <h4 style="margin:0 0 0.2rem 0;">{ENGINE_NAME}</h4>
            <h2 style="font-size:1rem; margin:0 0 0.6rem 0; letter-spacing:-0.01em;">{buy_label}</h2>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem 0.7rem;
                        padding-top:0.55rem; border-top:1px solid rgba(255,255,255,0.06);">
                {cells}
            </div>
        </div>
        """


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
        clr = _active_clr_settings()
        slot.markdown(_engine_card_html(clr, _edge_cache_get(_edge_key(*args, clr))),
                      unsafe_allow_html=True)
    except Exception:
        pass


def _render_engine_status_sidebar(current_universe: str, current_index,
                                  current_timeframe) -> tuple:
    """Sidebar Engine Status panel — visible in every mode.

    Deliberately compact: the engine, the measured verdict for the universe on screen (or an
    honest "not measured yet"), and the four facts a reader needs to interpret it — the two
    sides' intervals, the power behind them, the sample, and the fixed setup. The full
    per-era breakdown lives in the Edge Study panel under System Data; this card is a status
    line, not a report.

    There are no controls here. Parameters are all measured plateaus from the source study, so
    a slider would only invite fitting them to whatever universe is on screen — the exact thing
    that would destroy the credibility of the measurement above it. And the edge study is not
    opt-in: it runs on every run (see :func:`ensure_edge_study`), so there is nothing to tick.

    Caller must be inside a ``with st.sidebar:`` context. Returns the resolved
    :class:`CLRSettings` and stashes it in session state so renderers that do not take it as
    an argument can read it back.
    """
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Engine Status</div>', unsafe_allow_html=True)

    clr = _clr_settings(current_universe, current_index, current_timeframe)
    st.session_state["clr_settings"] = clr

    study = _edge_cache_get(_edge_key(current_universe, current_index, current_timeframe, clr))
    # Painted into a placeholder so it can be repainted after a run measures a study — the
    # sidebar renders BEFORE the analysis executes (single-pass render), so without this the
    # card would show "not measured" for one extra interaction after you measured.
    _slot = st.empty()
    _slot.markdown(_engine_card_html(clr, study), unsafe_allow_html=True)
    st.session_state["_engine_card_slot"] = _slot
    st.session_state["_engine_card_args"] = (current_universe, current_index, current_timeframe)

    return clr


if __name__ == "__main__":
    main()
