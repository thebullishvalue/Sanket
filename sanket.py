"""
Sanket - Market Signal Screener | A Pragyam Product Family Member
Cross-Sectional Reversion Ranker · Quantitative Signal Screener Terminal

Engine: cross-sectional short-horizon mean-reversion with a live alpha-health monitor.
See engine.py and ARCHITECTURE.md for the thesis, validation, and design rationale.
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
import warnings
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional
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

VERSION = "v4.0.6"

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
_MAX_DAYS_BACK = 500  # fetch the maximum once; all modes slice what they need
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
                         wt2_len, wt2_type, end_date) -> tuple:
    """Identity of an analyzed frame — everything that changes its computed values.

    The engine tag invalidates frames cached under a previous signal/feature engine:
    'rev1' = reversion + structural-divergence sets; 'rev2' = LIVE trigger/confluence
    sets; 'rev3' = selective 6-filter Set B; 'rev4' = Set B VWM clamp cross; 'rev5' =
    Set A is delta divergence gated by the thrust mean axis (vwm vs vwm_mean); 'rev6' =
    ATR(14)/Close < 0.05 volatility gate on both Set A & Set B.
    """
    return ("rev6", str(timeframe), int(reg_len), int(wt_n1), int(wt_n2),
            tuple(levels), int(wt2_len), str(wt2_type), end_date)


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

    Always fetches _MAX_DAYS_BACK days so screener, intelligence, and correlation
    can all slice from the same pool without re-fetching.  Correlation callers
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
# Alpha-health monitor — the reversion engine has no weights to calibrate. Instead
# it measures its OWN live edge: the trailing mean of daily cross-sectional IC of
# the reversion score vs realized forward return. Conviction auto-scales by this so
# the screen stands down when the edge is dormant. Measured once/day per universe.
# ──────────────────────────────────────────────────────────────────────────────
def _measure_trailing_ic(ts_data) -> tuple:
    """Compute the trailing daily cross-sectional IC of the reversion score from a
    harvested panel. Re-ranks each date's cross-section with engine.compute_ranking,
    then correlates the reversion score vs a forward-return column.

    Returns (trailing_ic, n_dates, t_stat). trailing_ic / t_stat are None when the
    panel is too thin. t_stat carries the 3-bar-overlap significance haircut so the
    UI can tell a real edge from a noise-level +0.01 IC.
    """
    if ts_data is None or getattr(ts_data, "empty", True):
        return None, 0, None
    df = ts_data.copy()
    # Pick a forward-return column: prefer Ret_3b, else the first available Ret_*.
    ret_cols = [c for c in df.columns if c.startswith("Ret_")]
    if not ret_cols:
        return None, 0, None
    fwd_col = "Ret_3b" if "Ret_3b" in ret_cols else ret_cols[0]
    ics = []
    for _date, day in df.groupby("Date"):
        try:
            ranked = eng.compute_ranking(day, alpha_health_mult=1.0)
        except Exception:
            continue
        # Score column: the reversion Priority_Long (or Rev_Score fallback).
        score_col = "Priority_Long" if "Priority_Long" in ranked.columns else (
            "Rev_Score" if "Rev_Score" in ranked.columns else None)
        if score_col is None or fwd_col not in ranked.columns:
            continue
        ic = eng.cross_sectional_ic(ranked[score_col], ranked[fwd_col])
        if ic is not None and np.isfinite(ic):
            ics.append(ic)
    if not ics:
        return None, 0, None
    window = np.asarray(ics[-60:], dtype=float)   # last ~60 daily ICs
    trailing_ic = float(window.mean())
    # Significance of the trailing mean IC. The daily ICs use a 3-bar-forward
    # return, so consecutive days overlap → the naive t overstates significance.
    # Haircut the effective sample size by the horizon (Newey-West-lite): a +0.010
    # IC at t≈1 is statistically indistinguishable from zero and must NOT read as a
    # confident "edge active". t is None when the window is too thin to be meaningful.
    t_stat = None
    if len(window) >= 10 and window.std(ddof=1) > 0:
        eff_n = max(len(window) / 3.0, 1.0)
        t_stat = float(trailing_ic / (window.std(ddof=1) / np.sqrt(eff_n)))
    return trailing_ic, len(ics), t_stat


def _ensure_alpha_health(universe, selected_index, timeframe, analysis_date,
                         reg_len, wt_n1, wt_n2, levels, wt2_len, wt2_type, calib_settings,
                         external_progress_slot=None, progress_offset=0, progress_scale=100):
    """Measure the live alpha-health (trailing realized IC) for the active universe.

    Harvests a recent lookback panel ending at analysis_date, computes the trailing
    cross-sectional IC of the reversion score vs forward return, maps it to a [0,1]
    conviction multiplier (engine.alpha_health), and stores both in session so the
    screener compute block scales conviction and the Intelligence tab / Engine Status
    panel can render from it.

    Returns a short status string: cached | measured | harvest_empty .
    """
    force     = bool(calib_settings.get("force"))
    today_str = _today_ist().strftime("%Y-%m-%d")
    key       = (universe, selected_index, timeframe)

    # Cheap guard — a fresh alpha-health measured today for this same key: skip recompute.
    if (not force
            and st.session_state.get("_alpha_health_key") == key
            and st.session_state.get("alpha_health_ic") is not None
            and str((st.session_state.get("opt_results") or {}).get("timestamp", ""))[:10] == today_str):
        console.detail(
            f"Alpha-health: reusing today's reading · trailing IC "
            f"{st.session_state.get('alpha_health_ic', float('nan')):+.3f}"
        )
        return "cached"

    # Harvest a lookback window ending at the analysis date, then measure realized edge.
    lookback = int(calib_settings.get("lookback_days", 730))
    start    = analysis_date - datetime.timedelta(days=lookback)
    console.detail(f"Alpha-health: {'forced ' if force else ''}measuring — harvesting ~{lookback}d ending {analysis_date}")
    run_timeseries_analysis(universe, selected_index, start, analysis_date,
                            reg_len, wt_n1, wt_n2, levels, timeframe,
                            wt2_len=wt2_len, wt2_type=wt2_type,
                            external_progress_slot=external_progress_slot,
                            progress_offset=progress_offset, progress_scale=progress_scale)
    ts_data = st.session_state.get("ts_results_df")

    trailing_ic, n_dates, t_stat = _measure_trailing_ic(ts_data)
    if trailing_ic is None:
        # Cold-start neutral — keep a dim, low-conviction screen rather than failing it.
        st.session_state["alpha_health_mult"] = 0.75
        st.session_state["alpha_health_ic"]   = None
        st.session_state["_alpha_health_key"] = key
        st.session_state["opt_results"] = {
            "trailing_ic":    None,
            "alpha_health":   0.75,
            "n_dates":        0,
            "t_stat":         None,
            "timestamp":      _today_ist().strftime("%Y-%m-%d %H:%M"),
            "universe":       universe,
            "selected_index": selected_index,
            "timeframe":      timeframe,
        }
        console.warning("Alpha-health: harvest empty / insufficient — cold-start neutral (0.75)")
        st.session_state["timeseries_done"] = False
        return "harvest_empty"

    health = eng.alpha_health(trailing_ic)
    st.session_state["alpha_health_mult"] = health
    st.session_state["alpha_health_ic"]   = trailing_ic
    st.session_state["alpha_health_t"]    = t_stat
    st.session_state["_alpha_health_key"] = key
    st.session_state["opt_results"] = {
        "trailing_ic":    trailing_ic,
        "alpha_health":   health,
        "n_dates":        n_dates,
        "t_stat":         t_stat,
        "timestamp":      _today_ist().strftime("%Y-%m-%d %H:%M"),
        "universe":       universe,
        "selected_index": selected_index,
        "timeframe":      timeframe,
    }
    _t_disp = f", t {t_stat:+.1f}" if t_stat is not None else ""
    console.detail(f"Alpha-health: trailing IC {trailing_ic:+.3f}{_t_disp} → health {health:.2f} over {n_dates} dates")
    # The harvest flag is an internal precondition here, not a Historical-Range deliverable.
    st.session_state["timeseries_done"] = False
    return "measured"


# Edge-state thresholds. The old badge flipped ACTIVE↔WEAK at a single IC=0.01
# cutoff — so a noise-level +0.0102 (t≈1, indistinguishable from zero) rendered a
# confident green "EDGE ACTIVE". Two fixes: (1) require BOTH a higher IC and a
# significant t before claiming an active edge; (2) hysteresis — a wide neutral
# band between ACTIVE and DORMANT so a reading hovering at the boundary doesn't
# oscillate. Significance is the real gate: mean IC alone is not enough.
_EDGE_IC_ACTIVE = 0.015    # IC must clear this AND be significant to read "active"
_EDGE_IC_DORMANT = 0.000   # below this = actively standing down
_EDGE_T_SIGNIF = 1.5       # trailing-IC t (overlap-adjusted) needed for "active"


def _edge_state(trailing_ic, t_stat):
    """Classify the live edge from IC + significance → (label, css_class, note).

    Honest about statistics: a positive-but-insignificant IC is 'MARGINAL', not
    'EDGE ACTIVE'. Returns a short note flagging when the reading is noise-level.
    """
    if trailing_ic is None:
        return "COLD START", "neutral", "no reading yet — measures on the next run"
    sig = t_stat is not None and abs(t_stat) >= _EDGE_T_SIGNIF
    t_txt = (f"t {t_stat:+.1f}" if t_stat is not None else "t n/a")
    if trailing_ic >= _EDGE_IC_ACTIVE and sig:
        return "EDGE ACTIVE", "success", f"significant ({t_txt})"
    if trailing_ic <= _EDGE_IC_DORMANT:
        return "DORMANT — standing down", "danger", f"no edge ({t_txt})"
    # positive IC but not both high AND significant → be honest it's marginal/noise
    if trailing_ic > 0 and not sig:
        return "MARGINAL — not significant", "warning", f"IC ~ noise ({t_txt})"
    return "WEAK", "warning", f"partial edge ({t_txt})"


def _render_intelligence_tab(universe, selected_index, timeframe):
    """Single-Date 'Intelligence' tab — the Alpha-Health Monitor.

    The ranker is a fixed, validated cross-sectional reversion model — there are no
    factor weights to calibrate. What it DOES measure live is its own realized edge:
    the trailing mean of daily cross-sectional IC of the reversion score vs forward
    return. Conviction auto-scales by that health. Rendered from the active reading
    (opt_results / alpha_health_*); the measurement itself runs inline on the screener
    click, so there is no button here."""
    res = st.session_state.get("opt_results") or {}
    trailing_ic = st.session_state.get("alpha_health_ic", res.get("trailing_ic"))
    health = float(st.session_state.get("alpha_health_mult", res.get("alpha_health", 1.0)) or 1.0)
    n_dates = int(res.get("n_dates", 0) or 0)
    t_stat = st.session_state.get("alpha_health_t", res.get("t_stat"))
    measured = trailing_ic is not None and isinstance(trailing_ic, (int, float))

    ui.render_section_header(
        "Alpha-Health Monitor",
        "Is the reversion edge working right now? — the system measures its own realized edge.",
        icon="brain", accent="violet",
    )

    # ── Interpretation tag — significance-gated, with hysteresis (see _edge_state) ──
    tag, tag_class, tag_note = _edge_state(trailing_ic if measured else None, t_stat)

    ic_t = f" · t {t_stat:+.1f}" if (measured and t_stat is not None) else ""
    ic_str     = f"{trailing_ic:+.3f}" if measured else "—"
    health_str = f"{health:.2f}×"
    health_class = "success" if health >= 0.85 else ("warning" if health >= 0.55 else "danger")
    dates_str  = f"{n_dates}" if n_dates else "—"

    st.markdown(f"""
    <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:1rem; margin-top:0.5rem;">
        <div class="metric-card {"info" if measured else "neutral"}" style="margin:0;">
            <h4>Trailing IC</h4><h2>{ic_str}</h2>
            <div class="sub-metric">realized IC · score vs fwd return{ic_t}</div>
        </div>
        <div class="metric-card {health_class}" style="margin:0;">
            <h4>Alpha-Health</h4><h2>{health_str}</h2>
            <div class="sub-metric">conviction multiplier · 0.35–1.00</div>
        </div>
        <div class="metric-card {tag_class}" style="margin:0;">
            <h4>Edge State</h4><h2 style="font-size:1.15rem;">{tag}</h2>
            <div class="sub-metric">{html.escape(tag_note)}</div>
        </div>
        <div class="metric-card info" style="margin:0;">
            <h4>Dates Measured</h4><h2>{dates_str}</h2>
            <div class="sub-metric">daily ICs in the trailing window</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div style="font-family:var(--data); font-size:0.72rem; color:var(--ink-tertiary); '
        'padding:0.9rem 0 0.1rem 0; line-height:1.6;">'
        'The ranker is a fixed, validated <b>cross-sectional short-horizon reversion</b> model — it '
        'fades over-extended names and buys over-sold ones across the universe. It has no weights to '
        'tune; instead it watches its <b>own realized edge</b>, the trailing mean daily cross-sectional '
        'IC of the reversion score against forward returns. Conviction <b>auto-scales by that health</b>: '
        'when the edge is active the screen runs at full conviction, and when it goes '
        '<b style="color:var(--ink-secondary);">dormant</b> the screen intentionally steps down to '
        'low conviction rather than forcing trades into a market the edge is not currently working in.'
        '</div>',
        unsafe_allow_html=True,
    )

    if not measured:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        ui.render_interpretation_card(
            "Cold start — edge not yet measured",
            "No trailing-IC reading for this universe / timeframe yet, so the screen runs at a neutral "
            "0.75× conviction. The alpha-health auto-measures once per day on the next run — or tick "
            "“Re-measure edge” in the sidebar to measure now.",
            "neutral",
        )

    # ── Reference: Context & Entry signal-aging columns ──
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    _render_aging_reference()


# Legend rows mirror the bands in _context_status / _entry_status — those helpers
# are the source of truth for the cell colors, so keep these hexes in sync.
_CONTEXT_LEGEND = [
    ("Confirmed", "#2DD4A8", "Intel confidence rose since the signal fired — the thesis is strengthening."),
    ("Holding",   "#A3E635", "Confidence broadly unchanged — the regime that backed the signal still holds."),
    ("Fading",    "#FB923C", "Confidence slipping — the supporting context is weakening."),
    ("Stale",     "#E8555A", "Confidence collapsed (or now very low) — the thesis has broken down."),
    ("New",       "#94A3B8", "Fired today — nothing has aged yet."),
]
_ENTRY_LEGEND = [
    ("Open",     "#2DD4A8", "Price has barely moved (under ½σ) — the entry is still fresh."),
    ("Running",  "#5EBFA8", "Moving your way (½–1½σ) — the move is in progress."),
    ("Extended", "#FB923C", "Stretched ≥ 1½σ — most of the move is spent; a late entry."),
    ("Adverse",  "#E8555A", "Gone ≥ 1σ against the signal — price moved the wrong way."),
    ("Now",      "#94A3B8", "Fired today — entry is current."),
]


def _render_aging_reference():
    """Reference guide for the Context & Entry columns on the signal tables.

    Context = has the thesis held since firing (Intel-confidence trajectory);
    Entry = has price already run (σ-scaled move since the fire bar). The two are
    orthogonal — one judges the signal, the other the timing. Rendered with the
    Intelligence tab's own token vocabulary (no new CSS classes)."""
    def _rows(legend):
        out = []
        for label, color, meaning in legend:
            out.append(
                '<div style="display:flex; align-items:baseline; gap:0.6rem; padding:0.32rem 0; '
                'border-bottom:1px solid var(--border-subtle);">'
                f'<span style="font-family:var(--data); font-size:0.7rem; font-weight:700; '
                f'color:{color}; min-width:78px; letter-spacing:0.02em;">{label}</span>'
                '<span style="font-family:var(--data); font-size:0.68rem; color:var(--ink-tertiary); '
                f'line-height:1.45;">{meaning}</span>'
                '</div>'
            )
        return "".join(out)

    ui.render_section_header(
        "Signal Aging Reference",
        "Context & Entry — the two columns beside Intel on the signal tables",
        icon="clock", accent="violet",
    )
    st.markdown(
        '<div style="font-family:var(--data); font-size:0.70rem; color:var(--ink-tertiary); '
        'padding:0.1rem 0 0.7rem 0; line-height:1.5;">'
        'On the Set A / Set B signal tables, each aged signal (1d / 2d / … ago) carries '
        'two <b>orthogonal</b> reads, both scored <b>at the bar it fired</b>. '
        '<b style="color:var(--ink-secondary);">Context</b> asks whether the signal is still good; '
        '<b style="color:var(--ink-secondary);">Entry</b> asks whether the move has already run.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div style="display:grid; grid-template-columns:repeat(2, 1fr); gap:1rem; margin-top:0.1rem;">
            <div class="metric-card neutral" style="margin:0; text-align:left;">
                <div style="font-family:var(--display); font-size:0.8rem; font-weight:700;
                            color:var(--ink-secondary); letter-spacing:0.02em; margin-bottom:0.1rem;">
                    Context <span style="color:var(--ink-tertiary); font-weight:500;">· thesis decay</span>
                </div>
                <div style="font-family:var(--data); font-size:0.66rem; color:var(--ink-tertiary);
                            line-height:1.4; padding-bottom:0.45rem;">
                    Intel confidence at the fire bar vs today.
                </div>
                {_rows(_CONTEXT_LEGEND)}
            </div>
            <div class="metric-card neutral" style="margin:0; text-align:left;">
                <div style="font-family:var(--display); font-size:0.8rem; font-weight:700;
                            color:var(--ink-secondary); letter-spacing:0.02em; margin-bottom:0.1rem;">
                    Entry <span style="color:var(--ink-tertiary); font-weight:500;">· move exhaustion</span>
                </div>
                <div style="font-family:var(--data); font-size:0.66rem; color:var(--ink-tertiary);
                            line-height:1.4; padding-bottom:0.45rem;">
                    Directional move since firing, scaled by the symbol's own volatility (σ).
                </div>
                {_rows(_ENTRY_LEGEND)}
            </div>
        </div>
        <div style="font-family:var(--data); font-size:0.66rem; color:var(--ink-tertiary);
                    padding:0.55rem 0 0.1rem 0; line-height:1.5;">
            A “—” means the value isn't available for that bar. Today's signals show
            <b>New</b> / <b>Now</b> — there is nothing yet to age.
        </div>
        """,
        unsafe_allow_html=True,
    )


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
            # SECOND "today" row next to the 00:00 one, double-counting today and corrupting
            # the 3/5-bar momentum oscillators.
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
    """Convert DataFrame to Excel bytes for download with a Legend sheet."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sanket_Quant_Data')
        
        # Add Legend for user clarity
        legend_data = {
            "Column Identifier": [
                "Priority_Long",
                "Priority_Short",
                "Bar_Delta",
                "CVD",
                "CVD_Slope",
                "Delta_Z",
                "Abs_Strength",
                "Regime_Confidence",
                "Vol_Regime",
                "Change_Point"
            ],
            "Metric Description": [
                "Master ranking score for bullish setups (normalized magnitude).",
                "Master ranking score for bearish setups (normalized magnitude).",
                "Inferred per-bar buy−sell volume delta (OHLC close-location proxy).",
                "Cumulative volume delta (running sum of Bar_Delta).",
                "3-bar change in CVD — flow building (+) or draining (−).",
                "Signed z-score of Bar_Delta vs its 20-bar distribution.",
                "Absorption strength: |Bar_Delta| ÷ its 20-bar average (order-flow context).",
                "Statistical probability (0.0 - 1.0) of the detected HMM regime.",
                "Volatility regime classification (Low/Normal/High) via GARCH analysis.",
                "Structural change point detection (CUSUM) identifying regime shifts."
            ]
        }
        pd.DataFrame(legend_data).to_excel(writer, index=False, sheet_name='Legend')
        
    return output.getvalue()

# ══════════════════════════════════════════════════════════════════════════════
# SHARED MATH HELPERS  (SMA + True Range — the only primitives the engine needs)
# ──────────────────────────────────────────────────────────────────────────────
#  The WRCI-era MA library (EMA/HMA/WMA/VWMA/ALMA/RMA, f_smooth, linreg, RSI) and
#  the Ehlers AutoTune filter were removed with the WRCI engine — the reversion
#  ranker uses only a simple SMA and ATR.
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
                      hci_thres=0.25, hci_look=102, hci_sig_len=53, hci_sig_type="SMA", hci_roc_len=15):
    """Per-symbol feature engine — reversion features (the ranking alpha) plus order-flow
    context (inferred delta, CVD, volume profile).

    The RANKING alpha is cross-sectional reversion (see engine.py / ARCHITECTURE.md);
    its per-symbol inputs are attached here via ``eng.add_reversion_features`` and ranked
    later across the universe by ``eng.compute_ranking``. The legacy WRCI / Conviction /
    Pulse / Liquidity / HCI / AutoTune engines and the momentum factor stack were removed
    (they anti-predicted on real data).

    Everything else here is DESCRIPTIVE ORDER-FLOW CONTEXT, not the ranking signal: the
    OHLC-proxy inferred delta / CVD / volume profile (ported from `Order Flow.pine`), and
    two LIVE same-bar signal-set overlays surfaced for the trader (compute_signal_sets):
      • long_cond/short_cond   — Set A · DELTA DIVERGENCE gated by the thrust mean
                                 axis: rawBull/rawBear (inferred_delta.pine) with
                                 vwm on the confirming side of vwm_mean (clamp.pine)
      • *_comp                 — Set B · CLAMP CROSS: VWM thrust re-enters the clamp
                                 band (bull = crosses up through the lower clamp,
                                 bear = crosses down through the upper clamp)
    These add no cross-sectional ranking edge (validated) — they are shown as flow colour,
    never as the rank.

    The unused WRCI-era params (n1/n2/obLevel*/osLevel*/wt2_*/hci_*) are retained in the
    signature only so existing call sites keep working; `reg_len` still drives the ATR
    window. `_analysis_params_sig` carries an engine tag so frames cached by the old
    engine are invalidated.
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
    # Pine's ta.atr is RMA-smoothed (Wilder), not SMA — matched exactly so the
    # Rel_Range used by the Set B absorption filter reproduces inferred_delta.pine.
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

    df = compute_signal_sets(df, high, low, close, vol,
                             bar_delta, cvd, cvd_ma)

    # ── REVERSION FEATURES (the actual ranking alpha; see engine.py / ARCHITECTURE.md) ──
    # Per-symbol features; the cross-sectional ranking (engine.compute_ranking) runs later
    # once the whole universe is assembled. Order-flow columns above are context only.
    df = eng.add_reversion_features(df)

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



def compute_signal_sets(df: pd.DataFrame,
                        high: pd.Series, low: pd.Series, close: pd.Series,
                        vol: pd.Series,
                        bar_delta: pd.Series, cvd: pd.Series, cvd_ma: pd.Series,
                        # Clamp engine (clamp.pine defaults)
                        thr_length: int = 20, thr_stat_len: int = 60,
                        thr_vol_norm_len: int = 50, base_k: float = 2.0,
                        use_adaptive_k: bool = True) -> pd.DataFrame:
    """LIVE same-bar signal sets: both fire ON the bar they occur (no delay).

    Both are built on the clamp.pine thrust engine — vwm = RVOL × %ROC (Relative
    mode) with an adaptive clamp band (mean ± k·σ, k stretched up to +50% by the
    volatility regime) and a mean axis (vwm_mean).

    Volatility gate (shared by Set A & Set B):
      ATR(14) / Close < 0.05 — filters out high-volatility names where the daily
      average true range exceeds 5% of the close price.

    Set A — DELTA DIVERGENCE confirmed by the thrust's side of the mean axis:
      bull (long_cond)  : rawBull (inferred_delta.pine — close down, inferred
                          delta positive, low at a 3-bar low) AND vwm > vwm_mean
                          (net thrust above the mean axis, force turning up)
                          AND ATR(14)/Close < 0.05.
      bear (short_cond) : rawBear (close up, delta negative, high at a 3-bar high)
                          AND vwm < vwm_mean (thrust below the mean axis)
                          AND ATR(14)/Close < 0.05.

    Set B — CLAMP CROSS (VWM thrust re-enters the adaptive clamp band):
      bull (long_cond_comp)  : vwm crosses UP through the LOWER clamp
                               (vwm ≤ clamp_lower last bar, > it now) — a
                               selling-thrust impact absorbed back inside
                               AND ATR(14)/Close < 0.05.
      bear (short_cond_comp) : vwm crosses DOWN through the UPPER clamp
                               (vwm ≥ clamp_upper last bar, < it now) — a
                               buying-thrust impact absorbed back inside
                               AND ATR(14)/Close < 0.05.
    Same direction convention as clamp.pine (below-lower = selling = bullish).

    Writes long_cond/short_cond (Set A), *_comp (Set B), and the flow Condition.
    """
    n = len(df)
    idx = df.index

    high_a  = high.to_numpy(dtype=float)
    low_a   = low.to_numpy(dtype=float)
    close_a = close.to_numpy(dtype=float)

    # ══ CLAMP ENGINE v2 (clamp.pine · Net Fractional Volume + Asymmetric MAD + Markov States) ══
    pc  = close.shift(1)
    tr  = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    atr_c  = pd.Series(_rma(tr.to_numpy(dtype=float), 14), index=idx)
    atr_ma = atr_c.rolling(50).mean()

    # Discrete Markov Volatility States
    atr_ratio = (atr_c / atr_ma).to_numpy()
    vol_state = np.zeros(n, dtype=int)
    curr_state = 0
    for i in range(n):
        r = atr_ratio[i]
        if np.isnan(r):
            pass
        elif r > 1.3:
            curr_state = 1
        elif curr_state == 1 and r < 1.05:
            curr_state = 0
        elif r < 0.75:
            curr_state = -1
        elif curr_state == -1 and r > 0.9:
            curr_state = 0
        vol_state[i] = curr_state

    if use_adaptive_k:
        dyn_k_vals = np.where(vol_state == 1, base_k * 1.3,
                     np.where(vol_state == -1, base_k * 0.9, base_k))
    else:
        dyn_k_vals = np.full(n, base_k)
    dyn_k = pd.Series(dyn_k_vals, index=idx)

    # Net Fractional Volume (Pseudo Order Flow)
    chg       = close - close.shift(thr_length)
    close_lag = close.shift(thr_length)
    velocity  = pd.Series(np.where(close_lag.to_numpy() != 0,
                                   (chg / close_lag * 100.0), 0.0), index=idx)
    
    h_l_a = high_a - low_a
    vol_a = vol.to_numpy(dtype=float)
    v_buy = vol_a * np.where(h_l_a != 0, (close_a - low_a) / h_l_a, 0.0)
    v_sell = vol_a * np.where(h_l_a != 0, (high_a - close_a) / h_l_a, 0.0)
    netDirVol = pd.Series(np.abs(v_buy - v_sell), index=idx)
    netDirVolSMA = netDirVol.rolling(thr_vol_norm_len).mean()
    rNetVol = netDirVol / netDirVolSMA.clip(lower=1e-10)
    
    # mass = Net Fractional Volume
    vwm = rNetVol * velocity
    
    # Causal Robust Normalization (Asymmetric MAD Envelopes)
    vwm_median = vwm.rolling(thr_stat_len).median()
    vwm_a = vwm.to_numpy()
    vwm_med_a = vwm_median.to_numpy()
    
    mad_up = np.zeros(n)
    mad_dn = np.zeros(n)
    
    for i in range(n):
        if i < thr_stat_len - 1 or np.isnan(vwm_med_a[i]):
            mad_up[i] = 0.0
            mad_dn[i] = 0.0
            continue
        
        start_idx = max(0, i - thr_stat_len + 1)
        window = vwm_a[start_idx : i + 1]
        med = vwm_med_a[i]
        
        up_devs = []
        dn_devs = []
        for val in window:
            if val > med:
                up_devs.append(abs(val - med))
            elif val < med:
                dn_devs.append(abs(val - med))
                
        mad_up[i] = np.median(up_devs) if len(up_devs) > 0 else 0.0
        mad_dn[i] = np.median(dn_devs) if len(dn_devs) > 0 else 0.0

    CONSISTENCY_CONST = 1.4826
    mad_up_s = pd.Series(mad_up, index=idx)
    mad_dn_s = pd.Series(mad_dn, index=idx)
    
    clamp_up = vwm_median + (dyn_k * CONSISTENCY_CONST * mad_up_s.clip(lower=1e-10))
    clamp_dn = vwm_median - (dyn_k * CONSISTENCY_CONST * mad_dn_s.clip(lower=1e-10))

    # Clamp series as numpy (shared by Set A's mean-axis filter and Set B's cross).
    v, v1   = vwm.to_numpy(), vwm.shift(1).to_numpy()
    vm      = vwm_median.to_numpy()  # Set A confirmation now uses vwm_median axis
    cu, cu1 = clamp_up.to_numpy(), clamp_up.shift(1).to_numpy()
    cd, cd1 = clamp_dn.to_numpy(), clamp_dn.shift(1).to_numpy()
    c1      = pc.to_numpy()

    # ── VOLATILITY GATE: ATR(14) / Close < 0.05 — shared by Set A & Set B ──────
    # Filters out high-volatility names where daily ATR exceeds 5% of the close.
    atr_pct = (atr_c / close.clip(lower=1e-10)).to_numpy(dtype=float)
    low_vol_ok = atr_pct < 0.05

    # ── SET A: bar-level delta divergence CONFIRMED by the thrust's side of the mean axis ──
    #   rawBull (inferred_delta.pine): close down, inferred delta positive, at a 3-bar low.
    #   rawBear:                       close up,   delta negative,           at a 3-bar high.
    #   Confirmation (clamp.pine mean axis = vwm_mean): the bull divergence only counts
    #   when net thrust is ABOVE the mean axis (vwm > vwm_mean, force turning up), the bear
    #   only when thrust is BELOW it (vwm < vwm_mean, force turning down).
    bd  = bar_delta.to_numpy(dtype=float)
    lo3 = low.rolling(3).min().to_numpy()
    hi3 = high.rolling(3).max().to_numpy()
    with np.errstate(invalid='ignore'):
        raw_bull = (close_a < c1) & (bd > 0) & (low_a <= lo3)
        raw_bear = (close_a > c1) & (bd < 0) & (high_a >= hi3)
        long_cond  = raw_bull & (v > vm) & low_vol_ok
        short_cond = raw_bear & (v < vm) & low_vol_ok
    long_cond  = np.where(np.isnan(v) | np.isnan(vm) | np.isnan(atr_pct), False, long_cond)
    short_cond = np.where(np.isnan(v) | np.isnan(vm) | np.isnan(atr_pct), False, short_cond)

    # ── SET B: VWM thrust crossing back INTO the clamp band ──
    #   bull  = vwm crosses UP through the LOWER clamp (was ≤ lower last bar, now
    #           above it) → a selling-thrust impact is being absorbed back inside.
    #   bear  = vwm crosses DOWN through the UPPER clamp (was ≥ upper last bar, now
    #           below it) → a buying-thrust impact absorbed. Matches clamp.pine's
    #           direction convention (below-lower = selling = bullish implication),
    #           i.e. the exhaustion re-entry without the price-still-pushing gate or
    #           the same-side cooldown — a cleaner "thrust returned to normal" cross.
    with np.errstate(invalid='ignore'):
        long_cond_comp  = (v1 <= cd1) & (v > cd) & low_vol_ok    # cross up through lower clamp
        short_cond_comp = (v1 >= cu1) & (v < cu) & low_vol_ok    # cross down through upper clamp
    long_cond_comp  = np.where(np.isnan(v1) | np.isnan(cd1) | np.isnan(cd) | np.isnan(atr_pct),
                               False, long_cond_comp)
    short_cond_comp = np.where(np.isnan(v1) | np.isnan(cu1) | np.isnan(cu) | np.isnan(atr_pct),
                               False, short_cond_comp)

    df['long_cond']       = long_cond
    df['short_cond']      = short_cond
    df['long_cond_comp']  = long_cond_comp
    df['short_cond_comp'] = short_cond_comp

    # Flow Condition (unchanged): where cumulative delta sits vs its 20-bar mean →
    # accumulation / distribution / neutral.
    cvd_dev = (cvd - cvd_ma)
    band    = cvd_dev.abs().rolling(20).mean().clip(lower=1e-9)
    df['Condition'] = np.select(
        [cvd_dev >  2 * band, cvd_dev >  band, cvd_dev < -2 * band, cvd_dev < -band],
        ['Accumulation+', 'Accumulation', 'Distribution+', 'Distribution'],
        default='Neutral'
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# REGIME INTELLIGENCE ENGINE (NIRNAY FEATURES)
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
    Apply joint-state regime intelligence over (F1_PriceMom, F2_VolQual, CVD flow).
    The three input dimensions are roughly orthogonal (price momentum, volume
    quality, cumulative-delta flow), so HMM's classification reflects true market
    state. The regime engine is order-flow-agnostic; its Vol_Regime and
    Regime_Confidence outputs feed the conviction read in engine.compute_ranking
    and the per-name risk context in the UI.
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
    """Return priority-ordered signal type for a single bar row (pandas Series).

    Priority: Set A (delta divergence) > Set B (clamp cross) > flow zone.
    Matches the vectorised np.select in the harvest path.
    """
    if row.get('long_cond'):       return "A: Long"
    if row.get('short_cond'):      return "A: Short"
    if row.get('long_cond_comp'):  return "B: Long"
    if row.get('short_cond_comp'): return "B: Short"
    cond = row.get('Condition', 'Neutral')
    return cond if cond != 'Neutral' else '-'


# The legacy WRCI-oscillator divergence detector (`calculate_divergences`, writing
# Bullish_Div / Bearish_Div) was removed in the order-flow rewrite: Set A IS the
# delta-divergence signal now (compute_signal_sets), so a separate price/oscillator
# divergence pass is redundant.

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
                REVERSION ENGINE
            </h3>
            <p>Sanket ranks the universe by short-horizon cross-sectional mean-reversion — names overextended against their own volatility tend to revert relative to peers.</p>
            <div class='spec'>
                <span>Primary:</span> ATR-normalized reversion (ret · dist-from-MA)<br>
                <span>Method:</span> Within-date robust z-score blend<br>
                <span>Validated:</span> IC ≈ +0.03 (t ≈ +8), walk-forward<br>
                <span>Sorting:</span> Rank by reversion score
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='system-card regime'>
            <h3>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>
                CONVICTION & EDGE
            </h3>
            <p>Conviction scales by a live alpha-health monitor — the system measures its own realized edge and stands down when reversion is dormant.</p>
            <div class='spec'>
                <span>Sides:</span> Long (oversold) / Short (overbought) tails<br>
                <span>Conviction:</span> Tail strength × alpha-health × regime<br>
                <span>Edge monitor:</span> Trailing realized cross-sectional IC<br>
                <span>Context:</span> Order-flow + vol-regime overlays
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='system-card strategies'>
            <h3>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
                UNIVERSE & MODES
            </h3>
            <p>Span eight asset classes through five analysis modes — from single-date snapshots to historical edge measurement.</p>
            <div class='spec'>
                <span>Coverage:</span> 8 asset classes · 500+ symbols<br>
                <span>Timeframes:</span> Daily · Weekly<br>
                <span>Modes:</span> 5 modes · snapshot to time-series<br>
                <span>Edge monitor:</span> Live alpha-health (realized IC)
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
           <span style="color:var(--ink-secondary); font-size:0.85em; margin-top:0.5rem; display:inline-block;">System will rank the cross-section by mean-reversion · scale conviction by live alpha-health · surface order-flow context</span></p>
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
    calib_settings: dict[str, Any]


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

        # ── Per-universe alpha-health reset (must run BEFORE the Passport renders) ──
        # When the (universe, selected_index, timeframe) triple changes — including the
        # click that triggered THIS rerun — clear the stale trailing-IC reading so the
        # alpha-health re-measures for the new universe on the next screener run. The
        # reversion engine has no per-universe profiles on disk to load.
        _current_uni_key = (universe, selected_index, timeframe)
        _previous_uni_key = st.session_state.get("_last_universe_key")
        if _previous_uni_key != _current_uni_key:
            st.session_state.pop("alpha_health_ic", None)
            st.session_state.pop("_alpha_health_key", None)
            if "opt_results" in st.session_state:
                del st.session_state["opt_results"]
            if _previous_uni_key is not None:
                _uni_label = (selected_index or universe or "—")
                console.detail(f"Universe changed · {_uni_label} · alpha-health will re-measure")
            st.session_state["_last_universe_key"] = _current_uni_key

        # Engine Status panel — rendered in every mode. Surfaces the engine name, the
        # live alpha-health / trailing-IC reading, and the Meta Filter controls.
        # Returns the resolved alpha-health settings (force / lookback / horizons).
        calib_settings = _render_model_passport_sidebar(universe, selected_index, timeframe, analysis_mode)

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
            calib_settings=calib_settings,
        )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SCREENER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_screener_analysis(universe, selected_index, analysis_date, reg_len, wt_n1, wt_n2, levels, timeframe, show_progress=True, external_progress_slot=None, progress_offset=0, progress_scale=100, wt2_len=20, wt2_type="ALMA"):
    """Execute the cross-sectional reversion screen and return ranked signals.

    Fetches market data for the universe, computes per-symbol reversion features +
    order-flow/regime context, then ranks the whole cross-section by the reversion
    score (engine.compute_ranking), scaling conviction by the live alpha-health read.

    Args:
        external_progress_slot: Optional Streamlit container for external progress tracking (e.g., from correlation analysis)
        progress_offset: Starting percentage for external progress tracking (default 0)
        progress_scale: Scale factor for progress percentage within external slot (default 100 = full)

    Returns: DataFrame with signals ranked by magnitude, or None on error.
    """
    obLevel1, obLevel2, osLevel1, osLevel2 = levels
    progress_slot = external_progress_slot if external_progress_slot is not None else (st.empty() if show_progress else None)

    if show_progress or external_progress_slot is not None:
        pct_val = progress_offset + (5 * progress_scale / 100)
        progress_bar(progress_slot, pct_val, "Initializing Reversion Engine", f"Universe: {universe}")
    
    console.start_phase("DATA ACQUISITION", 1, 2)
    console.section("Universe Configuration")
    console.item("Universe", universe)
    console.item("Selected Index", selected_index)
    console.item("Timeframe", timeframe)

    if universe == "India Indexes":
        stock_list, msg = get_index_stock_list(selected_index)
    elif universe == "Global Indexes":
        stock_list, msg = get_global_index_symbols()
    elif universe == "US Indexes":
        stock_list, msg = get_us_index_symbols(selected_index)
    elif universe == "Commodities":
        stock_list, msg = get_commodity_symbols(None)
    elif universe == "Currency":
        stock_list, msg = get_currency_symbols(None)
    elif universe == "Crypto":
        stock_list, msg = get_crypto_symbols(None)
    elif universe == "ETF Index":
        stock_list, msg = get_etf_symbols()
    elif universe == "Global Macro":
        stock_list, msg = get_global_macro_symbols()
    else:
        stock_list, msg = None, f"Unknown universe: {universe}"

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
    # display-only "% Chng Since" column, which uses the few buffer days after it),
    # while sharing the registry key the alpha-health harvest already populated for
    # this (universe, analysis_date) — no second yfinance round-trip on historical
    # runs. For the common analysis_date == today run this is identical to before.
    end_date = analysis_date if isinstance(analysis_date, datetime.date) else _today_ist()
    data_dict, fetch_msg = get_universe_data(stock_list, end_date=end_date)

    if not data_dict:
        console.error(fetch_msg)
        st.error(fetch_msg)
        return None

    console.success(f"Successfully downloaded data for {len(data_dict)} stocks")

    console.end_phase("DATA ACQUISITION")

    console.start_phase("REVERSION RANKING", 2, 2)

    console.section("Engine Parameters")
    console.item("Engine", "Cross-Sectional Reversion v1")
    console.item("Timeframe", timeframe)
    console.item("Reversion lookbacks", f"ret{eng.REV_RETURN_LAGS}  dist{eng.REV_MA_WINDOWS}")
    console.item("Alpha-health", f"{st.session_state.get('alpha_health_mult', 1.0):.2f}× (live edge scaling)")
    console.item("Instruments", f"{len(data_dict)} of {len(stock_list)} fetched successfully")
    if show_progress or external_progress_slot is not None:
        pct_val = progress_offset + (20 * progress_scale / 100)
        progress_bar(progress_slot, pct_val, "Ranking Cross-Section", f"{len(data_dict)} Stocks")

    results = []
    _failed_symbols = []
    # Per-symbol recent-bar feature windows, keyed by ticker. Powers the fire-bar
    # context status for aged order-flow context signals (1d/2d/… ago).
    intel_windows: dict = {}

    # If an edge-measurement harvest just ran for this exact universe + params + date,
    # its analyzed frames are cached — reuse them instead of recomputing the whole
    # per-stock pipeline.
    _cache_sig = _analysis_params_sig(timeframe, reg_len, wt_n1, wt_n2, levels,
                                      wt2_len, wt2_type, end_date)
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

            # Insufficient-data guard — applied on both cache hit and miss so a short
            # frame cached by the (unguarded) harvest can't slip a stock the screener
            # would otherwise skip.
            if len(df) < reg_len + 30:
                console.detail(f"{ticker}: Skipped (Insufficient data: {len(df)} rows)")
                continue

            if _cached is None:
                df = run_full_analysis(df, reg_len, wt_n1, wt_n2, obLevel1, obLevel2, osLevel1, osLevel2,
                                       wt2_len=wt2_len, wt2_type=wt2_type)
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

            # Capture the recent-bar feature window (offsets 0..4 = Today..Within-5d)
            # for fire-bar Intel scoring — the order-flow features signal_conf_features
            # reads at each fire bar (not just the snapshot bar).
            _win_cols = ['HMM_Bull', 'HMM_Bear', 'Vol_Regime', 'Regime_Confidence',
                         'Change_Point', 'long_cond', 'short_cond',
                         'long_cond_comp', 'short_cond_comp',
                         'F1_PriceMom', 'CVD', 'CVD_Slope', 'Delta_Z', 'Close']
            _win = df.iloc[max(0, idx_pos - 4): idx_pos + 1]
            intel_windows[ticker] = _win[[c for c in _win_cols if c in _win.columns]].copy()
            # Recent daily-return volatility — the asset-agnostic scale for the
            # Entry (move-exhaustion) check: how far has price run, in σ units.
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

            results.append({
                "% Chng Since": pct_chng_since,  # None when data unavailable — displays as NaN / "—"
                "Symbol": ticker,
                "DisplayName": display_name,
                "SimpleName": simple_name,
                "Signal": round(last_row['Delta_Z'], 2) if not pd.isna(last_row['Delta_Z']) else 0.0,
                "Bar_Delta": round(last_row['Bar_Delta'], 2) if not pd.isna(last_row['Bar_Delta']) else 0.0,
                "CVD": round(last_row['CVD'], 2) if not pd.isna(last_row['CVD']) else 0.0,
                "CVD_Slope": round(last_row['CVD_Slope'], 2) if not pd.isna(last_row['CVD_Slope']) else 0.0,
                "Delta_Z": round(last_row['Delta_Z'], 2) if not pd.isna(last_row['Delta_Z']) else 0.0,
                "Abs_Strength": round(last_row.get('Abs_Strength', 0), 2) if not pd.isna(last_row.get('Abs_Strength', 0)) else 0.0,
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
                # Reversion features — the actual ranking inputs (engine.compute_ranking
                # consumes these cross-sectionally to produce Rev_Score / Conviction / Side).
                "_rev_ret2":     last_row.get('_rev_ret2'),
                "_rev_ret5":     last_row.get('_rev_ret5'),
                "_rev_dist5":    last_row.get('_rev_dist5'),
                "_rev_dist10":   last_row.get('_rev_dist10'),
                "_rev_rngpos":   last_row.get('_rev_rngpos'),
                "ATR_Pct":       last_row.get('ATR_Pct'),
                # Set A · legacy L_/S_ alias of LA_/SA_ below
                # (kept for Range Study compat; reads the same long_cond column).
                "L_Today": "●" if sample_range.iloc[-1]['long_cond'] else "—",
                "L_1d": "●" if sample_range.iloc[-2]['long_cond'] else "—",
                "L_2d": "●" if sample_range.iloc[-3]['long_cond'] else "—",
                "L_3d": "●" if sample_range.iloc[-4]['long_cond'] else "—",
                "L_5d": "●" if sample_range.tail(5)['long_cond'].any() else "—",
                # (short side of the same legacy alias)
                "S_Today": "●" if sample_range.iloc[-1]['short_cond'] else "—",
                "S_1d": "●" if sample_range.iloc[-2]['short_cond'] else "—",
                "S_2d": "●" if sample_range.iloc[-3]['short_cond'] else "—",
                "S_3d": "●" if sample_range.iloc[-4]['short_cond'] else "—",
                "S_5d": "●" if sample_range.tail(5)['short_cond'].any() else "—",
                # Set A · Delta Divergence — Historical Long Signals
                "LA_Today": "●" if sample_range.iloc[-1]['long_cond'] else "—",
                "LA_1d": "●" if sample_range.iloc[-2]['long_cond'] else "—",
                "LA_2d": "●" if sample_range.iloc[-3]['long_cond'] else "—",
                "LA_3d": "●" if sample_range.iloc[-4]['long_cond'] else "—",
                "LA_5d": "●" if sample_range.tail(5)['long_cond'].any() else "—",
                # Set A · Delta Divergence — Historical Short Signals
                "SA_Today": "●" if sample_range.iloc[-1]['short_cond'] else "—",
                "SA_1d": "●" if sample_range.iloc[-2]['short_cond'] else "—",
                "SA_2d": "●" if sample_range.iloc[-3]['short_cond'] else "—",
                "SA_3d": "●" if sample_range.iloc[-4]['short_cond'] else "—",
                "SA_5d": "●" if sample_range.tail(5)['short_cond'].any() else "—",
                # Set B · Clamp Cross — Historical Long Signals
                "LB_Today": "●" if sample_range.iloc[-1]['long_cond_comp'] else "—",
                "LB_1d": "●" if sample_range.iloc[-2]['long_cond_comp'] else "—",
                "LB_2d": "●" if sample_range.iloc[-3]['long_cond_comp'] else "—",
                "LB_3d": "●" if sample_range.iloc[-4]['long_cond_comp'] else "—",
                "LB_5d": "●" if sample_range.tail(5)['long_cond_comp'].any() else "—",
                # Set B · Clamp Cross — Historical Short Signals
                "SB_Today": "●" if sample_range.iloc[-1]['short_cond_comp'] else "—",
                "SB_1d": "●" if sample_range.iloc[-2]['short_cond_comp'] else "—",
                "SB_2d": "●" if sample_range.iloc[-3]['short_cond_comp'] else "—",
                "SB_3d": "●" if sample_range.iloc[-4]['short_cond_comp'] else "—",
                "SB_5d": "●" if sample_range.tail(5)['short_cond_comp'].any() else "—",
                # Additional fields for detail cards
                "Osc_Value": round(last_row.get('Delta_Z', 0), 2),
                "MA_Alignment": int(last_row.get('MA_Alignment', 0)),
                "ZScore_Value": round(last_row.get('Delta_Z', 0), 2),
            })

            console.detail(f"[{i+1}/{len(data_dict)}] {ticker}: Delta_Z={last_row['Delta_Z']:+.2f}  Zone={last_row['Condition']}  Status={signal_type}")
            
        except Exception as e:
            console.failure(f"Analysis Failed: {ticker}", str(e))
            _failed_symbols.append(ticker)
            continue

    console.end_phase("REVERSION RANKING")
    if _cache_hits:
        console.detail(f"Analyzed-frame cache: reused {_cache_hits}/{len(data_dict)} frames from the edge-measurement harvest (skipped re-analysis)")
    # One-shot cache — release the harvested frames now that the screener has consumed them.
    _analyzed_cache_clear()

    _fail_count = len(_failed_symbols)
    console.summary("RUN SUMMARY", {
        "Universe": universe,
        "Universe Index": selected_index,
        "Total Symbols": len(stock_list),
        "Data Success": len(data_dict),
        "Analyzed Stocks": len(results),
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
    }
    # Fire-bar feature windows for the age-bucketed signal tables (per-symbol).
    st.session_state["intel_windows"] = intel_windows
    st.session_state["intel_fire_cache"] = {}   # invalidate memoized fire-bar scores
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
        else:
            st.info(
                f"**No signals found** — {_n_fetched} of {_n_total} symbols had data for {analysis_date}, "
                "but none produced an order-flow signal in the current timeframe. "
                "Try an adjacent trading date, or check that the selected date is a market session."
            )
        # Return empty DataFrame with expected columns to prevent downstream KeyErrors
        expected_cols = [
            "Symbol", "DisplayName", "SimpleName", "Signal", "Bar_Delta", "CVD", "CVD_Slope", "Delta_Z", "Zone", "SignalType", "Price", "PctChange",
            "L_Today", "L_1d", "L_2d", "L_3d", "L_5d", "S_Today", "S_1d", "S_2d", "S_3d", "S_5d",
            "LA_Today", "LA_1d", "LA_2d", "LA_3d", "LA_5d", "SA_Today", "SA_1d", "SA_2d", "SA_3d", "SA_5d",
            "LB_Today", "LB_1d", "LB_2d", "LB_3d", "LB_5d", "SB_Today", "SB_1d", "SB_2d", "SB_3d", "SB_5d",
            "Osc_Value", "MA_Alignment", "ZScore_Value",
        ]
        return pd.DataFrame(columns=expected_cols)

    results_df = pd.DataFrame(results)

    # Global ranking via the cross-sectional REVERSION engine (engine.py). The reversion
    # composite is the validated alpha; conviction is scaled by the live alpha-health monitor
    # (is the edge currently working?) and per-name vol-regime suitability. This single call
    # emits the full UI contract (Priority_*, Intel_Confidence, Meta_*) — see engine.py.
    if not results_df.empty:
        health = float(st.session_state.get("alpha_health_mult", 1.0))
        results_df = eng.compute_ranking(results_df, alpha_health_mult=health)

    return results_df


def run_timeseries_analysis(universe, selected_index, start_date, end_date, reg_len, wt_n1, wt_n2, levels, timeframe, wt2_len=20, wt2_type="ALMA",
                            external_progress_slot=None, progress_offset=0, progress_scale=100):
    """Compute the per-(date, symbol) factor frame for a date range.

    Pure compute path: fetches history, runs full / regime / divergence analyses on
    every symbol, builds the per-(date, symbol) row set, and stores ts_results_df +
    ts_meta in ``st.session_state``. **Does not render UI.** The dashboard is
    rendered separately by ``render_timeseries_dashboard()`` so it survives sidebar
    interactions / reruns.

    When ``external_progress_slot`` is supplied (the Single-Date alpha-health
    pre-pass), progress renders into that shared bar over
    [offset, offset+scale], so the whole run shows ONE continuous progress bar
    instead of a separate harvest bar followed by the screener bar.
    """
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

    if universe == "India Indexes":
        stock_list, _ = get_index_stock_list(selected_index)
    elif universe == "Global Indexes":
        stock_list, _ = get_global_index_symbols()
    elif universe == "US Indexes":
        stock_list, _ = get_us_index_symbols(selected_index)
    elif universe == "Commodities":
        stock_list, _ = get_commodity_symbols(None)
    elif universe == "Currency":
        stock_list, _ = get_currency_symbols(None)
    elif universe == "Crypto":
        stock_list, _ = get_crypto_symbols(None)
    elif universe == "ETF Index":
        stock_list, _ = get_etf_symbols()
    elif universe == "Global Macro":
        stock_list, _ = get_global_macro_symbols()
    else:
        stock_list = None

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
    console.start_phase("EDGE MEASUREMENT", 2, 2)
    start_harvest = time.time()

    _p(15, "Measuring Live Edge", f"{len(data_dict)} Stocks")
    all_results = []

    # Analyzed-frame cache for this run — lets the screener that follows skip
    # re-running the identical per-stock analysis pipeline (see helper comment).
    _cache_sig = _analysis_params_sig(timeframe, reg_len, wt_n1, wt_n2, levels,
                                      wt2_len, wt2_type, end_date)
    _analyzed_cache_reset(_cache_sig)

    for i, (ticker, df) in enumerate(data_dict.items()):
        try:
            elapsed = time.time() - start_harvest
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (len(data_dict) - (i + 1))
            eta_str = time.strftime("%M:%S", time.gmtime(remaining))

            # Local 15% -> 85% band for the per-symbol harvest loop.
            pct = 15 + (i + 1) / len(data_dict) * 70
            _p(pct, "Measuring Live Edge", f"{i + 1} / {len(data_dict)} Symbols · ETA {eta_str}")
            if timeframe == "Weekly":
                df = resample_to_weekly(df)
            df = run_full_analysis(df, reg_len, wt_n1, wt_n2, *levels,
                                   wt2_len=wt2_len, wt2_type=wt2_type)
            df = run_regime_analysis(df)
            # Cache the analyzed frame so run_screener_analysis can reuse it instead
            # of recomputing. Stored by reference — the harvest-only columns appended
            # below (Ret_*, SignalType) are harmless extras; the screener copies on read.
            _analyzed_cache_put(ticker, df, _cache_sig)

            # Calculate forward returns for each training horizon
            for h in eng.HOLD_HORIZONS:
                df[f'Ret_{h}b'] = df['Close'].shift(-h) / df['Close'] - 1

            # Vectorized SignalType per bar (priority: A divergence > B clamp-cross > Zone)
            df['SignalType'] = np.select(
                [
                    df['long_cond'],      df['short_cond'],
                    df['long_cond_comp'], df['short_cond_comp'],
                    df['Condition'] != 'Neutral',
                ],
                [
                    'A: Long', 'A: Short',
                    'B: Long', 'B: Short',
                    df['Condition'],
                ],
                default='-',
            )

            mask = (df.index.date >= start_date) & (df.index.date <= end_date)
            range_df = df.loc[mask]

            for date, row in range_df.iterrows():
                all_results.append({
                    'Date': date,
                    'Symbol': ticker,
                    'Signal': row['Delta_Z'],
                    'Bar_Delta': row['Bar_Delta'],
                    'CVD': row['CVD'],
                    'CVD_Slope': row['CVD_Slope'],
                    'Delta_Z': row['Delta_Z'],
                    'Zone': row['Condition'],
                    'LongSignal': row['long_cond'],
                    'ShortSignal': row['short_cond'],
                    # Set A + Set B live-signal booleans (delta-divergence / clamp-cross)
                    'long_cond': row['long_cond'],
                    'short_cond': row['short_cond'],
                    'long_cond_comp': row['long_cond_comp'],
                    'short_cond_comp': row['short_cond_comp'],
                    'SignalType': row['SignalType'],
                    # Regime Intelligence columns
                    'Regime': row.get('Regime', 'NEUTRAL'),
                    'HMM_Bull': row.get('HMM_Bull', 0),
                    'HMM_Bear': row.get('HMM_Bear', 0),
                    'Vol_Regime': row.get('Vol_Regime', 'NORMAL'),
                    'Change_Point': row.get('Change_Point', False),
                    'Regime_Confidence': row.get('Regime_Confidence', 0),
                    # Forward Returns for self-training
                    'Ret_2b': row.get('Ret_2b', 0),
                    'Ret_3b': row.get('Ret_3b', 0),
                    'Ret_5b': row.get('Ret_5b', 0),
                    'Ret_8b': row.get('Ret_8b', 0),
                    'Ret_13b': row.get('Ret_13b', 0),
                    # Optimization factors
                    'F1_PriceMom': row.get('F1_PriceMom', 0),
                    'F2_VolQual': row.get('F2_VolQual', 0),
                    # Reversion features (engine.compute_ranking inputs) — let the
                    # alpha-health monitor re-rank each harvested cross-section to
                    # measure trailing realized IC. Mirrors the live screener row.
                    '_rev_ret2':   row.get('_rev_ret2'),
                    '_rev_ret5':   row.get('_rev_ret5'),
                    '_rev_dist5':  row.get('_rev_dist5'),
                    '_rev_dist10': row.get('_rev_dist10'),
                    '_rev_rngpos': row.get('_rev_rngpos'),
                    'ATR_Pct':     row.get('ATR_Pct'),
                    'Close':       row.get('Close'),
                })
            
        except Exception as e:
            console.failure(f"Range Analysis Failed: {ticker}", str(e))
            continue
            
    console.success(f"Successfully processed {len(data_dict)} symbols for historical depth")
    console.end_phase("EDGE MEASUREMENT")

    if not all_results:
        if _own_slot:
            progress_slot.empty()
        st.error("No results generated for the selected timeframe.")
        return

    ts_df = pd.DataFrame(all_results)
    ts_df['Date'] = pd.to_datetime(ts_df['Date'])
    ts_df = ts_df.sort_values('Date')

    # The reversion engine has no per-row confidence model — the historical dashboard's
    # Avg_Intel aggregation degrades gracefully when Intel_Confidence is absent (see
    # _aggregate_timeseries). The trailing realized IC is measured separately by the
    # alpha-health monitor (_measure_trailing_ic) from this same harvested panel.
    daily_agg, summary = _aggregate_timeseries(ts_df)

    console.summary("HISTORICAL RANGE SUMMARY", {
        "Universe": universe,
        "Universe Index": selected_index,
        "Historical Range": f"{start_date} to {end_date}",
        "Total Signals Generated": summary['total_signals'],
        "Avg Signal Strength": round(summary['avg_signal'], 2),
        "Bias Ratio (L/S)": round(summary['overall_ratio'], 2),
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
    }

    # Only clear our OWN bar. When sharing the Single-Date bar, the screener that
    # follows keeps rendering into it (the 40→100% phase).
    if _own_slot:
        progress_slot.empty()


# ══════════════════════════════════════════════════════════════════════════════
# TIMESERIES — AGGREGATION + DASHBOARD RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def _aggregate_timeseries(ts_df):
    """Aggregate per-(date, symbol) factor frame into daily metrics + summary stats.

    Pure function — used by both ``run_timeseries_analysis`` (for the console
    summary on harvest) and ``render_timeseries_dashboard`` (re-rendered on every
    Streamlit run from session state, so sidebar interactions don't lose the view).
    """
    daily_agg = ts_df.groupby('Date').agg({
        'Signal': 'mean',
        'CVD': 'mean',
        'CVD_Slope': 'mean',
        'LongSignal': 'sum',
        'ShortSignal': 'sum',
        'Zone': lambda x: x.value_counts().idxmax() if len(x) > 0 else 'Neutral',
        'Regime': lambda x: x.value_counts().idxmax() if len(x) > 0 else 'NEUTRAL',
        'HMM_Bull': 'mean',
        'HMM_Bear': 'mean',
        'Vol_Regime': lambda x: x.value_counts().idxmax() if len(x) > 0 else 'NORMAL',
        'Change_Point': 'sum',
        'Regime_Confidence': 'mean',
    })

    daily_agg['TotalSignals'] = daily_agg['LongSignal'] + daily_agg['ShortSignal']
    daily_agg['L_S_Ratio']    = np.where(
        daily_agg['ShortSignal'] == 0,
        np.nan,                          # undefined (all longs, no shorts) — show as NaN in charts
        daily_agg['LongSignal'] / daily_agg['ShortSignal'],
    )
    daily_agg['Flow_Strength'] = daily_agg['Signal'].abs()

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

    # Mean Signal-Intelligence confidence of the fired signals each day (Intel is
    # NaN on non-fired bars, so this is a conviction-of-firing-signals breadth read).
    if 'Intel_Confidence' in ts_df.columns:
        daily_agg['Avg_Intel'] = ts_df.groupby('Date')['Intel_Confidence'].mean()
    else:
        daily_agg['Avg_Intel'] = np.nan

    summary = {
        'total_signals':       int(daily_agg['TotalSignals'].sum()),
        'total_buys':          int(daily_agg['LongSignal'].sum()),
        'total_sells':         int(daily_agg['ShortSignal'].sum()),
        'avg_signal':          float(daily_agg['Signal'].mean()),
        'overall_ratio':       float(daily_agg['LongSignal'].sum() / max(daily_agg['ShortSignal'].sum(), 1)),
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
    ui.render_section_header(f"Historical Range ({range_label})", icon="history", accent="violet")

    # ── Summary metric row (6 cards, mirrors single-date / pulse cadence) ──
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        ui.render_metric_card("Total Signals", str(summary['total_signals']),
                              f"{summary['total_buys']} long · {summary['total_sells']} short", "info")
    with c2:
        ui.render_metric_card("Avg Distribution", f"{summary['avg_oversold']:.1f}%",
                              timeframe_label, "success")
    with c3:
        ui.render_metric_card("Avg Accumulation", f"{summary['avg_overbought']:.1f}%",
                              timeframe_label, "danger")
    with c4:
        ui.render_metric_card("Period Regime", summary['dominant_regime'],
                              f"Bull: {summary['avg_bull_regime']:.0f}% | Bear: {summary['avg_bear_regime']:.0f}%",
                              "warning")
    with c5:
        ui.render_metric_card("L/S Ratio", f"{summary['overall_ratio']:.2f}",
                              f"{'Bullish' if summary['overall_ratio'] > 1 else 'Bearish'} bias", "info")
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
        ui.render_section_header("Flow-Zone Extremes",
                                 "Distribution / Accumulation Breadth Over Time",
                                 icon="activity", accent="cyan")
        fig_zones = go.Figure()
        fig_zones.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Oversold_Pct'],
                                       mode='lines', name='Distribution %',
                                       fill='tozeroy', fillcolor='rgba(52,211,153,0.12)',
                                       line=dict(color='#2DD4A8', width=2)))
        fig_zones.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Overbought_Pct'],
                                       mode='lines', name='Accumulation %',
                                       fill='tozeroy', fillcolor='rgba(251,113,133,0.12)',
                                       line=dict(color='#E8555A', width=2)))
        _pct_raw = max(daily_agg['Oversold_Pct'].max(), daily_agg['Overbought_Pct'].max())
        _pct_raw = float(_pct_raw) if pd.notna(_pct_raw) and np.isfinite(_pct_raw) else 0.0
        ymax = max(_pct_raw * 1.15, 5.0)   # floor at 5 so axis always renders sensibly
        fig_zones.update_layout(title='', height=350, hovermode='x unified',
                                yaxis=dict(range=[0, ymax], title='% of Universe'))
        apply_chart_theme(fig_zones)
        st.plotly_chart(fig_zones, width='stretch', key='chart_zones')

        st.markdown("<br>", unsafe_allow_html=True)
        ui.render_section_header("Signal Count by Date", "Long vs Short Signal Count per Session",
                                 icon="bar-chart", accent="info")
        fig_counts = go.Figure()
        fig_counts.add_trace(go.Bar(x=daily_agg.index, y=daily_agg['LongSignal'],
                                    name='Long Signals',
                                    marker=dict(color='#2DD4A8', line=dict(color='#2DD4A8', width=1))))
        fig_counts.add_trace(go.Bar(x=daily_agg.index, y=daily_agg['ShortSignal'],
                                    name='Short Signals',
                                    marker=dict(color='#E8555A', line=dict(color='#E8555A', width=1))))
        fig_counts.update_layout(title='', height=300, hovermode='x unified', barmode='group')
        apply_chart_theme(fig_counts)
        st.plotly_chart(fig_counts, width='stretch', key='chart_signal_counts')

    # ── TAB 2 · Transaction Dynamics ───────────────────────────────────────
    with tab2:
        ui.render_section_header("Transaction Signal Trends",
                                 "Buy / Sell Signal Counts Over Time",
                                 icon="zap", accent="emerald")
        fig_signals = go.Figure()
        fig_signals.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['LongSignal'],
                                         mode='lines+markers', name='Long Signals',
                                         line=dict(color='#2DD4A8', width=2),
                                         marker=dict(size=6, color='#2DD4A8')))
        fig_signals.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['ShortSignal'],
                                         mode='lines+markers', name='Short Signals',
                                         line=dict(color='#E8555A', width=2),
                                         marker=dict(size=6, color='#E8555A')))
        fig_signals.update_layout(title='', height=300, hovermode='x unified')
        apply_chart_theme(fig_signals)
        st.plotly_chart(fig_signals, width='stretch', key='chart_signals_overtime')

        st.markdown("<br>", unsafe_allow_html=True)
        ui.render_section_header("Flow Breadth", "Accumulation vs Distribution Over Time",
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
        ui.render_section_header("Aggregate Flow Skew", "Average Delta Z-Score Over Time",
                                 icon="activity", accent="rose")
        # Signal = per-name Delta_Z (clipped ±5); its daily cross-sectional MEAN
        # concentrates near zero (σ ≈ 1/√N ≈ 0.08 for ~150 names). The extreme
        # bands sit at ±0.25 (~3σ of that mean) — the old ±20 bands / ±80 axis
        # were sized for the removed WT1 ±100 oscillator and rendered a dead
        # flatline with never-triggering colors.
        _sig_band = 0.25
        colors = ['#2DD4A8' if v < -_sig_band else '#E8555A' if v > _sig_band else '#64748B'
                  for v in daily_agg['Signal']]
        fig_avg = go.Figure()
        fig_avg.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Signal'].clip(lower=0),
                                     fill='tozeroy', fillcolor='rgba(232,85,90,0.05)',
                                     line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig_avg.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Signal'].clip(upper=0),
                                     fill='tozeroy', fillcolor='rgba(45,212,168,0.05)',
                                     line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig_avg.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Signal'],
                                     mode='lines+markers', name='Avg Δ-Z',
                                     line=dict(color='#D4A853', width=2),
                                     marker=dict(size=6, color=colors)))
        fig_avg.add_hline(y=_sig_band,  line=dict(color='rgba(239,68,68,0.5)', width=1, dash='dash'))
        fig_avg.add_hline(y=-_sig_band, line=dict(color='rgba(16,185,129,0.5)', width=1, dash='dash'))
        fig_avg.add_hline(y=0,   line=dict(color='rgba(255,255,255,0.3)', width=1))
        _sig_span = float(np.nanmax(np.abs(daily_agg['Signal']))) if len(daily_agg) else 0.0
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
            ui.render_section_header("Distribution Metrics", "Signal Statistics",
                                     icon="database", accent="rose")
            signal_stats = {
                "Metric": ["Mean Signal", "Median Signal", "Min Signal", "Max Signal", "Std Dev"],
                "Value": [f"{daily_agg['Signal'].mean():.2f}",
                          f"{daily_agg['Signal'].median():.2f}",
                          f"{daily_agg['Signal'].min():.2f}",
                          f"{daily_agg['Signal'].max():.2f}",
                          f"{daily_agg['Signal'].std():.2f}"],
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
        display_cols = ['Date', 'LongSignal', 'ShortSignal', 'Signal',
                        'Oversold_Pct', 'Overbought_Pct',
                        'Regime_Bull_Pct', 'Regime_Bear_Pct', 'Change_Point']
        _has_intel = 'Avg_Intel' in display_ts.columns and display_ts['Avg_Intel'].notna().any()
        if _has_intel:
            display_cols.append('Avg_Intel')
        display_ts = display_ts[display_cols]
        display_ts.columns = ['Date', 'Long Sig', 'Short Sig', 'Avg Signal',
                              'Distribution %', 'Accumulation %',
                              'Bull Regime %', 'Bear Regime %', 'Change Pts'] + (['Avg Conviction'] if _has_intel else [])
        st.dataframe(
            display_ts, width='stretch', hide_index=True,
            column_config={
                'Date':         st.column_config.TextColumn(help="Trading day (YYYY-MM-DD)."),
                'Long Sig':     st.column_config.NumberColumn(help="Daily count of symbols firing a Set A bullish signal (delta divergence with thrust above the mean axis)."),
                'Short Sig':    st.column_config.NumberColumn(help="Daily count of symbols firing a Set A bearish signal (delta divergence with thrust below the mean axis)."),
                'Avg Signal':   st.column_config.NumberColumn(help="Cross-sectional mean of Delta_Z (signed delta z-score, clipped ±5) on this day. The daily mean concentrates near 0; ±0.25 is already a strongly one-sided tape."),
                'Distribution %': st.column_config.NumberColumn(help="Percent of universe in Distribution / Distribution+ flow zones (net selling flow)."),
                'Accumulation %': st.column_config.NumberColumn(help="Percent of universe in Accumulation / Accumulation+ flow zones (net buying flow)."),
                'Bull Regime %':st.column_config.NumberColumn(help="Percent of universe with HMM regime label containing 'BULL'."),
                'Bear Regime %':st.column_config.NumberColumn(help="Percent of universe with HMM regime label containing 'BEAR'."),
                'Change Pts':   st.column_config.NumberColumn(help="Sum of Change_Point flags — count of symbols with a regime-state transition on this day."),
                'Avg Conviction': st.column_config.NumberColumn(help="Mean conviction of the surfaced long/short candidates on this day (0–1) — tail strength × alpha-health × regime.", format="%.2f"),
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

def run_correlation_analysis(universe, selected_index, target_ticker, lookback, method, timeframe, analysis_date=None):
    """Execute correlation analysis between universe constituents and a target asset.

    Returns a dict with correlation data, rolling correlations, prices, and returns,
    plus a confluence score (|correlation| × reversion rank strength × conviction).
    """
    if analysis_date is None:
        analysis_date = _today_ist()
    progress_slot = st.empty()
    progress_bar(progress_slot, 5, "Initializing Correlation Engine", "Fetching Market Data")

    try:
        # Fetch universe symbols
        if universe == "India Indexes":
            stock_list, msg = get_index_stock_list(selected_index)
        elif universe == "Global Indexes":
            stock_list, msg = get_global_index_symbols()
        elif universe == "US Indexes":
            stock_list, msg = get_us_index_symbols(selected_index)
        elif universe == "Commodities":
            stock_list, msg = get_commodity_symbols(None)
        elif universe == "Currency":
            stock_list, msg = get_currency_symbols(None)
        elif universe == "Crypto":
            stock_list, msg = get_crypto_symbols(None)
        elif universe == "ETF Index":
            stock_list, msg = get_etf_symbols()
        else:
            st.error(f"Universe '{universe}' not supported")
            return None

        if not stock_list:
            st.error(f"Failed to fetch universe symbols: {msg}")
            return None

        console.item("Symbols fetched", len(stock_list))

        progress_bar(progress_slot, 15, "Fetching OHLCV Data", f"Symbols: {len(stock_list)}")

        # ── Universe data from registry (shared pool with screener / intelligence) ──
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
            wrci_results = _sdf
        else:
            # Measure alpha-health BEFORE screening, exactly like the Single-Date path —
            # so the Confluence Score (built from Priority_Long/Short) ranks with the
            # live edge-scaled conviction rather than a stale / cold-start reading.
            _corr_reg_len, _corr_n1, _corr_n2 = 20, 10, 21
            _corr_levels = (80, 40, -80, -40)
            _corr_wt2_len, _corr_wt2_type = 20, "ALMA"
            _calib = st.session_state.get("_calib_settings") or {}
            # Share the correlation bar: alpha-health harvest (if any) → 60-75%,
            # screener → 75-90%. No separate harvest bar.
            _ensure_alpha_health(
                universe, selected_index, timeframe, analysis_date,
                _corr_reg_len, _corr_n1, _corr_n2, _corr_levels,
                _corr_wt2_len, _corr_wt2_type, _calib,
                external_progress_slot=progress_slot, progress_offset=60, progress_scale=15,
            )
            wrci_results = run_screener_analysis(
                universe, selected_index, analysis_date,
                _corr_reg_len, _corr_n1, _corr_n2, _corr_levels, timeframe,
                show_progress=False, external_progress_slot=progress_slot, progress_offset=75, progress_scale=15,
                wt2_len=_corr_wt2_len, wt2_type=_corr_wt2_type,
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

            # Get WRCI data if available — pull priorities from the screener's
            # already-computed reversion ranking output so Correlation Analysis
            # benefits from the live edge-scaled conviction.
            wrci_signal = np.nan
            wrci_zone = "—"
            wrci_signal_type = "Neutral"
            priority_long = np.nan
            priority_short = np.nan
            intel_conf = np.nan
            intel_source = ''
            if wrci_results is not None and len(wrci_results) > 0:
                wrci_row = wrci_results[wrci_results['SimpleName'] == symbol.replace('.NS', '').replace('^', '')]
                if len(wrci_row) > 0:
                    wrci_signal = wrci_row['Signal'].values[0]
                    wrci_zone = wrci_row['Zone'].values[0]
                    wrci_signal_type = wrci_row['SignalType'].values[0]
                    if 'Priority_Long' in wrci_row.columns:
                        priority_long = wrci_row['Priority_Long'].values[0]
                    if 'Priority_Short' in wrci_row.columns:
                        priority_short = wrci_row['Priority_Short'].values[0]
                    # Layer-2 Signal Intelligence for this symbol's fired signal (NaN
                    # when it didn't fire) — used below to penalize the confluence of
                    # likely false positives.
                    if 'Intel_Confidence' in wrci_row.columns:
                        intel_conf = wrci_row['Intel_Confidence'].values[0]
                    if 'Intel_Source' in wrci_row.columns:
                        intel_source = wrci_row['Intel_Source'].values[0]

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
                'WRCI_Signal': wrci_signal,
                'WRCI_Zone': wrci_zone,
                'WRCI_Signal_Type': wrci_signal_type,
                'Priority_Long':  priority_long,
                'Priority_Short': priority_short,
                'Intel_Confidence': intel_conf,
                'Intel_Source':     intel_source,
            })

        corr_df = pd.DataFrame(corr_data_list)
        if len(corr_df) == 0:
            st.error("No correlation data could be computed")
            console.item("Error", "Empty correlation DataFrame")
            return None

        corr_df = corr_df.sort_values('Corr_Current', key=abs, ascending=False)

        # ── Confluence score ──────────────────────────────────────────────
        # |Corr| × normalized max(|Priority_Long|, |Priority_Short|), where the
        # priorities are the cross-sectional reversion scores (engine.py) — so
        # the confluence ranking carries the live, alpha-health-scaled read.
        # If Priority columns are missing entirely (defensive), drop back to
        # the legacy Delta_Z-based formula.
        if 'Priority_Long' in corr_df.columns and corr_df['Priority_Long'].notna().any():
            abs_pri = pd.concat(
                [corr_df['Priority_Long'].abs(), corr_df['Priority_Short'].abs()],
                axis=1,
            ).max(axis=1).fillna(0)
            pri_norm = abs_pri / max(abs_pri.max(), 1e-6)        # [0, 1]
            corr_df['Priority_Strength'] = pri_norm
            corr_df['Confluence_Score'] = (corr_df['Corr_Current'].abs() * pri_norm).clip(0.0, 1.0)
            console.item("Confluence formula", "|Corr| × reversion rank strength")

            # ── Conviction penalty ───────────────────────────────────────────
            # A high-correlation, high-priority name still deserves less weight when
            # the engine's conviction in its assigned side is low (thin tail, dormant
            # alpha-health, or hostile vol regime). Penalize the confluence of FIRED
            # signals by conviction: factor = 0.5 + 0.5·Conviction, so full conviction
            # is unchanged and near-zero conviction is halved (not zeroed — the
            # correlation read still carries information). Non-fired rows (Intel NaN)
            # are untouched.
            if 'Intel_Confidence' in corr_df.columns and corr_df['Intel_Confidence'].notna().any():
                _intel = corr_df['Intel_Confidence']
                _factor = np.where(_intel.notna(), 0.5 + 0.5 * _intel.fillna(0.0), 1.0)
                corr_df['Confluence_Raw'] = corr_df['Confluence_Score']
                corr_df['Confluence_Score'] = (corr_df['Confluence_Score'] * _factor).clip(0.0, 1.0)
                _n_penalized = int((_intel.notna() & (_intel < 0.5)).sum())
                console.item("Confluence formula", "|Corr| × reversion rank × (0.5 + 0.5·Conviction)")
                console.item("Low-conviction signals", f"{_n_penalized} candidate(s) below 0.50 conviction")
        else:
            # Normalise by the 95th-percentile absolute oscillator value so the
            # scale adapts to the current universe (weekly ≈ ±40, crypto ≈ ±200)
            # rather than assuming a fixed ±80 range.
            _osc_p95 = max(corr_df['WRCI_Signal'].abs().quantile(0.95), 1.0)
            corr_df['Confluence_Score'] = (
                corr_df['Corr_Current'].abs() * (corr_df['WRCI_Signal'].fillna(0).abs() / _osc_p95)
            ).clip(0.0, 1.0)
            console.item(
                "Confluence formula",
                f"|Corr| × |WRCI_Signal|/{_osc_p95:.1f} (fallback — no priority data; scale from p95)"
            )

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

def _side_palette(side: str) -> dict:
    """Side-keyed accent colors for long/short signal tables."""
    if side == 'long':
        return {
            "accent_light": _GREEN,
            "border_color": "rgba(45, 212, 168, 0.3)",
            "header_bg":    "rgba(45, 212, 168, 0.15)",
        }
    return {
        "accent_light": _RED,
        "border_color": "rgba(232, 85, 90, 0.3)",
        "header_bg":    "rgba(232, 85, 90, 0.15)",
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


def _build_confluence_table_html(df: pd.DataFrame) -> str:
    """Build ranked HTML table for confluence setups.

    Displays symbol, correlation, zone, signal, actual/expected/divergence, and confluence score.

    Returns: Complete HTML document string ready for st.components.v1.html().
    """
    table_rows = []
    if df.empty:
        table_rows.append(f"""
        <tr>
            <td colspan="10" style="
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
            zone = html.escape(str(row.get('WRCI_Zone', 'Neutral')))
            signal_type = html.escape(str(row.get('WRCI_Signal_Type', '—')))
            actual = float(row.get('PctChange', 0))
            expected = float(row.get('Expected_Change', 0))
            divergence = float(row.get('Divergence', 0))
            confluence = float(row.get('Confluence_Score', 0))
            
            # Format the Intel cell
            intel_cell, _ = _intel_cell_and_style(row.get('Intel_Confidence'), row.get('Intel_Source', ''), 'Off', 0.0)

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
                <td class="numeric">{zone}</td>
                <td class="numeric">{signal_type}</td>
                <td class="numeric" style="color: #94A3B8;">{actual:+.2f}%</td>
                <td class="numeric" style="color: #94A3B8;">{expected:+.2f}%</td>
                <td class="numeric" style="color: {div_color}; font-weight: 600;">{divergence:+.2f}%</td>
                {intel_cell}
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
                <th>Zone</th>
                <th>Type</th>
                <th class="numeric" title="Symbol's price change on the analysis date">Actual %</th>
                <th class="numeric" title="Expected move = target return × beta (rolling correlation × vol ratio over the lookback)">Expected %</th>
                <th class="numeric" title="Divergence = Actual − Expected (positive = outperforming expectation)">Div %</th>
                <th class="numeric" title="Conviction in the assigned side (0–1)">Intel</th>
                <th class="numeric" title="Confluence = |Correlation| × normalised Priority strength [0–1]">Confluence</th>
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

    tab1, tab2, tab3 = st.tabs([
        "Correlation Dashboard",
        "Trade Intelligence",
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
            "Trade Intelligence",
            "Confluence: Correlation × Reversion Rank",
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
                Each setup type is ranked by <span style="color:#38BDF8; font-weight:600;">Confluence Score</span> (0-1).
                Highest rank = strongest opportunity. Look for: <span style="font-weight:600;">(1) Score >0.7</span>,
                <span style="font-weight:600;">(2) |Div %| >3%</span>, <span style="font-weight:600;">(3) a confirming flow Zone (Accumulation+/Distribution+)</span>
            </div>
            <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:0.5rem; margin-top:0.75rem;">
                <div style="font-family:var(--data); font-size:0.65rem; color:var(--ink-secondary);">
                    <span style="color:#38BDF8; font-weight:600;">Corr</span> — Correlation strength
                </div>
                <div style="font-family:var(--data); font-size:0.65rem; color:var(--ink-secondary);">
                    <span style="color:#38BDF8; font-weight:600;">Zone</span> — Flow zone (Accumulation / Distribution)
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
            zone = row['WRCI_Zone']

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
                        st.components.v1.html(_build_confluence_table_html(top_half), height=100 + len(top_half) * 48)
                with col_right:
                    st.markdown(f"""<p style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem; font-weight:600;
                                   text-transform:uppercase; letter-spacing:0.1em; color:{config['color']};
                                   margin:0 0 0.4rem 0; display:flex; align-items:center; gap:0.35rem;">
                        Also Considered</p>""", unsafe_allow_html=True)
                    bottom_half = setup_data.iloc[5:10]
                    if len(bottom_half) > 0:
                        st.components.v1.html(_build_confluence_table_html(bottom_half), height=100 + len(bottom_half) * 48)
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

def _intel_filter_active():
    """Read the Meta Filter settings from session state.

    Returns (mode, threshold) where mode ∈ {'Off','Dim','Hide'}. Defaults to
    'Dim' at 0.45 (set in the sidebar expander); it applies only to fired-signal
    tables — never to the full-universe priority ranking (whose non-fired rows
    have no confidence score).
    """
    mode = st.session_state.get("intel_filter_mode", "Off")
    if mode not in ("Off", "Dim", "Hide"):
        mode = "Off"
    try:
        thr = float(st.session_state.get("intel_filter_threshold", 0.45))
    except (TypeError, ValueError):
        thr = 0.45
    return mode, thr


def _intel_cell_and_style(conf, source, mode, thr):
    """Render an Intel-Confidence table cell + optional row dim style.

    Returns (td_html, tr_style). conf is the row's Intel_Confidence (0–1 or
    NaN/None for non-fired). A filled ◆ marks a calibrated (Layer-2) score, a
    hollow ◇ the Layer-1 heuristic. In 'Dim' mode, rows below threshold are
    greyed; 'Hide' is handled upstream by dropping the rows.
    """
    if conf is None or pd.isna(conf):
        return '<td class="numeric" style="color:#4B5563;">—</td>', ''
    c = float(conf)
    if source == 'calibrated':
        # Calibrated → a real out-of-sample probability. Vivid semantic bands and
        # a % format so the value reads as P(true).
        if   c >= 0.65: col = '#2DD4A8'
        elif c >= 0.50: col = '#A3E635'
        elif c >= 0.35: col = '#D4A853'
        elif c >= 0.20: col = '#FB923C'
        else:           col = '#E8555A'
        txt   = f'◆ {c*100:.0f}%'
        title = f'Calibrated P(true) ≈ {c*100:.0f}%'
    else:
        # Heuristic → an indicative 0–1 index, NOT a probability. Muted (desaturated)
        # bands + "~" prefix so it is never read on the calibrated scale.
        if   c >= 0.65: col = '#7FA8A0'
        elif c >= 0.50: col = '#9DB07A'
        elif c >= 0.35: col = '#B0A079'
        elif c >= 0.20: col = '#B58E6E'
        else:           col = '#B57A7E'
        txt   = f'~{c:.2f}'
        title = f'Heuristic estimate {c:.2f} — not a calibrated probability'
    cell = (f'<td class="numeric" style="color:{col}; font-weight:700;" title="{title}">{txt}</td>')
    style = 'opacity:0.4;' if (mode == 'Dim' and c < thr) else ''
    return cell, style


def _meta_intel_cell(conv, tier=None, source=''):
    """Render a Layer-3 Meta Intelligence table cell (tier-banded fused score).

    conv ∈ [0,1] or NaN/None (non-fired / no snapshot fusion → '—'). The fused
    score blends cross-sectional Priority rank × per-signal Intel confidence.
    source 'meta' (the calibrated fused model, active or advisory) shows a filled
    ◆; 'fallback' (rank × confidence) a hollow ◇. Colour follows the 0–3 tier band.
    """
    if conv is None or pd.isna(conv):
        return '<td class="numeric" style="color:#4B5563;">—</td>'
    c = float(conv)
    if tier is None or pd.isna(tier):
        tier = 1 + sum(c >= b for b in (0.35, 0.55, 0.70))
    t = int(tier)
    if   c >= 0.70: col = '#2DD4A8'
    elif c >= 0.55: col = '#A3E635'
    elif c >= 0.35: col = '#D4A853'
    else:           col = '#FB923C'
    is_meta = str(source) == 'meta'
    mark  = '◆' if is_meta else '◇'
    txt   = f'{mark} {c*100:.0f}%'
    title = (f'Meta Intelligence {c*100:.0f}% · tier {t}/3 · '
             + ('fused model (rank × confidence)' if is_meta else 'fallback rank × confidence'))
    return f'<td class="numeric" style="color:{col}; font-weight:700;" title="{title}">{txt}</td>'


def _status_cell(status) -> str:
    """Render a (label, color, title) status tuple as a small table cell."""
    label, color, title = (status if isinstance(status, (tuple, list)) and len(status) == 3
                           else ('—', '#4B5563', ''))
    _t = html.escape(str(title)) if title else ''
    return (f'<td class="numeric" style="color:{color}; font-weight:700; font-size:0.62rem;" '
            f'title="{_t}">{html.escape(str(label))}</td>')


def _context_status(fire_c, today_c, offset):
    """Has the regime/momentum context that made the signal good held up to today?

    Compares confidence at the fire bar vs today (both per-symbol). Returns
    (label, color, title). Orthogonal to price — this is about the thesis, not
    whether the move already ran.
    """
    if fire_c is None or pd.isna(fire_c):
        return ('—', '#4B5563', '')
    if offset == 0:
        return ('New', '#94a3b8', 'fired today')
    if today_c is None or pd.isna(today_c):
        return ('—', '#4B5563', '')
    d = float(today_c) - float(fire_c)
    title = f'context: fire {float(fire_c):.2f} → now {float(today_c):.2f} (Δ{d:+.2f})'
    if float(today_c) < 0.20 or d <= -0.30:
        return ('Stale', '#E8555A', title)
    if d >= 0.05:
        return ('Confirmed', '#2DD4A8', title)
    if d <= -0.12:
        return ('Fading', '#FB923C', title)
    return ('Holding', '#A3E635', title)


def _entry_status(window, side: str, offset: int, row):
    """Has price already run since the signal fired — i.e. is the entry now late?

    Directional move from the fire bar to today, normalized by the symbol's own
    recent return volatility × √(bars elapsed) so the bands are asset-agnostic
    (σ units). Returns (label, color, title). Orthogonal to context.
    """
    if offset == 0:
        return ('Now', '#94a3b8', 'fresh — fired today')
    if window is None or len(window) == 0 or 'Close' not in getattr(window, 'columns', []):
        return ('—', '#4B5563', '')
    fidx = len(window) - 1 - offset
    if fidx < 0:
        return ('—', '#4B5563', '')
    fire_close = window['Close'].iloc[fidx]
    today_close = window['Close'].iloc[-1]
    if not (pd.notna(fire_close) and pd.notna(today_close) and fire_close > 0):
        return ('—', '#4B5563', '')
    side_sign = 1.0 if side == 'long' else -1.0
    dm = (float(today_close) - float(fire_close)) / float(fire_close) * side_sign
    rv = row.get('RetVol20')
    scale = (float(rv) * (offset ** 0.5)) if (rv is not None and pd.notna(rv) and float(rv) > 0) else None
    if scale and scale > 0:
        sig = dm / scale
        title = f'entry: {dm*100:+.1f}% since fire ({sig:+.1f}σ)'
        if sig <= -1.0: return ('Adverse', '#E8555A', title)
        if sig >= 1.5:  return ('Extended', '#FB923C', title)
        if sig >= 0.5:  return ('Running', '#5EBFA8', title)
        return ('Open', '#2DD4A8', title)
    # Fallback when volatility is unavailable — crude fixed % bands.
    title = f'entry: {dm*100:+.1f}% since fire'
    if dm <= -0.03: return ('Adverse', '#E8555A', title)
    if dm >= 0.06:  return ('Extended', '#FB923C', title)
    if dm >= 0.02:  return ('Running', '#5EBFA8', title)
    return ('Open', '#2DD4A8', title)


def _active_model_sig() -> str:
    """Cheap signature of the active confidence model — the reversion engine has none."""
    return 'heuristic'


def _cached_conf_series(symbol, window, side: str, condition_set: str):
    """Per-bar confidence for (symbol, side, set), memoized for the current screener run.

    The reversion engine has no per-bar confidence model, so there is no fire-bar
    confidence series to recompute — return empty arrays. Callers fall back to the
    row's own snapshot Intel_Confidence / Conviction. Kept memoized for API parity;
    the cache is cleared when a new screener run replaces intel_windows.
    """
    cache = st.session_state.setdefault("intel_fire_cache", {})
    key = (symbol, side, condition_set, _active_model_sig())
    hit = cache.get(key)
    if hit is not None:
        return hit
    res = (np.array([]), np.array([]))
    cache[key] = res
    return res


def _fire_bar_metrics(window, side: str, condition_set: str, offset: int, row) -> dict:
    """Per-signal fire-bar metrics: confidence (at fire), context decay, entry state.

    `window` is the symbol's recent-bar feature frame (chronological, last row =
    snapshot). `offset` 0 = Today … 4 = Within-5d. The Intel confidence is read at
    the fire bar; Context compares it to today; Entry measures the price move
    since firing. Falls back to snapshot confidence when the window is missing.
    """
    conf, src = np.nan, ''
    ctx = ('—', '#4B5563', '')
    confs = np.array([])
    if window is not None and len(window) and condition_set in ('A', 'B'):
        confs, srcs = _cached_conf_series(row.get('Symbol'), window, side, condition_set)
        fidx = len(confs) - 1 - offset
        if 0 <= fidx < len(confs):
            conf = confs[fidx]
            src = srcs[fidx] if fidx < len(srcs) else ''
            ctx = _context_status(conf, confs[-1], offset)
    if pd.isna(conf):   # window missing / out of range → snapshot fallback
        c = row.get('Intel_Confidence')
        conf = c if c is not None else np.nan
        src = row.get('Intel_Source', '') or ''
    entry = _entry_status(window, side, offset, row)
    return {'conf': conf, 'src': src, 'ctx': ctx, 'entry': entry}


def _bucket_signals_by_age(results_df: pd.DataFrame, side: str = 'long', condition_set: str = 'A', timeframe: str = 'Daily') -> dict:
    """Bucket signals by age (Today, 1d, 2d, 3d, 5d) with stats for timeline display.

    condition_set: 'A' = Delta Divergence (LA_/SA_), 'B' = Clamp Cross (LB_/SB_)
    timeframe: 'Daily' or 'Weekly' — determines age label names
    """
    if condition_set == 'A':
        prefix = 'LA' if side == 'long' else 'SA'
    elif condition_set == 'B':
        prefix = 'LB' if side == 'long' else 'SB'
    else:
        prefix = 'L' if side == 'long' else 'S'
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
        age_labels[4]: f"{prefix}_5d"
    }
    seen = set()

    # Fire-bar Intel scoring: each signal is scored at the bar it fired (its age
    # offset), not at the snapshot date. The result is attached to the row as
    # _fire_conf / _fire_src and reused for both Layer-3 Hide and table display.
    _filter_mode, _filter_thr = _intel_filter_active()
    _windows = st.session_state.get("intel_windows", {})

    for _offset, age in enumerate(buckets.keys()):
        col = col_map[age]
        subset = results_df[(results_df[col] == target_indicator) & (~results_df['Symbol'].isin(seen))]
        for _, r in subset.iterrows():
            sym = r['Symbol']
            m = _fire_bar_metrics(_windows.get(sym), side, condition_set, _offset, r)
            # Intel DISPLAY value — the real per-signal Intel confidence (fire-bar for
            # aged signals, snapshot for today). This is what the Intel column shows; it
            # is kept distinct from the Meta filter value below so the two columns never
            # mirror each other.
            fc = m['conf']
            _fc_src = m['src']
            # FILTER decision value (drives Hide here + Dim styling in the table) — the
            # Meta score on fired signals (rank × edge-scaled conviction), else the
            # fire-bar Intel for aged signals. The old "probation" gate keyed off a
            # Meta_Active column the removed fused model emitted; the reversion engine
            # never emits it, so gating Hide on it made Hide silently inert — the
            # engine's Meta score is always the live conviction fusion and may hide.
            _mv = r.get('Meta_Score')
            if _mv is not None and not pd.isna(_mv):
                _filter_val = float(_mv)
            else:
                _filter_val = fc
            _filter_may_hide = True
            # Hide mode — drop low-score signals (and don't let an older fire of the same
            # symbol resurface in a later bucket).
            if (_filter_mode == "Hide" and _filter_may_hide and _filter_val is not None
                    and not pd.isna(_filter_val) and _filter_val < _filter_thr):
                seen.add(sym)
                continue
            r = r.copy()
            r['_fire_conf'] = fc            # Intel column — REAL Intel confidence
            r['_fire_src'] = _fc_src
            r['_filter_val'] = _filter_val  # Meta (today) / Intel (aged) — drives Dim styling
            r['_ctx'] = m['ctx']      # (label, color, title) — context decay
            r['_entry'] = m['entry']  # (label, color, title) — move exhaustion
            buckets[age].append(r)
            seen.add(sym)

    # Compute stats for each bucket
    stats = {}
    for age, rows in buckets.items():
        if rows:
            signals = [r['Signal'] for r in rows]
            pct_changes = [r.get('PctChange', 0) for r in rows]
            convictions = [abs(r.get('Delta_Z', 0)) for r in rows]
            avg_signal = np.mean(signals)
            avg_pct_change = np.mean(pct_changes)
            avg_conviction = np.mean(convictions)
            count = len(rows)
            stats[age] = {
                'count': count,
                'avg_signal': avg_signal,
                'avg_pct_change': avg_pct_change,
                'avg_conviction': avg_conviction,
                'rows': rows
            }
        else:
            stats[age] = {'count': 0, 'avg_signal': 0, 'avg_pct_change': 0, 'rows': []}

    # Calculate trend: are signals strengthening (newer) or weakening (older)?
    newest_label = age_labels[0]  # "Today" or "This Week"
    older_labels = age_labels[1:]  # Rest of the labels

    newest_avg = stats[newest_label]['avg_signal'] if stats[newest_label]['count'] > 0 else 0
    older_avg = np.mean([stats[age]['avg_signal'] for age in older_labels if stats[age]['count'] > 0]) if any(stats[age]['count'] for age in older_labels) else 0

    # Signal = Delta_Z (clipped ±5); bucket averages move on a ~±1 scale, so the
    # strengthening/weakening threshold is 0.5 — the old ±5 threshold (sized for
    # the removed ±100 oscillator) was unreachable and pinned the badge on "Stable".
    _TREND_EPS = 0.5
    if newest_avg > older_avg + _TREND_EPS:
        trend = f"{SVGS['UP'].replace('12','14').replace('12','14')} Strengthening"
        trend_color = "#2DD4A8"
    elif newest_avg < older_avg - _TREND_EPS:
        trend = f"{SVGS['DOWN'].replace('12','14').replace('12','14')} Weakening"
        trend_color = "#E8555A"
    else:
        trend = "— Stable"
        trend_color = "#D4A853"

    return buckets, stats, trend, trend_color


def _build_signal_table_html(stats: dict, side: str = 'long', timeframe: str = 'Daily') -> str:
    """Build organized HTML table for signals grouped by age with section headers."""
    _pal = _side_palette(side)
    accent_light = _pal["accent_light"]
    border_color = _pal["border_color"]
    header_bg    = _pal["header_bg"]
    _filter_mode, _filter_thr = _intel_filter_active()

    table_rows = []
    if timeframe == 'Weekly':
        age_order = ["This Week", "1 Week Ago", "2 Weeks Ago", "3 Weeks Ago", "Within 5 Weeks"]
    else:
        age_order = ["Today", "1 Day Ago", "2 Days Ago", "3 Days Ago", "Within 5 Days"]

    for age in age_order:
        if stats[age]['count'] == 0:
            continue

        # Section header for this age group
        avg_signal = stats[age]['avg_signal']
        avg_pct = stats[age].get('avg_pct_change', 0)
        avg_conv = stats[age].get('avg_conviction', 0)
        count = stats[age]['count']
        table_rows.append(f"""
        <tr style="background: {header_bg}; border-bottom: 2px solid {border_color};">
            <td colspan="13" style="padding: 0.75rem 1rem; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.8rem !important; font-weight: 700; color: {accent_light}; text-transform: uppercase; letter-spacing: 0.05em;">
                {age} · {count} signal{'s' if count != 1 else ''} · Avg |Δ-Z|: {avg_conv:+.1f} · Avg %: {avg_pct:+.1f}
            </td>
        </tr>
        """)

        # Data rows for this age group
        for row in stats[age]['rows']:
            symbol = html.escape(str(row.get('DisplayName', row.get('Symbol', ''))))
            price = float(row.get('Price', 0))
            pct_change = float(row.get('PctChange', 0))
            signal = float(row.get('Signal', 0))
            bar_delta = float(row.get('Bar_Delta', 0))
            cvd = float(row.get('CVD', 0))
            cvd_slope = float(row.get('CVD_Slope', 0))
            delta_z = float(row.get('Delta_Z', 0))
            abs_strength = float(row.get('Abs_Strength', 0))
            abs_color = _signed_color(abs_strength - 1.0, pos="#fbbf24", neg="#38bdf8")  # >1× = amber
            signal_type = str(row.get('SignalType', '-'))

            pct_color        = _signed_color(pct_change)
            cvd_color        = _signed_color(cvd)
            cvd_slope_color  = _signed_color(cvd_slope, pos="#4a9eff", neg="#D4A853")
            cvd_slope_arrow  = _delta_arrow(cvd_slope)

            # Intel Confirmation cell — fire-bar confidence (set on the row by
            # _bucket_signals_by_age), falling back to the snapshot value.
            _conf_val = row.get('_fire_conf', row.get('Intel_Confidence'))
            _conf_src = row.get('_fire_src', row.get('Intel_Source', ''))
            # Intel cell shows the REAL Intel confidence (no dim from it). The row Dim
            # styling keys off the Meta filter value (Meta today / Intel aged), so the
            # Intel and Meta columns stay independent.
            intel_cell, _ = _intel_cell_and_style(_conf_val, _conf_src, 'Off', 0.0)
            _fv = row.get('_filter_val', _conf_val)
            _row_style = ('opacity:0.4;' if (_filter_mode == 'Dim' and _fv is not None
                          and not pd.isna(_fv) and _fv < _filter_thr) else '')
            # Context (thesis decay) + Entry (move exhaustion) status cells.
            ctx_cell   = _status_cell(row.get('_ctx',   ('—', '#4B5563', '')))
            entry_cell = _status_cell(row.get('_entry', ('—', '#4B5563', '')))
            # Layer-3 Meta Intelligence (today's snapshot fusion; '—' for aged rows).
            meta_cell  = _meta_intel_cell(row.get('Meta_Score'), row.get('Meta_Tier'),
                                          row.get('Meta_Source', ''))

            table_rows.append(f"""
            <tr style="{_row_style}">
                <td class="symbol">{symbol}</td>
                <td class="numeric currency">{price:,.2f}</td>
                <td class="numeric" style="color: {pct_color}; font-weight: 600;">{pct_change:+.2f}%</td>
                <td class="numeric" style="color: {accent_light}; font-weight: 600;">{signal:+.2f}</td>
                <td class="numeric" style="color: #D4A853; font-weight: 600;">{_human_vol(bar_delta)}</td>
                <td class="numeric" style="color: {cvd_color}; font-weight: 600;">{_human_vol(cvd)}</td>
                <td class="numeric" style="color: {cvd_slope_color}; font-size: 0.65rem; font-weight: 600;">{cvd_slope_arrow}{_human_vol(abs(cvd_slope), signed=False)}</td>
                <td class="numeric" style="color: #4a9eff; font-weight: 600;">{delta_z:+.2f}</td>
                <td class="numeric" style="color: {abs_color}; font-weight: 600;">{abs_strength:.2f}×</td>
                {intel_cell}
                {meta_cell}
                {ctx_cell}
                {entry_cell}
            </tr>
            """)

    if not table_rows:
        table_rows.append(f"""
        <tr>
            <td colspan="13" style="text-align:center; color:#374151; font-family:'IBM Plex Mono',monospace;
                font-size:0.72rem; letter-spacing:0.06em; padding:2.25rem 1rem;">
                — no signals detected —
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
                    <th class="numeric">Signal</th>
                    <th class="numeric">Bar Δ</th>
                    <th class="numeric">CVD</th>
                    <th class="numeric">CVD Slope</th>
                    <th class="numeric">Δ-Z</th>
                    <th class="numeric">Absorp</th>
                    <th class="numeric">Intel</th>
                    <th class="numeric" title="Fused score · reversion rank × conviction">Meta</th>
                    <th class="numeric">Context</th>
                    <th class="numeric">Entry</th>
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

def _build_narrative_table_html(df: pd.DataFrame, side: str = 'long') -> str:
    """Build a simplified HTML table for Pulse Narrative mode showing all symbols."""
    _pal = _side_palette(side)
    border_color = _pal["border_color"]  # accent_light unused in this builder

    table_rows = []
    if df.empty:
        table_rows.append(f"""
        <tr>
            <td colspan="11" style="text-align:center; color:#374151; font-family:'IBM Plex Mono',monospace;
                font-size:0.72rem; letter-spacing:0.06em; padding:2.25rem 1rem;">
                — no data available —
            </td>
        </tr>""")
    else:
        for _, row in df.iterrows():
            symbol = html.escape(str(row.get('DisplayName', row.get('Symbol', ''))))
            price = float(row.get('Price', 0))
            pct_change = float(row.get('PctChange', 0))
            signal = float(row.get('Signal', 0))
            bar_delta = float(row.get('Bar_Delta', 0))
            cvd = float(row.get('CVD', 0))
            cvd_slope = float(row.get('CVD_Slope', 0))
            delta_z = float(row.get('Delta_Z', 0))
            abs_strength = float(row.get('Abs_Strength', 0))
            abs_color = _signed_color(abs_strength - 1.0, pos="#fbbf24", neg="#38bdf8")

            pct_color       = _signed_color(pct_change)
            cvd_color       = _signed_color(cvd)
            cvd_slope_color = _signed_color(cvd_slope, pos="#4a9eff", neg="#D4A853")
            cvd_slope_arrow = _delta_arrow(cvd_slope)
            # Layer-2 Signal Intelligence — same snapshot Intel cell used elsewhere
            # (calibrated % or muted ~heuristic; "—" when this symbol has no fired signal).
            intel_cell, _ = _intel_cell_and_style(
                row.get('Intel_Confidence'), row.get('Intel_Source', ''), 'Off', 0.0)
            meta_cell = _meta_intel_cell(row.get('Meta_Score'), row.get('Meta_Tier'),
                                         row.get('Meta_Source', ''))

            table_rows.append(f"""
            <tr>
                <td class="symbol" style="color: #F1F5F9;">{symbol}</td>
                <td class="numeric currency">{price:,.2f}</td>
                <td class="numeric" style="color: {pct_color}; font-weight: 600;">{pct_change:+.2f}%</td>
                <td class="numeric" style="color: #60A5FA; font-weight: 600;">{signal:+.2f}</td>
                <td class="numeric" style="color: #D4A853; font-weight: 600;">{_human_vol(bar_delta)}</td>
                <td class="numeric" style="color: {cvd_color}; font-weight: 600;">{_human_vol(cvd)}</td>
                <td class="numeric" style="color: {cvd_slope_color}; font-size: 0.65rem; font-weight: 600;">{cvd_slope_arrow}{_human_vol(abs(cvd_slope), signed=False)}</td>
                <td class="numeric" style="color: #4a9eff; font-weight: 600;">{delta_z:+.2f}</td>
                <td class="numeric" style="color: {abs_color}; font-weight: 600;">{abs_strength:.2f}×</td>
                {intel_cell}
                {meta_cell}
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
                    <th class="numeric">Signal</th>
                    <th class="numeric">Bar Δ</th>
                    <th class="numeric">CVD</th>
                    <th class="numeric">CVD Slope</th>
                    <th class="numeric">Δ-Z</th>
                    <th class="numeric">Absorp</th>
                    <th class="numeric">Intel</th>
                    <th class="numeric" title="Fused score · reversion rank × conviction">Meta</th>
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



def _build_signal_strength_table_html(df: pd.DataFrame, side: str = 'long') -> str:
    """Build ranked HTML table for top reversion candidates by conviction.

    Creates styled HTML table with colored accent for side (long=green, short=red),
    displaying symbol, price, signal magnitude, trend direction, and zone status.
    Prioritizes conviction (reversion tail strength × alpha-health) as the ranking metric.

    Returns: Complete HTML document string ready for st.components.v1.html().
    """
    _pal = _side_palette(side)
    accent_light = _pal["accent_light"]
    border_color = _pal["border_color"]

    table_rows = []
    if df.empty:
        table_rows.append(f"""
        <tr>
            <td colspan="14" style="
                text-align: center;
                color: #374151;
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.72rem;
                letter-spacing: 0.06em;
                padding: 2.25rem 1rem;
            ">— no signals detected —</td>
        </tr>
        """)
    else:
        for idx, (_, row) in enumerate(df.iterrows(), 1):
            symbol = html.escape(str(row.get('DisplayName', row.get('Symbol', ''))))
            price = float(row.get('Price', 0))
            pct_change = float(row.get('PctChange', 0))
            signal = float(row.get('Signal', 0))
            bar_delta = float(row.get('Bar_Delta', 0))
            cvd = float(row.get('CVD', 0))
            cvd_slope = float(row.get('CVD_Slope', 0))
            delta_z = float(row.get('Delta_Z', 0))
            abs_strength = float(row.get('Abs_Strength', 0))
            abs_color = _signed_color(abs_strength - 1.0, pos="#fbbf24", neg="#38bdf8")

            rank_str = f"{idx:02d}"
            pct_color       = _signed_color(pct_change)
            cvd_slope_color = _signed_color(cvd_slope)
            cvd_slope_arrow = _delta_arrow(cvd_slope)

            # v3 Metrics
            pct_rank = float(row.get(f'Priority_{side.capitalize()}_pct', 0))
            hmm_bull = float(row.get('HMM_Bull', 0.5))
            hmm_bear = float(row.get('HMM_Bear', 0.5))
            vol_reg  = str(row.get('Vol_Regime', 'NORMAL'))

            # Regime Logic
            regime_tag = "NEUTRAL"
            regime_color = "#94a3b8"
            if side == 'long':
                if hmm_bull > 0.7: regime_tag, regime_color = "BULL", _GREEN
                elif hmm_bull < 0.3: regime_tag, regime_color = "BEAR", _RED
            else:
                if hmm_bear > 0.7: regime_tag, regime_color = "BEAR", _RED
                elif hmm_bear < 0.3: regime_tag, regime_color = "BULL", _GREEN
                
            vol_color = {"LOW": "#60a5fa", "NORMAL": "#94a3b8", "HIGH": "#fbbf24", "EXTREME": "#f87171"}.get(vol_reg, "#94a3b8")

            # Snapshot Intel confidence (today's, per-symbol). This is a ranking
            # table with no canonical fire bar, so only the snapshot value applies
            # (no Context/Entry, no Dim). '—' where the symbol has no fired signal.
            _intel_cell, _ = _intel_cell_and_style(
                row.get('Intel_Confidence'), row.get('Intel_Source', ''), 'Off', 0.0)
            _meta_cell = _meta_intel_cell(row.get('Meta_Score'), row.get('Meta_Tier'),
                                          row.get('Meta_Source', ''))

            table_rows.append(f"""
            <tr>
                <td class="numeric" style="color: #D4A853; font-weight: 700;">{rank_str}</td>
                <td class="symbol">{symbol}</td>
                <td class="numeric" style="color: #4a9eff; font-weight: 700;">TOP {min(100.0, 101-pct_rank):,.1f}%</td>
                <td class="numeric currency">{price:,.2f}</td>
                <td class="numeric" style="color: {pct_color}; font-weight: 600;">{pct_change:+.2f}%</td>
                <td class="numeric" style="color: {accent_light}; font-weight: 600;">{signal:+.2f}</td>
                <td class="numeric" style="color: #D4A853; font-weight: 600;">{_human_vol(bar_delta)}</td>
                <td class="numeric" style="color: {cvd_slope_color}; font-size: 0.65rem; font-weight: 600;">{cvd_slope_arrow}{_human_vol(abs(cvd_slope), signed=False)}</td>
                <td class="numeric" style="color: #4a9eff; font-weight: 600;">{delta_z:+.2f}</td>
                <td class="numeric" style="color: {regime_color}; font-weight: 700; font-size: 0.65rem;">{regime_tag}</td>
                <td class="numeric" style="color: {vol_color}; font-weight: 700; font-size: 0.65rem;">{vol_reg}</td>
                <td class="numeric" style="color: {abs_color}; font-weight: 600;">{abs_strength:.2f}×</td>
                {_intel_cell}
                {_meta_cell}
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
                    <th class="numeric">Percentile</th>
                    <th class="numeric">Price</th>
                    <th class="numeric">% Change</th>
                    <th class="numeric">Signal</th>
                    <th class="numeric">Bar Δ</th>
                    <th class="numeric">CVD Slope</th>
                    <th class="numeric">Δ-Z</th>
                    <th class="numeric">Regime</th>
                    <th class="numeric">Vol</th>
                    <th class="numeric">Absorp</th>
                    <th class="numeric">Intel</th>
                    <th class="numeric" title="Fused score · reversion rank × conviction">Meta</th>
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
    ("Set A · Delta Divergence",   "amber",
     "A bar-level inferred-delta divergence (inferred_delta.pine) confirmed by the thrust's "
     "side of its mean axis (clamp.pine). Bullish: close down on POSITIVE inferred delta at a "
     "3-bar low, while net thrust (VWM) sits ABOVE the mean axis — sellers pushed price down "
     "but flow is turning up. Bearish: close up on NEGATIVE delta at a 3-bar high, with thrust "
     "BELOW the mean axis. The mean-axis gate keeps only divergences the force is confirming."),
    ("Set B · Clamp Cross", "violet",
     "VWM thrust re-entering its adaptive clamp band (clamp.pine). Bullish when thrust crosses "
     "UP through the LOWER clamp — a selling-pressure impact being absorbed back to normal; "
     "bearish when it crosses DOWN through the UPPER clamp — a buying-pressure impact absorbed. "
     "It marks the moment force returns inside the band. Strongest reads pair a Set A "
     "divergence with a Set B cross in the same direction."),
]


def _render_system_data_tab(results_df, analysis_date, universe=None, selected_index=None):
    """System Data tab — exports, raw factor frame, and the signal-type legend.

    Used by both Single Date and Pulse Narrative modes (their tab_raw share content).
    Universe context is threaded through so download filenames stay self-describing.
    """
    ui.render_section_header(
        "System Data",
        "Exports, raw factor frame, and reference legends",
        icon="database", accent="cyan",
    )

    # ── Downloads ─────────────────────────────────────────────────────────
    bull_df = results_df[results_df['Signal'] > 0] if 'Signal' in results_df.columns else results_df.iloc[0:0]
    bear_df = results_df[results_df['Signal'] < 0] if 'Signal' in results_df.columns else results_df.iloc[0:0]

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
                f"All {len(results_df)} symbols with every computed factor. "
                "Includes a Legend sheet defining each column: signal-set conditions (Set A / Set B), "
                "factor descriptions (Price Momentum, Vol Quality, Bar Delta, CVD, CVD Slope, Δ-Z), "
                "and flow-zone/regime definitions."
            ),
        )
    with dl2:
        st.download_button(
            "▲ Bullish Only (Excel)",
            data=to_excel(bull_df),
            file_name=build_download_filename(
                "bullish", universe=universe, selected_index=selected_index,
                dates=analysis_date, ext="xlsx",
            ),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width='stretch',
            key="sysdata_dl_bull",
            disabled=len(bull_df) == 0,
            help=f"{len(bull_df)} symbols with Signal > 0 (bullish-leaning composite).",
        )
    with dl3:
        st.download_button(
            "▼ Bearish Only (Excel)",
            data=to_excel(bear_df),
            file_name=build_download_filename(
                "bearish", universe=universe, selected_index=selected_index,
                dates=analysis_date, ext="xlsx",
            ),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width='stretch',
            key="sysdata_dl_bear",
            disabled=len(bear_df) == 0,
            help=f"{len(bear_df)} symbols with Signal < 0 (bearish-leaning composite).",
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Raw Data Table ────────────────────────────────────────────────────
    ui.render_section_header(
        "Raw Factor Frame",
        f"{len(results_df)} symbols · sorted by Priority Long",
        icon="list", accent="emerald",
    )
    cols = ["DisplayName", "Price", "Signal", "SignalType",
            "Intel_Confidence", "Intel_Stars", "Intel_Source",
            "Abs_Strength", "Priority_Long",
            "Priority_Long_pct", "F1_PriceMom", "F2_VolQual"]
    if "% Chng Since" in results_df.columns and results_df["% Chng Since"].notna().any():
        cols.insert(2, "% Chng Since")
    cols += ["Bar_Delta", "CVD", "CVD_Slope", "Delta_Z"]
    l_cols = [c for c in results_df.columns
              if c.startswith('L_') and (c[2:].replace('d', '').isdigit() or c == 'L_Today')]
    cols += sorted(l_cols)
    cols = [c for c in cols if c in results_df.columns]
    # Rename internal factor column names to domain-readable labels for display
    _col_display_names = {
        "DisplayName":       "Symbol",
        "SignalType":        "Set",
        "Intel_Confidence":  "Intel Conf",
        "Intel_Stars":       "Intel ★",
        "Intel_Source":      "Intel Src",
        "F1_PriceMom":       "Price Momentum",
        "F2_VolQual":        "Vol Quality",
        "Priority_Long_pct": "Long Priority %ile",
        "Bar_Delta":         "Bar Δ",
        "CVD_Slope":         "CVD Slope",
        "Delta_Z":           "Δ-Z",
        "Abs_Strength":      "Absorption ×",
    }
    display_frame = results_df[cols].sort_values("Priority_Long", ascending=False).rename(columns=_col_display_names)
    _sysdata_colcfg = {
        "Intel Conf": st.column_config.ProgressColumn(
            help=("Conviction in the row's assigned side, in [0,1] — cross-sectional tail "
                  "strength × live alpha-health × vol-regime suitability (engine.compute_ranking). "
                  "NaN/empty = the row has no assigned side (middle of the cross-section)."),
            format="%.2f", min_value=0.0, max_value=1.0,
        ),
        "Intel ★": st.column_config.NumberColumn(
            help="Conviction rating 1–5 on fixed bands (5 = strongest). 0 = no assigned side.",
        ),
    }
    st.dataframe(display_frame, width='stretch', height=500, column_config=_sysdata_colcfg)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Signal Type Reference ─────────────────────────────────────────────
    ui.render_section_header(
        "Signal Type Reference",
        "The two order-flow signal sets — Set A · Set B",
        icon="info", accent="amber",
    )
    # One column per reference card so the three cards widen equally and fill the
    # row — a fixed 4-column grid would leave an empty slot / dead space on the right.
    ref_cols = st.columns(len(_SIGNAL_TYPE_REFERENCE))
    accent_var_map = {
        "amber":  "var(--amber)",
        "violet": "var(--violet)",
        "cyan":   "var(--cyan)",
        "rose":   "var(--rose)",
    }
    # min-height + flex layout keeps all four cards visually equal regardless
    # of body text length. Without it cards stretch to their own content because
    # Streamlit's columns don't enforce a shared height.
    SIG_CARD_MIN_H = "11rem"
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
        console.item("Ranking engine", "Cross-Sectional Reversion v1 (alpha-health monitored)")

    # Render sidebar and get parameters + run button state
    sb = render_sidebar()
    # Local aliases preserve the existing main() body unchanged; the win is that
    # the data flow from the sidebar is now name-keyed (sb.field) instead of a
    # 16-element positional tuple unpack.
    universe           = sb.universe
    selected_index     = sb.selected_index
    analysis_date      = sb.analysis_date
    reg_len            = sb.reg_len
    wt_n1              = sb.wt_n1
    wt_n2              = sb.wt_n2
    wt2_len            = sb.wt2_len
    wt2_type           = sb.wt2_type
    levels             = sb.levels
    timeframe          = sb.timeframe
    mode               = sb.mode
    start_date         = sb.start_date
    end_date           = sb.end_date
    run_clicked        = sb.run_clicked
    corr_target_ticker = sb.corr_target_ticker
    corr_lookback      = sb.corr_lookback
    corr_method        = sb.corr_method
    calib_settings     = sb.calib_settings
    st.session_state["_calib_settings"] = calib_settings

    # Per-universe profile sync now happens inside render_sidebar() right
    # before the Passport renders, so the Passport reflects the new universe
    # in the same render frame.

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

        if mode in ("Single Date", "Pulse Narrative"):
            header_text = "Institutional Signal Screener" if mode == "Single Date" else "Pulse Narrative Analysis"
            console.header(f"SANKET TERMINAL — {header_text}", VERSION)
            console.main_header("ANALYSIS RUN START", {
                "Universe": universe, "Index": selected_index, "Timeframe": timeframe,
                "Target Date": analysis_date, "Mode": mode,
            })
            # ONE progress bar for the whole run. If a fresh edge measurement is
            # needed it harvests into the first 40% ("Measuring Live Edge"), then the
            # screener renders 40→100% ("Screening") into the SAME bar — no second bar.
            # When the reading is cached (no harvest), the screener owns the full 0→100%.
            _run_slot = st.empty()
            _calib_status = _ensure_alpha_health(
                universe, selected_index, timeframe, analysis_date,
                reg_len, wt_n1, wt_n2, levels, wt2_len, wt2_type, calib_settings,
                external_progress_slot=_run_slot, progress_offset=0, progress_scale=40,
            )
            _harvested = _calib_status in ("measured", "harvest_empty")
            results_df = run_screener_analysis(
                universe, selected_index, analysis_date,
                reg_len, wt_n1, wt_n2, levels, timeframe,
                wt2_len=wt2_len, wt2_type=wt2_type,
                external_progress_slot=_run_slot,
                progress_offset=(40 if _harvested else 0),
                progress_scale=(60 if _harvested else 100),
            )
            _run_slot.empty()
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
            # NOTE: _ensure_alpha_health returns cached|measured|harvest_empty; the
            # sidebar Engine Status card picks up a fresh reading on the next natural
            # rerun (the old "tuned" rerun branch died with the calibrator).
            _ = _calib_status

        elif mode == "Historical Range":
            console.header("SANKET TERMINAL — Bulk Range Intelligence", VERSION)
            run_timeseries_analysis(
                universe, selected_index, start_date, end_date,
                reg_len, wt_n1, wt_n2, levels, timeframe,
                wt2_len=wt2_len, wt2_type=wt2_type,
            )
            # Standalone harvest — no screener follows to consume the analyzed-frame
            # cache the harvest just populated, so release it here. (In the Single-Date
            # / Correlation flows the screener consumes then clears it itself.)
            _analyzed_cache_clear()

        elif mode == "Correlation Analysis":
            corr_data = run_correlation_analysis(
                universe, selected_index, corr_target_ticker,
                corr_lookback, corr_method, timeframe, analysis_date,
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

    if show_landing:
        ui.render_header("Sanket", "Market Signal Screener · Cross-Sectional Reversion Engine")
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
            for _col in ['LA_Today','LA_1d','LA_2d','LA_3d','LA_5d','SA_Today','SA_1d','SA_2d','SA_3d','SA_5d','LB_Today','LB_1d','LB_2d','LB_3d','LB_5d','SB_Today','SB_1d','SB_2d','SB_3d','SB_5d']:
                if _col not in results_df.columns: results_df[_col] = "—"

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
                        f"{_pn_n} / {_pn_total} symbols · {_pn_date} · Full universe ranking by reversion score",
                        icon="zap", accent="amber"
                    )
                    avg_delta = results_df['Delta_Z'].mean()
                    avg_cvd_slope = results_df['CVD_Slope'].mean()
                    strong_flow = len(results_df[results_df['Delta_Z'].abs() > 1.5])
                    bullish_bias = (results_df['CVD_Slope'] > 0).sum() / len(results_df) * 100 if len(results_df) > 0 else 0
                    m1, m2, m3, m4 = st.columns(4)
                    with m1: ui.render_metric_card("Universe Δ-Z", f"{avg_delta:+.2f}", "Avg signed delta z-score", "neutral")
                    with m2: ui.render_metric_card("Universe CVD Slope", _human_vol(avg_cvd_slope), "Avg 3-bar flow build/drain", "neutral")
                    with m3: ui.render_metric_card("Strong Flow", str(strong_flow), f"{strong_flow/len(results_df)*100 if len(results_df)>0 else 0:.0f}% Universe", "info")
                    with m4: ui.render_metric_card("Flow Bias", f"{bullish_bias:.0f}%", "CVD slope > 0", "success" if bullish_bias > 50 else "danger")
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    bull_narr_tab, bear_narr_tab = st.tabs(["Bullish Priority Ranking", "Bearish Priority Ranking"])
                    with bull_narr_tab:
                        bull_priority_df = results_df.sort_values('Priority_Long', ascending=False)
                        st.components.v1.html(_build_narrative_table_html(bull_priority_df, side='long'), height=min(1200, 150 + len(bull_priority_df) * 52))
                    with bear_narr_tab:
                        bear_priority_df = results_df.sort_values('Priority_Short', ascending=False)
                        st.components.v1.html(_build_narrative_table_html(bear_priority_df, side='short'), height=min(1200, 150 + len(bear_priority_df) * 52))

                # ════ Pulse Narrative · TAB 2: SIGNAL STRENGTH ═════════════════════════════
                with tab_strength:
                    ui.render_section_header(
                        "Order-Flow Signal Strength",
                        "Top 10 long / short candidates by reversion priority",
                        icon="zap", accent="amber",
                    )
                    pn_top_longs  = results_df.sort_values('Priority_Long',  ascending=False).head(10)
                    pn_top_shorts = results_df.sort_values('Priority_Short', ascending=False).head(10)

                    pn_avg_delta  = results_df['Delta_Z'].abs().mean()
                    pn_avg_abs    = results_df['Abs_Strength'].mean()
                    pn_strong_p   = len(results_df[results_df['Delta_Z'].abs() > 1.5])
                    pn_strong_t   = len(results_df[results_df['Abs_Strength'] > 1.8])

                    s1, s2, s3, s4 = st.columns(4)
                    with s1: ui.render_metric_card("Avg Δ-Z",        f"{pn_avg_delta:.2f}", "Signed delta z-score (abs)", "neutral")
                    with s2: ui.render_metric_card("Avg Absorption", f"{pn_avg_abs:.2f}×",  "|Δ| vs 20-bar avg",          "neutral")
                    with s3: ui.render_metric_card("Strong Flow",    str(pn_strong_p),
                                                   f"{pn_strong_p/len(results_df)*100:.0f}% of universe", "info")
                    with s4: ui.render_metric_card("Absorption Hits", str(pn_strong_t),
                                                   f"{pn_strong_t/len(results_df)*100:.0f}% of universe", "info")

                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    pn_l, pn_s = st.columns(2)
                    with pn_l:
                        st.markdown(
                            '<p style="font-family:\'IBM Plex Mono\',monospace; font-size:0.62rem; '
                            'font-weight:600; text-transform:uppercase; letter-spacing:0.1em; '
                            'color:var(--emerald); margin:0 0 0.4rem 0;">Top 10 Bullish</p>',
                            unsafe_allow_html=True,
                        )
                        st.components.v1.html(
                            _build_signal_strength_table_html(pn_top_longs, side='long'),
                            height=150 + len(pn_top_longs) * 55,
                        )
                    with pn_s:
                        st.markdown(
                            '<p style="font-family:\'IBM Plex Mono\',monospace; font-size:0.62rem; '
                            'font-weight:600; text-transform:uppercase; letter-spacing:0.1em; '
                            'color:var(--rose); margin:0 0 0.4rem 0;">Top 10 Bearish</p>',
                            unsafe_allow_html=True,
                        )
                        st.components.v1.html(
                            _build_signal_strength_table_html(pn_top_shorts, side='short'),
                            height=150 + len(pn_top_shorts) * 55,
                        )

                # ════ Pulse Narrative · TAB 3: SYSTEM DATA ════════════════════════════════
                with tab_raw:
                    _render_system_data_tab(results_df, analysis_date,
                                            universe=universe, selected_index=selected_index)
            else:
                tab_signals, tab_strength, tab_intel, tab_raw = st.tabs(["Action Dashboard", "Signal Strength", "Intelligence", "System Data"])
                with tab_intel:
                    _render_intelligence_tab(universe, selected_index, timeframe)
                with tab_signals:
                    timeframe_label = "This Week's" if timeframe == 'Weekly' else "Today's"
                    _run_stats = st.session_state.get("screener_run_stats", {})
                    _n_analyzed = _run_stats.get("analyzed", len(results_df))
                    _n_universe = _run_stats.get("total_in_universe", _n_analyzed)
                    _date_str   = analysis_date.strftime("%d %b %Y") if hasattr(analysis_date, "strftime") else str(analysis_date)
                    ui.render_section_header(
                        f"{timeframe_label} Signals",
                        f"{_n_analyzed} / {_n_universe} symbols · {timeframe} · {_date_str} · "
                        "Delta Divergence (A) · Clamp Cross (B)",
                        icon="zap",
                        accent="amber"
                    )

                    # Meta Filter status banner (opt-in low-conviction suppression).
                    _if_mode, _if_thr = _intel_filter_active()
                    if _if_mode != "Off":
                        _if_verb = "hiding" if _if_mode == "Hide" else "dimming"
                        _if_src = "the reversion engine's Meta score (rank × edge-scaled conviction)"
                        st.markdown(
                            f'<div style="font-family:var(--data); font-size:0.66rem; color:var(--amber); '
                            f'background:rgba(212,168,83,0.08); border:1px solid rgba(212,168,83,0.22); '
                            f'border-radius:6px; padding:0.45rem 0.7rem; margin:0 0 0.7rem 0;">'
                            f'⚙ Meta Filter <b>{_if_mode}</b> — {_if_verb} signals with '
                            f'Meta score &lt; <b>{_if_thr:.2f}</b> · scored by {_if_src}. '
                            f'Today\'s fired signals use the Meta score; aged signals fall back to fire-bar Intel. '
                            f'Adjust in the sidebar ▸ Alpha-Health Monitor.</div>',
                            unsafe_allow_html=True,
                        )

                    # Cross-set veto inputs: any Set B fire (either side) in the window.
                    has_bullish_crossover = (results_df[['LB_Today', 'LB_1d', 'LB_2d', 'LB_3d', 'LB_5d']] != "—").any(axis=1)
                    has_bearish_crossover = (results_df[['SB_Today', 'SB_1d', 'SB_2d', 'SB_3d', 'SB_5d']] != "—").any(axis=1)

                    # Set A fires, vetoed by an opposite-side Set B fire in the window.
                    longs_a_df = results_df[(results_df['LA_5d'] != "—") & ~has_bearish_crossover].copy().sort_values('Priority_Long', ascending=False)
                    shorts_a_df = results_df[(results_df['SA_5d'] != "—") & ~has_bullish_crossover].copy().sort_values('Priority_Short', ascending=False)

                    # Set B fires
                    longs_b_df = results_df[results_df['LB_5d'] != "—"].copy().sort_values('Priority_Long', ascending=False)
                    shorts_b_df = results_df[results_df['SB_5d'] != "—"].copy().sort_values('Priority_Short', ascending=False)

                    if timeframe == 'Weekly':
                        _age_order = ["This Week", "1 Week Ago", "2 Weeks Ago", "3 Weeks Ago", "Within 5 Weeks"]
                    else:
                        _age_order = ["Today", "1 Day Ago", "2 Days Ago", "3 Days Ago", "Within 5 Days"]

                    has_signals = any(not df_.empty for df_ in [longs_a_df, shorts_a_df, longs_b_df, shorts_b_df])

                    if has_signals:
                        total_longs  = len(longs_a_df) + len(longs_b_df)
                        total_shorts = len(shorts_a_df) + len(shorts_b_df)
                        all_longs  = pd.concat([longs_a_df, longs_b_df]).drop_duplicates('Symbol').sort_values('Priority_Long', ascending=False)
                        all_shorts = pd.concat([shorts_a_df, shorts_b_df]).drop_duplicates('Symbol').sort_values('Priority_Short', ascending=False)

                        mc1, mc2, mc3, mc4 = st.columns(4)
                        with mc1: ui.render_metric_card("Long Signals", str(total_longs), f"A:{len(longs_a_df)} B:{len(longs_b_df)}", "success")
                        with mc2: ui.render_metric_card("Short Signals", str(total_shorts), f"A:{len(shorts_a_df)} B:{len(shorts_b_df)}", "danger")
                        with mc3:
                            strongest_long = all_longs.iloc[0] if not all_longs.empty else None
                            ui.render_metric_card("Strongest Long", strongest_long['SimpleName'] if strongest_long is not None else "—", f"Signal: {strongest_long['Signal']:.1f}" if strongest_long is not None else "No signals", "info")
                        with mc4:
                            strongest_short = all_shorts.iloc[0] if not all_shorts.empty else None
                            ui.render_metric_card("Strongest Short", strongest_short['SimpleName'] if strongest_short is not None else "—", f"Signal: {strongest_short['Signal']:.1f}" if strongest_short is not None else "No signals", "info")

                        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                        bull_tab, bear_tab = st.tabs(["Bullish Signals by Timing", "Bearish Signals by Timing"])
                        with bull_tab:
                            mom_bull_tab, cross_bull_tab, prio_bull_tab = st.tabs(["Set A", "Set B", "Priority Rank"])
                            with mom_bull_tab:
                                _, la_stats, _, _ = _bucket_signals_by_age(longs_a_df, side='long', condition_set='A', timeframe=timeframe)
                                la_html = _build_signal_table_html(la_stats, side='long', timeframe=timeframe)
                                _g = sum(1 for a in _age_order if la_stats[a]['count'] > 0)
                                _r = sum(la_stats[a]['count'] for a in _age_order)
                                st.components.v1.html(la_html, height=max(120 + _g * 60 + _r * 56, 150))
                            with cross_bull_tab:
                                _, lb_stats, _, _ = _bucket_signals_by_age(longs_b_df, side='long', condition_set='B', timeframe=timeframe)
                                lb_html = _build_signal_table_html(lb_stats, side='long', timeframe=timeframe)
                                _g = sum(1 for a in _age_order if lb_stats[a]['count'] > 0)
                                _r = sum(lb_stats[a]['count'] for a in _age_order)
                                st.components.v1.html(lb_html, height=max(70 + _g * 46 + _r * 44, 110))
                            with prio_bull_tab:
                                # Entire universe ranked by the LONG reversion priority —
                                # not gated by any signal set; this is the engine ranking.
                                _all_long = results_df.sort_values('Priority_Long', ascending=False)
                                st.markdown(
                                    '<div style="font-family:var(--data); font-size:0.66rem; color:var(--ink-tertiary); '
                                    'padding:0.2rem 0 0.6rem 0;">Full universe ranked by the long reversion priority '
                                    '(cross-sectional score, alpha-health-scaled) — independent of signal sets A–B.</div>',
                                    unsafe_allow_html=True,
                                )
                                st.components.v1.html(_build_signal_strength_table_html(_all_long, side='long'),
                                                      height=min(150 + len(_all_long) * 55, 900), scrolling=True)
                        with bear_tab:
                            mom_bear_tab, cross_bear_tab, prio_bear_tab = st.tabs(["Set A", "Set B", "Priority Rank"])
                            with mom_bear_tab:
                                _, sa_stats, _, _ = _bucket_signals_by_age(shorts_a_df, side='short', condition_set='A', timeframe=timeframe)
                                sa_html = _build_signal_table_html(sa_stats, side='short', timeframe=timeframe)
                                _g = sum(1 for a in _age_order if sa_stats[a]['count'] > 0)
                                _r = sum(sa_stats[a]['count'] for a in _age_order)
                                st.components.v1.html(sa_html, height=max(70 + _g * 46 + _r * 44, 110))
                            with cross_bear_tab:
                                _, sb_stats, _, _ = _bucket_signals_by_age(shorts_b_df, side='short', condition_set='B', timeframe=timeframe)
                                sb_html = _build_signal_table_html(sb_stats, side='short', timeframe=timeframe)
                                _g = sum(1 for a in _age_order if sb_stats[a]['count'] > 0)
                                _r = sum(sb_stats[a]['count'] for a in _age_order)
                                st.components.v1.html(sb_html, height=max(70 + _g * 46 + _r * 44, 110))
                            with prio_bear_tab:
                                # Entire universe ranked by the SHORT reversion priority.
                                _all_short = results_df.sort_values('Priority_Short', ascending=False)
                                st.markdown(
                                    '<div style="font-family:var(--data); font-size:0.66rem; color:var(--ink-tertiary); '
                                    'padding:0.2rem 0 0.6rem 0;">Full universe ranked by the short reversion priority '
                                    '(cross-sectional score, alpha-health-scaled) — independent of signal sets A–B.</div>',
                                    unsafe_allow_html=True,
                                )
                                st.components.v1.html(_build_signal_strength_table_html(_all_short, side='short'),
                                                      height=min(150 + len(_all_short) * 55, 900), scrolling=True)
                    else:
                        st.info(
                            f"**No signals found** for {selected_index} on {analysis_date} ({timeframe}). "
                            "All symbols were analyzed but none fired a Set A / Set B condition in the last 5 bars. "
                            "Try an adjacent trading date or switch to a broader universe."
                        )


                # Omni-channel base: include any stock with a signal in ANY set (A or B)
                long_sets = ['LA_5d', 'LB_5d']
                short_sets = ['SA_5d', 'SB_5d']
                _longs_base = results_df[results_df[long_sets].ne("—").any(axis=1)].copy()
                _shorts_base = results_df[results_df[short_sets].ne("—").any(axis=1)].copy()
                top_longs = _longs_base.sort_values('Priority_Long', ascending=False).head(10)
                top_shorts = _shorts_base.sort_values('Priority_Short', ascending=False).head(10)

                # Action Dashboard's own Signal Strength + System Data tabs.
                # Pulse Narrative has its own equivalents inside the `if` branch above
                # (different filtering — full-universe top-N rather than signal-set filter),
                # so these blocks must NOT escape the `else:` indentation level — that would
                # cause Pulse Narrative to register the same widget keys twice.

                # ════ Action Dashboard · TAB 2: SIGNAL STRENGTH ═══════════════════════
                with tab_strength:
                    ui.render_section_header(
                        "Order-Flow Signal Strength",
                        "Top signals ranked by reversion priority — Set A · Set B overlays",
                        icon="zap",
                        accent="amber"
                    )

                    # Strength metrics
                    avg_delta = results_df['Delta_Z'].abs().mean()
                    avg_abs = results_df['Abs_Strength'].mean()
                    strong_flow_count = len(results_df[results_df['Delta_Z'].abs() > 1.5])
                    absorb_count = len(results_df[results_df['Abs_Strength'] > 1.8])

                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                    with col_s1: ui.render_metric_card("Avg Δ-Z", f"{avg_delta:.2f}", "Signed delta z-score", "neutral")
                    with col_s2: ui.render_metric_card("Avg Absorption", f"{avg_abs:.2f}×", "|Δ| vs 20-bar avg", "neutral")
                    with col_s3: ui.render_metric_card("Strong Flow", str(strong_flow_count), f"{strong_flow_count/len(results_df)*100:.0f}% of universe", "info")
                    with col_s4: ui.render_metric_card("Absorption Hits", str(absorb_count), f"{absorb_count/len(results_df)*100:.0f}% of universe", "info")


                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

                    # ── column label renderer ──
                    def _col_label(side_label, side):
                        arrow = SVGS['LONG'].replace('currentColor', 'var(--emerald)') if side == 'long' else SVGS['SHORT'].replace('currentColor', 'var(--rose)')
                        color = 'var(--emerald)' if side == 'long' else 'var(--rose)'
                        return f"""
                        <p style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem; font-weight:600;
                                   text-transform:uppercase; letter-spacing:0.1em; color:{color};
                                   margin:0 0 0.4rem 0; display:flex; align-items:center; gap:0.35rem;">
                            {arrow} {side_label}
                        </p>"""

                    st.markdown(f"""
                    <div style="display:flex; align-items:baseline; gap:0.65rem; margin:1.75rem 0 0.9rem 0;
                                 padding-bottom:0.6rem; border-bottom:1px solid rgba(212,168,83,0.2);">
                        <span style="font-family:var(--display); font-size:0.62rem; font-weight:700;
                                     letter-spacing:0.12em; text-transform:uppercase; color:#D4A853;
                                     padding:0.18rem 0.5rem; background:rgba(212,168,83,0.1);
                                     border:1px solid rgba(212,168,83,0.3); border-radius:4px;">REVERSION ENGINE</span>
                        <span style="font-family:var(--display); font-size:1rem; font-weight:700;
                                     color:#F1F5F9; letter-spacing:0.04em;">Top 10 Rankings</span>
                    </div>
                    """, unsafe_allow_html=True)

                    _col_l, _col_s = st.columns(2)
                    with _col_l:
                        st.markdown(_col_label("Top 10 Bullish", "long"), unsafe_allow_html=True)
                        st.components.v1.html(_build_signal_strength_table_html(top_longs, side='long'), height=150 + len(top_longs)*55)
                    with _col_s:
                        st.markdown(_col_label("Top 10 Bearish", "short"), unsafe_allow_html=True)
                        st.components.v1.html(_build_signal_strength_table_html(top_shorts, side='short'), height=150 + len(top_shorts)*55)

                # ════ Action Dashboard · TAB 4: SYSTEM DATA ═══════════════════════════
                with tab_raw:
                    _render_system_data_tab(results_df, analysis_date,
                                            universe=universe, selected_index=selected_index)

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

def _render_model_passport_sidebar(current_universe: str, current_index, current_timeframe=None, analysis_mode=None):
    """Sidebar Engine Status panel — visible in every mode.

    The reversion engine is fixed and validated; there are no per-universe profiles to
    manage. This panel surfaces the engine name, the live alpha-health / trailing-IC
    reading from the last screener run, and the universe/timeframe it was measured on,
    plus a single 'Re-measure edge' control.

    Caller must be inside a ``with st.sidebar:`` context. Returns the settings dict the
    alpha-health monitor consumes: {force, lookback_days, horizons}.
    """
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Engine Status</div>', unsafe_allow_html=True)

    res = st.session_state.get("opt_results") or {}
    trailing_ic = st.session_state.get("alpha_health_ic", res.get("trailing_ic"))
    health = st.session_state.get("alpha_health_mult", res.get("alpha_health"))
    measured = trailing_ic is not None and isinstance(trailing_ic, (int, float))

    cal_universe  = res.get("universe") or None
    cal_index     = res.get("selected_index") or None
    cal_timeframe = res.get("timeframe") or None
    cal_label     = cal_index or cal_universe or "—"
    cur_label     = (current_index or current_universe or "—")
    universe_mismatch  = bool(res) and cal_label != "—" and cur_label != "—" and cal_label != cur_label
    timeframe_mismatch = (bool(res) and cal_timeframe and current_timeframe
                          and cal_timeframe != current_timeframe)
    mismatch = universe_mismatch or timeframe_mismatch

    t_stat = st.session_state.get("alpha_health_t", res.get("t_stat"))
    if measured:
        # Significance-gated + hysteresis (shared _edge_state, same as the tab).
        _lbl, card_class, _ = _edge_state(trailing_ic, t_stat)
        # Compact sidebar label (strip the trailing "— …" descriptor).
        state_label = _lbl.split(" — ")[0].split(" (")[0]
        ic_str     = f"{trailing_ic:+.3f}" + (f" · t{t_stat:+.1f}" if t_stat is not None else "")
        health_str = f"{float(health):.2f}×" if isinstance(health, (int, float)) else "—"
    else:
        state_label = "Cold Start"
        card_class  = "neutral"
        ic_str      = "—"
        health_str  = f"{float(health):.2f}×" if isinstance(health, (int, float)) else "—"
    if mismatch:
        card_class = "warning"
    updated     = res.get("timestamp", "—") or "—"
    ic_color    = "var(--emerald)" if (measured and trailing_ic > 0) else "var(--rose)" if measured else "var(--ink-secondary)"
    cal_tf_disp = cal_timeframe or "—"

    def _trim(s, n=22):
        s = str(s)
        return s if len(s) <= n else s[: n - 1] + "…"

    cal_label_disp = _trim(cal_label)

    st.markdown(f"""
    <div class="metric-card {card_class}" style="
            min-height:auto;
            padding:0.85rem 0.95rem;
            margin-bottom:0.7rem;
            animation:none;">
        <h4 style="margin:0 0 0.3rem 0;">Reversion Engine</h4>
        <h2 style="font-size:1.05rem; margin:0 0 0.7rem 0; letter-spacing:-0.01em;">{state_label}</h2>
        <div style="display:flex; flex-direction:column; gap:0.32rem;
                    padding-top:0.55rem;
                    border-top:1px solid rgba(255,255,255,0.06);">
            <div style="display:flex; justify-content:space-between; align-items:baseline; font-family:var(--data); font-size:0.62rem;">
                <span style="color:var(--ink-tertiary); text-transform:uppercase; letter-spacing:0.1em; font-size:0.58rem;">Engine</span>
                <span style="color:var(--ink-secondary); font-weight:500;">X-Sect Reversion v1</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:baseline; font-family:var(--data); font-size:0.62rem;">
                <span style="color:var(--ink-tertiary); text-transform:uppercase; letter-spacing:0.1em; font-size:0.58rem;">Measured on</span>
                <span style="color:var(--ink-secondary); font-weight:500; max-width:62%; text-align:right; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{cal_label_disp}</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:baseline; font-family:var(--data); font-size:0.62rem;">
                <span style="color:var(--ink-tertiary); text-transform:uppercase; letter-spacing:0.1em; font-size:0.58rem;">Depth</span>
                <span style="color:var(--ink-secondary); font-weight:500;">{cal_tf_disp}</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:baseline; font-family:var(--data); font-size:0.65rem;">
                <span style="color:var(--ink-tertiary); text-transform:uppercase; letter-spacing:0.1em; font-size:0.58rem;">Trailing IC</span>
                <span style="color:{ic_color}; font-weight:600;">{ic_str}</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:baseline; font-family:var(--data); font-size:0.65rem;">
                <span style="color:var(--ink-tertiary); text-transform:uppercase; letter-spacing:0.1em; font-size:0.58rem;">Alpha-Health</span>
                <span style="color:var(--ink-secondary); font-weight:600;">{health_str}</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:baseline; font-family:var(--data); font-size:0.6rem;">
                <span style="color:var(--ink-tertiary); text-transform:uppercase; letter-spacing:0.1em; font-size:0.58rem;">Updated</span>
                <span style="color:var(--ink-secondary);">{updated}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if mismatch:
        mismatch_lines = []
        if universe_mismatch:
            mismatch_lines.append(
                f"Edge measured on <b>{_trim(cal_label, 28)}</b><br>"
                f"Active universe is <b>{_trim(cur_label, 28)}</b>"
            )
        if timeframe_mismatch:
            mismatch_lines.append(
                f"Measured depth is <b>{cal_timeframe}</b><br>"
                f"Active depth is <b>{current_timeframe}</b>"
            )
        mismatch_body = "<br>".join(mismatch_lines)
        st.markdown(f"""
        <div style="font-family:var(--data); font-size:0.62rem; color:var(--amber);
                    background:rgba(212,168,83,0.08);
                    border:1px solid rgba(212,168,83,0.22);
                    border-radius:6px; padding:0.55rem 0.65rem;
                    margin-bottom:0.7rem; line-height:1.45;">
            <span style="font-weight:700;">Reading is from a different universe / depth.</span><br>
            {mismatch_body}<br>
            <span style="color:var(--ink-tertiary);">The alpha-health re-measures on the next run for
            the current selection — or use Re-measure edge below.</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Alpha-Health controls — directly below the status card. The engine is fixed;
    # the only knob is whether to force a fresh edge measurement this run.
    calib_force = False
    if analysis_mode in ("Single Date", "Pulse Narrative"):
        with st.expander("⚙ Alpha-Health Monitor", expanded=False):
            st.markdown(
                '<div style="font-family:var(--data); font-size:0.62rem; color:var(--ink-tertiary); '
                'line-height:1.55; padding:0 0 0.55rem 0;">The ranker measures its own realized edge '
                '(trailing cross-sectional IC) once per day per universe and scales conviction by it. '
                'Reuses today\'s reading otherwise.</div>',
                unsafe_allow_html=True,
            )
            calib_force = st.checkbox(
                "Re-measure edge this run", value=False, key="sb_calib_force",
                help="Re-harvest the lookback panel and re-measure the trailing IC even if today's reading exists.",
            )

            # ── Meta Filter (opt-in low-conviction suppression) ──
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown(
                '<div style="font-family:var(--data); font-size:0.62rem; color:var(--ink-tertiary); '
                'line-height:1.55; padding:0 0 0.4rem 0;">Filter fired signals by <b>Meta score</b> — the '
                'reversion engine\'s rank × edge-scaled conviction. Dim greys low-conviction signals; '
                'Hide removes them from the Action Dashboard. Off shows all signals.</div>',
                unsafe_allow_html=True,
            )
            st.session_state.setdefault("intel_filter_threshold", 0.45)
            st.session_state.setdefault("intel_filter_mode", "Dim")

            intel_filter_mode = st.radio(
                "Meta Filter", ["Off", "Dim", "Hide"],
                horizontal=True, key="intel_filter_mode",
                help="Off: show all. Dim: grey signals below the threshold. Hide: drop them entirely.",
            )
            intel_filter_threshold = st.slider(
                "Min Meta Score", min_value=0.0, max_value=1.0,
                step=0.05, key="intel_filter_threshold",
                disabled=(intel_filter_mode == "Off"),
                help="Fired signals with Meta score below this are dimmed or hidden.",
            )

    # Inline-harvest lookback ending at the analysis date: ~3y weekly, ~2y daily.
    calib_lookback_days = 1095 if current_timeframe == "Weekly" else 730
    _calib_settings = {
        "force":         calib_force,
        "lookback_days": calib_lookback_days,
        "horizons":      eng.HOLD_HORIZONS,
    }

    return _calib_settings


if __name__ == "__main__":
    main()
