"""
Sanket - Market Signal Screener | A Pragyam Product Family Member
WRCI Engine Quantitative Signal Screener Terminal
"""

import html
import re
import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import numpy as np
import plotly.graph_objects as go
import requests
import json
import io
import urllib3
import priority_engine as pe
import intelligence as intel
from io import BytesIO
from priority_engine import compute_priority
import warnings
import logging
import time
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
    layout="wide",
    initial_sidebar_state="expanded",
)

VERSION = "v3.2.0"

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


def _registry_ttl_seconds() -> int:
    """15 min during NSE market hours (Mon–Fri 09:15–15:30 IST), 90 min outside."""
    now = datetime.datetime.now(_IST)
    mo  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    mc  = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if now.weekday() < 5 and mo <= now <= mc:
        return 15 * 60
    return 90 * 60


def _registry_get(stock_list: list, end_date: datetime.date):
    """Return cached data_dict if still fresh for this universe+date, else None."""
    reg   = st.session_state.get(_REGISTRY_KEY, {})
    key   = frozenset(stock_list)
    entry = reg.get(key)
    if entry is None or entry["end_date"] != end_date:
        return None
    age = (datetime.datetime.now(_IST) - entry["fetched_at"]).total_seconds()
    if age > _registry_ttl_seconds():
        return None
    return entry["data"]


def _registry_put(stock_list: list, end_date: datetime.date, data_dict: dict):
    """Store data_dict in the session-state registry under frozenset(stock_list).

    DataFrames are stored as copies so downstream mutation (adding indicator
    columns) never corrupts the cached source data.
    """
    if _REGISTRY_KEY not in st.session_state:
        st.session_state[_REGISTRY_KEY] = {}
    st.session_state[_REGISTRY_KEY][frozenset(stock_list)] = {
        "data":       {k: v.copy() for k, v in data_dict.items()},
        "end_date":   end_date,
        "fetched_at": datetime.datetime.now(_IST),
    }


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
if "active_weights" not in st.session_state:
    st.session_state["active_weights"] = pe.DEFAULT_W.copy()
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
# Per-session weight helpers — keep calibrated weights in session_state so that
# concurrent users on shared Streamlit Cloud deployments cannot overwrite each
# other's active profile (the module-level `active_W` global in priority_engine
# is shared across all sessions in the same process).
# ──────────────────────────────────────────────────────────────────────────────
def _set_active_weights(w: dict):
    """Activate weights in both session_state (per-user) and pe module global (fallback)."""
    st.session_state["active_weights"] = w
    pe.set_active_weights(w)

def _get_active_weights() -> dict:
    """Return the active weights for this session, falling back to pe defaults."""
    return st.session_state.get("active_weights", pe.DEFAULT_W)

# ══════════════════════════════════════════════════════════════════════════════
# INITIALIZE UI
# ══════════════════════════════════════════════════════════════════════════════
inject_css()
ui.render_theme_toggle()

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

@st.cache_data(ttl=3600, show_spinner=False)
def get_fno_stock_list():
    """Fetch F&O eligible stocks from NSE with multiple fallback sources."""
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
                if symbols:
                    symbols_ns = [str(s) + ".NS" for s in symbols if s and str(s).strip()]
                    return symbols_ns, f"✓ Fetched {len(symbols_ns)} F&O securities"
    except Exception as e:
        console.detail(f"F&O source 1 (NSE JSON) failed: {type(e).__name__}: {e}")

    try:
        stock_data = nse_get_advances_declines()
        if isinstance(stock_data, pd.DataFrame):
            symbols = None
            if 'SYMBOL' in stock_data.columns:
                symbols = stock_data['SYMBOL'].tolist()
            elif 'symbol' in stock_data.columns:
                symbols = stock_data['symbol'].tolist()
            elif len(stock_data.index) > 0 and not isinstance(stock_data.index, pd.RangeIndex):
                symbols = stock_data.index.tolist()

            if symbols:
                symbols_ns = [str(s) + ".NS" for s in symbols if s and str(s).strip()]
                if symbols_ns:
                    return symbols_ns, f"✓ Fetched {len(symbols_ns)} F&O securities"
    except Exception as e:
        console.detail(f"F&O source 2 (advances/declines) failed: {type(e).__name__}: {e}")

    try:
        url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, verify=False, timeout=10)
        if response.status_code == 200:
            csv_file = io.StringIO(response.text)
            stock_df = pd.read_csv(csv_file)
            if 'Symbol' in stock_df.columns:
                symbols = stock_df['Symbol'].tolist()
                symbols_ns = [str(s) + ".NS" for s in symbols if s and str(s).strip()]
                return symbols_ns, f"✓ Fetched {len(symbols_ns)} stocks (NIFTY 500 fallback)"
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
    url = INDEX_URL_MAP.get(index)
    if url:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0',
            }
            session = requests.Session()
            session.get("https://archives.nseindia.com", headers=headers, verify=False, timeout=10)
            response = session.get(url, headers=headers, verify=False, timeout=15)
            response.raise_for_status()
            stock_df = pd.read_csv(io.StringIO(response.text))
            symbol_col = next((c for c in stock_df.columns if c.lower() == 'symbol'), None)
            if symbol_col:
                symbols = stock_df[symbol_col].tolist()
                symbols_ns = [str(s) + ".NS" for s in symbols if s and str(s).strip()]
                if symbols_ns:
                    return symbols_ns, f"✓ Fetched {len(symbols_ns)} constituents (NSE archive)"
        except Exception as e:
            console.detail(f"Index source 2 (NSE archive CSV) failed for '{index}': {type(e).__name__}: {e}")

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
            has_today = any(idx.date() == _ist_today for idx in sample_df.index)
            if not has_today:
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
                                    new_dates = live_ticker.index.difference(hist_df.index)
                                    if len(new_dates) > 0:
                                        data_dict[ticker] = pd.concat([hist_df, live_ticker.loc[new_dates]]).sort_index()
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
                "Pulse", 
                "Conviction", 
                "Regime_Confidence", 
                "Vol_Regime", 
                "Change_Point"
            ],
            "Metric Description": [
                "Master ranking score for bullish setups (normalized magnitude).",
                "Master ranking score for bearish setups (normalized magnitude).",
                "Abnormal Acceleration: 5D Velocity modulated by 20D Volatility Z-Score.",
                "Systemic conviction based on fractal trend and momentum confluence.",
                "Statistical probability (0.0 - 1.0) of the detected HMM regime.",
                "Volatility regime classification (Low/Normal/High) via GARCH analysis.",
                "Structural change point detection (CUSUM) identifying regime shifts."
            ]
        }
        pd.DataFrame(legend_data).to_excel(writer, index=False, sheet_name='Legend')
        
    return output.getvalue()

# ══════════════════════════════════════════════════════════════════════════════
# WRCI ENGINE: WAVE-REGIME COMPOSITE INDEX CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def calculate_wma(series, length):
    if length <= 1:
        return series
    weights = np.arange(1, length + 1)
    return series.rolling(window=length).apply(lambda vars: np.dot(vars, weights) / weights.sum(), raw=True)


def calculate_hma(series, length):
    if length <= 1:
        return series
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    wma_half = calculate_wma(series, half_length)
    wma_full = calculate_wma(series, length)
    diff = 2 * wma_half - wma_full
    return calculate_wma(diff, sqrt_length)


def calculate_ema(series, length):
    """
    Exponential Moving Average matched to TradingView's ta.ema.
    Initializes with SMA and follows the recursive formula.
    """
    if length <= 1:
        return series
    
    # Calculate initial SMA for startup
    sma = series.rolling(window=length, min_periods=length).mean()
    
    # Find the first valid SMA index
    first_idx = sma.first_valid_index()
    if first_idx is None:
        return pd.Series(np.nan, index=series.index)
        
    start_pos = series.index.get_loc(first_idx)
    alpha = 2.0 / (length + 1)
    
    # Recursive calculation
    values = series.values
    ema_values = np.empty(len(series))
    ema_values.fill(np.nan)
    ema_values[start_pos] = sma.loc[first_idx]
    
    for i in range(start_pos + 1, len(series)):
        if np.isnan(values[i]):
            # If current price is NaN, EMA is NaN but state is preserved
            ema_values[i] = np.nan
        else:
            # If previous EMA was NaN (due to NaN price), find the last valid EMA
            prev_ema = ema_values[i-1]
            if np.isnan(prev_ema):
                # Look back for last valid EMA to continue recursion
                # Standard TV behavior: if price was NaN, recursion skips it
                j = i - 1
                while j >= start_pos and np.isnan(ema_values[j]):
                    j -= 1
                prev_ema = ema_values[j] if j >= start_pos else values[i]
            
            ema_values[i] = (values[i] - prev_ema) * alpha + prev_ema
            
    return pd.Series(ema_values, index=series.index)


def calculate_linreg(series, length, offset=0):
    """Calculate the Linear Regression endpoint."""
    def _linreg_val(y):
        if np.isnan(y).any():
            return np.nan
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        return slope * (len(y) - 1 - offset) + intercept

    return series.rolling(window=length).apply(_linreg_val, raw=True)


def calculate_true_range(df):
    """Standard True Range calculation."""
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def compute_rsi(series, length=14):
    """RSI calculation using RMA (TradingView standard)."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    alpha = 1.0 / length
    roll_up = up.ewm(alpha=alpha, adjust=False).mean()
    roll_down = down.ewm(alpha=alpha, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def calculate_trend_count(series, length):
    trend = pd.Series(0.0, index=series.index)
    for i in range(1, length + 1):
        trend += np.where(series > series.shift(i), 1, -1)
    return trend


def run_full_analysis(df, reg_len=20, n1=10, n2=21, obLevel1=80, obLevel2=40, osLevel1=-80, osLevel2=-40):
    reg_len = max(reg_len, 2)
    # Auto-correct inverted OB levels (obLevel1 must be the stronger/higher bound)
    if obLevel1 < obLevel2:
        obLevel1, obLevel2 = obLevel2, obLevel1
    # Auto-correct inverted OS levels (osLevel1 must be the stronger/lower bound)
    if osLevel1 > osLevel2:
        osLevel1, osLevel2 = osLevel2, osLevel1

    hlc3 = (df['High'] + df['Low'] + df['Close']) / 3.0
    vol = df['Volume']
    
    # Institutional Volume Fallback: Historically used for VWMA-based indicators on indexes; 
    # maintained for volume-trend calculations even after transition to EMA core.
    if vol.sum() == 0:
        vol = pd.Series(1.0, index=df.index)
    
    hma_p = calculate_hma(hlc3, 15)
    hma_v = calculate_hma(vol, 15)

    trend = calculate_trend_count(hma_p, reg_len)
    voltrend_raw = calculate_trend_count(hma_v, reg_len)

    coeff = 10.0 / reg_len
    norm_trend = (trend * coeff) * 10.0
    voltrend = voltrend_raw * coeff

    ap = hlc3
    esa = calculate_ema(ap, n1)
    d = calculate_ema((ap - esa).abs(), n1)
    ci = (ap - esa) / np.maximum(0.015 * d, 1e-6)
    tci = calculate_ema(ci, n2)

    wt1 = tci
    wt2 = wt1.rolling(window=4).mean()

    # ── CONVICTION V3 ENGINE ────────────────────────────────────────────────
    # Component 1: Trend Strength (structural, slow, magnitude-aware)
    # Slope per bar = linreg endpoint(today) − linreg endpoint shifted back 1 bar on the same line.
    # Mirrors Pine's `ta.linreg(src, len, 0) - ta.linreg(src, len, 1)`.
    hma_close = calculate_hma(df['Close'], reg_len)
    slope     = calculate_linreg(hma_close, reg_len, offset=0) \
              - calculate_linreg(hma_close, reg_len, offset=1)
    avg_price = df['Close'].rolling(reg_len).mean().replace(0, np.nan)
    slope_pct = slope / avg_price
    
    tr = calculate_true_range(df)
    atr = tr.rolling(reg_len).mean()
    atr_pct = (atr / avg_price).replace(0, np.nan)
    
    trend_str = (slope_pct / atr_pct).clip(-3, 3) / 3.0  # bounded [-1, +1]

    # Component 2: Momentum Quality (tactical, medium)
    wt_sep = wt1 - wt2
    wt_sep_p = wt_sep / wt_sep.rolling(60).std(ddof=0).clip(lower=1e-6)
    wt_sep_n = np.tanh(wt_sep_p / 2.0)  # bounded [-1, +1]

    # Component 3: Participation (volume + RSI confluence)
    vol_z = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std(ddof=0).clip(lower=1e-6)
    vol_n = np.tanh(vol_z / 2.0)
    price_dir = np.sign(df['Close'] - df['Close'].shift(5))
    participation = price_dir * vol_n.abs()  # [-1, +1]
    
    rsi_14 = compute_rsi(df['Close'], 14)
    rsi_norm = (rsi_14 - 50) / 50.0  # [-1, +1]
    flow = 0.7 * participation + 0.3 * rsi_norm

    # Composite Conviction [−100, +100]
    conviction = (100 * (0.50 * trend_str + 0.30 * wt_sep_n + 0.20 * flow)).fillna(0)

    # ── PULSE V3 ENGINE ─────────────────────────────────────────────────────
    # Use 3-bar velocity and 30-bar baseline (no overlap)
    # Pine's nz(conviction[3], conviction) — substitute current value when shifted is NaN.
    conv_lag3 = conviction.shift(3).fillna(conviction)
    conv_vel_3 = conviction - conv_lag3
    baseline_30 = conv_lag3.rolling(27).mean()
    baseline_std = conv_lag3.rolling(27).std(ddof=0)

    NOISE_FLOOR = 1.5
    denom = np.maximum(baseline_std, NOISE_FLOOR)
    conv_z_3 = ((conviction - baseline_30) / denom).clip(-5, 5)

    vel_baseline = conv_vel_3.rolling(60).mean()
    vel_std = conv_vel_3.rolling(60).std(ddof=0)
    vel_z = ((conv_vel_3 - vel_baseline) / np.maximum(vel_std, NOISE_FLOOR)).clip(-5, 5)
    
    # Volume Factor [0.7, 1.0]
    vol_align = np.tanh(vol_z / 2.0) * np.sign(conv_vel_3)
    vol_factor = 0.85 + 0.15 * vol_align
    
    # Price-Action Factor [0.7, 1.0]
    close_lag3 = df['Close'].shift(3).fillna(df['Close'])
    ret_3 = (df['Close'] - close_lag3) / close_lag3.clip(lower=0.001)
    atr_14 = tr.rolling(14).mean()
    atr_pct_14 = atr_14 / df['Close'].clip(lower=0.001)
    ret_z = (ret_3 / atr_pct_14).clip(-5, 5)
    price_align = np.tanh(ret_z * np.sign(conv_vel_3) / 2.0)
    price_factor = 0.85 + 0.15 * price_align
    
    # Roll-over Correction (Amplify sign misalignment)
    sign_misalign = (np.sign(conviction) != np.sign(conv_vel_3)).astype(float)
    turn_amplifier = 1.0 + 0.30 * sign_misalign * np.tanh(np.abs(conv_vel_3) / 8.0)
    
    # Composite Pulse [−6, +6]
    pulse_core = np.sign(conv_vel_3) * np.sqrt((vel_z * conv_z_3).abs())
    pulse = (pulse_core * vol_factor * price_factor * turn_amplifier).clip(-6, 6).fillna(0)

    # ── F1 · PRICE MOMENTUM (orthogonal to oscillator) ─────────────────────────
    # 5-bar log return, ATR-normalized. Direct measure of "is it moving?"
    close_lag5 = df['Close'].shift(5).fillna(df['Close'])
    log_ret_5  = np.log(df['Close'] / close_lag5)
    atr_14_v4  = tr.rolling(14).mean()
    atr_pct_v4 = atr_14_v4 / df['Close']
    F1_PriceMom = (log_ret_5 / atr_pct_v4.clip(lower=1e-6)).clip(-5, 5).fillna(0)

    # ── F2 · VOLUME QUALITY (signed, smoothed) ─────────────────────────────────
    # Volume z-score signed by price direction. Positive = real flow with the move.
    vol_mean   = df['Volume'].rolling(20).mean()
    vol_std    = df['Volume'].rolling(20).std(ddof=0).clip(lower=1e-6)
    vol_z_raw  = (df['Volume'] - vol_mean) / vol_std
    price_dir_5  = np.sign(df['Close'] - close_lag5)
    F2_VolQual  = (vol_z_raw * price_dir_5).rolling(5).mean().clip(-5, 5).fillna(0)

    # ── UPDATE DATAFRAME ────────────────────────────────────────────────────
    df['F1_PriceMom']      = F1_PriceMom
    df['F2_VolQual']       = F2_VolQual
    df['Unified_Osc']      = wt1
    df['Signal_Line']      = wt2
    df['WT1']              = wt1
    df['Norm_Trend']       = norm_trend
    df['Conviction']       = conviction
    df['Pulse']            = pulse
    df['VolTrend']         = voltrend
    df['WT1_5ago']         = wt1.shift(5)
    df['Recent_Travel']    = wt1 - wt1.shift(5)
    df['Conviction_Delta'] = conviction.diff().fillna(0)
    df['Pulse_Delta']      = pulse.diff().fillna(0)
    df['ZScore']           = conv_z_3.fillna(0)

    # ── MA ALIGNMENT ────────────────────────────────────────────────────────
    # Calculate alignment of 5 major EMAs (8, 21, 50, 100, 200)
    ma_counts = pd.Series(0, index=df.index)
    for ma in [8, 21, 50, 100, 200]:
        ema = df['Close'].ewm(span=ma, adjust=False).mean()
        ma_counts += (df['Close'] > ema).astype(int)
    df['MA_Alignment'] = ma_counts

    # ── Step 2: Self-normalized thresholds for narrative matrix ───────────────
    # Rolling percentile boundaries replace hardcoded ±5
    pd_roll_std = df['Pulse_Delta'].rolling(30, min_periods=10).std(ddof=0).clip(lower=1e-6)
    cd_roll_std = df['Conviction_Delta'].rolling(30, min_periods=10).std(ddof=0).clip(lower=1e-6)

    # Normalize deltas to their local volatility (z-score of the delta)
    pd_norm = df['Pulse_Delta'] / pd_roll_std
    cd_norm = df['Conviction_Delta'] / cd_roll_std

    def categorize_narrative(p_norm, c_norm):
        """
        Direction-aware narrative matrix.
        Version: v2 (Normalized deltas)
        Uses normalized deltas (σ-units) instead of hardcoded ±5.
        Thresholds: ±1σ for ">>"/"<<", 0 for ">/"<"
        """
        p_state = ">>" if p_norm > 1.0 else ">" if p_norm > 0 else "<" if p_norm > -1.0 else "<<"
        c_state = ">>" if c_norm > 1.0 else ">" if c_norm > 0 else "<" if c_norm > -1.0 else "<<"

        matrix = {
            (">>", ">>"): ("SQUEEZE",  2.0, "#22c55e"),
            (">>", ">"):  ("HYPER",    1.6, "#34d399"),
            (">>", "<"):  ("EXHAUST",  0.6, "#fbbf24"),
            (">>", "<<"): ("CHAOS",    0.4, "#f87171"),
            (">",  ">>"): ("IGNITE",   1.4, "#4a9eff"),
            (">",  ">"):  ("ORGANIC",  1.2, "#38bdf8"),
            (">",  "<"):  ("EXPAND",   0.9, "#94a3b8"),
            (">",  "<<"): ("POP",      0.7, "#f87171"),
            ("<",  ">>"): ("HARDEN",   1.5, "#22c55e"),
            ("<",  ">"):  ("STEALTH",  1.3, "#818cf8"),
            ("<",  "<"):  ("RETRACE",  0.8, "#cbd5e1"),
            ("<",  "<<"): ("LIQUID",   0.3, "#ef4444"),
            ("<<", ">>"): ("LOAD",     1.7, "#4a9eff"),
            ("<<", ">"):  ("TRAP",     0.5, "#fbbf24"),
            ("<<", "<"):  ("CAPITUL",  0.2, "#ef4444"),
            ("<<", "<<"): ("CRASH",    0.1, "#b91c1c"),
        }
        return matrix.get((p_state, c_state), ("NEUTRAL", 1.0, "#94a3b8"))

    narrative_results = [
        categorize_narrative(p, c)
        for p, c in zip(pd_norm.fillna(0), cd_norm.fillna(0))
    ]
    df['Narrative']      = [n[0] for n in narrative_results]
    df['Priority_Mult']  = [n[1] for n in narrative_results]
    df['Narrative_Color']= [n[2] for n in narrative_results]

    # ── Step 3: Zone Depth Factor ─────────────────────────────────────────────
    # Rewards depth of oscillator position (composite_line is WRCI WT1)
    osc_val = wt1
    
    # Zone boundaries (consistent with WRCI script)
    obLevel1, obLevel2 = 80, 40

    # Bullish zone depth: how deep into oversold are we? (OS = negative WT1)
    bull_zone_depth = ((-osc_val - obLevel2) / (obLevel1 - obLevel2)).clip(0, 1).fillna(0)
    # Bearish zone depth: how deep into overbought are we?
    bear_zone_depth = ((osc_val - obLevel2) / (obLevel1 - obLevel2)).clip(0, 1).fillna(0)

    df['Bull_Zone_Depth'] = bull_zone_depth
    df['Bear_Zone_Depth'] = bear_zone_depth

    # Base Crossings
    sig_bull_cross = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    sig_bear_cross = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))

    # Set B: Crossover — Contrarian signals inside extreme zones
    crossover_long  = sig_bear_cross & (wt1 < osLevel2)
    crossover_short = sig_bull_cross & (wt1 > obLevel2)

    # Set A: Momentum — Trend signals (Mutually Exclusive with Set B)
    momentum_long   = sig_bull_cross & (~crossover_short)
    momentum_short  = sig_bear_cross & (~crossover_long)

    # Set C: Threshold — freshly entering OS/OB zone with signal-line validation
    threshold_long  = (wt1 < osLevel2) & (wt1.shift(1) >= osLevel2) & (wt2 > osLevel2)
    threshold_short = (wt1 > obLevel2) & (wt1.shift(1) <= obLevel2) & (wt2 < obLevel2)

    df['long_cond'] = momentum_long
    df['short_cond'] = momentum_short
    df['long_cond_comp'] = crossover_long
    df['short_cond_comp'] = crossover_short
    df['long_cond_wt'] = threshold_long
    df['short_cond_wt'] = threshold_short

    df['Condition'] = np.select(
        [wt1 > obLevel1, wt1 > obLevel2, wt1 < osLevel1, wt1 < osLevel2],
        ['OB Extreme', 'OB', 'OS Extreme', 'OS'],
        default='Neutral'
    )

    # Set D: Squeeze — Squeeze Momentum
    df = compute_squeeze_momentum(df, length=20)
    sqz_release = df['SQZ_Off'] & df['SQZ_On'].shift(1)
    df['long_cond_sqz'] = sqz_release & (df['SQZ_Val'] > df['SQZ_Val'].shift(1))
    df['short_cond_sqz'] = sqz_release & (df['SQZ_Val'] < df['SQZ_Val'].shift(1))

    return df

# ══════════════════════════════════════════════════════════════════════════════
# SQUEEZE MOMENTUM ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def compute_squeeze_momentum(df: pd.DataFrame, length: int = 20, mult: float = 1.5, 
                             length_kc: int = 20, mult_kc: float = 1.5, 
                             use_true_range: bool = True) -> pd.DataFrame:
    """
    Squeeze Momentum Indicator
    Ported exactly from Pine Script.
    """
    df = df.copy()
    
    # Calculate BB
    source = df['Close']
    basis = source.rolling(window=length).mean()
    dev = mult * source.rolling(window=length).std(ddof=0)
    upperBB = basis + dev
    lowerBB = basis - dev
    
    # Calculate KC
    ma = source.rolling(window=length_kc).mean()
    if use_true_range:
        prev_close = df['Close'].shift(1)
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - prev_close).abs()
        tr3 = (df['Low'] - prev_close).abs()
        range_val = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    else:
        range_val = df['High'] - df['Low']
        
    rangema = range_val.rolling(window=length_kc).mean()
    upperKC = ma + rangema * mult_kc
    lowerKC = ma - rangema * mult_kc
    
    # Squeeze States
    sqzOn = (lowerBB > lowerKC) & (upperBB < upperKC)
    sqzOff = (lowerBB < lowerKC) & (upperBB > upperKC)
    noSqz = (~sqzOn) & (~sqzOff)
    
    # Linear Regression of Delta
    highest_high = df['High'].rolling(window=length).max()
    lowest_low = df['Low'].rolling(window=length).min()
    
    avg_hl = (highest_high + lowest_low) / 2.0
    sma_close = df['Close'].rolling(window=length).mean()
    
    inner_avg = (avg_hl + sma_close) / 2.0
    delta = source - inner_avg
    
    # Real ta.linreg endpoint
    val = calculate_linreg(delta, length, 0)
    
    # Assign Colors
    prev_val = val.shift(1).fillna(0)
    bcolor = np.where(
        val > 0,
        np.where(val > prev_val, "lime", "green"),
        np.where(val < prev_val, "red", "maroon")
    )
    scolor = np.where(noSqz, "blue", np.where(sqzOn, "black", "gray"))
    
    # Add to DataFrame
    df['SQZ_Val'] = val
    df['SQZ_BColor'] = bcolor
    df['SQZ_SColor'] = scolor
    df['SQZ_On'] = sqzOn
    df['SQZ_Off'] = sqzOff
    df['SQZ_NoSqz'] = noSqz
    
    return df

# ══════════════════════════════════════════════════════════════════════════════
# REGIME INTELLIGENCE ENGINE (NIRNAY FEATURES)
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveHMM:
    """Hidden Markov Model for regime state discovery - classifies WRCI signals"""
    
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
        
        return {"BULL": updated[0], "NEUTRAL": updated[1], "BEAR": updated[2]}
    
    def reset(self):
        self.state_probabilities = np.array([0.33, 0.34, 0.33])
        self.observation_history = []
        self.state_history = []


class GARCHDetector:
    """GARCH-inspired volatility regime detection for WRCI signal variance"""
    
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
        self.current_variance = np.clip(new_var, 0.001, 1.0)
        
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
    
    def reset(self):
        self.current_variance = 0.04
        self.shock_history = []


class CUSUMDetector:
    """CUSUM change point detection for WRCI signal regime shifts"""
    
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
    
    def reset(self):
        self.positive_cusum = 0.0
        self.negative_cusum = 0.0
        self.value_history = []


class AdaptiveKalmanFilter:
    """Kalman filter for WRCI signal smoothing"""
    
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
    
    def reset(self, initial=0.0):
        self.estimate = initial
        self.error_covariance = 1.0
        self.innovation_history = []


def run_regime_analysis(df):
    """
    Apply joint-state regime intelligence over (F1_PriceMom, F2_VolQual, Conviction/20).
    Replaces the previous WT1-only HMM. The three input dimensions are roughly
    orthogonal, so HMM's classification now reflects true market state, not a
    re-statement of WaveTrend.
    """
    hmm    = AdaptiveHMM()
    garch  = GARCHDetector()
    cusum  = CUSUMDetector()
    kalman = AdaptiveKalmanFilter()

    regimes, hmm_bulls, hmm_bears, vol_regimes = [], [], [], []
    change_points, confidences, signal_history = [], [], []

    f1_vals = df['F1_PriceMom'].values
    f2_vals = df['F2_VolQual'].values
    cv_vals = (df['Conviction'] / 20.0).values  # rescale to ~[-5, +5]

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
    signal_history.clear()   # reset history list; detector internal state is kept

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

    Priority: B (composite) > A (momentum) > C (threshold) > D (squeeze) > Zone.
    Matches the vectorised np.select used in the timeseries harvest path.
    """
    if row.get('long_cond_comp'):  return "B: Long"
    if row.get('short_cond_comp'): return "B: Short"
    if row.get('long_cond'):       return "A: Long"
    if row.get('short_cond'):      return "A: Short"
    if row.get('long_cond_wt'):    return "C: Long"
    if row.get('short_cond_wt'):   return "C: Short"
    if row.get('long_cond_sqz'):   return "D: Long"
    if row.get('short_cond_sqz'):  return "D: Short"
    cond = row.get('Condition', 'Neutral')
    return cond if cond != 'Neutral' else '-'


def calculate_divergences(df, lookback: int = 20, timeframe: str = "Daily"):
    """
    Peak-trough divergence over `lookback` bars. Replaces 1-bar comparison
    which produced ~30% false positives.

    Bullish divergence: latest local price-low is LOWER than previous,
                        but latest WT1 local-low is HIGHER (in OS context).
    Bearish divergence: latest local price-high is HIGHER than previous,
                        but latest WT1 local-high is LOWER (in OB context).

    `order` is scaled to timeframe so that the neighborhood covers a similar
    real-time span on both daily and weekly charts: order=3 ≈ 1 week on daily,
    order=2 ≈ 4 weeks on weekly.
    """
    from scipy.signal import argrelextrema
    close = df['Close'].values
    osc   = df['Unified_Osc'].values
    n     = len(df)
    bull  = np.zeros(n, dtype=bool)
    bear  = np.zeros(n, dtype=bool)
    order = 2 if timeframe == "Weekly" else 3

    for i in range(lookback, n):
        wc = close[i - lookback : i + 1]
        wo = osc[i - lookback : i + 1]
        c_lows  = argrelextrema(wc, np.less,    order=order)[0]
        c_highs = argrelextrema(wc, np.greater, order=order)[0]
        o_lows  = argrelextrema(wo, np.less,    order=order)[0]
        o_highs = argrelextrema(wo, np.greater, order=order)[0]

        # Adaptive OS/OB thresholds: use the 30th/70th percentile of the window
        # so the trigger scales with the actual oscillator range for this asset,
        # rather than assuming a fixed [-100, +100] normalized range.
        bull_thresh = min(-30.0, float(np.percentile(wo, 30)))
        bear_thresh = max( 30.0, float(np.percentile(wo, 70)))

        if len(c_lows) >= 2 and len(o_lows) >= 2:
            if (wc[c_lows[-1]] < wc[c_lows[-2]]
                and wo[o_lows[-1]] > wo[o_lows[-2]]
                and wo[o_lows[-1]] < bull_thresh):
                bull[i] = True
        if len(c_highs) >= 2 and len(o_highs) >= 2:
            if (wc[c_highs[-1]] > wc[c_highs[-2]]
                and wo[o_highs[-1]] < wo[o_highs[-2]]
                and wo[o_highs[-1]] > bear_thresh):
                bear[i] = True

    df['Bullish_Div'] = bull
    df['Bearish_Div'] = bear
    return df

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
                PULSE ENGINE
            </h3>
            <p>Sanket Pulse Engine identifies Abnormal Acceleration (Velocity * Z-Score) to surface high-conviction ignition events.</p>
            <div class='spec'>
                <span>Primary:</span> Abnormal Acceleration (Pulse)<br>
                <span>Secondary:</span> Signal Conviction Score<br>
                <span>Metric:</span> 5D Velocity * 20D Vol Z-Score<br>
                <span>Sorting:</span> Rank by Pulse Strength
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='system-card regime'>
            <h3>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>
                SIGNAL STRUCTURE
            </h3>
            <p>Hierarchical signal generation (Sets A-D) contextualized by Pulse and structural trend regime alignment.</p>
            <div class='spec'>
                <span>Sets:</span> Momentum / Crossover / Threshold / Squeeze<br>
                <span>Ranking:</span> Sorted by Abnormal Acceleration<br>
                <span>Long/Short:</span> Dual-sided directional logic<br>
                <span>Timing:</span> Age-weighted signal aging
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
            <p>Span eight asset classes through five analysis modes — from single-date snapshots to self-tuning calibration profiles.</p>
            <div class='spec'>
                <span>Coverage:</span> 8 asset classes · 500+ symbols<br>
                <span>Timeframes:</span> Daily · Weekly<br>
                <span>Modes:</span> 5 modes · snapshot to self-tuning<br>
                <span>Calibration:</span> Per-universe priority profiles
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
           <span style="color:var(--ink-secondary); font-size:0.85em; margin-top:0.5rem; display:inline-block;">System will compute Wave Trend oscillations · Analyze Abnormal Acceleration · Rank by calibrated Priority</span></p>
    </div>
    """, unsafe_allow_html=True)


def get_signal_strength_score(row):
    """Calculate signal strength from magnitude with diminishing returns above 50.

    Returns: Strength score (0-100) where magnitude 0-50 = linear, >50 = diminishing returns.
    """
    base_score = abs(row.get('Signal', 0))
    if base_score > 50:
        base_score = 50 + (base_score - 50) * 0.5
    return min(100, base_score)


# ══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS & SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
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
            ["Single Date", "Historical Range", "Correlation Analysis", "Pulse Narrative", "Intelligence (Self-Tuning)"],
            key="sb_mode",
            label_visibility="collapsed",
        )

        if analysis_mode in ["Single Date", "Pulse Narrative"]:
            st.markdown('<div class="sidebar-title">Analysis Date</div>', unsafe_allow_html=True)
            analysis_date = st.date_input("Date", _today_ist(), max_value=_today_ist(), key="sb_analysis_date", label_visibility="collapsed")
            start_date_hist, end_date_hist = None, None
            corr_target_ticker, corr_lookback, corr_method = None, 90, "Pearson"
        elif analysis_mode in ["Historical Range", "Intelligence (Self-Tuning)"]:
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

        # WRCI Engine — hardcoded defaults
        reg_len, wt_n1, wt_n2 = 20, 10, 21
        obLevel1, obLevel2, osLevel1, osLevel2 = 80, 40, -80, -40

        # ── Intelligence-mode-only inputs ────────────────────────────────
        if analysis_mode == "Intelligence (Self-Tuning)":
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-title">Calibration Settings</div>', unsafe_allow_html=True)

            # Default 75 trials — asymmetric search space is 21-dimensional, needs
            # a bit more exploration than the symmetric (13-dim) era's 50-default.
            calib_trials = st.slider(
                "Search Trials", min_value=20, max_value=200, value=75, step=5,
                key="sb_calib_trials",
                help=(
                    "Number of weight configurations Optuna's TPE sampler will try. "
                    "Independent of date range and universe size. Asymmetric search "
                    "covers 21 dimensions (long + short factor weights) — 75 trials "
                    "is a reasonable starting point; bump to 150 for tighter convergence."
                ),
            )
            # Slider is in percentage units to match the body summary line ("Split 70 / 30").
            calib_train_pct = st.slider(
                "Train / Val Split", min_value=50, max_value=90, value=70, step=5,
                key="sb_calib_train_pct",
                help=(
                    "Percent of dates used to fit weights. Remainder is held out "
                    "for out-of-sample (validation) IR."
                ),
                format="%d%%",
            )
            calib_train_frac = calib_train_pct / 100.0
        else:
            calib_trials, calib_train_frac = 75, 0.70

        calib_settings = {
            "trials":     calib_trials,
            "train_frac": calib_train_frac,
            "horizons":   pe.HOLD_HORIZONS,
        }

        # ── Date-range validation (Historical Range / Intelligence only) ──
        date_range_valid = True
        if analysis_mode in ("Historical Range", "Intelligence (Self-Tuning)"):
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
            "Intelligence (Self-Tuning)": "◈ RUN HARVEST",
        }
        run_clicked = st.button(
            _RUN_LABELS.get(analysis_mode, "◈ RUN ANALYSIS"),
            type="primary", width='stretch',
            disabled=not date_range_valid,
        )

        # ── Per-universe profile sync (must run BEFORE Passport renders) ──
        # When the (universe, selected_index, timeframe) triple changes — including
        # the click that triggered THIS rerun — load the matching profile from disk
        # so the Passport reflects the new universe/depth in the same render frame
        # instead of lagging by one interaction.
        _current_uni_key = (universe, selected_index, timeframe)
        _previous_uni_key = st.session_state.get("_last_universe_key")
        if _previous_uni_key != _current_uni_key:
            _profile = pe.load_profile_for(universe, selected_index, timeframe)
            _uni_label = (selected_index or universe or "—")
            if _profile and isinstance(_profile.get("weights"), dict):
                _set_active_weights(_profile["weights"])
                st.session_state["opt_results"] = _profile
                # Don't log the very first sync of a session (already covered by the
                # session-start banner); only log genuine universe transitions.
                if _previous_uni_key is not None:
                    _ir_v = _profile.get("val_score")
                    _ir_s = f"{_ir_v:+.3f}" if isinstance(_ir_v, (int, float)) else "—"
                    console.detail(
                        f"Profile loaded · {_uni_label} · val IR {_ir_s}"
                    )
            else:
                _set_active_weights(pe.DEFAULT_W)
                if "opt_results" in st.session_state:
                    del st.session_state["opt_results"]
                if _previous_uni_key is not None:
                    console.detail(
                        f"No profile for {_uni_label} · reverted to factory defaults"
                    )
            st.session_state["_last_universe_key"] = _current_uni_key

        # Model Passport — rendered in every mode. Surfaces the active priority
        # profile (default vs calibrated) and warns when the calibrated universe
        # differs from the current sidebar selection.
        _render_model_passport_sidebar(universe, selected_index, timeframe)

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

        return universe, selected_index, analysis_date, reg_len, wt_n1, wt_n2, (obLevel1, obLevel2, osLevel1, osLevel2), timeframe, analysis_mode, start_date_hist, end_date_hist, run_clicked, corr_target_ticker, corr_lookback, corr_method, calib_settings


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SCREENER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_screener_analysis(universe, selected_index, analysis_date, reg_len, wt_n1, wt_n2, levels, timeframe, show_progress=True, external_progress_slot=None, progress_offset=0, progress_scale=100):
    """Execute WRCI momentum analysis on universe symbols and return ranked signals.

    Fetches market data for universe, computes Wave Trend oscillations, calculates
    signal magnitude and trend values, detects overbought/oversold zones.

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
        progress_bar(progress_slot, pct_val, "Initializing WRCI engine", f"Universe: {universe}")
    
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
        progress_bar(progress_slot, pct_val, "Fetching Market Data", f"{len(stock_list)} stocks")
    # Fetch up to today (IST) — always hits registry first; only goes to yfinance on miss
    data_dict, fetch_msg = get_universe_data(stock_list, end_date=_today_ist())

    if not data_dict:
        console.error(fetch_msg)
        st.error(fetch_msg)
        return None

    console.success(f"Successfully downloaded data for {len(data_dict)} stocks")
    console.end_phase("DATA ACQUISITION")

    console.start_phase("WRCI MOMENTUM ANALYSIS", 2, 2)

    console.section("Analysis Parameters")
    console.item("Timeframe", timeframe)
    console.item("Regression Length", reg_len)
    console.item("Wave Trend", f"N1={wt_n1}  N2={wt_n2}")
    console.item("OB Levels", f"{obLevel1} / {obLevel2}")
    console.item("OS Levels", f"{osLevel1} / {osLevel2}")
    console.item("Instruments", f"{len(data_dict)} of {len(stock_list)} fetched successfully")
    if show_progress or external_progress_slot is not None:
        pct_val = progress_offset + (20 * progress_scale / 100)
        progress_bar(progress_slot, pct_val, "Analyzing WRCI momentum", f"{len(data_dict)} stocks")

    results = []
    _failed_symbols = []

    _tf_label = "weekly" if timeframe == "Weekly" else "daily"
    console.section(f"Signal Analysis — {len(data_dict)} {_tf_label} instruments")

    for i, (ticker, df) in enumerate(data_dict.items()):
        try:
            pct = int(progress_offset + (20 + (i + 1) / len(data_dict) * 75) * progress_scale / 100)
            if show_progress or external_progress_slot is not None:
                progress_bar(progress_slot, pct, f"Analyzing Signals", f"{i + 1}/{len(data_dict)} stocks")

            if timeframe == "Weekly":
                df = resample_to_weekly(df)

            if len(df) < reg_len + 30:
                console.detail(f"{ticker}: Skipped (Insufficient data: {len(df)} rows)")
                continue

            df = run_full_analysis(df, reg_len, wt_n1, wt_n2, obLevel1, obLevel2, osLevel1, osLevel2)
            df = run_regime_analysis(df)        # adds HMM_Bull/Bear, Vol_Regime, Change_Point, Regime_Confidence
            df = calculate_divergences(df, timeframe=timeframe)      # adds Bullish_Div, Bearish_Div

            # Sample at analysis_date or last available
            df.index = pd.to_datetime(df.index)
            target_dt = pd.to_datetime(analysis_date)

            if target_dt in df.index:
                idx_pos = df.index.get_loc(target_dt)
            else:
                idx_pos = len(df) - 1
                console.detail(f"{ticker}: analysis_date {analysis_date} not in data — using nearest available {df.index[idx_pos].date()}")

            if idx_pos < 5:
                continue

            # Get historical signals for tracking (Today, 1d, 2d, 3d, Within 5d)
            sample_range = df.iloc[max(0, idx_pos - 5) : idx_pos + 1]

            last_row = df.iloc[idx_pos]

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
                "Signal": round(last_row['Unified_Osc'], 2) if not pd.isna(last_row['Unified_Osc']) else 0.0,
                "Trend": round(last_row['Norm_Trend'], 2) if not pd.isna(last_row['Norm_Trend']) else 0.0,
                "Conviction": round(last_row['Conviction'], 2) if not pd.isna(last_row['Conviction']) else 0.0,
                "Conviction_Delta": round(last_row['Conviction_Delta'], 2) if not pd.isna(last_row['Conviction_Delta']) else 0.0,
                "Pulse": round(last_row['Pulse'], 2) if not pd.isna(last_row['Pulse']) else 0.0,
                "Pulse_Delta": round(last_row['Pulse_Delta'], 2) if not pd.isna(last_row['Pulse_Delta']) else 0.0,
                "Narrative": last_row['Narrative'],
                "Narrative_Color": last_row['Narrative_Color'],
                "Wave": round(last_row['WT1'], 2) if not pd.isna(last_row['WT1']) else 0.0,
                "Zone": last_row['Condition'],
                "SignalType": signal_type,
                "Price": round(last_row['Close'], 2),
                "PctChange": round(pct_change, 2),
                # v3 Metrics for Engine 2.0
                "WT1_5ago":      round(df.iloc[idx_pos-5]['WT1'], 2) if idx_pos >= 5 else 0.0,
                "VolTrend":      round(last_row.get('VolTrend', 0), 3),
                "HMM_Bull":      float(last_row.get('HMM_Bull', 0.33)),
                "HMM_Bear":      float(last_row.get('HMM_Bear', 0.33)),
                "Vol_Regime":    str(last_row.get('Vol_Regime', 'NORMAL')),
                "Change_Point":  bool(last_row.get('Change_Point', False)),
                "Regime_Confidence": float(last_row.get('Regime_Confidence', 0.0)),
                "Bullish_Div":   bool(last_row.get('Bullish_Div', False)),
                "Bearish_Div":   bool(last_row.get('Bearish_Div', False)),
                "F1_PriceMom":   float(last_row.get('F1_PriceMom', 0)),
                "F2_VolQual":    float(last_row.get('F2_VolQual', 0)),
                # Set C: Momentum — Historical Long Signals (kept for Range Study compat)
                "L_Today": "●" if sample_range.iloc[-1]['long_cond'] else "—",
                "L_1d": "●" if sample_range.iloc[-2]['long_cond'] else "—",
                "L_2d": "●" if sample_range.iloc[-3]['long_cond'] else "—",
                "L_3d": "●" if sample_range.iloc[-4]['long_cond'] else "—",
                "L_5d": "●" if sample_range.tail(5)['long_cond'].any() else "—",
                # Set C: Momentum — Historical Short Signals
                "S_Today": "●" if sample_range.iloc[-1]['short_cond'] else "—",
                "S_1d": "●" if sample_range.iloc[-2]['short_cond'] else "—",
                "S_2d": "●" if sample_range.iloc[-3]['short_cond'] else "—",
                "S_3d": "●" if sample_range.iloc[-4]['short_cond'] else "—",
                "S_5d": "●" if sample_range.tail(5)['short_cond'].any() else "—",
                # Set A: Momentum — Historical Long Signals
                "LA_Today": "●" if sample_range.iloc[-1]['long_cond'] else "—",
                "LA_1d": "●" if sample_range.iloc[-2]['long_cond'] else "—",
                "LA_2d": "●" if sample_range.iloc[-3]['long_cond'] else "—",
                "LA_3d": "●" if sample_range.iloc[-4]['long_cond'] else "—",
                "LA_5d": "●" if sample_range.tail(5)['long_cond'].any() else "—",
                # Set A: Momentum — Historical Short Signals
                "SA_Today": "●" if sample_range.iloc[-1]['short_cond'] else "—",
                "SA_1d": "●" if sample_range.iloc[-2]['short_cond'] else "—",
                "SA_2d": "●" if sample_range.iloc[-3]['short_cond'] else "—",
                "SA_3d": "●" if sample_range.iloc[-4]['short_cond'] else "—",
                "SA_5d": "●" if sample_range.tail(5)['short_cond'].any() else "—",
                # Set B: Crossover — Historical Long Signals
                "LB_Today": "●" if sample_range.iloc[-1]['long_cond_comp'] else "—",
                "LB_1d": "●" if sample_range.iloc[-2]['long_cond_comp'] else "—",
                "LB_2d": "●" if sample_range.iloc[-3]['long_cond_comp'] else "—",
                "LB_3d": "●" if sample_range.iloc[-4]['long_cond_comp'] else "—",
                "LB_5d": "●" if sample_range.tail(5)['long_cond_comp'].any() else "—",
                # Set B: Crossover — Historical Short Signals
                "SB_Today": "●" if sample_range.iloc[-1]['short_cond_comp'] else "—",
                "SB_1d": "●" if sample_range.iloc[-2]['short_cond_comp'] else "—",
                "SB_2d": "●" if sample_range.iloc[-3]['short_cond_comp'] else "—",
                "SB_3d": "●" if sample_range.iloc[-4]['short_cond_comp'] else "—",
                "SB_5d": "●" if sample_range.tail(5)['short_cond_comp'].any() else "—",
                # Set C: Threshold — Historical Long Signals
                "LC_Today": "●" if sample_range.iloc[-1]['long_cond_wt'] else "—",
                "LC_1d": "●" if sample_range.iloc[-2]['long_cond_wt'] else "—",
                "LC_2d": "●" if sample_range.iloc[-3]['long_cond_wt'] else "—",
                "LC_3d": "●" if sample_range.iloc[-4]['long_cond_wt'] else "—",
                "LC_5d": "●" if sample_range.tail(5)['long_cond_wt'].any() else "—",
                # Set C: Threshold — Historical Short Signals
                "SC_Today": "●" if sample_range.iloc[-1]['short_cond_wt'] else "—",
                "SC_1d": "●" if sample_range.iloc[-2]['short_cond_wt'] else "—",
                "SC_2d": "●" if sample_range.iloc[-3]['short_cond_wt'] else "—",
                "SC_3d": "●" if sample_range.iloc[-4]['short_cond_wt'] else "—",
                "SC_5d": "●" if sample_range.tail(5)['short_cond_wt'].any() else "—",
                # Set D: Squeeze — Historical Long Signals
                "LD_Today": "●" if sample_range.iloc[-1]['long_cond_sqz'] else "—",
                "LD_1d": "●" if sample_range.iloc[-2]['long_cond_sqz'] else "—",
                "LD_2d": "●" if sample_range.iloc[-3]['long_cond_sqz'] else "—",
                "LD_3d": "●" if sample_range.iloc[-4]['long_cond_sqz'] else "—",
                "LD_5d": "●" if sample_range.tail(5)['long_cond_sqz'].any() else "—",
                # Set D: Squeeze — Historical Short Signals
                "SD_Today": "●" if sample_range.iloc[-1]['short_cond_sqz'] else "—",
                "SD_1d": "●" if sample_range.iloc[-2]['short_cond_sqz'] else "—",
                "SD_2d": "●" if sample_range.iloc[-3]['short_cond_sqz'] else "—",
                "SD_3d": "●" if sample_range.iloc[-4]['short_cond_sqz'] else "—",
                "SD_5d": "●" if sample_range.tail(5)['short_cond_sqz'].any() else "—",
                # Additional fields for detail cards
                "Osc_Value": round(last_row.get('Unified_Osc', 0), 2),
                "MA_Alignment": int(last_row.get('MA_Alignment', 0)),
                "ZScore_Value": round(last_row.get('ZScore', 0), 2),
            })
            
            console.detail(f"[{i+1}/{len(data_dict)}] {ticker}: Signal={last_row['Unified_Osc']:+.2f}  Zone={last_row['Condition']}  Status={signal_type}")
            
        except Exception as e:
            console.failure(f"Analysis Failed: {ticker}", str(e))
            _failed_symbols.append(ticker)
            continue

    console.end_phase("WRCI MOMENTUM ANALYSIS")

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
    console.line('═', 70)
    
    if show_progress or external_progress_slot is not None:
        pct_val = progress_offset + (95 * progress_scale / 100) if external_progress_slot else 100
        progress_bar(progress_slot, pct_val, "Analysis Complete", f"{len(results)} stocks analyzed")
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
                "but none produced a WRCI signal in the current timeframe. "
                "Try an adjacent trading date, or check that the selected date is a market session."
            )
        # Return empty DataFrame with expected columns to prevent downstream KeyErrors
        expected_cols = [
            "Symbol", "DisplayName", "SimpleName", "Signal", "Trend", "Wave", "Zone", "SignalType", "Price", "PctChange",
            "L_Today", "L_1d", "L_2d", "L_3d", "L_5d", "S_Today", "S_1d", "S_2d", "S_3d", "S_5d",
            "LA_Today", "LA_1d", "LA_2d", "LA_3d", "LA_5d", "SA_Today", "SA_1d", "SA_2d", "SA_3d", "SA_5d",
            "LB_Today", "LB_1d", "LB_2d", "LB_3d", "LB_5d", "SB_Today", "SB_1d", "SB_2d", "SB_3d", "SB_5d",
            "LC_Today", "LC_1d", "LC_2d", "LC_3d", "LC_5d", "SC_Today", "SC_1d", "SC_2d", "SC_3d", "SC_5d",
            "LD_Today", "LD_1d", "LD_2d", "LD_3d", "LD_5d", "SD_Today", "SD_1d", "SD_2d", "SD_3d", "SD_5d",
            "Osc_Value", "MA_Alignment", "ZScore_Value",
        ]
        return pd.DataFrame(columns=expected_cols)

    results_df = pd.DataFrame(results)
    
    # Global ranking via Priority Engine — pass weights explicitly from session_state
    # to prevent cross-session weight bleed on shared Streamlit Cloud deployments.
    if not results_df.empty:
        results_df = compute_priority(results_df, weights=_get_active_weights())
        # Default sort by Long Percentile for the global table
        results_df = results_df.sort_values('Priority_Long', ascending=False)
        
    return results_df


def run_timeseries_analysis(universe, selected_index, start_date, end_date, reg_len, wt_n1, wt_n2, levels, timeframe):
    """Compute the WRCI time-series factor frame for a date range.

    Pure compute path: fetches history, runs full / regime / divergence analyses on
    every symbol, builds the per-(date, symbol) row set, and stores ts_results_df +
    ts_meta in ``st.session_state``. **Does not render UI.** The dashboard is
    rendered separately by ``render_timeseries_dashboard()`` so it survives sidebar
    interactions / reruns.
    """
    progress_slot = st.empty()
    progress_bar(progress_slot, 5, "Fetching Historical Depth", f"Date range: {start_date} to {end_date}")

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
    console.start_phase("INTELLIGENCE HARVESTING", 2, 2)
    start_harvest = time.time()

    progress_bar(progress_slot, 15, "Initializing Processing Intelligence", f"{len(data_dict)} stocks")
    all_results = []

    for i, (ticker, df) in enumerate(data_dict.items()):
        try:
            elapsed = time.time() - start_harvest
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (len(data_dict) - (i + 1))
            eta_str = time.strftime("%M:%S", time.gmtime(remaining))
            
            # Global progress scale: 15% -> 85% for harvesting
            pct = int(15 + (i + 1) / len(data_dict) * 70)
            progress_bar(progress_slot, pct, "Intelligence Harvesting", f"Processing {i + 1}/{len(data_dict)} Symbols | ETA: {eta_str}")
            if timeframe == "Weekly":
                df = resample_to_weekly(df)
            df = run_full_analysis(df, reg_len, wt_n1, wt_n2, *levels)
            df = run_regime_analysis(df)
            df = calculate_divergences(df, timeframe=timeframe)

            # Calculate forward returns for each training horizon
            for h in pe.HOLD_HORIZONS:
                df[f'Ret_{h}b'] = df['Close'].shift(-h) / df['Close'] - 1

            # Vectorized SignalType per bar (priority order: B > A > C > D > Zone)
            df['SignalType'] = np.select(
                [
                    df['long_cond_comp'], df['short_cond_comp'],
                    df['long_cond'],      df['short_cond'],
                    df['long_cond_wt'],   df['short_cond_wt'],
                    df['long_cond_sqz'],  df['short_cond_sqz'],
                    df['Condition'] != 'Neutral',
                ],
                [
                    'B: Long', 'B: Short',
                    'A: Long', 'A: Short',
                    'C: Long', 'C: Short',
                    'D: Long', 'D: Short',
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
                    'Signal': row['Unified_Osc'],
                    'Trend': row['Norm_Trend'],
                    'Conviction': row['Conviction'],
                    'Wave': row['WT1'],
                    'WT1_5ago': row.get('WT1_5ago', row['WT1']),
                    'Zone': row['Condition'],
                    'LongSignal': row['long_cond'],
                    'ShortSignal': row['short_cond'],
                    'SignalType': row['SignalType'],
                    # Regime Intelligence columns
                    'Regime': row.get('Regime', 'NEUTRAL'),
                    'HMM_Bull': row.get('HMM_Bull', 0),
                    'HMM_Bear': row.get('HMM_Bear', 0),
                    'Vol_Regime': row.get('Vol_Regime', 'NORMAL'),
                    'Change_Point': row.get('Change_Point', False),
                    'Regime_Confidence': row.get('Regime_Confidence', 0),
                    'Bullish_Div': row.get('Bullish_Div', False),
                    'Bearish_Div': row.get('Bearish_Div', False),
                    # Forward Returns for self-training
                    'Ret_2b': row.get('Ret_2b', 0),
                    'Ret_3b': row.get('Ret_3b', 0),
                    'Ret_5b': row.get('Ret_5b', 0),
                    'Ret_8b': row.get('Ret_8b', 0),
                    'Ret_13b': row.get('Ret_13b', 0),
                    # Optimization factors
                    'F1_PriceMom': row.get('F1_PriceMom', 0),
                    'F2_VolQual': row.get('F2_VolQual', 0),
                    'Pulse': row.get('Pulse', 0),
                })
            
        except Exception as e:
            console.failure(f"Range Analysis Failed: {ticker}", str(e))
            continue
            
    console.success(f"Successfully processed {len(data_dict)} symbols for historical depth")
    console.end_phase("INTELLIGENCE HARVESTING")

    progress_slot.empty()
    if not all_results:
        st.error("No results generated for the selected timeframe.")
        return

    ts_df = pd.DataFrame(all_results)
    ts_df['Date'] = pd.to_datetime(ts_df['Date'])
    ts_df = ts_df.sort_values('Date')

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
        'Trend': 'mean',
        'Wave': 'mean',
        'LongSignal': 'sum',
        'ShortSignal': 'sum',
        'Zone': lambda x: x.value_counts().idxmax() if len(x) > 0 else 'Neutral',
        'Regime': lambda x: x.value_counts().idxmax() if len(x) > 0 else 'NEUTRAL',
        'HMM_Bull': 'mean',
        'HMM_Bear': 'mean',
        'Vol_Regime': lambda x: x.value_counts().idxmax() if len(x) > 0 else 'NORMAL',
        'Change_Point': 'sum',
        'Regime_Confidence': 'mean',
        'Bullish_Div': 'sum',
        'Bearish_Div': 'sum',
    })

    daily_agg['TotalSignals'] = daily_agg['LongSignal'] + daily_agg['ShortSignal']
    daily_agg['L_S_Ratio']    = np.where(
        daily_agg['ShortSignal'] == 0,
        np.nan,                          # undefined (all longs, no shorts) — show as NaN in charts
        daily_agg['LongSignal'] / daily_agg['ShortSignal'],
    )
    daily_agg['Conviction']   = daily_agg['Signal'].abs()

    zone_counts   = ts_df.groupby('Date')['Zone'].apply(lambda x: (x.isin(['OB Extreme', 'OB'])).sum())
    os_counts     = ts_df.groupby('Date')['Zone'].apply(lambda x: (x.isin(['OS Extreme', 'OS'])).sum())
    total_per_day = ts_df.groupby('Date').size()
    daily_agg['Oversold_Pct']   = (zone_counts / total_per_day * 100).fillna(0)
    daily_agg['Overbought_Pct'] = (os_counts   / total_per_day * 100).fillna(0)

    regime_bull  = ts_df.groupby('Date')['Regime'].apply(lambda x: x.str.contains('BULL', na=False).sum())
    regime_bear  = ts_df.groupby('Date')['Regime'].apply(lambda x: x.str.contains('BEAR', na=False).sum())
    regime_trans = ts_df.groupby('Date')['Regime'].apply(lambda x: (x == 'TRANSITION').sum())
    daily_agg['Regime_Bull_Pct']       = (regime_bull  / total_per_day * 100).fillna(0)
    daily_agg['Regime_Bear_Pct']       = (regime_bear  / total_per_day * 100).fillna(0)
    daily_agg['Regime_Transition_Pct'] = (regime_trans / total_per_day * 100).fillna(0)

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
        ui.render_metric_card("Avg Oversold", f"{summary['avg_oversold']:.1f}%",
                              timeframe_label, "success")
    with c3:
        ui.render_metric_card("Avg Overbought", f"{summary['avg_overbought']:.1f}%",
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
        ui.render_section_header("Extreme Signal Trends",
                                 "Overbought / Oversold Distribution Over Time",
                                 icon="activity", accent="cyan")
        fig_zones = go.Figure()
        fig_zones.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Oversold_Pct'],
                                       mode='lines', name='Oversold %',
                                       fill='tozeroy', fillcolor='rgba(52,211,153,0.12)',
                                       line=dict(color='#2DD4A8', width=2)))
        fig_zones.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Overbought_Pct'],
                                       mode='lines', name='Overbought %',
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
        ui.render_section_header("Divergence Persistence", "Divergence Signals Over Time",
                                 icon="trending-up", accent="amber")
        fig_div = go.Figure()
        fig_div.add_trace(go.Bar(x=daily_agg.index, y=daily_agg['Bullish_Div'],
                                 name='Bullish Divergence',
                                 marker=dict(color='#D4A853', line=dict(color='#D4A853', width=1))))
        fig_div.add_trace(go.Bar(x=daily_agg.index, y=-daily_agg['Bearish_Div'],
                                 name='Bearish Divergence',
                                 marker=dict(color='#06B6D4', line=dict(color='#06B6D4', width=1))))
        fig_div.update_layout(title='', height=300, hovermode='x unified', barmode='relative')
        apply_chart_theme(fig_div)
        st.plotly_chart(fig_div, width='stretch', key='chart_divergence')

    # ── TAB 3 · Regime Analysis ────────────────────────────────────────────
    with tab3:
        ui.render_section_header("Aggregate Signal Momentum", "Average Signal Value Over Time",
                                 icon="activity", accent="rose")
        colors = ['#2DD4A8' if v < -20 else '#E8555A' if v > 20 else '#64748B' for v in daily_agg['Signal']]
        fig_avg = go.Figure()
        fig_avg.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Signal'].clip(lower=0),
                                     fill='tozeroy', fillcolor='rgba(232,85,90,0.05)',
                                     line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig_avg.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Signal'].clip(upper=0),
                                     fill='tozeroy', fillcolor='rgba(45,212,168,0.05)',
                                     line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig_avg.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Signal'],
                                     mode='lines+markers', name='Avg Signal',
                                     line=dict(color='#D4A853', width=2),
                                     marker=dict(size=6, color=colors)))
        fig_avg.add_hline(y=20,  line=dict(color='rgba(239,68,68,0.5)', width=1, dash='dash'))
        fig_avg.add_hline(y=-20, line=dict(color='rgba(16,185,129,0.5)', width=1, dash='dash'))
        fig_avg.add_hline(y=0,   line=dict(color='rgba(255,255,255,0.3)', width=1))
        fig_avg.update_layout(title='', height=300, hovermode='x unified', yaxis=dict(range=[-80, 80]))
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
        fig_vol.add_trace(go.Scatter(x=daily_agg.index, y=vol_high.fillna(0),
                                     mode='lines+markers', name='High Vol %',
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
        display_ts = display_ts[display_cols]
        display_ts.columns = ['Date', 'Long Sig', 'Short Sig', 'Avg Signal',
                              'Oversold %', 'Overbought %',
                              'Bull Regime %', 'Bear Regime %', 'Change Pts']
        st.dataframe(
            display_ts, width='stretch', hide_index=True,
            column_config={
                'Date':         st.column_config.TextColumn(help="Trading day (YYYY-MM-DD)."),
                'Long Sig':     st.column_config.NumberColumn(help="Daily count of symbols firing a long-momentum signal (Set A long_cond)."),
                'Short Sig':    st.column_config.NumberColumn(help="Daily count of symbols firing a short-momentum signal (Set A short_cond)."),
                'Avg Signal':   st.column_config.NumberColumn(help="Cross-sectional mean of WT1 (Composite Index) on this day. Range ≈ ±100."),
                'Oversold %':   st.column_config.NumberColumn(help="Percent of universe in OB / OB-Extreme zones — interpreted as oversold-side breadth."),
                'Overbought %': st.column_config.NumberColumn(help="Percent of universe in OS / OS-Extreme zones — interpreted as overbought-side breadth."),
                'Bull Regime %':st.column_config.NumberColumn(help="Percent of universe with HMM regime label containing 'BULL'."),
                'Bear Regime %':st.column_config.NumberColumn(help="Percent of universe with HMM regime label containing 'BEAR'."),
                'Change Pts':   st.column_config.NumberColumn(help="Sum of Change_Point flags — count of symbols with a regime-state transition on this day."),
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
    plus WRCI confluence scoring for trade intelligence.
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
            target_raw, _ = fetch_batch_data(
                [target_ticker], end_date=analysis_date, days_back=_MAX_DAYS_BACK
            )
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
        rolling_corr_dict = {}
        console.item("Computing rolling correlations", f"method={method}, lookback={lookback}, cols={len(universe_returns.columns)}")

        try:
            # Convert to numpy for faster computation
            target_vals = target_returns.values

            for col in universe_returns.columns:
                try:
                    col_vals = universe_returns[col].fillna(0.0).values

                    # Compute rolling correlation using DataFrame.rolling.corr()
                    temp_df = pd.DataFrame({
                        'col': col_vals,
                        'target': target_vals
                    })
                    rolling_corr = temp_df['col'].rolling(window=lookback).corr(temp_df['target'])
                    rolling_corr_dict[col] = rolling_corr
                except Exception as col_err:
                    console.item(f"Skipping column {col}", str(col_err)[:50])
                    continue

            console.item("Rolling corr dict entries", len(rolling_corr_dict))

            if len(rolling_corr_dict) == 0:
                st.error("Could not compute rolling correlations for any column")
                return None

            rolling_corr_df = pd.DataFrame(rolling_corr_dict)
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
            wrci_results = run_screener_analysis(
                universe, selected_index, analysis_date, 20, 10, 21, (80, 40, -80, -40), timeframe,
                show_progress=False, external_progress_slot=progress_slot, progress_offset=75, progress_scale=15
            )

        progress_bar(progress_slot, 90, "Building Results DataFrame", "Computing Divergence Metrics")

        # Build correlation results dataframe
        corr_data_list = []
        for symbol in universe_returns.columns:
            if symbol not in close_df.columns or symbol not in current_corr.index:
                continue

            # Get current data
            current_price = close_df[symbol].iloc[-1] if symbol in close_df.columns else np.nan
            price_change = (close_df[symbol].pct_change().iloc[-1] * 100) if symbol in close_df.columns else np.nan
            target_price = close_df[target_ticker].iloc[-1]
            target_change = (close_df[target_ticker].pct_change().iloc[-1] * 100)

            # Get WRCI data if available — pull calibrated priorities from the
            # screener's already-computed compute_priority output so Correlation
            # Analysis benefits from any active Intelligence-mode calibration.
            wrci_signal = np.nan
            wrci_zone = "—"
            wrci_signal_type = "Neutral"
            priority_long = np.nan
            priority_short = np.nan
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

            # Compute divergence
            expected_change = current_corr[symbol] * target_change
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
            })

        corr_df = pd.DataFrame(corr_data_list)
        if len(corr_df) == 0:
            st.error("No correlation data could be computed")
            console.item("Error", "Empty correlation DataFrame")
            return None

        corr_df = corr_df.sort_values('Corr_Current', key=abs, ascending=False)

        # ── Confluence score — now calibration-aware ─────────────────────
        # Old: |Corr| × |WRCI_Signal| / 80  (raw oscillator only).
        # New: |Corr| × normalized max(|Priority_Long|, |Priority_Short|).
        # When Intelligence-mode calibration is active, Priority_Long/Short
        # already incorporates the universe-specific calibrated weights, so
        # the confluence ranking automatically benefits from the learning.
        # When defaults are active, falls back to default-weighted priorities.
        # If Priority columns are missing entirely (defensive), drop back to
        # the legacy WRCI_Signal formula.
        if 'Priority_Long' in corr_df.columns and corr_df['Priority_Long'].notna().any():
            abs_pri = pd.concat(
                [corr_df['Priority_Long'].abs(), corr_df['Priority_Short'].abs()],
                axis=1,
            ).max(axis=1).fillna(0)
            pri_norm = abs_pri / max(abs_pri.max(), 1e-6)        # [0, 1]
            corr_df['Priority_Strength'] = pri_norm
            corr_df['Confluence_Score'] = (corr_df['Corr_Current'].abs() * pri_norm).clip(0.0, 1.0)
            console.item("Confluence formula", "|Corr| × calibrated Priority strength")
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

def _build_confluence_table_html(df: pd.DataFrame) -> str:
    """Build ranked HTML table for confluence setups.

    Displays symbol, correlation, zone, signal, actual/expected/divergence, and confluence score.

    Returns: Complete HTML document string ready for st.components.v1.html().
    """
    import html as html_module

    table_rows = []
    if df.empty:
        table_rows.append(f"""
        <tr>
            <td colspan="8" style="
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
            symbol = html_module.escape(str(row.get('SimpleName', '')))
            corr = float(row.get('Corr_Current', 0))
            zone = html_module.escape(str(row.get('WRCI_Zone', 'Neutral')))
            signal_type = html_module.escape(str(row.get('WRCI_Signal_Type', '—')))
            actual = float(row.get('PctChange', 0))
            expected = float(row.get('Expected_Change', 0))
            divergence = float(row.get('Divergence', 0))
            confluence = float(row.get('Confluence_Score', 0))

            corr_color = "#34D399" if corr > 0 else "#FB7185"
            div_color = "#34D399" if divergence > 0 else "#FB7185"
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
                <th class="numeric" title="Expected move = target asset return × rolling correlation">Expected %</th>
                <th class="numeric" title="Divergence = Actual − Expected (positive = outperforming expectation)">Div %</th>
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
            "Confluence: Correlation × Momentum Signals",
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
                <span style="font-weight:600;">(2) |Div %| >3%</span>, <span style="font-weight:600;">(3) Confirmed Zone (OB/OS Extreme)</span>
            </div>
            <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:0.5rem; margin-top:0.75rem;">
                <div style="font-family:var(--data); font-size:0.65rem; color:var(--ink-secondary);">
                    <span style="color:#38BDF8; font-weight:600;">Corr</span> — Correlation strength
                </div>
                <div style="font-family:var(--data); font-size:0.65rem; color:var(--ink-secondary);">
                    <span style="color:#38BDF8; font-weight:600;">Zone</span> — Momentum extreme (OB/OS)
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
        # zone conditions ensure the oscillator agrees with the setup direction.
        _CORR_THRESH = 0.4   # minimum |correlation| to consider a relationship directional
        _DIV_THRESH  = 2.0   # minimum % divergence to flag a laggard / runaway
        def classify_setup(row):
            corr = row['Corr_Current']
            div = row['Divergence']
            zone = row['WRCI_Zone']

            if corr > _CORR_THRESH and div > _DIV_THRESH and zone in ['OS', 'OS Extreme']:
                return "LAGGARD"
            elif corr > _CORR_THRESH and div < -_DIV_THRESH and zone in ['OB', 'OB Extreme']:
                return "RUNAWAY"
            elif abs(corr) < 0.2:
                return "CONVERGING"
            elif corr < -_CORR_THRESH and div < -_DIV_THRESH and zone in ['OB', 'OB Extreme']:
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
                "rationale": "Stock lagging expected move — expect catch-up rally to target's pace",
                "validate": "Check that Zone is OS/OS Extreme and Div % is positive & large (>3%)",
                "risk": "Correlation may break; stock continues lagging instead of catching up"
            },
            "RUNAWAY": {
                "action": "SHORT",
                "rationale": "Stock overextended vs expected move — expect pullback to fair value",
                "validate": "Check that Zone is OB/OB Extreme and Div % is negative & large (<-3%)",
                "risk": "Stock may continue running; wait for Zone to weaken before shorting"
            },
            "CONVERGING": {
                "action": "DE-RISK",
                "rationale": "Correlation collapsing — pair-trade falling apart, avoid new entries",
                "validate": "Corr close to 0 or unstable; watch for re-correlation before re-entering",
                "risk": "Old positions may unwind suddenly; previous divergence trades may fail"
            },
            "CONTRA": {
                "action": "LONG (vs target down)",
                "rationale": "Negative correlation + target down — expect rally when target recovers",
                "validate": "Check Corr is strongly negative (<-0.6) and Zone is OB/OB Extreme",
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

def _bucket_signals_by_age(results_df: pd.DataFrame, side: str = 'long', condition_set: str = 'C', timeframe: str = 'Daily') -> dict:
    """Bucket signals by age (Today, 1d, 2d, 3d, 5d) with stats for timeline display.

    condition_set: 'A' = Momentum (LA_/SA_), 'B' = Crossover (LB_/SB_), 'C' = Threshold (LC_/SC_)
    timeframe: 'Daily' or 'Weekly' — determines age label names
    """
    if condition_set == 'A':
        prefix = 'LA' if side == 'long' else 'SA'
    elif condition_set == 'B':
        prefix = 'LB' if side == 'long' else 'SB'
    elif condition_set == 'C':
        prefix = 'LC' if side == 'long' else 'SC'
    elif condition_set == 'D':
        prefix = 'LD' if side == 'long' else 'SD'
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

    for age in buckets.keys():
        col = col_map[age]
        subset = results_df[(results_df[col] == target_indicator) & (~results_df['Symbol'].isin(seen))]
        for _, r in subset.iterrows():
            buckets[age].append(r)
            seen.add(r['Symbol'])

    # Compute stats for each bucket
    stats = {}
    for age, rows in buckets.items():
        if rows:
            signals = [r['Signal'] for r in rows]
            pct_changes = [r.get('PctChange', 0) for r in rows]
            convictions = [r.get('Conviction', 0) for r in rows]
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

    if newest_avg > older_avg + 5:
        trend = f"{SVGS['UP'].replace('12','14').replace('12','14')} Strengthening"
        trend_color = "#2DD4A8"
    elif newest_avg < older_avg - 5:
        trend = f"{SVGS['DOWN'].replace('12','14').replace('12','14')} Weakening"
        trend_color = "#E8555A"
    else:
        trend = "— Stable"
        trend_color = "#D4A853"

    return buckets, stats, trend, trend_color


def _build_signal_table_html(stats: dict, side: str = 'long', timeframe: str = 'Daily') -> str:
    """Build organized HTML table for signals grouped by age with section headers."""
    import html as html_module

    accent_light = "#34D399" if side == 'long' else "#FB7185"
    border_color = "rgba(45, 212, 168, 0.3)" if side == 'long' else "rgba(232, 85, 90, 0.3)"
    header_bg = "rgba(45, 212, 168, 0.15)" if side == 'long' else "rgba(232, 85, 90, 0.15)"

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
            <td colspan="9" style="padding: 0.75rem 1rem; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.8rem !important; font-weight: 700; color: {accent_light}; text-transform: uppercase; letter-spacing: 0.05em;">
                {age} · {count} signal{'s' if count != 1 else ''} · Avg Conv: {avg_conv:+.1f} · Avg %: {avg_pct:+.1f}
            </td>
        </tr>
        """)

        # Data rows for this age group
        _zone_colors = {"OB Extreme": "#FB7185", "OB": "#FCA5A5",
                        "OS Extreme": "#34D399", "OS": "#86EFAC"}
        for row in stats[age]['rows']:
            symbol = html_module.escape(str(row.get('DisplayName', row.get('Symbol', ''))))
            price = float(row.get('Price', 0))
            pct_change = float(row.get('PctChange', 0))
            signal = float(row.get('Signal', 0))
            trend = float(row.get('Trend', 0))
            conviction = float(row.get('Conviction', 0))
            pulse = float(row.get('Pulse', 0))
            conv_delta = float(row.get('Conviction_Delta', 0))
            pulse_delta = float(row.get('Pulse_Delta', 0))
            narrative = str(row.get('Narrative', 'NEUTRAL'))
            narr_color = str(row.get('Narrative_Color', '#94a3b8'))
            signal_type = str(row.get('SignalType', '-'))

            # Color % change: green for positive, red for negative
            pct_color = "#34D399" if pct_change >= 0 else "#FB7185"
            
            # Delta formatting
            conv_delta_color = "#34D399" if conv_delta >= 0 else "#FB7185"
            pulse_delta_color = "#4a9eff" if pulse_delta >= 0 else "#D4A853"
            conv_delta_arrow = "↑" if conv_delta >= 0 else "↓"
            pulse_delta_arrow = "↑" if pulse_delta >= 0 else "↓"

            table_rows.append(f"""
            <tr>
                <td class="symbol">{symbol}</td>
                <td class="numeric currency">{price:,.2f}</td>
                <td class="numeric" style="color: {pct_color}; font-weight: 600;">{pct_change:+.2f}%</td>
                <td class="numeric" style="color: {accent_light}; font-weight: 600;">{signal:+.2f}</td>
                <td class="numeric" style="color: #D4A853; font-weight: 600;">{conviction:+.2f}</td>
                <td class="numeric" style="color: {conv_delta_color}; font-size: 0.65rem; font-weight: 600;">{conv_delta_arrow}{abs(conv_delta):.1f}</td>
                <td class="numeric" style="color: #4a9eff; font-weight: 600;">{pulse:+.2f}</td>
                <td class="numeric" style="color: {pulse_delta_color}; font-size: 0.65rem; font-weight: 600;">{pulse_delta_arrow}{abs(pulse_delta):.1f}</td>
                <td class="numeric" style="color: {narr_color}; font-weight: 700; letter-spacing: 0.05em; font-size: 0.68rem !important;">{narrative}</td>
            </tr>
            """)

    if not table_rows:
        table_rows.append(f"""
        <tr>
            <td colspan="9" style="text-align:center; color:#374151; font-family:'IBM Plex Mono',monospace;
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
                    <th class="numeric">Conv</th>
                    <th class="numeric">Δ Conv</th>
                    <th class="numeric">Pulse</th>
                    <th class="numeric">Δ Pulse</th>
                    <th class="numeric">Narrative</th>
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
    import html as html_module

    accent_light = "#34D399" if side == 'long' else "#FB7185"
    border_color = "rgba(45, 212, 168, 0.3)" if side == 'long' else "rgba(232, 85, 90, 0.3)"
    
    table_rows = []
    if df.empty:
        table_rows.append(f"""
        <tr>
            <td colspan="9" style="text-align:center; color:#374151; font-family:'IBM Plex Mono',monospace;
                font-size:0.72rem; letter-spacing:0.06em; padding:2.25rem 1rem;">
                — no data available —
            </td>
        </tr>""")
    else:
        for _, row in df.iterrows():
            symbol = html_module.escape(str(row.get('DisplayName', row.get('Symbol', ''))))
            price = float(row.get('Price', 0))
            pct_change = float(row.get('PctChange', 0))
            signal = float(row.get('Signal', 0))
            conviction = float(row.get('Conviction', 0))
            pulse = float(row.get('Pulse', 0))
            conv_delta = float(row.get('Conviction_Delta', 0))
            pulse_delta = float(row.get('Pulse_Delta', 0))
            narrative = str(row.get('Narrative', 'NEUTRAL'))
            narr_color = str(row.get('Narrative_Color', '#94a3b8'))

            pct_color = "#34D399" if pct_change >= 0 else "#FB7185"
            conv_delta_color = "#34D399" if conv_delta >= 0 else "#FB7185"
            pulse_delta_color = "#4a9eff" if pulse_delta >= 0 else "#D4A853"
            conv_delta_arrow = "↑" if conv_delta >= 0 else "↓"
            pulse_delta_arrow = "↑" if pulse_delta >= 0 else "↓"

            table_rows.append(f"""
            <tr>
                <td class="symbol" style="color: #F1F5F9;">{symbol}</td>
                <td class="numeric currency">{price:,.2f}</td>
                <td class="numeric" style="color: {pct_color}; font-weight: 600;">{pct_change:+.2f}%</td>
                <td class="numeric" style="color: #60A5FA; font-weight: 600;">{signal:+.2f}</td>
                <td class="numeric" style="color: #D4A853; font-weight: 600;">{conviction:+.2f}</td>
                <td class="numeric" style="color: {conv_delta_color}; font-size: 0.65rem; font-weight: 600;">{conv_delta_arrow}{abs(conv_delta):.1f}</td>
                <td class="numeric" style="color: #4a9eff; font-weight: 600;">{pulse:+.2f}</td>
                <td class="numeric" style="color: {pulse_delta_color}; font-size: 0.65rem; font-weight: 600;">{pulse_delta_arrow}{abs(pulse_delta):.1f}</td>
                <td class="numeric" style="color: {narr_color}; font-weight: 700; letter-spacing: 0.05em; font-size: 0.68rem !important;">{narrative}</td>
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
                    <th class="numeric">Conv</th>
                    <th class="numeric">Δ Conv</th>
                    <th class="numeric">Pulse</th>
                    <th class="numeric">Δ Pulse</th>
                    <th class="numeric">Narrative</th>
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
    """Build ranked HTML table for top signals by Abnormal Acceleration (Pulse).

    Creates styled HTML table with colored accent for side (long=green, short=red),
    displaying symbol, price, signal magnitude, trend direction, and zone status.
    Prioritizes Pulse (Velocity * Z-Score) as the ranking metric.

    Returns: Complete HTML document string ready for st.components.v1.html().
    """
    import html as html_module

    accent_light = "#34D399" if side == 'long' else "#FB7185"
    border_color = "rgba(45, 212, 168, 0.3)" if side == 'long' else "rgba(232, 85, 90, 0.3)"

    table_rows = []
    if df.empty:
        table_rows.append(f"""
        <tr>
            <td colspan="7" style="
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
        _zone_colors = {"OB Extreme": "#FB7185", "OB": "#FCA5A5",
                        "OS Extreme": "#34D399", "OS": "#86EFAC"}
        for idx, (_, row) in enumerate(df.iterrows(), 1):
            symbol = html_module.escape(str(row.get('DisplayName', row.get('Symbol', ''))))
            price = float(row.get('Price', 0))
            pct_change = float(row.get('PctChange', 0))
            signal = float(row.get('Signal', 0))
            conviction = float(row.get('Conviction', 0))
            pulse = float(row.get('Pulse', 0))
            conv_delta = float(row.get('Conviction_Delta', 0))
            narrative = str(row.get('Narrative', 'NEUTRAL'))
            narr_color = str(row.get('Narrative_Color', '#94a3b8'))

            rank_str = f"{idx:02d}"
            pct_color = "#34D399" if pct_change >= 0 else "#FB7185"
            conv_delta_color = "#34D399" if conv_delta >= 0 else "#FB7185"
            conv_delta_arrow = "↑" if conv_delta >= 0 else "↓"

            # v3 Metrics
            pct_rank = float(row.get(f'Priority_{side.capitalize()}_pct', 0))
            hmm_bull = float(row.get('HMM_Bull', 0.5))
            hmm_bear = float(row.get('HMM_Bear', 0.5))
            vol_reg  = str(row.get('Vol_Regime', 'NORMAL'))
            
            # Regime Logic
            regime_tag = "NEUTRAL"
            regime_color = "#94a3b8"
            if side == 'long':
                if hmm_bull > 0.7: regime_tag, regime_color = "BULL", "#34D399"
                elif hmm_bull < 0.3: regime_tag, regime_color = "BEAR", "#FB7185"
            else:
                if hmm_bear > 0.7: regime_tag, regime_color = "BEAR", "#FB7185"
                elif hmm_bear < 0.3: regime_tag, regime_color = "BULL", "#34D399"
                
            vol_color = {"LOW": "#60a5fa", "NORMAL": "#94a3b8", "HIGH": "#fbbf24", "EXTREME": "#f87171"}.get(vol_reg, "#94a3b8")

            table_rows.append(f"""
            <tr>
                <td class="numeric" style="color: #D4A853; font-weight: 700;">{rank_str}</td>
                <td class="symbol">{symbol}</td>
                <td class="numeric" style="color: #4a9eff; font-weight: 700;">TOP {101-pct_rank:,.1f}%</td>
                <td class="numeric currency">{price:,.2f}</td>
                <td class="numeric" style="color: {pct_color}; font-weight: 600;">{pct_change:+.2f}%</td>
                <td class="numeric" style="color: {accent_light}; font-weight: 600;">{signal:+.2f}</td>
                <td class="numeric" style="color: #D4A853; font-weight: 600;">{conviction:+.2f}</td>
                <td class="numeric" style="color: {conv_delta_color}; font-size: 0.65rem; font-weight: 600;">{conv_delta_arrow}{abs(conv_delta):.1f}</td>
                <td class="numeric" style="color: #4a9eff; font-weight: 600;">{pulse:+.2f}</td>
                <td class="numeric" style="color: {regime_color}; font-weight: 700; font-size: 0.65rem;">{regime_tag}</td>
                <td class="numeric" style="color: {vol_color}; font-weight: 700; font-size: 0.65rem;">{vol_reg}</td>
                <td class="numeric" style="color: {narr_color}; font-weight: 700; letter-spacing: 0.05em; font-size: 0.68rem !important;">{narrative}</td>
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
                    <th class="numeric">Conv</th>
                    <th class="numeric">Δ Conv</th>
                    <th class="numeric">Pulse</th>
                    <th class="numeric">Regime</th>
                    <th class="numeric">Vol</th>
                    <th class="numeric">Narrative</th>
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
    ("Set A · Momentum",  "amber",
     "Composite Line crosses Signal Line anywhere outside the ±40 extreme zones. "
     "Captures building momentum — the broadest active-trend signal."),
    ("Set B · Crossover", "violet",
     "Lines cross inside the ±40 extreme zones (contrarian). "
     "Marks momentum exhaustion — high-precision reversal timing."),
    ("Set C · Threshold", "cyan",
     "Composite Line freshly enters OS / OB zone from neutral. "
     "Earliest actionable signal — sorted by Pulse for recency."),
    ("Set D · Squeeze",   "rose",
     "Squeeze release fires (Bollinger Bands exit Keltner Channels). "
     "Volatility expansion incoming — direction inferred from SQZ momentum."),
]

# Narrative grid: rows are Pulse states (top → bottom: SURGE, FIRM, SOFT, CRUSH).
# Cols are Conviction states (left → right: LEAD, DEEP, LIGHT, HOLLOW).
# Each cell: (label, multiplier, color). Colors mirror priority_engine narratives.
_NARRATIVE_GRID = [
    # SURGE row (>> Pulse)
    [("SQUEEZE", 2.0, "#22c55e"), ("HYPER",   1.6, "#34d399"), ("EXHAUST", 0.6, "#fbbf24"), ("CHAOS",   0.4, "#f87171")],
    # FIRM row (> Pulse)
    [("IGNITE",  1.4, "#4a9eff"), ("ORGANIC", 1.2, "#38bdf8"), ("EXPAND",  0.9, "#94a3b8"), ("POP",     0.7, "#f87171")],
    # SOFT row (< Pulse)
    [("HARDEN",  1.5, "#22c55e"), ("STEALTH", 1.3, "#818cf8"), ("RETRACE", 0.8, "#cbd5e1"), ("LIQUID",  0.3, "#ef4444")],
    # CRUSH row (<< Pulse)
    [("LOAD",    1.7, "#4a9eff"), ("TRAP",    0.5, "#fbbf24"), ("CAPITUL", 0.2, "#ef4444"), ("CRASH",   0.1, "#b91c1c")],
]
_PULSE_AXIS_LABELS      = [("SURGE", ">>"), ("FIRM", ">"), ("SOFT", "<"), ("CRUSH", "<<")]
_CONVICTION_AXIS_LABELS = [("LEAD", ">>"),  ("DEEP", ">"), ("LIGHT", "<"), ("HOLLOW", "<<")]


def _render_system_data_tab(results_df, analysis_date, universe=None, selected_index=None):
    """System Data tab — exports, raw factor frame, signal-type legend, narrative matrix.

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
                "Includes a Legend sheet defining each column: signal conditions (A/B/C/D), "
                "factor descriptions (Price Momentum, Vol Quality, Conviction, Pulse, Wave), "
                "and zone/regime definitions."
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
    cols = ["DisplayName", "Price", "Signal", "Narrative", "Priority_Long",
            "Priority_Long_pct", "F1_PriceMom", "F2_VolQual"]
    if "% Chng Since" in results_df.columns and results_df["% Chng Since"].notna().any():
        cols.insert(2, "% Chng Since")
    cols += ["Conviction", "Conviction_Delta", "Pulse", "Pulse_Delta", "Wave"]
    l_cols = [c for c in results_df.columns
              if c.startswith('L_') and (c[2:].replace('d', '').isdigit() or c == 'L_Today')]
    cols += sorted(l_cols)
    cols = [c for c in cols if c in results_df.columns]
    # Rename internal factor column names to domain-readable labels for display
    _col_display_names = {
        "DisplayName":       "Symbol",
        "F1_PriceMom":       "Price Momentum",
        "F2_VolQual":        "Vol Quality",
        "Priority_Long_pct": "Long Priority %ile",
        "Conviction_Delta":  "Conv Δ",
        "Pulse_Delta":       "Pulse Δ",
    }
    display_frame = results_df[cols].sort_values("Priority_Long", ascending=False).rename(columns=_col_display_names)
    st.dataframe(display_frame, width='stretch', height=500)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Signal Type Reference ─────────────────────────────────────────────
    ui.render_section_header(
        "Signal Type Reference",
        "Four signal classes used across screens — A · B · C · D",
        icon="info", accent="amber",
    )
    ref_cols = st.columns(4)
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

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Narrative Matrix (4×4) ───────────────────────────────────────────
    ui.render_section_header(
        "Narrative Matrix",
        "16 narratives indexed by Pulse state (rows) × Conviction state (columns) · multiplier shown per cell",
        icon="grid", accent="violet",
    )

    # Build header row (Conviction axis labels)
    header_cells = [
        '<th style="padding:0.5rem 0.4rem; background:transparent; border-bottom:1px solid var(--border); '
        'font-family:var(--data); font-size:0.55rem; color:var(--ink-tertiary); '
        'text-transform:uppercase; letter-spacing:0.12em; text-align:center; width:8%;">'
        '</th>'  # corner empty
    ]
    for label, sign in _CONVICTION_AXIS_LABELS:
        header_cells.append(
            f'<th style="padding:0.5rem 0.4rem; background:transparent; '
            f'border-bottom:1px solid var(--border); '
            f'font-family:var(--data); font-size:0.6rem; color:var(--ink-secondary); '
            f'text-transform:uppercase; letter-spacing:0.1em; text-align:center; width:23%;">'
            f'<div style="font-weight:700; color:var(--ink-primary);">{label}</div>'
            f'<div style="font-size:0.55rem; color:var(--ink-tertiary); margin-top:0.1rem;">{sign}</div>'
            f'</th>'
        )

    # Build body rows
    body_rows = []
    for row_idx, (pulse_label, pulse_sign) in enumerate(_PULSE_AXIS_LABELS):
        row_cells = [
            f'<td style="padding:0.6rem 0.4rem; border-right:1px solid var(--border); '
            f'background:transparent; vertical-align:middle; text-align:center;">'
            f'<div style="font-family:var(--display); font-size:0.62rem; font-weight:700; '
            f'color:var(--ink-primary); letter-spacing:0.08em;">{pulse_label}</div>'
            f'<div style="font-family:var(--data); font-size:0.55rem; color:var(--ink-tertiary);">{pulse_sign}</div>'
            f'</td>'
        ]
        for label, mult, color in _NARRATIVE_GRID[row_idx]:
            opacity = max(0.06, min(0.18, mult / 12.0))  # vivid for high mult, faint for low
            row_cells.append(
                f'<td style="padding:0.55rem 0.4rem; border:1px solid rgba(255,255,255,0.04); '
                f'border-left:3px solid {color}; '
                f'background:linear-gradient(180deg, rgba({_hex_to_rgb(color)},{opacity:.3f}) 0%, rgba(0,0,0,0) 100%); '
                f'vertical-align:middle; text-align:center;">'
                f'<div style="font-family:var(--display); font-size:0.72rem; font-weight:700; '
                f'color:{color}; letter-spacing:0.04em;">{label}</div>'
                f'<div style="font-family:var(--data); font-size:0.62rem; color:var(--ink-secondary); '
                f'margin-top:0.15rem; font-weight:600;">{mult:.1f}×</div>'
                f'</td>'
            )
        body_rows.append('<tr>' + ''.join(row_cells) + '</tr>')

    matrix_html = (
        '<table style="width:100%; border-collapse:collapse; '
        'font-family:var(--data); margin:0.4rem 0;">'
        '<thead><tr>' + ''.join(header_cells) + '</tr></thead>'
        '<tbody>' + ''.join(body_rows) + '</tbody>'
        '</table>'
    )
    st.markdown(matrix_html, unsafe_allow_html=True)

    st.markdown(
        '<div style="font-family:var(--data); font-size:0.62rem; color:var(--ink-tertiary); '
        'padding-top:0.6rem; line-height:1.5;">'
        'Multiplier scales the priority score. Cells with multiplier &gt; 1 amplify ranking '
        '(directional clarity), &lt; 1 dampen it (regime noise / exhaustion). The active '
        'narrative for each symbol appears in the <b>Narrative</b> column of the table above.'
        '</div>',
        unsafe_allow_html=True,
    )


def _hex_to_rgb(hex_color: str) -> str:
    """'#22c55e' → '34, 197, 94' for use inside rgba()."""
    h = hex_color.lstrip('#')
    if len(h) != 6:
        return "148, 163, 184"  # slate fallback
    return f"{int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}"


def _build_active_weights_table_html(active_weights: dict) -> str:
    """Render active β/γ/tier weights as an HTML table consistent with screener tables.

    Splits long/short pairs into a side-by-side layout so the user can see the
    asymmetry the optimizer found. Tier multipliers (shared) are listed below.
    """
    factor_pairs = [
        ("F1 · PriceMom",   "beta_F1_pricemom_long",   "beta_F1_pricemom_short"),
        ("F2 · VolQual",    "beta_F2_volqual_long",    "beta_F2_volqual_short"),
        ("F3 · Wave",       "beta_F3_wave_long",       "beta_F3_wave_short"),
        ("F4 · Pulse",      "beta_F4_pulse_long",      "beta_F4_pulse_short"),
        ("F5 · Regime",     "beta_F5_regime_long",     "beta_F5_regime_short"),
        ("F6 · X-Sect",     "beta_F6_xsect_long",      "beta_F6_xsect_short"),
        ("γ · Reversion",   "gamma_reversion_long",    "gamma_reversion_short"),
        ("γ · Divergence",  "gamma_divergence_long",   "gamma_divergence_short"),
    ]
    tier_pairs = [
        ("Set A · Momentum",  "tier_A_mult"),
        ("Set B · Crossover", "tier_B_mult"),
        ("Set C · Threshold", "tier_C_mult"),
        ("Set D · Squeeze",   "tier_D_mult"),
        ("Default",           "tier_default_mult"),
    ]

    factor_rows = []
    for label, lk, sk in factor_pairs:
        lv = float(active_weights.get(lk, 0.0))
        sv = float(active_weights.get(sk, 0.0))
        delta = lv - sv
        delta_color = "#34D399" if delta > 0.5 else "#FB7185" if delta < -0.5 else "#94A3B8"
        delta_sign  = "+" if delta >= 0 else ""
        factor_rows.append(f"""
            <tr>
                <td class="aw-label">{label}</td>
                <td class="aw-num aw-long">{lv:.2f}</td>
                <td class="aw-num aw-short">{sv:.2f}</td>
                <td class="aw-num aw-delta" style="color:{delta_color};">{delta_sign}{delta:.2f}</td>
            </tr>
        """)

    tier_rows = []
    for label, key in tier_pairs:
        v = float(active_weights.get(key, 1.0))
        v_color = "#34D399" if v > 1.05 else "#FB7185" if v < 0.95 else "#94A3B8"
        tier_rows.append(f"""
            <tr>
                <td class="aw-label">{label}</td>
                <td class="aw-num" style="color:{v_color};">{v:.2f}×</td>
            </tr>
        """)

    return f"""
    <html><head><style>
        body {{
            margin:0; padding:0;
            background:transparent;
            font-family:'IBM Plex Mono', ui-monospace, Menlo, monospace;
            color:#E2E8F0;
        }}
        .aw-table {{
            width:100%;
            border-collapse:collapse;
            margin-bottom:1rem;
            background:rgba(255,255,255,0.015);
            border:1px solid rgba(255,255,255,0.06);
            border-radius:6px;
            overflow:hidden;
        }}
        .aw-table thead th {{
            font-family:'Space Grotesk', system-ui, sans-serif;
            font-size:0.58rem;
            font-weight:700;
            color:#64748B;
            text-transform:uppercase;
            letter-spacing:0.12em;
            text-align:right;
            padding:0.55rem 0.7rem;
            background:rgba(0,0,0,0.18);
            border-bottom:1px solid rgba(255,255,255,0.06);
        }}
        .aw-table thead th:first-child {{ text-align:left; }}
        .aw-table tbody tr {{
            border-bottom:1px solid rgba(255,255,255,0.03);
        }}
        .aw-table tbody tr:last-child {{ border-bottom:0; }}
        .aw-table tbody tr:hover {{ background:rgba(255,255,255,0.018); }}
        .aw-label {{
            font-family:'Space Grotesk', system-ui, sans-serif;
            font-size:0.7rem;
            font-weight:600;
            color:#CBD5E1;
            padding:0.5rem 0.7rem;
            text-align:left;
            letter-spacing:0.02em;
        }}
        .aw-num {{
            font-size:0.7rem;
            font-weight:600;
            text-align:right;
            padding:0.5rem 0.7rem;
            font-variant-numeric:tabular-nums;
        }}
        .aw-long  {{ color:#2DD4A8; }}
        .aw-short {{ color:#E8555A; }}
        .aw-delta {{ font-size:0.65rem; font-weight:500; }}

        .aw-section-title {{
            font-family:'Space Grotesk', system-ui, sans-serif;
            font-size:0.6rem;
            font-weight:700;
            color:#D4A853;
            text-transform:uppercase;
            letter-spacing:0.14em;
            padding:0.5rem 0 0.4rem 0.1rem;
            margin-top:0.4rem;
        }}
        .aw-footnote {{
            font-family:'IBM Plex Mono', monospace;
            font-size:0.6rem;
            color:#475569;
            line-height:1.5;
            padding:0.5rem 0.2rem 0;
        }}
    </style></head><body>

    <div class="aw-section-title">FACTOR WEIGHTS · LONG vs SHORT</div>
    <table class="aw-table">
        <thead>
            <tr>
                <th>Factor</th>
                <th>Long</th>
                <th>Short</th>
                <th>Δ</th>
            </tr>
        </thead>
        <tbody>{''.join(factor_rows)}</tbody>
    </table>

    <div class="aw-section-title">TIER MULTIPLIERS · SHARED</div>
    <table class="aw-table">
        <thead>
            <tr>
                <th>Signal Class</th>
                <th>Multiplier</th>
            </tr>
        </thead>
        <tbody>{''.join(tier_rows)}</tbody>
    </table>

    <div class="aw-footnote">
        Long / Short are independent weight vectors — Δ &gt; 0 means the factor
        contributes more to the bullish ranking than the bearish, and vice versa.
        Tier multipliers scale signal-class quality and are direction-agnostic.
    </div>

    </body></html>
    """


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
        _profiles = pe.list_profiles()
        if _profiles:
            console.item("Calibrated profiles", f"{len(_profiles)} on disk")
            for _p in _profiles[:3]:  # show up to 3 entries
                _label = _p.get("selected_index") or _p.get("universe") or "—"
                _ir    = _p.get("val_score")
                _ir_s  = f"{_ir:+.3f}" if isinstance(_ir, (int, float)) else "—"
                console.item(f"  · {_label}", f"val IR {_ir_s} · {_p.get('timestamp', '—')}")
        else:
            console.item("Calibrated profiles", "0 (running on factory defaults)")

    # Render sidebar and get parameters + run button state
    sidebar_out = render_sidebar()
    universe, selected_index, analysis_date, reg_len, wt_n1, wt_n2, levels, timeframe, mode, start_date, end_date, run_clicked, corr_target_ticker, corr_lookback, corr_method, calib_settings = sidebar_out
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
            results_df = run_screener_analysis(
                universe, selected_index, analysis_date,
                reg_len, wt_n1, wt_n2, levels, timeframe,
            )
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
            console.header("SANKET TERMINAL — Bulk Range Intelligence", VERSION)
            run_timeseries_analysis(
                universe, selected_index, start_date, end_date,
                reg_len, wt_n1, wt_n2, levels, timeframe,
            )

        elif mode == "Intelligence (Self-Tuning)":
            console.header("SANKET TERMINAL — Intelligence Harvest", VERSION)
            run_timeseries_analysis(
                universe, selected_index, start_date, end_date,
                reg_len, wt_n1, wt_n2, levels, timeframe,
            )

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
    elif mode in ("Historical Range", "Intelligence (Self-Tuning)") and not st.session_state.get("timeseries_done"):
        show_landing = True

    if show_landing:
        ui.render_header("SANKET", "Market Signal Screener · संकेत · WRCI Engine")
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
            for _col in ['LA_Today','LA_1d','LA_2d','LA_3d','LA_5d','SA_Today','SA_1d','SA_2d','SA_3d','SA_5d','LB_Today','LB_1d','LB_2d','LB_3d','LB_5d','SB_Today','SB_1d','SB_2d','SB_3d','SB_5d','LC_Today','LC_1d','LC_2d','LC_3d','LC_5d','SC_Today','SC_1d','SC_2d','SC_3d','SC_5d']:
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
                        f"{_pn_n} / {_pn_total} symbols · {_pn_date} · Full universe ranking by Abnormal Acceleration",
                        icon="zap", accent="amber"
                    )
                    avg_pulse = results_df['Pulse'].mean()
                    avg_conv = results_df['Conviction'].mean()
                    strong_pulse = len(results_df[results_df['Pulse'].abs() > 10])
                    bullish_bias = (results_df['Signal'] > 0).sum() / len(results_df) * 100 if len(results_df) > 0 else 0
                    m1, m2, m3, m4 = st.columns(4)
                    with m1: ui.render_metric_card("Universe Pulse", f"{avg_pulse:+.2f}", "Avg Acceleration (range ±100)", "neutral")
                    with m2: ui.render_metric_card("Universe Conv", f"{avg_conv:+.2f}", "Avg Conviction (range ±100)", "neutral")
                    with m3: ui.render_metric_card("High Pulse", str(strong_pulse), f"{strong_pulse/len(results_df)*100 if len(results_df)>0 else 0:.0f}% Universe", "info")
                    with m4: ui.render_metric_card("Bullish Bias", f"{bullish_bias:.0f}%", "Signal > 0", "success" if bullish_bias > 50 else "danger")
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
                        "Abnormal Acceleration (Pulse)",
                        "Top 10 long / short candidates by calibrated Priority",
                        icon="zap", accent="amber",
                    )
                    pn_top_longs  = results_df.sort_values('Priority_Long',  ascending=False).head(10)
                    pn_top_shorts = results_df.sort_values('Priority_Short', ascending=False).head(10)

                    pn_avg_pulse  = results_df['Pulse'].abs().mean()
                    pn_avg_conv   = results_df['Conviction'].abs().mean()
                    pn_strong_p   = len(results_df[results_df['Pulse'].abs() > 10])
                    pn_strong_t   = len(results_df[results_df['Trend'].abs() > 30])

                    s1, s2, s3, s4 = st.columns(4)
                    with s1: ui.render_metric_card("Avg Pulse",      f"{pn_avg_pulse:.1f}", "Abnormal Acceleration (abs, ±100)", "neutral")
                    with s2: ui.render_metric_card("Avg Conviction", f"{pn_avg_conv:.1f}",  "Blended confluence (abs, ±100)",    "neutral")
                    with s3: ui.render_metric_card("Strong Pulse",   str(pn_strong_p),
                                                   f"{pn_strong_p/len(results_df)*100:.0f}% of universe", "info")
                    with s4: ui.render_metric_card("Strong Trends",  str(pn_strong_t),
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
                tab_signals, tab_strength, tab_raw = st.tabs(["Action Dashboard", "Signal Strength", "System Data"])
                with tab_signals:
                    timeframe_label = "This Week's" if timeframe == 'Weekly' else "Today's"
                    _run_stats = st.session_state.get("screener_run_stats", {})
                    _n_analyzed = _run_stats.get("analyzed", len(results_df))
                    _n_universe = _run_stats.get("total_in_universe", _n_analyzed)
                    _date_str   = analysis_date.strftime("%d %b %Y") if hasattr(analysis_date, "strftime") else str(analysis_date)
                    ui.render_section_header(
                        f"{timeframe_label} Signals",
                        f"{_n_analyzed} / {_n_universe} symbols · {timeframe} · {_date_str} · "
                        "Momentum (A) · Crossover (B) · Threshold (C) · Squeeze (D)",
                        icon="zap",
                        accent="amber"
                    )

                    # Set C: Momentum — sorted by Directional Priority v3
                    longs_df = results_df[results_df['L_5d'] != "—"].copy().sort_values('Priority_Long', ascending=False)
                    shorts_df = results_df[results_df['S_5d'] != "—"].copy().sort_values('Priority_Short', ascending=False)

                    # Set A: Momentum — broad crossover anywhere
                    has_bullish_crossover = (results_df[['LB_Today', 'LB_1d', 'LB_2d', 'LB_3d', 'LB_5d']] != "—").any(axis=1)
                    has_bearish_crossover = (results_df[['SB_Today', 'SB_1d', 'SB_2d', 'SB_3d', 'SB_5d']] != "—").any(axis=1)

                    longs_a_df = results_df[(results_df['LA_5d'] != "—") & ~has_bearish_crossover].copy().sort_values('Priority_Long', ascending=False)
                    shorts_a_df = results_df[(results_df['SA_5d'] != "—") & ~has_bullish_crossover].copy().sort_values('Priority_Short', ascending=False)

                    # Set B: Crossover
                    longs_b_df = results_df[results_df['LB_5d'] != "—"].copy().sort_values('Priority_Long', ascending=False)
                    shorts_b_df = results_df[results_df['SB_5d'] != "—"].copy().sort_values('Priority_Short', ascending=False)

                    # Set C: Threshold
                    longs_c_df = results_df[results_df['LC_5d'] != "—"].copy().sort_values('Priority_Long', ascending=False)
                    shorts_c_df = results_df[results_df['SC_5d'] != "—"].copy().sort_values('Priority_Short', ascending=False)

                    # Set D: Squeeze
                    longs_d_df = results_df[results_df['LD_5d'] != "—"].copy().sort_values('Priority_Long', ascending=False)
                    shorts_d_df = results_df[results_df['SD_5d'] != "—"].copy().sort_values('Priority_Short', ascending=False)

                    if timeframe == 'Weekly':
                        _age_order = ["This Week", "1 Week Ago", "2 Weeks Ago", "3 Weeks Ago", "Within 5 Weeks"]
                    else:
                        _age_order = ["Today", "1 Day Ago", "2 Days Ago", "3 Days Ago", "Within 5 Days"]

                    has_signals = any(not df_.empty for df_ in [longs_a_df, shorts_a_df, longs_b_df, shorts_b_df, longs_c_df, shorts_c_df, longs_d_df, shorts_d_df])

                    if has_signals:
                        total_longs  = len(longs_a_df) + len(longs_b_df) + len(longs_c_df) + len(longs_d_df)
                        total_shorts = len(shorts_a_df) + len(shorts_b_df) + len(shorts_c_df) + len(shorts_d_df)
                        all_longs  = pd.concat([longs_a_df, longs_b_df, longs_c_df, longs_d_df]).drop_duplicates('Symbol').sort_values('Priority_Long', ascending=False)
                        all_shorts = pd.concat([shorts_a_df, shorts_b_df, shorts_c_df, shorts_d_df]).drop_duplicates('Symbol').sort_values('Priority_Short', ascending=False)

                        mc1, mc2, mc3, mc4 = st.columns(4)
                        with mc1: ui.render_metric_card("Long Signals", str(total_longs), f"A:{len(longs_a_df)} B:{len(longs_b_df)} C:{len(longs_c_df)} D:{len(longs_d_df)}", "success")
                        with mc2: ui.render_metric_card("Short Signals", str(total_shorts), f"A:{len(shorts_a_df)} B:{len(shorts_b_df)} C:{len(shorts_c_df)} D:{len(shorts_d_df)}", "danger")
                        with mc3:
                            strongest_long = all_longs.iloc[0] if not all_longs.empty else None
                            ui.render_metric_card("Strongest Long", strongest_long['SimpleName'] if strongest_long is not None else "—", f"Signal: {strongest_long['Signal']:.1f}" if strongest_long is not None else "No signals", "info")
                        with mc4:
                            strongest_short = all_shorts.iloc[0] if not all_shorts.empty else None
                            ui.render_metric_card("Strongest Short", strongest_short['SimpleName'] if strongest_short is not None else "—", f"Signal: {strongest_short['Signal']:.1f}" if strongest_short is not None else "No signals", "info")

                        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                        bull_tab, bear_tab = st.tabs(["Bullish Signals by Timing", "Bearish Signals by Timing"])
                        with bull_tab:
                            mom_bull_tab, cross_bull_tab, thresh_bull_tab, sqz_bull_tab = st.tabs(["Momentum", "Crossover", "Threshold", "Squeeze"])
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
                            with thresh_bull_tab:
                                _, lc_stats, _, _ = _bucket_signals_by_age(longs_c_df, side='long', condition_set='C', timeframe=timeframe)
                                lc_html = _build_signal_table_html(lc_stats, side='long', timeframe=timeframe)
                                _g = sum(1 for a in _age_order if lc_stats[a]['count'] > 0)
                                _r = sum(lc_stats[a]['count'] for a in _age_order)
                                st.components.v1.html(lc_html, height=max(70 + _g * 46 + _r * 44, 110))
                            with sqz_bull_tab:
                                _, ld_stats, _, _ = _bucket_signals_by_age(longs_d_df, side='long', condition_set='D', timeframe=timeframe)
                                ld_html = _build_signal_table_html(ld_stats, side='long', timeframe=timeframe)
                                _g = sum(1 for a in _age_order if ld_stats[a]['count'] > 0)
                                _r = sum(ld_stats[a]['count'] for a in _age_order)
                                st.components.v1.html(ld_html, height=max(70 + _g * 46 + _r * 44, 110))
                        with bear_tab:
                            mom_bear_tab, cross_bear_tab, thresh_bear_tab, sqz_bear_tab = st.tabs(["Momentum", "Crossover", "Threshold", "Squeeze"])
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
                            with thresh_bear_tab:
                                _, sc_stats, _, _ = _bucket_signals_by_age(shorts_c_df, side='short', condition_set='C', timeframe=timeframe)
                                sc_html = _build_signal_table_html(sc_stats, side='short', timeframe=timeframe)
                                _g = sum(1 for a in _age_order if sc_stats[a]['count'] > 0)
                                _r = sum(sc_stats[a]['count'] for a in _age_order)
                                st.components.v1.html(sc_html, height=max(70 + _g * 46 + _r * 44, 110))
                            with sqz_bear_tab:
                                _, sd_stats, _, _ = _bucket_signals_by_age(shorts_d_df, side='short', condition_set='D', timeframe=timeframe)
                                sd_html = _build_signal_table_html(sd_stats, side='short', timeframe=timeframe)
                                _g = sum(1 for a in _age_order if sd_stats[a]['count'] > 0)
                                _r = sum(sd_stats[a]['count'] for a in _age_order)
                                st.components.v1.html(sd_html, height=max(70 + _g * 46 + _r * 44, 110))
                    else:
                        st.info(
                            f"**No signals found** for {selected_index} on {analysis_date} ({timeframe}). "
                            "All symbols were analyzed but none crossed the WRCI signal thresholds. "
                            "Try an adjacent trading date or switch to a broader universe."
                        )


                # Omni-channel base: include any stock with a signal in ANY set (A, B, C, or D)
                long_sets = ['LA_5d', 'LB_5d', 'LC_5d', 'LD_5d']
                short_sets = ['SA_5d', 'SB_5d', 'SC_5d', 'SD_5d']
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
                        "Abnormal Acceleration (Pulse)",
                        "Top signals ranked by Pulse — Momentum (A) · Crossover (B) · Threshold (C) · Squeeze (D)",
                        icon="zap",
                        accent="amber"
                    )

                    # Strength metrics
                    avg_pulse = results_df['Pulse'].abs().mean()
                    avg_conv = results_df['Conviction'].abs().mean()
                    strong_pulse_count = len(results_df[results_df['Pulse'].abs() > 10])
                    strong_trend_count = len(results_df[results_df['Trend'].abs() > 30])

                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                    with col_s1: ui.render_metric_card("Avg Pulse", f"{avg_pulse:.1f}", "Abnormal Acceleration", "neutral")
                    with col_s2: ui.render_metric_card("Avg Conviction", f"{avg_conv:.1f}", "Blended confluence", "neutral")
                    with col_s3: ui.render_metric_card("Strong Pulse", str(strong_pulse_count), f"{strong_pulse_count/len(results_df)*100:.0f}% of universe", "info")
                    with col_s4: ui.render_metric_card("Strong Trends", str(strong_trend_count), f"{strong_trend_count/len(results_df)*100:.0f}% of universe", "info")


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
                                     border:1px solid rgba(212,168,83,0.3); border-radius:4px;">PRIORITY ENGINE</span>
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
        # so sidebar interactions don't blank the view. Intelligence mode shows
        # the Intelligence Center instead — the harvest is just a precondition,
        # not the deliverable.
        if st.session_state.get("timeseries_done") and mode == "Historical Range":
            render_timeseries_dashboard()

        # ── Correlation results ───────────────────────────────────────────
        if mode == "Correlation Analysis" and st.session_state.get("corr_data") is not None:
            render_correlation_results(st.session_state["corr_data"])

        # ── Intelligence Center (calibration UI) ──────────────────────────
        if st.session_state.get("timeseries_done") and mode == "Intelligence (Self-Tuning)":
            render_intelligence_center()

        # Always render footer
        render_footer()

def _render_model_passport_sidebar(current_universe: str, current_index, current_timeframe=None):
    """Sidebar Passport — visible in every mode.

    Surfaces:
      • Profile state (Default / Calibrated)
      • The universe + timeframe the profile was fit on
      • Train + Val IR + last-updated timestamp
      • A mismatch banner when the calibrated universe or timeframe ≠ current
        sidebar selection (weights learned on daily data don't generalize to
        weekly and vice-versa).

    Caller must be inside a ``with st.sidebar:`` context.
    """
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Model Passport</div>', unsafe_allow_html=True)

    res = st.session_state.get("opt_results")

    # What universe/timeframe is this profile from? What's selected now?
    cal_universe  = (res.get("universe")       if res else None) or None
    cal_index     = (res.get("selected_index") if res else None) or None
    cal_timeframe = (res.get("timeframe")      if res else None) or None
    cal_label     = cal_index or cal_universe or "—"
    cur_label     = (current_index or current_universe or "—")
    universe_mismatch  = bool(res) and cal_label != "—" and cur_label != "—" and cal_label != cur_label
    timeframe_mismatch = (bool(res) and cal_timeframe and current_timeframe
                          and cal_timeframe != current_timeframe)
    mismatch = universe_mismatch or timeframe_mismatch

    if res:
        train_v = res.get('train_score', 0.0) or 0.0
        val_v   = res.get('val_score',   0.0) or 0.0
        train_str = f"{train_v:+.3f}"
        val_str   = f"{val_v:+.3f}"
        updated   = res.get('timestamp', '—')
        train_color = "var(--emerald)" if train_v > 0 else "var(--rose)"
        val_color   = "var(--emerald)" if val_v   > 0 else "var(--rose)"
        cal_tf_disp = cal_timeframe or "—"
        if mismatch:
            profile_label = "Calibrated · ⚠"
            card_class    = "warning"
        else:
            profile_label = "Calibrated"
            card_class    = "success" if (val_v > 0 and train_v > 0) else "warning"
    else:
        profile_label = "Default"
        train_str = val_str = updated = "—"
        cal_tf_disp = "—"
        train_color = val_color = "var(--ink-secondary)"
        card_class  = "neutral"

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
        <h4 style="margin:0 0 0.3rem 0;">Profile</h4>
        <h2 style="font-size:1.05rem; margin:0 0 0.7rem 0; letter-spacing:-0.01em;">{profile_label}</h2>
        <div style="display:flex; flex-direction:column; gap:0.32rem;
                    padding-top:0.55rem;
                    border-top:1px solid rgba(255,255,255,0.06);">
            <div style="display:flex; justify-content:space-between; align-items:baseline; font-family:var(--data); font-size:0.62rem;">
                <span style="color:var(--ink-tertiary); text-transform:uppercase; letter-spacing:0.1em; font-size:0.58rem;">Trained on</span>
                <span style="color:var(--ink-secondary); font-weight:500; max-width:62%; text-align:right; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{cal_label_disp}</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:baseline; font-family:var(--data); font-size:0.62rem;">
                <span style="color:var(--ink-tertiary); text-transform:uppercase; letter-spacing:0.1em; font-size:0.58rem;">Depth</span>
                <span style="color:var(--ink-secondary); font-weight:500;">{cal_tf_disp}</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:baseline; font-family:var(--data); font-size:0.65rem;">
                <span style="color:var(--ink-tertiary); text-transform:uppercase; letter-spacing:0.1em; font-size:0.58rem;">Train IR</span>
                <span style="color:{train_color}; font-weight:600;">{train_str}</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:baseline; font-family:var(--data); font-size:0.65rem;">
                <span style="color:var(--ink-tertiary); text-transform:uppercase; letter-spacing:0.1em; font-size:0.58rem;">Val IR</span>
                <span style="color:{val_color}; font-weight:600;">{val_str}</span>
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
                f"Profile fit on <b>{_trim(cal_label, 28)}</b><br>"
                f"Active universe is <b>{_trim(cur_label, 28)}</b>"
            )
        if timeframe_mismatch:
            mismatch_lines.append(
                f"Profile depth is <b>{cal_timeframe}</b><br>"
                f"Active depth is <b>{current_timeframe}</b>"
            )
        mismatch_body = "<br>".join(mismatch_lines)
        st.markdown(f"""
        <div style="font-family:var(--data); font-size:0.62rem; color:var(--amber);
                    background:rgba(212,168,83,0.08);
                    border:1px solid rgba(212,168,83,0.22);
                    border-radius:6px; padding:0.55rem 0.65rem;
                    margin-bottom:0.7rem; line-height:1.45;">
            <span style="font-weight:700;">Profile mismatch — calibrated weights are still active.</span><br>
            {mismatch_body}<br>
            <span style="color:var(--ink-tertiary);">Rankings are using weights fit on a different universe or timeframe.
            Factors learned for one market do not generalise to another.
            Reset to defaults or run a new calibration for the current selection.</span>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("↑ Import Profile", expanded=False):
        uploaded = st.file_uploader(" ", type=["json"], label_visibility="collapsed", key="passport_uploader")
        if uploaded:
            try:
                payload = json.load(uploaded)
                # Accept the v2 full opt_results shape AND legacy weights-only dicts.
                if isinstance(payload, dict) and isinstance(payload.get("weights"), dict):
                    _set_active_weights(payload["weights"])
                    st.session_state["opt_results"] = payload
                    pe.save_profile(payload)
                    _imp_label = payload.get("selected_index") or payload.get("universe") or "—"
                    console.success(f"Profile imported · {_imp_label} · persisted to disk")
                else:
                    _set_active_weights(payload)  # legacy: file IS a weights dict
                    _had_calibration = "opt_results" in st.session_state
                    if _had_calibration:
                        del st.session_state["opt_results"]
                    pe.delete_profile()
                    console.success("Profile imported · legacy weights (no calibration metadata)")
                    if _had_calibration:
                        st.warning(
                            "Legacy weights-only file imported. Your previous calibrated profile "
                            "(train/val scores, sensitivity data) has been cleared. "
                            "Re-run Intelligence mode to produce a new calibrated profile."
                        )
                # Toast survives the rerun; success card alone would blink-and-disappear.
                # Note: Streamlit's icon= validates against an emoji whitelist — '✓' (U+2713)
                # is rejected as a "shortcode". Using '✅' (U+2705) which IS a valid emoji.
                st.toast("Profile imported.", icon="✅")
                st.success("Profile imported.")
            except Exception as e:
                st.error(f"Import failed: {e}")

    # Export the full opt_results when calibrated; raw weights otherwise.
    # Filename: sanket_profile_<universe_slug>_<timestamp>.json so the file
    # is self-describing — important when users keep multiple profiles.
    res = st.session_state.get("opt_results")
    export_payload = res or {"weights": _get_active_weights()}
    if res:
        # Always use the profile's own universe/index — never the sidebar selection.
        # If the profile pre-dates universe stamping (None), omit those parts rather
        # than silently substituting the sidebar value, which could mislabel the file.
        export_universe = res.get("universe")
        export_index    = res.get("selected_index")
        # Timestamp from opt_results may be 'YYYY-MM-DD HH:MM' — convert to date slug.
        ts = res.get("timestamp") or ""
        export_date = ts.split(" ")[0] if ts else None
    else:
        export_universe = current_universe
        export_index    = current_index
        export_date     = _today_ist()
    st.download_button(
        "↓ Export Profile",
        data=json.dumps(export_payload, indent=2, default=str),
        file_name=build_download_filename(
            "profile", universe=export_universe, selected_index=export_index,
            dates=export_date, ext="json",
        ),
        mime="application/json",
        width='stretch',
        key="passport_export",
    )
    if st.button("↺ Reset to Defaults", width='stretch', key="passport_reset"):
        _set_active_weights(pe.DEFAULT_W)
        if "opt_results" in st.session_state:
            del st.session_state["opt_results"]
        # Only delete THIS universe+timeframe profile — others are preserved.
        pe.delete_profile(current_universe, current_index, current_timeframe)
        _reset_label = (current_index or current_universe or "—")
        console.detail(f"Profile reset · {_reset_label} · disk profile cleared")
        st.rerun()


def render_intelligence_center():
    """Self-Tuning calibration dashboard. Two tabs: Calibration · Sensitivity.
    Sidebar carries the Model Passport (rendered by ``render_sidebar``)."""
    ui.render_section_header(
        "Intelligence Center",
        "Self-Training Priority Engine Calibration",
        icon="brain", accent="violet",
    )

    if not st.session_state.get("timeseries_done"):
        st.info("Run a Bulk Range analysis first to generate the training dataset.")
        return

    ts_data = st.session_state.get("ts_results_df")
    settings = st.session_state.get("_calib_settings",
                                    {"trials": 50, "train_frac": 0.70, "horizons": pe.HOLD_HORIZONS})
    s_trials  = int(settings.get("trials", 50))
    s_split   = float(settings.get("train_frac", 0.70))
    s_horizon = settings.get("horizons", pe.HOLD_HORIZONS)
    train_pct = int(round(s_split * 100))

    # ── Dataset summary row (mirrors bulk-range "Total Signals / Avg Oversold / …" rhythm) ──
    ts_dates   = int(ts_data['Date'].nunique())
    ts_symbols = int(ts_data['Symbol'].nunique())
    ts_rows    = int(len(ts_data))
    n_train_d  = max(1, int(ts_dates * s_split))
    n_val_d    = max(0, ts_dates - n_train_d)

    active_weights = _get_active_weights()
    is_default = (active_weights == pe.DEFAULT_W)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: ui.render_metric_card("Dates",        f"{ts_dates}",            "in window",     "info")
    with c2: ui.render_metric_card("Symbols",      f"{ts_symbols}",          "universe",      "info")
    with c3: ui.render_metric_card("Observations", f"{ts_rows:,}",           "factor rows",   "info")
    with c4: ui.render_metric_card("Train / Val",  f"{n_train_d} / {n_val_d}", f"{train_pct} / {100 - train_pct} split", "warning")
    with c5: ui.render_metric_card("Profile",      "Default" if is_default else "Calibrated",
                                                   "active engine state",
                                                   "neutral" if is_default else "success")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Tabs (same simple titles as bulk-range view) ──
    tab_calib, tab_sens = st.tabs(["Calibration", "Factor Sensitivity"])

    # ════ TAB 1 · Calibration ═════════════════════════════════════════════
    with tab_calib:
        col_t1, col_t2 = st.columns([2, 1])
        with col_t1:
            ui.render_section_header(
                "Optimization Engine",
                "Optuna TPE search · IR of daily Spearman IC across hold horizons",
                icon="zap", accent="violet",
            )
            # Settings summary line — settings are configured in the sidebar.
            st.markdown(f"""
            <div style="font-family:var(--data); font-size:0.72rem; color:var(--ink-tertiary);
                        display:flex; flex-wrap:wrap; gap:1.4rem; padding:0.8rem 0 1rem 0;">
                <span>Trials <b style="color:var(--ink-secondary);">{s_trials}</b></span>
                <span>Split <b style="color:var(--ink-secondary);">{train_pct} / {100 - train_pct}</b></span>
                <span>Horizons <b style="color:var(--ink-secondary);">{', '.join(map(str, s_horizon))}</b></span>
                <span style="font-size:0.65rem;">— change these in the sidebar</span>
            </div>
            """, unsafe_allow_html=True)

            if st.button("◈ START SELF-CALIBRATION", width='stretch', type="primary"):
                run_priority_optimization(ts_data, settings)

            # Latest calibration diagnostics — same metric-row rhythm as the rest of the app
            if "opt_results" in st.session_state:
                res = st.session_state["opt_results"]
                train_v = res.get('train_score', 0.0) or 0.0
                val_v   = res.get('val_score',   0.0) or 0.0
                stability = (val_v / train_v * 100) if train_v not in (0, None) else 0.0

                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                ui.render_section_header(
                    "Calibration Diagnostics",
                    f"Last run · {res.get('timestamp', '—')}",
                    icon="activity", accent="emerald",
                )
                _is_overfit = train_v > 0.05 and val_v < train_v * 0.3
                _is_low_ir  = val_v <= 0.0
                d_r1c1, d_r1c2 = st.columns(2)
                with d_r1c1:
                    ui.render_metric_card("Train IR", f"{train_v:+.4f}", "in-sample fit",
                                          "success" if train_v > 0 else "warning")
                with d_r1c2:
                    _val_sub = "out-of-sample · IC rank corr"
                    ui.render_metric_card("Validation IR", f"{val_v:+.4f}", _val_sub,
                                          "success" if val_v > 0.02 else ("warning" if val_v > 0 else "danger"))

                st.markdown('<div style="margin-bottom: 0.8rem;"></div>', unsafe_allow_html=True)

                d_r2c1, d_r2c2 = st.columns(2)
                with d_r2c1:
                    ui.render_metric_card("Stability", f"{stability:.0f}%", "Val / Train ratio",
                                          "info" if 30 < stability < 130 else "warning")
                with d_r2c2:
                    if _is_low_ir:
                        _risk_label, _risk_sub, _risk_kind = "No Edge", "Val IR ≤ 0 — reset or recalibrate", "danger"
                    elif _is_overfit:
                        _risk_label, _risk_sub, _risk_kind = "Overfit", "Train >> Val — reduce trials or extend range", "warning"
                    else:
                        _risk_label, _risk_sub, _risk_kind = "Quality OK", "No overfit or IR issues detected", "success"
                    ui.render_metric_card("Quality Check", _risk_label, _risk_sub, _risk_kind)

        with col_t2:
            ui.render_section_header(
                "Active Weights",
                "factor coefficients in basis points · long vs short",
                icon="grid", accent="amber",
            )
            st.components.v1.html(
                _build_active_weights_table_html(active_weights),
                height=720,
                scrolling=False,
            )

    # ════ TAB 2 · Factor Sensitivity ══════════════════════════════════════
    with tab_sens:
        if "opt_results" not in st.session_state:
            st.info("Run a calibration to populate the factor importance chart.")
        else:
            res = st.session_state["opt_results"]
            sensitivity = res.get('sensitivity', {}) or {}

            if not sensitivity:
                st.info("Importance scores are not yet available — try increasing the trial count.")
            else:
                ui.render_section_header(
                    "Factor Importance",
                    "Optuna fANOVA · share of objective variance attributable to each factor",
                    icon="bar-chart", accent="amber",
                )

                # NB: get_param_importance() already returns percentages summing to 100.
                sens_df = pd.DataFrame(sensitivity.items(), columns=['Factor', 'Importance'])
                sens_df = sens_df.sort_values('Importance', ascending=True)

                top_factor = sens_df.iloc[-1]['Factor']
                top_share  = float(sens_df.iloc[-1]['Importance'])

                sc1, sc2 = st.columns([2, 1])
                with sc1:
                    fig_sens = go.Figure(go.Bar(
                        x=sens_df['Importance'],
                        y=sens_df['Factor'],
                        orientation='h',
                        marker=dict(
                            color=sens_df['Importance'],
                            colorscale=[[0, '#1E293B'], [1, '#D4A853']],
                            line=dict(width=0),
                        ),
                        hovertemplate="<b>%{y}</b><br>Importance: %{x:.1f}%<extra></extra>",
                    ))
                    fig_sens.update_layout(
                        height=440, showlegend=False,
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis=dict(title="% of objective variance", gridcolor='rgba(255,255,255,0.05)', zeroline=False),
                        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    )
                    apply_chart_theme(fig_sens)
                    st.plotly_chart(fig_sens, width='stretch', key='chart_sensitivity')

                with sc2:
                    ui.render_interpretation_card(
                        "Dominant Factor",
                        f"{top_factor} drives {top_share:.1f}% of the objective's variance "
                        f"across the search. It is the primary lever for the current dataset.",
                        "amber",
                    )
                    st.markdown("<br>", unsafe_allow_html=True)
                    ui.render_interpretation_card(
                        "Reading the Chart",
                        "A single tall bar means the optimizer is highly sensitive to one factor — "
                        "trend-clear regime. A flat distribution means importance is diffuse — "
                        "high-entropy regime where no factor dominates.",
                        "neutral",
                    )


# ═══════════════════════════════════════════════════════════════════════════
# CALIBRATION RUNNER — phase-aware progress, throttled logging, abort guard
# ═══════════════════════════════════════════════════════════════════════════

# Phase boundaries on the unified progress bar (build → calibrate → validate → apply).
# Most visible movement is allocated to the calibration loop.
_CALIB_PHASES = [
    ("Building dataset",      6),    # phase 1 → 0–6 %
    ("Calibrating weights",  92),    # phase 2 → 6–92 %  (Optuna trials live here)
    ("Validating on holdout", 97),   # phase 3 → 92–97 %
    ("Applying weights",     100),   # phase 4 → 97–100 %
]


def run_priority_optimization(ts_data, calib_settings):
    """Phase-aware calibration runner.

    Pipeline: build precomputed dataset → Optuna TPE search → out-of-sample
    validation → activate best weights. Single themed progress card spans
    all four phases. Terminal logs phase transitions and quartile checkpoints
    only — no per-trial spam.

    Args:
        ts_data: time-series factor frame from run_timeseries_analysis.
        calib_settings: dict {trials, train_frac, horizons}.
    """
    n_trials   = int(calib_settings.get("trials", 50))
    train_frac = float(calib_settings.get("train_frac", 0.70))
    horizons   = list(calib_settings.get("horizons", pe.HOLD_HORIZONS))

    progress_slot = st.empty()
    start_time    = time.time()

    # ─── Phase 1 / 4 · Build dataset ──────────────────────────────────────
    progress_bar(progress_slot, 1, "Calibration Engine", "Phase 1 / 4 · Building dataset")
    console.section("QUANT CALIBRATION", phase="INTELLIGENCE")

    tuner = intel.PriorityTuner(
        ts_data,
        hold_periods=horizons,
        train_frac=train_frac,
    )
    n_train_dates = tuner._train_pre.n_groups if not tuner._train_pre.empty else 0
    n_val_dates   = tuner._val_pre.n_groups   if not tuner._val_pre.empty   else 0
    n_train_rows  = tuner._train_pre.n_rows   if not tuner._train_pre.empty else 0
    n_val_rows    = tuner._val_pre.n_rows     if not tuner._val_pre.empty   else 0

    console.item("Trials",           n_trials)
    console.item("Train dates",      f"{n_train_dates}  ({n_train_rows} rows)")
    console.item("Validation dates", f"{n_val_dates}  ({n_val_rows} rows)")
    console.item("Hold horizons",    str(horizons))
    console.item("Train / val split", f"{int(train_frac * 100)} / {100 - int(train_frac * 100)}")

    # Hard guard: not enough usable training data
    MIN_TRAIN_DATES = 10
    if n_train_dates < MIN_TRAIN_DATES:
        progress_slot.empty()
        st.error(
            f"**Not enough training data** — only {n_train_dates} usable date(s) after "
            f"dropping the trailing {max(horizons)}-bar boundary (forward returns NaN there). "
            f"Need ≥ {MIN_TRAIN_DATES}. Increase the historical range to ~60+ trading days "
            f"for meaningful calibration."
        )
        console.warning(
            f"Calibration aborted — only {n_train_dates} usable training dates "
            f"(need ≥ {MIN_TRAIN_DATES})."
        )
        return

    # Soft warning: sparse cross-section makes IC-based ranking unreliable.
    # IC measures rank correlation across symbols per date; with fewer than ~20
    # symbols per date the rank has too few positions to produce stable IC estimates.
    MIN_SYMBOLS_FOR_IC = 20
    avg_symbols_per_date = n_train_rows / max(n_train_dates, 1)
    if avg_symbols_per_date < MIN_SYMBOLS_FOR_IC:
        st.warning(
            f"**Small universe detected** — {avg_symbols_per_date:.0f} symbols/date on average "
            f"(recommended ≥ {MIN_SYMBOLS_FOR_IC}). IC-based calibration can produce noisy, "
            f"overfit weights with sparse cross-sections. Proceed, but treat the calibrated "
            f"profile as experimental and validate out-of-sample carefully."
        )
        console.warning(
            f"Small universe: ~{avg_symbols_per_date:.0f} symbols/date — "
            f"IC estimates may be noisy (recommend ≥ {MIN_SYMBOLS_FOR_IC})."
        )

    progress_bar(progress_slot, _CALIB_PHASES[0][1], "Calibration Engine", "Phase 1 / 4 · Dataset built")
    console.detail("[1/4] Dataset built ✓")

    # ─── Phase 2 / 4 · Calibrate weights ──────────────────────────────────
    progress_bar(
        progress_slot, _CALIB_PHASES[0][1],
        "Calibration Engine",
        f"Phase 2 / 4 · Calibrating weights · trial 0 / {n_trials}",
    )
    console.detail(f"[2/4] Calibrating · {n_trials} trials over {n_train_dates} dates")

    quartile_marks = {max(1, n_trials // 4), max(1, n_trials // 2),
                      max(1, 3 * n_trials // 4), n_trials}
    best_so_far = -float("inf")
    p_opt_start = _CALIB_PHASES[0][1]
    p_opt_end   = _CALIB_PHASES[1][1]

    def on_trial(trial_num, total, score):
        nonlocal best_so_far
        done = trial_num + 1
        best_so_far = max(best_so_far, score)
        elapsed = time.time() - start_time
        remaining = (elapsed / done) * (total - done)
        eta_str = time.strftime("%M:%S", time.gmtime(remaining))

        pct = int(p_opt_start + (done / total) * (p_opt_end - p_opt_start))
        progress_bar(
            progress_slot, pct,
            "Calibration Engine",
            f"Phase 2 / 4 · trial {done} / {total} · best IR {best_so_far:+.3f} · ETA {eta_str}",
        )

        # Terminal: quartile checkpoints only.
        if done in quartile_marks:
            console.detail(f"      · {done:>3} / {total} trials · best IR {best_so_far:+.3f}")

    best_w, train_score = tuner.optimize(n_trials=n_trials, progress_callback=on_trial)

    # ─── Phase 3 / 4 · Validate on holdout ────────────────────────────────
    progress_bar(
        progress_slot, _CALIB_PHASES[2][1],
        "Calibration Engine",
        f"Phase 3 / 4 · validating on {n_val_dates} held-out date(s)",
    )
    console.detail("[3/4] Validating on holdout dates")
    val_score = tuner.evaluate_validation()

    # ─── Phase 4 / 4 · Apply weights ──────────────────────────────────────
    importance = tuner.get_param_importance()
    top_factor = max(importance, key=importance.get) if importance else "—"
    top_share  = importance.get(top_factor, 0.0) if importance else 0.0

    progress_bar(
        progress_slot, 99,
        "Calibration Engine",
        f"Phase 4 / 4 · activating weights · top factor {top_factor}",
    )
    console.detail("[4/4] Activating weights · storing profile")

    _set_active_weights(best_w)
    ts_meta = st.session_state.get("ts_meta") or {}
    opt_results = {
        "weights":        best_w,
        "train_score":    train_score,
        "val_score":      val_score,
        "sensitivity":    importance,
        "timestamp":      datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "universe":       ts_meta.get("universe"),
        "selected_index": ts_meta.get("selected_index"),
        "timeframe":      ts_meta.get("timeframe"),
    }
    st.session_state["opt_results"] = opt_results
    _persist_ok = pe.save_profile(opt_results)  # best-effort disk persistence
    _persist_label = ts_meta.get("selected_index") or ts_meta.get("universe") or "—"
    if _persist_ok:
        console.detail(f"Profile persisted to disk · key='{_persist_label}'")
    else:
        console.warning(f"Profile persist failed (disk write error) · key='{_persist_label}'")

    # ─── Final summary ────────────────────────────────────────────────────
    duration_str = time.strftime("%M:%S", time.gmtime(time.time() - start_time))
    # Three distinct quality states — overfit and low-IR are separate failure modes.
    overfit  = train_score > 0.05 and val_score < train_score * 0.3
    low_ir   = val_score <= 0.0   # negative or zero validation IC — profile adds no edge
    cal_risk = overfit or low_ir

    if low_ir:
        cal_quality = "negative val IR — profile has no demonstrated edge"
    elif overfit:
        cal_quality = "overfit — train IR significantly exceeds validation IR"
    else:
        cal_quality = "none"

    progress_bar(progress_slot, 100, "Calibration Engine",
                 f"Complete · Train {train_score:+.3f} · Val {val_score:+.3f} · {duration_str}")

    _cal_label = ts_meta.get("selected_index") or ts_meta.get("universe") or "—"
    console.summary("CALIBRATION COMPLETE", {
        "Universe":      _cal_label,
        "Train IR":      f"{train_score:+.4f}",
        "Validation IR": f"{val_score:+.4f}",
        "Top factor":    f"{top_factor} ({top_share:.1f}%)",
        "Trials":        n_trials,
        "Duration":      duration_str,
        "Quality risk":  cal_quality,
        "Profile saved": "yes" if _persist_ok else "no (disk error)",
    })

    overfit_suffix = f" ⚠ {cal_quality}" if cal_risk else ""
    st.toast(
        f"Calibration complete · Train {train_score:+.3f} · Val {val_score:+.3f}{overfit_suffix}",
        icon="🎯" if not cal_risk else "⚠️",
    )

    st.rerun()

if __name__ == "__main__":
    main()
