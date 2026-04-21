"""
Sanket - Market Signal Screener | A Pragyam Product Family Member
WRCI Engine (Wave-Regime Composite Index) Quantitative Signal Scanner
"""

import html
import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import numpy as np
import plotly.graph_objects as go
import requests
import io
import urllib3
import warnings
import logging
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

VERSION = "v1.0.0"

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

if "results_df" not in st.session_state:
    st.session_state["results_df"] = None
if "run_screener_flag" not in st.session_state:
    st.session_state["run_screener_flag"] = False
if "run_timeseries_flag" not in st.session_state:
    st.session_state["run_timeseries_flag"] = False
if "timeseries_done" not in st.session_state:
    st.session_state["timeseries_done"] = False
if "run_error" not in st.session_state:
    st.session_state["run_error"] = None

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

UNIVERSE_OPTIONS = ["India Indexes", "US Indexes", "ETF Index", "Commodities", "Currency", "Crypto"]
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

# Asset Name Lookup for friendly display (Reverse map tickers to names)
ASSET_NAME_LOOKUP = {v: k for k, v in {**COMMODITY_MAP, **CURRENCY_MAP, **CRYPTO_MAP}.items()}

# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
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
    except Exception:
        pass

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
    except Exception:
        pass

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
    except Exception:
        pass

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
    except Exception:
        pass

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
        except Exception:
            pass

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


def get_us_index_symbols(index_name):
    """Get symbols for US index from Yahoo Finance."""
    index_map = {
        "S&P 500": ["^GSPC"] + [f"^GSPC-{i}" for i in range(1, 100)],
        "DOW JONES": ["^DJI"],
        "NASDAQ 100": ["^NDX"]
    }
    if index_name in index_map:
        # For US indices, return major constituent tickers via yfinance
        return index_map[index_name], f"✓ Fetched {index_name}"
    return None, f"Unknown US index: {index_name}"


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
        end_date = datetime.date.today()
    
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
            
        if isinstance(all_data, pd.DataFrame) and isinstance(all_data.columns, pd.MultiIndex):
            data_dict = {}
            for ticker in stock_list:
                try:
                    ticker_df = all_data.xs(ticker, level=0, axis=1)
                    if not ticker_df.empty and not ticker_df['Close'].isnull().all():
                        data_dict[ticker] = ticker_df.copy()
                except KeyError:
                    pass
        elif isinstance(all_data, dict):
            data_dict = {t:df.copy() for t,df in all_data.items() if not df.empty and not df['Close'].isnull().all()}
        else:
             return None, "Unexpected data structure"

        if include_live and end_date == datetime.date.today() and data_dict:
            sample_df = list(data_dict.values())[0]
            sample_df.index = pd.to_datetime(sample_df.index)
            if sample_df.index.tz is not None:
                 sample_df.index = sample_df.index.tz_localize(None)
            
            has_today = any(idx.date() == datetime.date.today() for idx in sample_df.index)
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
                                    if hist_df.index.tz is not None: hist_df.index = hist_df.index.tz_localize(None)
                                    live_ticker.index = pd.to_datetime(live_ticker.index)
                                    if live_ticker.index.tz is not None: live_ticker.index = live_ticker.index.tz_localize(None)
                                    new_dates = live_ticker.index.difference(hist_df.index)
                                    if len(new_dates) > 0:
                                        data_dict[ticker] = pd.concat([hist_df, live_ticker.loc[new_dates]]).sort_index()
                            except KeyError: pass
                except Exception: pass
        return data_dict, f"✓ Downloaded {len(data_dict)} tickers"
    except Exception as e:
        return None, f"Download error: {e}"


def resample_to_weekly(df):
    if df is None or df.empty:
        return df
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    weekly = df.resample('W-FRI').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    return weekly

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


def calculate_trend_count(series, length):
    trend = pd.Series(0.0, index=series.index)
    for i in range(1, length + 1):
        trend += np.where(series > series.shift(i), 1, -1)
    return trend


def run_full_analysis(df, reg_len=20, wt_n1=10, wt_n2=21, obLevel1=80, obLevel2=40, osLevel1=-80, osLevel2=-40):
    hlc3 = (df['High'] + df['Low'] + df['Close']) / 3.0
    hma_p = calculate_hma(hlc3, 15)
    hma_v = calculate_hma(df['Volume'], 15)

    trend = calculate_trend_count(hma_p, reg_len)
    voltrend = calculate_trend_count(hma_v, reg_len)

    coeff = 10.0 / reg_len
    norm_trend = (trend * coeff) * 10.0

    ap = hlc3
    esa = ap.ewm(span=wt_n1, adjust=False).mean()
    d = (ap - esa).abs().ewm(span=wt_n1, adjust=False).mean()
    ci = (ap - esa) / (0.015 * d).replace(0, np.nan)
    wt1 = ci.ewm(span=wt_n2, adjust=False).mean()

    composite_line = (wt1 + norm_trend) / 2.0
    composite_signal = composite_line.rolling(window=4).mean()

    df['Unified_Osc'] = composite_line
    df['Signal_Line'] = composite_signal
    df['WT1'] = wt1
    df['Norm_Trend'] = norm_trend
    
    df['long_cond'] = (composite_line > composite_signal) & (composite_line.shift(1) <= composite_signal.shift(1))
    df['short_cond'] = (composite_line < composite_signal) & (composite_line.shift(1) >= composite_signal.shift(1))

    df['Condition'] = np.select(
        [composite_line > obLevel1, composite_line > obLevel2, composite_line < osLevel1, composite_line < osLevel2],
        ['OB Extreme', 'OB', 'OS Extreme', 'OS'],
        default='Neutral'
    )

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
                SIGNAL ENGINE
            </h3>
            <p>Wave Trend Composite Index (WRCI) identifies momentum signals and trend strength across your universe with daily updates.</p>
            <div class='spec'>
                <span>Detection:</span> Wave Trend signals (bullish/bearish)<br>
                <span>Scoring:</span> Signal magnitude + trend direction<br>
                <span>Output:</span> Signal strength, zone, trend value<br>
                <span>Refresh:</span> Daily updates
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='system-card regime'>
            <h3>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>
                SIGNAL TYPES
            </h3>
            <p>Rank momentum signals by strength, identify overbought/oversold zones, and track trend direction for each symbol.</p>
            <div class='spec'>
                <span>Long:</span> Bullish signal (positive Wave Trend)<br>
                <span>Short:</span> Bearish signal (negative Wave Trend)<br>
                <span>OB/OS:</span> Overbought/Oversold zones<br>
                <span>Trend:</span> Direction + strength value
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='system-card strategies'>
            <h3>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
                UNIVERSE COVERAGE
            </h3>
            <p>Scan F&O stocks or entire index constituents. Filter by timeframe. Customize sensitivity thresholds.</p>
            <div class='spec'>
                <span>Universes:</span> F&O + Indices + ETFs<br>
                <span>Timeframes:</span> Daily · Weekly<br>
                <span>Symbols:</span> Up to 500<br>
                <span>Modes:</span> Point + Time Series
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
        <p>Configure via the <strong>Sidebar</strong>: select <strong>Universe</strong>, <strong>Timeframe</strong>, <strong>Temporal Range</strong>, and <strong>Engine Settings</strong>.<br>
           Click <strong>RUN SCREENER</strong> to analyze and discover today's signals.<br>
           <span style="color:var(--ink-secondary); font-size:0.85em; margin-top:0.5rem; display:inline-block;">System will compute Wave Trend oscillations · Calculate signal magnitude · Rank by strength</span></p>
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


def render_signal_detail_card(symbol, price, signal_val, trend_val, zone, signal_type, rsi_val, osc_val, zscore_val, ma_count):
    """Render detailed signal card with strength indicator and technical confirmations.

    Displays signal magnitude, trend direction, zone status, and technical confirmations
    (RSI levels, oscillator state) to provide comprehensive signal context.

    Returns: Renders to Streamlit; no return value.
    """
    signal_strength = get_signal_strength_score({'Signal': signal_val})

    # Determine signal quality
    if signal_strength >= 65:
        icon = SVGS["DOT"].replace('currentColor', 'var(--emerald)')
        label = "Strong"
    elif signal_strength >= 50:
        icon = SVGS["DOT"].replace('currentColor', 'var(--info)')
        label = "Moderate"
    elif signal_strength >= 35:
        icon = SVGS["DOT"].replace('currentColor', 'var(--amber)')
        label = "Weak"
    else:
        icon = SVGS["DOT"].replace('currentColor', 'var(--rose)')
        label = "Very Weak"

    # Technical confirmation indicators
    confirmations = []
    if pd.notna(rsi_val):
        if rsi_val > 70:
            confirmations.append(("RSI Overbought", SVGS["UP"].replace('currentColor', 'var(--rose)'), "var(--rose)"))
        elif rsi_val < 30:
            confirmations.append(("RSI Oversold", SVGS["DOWN"].replace('currentColor', 'var(--emerald)'), "var(--emerald)"))
        else:
            confirmations.append(("RSI Neutral", "—", "var(--amber)"))

    trend_label = "Strong" if abs(trend_val) > 30 else "Moderate" if abs(trend_val) > 15 else "Weak"
    trend_icon = SVGS["UP"].replace('currentColor', 'var(--emerald)') if trend_val > 0 else SVGS["DOWN"].replace('currentColor', 'var(--rose)')

    st.markdown(f"""
    <div style="background: linear-gradient(145deg, var(--glass) 0%, rgba(17, 24, 39, 0.4) 100%);
                border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem; margin-bottom: 0.75rem;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;">
            <div>
                <div style="font-family: var(--display); font-size: 1rem; font-weight: 700; color: var(--ink-primary);">
                    {symbol.replace('.NS', '')}
                </div>
                <div style="font-family: var(--data); font-size: 0.8rem; color: var(--ink-secondary);">
                    ₹{price:,.2f}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-family: var(--data); font-size: 1.25rem; font-weight: 700; color: var(--amber);">
                    {signal_strength:.0f}%
                </div>
                <div style="font-family: var(--data); font-size: 0.7rem; color: var(--ink-secondary); text-transform: uppercase; letter-spacing: 0.05em;">
                    Strength
                </div>
            </div>
        </div>

        <div style="background: rgba(255,255,255,0.02); border-radius: 6px; padding: 0.75rem; margin-bottom: 0.75rem;">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; font-family: var(--data); font-size: 0.75rem;">
                <div>
                    <span style="color: var(--ink-tertiary); text-transform: uppercase; font-size: 0.65rem;">Signal Type</span><br>
                    <span style="color: var(--ink-primary); font-weight: 600;">{icon} {signal_type}</span>
                </div>
                <div>
                    <span style="color: var(--ink-tertiary); text-transform: uppercase; font-size: 0.65rem;">Trend Strength</span><br>
                    <span style="color: var(--ink-primary); font-weight: 600;">{trend_icon} {trend_label}</span>
                </div>
                <div>
                    <span style="color: var(--ink-tertiary); text-transform: uppercase; font-size: 0.65rem;">Zone</span><br>
                    <span style="color: var(--ink-primary); font-weight: 600;">{zone}</span>
                </div>
                <div>
                    <span style="color: var(--ink-tertiary); text-transform: uppercase; font-size: 0.65rem;">MA Alignment</span><br>
                    <span style="color: var(--ink-primary); font-weight: 600;">{int(ma_count) if pd.notna(ma_count) else 0}/5</span>
                </div>
            </div>
        </div>

        <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; font-family: var(--data); font-size: 0.7rem;">
            <div style="padding: 0.35rem 0.75rem; background: rgba(212,168,83,0.1); border-radius: 4px; border: 1px solid rgba(212,168,83,0.2); color: var(--amber);">
                ◈ Signal: {signal_val:+.2f}
            </div>
            <div style="padding: 0.35rem 0.75rem; background: rgba(45,212,168,0.1); border-radius: 4px; border: 1px solid rgba(45,212,168,0.2); color: var(--emerald);">
                ≈ Wave: {osc_val:+.2f}
            </div>
            {f'<div style="padding: 0.35rem 0.75rem; background: rgba(232,85,90,0.1); border-radius: 4px; border: 1px solid rgba(232,85,90,0.2); color: var(--rose);">RSI: {rsi_val:.0f}</div>' if pd.notna(rsi_val) else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)

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
        timeframe = st.radio("Timeframe", TIMEFRAME_OPTIONS, horizontal=True, label_visibility="collapsed")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Universe Selection
        st.markdown('<div class="sidebar-title">Universe Selection</div>', unsafe_allow_html=True)
        universe = st.selectbox("Universe", UNIVERSE_OPTIONS, label_visibility="collapsed")
        selected_index = None

        if universe == "India Indexes":
            selected_index = st.selectbox("Index", INDEX_LIST, index=INDEX_LIST.index("Benchmark Indexes"), label_visibility="collapsed")
        elif universe == "US Indexes":
            selected_index = st.selectbox("Index", US_INDEX_LIST, index=US_INDEX_LIST.index("DOW JONES"), label_visibility="collapsed")
        elif universe == "ETF Index":
            selected_index = "NSE ETF Universe"
        elif universe == "Commodities":
            selected_index = "Global Commodities"
        elif universe == "Currency":
            selected_index = "Major FX Pairs"
        elif universe == "Crypto":
            selected_index = "Digital Assets (Top 20)"

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Temporal Range Section
        st.markdown('<div class="sidebar-title">Temporal Range</div>', unsafe_allow_html=True)
        analysis_mode = st.radio("Mode", ["Snapshot", "Range Study"], horizontal=True, label_visibility="collapsed")

        if analysis_mode == "Snapshot":
            analysis_date = st.date_input("Date", datetime.date.today(), max_value=datetime.date.today(), label_visibility="collapsed")
            start_date_hist, end_date_hist = None, None
        else:
            analysis_date = datetime.date.today()
            col_date1, col_date2 = st.columns(2)
            with col_date1: start_date_hist = st.date_input("Start", datetime.date.today() - datetime.timedelta(days=30), label_visibility="collapsed")
            with col_date2: end_date_hist = st.date_input("End", datetime.date.today(), label_visibility="collapsed")

        # WRCI Engine — hardcoded defaults
        reg_len, wt_n1, wt_n2 = 20, 10, 21
        obLevel1, obLevel2, osLevel1, osLevel2 = 80, 40, -80, -40

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Run Button
        run_clicked = st.button("◈ RUN SCREENER", type="primary", width='stretch', use_container_width=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # System Spec Card
        try:
            if universe == "India Indexes" and selected_index:
                symbols_count = len(get_index_stock_list(selected_index)[0] or [])
                universe_display = selected_index
            elif universe == "US Indexes" and selected_index:
                symbols_count = len(get_us_index_symbols(selected_index)[0] or [])
                universe_display = selected_index
            elif universe == "Commodities" and selected_index:
                symbols_count = len(get_commodity_symbols(selected_index)[0] or [])
                universe_display = selected_index
            elif universe == "Currency" and selected_index:
                symbols_count = len(get_currency_symbols(selected_index)[0] or [])
                universe_display = selected_index
            elif universe == "ETF Index":
                symbols_count = len(ETF_LIST)
                universe_display = "NSE ETFs"
            else:
                symbols_count = "—"
                universe_display = universe
        except:
            symbols_count = "—"
            universe_display = universe

        st.markdown(f"""
        <div class="system-spec">
            <div class="spec-row"><span class="spec-label">Version</span><span class="spec-value">{VERSION}</span></div>
            <div class="spec-row"><span class="spec-label">Universe</span><span class="spec-value" style="font-size:0.7rem;">{universe_display}</span></div>
            <div class="spec-row"><span class="spec-label">Timeframe</span><span class="spec-value">{timeframe}</span></div>
            <div class="spec-row"><span class="spec-label">Mode</span><span class="spec-value" style="font-size:0.7rem;">{analysis_mode}</span></div>
        </div>
        """, unsafe_allow_html=True)

        return universe, selected_index, analysis_date, reg_len, wt_n1, wt_n2, (obLevel1, obLevel2, osLevel1, osLevel2), timeframe, analysis_mode, start_date_hist, end_date_hist, run_clicked


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SCREENER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_screener_analysis(universe, selected_index, analysis_date, reg_len, wt_n1, wt_n2, levels, timeframe):
    """Execute WRCI momentum analysis on universe symbols and return ranked signals.

    Fetches market data for universe, computes Wave Trend oscillations, calculates
    signal magnitude and trend values, detects overbought/oversold zones.

    Returns: DataFrame with signals ranked by magnitude, or None on error.
    """
    obLevel1, obLevel2, osLevel1, osLevel2 = levels
    progress_slot = st.empty()

    progress_bar(progress_slot, 5, "Initializing WRCI engine", f"Universe: {universe}")
    
    console.start_phase("DATA ACQUISITION", 1, 2)
    console.section("Universe Configuration")
    console.item("Universe", universe)
    console.item("Selected Index", selected_index)
    console.item("Timeframe", timeframe)

    if universe == "India Indexes":
        stock_list, msg = get_index_stock_list(selected_index)
    elif universe == "US Indexes":
        stock_list, msg = get_us_index_symbols(selected_index)
    elif universe == "Commodities":
        stock_list, msg = get_commodity_symbols(None)  # Runs all commodities
    elif universe == "Currency":
        stock_list, msg = get_currency_symbols(None)   # Runs all pairs
    elif universe == "Crypto":
        stock_list, msg = get_crypto_symbols(None)     # Runs all crypto
    elif universe == "ETF Index":
        stock_list, msg = get_etf_symbols()
    else:
        stock_list, msg = None, f"Unknown universe: {universe}"

    if not stock_list:
        console.error(msg)
        st.error(msg)
        return None

    console.success(f"Fetched {len(stock_list)} symbols for {selected_index}")
    console.section("Market Data Fetch")
    progress_bar(progress_slot, 15, "Fetching market data", f"{len(stock_list)} stocks")
    data_dict, fetch_msg = fetch_batch_data(stock_list, end_date=analysis_date)

    if not data_dict:
        console.error(fetch_msg)
        st.error(fetch_msg)
        return None

    console.success(f"Successfully downloaded data for {len(data_dict)} stocks")
    console.end_phase("DATA ACQUISITION")

    console.start_phase("WRCI MOMENTUM ANALYSIS", 2, 2)
    console.section("Technical Diagnostics")
    progress_bar(progress_slot, 20, "Analyzing WRCI momentum", f"{len(data_dict)} stocks")
    results = []

    for i, (ticker, df) in enumerate(data_dict.items()):
        try:
            pct = int(20 + (i + 1) / len(data_dict) * 75)
            progress_bar(progress_slot, pct, f"Analyzing signals", f"{i + 1}/{len(data_dict)} stocks")

            if timeframe == "Weekly":
                df = resample_to_weekly(df)

            if len(df) < reg_len + 30:
                console.detail(f"{ticker}: Skipped (Insufficient data: {len(df)} rows)")
                continue

            df = run_full_analysis(df, reg_len, wt_n1, wt_n2, obLevel1, obLevel2, osLevel1, osLevel2)

            # Sample at analysis_date or last available
            df.index = pd.to_datetime(df.index)
            target_dt = pd.to_datetime(analysis_date)

            if target_dt in df.index:
                idx_pos = df.index.get_loc(target_dt)
            else:
                idx_pos = len(df) - 1

            if idx_pos < 5:
                continue

            # Get historical signals for tracking (Today, 1d, 2d, 3d, Within 5d)
            sample_range = df.iloc[max(0, idx_pos - 5) : idx_pos + 1]

            last_row = df.iloc[idx_pos]

            # Build Signal String
            signal_type = "Neutral"
            if last_row['long_cond']:
                signal_type = "Long Cross"
            elif last_row['short_cond']:
                signal_type = "Short Cross"
            elif last_row['Condition'] != 'Neutral':
                signal_type = last_row['Condition']

            # Clean display names
            simple_name = ticker.replace(".NS", "").lstrip("^")
            friendly_name = ASSET_NAME_LOOKUP.get(ticker)
            if friendly_name:
                display_name = f"{ticker} ({friendly_name})"
            else:
                display_name = simple_name

            results.append({
                "Symbol": ticker,
                "DisplayName": display_name,
                "SimpleName": simple_name,
                "Signal": round(last_row['Unified_Osc'], 2),
                "Trend": round(last_row['Norm_Trend'], 2),
                "Wave": round(last_row['WT1'], 2),
                "Zone": last_row['Condition'],
                "SignalType": signal_type,
                "Price": round(last_row['Close'], 2),
                # Historical Long Signals
                "L_Today": "●" if sample_range.iloc[-1]['long_cond'] else "—",
                "L_1d": "●" if sample_range.iloc[-2]['long_cond'] else "—",
                "L_2d": "●" if sample_range.iloc[-3]['long_cond'] else "—",
                "L_3d": "●" if sample_range.iloc[-4]['long_cond'] else "—",
                "L_5d": "●" if sample_range.iloc[: idx_pos + 1].tail(5)['long_cond'].any() else "—",
                # Historical Short Signals
                "S_Today": "●" if sample_range.iloc[-1]['short_cond'] else "—",
                "S_1d": "●" if sample_range.iloc[-2]['short_cond'] else "—",
                "S_2d": "●" if sample_range.iloc[-3]['short_cond'] else "—",
                "S_3d": "●" if sample_range.iloc[-4]['short_cond'] else "—",
                "S_5d": "●" if sample_range.iloc[: idx_pos + 1].tail(5)['short_cond'].any() else "—",
                # Additional fields for detail cards
                "Osc_Value": round(last_row.get('Unified_Osc', 0), 2),
                "MA_Alignment": 5,  # Placeholder
                "ZScore_Value": 0,  # Placeholder
            })
            
            console.detail(f"[{i+1}/{len(data_dict)}] {ticker}: Signal={last_row['Unified_Osc']:+.2f} Zone={last_row['Condition']} Status={signal_type}")
            
        except Exception as e:
            console.failure(f"Analysis Failed: {ticker}", str(e))
            continue

    console.end_phase("WRCI MOMENTUM ANALYSIS")
    
    console.summary("RUN SUMMARY", {
        "Universe": universe,
        "Universe Index": selected_index,
        "Total Symbols": len(stock_list),
        "Data Success": len(data_dict),
        "Analyzed Stocks": len(results),
        "Analysis Date": analysis_date,
        "Status": "COMPLETE"
    })
    console.line('═', 70)
    
    progress_bar(progress_slot, 100, "Analysis complete", f"{len(results)} stocks analyzed")
    progress_slot.empty()

    if not results:
        st.warning("No stocks met the analysis criteria.")
        # Return empty DataFrame with expected columns to prevent downstream KeyErrors
        expected_cols = ["Symbol", "DisplayName", "SimpleName", "Signal", "Trend", "Wave", "Zone", "SignalType", "Price", "L_Today", "L_1d", "L_2d", "L_3d", "L_5d", "S_Today", "S_1d", "S_2d", "S_3d", "S_5d", "Osc_Value", "MA_Alignment", "ZScore_Value"]
        return pd.DataFrame(columns=expected_cols)

    results_df = pd.DataFrame(results)
    return results_df


def run_timeseries_analysis(universe, selected_index, start_date, end_date, reg_len, wt_n1, wt_n2, levels, timeframe):
    """Execute WRCI analysis across historical date range for signal evolution tracking.

    Differs from run_screener_analysis: processes 500+ days of history to detect
    signal emergence, persistence, and fade patterns over time for timeline visualization.

    Returns: Dict with per-date results for historical signal tracking.
    """
    progress_slot = st.empty()
    progress_bar(progress_slot, 5, "Fetching historical depth", f"Date range: {start_date} to {end_date}")

    console.start_phase("HISTORICAL ACQUISITION", 1, 2)
    console.section("Range Configuration")
    console.item("Universe", universe)
    console.item("Selected Index", selected_index)
    console.item("Start Date", start_date)
    console.item("End Date", end_date)
    console.item("Timeframe", timeframe)

    if universe == "India Indexes":
        stock_list, _ = get_index_stock_list(selected_index)
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
    else:
        stock_list = None

    if not stock_list:
        console.error("Failed to retrieve stock list")
        st.error("Failed to retrieve stock list")
        return

    console.success(f"Fetched {len(stock_list)} symbols for {selected_index}")
    console.section("Mass Historical Download")
    data_dict, msg = fetch_batch_data(stock_list, end_date=end_date, days_back=500)

    if not data_dict:
        console.error("No historical data available")
        st.error("No historical data available for selected range.")
        return

    console.success(f"Downloaded depth for {len(data_dict)} entities")
    console.end_phase("HISTORICAL ACQUISITION")

    progress_bar(progress_slot, 15, "Processing WRCI oscillations", f"{len(data_dict)} stocks")
    all_results = []

    for i, (ticker, df) in enumerate(data_dict.items()):
        try:
            pct = int(15 + (i + 1) / len(data_dict) * 70)
            progress_bar(progress_slot, pct, f"Analyzing signals", f"{i + 1}/{len(data_dict)} stocks")
            if timeframe == "Weekly":
                df = resample_to_weekly(df)
            df = run_full_analysis(df, reg_len, wt_n1, wt_n2, *levels)

            # Filter for requested range
            mask = (df.index.date >= start_date) & (df.index.date <= end_date)
            range_df = df.loc[mask]

            for date, row in range_df.iterrows():
                all_results.append({
                    'Date': date,
                    'Symbol': ticker,
                    'Signal': row['Unified_Osc'],
                    'Trend': row['Norm_Trend'],
                    'Wave': row['WT1'],
                    'Zone': row['Condition'],
                    'LongSignal': row['long_cond'],
                    'ShortSignal': row['short_cond']
                })
            
            console.detail(f"[{i+1}/{len(data_dict)}] {ticker}: {len(range_df)} data points processed")
            
        except Exception as e:
            console.failure(f"Range Analysis Failed: {ticker}", str(e))
            continue

    console.end_phase("WRCI RANGE ANALYSIS")

    progress_slot.empty()
    if not all_results:
        st.error("No results generated for the selected timeframe.")
        return

    ts_df = pd.DataFrame(all_results)
    daily_agg = ts_df.groupby('Date').agg({
        'Signal': 'mean',
        'Trend': 'mean',
        'Wave': 'mean',
        'LongSignal': 'sum',
        'ShortSignal': 'sum',
        'Zone': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Neutral'
    })

    # Compute additional metrics
    daily_agg['TotalSignals'] = daily_agg['LongSignal'] + daily_agg['ShortSignal']
    daily_agg['L_S_Ratio'] = daily_agg['LongSignal'] / (daily_agg['ShortSignal'] + 0.01)
    daily_agg['Conviction'] = daily_agg['Signal'].abs()

    # Summary metrics
    total_signals = daily_agg['TotalSignals'].sum()
    avg_signal = daily_agg['Signal'].mean()
    overall_ratio = daily_agg['LongSignal'].sum() / max(daily_agg['ShortSignal'].sum(), 1)
    most_common_zone = ts_df['Zone'].mode()[0] if len(ts_df['Zone'].mode()) > 0 else 'Neutral'

    console.summary("RANGE STUDY SUMMARY", {
        "Universe": universe,
        "Universe Index": selected_index,
        "Range Study": f"{start_date} to {end_date}",
        "Total Signals Generated": int(total_signals),
        "Avg Signal Strength": round(avg_signal, 2),
        "Bias Ratio (L/S)": round(overall_ratio, 2),
        "Dominant Regime": most_common_zone,
        "Status": "COMPLETE"
    })
    console.line('═', 70)

    progress_bar(progress_slot, 100, "Range study complete", f"{int(total_signals)} signals analyzed")
    progress_slot.empty()
    st.session_state["timeseries_done"] = True

    ui.render_section_header(f"Range Study ({start_date} to {end_date})", icon="history", accent="violet")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        ui.render_metric_card("Total Signals", str(int(total_signals)), f"{int(daily_agg['LongSignal'].sum())} long · {int(daily_agg['ShortSignal'].sum())} short", "info")
    with col_m2:
        ui.render_metric_card("Avg Signal Strength", f"{avg_signal:+.1f}", "Across all days", "neutral")
    with col_m3:
        ui.render_metric_card("L/S Ratio", f"{overall_ratio:.2f}", f"{'Bullish' if overall_ratio > 1 else 'Bearish'} bias", "info")
    with col_m4:
        ui.render_metric_card("Dominant Zone", most_common_zone, "Most common regime", "warning")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Signal Frequency over time (line chart)
    st.markdown(f'<h4 style="font-family: var(--data); font-size: 0.9rem; color: var(--ink-secondary); text-transform: uppercase; margin-bottom: 1rem; letter-spacing: 0.08em; display: flex; align-items: center; gap: 0.5rem;">{SVGS["CHART"]} Signal Frequency</h4>', unsafe_allow_html=True)
    fig_freq = go.Figure()
    fig_freq.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['LongSignal'], fill='tozeroy', name='Long Signals', line=dict(color='#2DD4A8', width=2), fillcolor='rgba(45, 212, 168, 0.2)'))
    fig_freq.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['ShortSignal'], fill='tozeroy', name='Short Signals', line=dict(color='#E8555A', width=2), fillcolor='rgba(232, 85, 90, 0.2)'))
    fig_freq.update_layout(title='', height=300, hovermode='x unified')
    apply_chart_theme(fig_freq)
    st.plotly_chart(fig_freq, width='stretch')

    # Signal Strength Trend (line chart)
    st.markdown(f'<h4 style="font-family: var(--data); font-size: 0.9rem; color: var(--ink-secondary); text-transform: uppercase; margin-bottom: 1rem; letter-spacing: 0.08em; display: flex; align-items: center; gap: 0.5rem;">{SVGS["UP"]} Signal Strength Trend</h4>', unsafe_allow_html=True)
    fig_sig = go.Figure()
    fig_sig.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Signal'], fill='tozeroy', name='Avg Signal', line=dict(color='#D4A853', width=2), fillcolor='rgba(212, 168, 83, 0.2)'))
    fig_sig.update_layout(title='', height=300, yaxis=dict(title=dict(text='Avg Signal Strength', font=dict(size=11, color='#94A3B8'))), hovermode='x unified')
    apply_chart_theme(fig_sig)
    st.plotly_chart(fig_sig, width='stretch')

    # Long/Short Ratio (line chart)
    st.markdown(f'<h4 style="font-family: var(--data); font-size: 0.9rem; color: var(--ink-secondary); text-transform: uppercase; margin-bottom: 1rem; letter-spacing: 0.08em; display: flex; align-items: center; gap: 0.5rem;">{SVGS["STRENGTH"]} Long/Short Ratio</h4>', unsafe_allow_html=True)
    fig_ratio = go.Figure()
    fig_ratio.add_hline(y=1.0, line_dash="dash", line_color="#D4A853", annotation_text="Neutral", annotation_position="right")
    fig_ratio.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['L_S_Ratio'], fill='tozeroy', name='L/S Ratio', line=dict(color='#06B6D4', width=2), fillcolor='rgba(6, 182, 212, 0.2)'))
    fig_ratio.update_layout(title='', height=300, yaxis=dict(title=dict(text='Ratio', font=dict(size=11, color='#94A3B8'))), hovermode='x unified')
    apply_chart_theme(fig_ratio)
    st.plotly_chart(fig_ratio, width='stretch')

    # Signal Conviction trend (line chart)
    st.markdown(f'<h4 style="font-family: var(--data); font-size: 0.9rem; color: var(--ink-secondary); text-transform: uppercase; margin-bottom: 1rem; letter-spacing: 0.08em; display: flex; align-items: center; gap: 0.5rem;">{SVGS["STRENGTH"]} Signal Conviction (Strength Confidence)</h4>', unsafe_allow_html=True)
    fig_conviction = go.Figure()
    fig_conviction.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Conviction'], fill='tozeroy', name='Conviction', line=dict(color='#8B5CF6', width=2), fillcolor='rgba(139, 92, 246, 0.2)'))
    fig_conviction.update_layout(title='', height=300, yaxis_title='Avg Absolute Signal Strength', hovermode='x unified')
    apply_chart_theme(fig_conviction)
    st.plotly_chart(fig_conviction, width='stretch')

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    render_footer()


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS FOR TAB RENDERING
# ══════════════════════════════════════════════════════════════════════════════

def _bucket_signals_by_age(results_df: pd.DataFrame, side: str = 'long') -> dict:
    """Bucket signals by age (Today, 1d, 2d, 3d, 5d) with stats for timeline display."""
    prefix = 'L' if side == 'long' else 'S'
    target_indicator = "●"
    buckets = {
        "Today": [],
        "1 Day Ago": [],
        "2 Days Ago": [],
        "3 Days Ago": [],
        "Within 5 Days": []
    }
    col_map = {
        "Today": f"{prefix}_Today",
        "1 Day Ago": f"{prefix}_1d",
        "2 Days Ago": f"{prefix}_2d",
        "3 Days Ago": f"{prefix}_3d",
        "Within 5 Days": f"{prefix}_5d"
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
            avg_signal = np.mean(signals)
            count = len(rows)
            stats[age] = {
                'count': count,
                'avg_signal': avg_signal,
                'rows': rows
            }
        else:
            stats[age] = {'count': 0, 'avg_signal': 0, 'rows': []}

    # Calculate trend: are signals strengthening (newer) or weakening (older)?
    today_avg = stats["Today"]['avg_signal'] if stats["Today"]['count'] > 0 else 0
    older_avg = np.mean([stats[age]['avg_signal'] for age in ["1 Day Ago", "2 Days Ago", "3 Days Ago", "Within 5 Days"] if stats[age]['count'] > 0]) if any(stats[age]['count'] for age in ["1 Day Ago", "2 Days Ago", "3 Days Ago", "Within 5 Days"]) else 0

    if today_avg > older_avg + 5:
        trend = f"{SVGS['UP'].replace('12','14').replace('12','14')} Strengthening"
        trend_color = "#2DD4A8"
    elif today_avg < older_avg - 5:
        trend = f"{SVGS['DOWN'].replace('12','14').replace('12','14')} Weakening"
        trend_color = "#E8555A"
    else:
        trend = "— Stable"
        trend_color = "#D4A853"

    return buckets, stats, trend, trend_color


def _render_signal_legend(side: str = 'long') -> None:
    """Render context-aware interpretation legend below a timing table."""
    if side == 'long':
        signal_desc  = "Positive WRCI value — the oscillator has crossed upward, indicating building bullish momentum. Higher magnitude = stronger push."
        trend_desc   = "Positive = uptrend confirming the signal. Negative = downtrend still in place despite the bullish cross."
        timing_desc  = "Older bullish signals are more reliable — the upside shift has had time to prove itself. Today&rsquo;s signal is fresh and may still be developing."
        together_good = "Signal &#x2B; | Trend &#x2B; = high conviction long — momentum and direction fully aligned."
        together_mixed = "Signal &#x2B; | Trend &#x2212; = bullish cross against a downtrend. Likely a counter-trend bounce — wait for Trend to turn positive before committing."
    else:
        signal_desc  = "Negative WRCI value — the oscillator has crossed downward, indicating building selling pressure. Higher magnitude (more negative) = stronger push."
        trend_desc   = "Negative = downtrend confirming the signal. Positive = uptrend still in place despite the bearish cross."
        timing_desc  = "Older bearish signals are more reliable — the downside shift has confirmed over time. Today&rsquo;s signal is fresh and may still be developing."
        together_good = "Signal &#x2212; | Trend &#x2212; = high conviction short — momentum and direction fully aligned."
        together_mixed = "Signal &#x2212; | Trend &#x2B; = bearish cross inside an uptrend. Possible exhaustion or pullback — not a clean short until the trend turns negative."

    st.markdown(f"""
    <div style="
        position: relative;
        margin-top: 1.25rem;
        padding: 0.85rem 1rem 0.85rem 1rem;
        background: rgba(255, 255, 255, 0.015);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        overflow: hidden;
    ">
        <div style="
            position: absolute; top: 0; left: 0; right: 0; height: 1px;
            background: linear-gradient(90deg, #D4A853 0%, rgba(212,168,83,0.25) 60%, transparent 100%);
            opacity: 0.55;
        "></div>
        <div style="
            display: flex; align-items: center; gap: 0.45rem;
            margin-bottom: 0.65rem;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.78rem; font-weight: 600;
            letter-spacing: 0.12em; text-transform: uppercase;
            color: #D4A853;
        ">
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#D4A853" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
            How to read this table
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.45rem 1.5rem;">
            <div>
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.82rem; font-weight:600; color:#F1F5F9;">Timing</span>
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:#4B5563;"> · </span>
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:#94A3B8;">{timing_desc}</span>
            </div>
            <div>
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.82rem; font-weight:600; color:#F1F5F9;">Signal</span>
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:#4B5563;"> · </span>
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:#94A3B8;">{signal_desc}</span>
            </div>
            <div>
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.82rem; font-weight:600; color:#F1F5F9;">Trend</span>
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:#4B5563;"> · </span>
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:#94A3B8;">{trend_desc}</span>
            </div>
        </div>
        <div style="margin-top: 0.6rem; padding-top: 0.6rem; border-top: 1px solid rgba(255,255,255,0.05);">
            <span style="font-family:'IBM Plex Mono',monospace; font-size:0.82rem; font-weight:600; color:#F1F5F9;">Reading Signal &amp; Trend together</span>
            <span style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:#4B5563;"> · </span>
            <span style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:#94A3B8;">{together_good} &nbsp;&nbsp;{together_mixed}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _build_signal_table_html(stats: dict, side: str = 'long') -> str:
    """Build organized HTML table for signals grouped by age with section headers."""
    import html as html_module

    accent_light = "#34D399" if side == 'long' else "#FB7185"
    border_color = "rgba(45, 212, 168, 0.3)" if side == 'long' else "rgba(232, 85, 90, 0.3)"
    header_bg = "rgba(45, 212, 168, 0.15)" if side == 'long' else "rgba(232, 85, 90, 0.15)"

    table_rows = []
    age_order = ["Today", "1 Day Ago", "2 Days Ago", "3 Days Ago", "Within 5 Days"]

    for age in age_order:
        if stats[age]['count'] == 0:
            continue

        # Section header for this age group
        avg_signal = stats[age]['avg_signal']
        count = stats[age]['count']
        table_rows.append(f"""
        <tr style="background: {header_bg}; border-bottom: 2px solid {border_color};">
            <td colspan="6" style="padding: 0.75rem 1rem; font-family: var(--display); font-size: 0.8rem; font-weight: 700; color: {accent_light}; text-transform: uppercase; letter-spacing: 0.05em;">
                {age} · {count} signal{'s' if count != 1 else ''} · Avg: {avg_signal:+.1f}
            </td>
        </tr>
        """)

        # Data rows for this age group
        for row in stats[age]['rows']:
            symbol = html_module.escape(str(row.get('DisplayName', row.get('Symbol', ''))))
            price = float(row.get('Price', 0))
            signal = float(row.get('Signal', 0))
            trend = float(row.get('Trend', 0))
            zone = html_module.escape(str(row.get('Zone', '—')))

            table_rows.append(f"""
            <tr>
                <td class="symbol">{symbol}</td>
                <td class="numeric currency">₹{price:,.2f}</td>
                <td class="numeric" style="color: {accent_light}; font-weight: 600;">{signal:+.2f}</td>
                <td class="numeric" style="color: {accent_light}; font-weight: 600;">{trend:+.2f}</td>
                <td class="numeric">{zone}</td>
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
            padding: 0.5rem 0.5rem 1.5rem 0.5rem;
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
            font-size: 0.62rem;
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
            font-size: 0.75rem;
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
                    <th class="numeric">Price (₹)</th>
                    <th class="numeric">Signal</th>
                    <th class="numeric">Trend</th>
                    <th class="numeric">Zone</th>
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


def _build_signal_strength_table_html(df: pd.DataFrame, side: str = 'long') -> str:
    """Build ranked HTML table for top signals by magnitude.

    Creates styled HTML table with colored accent for side (long=green, short=red),
    displaying symbol, price, signal magnitude, trend direction, and zone status.

    Returns: Complete HTML document string ready for st.components.v1.html().
    """
    import html as html_module

    accent_light = "#34D399" if side == 'long' else "#FB7185"
    border_color = "rgba(45, 212, 168, 0.3)" if side == 'long' else "rgba(232, 85, 90, 0.3)"

    table_rows = []
    for idx, (_, row) in enumerate(df.iterrows(), 1):
        symbol = html_module.escape(str(row.get('DisplayName', row.get('Symbol', ''))))
        price = float(row.get('Price', 0))
        signal = float(row.get('Signal', 0))
        trend = float(row.get('Trend', 0))
        zone = html_module.escape(str(row.get('Zone', '—')))

        rank_str = f"{idx:02d}"

        table_rows.append(f"""
        <tr>
            <td class="numeric" style="color: #D4A853; font-weight: 700;">{rank_str}</td>
            <td class="symbol">{symbol}</td>
            <td class="numeric currency">₹{price:,.2f}</td>
            <td class="numeric" style="color: {accent_light}; font-weight: 600;">{signal:+.2f}</td>
            <td class="numeric" style="color: {accent_light}; font-weight: 600;">{trend:+.2f}</td>
            <td class="numeric">{zone}</td>
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
            font-size: 0.62rem;
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
            font-size: 0.75rem;
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
                    <th class="numeric">Price (₹)</th>
                    <th class="numeric">Signal</th>
                    <th class="numeric">Trend</th>
                    <th class="numeric">Zone</th>
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




def main():
    """Main app entry point with state-based flow."""
    # Render sidebar and get parameters + run button state
    sidebar_out = render_sidebar()
    universe, selected_index, analysis_date, reg_len, wt_n1, wt_n2, levels, timeframe, mode, start_date, end_date, run_clicked = sidebar_out

    # Handle run button click
    if run_clicked:
        st.session_state["run_screener_flag"] = True
        st.session_state["timeseries_done"] = False
        st.session_state["results_df"] = None
        st.session_state["run_error"] = None
        st.rerun()

    # Reset timeseries_done if mode switches to Snapshot
    if mode == "Snapshot" and st.session_state.get("timeseries_done"):
        st.session_state["timeseries_done"] = False
        st.rerun()

    # Show landing page if no results yet AND not in time-series display mode
    if st.session_state["results_df"] is None and not st.session_state.get("run_screener_flag") and not st.session_state.get("timeseries_done"):
        ui.render_header("SANKET", "Market Signal Screener · संकेत · WRCI Engine")
        if st.session_state.get("run_error"):
            st.error(st.session_state["run_error"])
        render_landing_page()
        render_footer()
    else:
        # Run analysis if flagged
        if st.session_state.get("run_screener_flag"):
            if mode == "Snapshot":
                # Console header for local terminal monitoring
                console.header("SANKET TERMINAL — Institutional Signal Screener", f"v{VERSION}")
                console.main_header("ANALYSIS RUN START", {
                    "Universe": universe,
                    "Index": selected_index,
                    "Timeframe": timeframe,
                    "Target Date": analysis_date
                })

                results_df = run_screener_analysis(
                    universe, selected_index, analysis_date, reg_len, wt_n1, wt_n2, levels, timeframe
                )
                if results_df is None:
                    st.session_state["run_error"] = f"Failed to fetch constituents for '{selected_index}'. Check your internet connection or try a different index."
                else:
                    st.session_state["run_error"] = None
                st.session_state["results_df"] = results_df
                st.session_state["run_screener_flag"] = False
                st.rerun()
            else:
                # Time-series (Range Study) — Console Logging
                console.header("SANKET TERMINAL — Bulk Range Intelligence", f"v{VERSION}")
                console.main_header("RANGE STUDY START", {
                    "Universe": universe,
                    "Index": selected_index,
                    "Start Date": start_date,
                    "End Date": end_date,
                    "Timeframe": timeframe
                })
                
                # Time-series renders inline, so no rerun needed
                run_timeseries_analysis(
                    universe, selected_index, start_date, end_date, reg_len, wt_n1, wt_n2, levels, timeframe
                )
                st.session_state["run_screener_flag"] = False

        # Display single-date results (skip if time-series already rendered)
        if st.session_state["results_df"] is not None and not st.session_state.get("timeseries_done"):
            results_df = st.session_state["results_df"]
            
            # Safety: Ensure required columns exist (handles stale session state)
            if 'SimpleName' not in results_df.columns and not results_df.empty:
                results_df['SimpleName'] = results_df['Symbol'].str.replace(".NS", "", regex=False).str.lstrip("^")

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # ════════════════════════════════════════════════════════════════════════════
            # NEW TAB STRUCTURE — ACTION-FOCUSED DESIGN
            # ════════════════════════════════════════════════════════════════════════════

            tab_signals, tab_strength, tab_raw = st.tabs([
                "Action Dashboard",
                "Signal Strength",
                "System Data"
            ])

            # ════ TAB 1: ACTION DASHBOARD ════════════════════════════════════════════════
            with tab_signals:
                ui.render_section_header(
                    "Today's Signals",
                    "Momentum signals ranked by strength across bullish and bearish sides",
                    icon="zap",
                    accent="amber"
                )

                # Split into longs and shorts
                longs_df = results_df[results_df['L_5d'] != "—"].copy().sort_values('Signal', ascending=False)
                shorts_df = results_df[results_df['S_5d'] != "—"].copy().sort_values('Signal', ascending=True)

                if not longs_df.empty or not shorts_df.empty:
                    # Summary metrics
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    with mc1:
                        ui.render_metric_card(
                            "Long Signals",
                            str(len(longs_df)),
                            f"{len(longs_df)/len(results_df)*100:.0f}% of universe",
                            "success"
                        )
                    with mc2:
                        ui.render_metric_card(
                            "Short Signals",
                            str(len(shorts_df)),
                            f"{len(shorts_df)/len(results_df)*100:.0f}% of universe",
                            "danger"
                        )
                    with mc3:
                        strongest_long = longs_df.iloc[0] if not longs_df.empty else None
                        ui.render_metric_card(
                            "Strongest Long",
                            strongest_long['SimpleName'] if strongest_long is not None else "—",
                            f"Signal: {strongest_long['Signal']:.1f}" if strongest_long is not None else "No signals",
                            "info"
                        )
                    with mc4:
                        strongest_short = shorts_df.iloc[0] if not shorts_df.empty else None
                        ui.render_metric_card(
                            "Weakest Short",
                            strongest_short['SimpleName'] if strongest_short is not None else "—",
                            f"Signal: {strongest_short['Signal']:.1f}" if strongest_short is not None else "No signals",
                            "info"
                        )

                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

                    # Inject SVG icons into nested sub-tab labels via CSS ::before pseudo-elements
                    st.markdown("""
                    <style>
                    [data-testid="stTabs"] [data-testid="stTabs"] button[role="tab"]:nth-of-type(1) [data-testid="stMarkdownContainer"] p::before {
                        content: '';
                        display: inline-block;
                        width: 14px;
                        height: 14px;
                        margin-right: 8px;
                        vertical-align: -2px;
                        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%2334D399' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'><path d='m5 12 7-7 7 7'/><path d='M12 19V5'/></svg>");
                        background-repeat: no-repeat;
                        background-size: contain;
                    }
                    [data-testid="stTabs"] [data-testid="stTabs"] button[role="tab"]:nth-of-type(2) [data-testid="stMarkdownContainer"] p::before {
                        content: '';
                        display: inline-block;
                        width: 14px;
                        height: 14px;
                        margin-right: 8px;
                        vertical-align: -2px;
                        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%23FB7185' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'><path d='M12 5v14'/><path d='m19 12-7 7-7-7'/></svg>");
                        background-repeat: no-repeat;
                        background-size: contain;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                    # Sub-tabs for Bullish and Bearish signals (side-by-side navigation instead of vertical stacking)
                    bull_tab, bear_tab = st.tabs(["Bullish Signals by Timing", "Bearish Signals by Timing"])

                    _age_order = ["Today", "1 Day Ago", "2 Days Ago", "3 Days Ago", "Within 5 Days"]

                    with bull_tab:
                        if not longs_df.empty:
                            _, long_stats, _, _ = _bucket_signals_by_age(longs_df, side='long')
                            long_table_html = _build_signal_table_html(long_stats, side='long')
                            _groups = sum(1 for a in _age_order if long_stats[a]['count'] > 0)
                            _rows = sum(long_stats[a]['count'] for a in _age_order)
                            st.components.v1.html(long_table_html, height=70 + _groups * 46 + _rows * 44)
                        else:
                            st.info("No bullish signals detected.")
                        _render_signal_legend(side='long')

                    with bear_tab:
                        if not shorts_df.empty:
                            _, short_stats, _, _ = _bucket_signals_by_age(shorts_df, side='short')
                            short_table_html = _build_signal_table_html(short_stats, side='short')
                            _groups = sum(1 for a in _age_order if short_stats[a]['count'] > 0)
                            _rows = sum(short_stats[a]['count'] for a in _age_order)
                            st.components.v1.html(short_table_html, height=70 + _groups * 46 + _rows * 44)
                        else:
                            st.info("No bearish signals detected.")
                        _render_signal_legend(side='short')

                else:
                    st.info("No signals detected in the specified universe and timeframe.")


            # ════ TAB 3: SIGNAL STRENGTH ANALYSIS ════════════════════════════════════════
            with tab_strength:
                ui.render_section_header(
                    "Signal Strength Analysis",
                    "Top momentum signals ranked by magnitude, with zone and trend context",
                    icon="target",
                    accent="emerald"
                )

                # Strength metrics
                avg_signal_str = results_df['Signal'].abs().mean()
                avg_trend_str = results_df['Trend'].abs().mean()
                strong_trend_count = len(results_df[results_df['Trend'].abs() > 30])

                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    ui.render_metric_card("Avg Signal Magnitude", f"{avg_signal_str:.1f}", "Average across all symbols", "neutral")
                with col_s2:
                    ui.render_metric_card("Avg Trend Value", f"{avg_trend_str:.1f}", "Directional strength", "neutral")
                with col_s3:
                    ui.render_metric_card("Strong Trends", str(strong_trend_count), f"{strong_trend_count/len(results_df)*100:.0f}% of universe", "info")

                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

                # Top conviction signals — side-by-side tables (long vs short)
                top_longs = longs_df.head(10)
                top_shorts = shorts_df.head(10)

                col_l, col_s = st.columns(2)

                with col_l:
                    st.markdown(f"""
                    <h4 style="font-family: var(--display); font-size: 0.9rem; color: var(--emerald); margin: 1.5rem 0 1rem 0; text-transform: uppercase; letter-spacing: 0.08em; display: flex; align-items: center; gap: 0.5rem;">
                        {SVGS['LONG'].replace('currentColor', 'var(--emerald)')} Strongest Bullish Signals
                    </h4>
                    """, unsafe_allow_html=True)
                    if not top_longs.empty:
                        long_conviction_html = _build_signal_strength_table_html(top_longs, side='long')
                        st.components.v1.html(long_conviction_html, height=94 + len(top_longs) * 40)
                    else:
                        st.info("No bullish signals detected in this period.")

                with col_s:
                    st.markdown(f"""
                    <h4 style="font-family: var(--display); font-size: 0.9rem; color: var(--rose); margin: 1.5rem 0 1rem 0; text-transform: uppercase; letter-spacing: 0.08em; display: flex; align-items: center; gap: 0.5rem;">
                        {SVGS['SHORT'].replace('currentColor', 'var(--rose)')} Strongest Bearish Signals
                    </h4>
                    """, unsafe_allow_html=True)
                    if not top_shorts.empty:
                        short_conviction_html = _build_signal_strength_table_html(top_shorts, side='short')
                        st.components.v1.html(short_conviction_html, height=94 + len(top_shorts) * 40)
                    else:
                        st.info("No bearish signals detected in this period.")

            # ════ TAB 4: SYSTEM DATA ════════════════════════════════════════════════════
            with tab_raw:
                ui.render_section_header(
                    "System Raw Data",
                    "Complete underlying data for analysis and model validation",
                    icon="database",
                    accent="cyan"
                )

                st.markdown("""
                <p style="font-family: var(--data); font-size: 0.8rem; color: var(--ink-secondary); margin-bottom: 1rem;">
                    All WRCI engine outputs including oscillator values, trend metrics, zones, and historical signal history.
                </p>
                """, unsafe_allow_html=True)

                # Show all data with historical signals
                display_df = results_df[[
                    "DisplayName", "Price", "Signal", "Trend", "Wave", "Zone",
                    "SignalType", "L_Today", "L_1d", "L_2d", "L_3d", "L_5d",
                    "S_Today", "S_1d", "S_2d", "S_3d", "S_5d"
                ]].sort_values("Signal", ascending=False)

                st.dataframe(display_df, width='stretch', height=500)

                # ── EXPORT SECTION ─────────────────────────────────────────────────────────
                st.markdown('<div class="section-divider" style="margin-top: 2rem;"></div>', unsafe_allow_html=True)

                ui.render_section_header(
                    "Export Quant Dataset",
                    "Signal archives by timing and top-ranked strength lists",
                    icon="download",
                    accent="cyan"
                )

                # Row 1 — full signal lists by timing
                st.markdown('<p style="font-family:\'IBM Plex Mono\',monospace; font-size:0.65rem; color:#4B5563; text-transform:uppercase; letter-spacing:0.08em; margin: 0.5rem 0 0.4rem 0;">Signals by Timing</p>', unsafe_allow_html=True)
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        label="↑  Bullish Signals",
                        data=longs_df.to_csv(index=False).encode('utf-8'),
                        file_name=f"bullish_signals_{analysis_date}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="dl_bullish_timing",
                        help="All active bullish signals grouped by timing"
                    )
                with dl_col2:
                    st.download_button(
                        label="↓  Bearish Signals",
                        data=shorts_df.to_csv(index=False).encode('utf-8'),
                        file_name=f"bearish_signals_{analysis_date}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="dl_bearish_timing",
                        help="All active bearish signals grouped by timing"
                    )

                # Row 2 — top 10 by signal strength
                st.markdown('<p style="font-family:\'IBM Plex Mono\',monospace; font-size:0.65rem; color:#4B5563; text-transform:uppercase; letter-spacing:0.08em; margin: 0.9rem 0 0.4rem 0;">Top 10 by Strength</p>', unsafe_allow_html=True)
                dl_col3, dl_col4 = st.columns(2)
                with dl_col3:
                    st.download_button(
                        label="↑  Top 10 Bullish",
                        data=top_longs.to_csv(index=False).encode('utf-8'),
                        file_name=f"top10_bullish_{analysis_date}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="dl_top10_bullish",
                        help="Top 10 bullish signals ranked by signal magnitude"
                    )
                with dl_col4:
                    st.download_button(
                        label="↓  Top 10 Bearish",
                        data=top_shorts.to_csv(index=False).encode('utf-8'),
                        file_name=f"top10_bearish_{analysis_date}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="dl_top10_bearish",
                        help="Top 10 bearish signals ranked by signal magnitude"
                    )

            render_footer()


if __name__ == "__main__":
    main()
