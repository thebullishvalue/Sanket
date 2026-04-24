"""
SANKET | Unified Market Analysis (UMA) Intelligence Terminal
WRCI Ensemble Engine — Wave-Regime Composite Index for Multi-Asset Screening
Powered by Pragyam Quantitative Intelligence
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
import gc
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

VERSION = "v1.2.0"

# ══════════════════════════════════════════════════════════════════════════════
# QUANTITATIVE INFRASTRUCTURE & STATE MANAGEMENT
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
# ASSET UNIVERSE & INSTRUMENT TAXONOMY
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

UNIVERSE_OPTIONS = ["India Indexes", "US Indexes", "ETF Index", "Commodities", "Currency", "Crypto", "Bond Yields"]
TIMEFRAME_OPTIONS = ["Daily", "Weekly", "Monthly"]

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

# Bond Yield Universe (Stooq Tickers)
BOND_YIELD_MAP = {
    "US 10Y": "10YUSY.B",
    "US 2Y": "02YUSY.B",
    "US 30Y": "30YUSY.B",
    "Japan 10Y": "10YJPY.B",
    "Japan 2Y": "02YJPY.B",
    "China 10Y": "10YCNY.B",
    "China 2Y": "02YCNY.B",
    "UK 10Y": "10YGBY.B",
    "UK 2Y": "02YGBY.B",
    "India 10Y": "10YINY.B",
    "India 2Y": "02YINY.B",
    "Germany 10Y": "10YDEY.B",
    "Germany 2Y": "02YDEY.B",
}
BOND_YIELD_LIST = list(BOND_YIELD_MAP.keys())

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
    "USD/BRL": "USDBRL=X",
}
CURRENCY_LIST = list(CURRENCY_MAP.keys())

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
ASSET_NAME_LOOKUP = {v: k for k, v in {**COMMODITY_MAP, **CURRENCY_MAP, **CRYPTO_MAP, **BOND_YIELD_MAP}.items()}

# ══════════════════════════════════════════════════════════════════════════════
# INSTRUMENT DISCOVERY & DATA PIPELINES
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
@st.cache_data(ttl=3600, show_spinner=False)
def get_fno_stock_list():
    """
    Identifies high-liquidity F&O constituents via a resilient multi-source pipeline.
    Prioritizes real-time NSE JSON APIs with fallback to historical CSV archives 
    to ensure universe continuity even during exchange downtime.
    """
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


# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSE DATA FETCHERS
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_index_from_wikipedia(index):
    """
    Scrapes Wikipedia for index constituents when official NSE sources fail.
    Uses pattern matching to identify symbol/ticker columns across various table formats.
    """
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


class SymbolFetcher:
    """
    Orchestrates the discovery of tradable instruments across multi-asset universes.
    Standardizes ticker formatting (.NS, =X, -USD) to ensure seamless integration 
    with the quantitative analysis engine.
    """
    
    @staticmethod
    def get_fno_list():
        return get_fno_stock_list()

    @staticmethod
    def get_india_index(index):
        if index == "F&O Stocks": return SymbolFetcher.get_fno_list()
        if index == "Benchmark Indexes": return BENCHMARK_INDEXES_LIST, f"✓ Loaded {len(BENCHMARK_INDEXES_LIST)} benchmark index instruments"
        
        # Source 1: NSE API
        try:
            import urllib.parse
            api_url = f"https://www.nseindia.com/api/equity-stockIndices?index={urllib.parse.quote(index)}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/json',
                'Referer': 'https://www.nseindia.com/market-data/live-equity-market',
            }
            session = requests.Session()
            session.get("https://www.nseindia.com", headers=headers, timeout=10)
            response = session.get(api_url, headers=headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    symbols = [item['symbol'] for item in data['data'] if 'symbol' in item]
                    symbols = [str(s) + ".NS" for s in symbols[1:] if s and str(s).strip()]
                    if symbols: return symbols, f"✓ Fetched {len(symbols)} constituents (NSE API)"
        except Exception: pass

        # Source 2: NSE CSV Fallback
        url = INDEX_URL_MAP.get(index)
        if url:
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(url, headers=headers, verify=False, timeout=15)
                if response.status_code == 200:
                    stock_df = pd.read_csv(io.StringIO(response.text))
                    symbol_col = next((c for c in stock_df.columns if c.lower() == 'symbol'), None)
                    if symbol_col:
                        symbols = [str(s) + ".NS" for s in stock_df[symbol_col].tolist() if s and str(s).strip()]
                        if symbols: return symbols, f"✓ Fetched {len(symbols)} constituents (NSE archive)"
            except Exception: pass

        # Source 3: Wikipedia Fallback
        return _fetch_index_from_wikipedia(index)

    @staticmethod
    def get_us_index(index):
        index_map = {"S&P 500": ["^GSPC"], "DOW JONES": ["^DJI"], "NASDAQ 100": ["^NDX"]}
        if index in index_map: return index_map[index], f"✓ Fetched {index}"
        return None, f"Unknown US index: {index}"

    @staticmethod
    def get_map_universe(name, mapping, category_label):
        if name is None or name.startswith("All "):
            return list(mapping.values()), f"✓ Fetched {len(mapping)} {category_label}"
        symbol = mapping.get(name)
        if symbol: return [symbol], f"✓ Fetched {name}"
        return None, f"Unknown {category_label}: {name}"

def get_universe_symbols(universe, index_or_ticker):
    """Primary interface for universe discovery."""
    if universe == "India Indexes": return SymbolFetcher.get_india_index(index_or_ticker)
    if universe == "US Indexes": return SymbolFetcher.get_us_index(index_or_ticker)
    if universe == "ETF Index": return ETF_LIST, f"✓ Loaded {len(ETF_LIST)} ETFs"
    if universe == "Commodities": return SymbolFetcher.get_map_universe(index_or_ticker, COMMODITY_MAP, "commodities")
    if universe == "Currency": return SymbolFetcher.get_map_universe(index_or_ticker, CURRENCY_MAP, "currency pairs")
    if universe == "Crypto": return SymbolFetcher.get_map_universe(index_or_ticker, CRYPTO_MAP, "digital assets")
    if universe == "Bond Yields": return SymbolFetcher.get_map_universe(index_or_ticker, BOND_YIELD_MAP, "bond yields")
    return None, f"Unknown universe: {universe}"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stooq_data(ticker, days_back=500):
    """Fetch historical data from Stooq CSV API."""
    try:
        # Ticker on Stooq is case insensitive for URL but usually uppercase
        url = f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d&e=csv"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': f'https://stooq.com/q/d/?s={ticker.lower()}'
        }
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            # Check if response is actually a CSV or the "apikey" message
            if "Get your apikey" in response.text:
                console.warning(f"Stooq API Key Required for {ticker}. Skipping...")
                return pd.DataFrame()
            
            df = pd.read_csv(io.StringIO(response.text), on_bad_lines='skip', engine='python')
            if not df.empty and 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                # Stooq often provides: Date, Open, High, Low, Close, Volume
                # Filter by days_back
                start_date = datetime.date.today() - datetime.timedelta(days=days_back)
                df = df[df.index >= pd.to_datetime(start_date)]
                return df
    except Exception as e:
        console.warning(f"Stooq Error: {ticker}: {str(e)}")
    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_macro_drivers(days_back=500):
    """
    Fetch all macro drivers (Yields, Commodities, FX, DXY) required for MMR.
    """
    macro_data = {}
    
    # 1. Bond Yields (Stooq with YFinance Fallback for US)
    for name, ticker in BOND_YIELD_MAP.items():
        df = fetch_stooq_data(ticker, days_back)
        if df.empty and name.startswith("US "):
            # Fallback for US Yields using YFinance tickers
            yf_map = {"US 10Y": "^TNX", "US 30Y": "^TYX", "US 5Y": "^FVX"}
            if name in yf_map:
                console.item("Fallback", f"Fetching {name} from YFinance...")
                yf_df, _ = fetch_batch_data([yf_map[name]], days_back=days_back, include_live=True)
                if yf_df and yf_map[name] in yf_df:
                    df = yf_df[yf_map[name]]
        
        if not df.empty:
            macro_data[name] = df
            
    # 2. Commodities (YFinance)
    commodity_tickers = list(COMMODITY_MAP.values())
    comm_data, _ = fetch_batch_data(commodity_tickers, days_back=days_back, include_live=True)
    if comm_data:
        for name, ticker in COMMODITY_MAP.items():
            if ticker in comm_data:
                macro_data[name] = comm_data[ticker]
                
    # 3. Currencies (YFinance)
    currency_tickers = list(CURRENCY_MAP.values())
    curr_data, _ = fetch_batch_data(currency_tickers, days_back=days_back, include_live=True)
    if curr_data:
        for name, ticker in CURRENCY_MAP.items():
            if ticker in curr_data:
                macro_data[name] = curr_data[ticker]
                
    # 4. Special Drivers: DXY, GOLD, SILVER (if not already fetched)
    special = {"DXY": "DX-Y.NYB", "GOLD": "GC=F", "SILVER": "SI=F"}
    for name, ticker in special.items():
        if name not in macro_data:
            s_data, _ = fetch_batch_data([ticker], days_back=days_back, include_live=True)
            if s_data and ticker in s_data:
                macro_data[name] = s_data[ticker]
                
    return macro_data


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
            return {}, "No data returned"
            
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
             return {}, "Unexpected data structure"

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


def f_zscore_clipped(series, window, clip_threshold=3.0):
    """Compute rolling z-score clipped to +/- threshold. Resilient to NaNs."""
    mean = series.rolling(window=window, min_periods=window//4).mean()
    std = series.rolling(window=window, min_periods=window//4).std()
    z = (series - mean) / std.replace(0, np.nan)
    return z.fillna(0).clip(lower=-clip_threshold, upper=clip_threshold)


def f_sigmoid(z, scale=1.0):
    """Sigmoid transformation: 2/(1+exp(-z/scale)) - 1. Resilient to NaNs."""
    # Ensure z is a Series or handle single values
    if isinstance(z, (pd.Series, np.ndarray)):
        res = 2.0 / (1.0 + np.exp(-z.astype(float) / scale)) - 1.0
        return pd.Series(res, index=z.index) if isinstance(z, pd.Series) else res
    else:
        return 2.0 / (1.0 + np.exp(-float(z) / scale)) - 1.0




def compute_hurst(price_series, short=10, long=50, sample_len=100):
    """
    Compute Hurst exponent via variance ratio method (as in Pine Script).
    Returns hurst_clipped between 0.1 and 0.9.
    """
    log_returns = np.log(price_series / price_series.shift(1)).dropna()
    if len(log_returns) < max(short, long) + sample_len:
        return pd.Series(0.5, index=price_series.index)
    
    ret_short = log_returns.rolling(short).sum()
    ret_long = log_returns.rolling(long).sum()
    
    std_short = ret_short.rolling(sample_len).std()
    std_long = ret_long.rolling(sample_len).std()
    
    log_ratio_std = np.log(std_long / std_short.replace(0, np.nan))
    log_ratio_tau = np.log(long / short)
    hurst_raw = log_ratio_std / log_ratio_tau
    
    hurst_smooth = hurst_raw.ewm(span=10, adjust=False).mean()
    hurst_clipped = hurst_smooth.clip(lower=0.1, upper=0.9)
    return hurst_clipped.fillna(0.5)


def compute_vol_stress(df, length=20):
    """
    Compute volatility stress metrics. Resilient to NaNs.
    """
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # ATR Proxy
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    atr_ref = tr.rolling(14).mean().fillna(method='ffill').fillna(0)
    
    atr_ref_mean = atr_ref.rolling(length).mean()
    vov_raw = atr_ref / atr_ref_mean.replace(0, np.nan)
    vov_z = f_zscore_clipped(vov_raw, length, 3.0)
    
    atr_s = tr.rolling(5).mean().fillna(0)
    atr_l = tr.rolling(20).mean().fillna(0)
    vts_raw = atr_l / atr_s.replace(0, np.nan)
    vts_z = f_zscore_clipped(vts_raw, length, 3.0)
    
    vol_stress_z = (vov_z.fillna(0) + vts_z.fillna(0)) / np.sqrt(2.0)
    vol_stress_sigmoid = f_sigmoid(vol_stress_z, 1.5).fillna(0)
    
    vts_regime = np.select(
        [vts_raw.fillna(1.0) > 1.15, vts_raw.fillna(1.0) > 1.0, vts_raw.fillna(1.0) < 0.85, vts_raw.fillna(1.0) < 1.0],
        [2, 1, -2, -1],
        default=0
    )
    return vol_stress_sigmoid, pd.Series(vts_regime, index=df.index)


def compute_mmr(src_y, macro_data_dict, window_reg=20, window_z=20, z_clip=3.0):
    """
    Compute Macro Multiple Regression (MMR) signal using Orthogonalized Gram-Schmidt predictors.
    """
    if not macro_data_dict:
        return pd.Series(0.0, index=src_y.index), 0.0
    
    # 1. Calculate correlations
    corrs = {}
    for name, df in macro_data_dict.items():
        if not df.empty and 'Close' in df.columns:
            aligned_close = df['Close'].reindex(src_y.index, method='ffill')
            corrs[name] = src_y.corr(aligned_close)
    
    if not corrs:
        return pd.Series(0.0, index=src_y.index), 0.0
    
    # 2. Sort and pick top 3 predictors (as per Pine Script)
    sorted_drivers = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
    top_3 = sorted_drivers[:3]
    
    # Extract predictor series
    X = []
    for name, c in top_3:
        df = macro_data_dict[name]
        X.append(df['Close'].reindex(src_y.index, method='ffill'))
    
    if len(X) < 1:
        return pd.Series(0.0, index=src_y.index), 0.0

    # 3. Gram-Schmidt Orthogonalization & Multiple Regression
    # We'll use a simplified version of the Pine Script GS process
    # for rolling regression if needed, but here we do it on the full series
    # for point-in-time calculation (or rolling if we want a trace)
    
    def get_variance(s, w): return s.rolling(w).var()
    def get_covariance(s1, s2, w): return s1.rolling(w).cov(s2)
    
    # x1, x2, x3
    x1 = X[0]
    u1 = x1
    var_u1 = get_variance(u1, window_reg)
    
    if len(X) > 1:
        x2 = X[1]
        cov_x2_u1 = get_covariance(x2, u1, window_reg)
        proj_x2_u1 = (cov_x2_u1 / var_u1.replace(0, np.nan)) * u1
        u2 = x2 - proj_x2_u1.fillna(0)
    else:
        u2 = pd.Series(0.0, index=src_y.index)
        
    if len(X) > 2:
        x3 = X[2]
        var_u2 = get_variance(u2, window_reg)
        cov_x3_u1 = get_covariance(x3, u1, window_reg)
        cov_x3_u2 = get_covariance(x3, u2, window_reg)
        proj_x3_u1 = (cov_x3_u1 / var_u1.replace(0, np.nan)) * u1
        proj_x3_u2 = (cov_x3_u2 / var_u2.replace(0, np.nan)) * u2
        u3 = x3 - proj_x3_u1.fillna(0) - proj_x3_u2.fillna(0)
    else:
        u3 = pd.Series(0.0, index=src_y.index)
        
    # Beta calculations
    def get_slope(u, y, w):
        var_u = get_variance(u, w)
        cov_uy = get_covariance(u, y, w)
        return (cov_uy / var_u.replace(0, np.nan)).fillna(0)
    
    b1 = get_slope(u1, src_y, window_reg)
    b2 = get_slope(u2, src_y, window_reg)
    b3 = get_slope(u3, src_y, window_reg)
    
    m_y = src_y.rolling(window_reg).mean()
    m_u1 = u1.rolling(window_reg).mean()
    m_u2 = u2.rolling(window_reg).mean()
    m_u3 = u3.rolling(window_reg).mean()
    
    intercept = m_y - (b1 * m_u1) - (b2 * m_u2) - (b3 * m_u3)
    y_pred = intercept + (b1 * u1) + (b2 * u2) + (b3 * u3)
    
    # Model Quality (R2)
    ssr = ((y_pred - m_y)**2).rolling(window_reg).mean()
    sst = get_variance(src_y, window_reg)
    model_r2 = (ssr / sst.replace(0, np.nan)).clip(0, 1).fillna(0)
    
    # MMR Signal
    deviation = src_y - y_pred
    deviation_z = f_zscore_clipped(deviation, window_z, z_clip)
    mmr_signal = f_sigmoid(deviation_z, 1.5)
    
    return mmr_signal, model_r2, top_3


def compute_msf(df, params=None):
    """
    Compute MSF (Momentum Structure Flow) signal per Pine Script UMA v6.
    
    Args:
        df: DataFrame with columns ['Open','High','Low','Close','Volume']
        params: dict with keys: length, rocLength, wt_channel_len, wt_avg_len, 
                entropy_dampen_scale, hurst_influence, vol_dampen_scale
    
    Returns:
        tuple: (msf_signal, msf_clarity, components_dict)
    """
    if params is None:
        params = {
            'length': 20,
            'rocLength': 14,
            'wt_channel_len': 10,
            'wt_avg_len': 21,
            'entropy_dampen_scale': 0.25,
            'hurst_influence': 0.3,
            'vol_dampen_scale': 0.15,
        }
    length = params['length']
    rocLength = params['rocLength']
    wt_channel_len = params['wt_channel_len']
    wt_avg_len = params['wt_avg_len']
    entropy_dampen_scale = params['entropy_dampen_scale']
    hurst_influence = params['hurst_influence']
    vol_dampen_scale = params['vol_dampen_scale']
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    hlc3 = (high + low + close) / 3.0
    
    # --- 3.1 Momentum Component (ROC-based) ---
    roc_raw = close.pct_change(periods=rocLength) * 100
    roc_z = f_zscore_clipped(roc_raw, length, 3.0)
    momentum_norm = f_sigmoid(roc_z, 1.5)
    
    # --- 3.2 Market Microstructure Component ---
    open_ = df['Open']
    intrabar_direction = (high + low) / 2 - open_
    vol_ma = volume.rolling(window=length).mean()
    vol_ratio = volume / vol_ma
    vw_direction = (intrabar_direction * vol_ratio).rolling(window=length).mean()
    price_change_impact = close - close.shift(5)
    vw_impact = (price_change_impact * vol_ratio).rolling(window=length).mean()
    microstructure_raw = vw_direction - vw_impact
    microstructure_z = f_zscore_clipped(microstructure_raw, length, 3.0)
    microstructure_norm = f_sigmoid(microstructure_z, 1.5)
    
    # --- 3.3 Volatility Regime (Confidence Bands) ---
    price_mean = close.rolling(window=length).mean()
    price_std = close.rolling(window=length).std()
    conf_mult = 1.96
    upper_bound = price_mean + conf_mult * price_std
    lower_bound = price_mean - conf_mult * price_std
    band_width = upper_bound - lower_bound
    price_position = band_width.where(band_width > 0, 0).pipe(lambda bw: (close - lower_bound) / bw * 2 - 1)
    price_position_clipped = price_position.clip(lower=-1.5, upper=1.5)
    
    # --- 3.4 Composite Trend ---
    trend_fast = close.rolling(5).mean()
    trend_slow = price_mean
    trend_diff_z = f_zscore_clipped(trend_fast - trend_slow, length, 3.0)
    
    momentum_accel_raw = close.diff(5).diff(5)
    momentum_accel_z = f_zscore_clipped(momentum_accel_raw, length, 3.0)
    
    atr_val = (high - low).rolling(14).mean()
    vol_adj_mom_raw = close.diff(5) / atr_val
    vol_adj_mom_z = f_zscore_clipped(vol_adj_mom_raw, length, 3.0)
    
    mean_reversion_z = f_zscore_clipped(close - price_mean, length, 3.0)
    
    composite_trend_z = (trend_diff_z.fillna(0) + momentum_accel_z.fillna(0) + vol_adj_mom_z.fillna(0) + mean_reversion_z.fillna(0)) / 4.0
    composite_trend_norm = f_sigmoid(composite_trend_z, 1.5).fillna(0)
    
    # --- 3.5 Accumulation/Distribution ---
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    mf_positive = money_flow.where(close > close.shift(1), 0)
    mf_negative = money_flow.where(close < close.shift(1), 0)
    mf_pos_smooth = mf_positive.rolling(window=length).mean()
    mf_neg_smooth = mf_negative.rolling(window=length).mean()
    mf_total = mf_pos_smooth + mf_neg_smooth
    accum_ratio = mf_pos_smooth / mf_total.replace(0, np.nan)
    accum_norm = 2.0 * (accum_ratio - 0.5)
    
    # --- 3.6 Regime Counter ---
    pct_change = close.pct_change()
    threshold_pct = 0.33 / 100.0
    sign_series = np.where(pct_change > threshold_pct, 1, np.where(pct_change < -threshold_pct, -1, 0))
    regime_raw = pd.Series(sign_series, index=close.index).rolling(window=length).sum()
    regime_z = f_zscore_clipped(regime_raw, length, 3.0)
    regime_norm = f_sigmoid(regime_z, 1.5)
    
    # --- 3.7 RSI Component ---
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss
    rsi_value = 100 - (100 / (1 + rs))
    
    # --- 3.8 WaveTrend Cycle Component ---
    wt_ap = hlc3
    wt_esa = wt_ap.ewm(span=wt_channel_len, adjust=False).mean()
    wt_d = (wt_ap - wt_esa).abs().ewm(span=wt_channel_len, adjust=False).mean()
    wt_ci = (wt_ap - wt_esa) / (0.015 * wt_d).replace(0, np.nan)
    wt1 = wt_ci.ewm(span=wt_avg_len, adjust=False).mean()
    wt2 = wt1.rolling(window=4).mean()
    wt_hist = wt1 - wt2
    
    wt_z = f_zscore_clipped(wt1, length, 3.0)
    wavetrend_norm = f_sigmoid(wt_z, 1.5)
    
    # --- Modulators ---
    entropy_norm = compute_permutation_entropy(close, 50)
    # entropy_mod = 1.0 - entropy_dampen_scale * f_sigmoid(entropy_z, 1.5)
    entropy_z = f_zscore_clipped(entropy_norm, length, 3.0)
    entropy_mod = 1.0 - entropy_dampen_scale * f_sigmoid(entropy_z, 1.5).clip(lower=0)
    
    hurst_clipped = compute_hurst(close, short=10, long=50, sample_len=100)
    hurst_baseline = hurst_clipped.rolling(length * 5).mean()
    hurst_centered = hurst_clipped - hurst_baseline.fillna(0.5)
    hurst_tilt = (hurst_centered * 2.5).clip(-1, 1)
    
    h_w_mom = 1.0 + hurst_tilt * hurst_influence
    h_w_str = 1.0 + hurst_tilt * hurst_influence
    h_w_flo = 1.0 - hurst_tilt * hurst_influence * 0.5
    h_w_cyc = 1.0 - hurst_tilt * hurst_influence
    h_w_denom = np.sqrt(h_w_mom**2 + h_w_str**2 + h_w_flo**2 + h_w_cyc**2)
    
    vol_stress_sigmoid, vts_regime = compute_vol_stress(df, length)
    vol_mod = 1.0 - vol_dampen_scale * vol_stress_sigmoid.clip(lower=0)
    
    # --- MSF Composite Signal ---
    osc_momentum = momentum_norm
    osc_structure = (microstructure_norm + composite_trend_norm) / np.sqrt(2.0)
    osc_flow = (accum_norm + regime_norm) / np.sqrt(2.0)
    osc_cycle = wavetrend_norm
    
    msf_raw_weighted = (h_w_mom * osc_momentum.fillna(0) + h_w_str * osc_structure.fillna(0) + h_w_flo * osc_flow.fillna(0) + h_w_cyc * osc_cycle.fillna(0))
    msf_raw = msf_raw_weighted / h_w_denom.replace(0, np.nan)
    msf_pre_mod = f_sigmoid(msf_raw * 2.0, 1.0)
    msf_signal = (msf_pre_mod * entropy_mod.fillna(1.0) * vol_mod.fillna(1.0)).clip(-1, 1).fillna(0)
    return {
        'msf_signal': msf_signal,
        'msf_clarity': msf_signal.abs().fillna(0),
        'osc_momentum': osc_momentum.fillna(0),
        'osc_structure': osc_structure.fillna(0),
        'osc_flow': osc_flow.fillna(0),
        'osc_cycle': osc_cycle.fillna(0),
        'wt1': wt1.fillna(0),
        'wt2': wt2.fillna(0),
        'wt_hist': wt_hist.fillna(0),
        'price_position': price_position_clipped.fillna(0),
        'rsi_value': rsi_value.fillna(50),
        'entropy_norm': entropy_norm.fillna(0.5),
        'hurst_clipped': hurst_clipped.fillna(0.5),
        'vol_stress': vol_stress_sigmoid.fillna(0),
        'vts_regime': vts_regime.fillna(0),
        'vol_mod': vol_mod.fillna(1.0),
        'trend_norm': composite_trend_norm.fillna(0),
    }


def run_uma_v6_analysis(df_daily, timeframe, macro_data=None, params=None):
    """
    Executes the UMA v6 Intelligence Ensemble, synthesizing Macro-Market Regimes (MMR)
    with Market Signal Fusion (MSF) components. 
    
    Acts as the primary analytical brain, applying volatility-adjusted trend scoring 
    and entropy-based noise filtering to identify institutional-grade setups.
    """
    if params is None:
        params = {
            'length': 20,
            'rocLength': 14,
            'wt_channel_len': 10,
            'wt_avg_len': 21,
            'entropy_dampen_scale': 0.25,
            'hurst_influence': 0.3,
            'vol_dampen_scale': 0.15,
            'msf_weight_base': 0.5,
            'regime_sensitivity': 1.5
        }
    
    # 1. Frequency handling
    if timeframe == "Weekly":
        primary_df = resample_to_weekly(df_daily)
    else:
        primary_df = df_daily
        
    # Initialize columns with safe defaults
    uma_cols = [
        'Unified_Osc', 'MSF_Osc', 'MMR_Osc', 'Signal_Line', 'Norm_Trend', 
        'WT1', 'WT2', 'Entropy', 'Hurst', 'Vol_Stress', 'MMR_Quality', 
        'MSF_Weight', 'MMR_Weight', 'Condition', 'is_tier3_buy', 'is_tier3_sell',
        'long_cond_momentum', 'short_cond_momentum', 'long_cond_threshold', 'short_cond_threshold',
        'long_cond_crossover', 'short_cond_crossover'
    ]
    for col in uma_cols:
        if col not in primary_df.columns:
            primary_df[col] = 0.0 if 'cond' not in col and 'is_tier' not in col else False
            if col == 'Condition': primary_df[col] = 'Neutral'
            if col == 'Hurst': primary_df[col] = 0.5
            if col == 'Entropy': primary_df[col] = 0.5
            
    if len(primary_df) < 50:
        return primary_df
    
    # 2. MSF Analysis
    msf_res = compute_msf(primary_df, params)
    
    # 3. MMR Analysis
    mmr_signal, mmr_quality, top_3 = compute_mmr(primary_df['Close'], macro_data, window_reg=params['length'])
    
    # 4. Signal Integration (Section 9 of Pine Script)
    msf_signal = msf_res['msf_signal'].fillna(0)
    msf_clarity = msf_res['msf_clarity'].fillna(0)
    mmr_clarity = mmr_signal.abs().fillna(0)
    
    # 4. Core Mathematical Layer (V4 Parity)
    # This ensures Signal, Trend, and Wave match the v4 system exactly
    hlc3 = (primary_df['High'] + primary_df['Low'] + primary_df['Close']) / 3.0
    wt_n1, wt_n2 = 10, 21
    reg_len = params.get('length', 20)
    
    # 1. Primary Momentum Oscillator: WaveTrend (WT1/WT2)
    # Computes an EMA-smoothed channel index (HLC3 relative to its own moving average)
    # to identify overextended price cycles relative to short-term mean volatility.
    esa = hlc3.ewm(span=wt_n1, adjust=False).mean()
    d = (hlc3 - esa).abs().ewm(span=wt_n1, adjust=False).mean()
    ci = (hlc3 - esa) / (0.015 * d).replace(0, np.nan)
    wt1 = ci.ewm(span=wt_n2, adjust=False).mean()
    wt2 = wt1.rolling(window=4).mean()
    
    # 2. Structural Trend Component: Normalized HMA Count
    # Quantifies the current price location relative to a Hull Moving Average (HMA)
    # over the last 'reg_len' periods to detect directional drift and trend maturity.
    hma_p = calculate_hma(hlc3, 15)
    trend_count = calculate_trend_count(hma_p, reg_len)
    norm_trend = trend_count * (100.0 / reg_len)
    
    # 3. Core Ensemble Signal (V4 Baseline)
    # Fuses momentum (WaveTrend) and trend drift (HMA Count) into a single unified index.
    v4_signal = (wt1 + norm_trend) / 2.0
    
    # 4. Intelligence Synthesis: MSF & MMR Weights
    # Dynamically allocates conviction weight between local momentum (MSF) 
    # and macro-regime (MMR) confirmation based on the relative clarity of each layer.
    sens = params.get('regime_sensitivity', 1.5)
    msf_clarity_scaled = msf_res['msf_clarity'].fillna(0) ** sens
    mmr_clarity_scaled = (mmr_signal.abs().fillna(0) * mmr_quality) ** sens
    clarity_sum = msf_clarity_scaled + mmr_clarity_scaled + 0.001
    msf_w_norm = msf_clarity_scaled / clarity_sum
    mmr_w_norm = mmr_clarity_scaled / clarity_sum
    
    # 6. Populate Core Columns
    primary_df['Unified_Osc'] = v4_signal.fillna(0)
    primary_df['Signal_Line'] = primary_df['Unified_Osc'].rolling(4).mean().fillna(method='bfill').fillna(0)
    primary_df['WT1'] = wt1
    primary_df['WT2'] = wt2
    primary_df['Norm_Trend'] = norm_trend
    
    # Secondary v6 Indicator Columns
    primary_df['MSF_Osc'] = msf_signal * 100.0
    primary_df['MMR_Osc'] = mmr_signal.fillna(0) * 100.0
    primary_df['Entropy'] = msf_res['entropy_norm'].fillna(0.5)
    primary_df['Hurst'] = msf_res['hurst_clipped'].fillna(0.5)
    primary_df['Vol_Stress'] = msf_res['vol_stress'].fillna(0)
    primary_df['MMR_Quality'] = mmr_quality
    primary_df['MSF_Weight'] = msf_w_norm
    primary_df['MMR_Weight'] = mmr_w_norm

    # 6. Signal Conditions — Aligned with README.md / v4
    
    # ── SET A: Threshold — composite crosses extreme level with signal confirmation ──
    # Long: Composite crosses below -40 (oversold entry), Signal_Line stays above -40
    # Short: Composite crosses above +40 (overbought entry), Signal_Line stays below +40
    primary_df['long_cond_threshold'] = (primary_df['Unified_Osc'] < -40) & (primary_df['Unified_Osc'].shift(1) >= -40) & (primary_df['Signal_Line'] > -40)
    primary_df['short_cond_threshold'] = (primary_df['Unified_Osc'] > 40) & (primary_df['Unified_Osc'].shift(1) <= 40) & (primary_df['Signal_Line'] < 40)
    
    # ── SET B: Crossover — composite crosses its signal line in extreme zone ──
    # Long: Composite crosses below signal line while already in oversold (< -40)
    # Short: Composite crosses above signal line while already in overbought (> +40)
    primary_df['long_cond_crossover'] = (primary_df['Unified_Osc'] < primary_df['Signal_Line']) & (primary_df['Unified_Osc'].shift(1) >= primary_df['Signal_Line'].shift(1)) & (primary_df['Unified_Osc'] < -40)
    primary_df['short_cond_crossover'] = (primary_df['Unified_Osc'] > primary_df['Signal_Line']) & (primary_df['Unified_Osc'].shift(1) <= primary_df['Signal_Line'].shift(1)) & (primary_df['Unified_Osc'] > 40)
    
    # ── SET C: Momentum — pure crossover, no level filter (for Range Study only) ──
    primary_df['long_cond_momentum'] = (primary_df['Unified_Osc'] > primary_df['Signal_Line']) & (primary_df['Unified_Osc'].shift(1) <= primary_df['Signal_Line'].shift(1))
    primary_df['short_cond_momentum'] = (primary_df['Unified_Osc'] < primary_df['Signal_Line']) & (primary_df['Unified_Osc'].shift(1) >= primary_df['Signal_Line'].shift(1))
    
    # ── Zone Classification — 80/40/-40/-80 ──
    primary_df['Condition'] = np.select(
        [primary_df['Unified_Osc'] > 80, primary_df['Unified_Osc'] > 40, primary_df['Unified_Osc'] < -80, primary_df['Unified_Osc'] < -40],
        ['OB Extreme', 'OB', 'OS Extreme', 'OS'],
        default='Neutral'
    )
    
    return primary_df


def compute_permutation_entropy(series, window=50):
    """
    Compute rolling Permutation Entropy (order-3 patterns).
    Measures market disorder: 0 = ordered, 1 = chaotic.
    """
    if len(series) < window + 3:
        return pd.Series(0.5, index=series.index)
    
    # Differential patterns
    d = series.diff()
    # Patterns of 3 consecutive moves
    # p1: UP UP, p2: UP DOWN (but newest > middle?), etc.
    # We follow the Pine Script logic precisely:
    # d_oldest = close[2]-close[3], d_middle = close[1]-close[2], d_newest = close[0]-close[1]
    
    dm = d.shift(1)
    do = d.shift(2)
    
    p1 = (do < dm) & (dm < d)
    p2 = (do < d) & (d < dm)
    p3 = (dm < do) & (do < d)
    p4 = (dm < d) & (d < do)
    p5 = (d < do) & (do < dm)
    p6 = (d < dm) & (dm < do)
    
    freqs = []
    for p in [p1, p2, p3, p4, p5, p6]:
        freqs.append(p.astype(float).rolling(window).mean())
    
    # Entropy calculation: -sum(p * log(p))
    def safe_log(x):
        return np.log(x.where(x > 1e-10, np.nan))
    
    h_raw = 0
    for f in freqs:
        h_raw -= f * safe_log(f).fillna(0)
        
    h_max = np.log(6)
    entropy_norm = h_raw / h_max
    return entropy_norm.fillna(0.5)

# ══════════════════════════════════════════════════════════════════════════════
# DIVERGENCE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def calculate_divergences(df, window=5):
    """
    Detect Regular and Hidden Divergences between Price and Unified_Osc.
    """
    df = df.copy()
    close = df['Close']
    osc = df['Unified_Osc']
    
    # Simple Fractal/Pivot point detection
    def is_pivot_high(s):
        return (s.shift(2) < s.shift(1)) & (s.shift(1) > s)
    def is_pivot_low(s):
        return (s.shift(2) > s.shift(1)) & (s.shift(1) < s)

    # Note: These are lagged by 1 bar as we need to see the 'turn'
    price_ph = is_pivot_high(close)
    price_pl = is_pivot_low(close)
    osc_ph = is_pivot_high(osc)
    osc_pl = is_pivot_low(osc)
    
    # Basic Divergence (Price making higher high, Osc making lower high)
    # This is a simplified version; real divergence needs previous pivot comparison
    # We'll tag bars where current pivot vs last pivot shows divergence
    
    df['Div_Bull'] = False
    df['Div_Bear'] = False
    
    # We'll just flag fractal points for UI highlighting for now
    df['Pivot_High'] = price_ph
    df['Pivot_Low'] = price_pl
    
    return df

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


def run_full_analysis(df_daily, timeframe, params=None, macro_data=None):
    """
    Centralized orchestration entry point. 
    Integrates time-series preprocessing with the UMA v6 Intelligence Engine 
    to generate authoritative multi-asset signal matrices.
    """
    return run_uma_v6_analysis(df_daily, timeframe, macro_data, params)

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
        self.emission_means = np.array([0.6, 0.0, -0.6])
        self.emission_stds = np.array([0.3, 0.25, 0.3])
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
            updated = np.array([0.33, 0.34, 0.33])
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
        self.positive_cusum = max(0, self.positive_cusum + z - self.drift)
        self.negative_cusum = max(0, self.negative_cusum - z - self.drift)
        
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
    Orchestrates the regime intelligence layer using HMM, GARCH, and CUSUM models.
    
    Detects structural trend shifts and volatility clusters to dynamically adjust 
    signal conviction based on the prevailing market regime.
    """
    hmm = AdaptiveHMM()
    garch = GARCHDetector()
    cusum = CUSUMDetector()
    kalman = AdaptiveKalmanFilter()
    
    regimes = []
    hmm_bulls = []
    hmm_bears = []
    vol_regimes = []
    change_points = []
    confidences = []
    signal_history = []
    
    unified_vals = df['Unified_Osc'].values
    
    for i in range(len(df)):
        sig = unified_vals[i] if not np.isnan(unified_vals[i]) else 0
        filtered = kalman.update(sig / 10.0)
        
        shock = sig - signal_history[-1] if signal_history else 0
        garch.update(shock)
        vol_regime, _ = garch.get_regime()
        
        hmm_probs = hmm.update(filtered)
        change = cusum.update(filtered)
        
        bull_p = hmm_probs['BULL']
        bear_p = hmm_probs['BEAR']
        
        if change:
            regime = "TRANSITION"
        elif bull_p > 0.6:
            regime = "BULL"
        elif bear_p > 0.6:
            regime = "BEAR"
        elif bull_p > 0.4:
            regime = "WEAK_BULL"
        elif bear_p > 0.4:
            regime = "WEAK_BEAR"
        else:
            regime = "NEUTRAL"
        
        regimes.append(regime)
        hmm_bulls.append(bull_p)
        hmm_bears.append(bear_p)
        vol_regimes.append(vol_regime)
        change_points.append(change)
        confidences.append(max(bull_p, bear_p, hmm_probs['NEUTRAL']))
        signal_history.append(sig)
    
    df['Regime'] = regimes
    df['HMM_Bull'] = hmm_bulls
    df['HMM_Bear'] = hmm_bears
    df['Vol_Regime'] = vol_regimes
    df['Change_Point'] = change_points
    df['Confidence'] = confidences
    
    return df


def calculate_divergences(df):
    """Calculate bullish and bearish divergences for WRCI signals"""
    osc_rising = df['Unified_Osc'] > df['Unified_Osc'].shift(1)
    price_falling = df['Close'] < df['Close'].shift(1)
    osc_falling = df['Unified_Osc'] < df['Unified_Osc'].shift(1)
    price_rising = df['Close'] > df['Close'].shift(1)

    df['Bullish_Div'] = osc_rising & price_falling & (df['Unified_Osc'] < -5)
    df['Bearish_Div'] = osc_falling & price_rising & (df['Unified_Osc'] > 5)

    return df


def calculate_uma_states(df):
    """
    Synthesizes momentum oscillators and macro regimes into actionable UMA States.
    
    Filters market noise by requiring regime-momentum confluence for 'Confirmed' signals 
    and identifies structural imbalances via Bullish/Bearish Divergence detection.
    """
    states = pd.Series('-', index=df.index)

    # Bullish divergence detected
    if 'Bullish_Div' in df.columns:
        states = states.where(~df['Bullish_Div'], other='Bullish Div')

    # Bearish divergence detected
    if 'Bearish_Div' in df.columns:
        states = states.where(~df['Bearish_Div'], other='Bearish Div')

    # Confirmed Bullish: MSF & MMR both bullish agreement + deep oversold
    unified = df['Unified_Osc']
    confirmed_bull = (df.get('MSF_Osc', 0) > 40) & (df.get('MMR_Osc', 0) > 40) & (unified < 10)
    states = states.where(~confirmed_bull, other='Confirmed Bullish')

    # Confirmed Bearish: MSF & MMR both bearish agreement + deep overbought
    confirmed_bear = (df.get('MSF_Osc', 0) < -40) & (df.get('MMR_Osc', 0) < -40) & (unified > -10)
    states = states.where(~confirmed_bear, other='Confirmed Bearish')

    df['UMA State'] = states
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
                ENSEMBLE INTELLIGENCE
            </h3>
            <p>The WRCI Engine synthesizes Wave Momentum with structural Trend Drift to identify high-probability institutional entry and exit points.</p>
            <div class='spec'>
                <span>Core Math:</span> WRCI Momentum Ensemble<br>
                <span>Sensitivity:</span> Multi-Timeframe Convergence<br>
                <span>Intelligence:</span> v6 Regime-Adaptive Filters<br>
                <span>Reliability:</span> Institutional Baseline
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='system-card regime'>
            <h3>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>
                STRATEGIC DIAGNOSTICS
            </h3>
            <p>Detect structural imbalances via Bullish/Bearish Divergence and confirm trend maturity using Normalized Trend Drift analysis.</p>
            <div class='spec'>
                <span>States:</span> Confirmed Bullish / Bearish<br>
                <span>Dynamics:</span> Divergence & Mean Reversion<br>
                <span>Zones:</span> 80/40/-40/-80 Extremes<br>
                <span>Clarity:</span> Entropy-Based Noise Reduction
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='system-card strategies'>
            <h3>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
                GLOBAL TAXONOMY
            </h3>
            <p>Seamlessly scan across Equities, Commodities, Currencies, and Crypto universes with localized data discovery pipelines.</p>
            <div class='spec'>
                <span>Equities:</span> Nifty 50/500 + F&O + US<br>
                <span>Macro:</span> FX + Yields + Commodities<br>
                <span>Discovery:</span> Auto-Resilient Pipelines<br>
                <span>Depth:</span> Full Time-Series Analysis
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
        elif universe == "Bond Yields":
            selected_index = "Global Bond Yields"

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
            with col_date1: start_date_hist = st.date_input("Start", datetime.date.today() - datetime.timedelta(days=300), label_visibility="collapsed")
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
                symbols_count = len(get_universe_symbols(universe, selected_index)[0] or [])
                universe_display = selected_index
            elif universe == "US Indexes" and selected_index:
                symbols_count = len(get_universe_symbols(universe, selected_index)[0] or [])
                universe_display = selected_index
            elif universe == "Commodities" and selected_index:
                symbols_count = len(get_universe_symbols(universe, selected_index)[0] or [])
                universe_display = selected_index
            elif universe == "Currency" and selected_index:
                symbols_count = len(get_universe_symbols(universe, selected_index)[0] or [])
                universe_display = selected_index
            elif universe == "Bond Yields" and selected_index:
                symbols_count = len(get_universe_symbols(universe, selected_index)[0] or [])
                universe_display = selected_index
            elif universe == "ETF Index":
                symbols_count = len(get_universe_symbols(universe, None)[0] or [])
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
    days_back = 500  # Default lookback for UMA v6 analysis

    progress_bar(progress_slot, 5, "Initializing UMA engine", f"Universe: {universe}")
    
    console.start_phase("DATA ACQUISITION", 1, 2)
    console.section("Universe Configuration")
    console.item("Universe", universe)
    console.item("Selected Index", selected_index)
    console.item("Timeframe", timeframe)

    if universe == "India Indexes":
        stock_list, msg = get_universe_symbols(universe, selected_index)
    elif universe == "US Indexes":
        stock_list, msg = get_universe_symbols(universe, selected_index)
    elif universe == "Commodities":
        stock_list, msg = get_universe_symbols(universe, None)  # Runs all commodities
    elif universe == "Currency":
        stock_list, msg = get_universe_symbols(universe, None)   # Runs all pairs
    elif universe == "Bond Yields":
        stock_list, msg = get_universe_symbols(universe, None) # Runs all yields
    elif universe == "Crypto":
        stock_list, msg = get_universe_symbols(universe, None)     # Runs all crypto
    elif universe == "ETF Index":
        stock_list, msg = get_universe_symbols(universe, None)
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
    console.start_phase("MACRO DRIVER ACQUISITION", 1, 3)
    macro_data = fetch_macro_drivers(days_back=days_back)
    console.success(f"Fetched {len(macro_data)} macro drivers for MMR engine")
    
    console.start_phase("UMA v6 ENGINE ANALYSIS", 2, 3)
    console.section("Technical & Macro Diagnostics")
    progress_bar(progress_slot, 20, "Analyzing UMA signals", f"{len(data_dict)} stocks")
    results = []

    # Define parameters (could be pulled from UI later)
    uma_params = {
        'length': reg_len,
        'rocLength': 14,
        'wt_channel_len': wt_n1,
        'wt_avg_len': wt_n2,
        'entropy_dampen_scale': 0.25,
        'hurst_influence': 0.3,
        'vol_dampen_scale': 0.15,
        'msf_weight_base': 0.5,
        'regime_sensitivity': 1.5
    }

    for i, (ticker, df) in enumerate(data_dict.items()):
        try:
            pct = int(20 + (i + 1) / len(data_dict) * 75)
            progress_bar(progress_slot, pct, f"Analyzing UMA v6", f"{i + 1}/{len(data_dict)} symbols")

            # Run UMA v6 analysis
            df = run_full_analysis(df, timeframe, params=uma_params, macro_data=macro_data)

            # Skip if not enough data
            if len(df) < reg_len + 30:
                console.detail(f"{ticker}: Skipped (Insufficient data: {len(df)} rows on {timeframe})")
                continue

            # Enrich with regime intelligence
            df = run_regime_analysis(df)
            df = calculate_divergences(df)
            df = calculate_uma_states(df)

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
            # Defensive indexing for short dataframes
            data_len = len(df)
            hist_depth = 6 # Need today + 5 back
            
            if data_len < hist_depth:
                # Pad with empty rows if necessary
                pad_len = hist_depth - data_len
                pad_df = pd.DataFrame(index=[None]*pad_len, columns=df.columns).fillna(False)
                sample_range = pd.concat([pad_df, df]).tail(hist_depth)
            else:
                sample_range = df.iloc[max(0, idx_pos - (hist_depth - 1)) : idx_pos + 1]

            last_row = df.iloc[idx_pos]

            # Build Signal String from independent logics
            signal_type = "Neutral"
            if last_row['long_cond_threshold']:
                signal_type = "Threshold Long"
            elif last_row['short_cond_threshold']:
                signal_type = "Threshold Short"
            elif last_row['long_cond_crossover']:
                signal_type = "Crossover Long"
            elif last_row['short_cond_crossover']:
                signal_type = "Crossover Short"
            elif last_row['Condition'] != 'Neutral':
                signal_type = last_row['Condition']
            
            # Additional flags for easier filtering
            long_thresh = last_row['long_cond_threshold']
            short_thresh = last_row['short_cond_threshold']
            long_cross = last_row['long_cond_crossover']
            short_cross = last_row['short_cond_crossover']

            # Clean display names
            simple_name = ticker.replace(".NS", "").lstrip("^")
            friendly_name = ASSET_NAME_LOOKUP.get(ticker)
            display_name = f"{ticker} ({friendly_name})" if friendly_name else simple_name

            results.append({
                "Symbol": ticker,
                "DisplayName": display_name,
                "SimpleName": simple_name,
                "Signal": round(last_row['Unified_Osc'], 2),
                "Trend": round(last_row['Norm_Trend'], 2),
                "Wave": round(last_row['WT1'], 2),
                "Zone": last_row['Condition'],
                "UMA State": last_row.get('UMA State', ''),
                "Entropy": round(last_row['Entropy'], 3),
                "Hurst": round(last_row['Hurst'], 3),
                "VolStress": round(last_row['Vol_Stress'], 2),
                "MMR_Qual": round(last_row['MMR_Quality'], 2),
                "MSF_Weight": round(last_row['MSF_Weight'], 2),
                "MSF_Osc": round(last_row.get('MSF_Osc', 0), 2),
                "MMR_Osc": round(last_row.get('MMR_Osc', 0), 2),
                "SignalType": signal_type,
                "Price": round(last_row['Close'], 2),
                # Flags
                "LongSignal_Thresh": long_thresh,
                "ShortSignal_Thresh": short_thresh,
                "LongSignal_Cross": long_cross,
                "ShortSignal_Cross": short_cross,
                # Historical Long Signals — Set A (Threshold)
                "L_Thresh_Today": "●" if sample_range.iloc[-1]['long_cond_threshold'] else "—",
                "L_Thresh_1d": "●" if sample_range.iloc[-2]['long_cond_threshold'] else "—",
                "L_Thresh_2d": "●" if sample_range.iloc[-3]['long_cond_threshold'] else "—",
                "L_Thresh_3d": "●" if sample_range.iloc[-4]['long_cond_threshold'] else "—",
                "L_Thresh_5d": "●" if sample_range['long_cond_threshold'].any() else "—",
                # Historical Short Signals — Set A (Threshold)
                "S_Thresh_Today": "●" if sample_range.iloc[-1]['short_cond_threshold'] else "—",
                "S_Thresh_1d": "●" if sample_range.iloc[-2]['short_cond_threshold'] else "—",
                "S_Thresh_2d": "●" if sample_range.iloc[-3]['short_cond_threshold'] else "—",
                "S_Thresh_3d": "●" if sample_range.iloc[-4]['short_cond_threshold'] else "—",
                "S_Thresh_5d": "●" if sample_range.iloc[: idx_pos + 1].tail(5)['short_cond_threshold'].any() else "—",
                # Historical Long Signals — Set B (Crossover)
                "L_Comp_Today": "●" if sample_range.iloc[-1]['long_cond_crossover'] else "—",
                "L_Comp_1d": "●" if sample_range.iloc[-2]['long_cond_crossover'] else "—",
                "L_Comp_2d": "●" if sample_range.iloc[-3]['long_cond_crossover'] else "—",
                "L_Comp_3d": "●" if sample_range.iloc[-4]['long_cond_crossover'] else "—",
                "L_Comp_5d": "●" if sample_range.iloc[: idx_pos + 1].tail(5)['long_cond_crossover'].any() else "—",
                # Historical Short Signals — Set B (Crossover)
                "S_Comp_Today": "●" if sample_range.iloc[-1]['short_cond_crossover'] else "—",
                "S_Comp_1d": "●" if sample_range.iloc[-2]['short_cond_crossover'] else "—",
                "S_Comp_2d": "●" if sample_range.iloc[-3]['short_cond_crossover'] else "—",
                "S_Comp_3d": "●" if sample_range.iloc[-4]['short_cond_crossover'] else "—",
                "S_Comp_5d": "●" if sample_range.iloc[: idx_pos + 1].tail(5)['short_cond_crossover'].any() else "—",
                # Signal flags (for filtering)
                "LongSignal_Thresh": last_row.get('long_cond_threshold', False),
                "ShortSignal_Thresh": last_row.get('short_cond_threshold', False),
                "LongSignal_Comp": last_row.get('long_cond_crossover', False),
                "ShortSignal_Comp": last_row.get('short_cond_crossover', False),
                # Additional fields for detail cards
                "Osc_Value": round(last_row.get('Unified_Osc', 0), 2),
                "MA_Alignment": 0,  # Placeholder — not computed
                "ZScore_Value": 0,  # Placeholder — not computed
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
        expected_cols = [
            "Symbol", "DisplayName", "SimpleName", "Signal", "Trend", "Wave", "Zone",
            "Conviction", "SignalType", "Price",
            "Entropy", "Hurst", "VolStress", "MMR_Qual", "MSF_Weight",
            "L_Thresh_Today", "L_Thresh_1d", "L_Thresh_2d", "L_Thresh_3d", "L_Thresh_5d",
            "S_Thresh_Today", "S_Thresh_1d", "S_Thresh_2d", "S_Thresh_3d", "S_Thresh_5d",
            "L_Comp_Today", "L_Comp_1d", "L_Comp_2d", "L_Comp_3d", "L_Comp_5d",
            "S_Comp_Today", "S_Comp_1d", "S_Comp_2d", "S_Comp_3d", "S_Comp_5d",
            "LongSignal_Thresh", "ShortSignal_Thresh", "LongSignal_Comp", "ShortSignal_Comp",
            "Osc_Value", "MA_Alignment", "ZScore_Value"
        ]
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

    console.start_phase("HISTORICAL ACQUISITION", 1, 3)
    console.section("Range Configuration")
    console.item("Universe", universe)
    console.item("Selected Index", selected_index)
    console.item("Start Date", start_date)
    console.item("End Date", end_date)
    console.item("Timeframe", timeframe)

    if universe == "India Indexes":
        stock_list, _ = get_universe_symbols(universe, selected_index)
    elif universe == "US Indexes":
        stock_list, _ = get_universe_symbols(universe, selected_index)
    elif universe == "Commodities":
        stock_list, _ = get_universe_symbols(universe, None)
    elif universe == "Currency":
        stock_list, _ = get_universe_symbols(universe, None)
    elif universe == "Bond Yields":
        stock_list, _ = get_universe_symbols(universe, None)
    elif universe == "Crypto":
        stock_list, _ = get_universe_symbols(universe, None)
    elif universe == "ETF Index":
        stock_list, _ = get_universe_symbols(universe, None)
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
    
    console.start_phase("MACRO HISTORICAL ACQUISITION", 2, 3)
    # Fetch macro data for the entire range
    macro_data = fetch_macro_drivers(days_back=500)
    console.success(f"Fetched {len(macro_data)} macro drivers for MMR history")
    console.end_phase("MACRO HISTORICAL ACQUISITION")

    console.start_phase("WRCI RANGE ANALYSIS", 3, 3)

    progress_bar(progress_slot, 15, "Processing UMA v6 + Regime Intelligence", f"{len(data_dict)} symbols")
    all_results = []
    
    if not data_dict:
        console.error("No valid market data retrieved for the selected universe.")
        st.error("No valid market data retrieved for selected universe/index.")
        return

    uma_params = {
        'length': reg_len,
        'rocLength': 14,
        'wt_channel_len': wt_n1,
        'wt_avg_len': wt_n2,
        'regime_sensitivity': 1.5,
        'entropy_dampen_scale': 0.25,
        'hurst_influence': 0.3,
        'vol_dampen_scale': 0.15,
        'msf_weight_base': 0.5
    }

    for i, (ticker, df) in enumerate(data_dict.items()):
        try:
            pct = int(15 + (i + 1) / len(data_dict) * 70)
            progress_bar(progress_slot, pct, f"Analyzing UMA v6", f"{i + 1}/{len(data_dict)} symbols")
            # Run UMA v6 analysis
            df = run_full_analysis(df, timeframe, params=uma_params, macro_data=macro_data)

            if len(df) < reg_len + 30:
                console.detail(f"{ticker}: Skipped (Insufficient data: {len(df)} rows on {timeframe})")
                continue

            # Apply Regime Intelligence & UMA States
            df = run_regime_analysis(df)
            df = calculate_divergences(df)
            df = calculate_uma_states(df)

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
                    'LongSignal': row['long_cond_threshold'] or row['long_cond_crossover'],
                    'ShortSignal': row['short_cond_threshold'] or row['short_cond_crossover'],
                    # Regime Intelligence columns
                    'Regime': row.get('Regime', 'NEUTRAL'),
                    'HMM_Bull': row.get('HMM_Bull', 0),
                    'HMM_Bear': row.get('HMM_Bear', 0),
                    'Vol_Regime': row.get('Vol_Regime', 'NORMAL'),
                    'Change_Point': row.get('Change_Point', False),
                    'Confidence': row.get('Confidence', 0),
                    'Bullish_Div': row.get('Bullish_Div', False),
                    'Bearish_Div': row.get('Bearish_Div', False),
                    'UMA State': row.get('UMA State', ''),
                })
            
            console.detail(f"[{i+1}/{len(data_dict)}] {ticker}: {len(range_df)} data points processed")
            
            # Memory Management: Free up the large analytical dataframe
            del df
            if i % 20 == 0:
                gc.collect()
            
        except Exception as e:
            console.failure(f"Range Analysis Failed: {ticker}", str(e))
            continue

    console.end_phase("WRCI RANGE ANALYSIS")

    progress_slot.empty()
    if not all_results:
        st.error("No results generated for the selected timeframe.")
        return

    ts_df = pd.DataFrame(all_results)
    ts_df['Date'] = pd.to_datetime(ts_df['Date'])
    ts_df = ts_df.sort_values('Date')

    # Aggregate daily metrics - WRCI + Regime Intelligence
    daily_agg = ts_df.groupby('Date').agg({
        'Signal': 'mean',
        'Trend': 'mean',
        'Wave': 'mean',
        'LongSignal': 'sum',
        'ShortSignal': 'sum',
        'Zone': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Neutral',
        # Regime aggregations
        'Regime': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'NEUTRAL',
        'HMM_Bull': 'mean',
        'HMM_Bear': 'mean',
        'Vol_Regime': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'NORMAL',
        'Change_Point': 'sum',
        'Confidence': 'mean',
        'Bullish_Div': 'sum',
        'Bearish_Div': 'sum',
    })

    daily_agg['TotalSignals'] = daily_agg['LongSignal'] + daily_agg['ShortSignal']
    daily_agg['L_S_Ratio'] = daily_agg['LongSignal'] / (daily_agg['ShortSignal'] + 0.01)
    daily_agg['Conviction'] = daily_agg['Signal'].abs()

    # Compute zone percentages
    ob_counts = ts_df.groupby('Date')['Zone'].apply(lambda x: (x.isin(['OB Extreme', 'OB'])).sum())
    os_counts = ts_df.groupby('Date')['Zone'].apply(lambda x: (x.isin(['OS Extreme', 'OS'])).sum())
    total_per_day = ts_df.groupby('Date').size()
    daily_agg['Oversold_Pct'] = (os_counts / total_per_day * 100).fillna(0)
    daily_agg['Overbought_Pct'] = (ob_counts / total_per_day * 100).fillna(0)

    # Compute regime percentages
    regime_bull = ts_df.groupby('Date')['Regime'].apply(lambda x: x.str.contains('BULL', na=False).sum())
    regime_bear = ts_df.groupby('Date')['Regime'].apply(lambda x: x.str.contains('BEAR', na=False).sum())
    regime_trans = ts_df.groupby('Date')['Regime'].apply(lambda x: (x == 'TRANSITION').sum())
    daily_agg['Regime_Bull_Pct'] = (regime_bull / total_per_day * 100).fillna(0)
    daily_agg['Regime_Bear_Pct'] = (regime_bear / total_per_day * 100).fillna(0)
    daily_agg['Regime_Transition_Pct'] = (regime_trans / total_per_day * 100).fillna(0)

    # Summary metrics
    total_signals = daily_agg['TotalSignals'].sum()
    avg_signal = daily_agg['Signal'].mean()
    overall_ratio = daily_agg['LongSignal'].sum() / max(daily_agg['ShortSignal'].sum(), 1)
    most_common_zone = ts_df['Zone'].mode()[0] if len(ts_df['Zone'].mode()) > 0 else 'Neutral'
    dominant_regime = ts_df['Regime'].mode()[0] if len(ts_df['Regime'].mode()) > 0 else 'NEUTRAL'
    
    avg_oversold = daily_agg['Oversold_Pct'].mean()
    avg_overbought = daily_agg['Overbought_Pct'].mean()
    total_buys = int(daily_agg['LongSignal'].sum())
    total_sells = int(daily_agg['ShortSignal'].sum())
    avg_bull_regime = daily_agg['Regime_Bull_Pct'].mean()
    avg_bear_regime = daily_agg['Regime_Bear_Pct'].mean()
    total_change_points = int(daily_agg['Change_Point'].sum())

    console.summary("RANGE STUDY SUMMARY", {
        "Universe": universe,
        "Universe Index": selected_index,
        "Range Study": f"{start_date} to {end_date}",
        "Total Signals Generated": int(total_signals),
        "Avg Signal Strength": round(avg_signal, 2),
        "Bias Ratio (L/S)": round(overall_ratio, 2),
        "Dominant Zone": most_common_zone,
        "HMM Regime": dominant_regime,
        "Status": "COMPLETE"
    })
    console.line('═', 70)

    progress_bar(progress_slot, 100, "Range study complete", f"{int(total_signals)} signals analyzed")
    progress_slot.empty()
    st.session_state["timeseries_done"] = True

    ui.render_section_header(f"Range Study ({start_date} to {end_date})", icon="history", accent="violet")

    # Summary metric cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        ui.render_metric_card("Total Signals", str(int(total_signals)), f"{total_buys} long · {total_sells} short", "info")
    with c2:
        ui.render_metric_card("Avg Oversold", f"{avg_oversold:.1f}%", "Daily Average", "success")
    with c3:
        ui.render_metric_card("Avg Overbought", f"{avg_overbought:.1f}%", "Daily Average", "danger")
    with c4:
        ui.render_metric_card("Period Regime", dominant_regime, f"Bull: {avg_bull_regime:.0f}% | Bear: {avg_bear_regime:.0f}%", "warning")
    with c5:
        ui.render_metric_card("L/S Ratio", f"{overall_ratio:.2f}", f"{'Bullish' if overall_ratio > 1 else 'Bearish'} bias", "info")
    with c6:
        ui.render_metric_card("Trading Days", str(len(daily_agg)), "Analyzed", "neutral")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Create 4 tabs like NIRNAY
    tab1, tab2, tab3, tab4 = st.tabs([
        "Signal Dashboard",
        "Transaction Dynamics", 
        "Regime Analysis",
        "Data Terminal"
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1: SIGNAL DASHBOARD
    # ═══════════════════════════════════════════════════════════════════
    with tab1:
        ui.render_section_header("Extreme Signal Trends", "Overbought / Oversold Distribution Over Time", icon="activity", accent="cyan")
        
        fig_zones = go.Figure()
        fig_zones.add_trace(go.Scatter(
            x=daily_agg.index, y=daily_agg['Oversold_Pct'],
            mode='lines', name='Oversold %',
            fill='tozeroy', fillcolor='rgba(52,211,153,0.12)',
            line=dict(color='#2DD4A8', width=2)
        ))
        fig_zones.add_trace(go.Scatter(
            x=daily_agg.index, y=daily_agg['Overbought_Pct'],
            mode='lines', name='Overbought %',
            fill='tozeroy', fillcolor='rgba(251,113,133,0.12)',
            line=dict(color='#E8555A', width=2)
        ))
        ymax = max(daily_agg['Oversold_Pct'].max(), daily_agg['Overbought_Pct'].max()) * 1.15
        fig_zones.update_layout(title='', height=350, hovermode='x unified', yaxis=dict(range=[0, ymax]))
        apply_chart_theme(fig_zones)
        st.plotly_chart(fig_zones, width='stretch')

        st.markdown("<br>", unsafe_allow_html=True)
        ui.render_section_header("Signal Volume Trends", "Raw Counts Over Time", icon="bar-chart", accent="info")
        
        fig_counts = go.Figure()
        fig_counts.add_trace(go.Bar(
            x=daily_agg.index, y=daily_agg['LongSignal'],
            name='Oversold', 
            marker=dict(color='#2DD4A8', line=dict(color='#2DD4A8', width=1))
        ))
        fig_counts.add_trace(go.Bar(
            x=daily_agg.index, y=daily_agg['ShortSignal'],
            name='Overbought', 
            marker=dict(color='#E8555A', line=dict(color='#E8555A', width=1))
        ))
        fig_counts.update_layout(title='', height=300, hovermode='x unified', barmode='group')
        apply_chart_theme(fig_counts)
        st.plotly_chart(fig_counts, width='stretch')

    # ════════════════════════════════════���═���════════════════════════════════════
    # TAB 2: TRANSACTION DYNAMICS
    # ═══════════════════════════════════════════════════════════════════════════
    with tab2:
        ui.render_section_header("Transaction Signal Trends", "Buy / Sell Signal Counts Over Time", icon="zap", accent="emerald")
        
        fig_signals = go.Figure()
        fig_signals.add_trace(go.Scatter(
            x=daily_agg.index, y=daily_agg['LongSignal'],
            mode='lines+markers', name='Long Signals',
            line=dict(color='#2DD4A8', width=2),
            marker=dict(size=6, color='#2DD4A8')
        ))
        fig_signals.add_trace(go.Scatter(
            x=daily_agg.index, y=daily_agg['ShortSignal'],
            mode='lines+markers', name='Short Signals',
            line=dict(color='#E8555A', width=2),
            marker=dict(size=6, color='#E8555A')
        ))
        fig_signals.update_layout(title='', height=300, hovermode='x unified')
        apply_chart_theme(fig_signals)
        st.plotly_chart(fig_signals, width='stretch')

        st.markdown("<br>", unsafe_allow_html=True)
        ui.render_section_header("Divergence Persistence", "Divergence Signals Over Time", icon="trending-up", accent="amber")
        
        fig_div = go.Figure()
        fig_div.add_trace(go.Bar(
            x=daily_agg.index, y=daily_agg['Bullish_Div'],
            name='Bullish Divergence', 
            marker=dict(color='#D4A853', line=dict(color='#D4A853', width=1))
        ))
        fig_div.add_trace(go.Bar(
            x=daily_agg.index, y=-daily_agg['Bearish_Div'],
            name='Bearish Divergence', 
            marker=dict(color='#06B6D4', line=dict(color='#06B6D4', width=1))
        ))
        fig_div.update_layout(title='', height=300, hovermode='x unified', barmode='relative')
        apply_chart_theme(fig_div)
        st.plotly_chart(fig_div, width='stretch')

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3: REGIME ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    with tab3:
        ui.render_section_header("Aggregate Signal Momentum", "Average Signal Value Over Time", icon="activity", accent="rose")
        
        colors = ['#2DD4A8' if v < -20 else '#E8555A' if v > 20 else '#64748B' for v in daily_agg['Signal']]
        
        fig_avg = go.Figure()
        fig_avg.add_trace(go.Scatter(
            x=daily_agg.index, y=daily_agg['Signal'].clip(lower=0),
            fill='tozeroy', fillcolor='rgba(232,85,90,0.05)',
            line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig_avg.add_trace(go.Scatter(
            x=daily_agg.index, y=daily_agg['Signal'].clip(upper=0),
            fill='tozeroy', fillcolor='rgba(45,212,168,0.05)',
            line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig_avg.add_trace(go.Scatter(
            x=daily_agg.index, y=daily_agg['Signal'],
            mode='lines+markers', name='Avg Signal',
            line=dict(color='#D4A853', width=2),
            marker=dict(size=6, color=colors)
        ))
        fig_avg.add_hline(y=20, line=dict(color='rgba(239,68,68,0.5)', width=1, dash='dash'))
        fig_avg.add_hline(y=-20, line=dict(color='rgba(16,185,129,0.5)', width=1, dash='dash'))
        fig_avg.add_hline(y=0, line=dict(color='rgba(255,255,255,0.3)', width=1))
        fig_avg.update_layout(title='', height=300, hovermode='x unified', yaxis=dict(range=[-80, 80]))
        apply_chart_theme(fig_avg)
        st.plotly_chart(fig_avg, width='stretch')

        st.markdown("<br>", unsafe_allow_html=True)
        ui.render_section_header("HMM Regime Distribution Over Time", "Percentage of symbols in each HMM regime daily", icon="activity", accent="cyan")
        
        fig_regime = go.Figure()
        fig_regime.add_trace(go.Scatter(
            x=daily_agg.index, y=daily_agg['Regime_Bull_Pct'],
            mode='lines', name='Bull Regime %',
            fill='tozeroy', fillcolor='rgba(52,211,153,0.12)',
            line=dict(color='#2DD4A8', width=2)
        ))
        fig_regime.add_trace(go.Scatter(
            x=daily_agg.index, y=daily_agg['Regime_Bear_Pct'],
            mode='lines', name='Bear Regime %',
            fill='tozeroy', fillcolor='rgba(232,85,90,0.12)',
            line=dict(color='#E8555A', width=2)
        ))
        fig_regime.update_layout(title='', height=300, hovermode='x unified', yaxis=dict(range=[0, 100]))
        apply_chart_theme(fig_regime)
        st.plotly_chart(fig_regime, width='stretch')

        st.markdown("<br>", unsafe_allow_html=True)
        ui.render_section_header("Volatility Dynamics", "Volatility Regime & Change Points Over Time", icon="shield", accent="amber")
        
        # Compute high vol percentage
        vol_high = ts_df.groupby('Date')['Vol_Regime'].apply(lambda x: (x.isin(['HIGH', 'EXTREME'])).sum() / len(x) * 100)
        
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=daily_agg.index, y=vol_high.fillna(0),
            mode='lines+markers', name='High Vol %',
            line=dict(color='#D4A853', width=2),
            marker=dict(size=5)
        ))
        fig_vol.add_trace(go.Bar(
            x=daily_agg.index, y=daily_agg['Change_Point'],
            name='Change Points',
            marker=dict(color='#A855F7', opacity=0.7)
        ))
        fig_vol.update_layout(title='', height=250, hovermode='x unified')
        apply_chart_theme(fig_vol)
        st.plotly_chart(fig_vol, width='stretch')

        st.markdown("<br>", unsafe_allow_html=True)
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            ui.render_section_header("State Transition Metrics", "HMM Regime Statistics", icon="bar-chart", accent="emerald")
            regime_stats = {
                "Metric": ["Avg Bull Regime %", "Avg Bear Regime %", "Total Change Points", "Avg High Vol %"],
                "Value": [f"{avg_bull_regime:.1f}%", f"{avg_bear_regime:.1f}%", f"{total_change_points}", f"{vol_high.mean():.1f}%"]
            }
            st.dataframe(pd.DataFrame(regime_stats), width='stretch', hide_index=True)
        
        with col_r2:
            ui.render_section_header("Distribution Metrics", "Signal Statistics", icon="database", accent="rose")
            signal_stats = {
                "Metric": ["Mean Signal", "Median Signal", "Min Signal", "Max Signal", "Std Dev"],
                "Value": [
                    f"{daily_agg['Signal'].mean():.2f}",
                    f"{daily_agg['Signal'].median():.2f}",
                    f"{daily_agg['Signal'].min():.2f}",
                    f"{daily_agg['Signal'].max():.2f}",
                    f"{daily_agg['Signal'].std():.2f}"
                ]
            }
            st.dataframe(pd.DataFrame(signal_stats), width='stretch', hide_index=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 4: DATA TERMINAL
    # ═══════════════════════════════════════════════════════════════════════════
    with tab4:
        ui.render_section_header("Analytical Data", f"Daily Time Series ({len(daily_agg)} days)", icon="list", accent="cyan")
        
        display_ts = daily_agg.copy()
        display_ts.index = display_ts.index.strftime('%Y-%m-%d')
        display_ts = display_ts.reset_index().rename(columns={'Date': 'Date'})
        
        # Select columns to display
        display_cols = ['Date', 'LongSignal', 'ShortSignal', 'Signal', 'Oversold_Pct', 'Overbought_Pct', 
                      'Regime_Bull_Pct', 'Regime_Bear_Pct', 'Change_Point']
        display_ts = display_ts[display_cols]
        display_ts.columns = ['Date', 'Long Sig', 'Short Sig', 'Avg Signal', 'Oversold %', 'Overbought %',
                           'Bull Regime %', 'Bear Regime %', 'Change Pts']
        
        st.dataframe(display_ts, width='stretch', hide_index=True, height=500)
        
        st.markdown("<br>", unsafe_allow_html=True)
        csv_data = ts_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Report (CSV)",
            data=csv_data,
            file_name=f"sanket_range_study_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    render_footer()


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS FOR TAB RENDERING
# ══════════════════════════════════════════════════════════════════════════════

def _bucket_signals_by_age(results_df: pd.DataFrame, side: str = 'long', prefix: str = '') -> dict:
    """Bucket signals by age (Today, 1d, 2d, 3d, 5d) with stats for timeline display.

    Args:
        results_df: DataFrame containing signal data
        side: 'long' or 'short'
        prefix: Column prefix — 'Thresh' for Threshold set, 'Comp' for Crossover set
    """
    base_prefix = 'L' if side == 'long' else 'S'
    full_prefix = f"{base_prefix}_{prefix}" if prefix else base_prefix
    target_indicator = "●"
    buckets = {
        "Today": [],
        "1 Day Ago": [],
        "2 Days Ago": [],
        "3 Days Ago": [],
        "Within 5 Days": []
    }
    col_map = {
        "Today": f"{full_prefix}_Today",
        "1 Day Ago": f"{full_prefix}_1d",
        "2 Days Ago": f"{full_prefix}_2d",
        "3 Days Ago": f"{full_prefix}_3d",
        "Within 5 Days": f"{full_prefix}_5d"
    }
    seen = set()

    for age in buckets.keys():
        col = col_map[age]
        # Gracefully handle missing columns (fallback to all-False)
        if col not in results_df.columns:
            continue
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
            uma_state = html_module.escape(str(row.get('UMA State') or '—'))
            zone = html_module.escape(str(row.get('Zone', '—')))

            table_rows.append(f"""
            <tr>
                <td class="symbol">{symbol}</td>
                <td class="numeric currency">₹{price:,.2f}</td>
                <td class="numeric" style="color: {accent_light}; font-weight: 600;">{signal:+.2f}</td>
                <td class="numeric" style="color: {accent_light}; font-weight: 600;">{trend:+.2f}</td>
                <td class="numeric">{uma_state}</td>
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
             min-width: 600px;
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
                      <th class="numeric">UMA State</th>
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
        uma_state = html_module.escape(str(row.get('UMA State') or '—'))
        zone = html_module.escape(str(row.get('Zone', '—')))

        rank_str = f"{idx:02d}"

        table_rows.append(f"""
        <tr>
            <td class="numeric" style="color: #D4A853; font-weight: 700;">{rank_str}</td>
            <td class="symbol">{symbol}</td>
            <td class="numeric currency">₹{price:,.2f}</td>
            <td class="numeric" style="color: {accent_light}; font-weight: 600;">{signal:+.2f}</td>
            <td class="numeric" style="color: {accent_light}; font-weight: 600;">{trend:+.2f}</td>
            <td class="numeric">{uma_state}</td>
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
             min-width: 720px;
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
                    <th class="numeric">UMA State</th>
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

                # Split into longs and shorts — include symbols from either signal set
                longs_df = results_df[(results_df['L_Thresh_5d'] != "—") | (results_df['L_Comp_5d'] != "—")].copy().sort_values('Signal', ascending=False)
                shorts_df = results_df[(results_df['S_Thresh_5d'] != "—") | (results_df['S_Comp_5d'] != "—")].copy().sort_values('Signal', ascending=True)

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

                    # ═══════════════════════════════════════════════════════════════════════════
                    # BULLISH & BEARISH SIGNALS BY TIMING — DUAL-SIGNAL SET TABS
                    # ═══════════════════════════════════════════════════════════════════════════
                    # Main tabs: Bullish | Bearish
                    # Each contains nested tabs: Threshold (Set A) | Crossover (Set B)
                    bull_tab, bear_tab = st.tabs(["Bullish Signals by Timing", "Bearish Signals by Timing"])

                    _age_order = ["Today", "1 Day Ago", "2 Days Ago", "3 Days Ago", "Within 5 Days"]

                    with bull_tab:
                        if not longs_df.empty:
                            # Nested tabs for signal sets
                            inner_thresh, inner_comp = st.tabs(["Threshold", "Crossover"])

                            with inner_thresh:
                                _, stats_thresh, _, _ = _bucket_signals_by_age(longs_df, side='long', prefix='Thresh')
                                if any(s['count'] > 0 for s in stats_thresh.values()):
                                    html_thresh = _build_signal_table_html(stats_thresh, side='long')
                                    _groups = sum(1 for a in _age_order if stats_thresh[a]['count'] > 0)
                                    _rows = sum(stats_thresh[a]['count'] for a in _age_order)
                                    st.components.v1.html(html_thresh, height=70 + _groups * 46 + _rows * 44)
                                else:
                                    st.info("No Threshold long signals in this universe.")

                            with inner_comp:
                                _, stats_comp, _, _ = _bucket_signals_by_age(longs_df, side='long', prefix='Comp')
                                if any(s['count'] > 0 for s in stats_comp.values()):
                                    html_comp = _build_signal_table_html(stats_comp, side='long')
                                    _groups = sum(1 for a in _age_order if stats_comp[a]['count'] > 0)
                                    _rows = sum(stats_comp[a]['count'] for a in _age_order)
                                    st.components.v1.html(html_comp, height=70 + _groups * 46 + _rows * 44)
                                else:
                                    st.info("No Crossover long signals in this universe.")
                        else:
                            st.info("No bullish signals detected in either set.")
                        _render_signal_legend(side='long')

                    with bear_tab:
                        if not shorts_df.empty:
                            # Nested tabs for signal sets
                            inner_thresh, inner_comp = st.tabs(["Threshold", "Crossover"])

                            with inner_thresh:
                                _, stats_thresh, _, _ = _bucket_signals_by_age(shorts_df, side='short', prefix='Thresh')
                                if any(s['count'] > 0 for s in stats_thresh.values()):
                                    html_thresh = _build_signal_table_html(stats_thresh, side='short')
                                    _groups = sum(1 for a in _age_order if stats_thresh[a]['count'] > 0)
                                    _rows = sum(stats_thresh[a]['count'] for a in _age_order)
                                    st.components.v1.html(html_thresh, height=70 + _groups * 46 + _rows * 44)
                                else:
                                    st.info("No Threshold short signals in this universe.")

                            with inner_comp:
                                _, stats_comp, _, _ = _bucket_signals_by_age(shorts_df, side='short', prefix='Comp')
                                if any(s['count'] > 0 for s in stats_comp.values()):
                                    html_comp = _build_signal_table_html(stats_comp, side='short')
                                    _groups = sum(1 for a in _age_order if stats_comp[a]['count'] > 0)
                                    _rows = sum(stats_comp[a]['count'] for a in _age_order)
                                    st.components.v1.html(html_comp, height=70 + _groups * 46 + _rows * 44)
                                else:
                                    st.info("No Crossover short signals in this universe.")
                        else:
                            st.info("No bearish signals detected in either set.")
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
                avg_signal_str = results_df['Signal'].abs().mean() if not results_df.empty else 0.0
                avg_trend_str = results_df['Trend'].abs().mean() if not results_df.empty else 0.0
                strong_trend_count = len(results_df[results_df['Trend'].abs() > 30])

                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    ui.render_metric_card("Avg Signal Magnitude", f"{avg_signal_str:.1f}", "Average across all symbols", "neutral")
                with col_s2:
                    ui.render_metric_card("Avg Trend Value", f"{avg_trend_str:.1f}", "Directional strength", "neutral")
                with col_s3:
                    pct = (strong_trend_count / len(results_df) * 100) if len(results_df) > 0 else 0
                    ui.render_metric_card("Strong Trends", str(strong_trend_count), f"{pct:.0f}% of universe", "info")

                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

                # ═══════════════════════════════════════════════════════════════════════════
                # SIGNAL STRENGTH — SPLIT BY SIGNAL SET
                # ═══════════════════════════════════════════════════════════════════════════

                ui.render_section_header(
                    "Threshold Intelligence (Set A)",
                    "Composite signals crossing institutional extreme levels with signal confirmation",
                    icon="zap",
                    accent="emerald"
                )
                col_thresh_l, col_thresh_s = st.columns(2)
                with col_thresh_l:
                    st.markdown('<p style="font-family:\'IBM Plex Mono\',monospace; font-size:0.65rem; font-weight:600; color:var(--emerald-bright); text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.6rem;">Top 10 Bullish — Threshold (Active 5d)</p>', unsafe_allow_html=True)
                    top_thresh_longs = longs_df[longs_df['L_Thresh_5d'] != "—"].head(10)
                    if not top_thresh_longs.empty:
                        html_thresh_l = _build_signal_strength_table_html(top_thresh_longs, side='long')
                        st.components.v1.html(html_thresh_l, height=94 + len(top_thresh_longs) * 40)
                    else:
                        st.info("No Threshold long signals.")
                with col_thresh_s:
                    st.markdown('<p style="font-family:\'IBM Plex Mono\',monospace; font-size:0.65rem; font-weight:600; color:var(--rose-bright); text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.6rem;">Top 10 Bearish — Threshold (Active 5d)</p>', unsafe_allow_html=True)
                    top_thresh_shorts = shorts_df[shorts_df['S_Thresh_5d'] != "—"].head(10)
                    if not top_thresh_shorts.empty:
                        html_thresh_s = _build_signal_strength_table_html(top_thresh_shorts, side='short')
                        st.components.v1.html(html_thresh_s, height=94 + len(top_thresh_shorts) * 40)
                    else:
                        st.info("No Threshold short signals.")

                st.markdown('<div class="section-divider" style="margin: 2rem 0;"></div>', unsafe_allow_html=True)

                ui.render_section_header(
                    "Crossover Intelligence (Set B)",
                    "Trend exhaustion and momentum reversal signals at extreme zones",
                    icon="activity",
                    accent="cyan"
                )
                col_comp_l, col_comp_s = st.columns(2)
                with col_comp_l:
                    st.markdown('<p style="font-family:\'IBM Plex Mono\',monospace; font-size:0.65rem; font-weight:600; color:var(--emerald-bright); text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.6rem;">Top 10 Bullish — Crossover (Active 5d)</p>', unsafe_allow_html=True)
                    top_comp_longs = longs_df[longs_df['L_Comp_5d'] != "—"].head(10)
                    if not top_comp_longs.empty:
                        html_comp_l = _build_signal_strength_table_html(top_comp_longs, side='long')
                        st.components.v1.html(html_comp_l, height=94 + len(top_comp_longs) * 40)
                    else:
                        st.info("No Crossover long signals.")
                with col_comp_s:
                    st.markdown('<p style="font-family:\'IBM Plex Mono\',monospace; font-size:0.65rem; font-weight:600; color:var(--rose-bright); text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.6rem;">Top 10 Bearish — Crossover (Active 5d)</p>', unsafe_allow_html=True)
                    top_comp_shorts = shorts_df[shorts_df['S_Comp_5d'] != "—"].head(10)
                    if not top_comp_shorts.empty:
                        html_comp_s = _build_signal_strength_table_html(top_comp_shorts, side='short')
                        st.components.v1.html(html_comp_s, height=94 + len(top_comp_shorts) * 40)
                    else:
                        st.info("No Crossover short signals.")


            # ════ TAB 4: SYSTEM DATA ════════════════════════════════════════════════════
            with tab_raw:
                ui.render_section_header(
                    "System Raw Data",
                    "Complete underlying data for analysis and model validation",
                    icon="database",
                    accent="cyan"
                )


                # Show all data with historical signals
                display_df = results_df[[
                    "DisplayName", "Price", "Signal", "Trend", "Wave", "UMA State", "Zone",
                    "Entropy", "Hurst", "VolStress", "MMR_Qual", "MSF_Weight", "MSF_Osc", "MMR_Osc",
                    # Threshold historical
                    "L_Thresh_Today", "L_Thresh_1d", "L_Thresh_2d", "L_Thresh_3d", "L_Thresh_5d",
                    "S_Thresh_Today", "S_Thresh_1d", "S_Thresh_2d", "S_Thresh_3d", "S_Thresh_5d",
                    # Crossover historical
                    "L_Comp_Today", "L_Comp_1d", "L_Comp_2d", "L_Comp_3d", "L_Comp_5d",
                    "S_Comp_Today", "S_Comp_1d", "S_Comp_2d", "S_Comp_3d", "S_Comp_5d",
                ]].sort_values("Signal", ascending=False)

                st.dataframe(display_df, width='stretch', height=500)

                # ── EXPORT SECTION ─────────────────────────────────────────────────────────
                st.markdown('<div class="section-divider" style="margin-top: 2rem;"></div>', unsafe_allow_html=True)

                ui.render_section_header(
                    "Export Signal Sets",
                    "Separate CSV downloads for each signal logic — Threshold and Crossover",
                    icon="download",
                    accent="cyan"
                )

                # ── Row 1: Threshold Set ──
                st.markdown('<p style="font-family:\'IBM Plex Mono\',monospace; font-size:0.65rem; color:#4B5563; text-transform:uppercase; letter-spacing:0.08em; margin: 0.5rem 0 0.4rem 0;">Threshold Signals (Set A)</p>', unsafe_allow_html=True)
                dl_t1, dl_t2 = st.columns(2)
                with dl_t1:
                    thresh_longs = longs_df[longs_df['LongSignal_Thresh']].copy()
                    st.download_button(
                        label="↑  Threshold Bullish",
                        data=thresh_longs.to_csv(index=False).encode('utf-8') if not thresh_longs.empty else "No signals".encode('utf-8'),
                        file_name=f"threshold_bullish_{analysis_date}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="dl_thresh_long",
                        disabled=thresh_longs.empty,
                        help="Long signals from Threshold logic (composite crosses extreme level)"
                    )
                with dl_t2:
                    thresh_shorts = shorts_df[shorts_df['ShortSignal_Thresh']].copy()
                    st.download_button(
                        label="↓  Threshold Bearish",
                        data=thresh_shorts.to_csv(index=False).encode('utf-8') if not thresh_shorts.empty else "No signals".encode('utf-8'),
                        file_name=f"threshold_bearish_{analysis_date}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="dl_thresh_short",
                        disabled=thresh_shorts.empty,
                        help="Short signals from Threshold logic (composite crosses extreme level)"
                    )

                st.markdown('<div class="section-divider" style="margin: 1rem 0;"></div>', unsafe_allow_html=True)

                # ── Row 2: Crossover Set ──
                st.markdown('<p style="font-family:\'IBM Plex Mono\',monospace; font-size:0.65rem; color:#4B5563; text-transform:uppercase; letter-spacing:0.08em; margin: 0.5rem 0 0.4rem 0;">Crossover Signals (Set B)</p>', unsafe_allow_html=True)
                dl_c1, dl_c2 = st.columns(2)
                with dl_c1:
                    comp_longs = longs_df[longs_df['LongSignal_Comp']].copy()
                    st.download_button(
                        label="↑  Crossover Bullish",
                        data=comp_longs.to_csv(index=False).encode('utf-8') if not comp_longs.empty else "No signals".encode('utf-8'),
                        file_name=f"crossover_bullish_{analysis_date}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="dl_comp_long",
                        disabled=comp_longs.empty,
                        help="Long signals from Crossover logic (composite crosses signal line in oversold)"
                    )
                with dl_c2:
                    comp_shorts = shorts_df[shorts_df['ShortSignal_Comp']].copy()
                    st.download_button(
                        label="↓  Crossover Bearish",
                        data=comp_shorts.to_csv(index=False).encode('utf-8') if not comp_shorts.empty else "No signals".encode('utf-8'),
                        file_name=f"crossover_bearish_{analysis_date}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="dl_comp_short",
                        disabled=comp_shorts.empty,
                        help="Short signals from Crossover logic (composite crosses signal line in overbought)"
                    )

            render_footer()


if __name__ == "__main__":
    main()
