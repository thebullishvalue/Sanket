"""
Sanket - Market Signal Screener | A Pragyam Product Family Member
MSF + MMR Quantitative Signal Scanner for Indian Markets
"""

import streamlit as st
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import datetime
import numpy as np
import plotly.graph_objects as go
import requests
import io
import urllib3
from nsepython import nse_get_advances_declines

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Sanket | Market Signal Screener",
    layout="wide",
    initial_sidebar_state="expanded"
)

VERSION = "v1.1.0 - MMR Engine"

# ══════════════════════════════════════════════════════════════════════════════
# PRAGYAM DESIGN SYSTEM CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary-color: #FFC300;
        --primary-rgb: 255, 195, 0;
        --background-color: #0F0F0F;
        --secondary-background-color: #1A1A1A;
        --bg-card: #1A1A1A;
        --bg-elevated: #2A2A2A;
        --text-primary: #EAEAEA;
        --text-secondary: #EAEAEA;
        --text-muted: #888888;
        --border-color: #2A2A2A;
        --border-light: #3A3A3A;
        --success-green: #10b981;
        --danger-red: #ef4444;
        --warning-amber: #f59e0b;
        --info-cyan: #06b6d4;
        --neutral: #888888;
    }
    
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .main, [data-testid="stSidebar"] { background-color: var(--background-color); color: var(--text-primary); }
    .stApp > header { background-color: transparent; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .block-container { padding-top: 3.5rem; max-width: 90%; padding-left: 2rem; padding-right: 2rem; }
    
    /* Sidebar toggle button - always visible */
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        background-color: var(--secondary-background-color) !important;
        border: 2px solid var(--primary-color) !important;
        border-radius: 8px !important;
        padding: 10px !important;
        margin: 12px !important;
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.4) !important;
        z-index: 999999 !important;
        position: fixed !important;
        top: 14px !important;
        left: 14px !important;
        width: 40px !important;
        height: 40px !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    [data-testid="collapsedControl"]:hover {
        background-color: rgba(var(--primary-rgb), 0.2) !important;
        box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.6) !important;
        transform: scale(1.05);
    }
    
    [data-testid="collapsedControl"] svg {
        stroke: var(--primary-color) !important;
        width: 20px !important;
        height: 20px !important;
    }
    
    [data-testid="stSidebar"] button[kind="header"] {
        background-color: transparent !important;
        border: none !important;
    }
    
    [data-testid="stSidebar"] button[kind="header"] svg {
        stroke: var(--primary-color) !important;
    }
    
    button[kind="header"] {
        z-index: 999999 !important;
    }
    
    .premium-header {
        background: var(--secondary-background-color);
        padding: 1.25rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.1);
        border: 1px solid var(--border-color);
        position: relative;
        overflow: hidden;
        margin-top: 1rem;
    }
    
    .premium-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 20% 50%, rgba(var(--primary-rgb),0.08) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .premium-header h1 { margin: 0; font-size: 2rem; font-weight: 700; color: var(--text-primary); letter-spacing: -0.50px; position: relative; }
    .premium-header .tagline { color: var(--text-muted); font-size: 0.9rem; margin-top: 0.25rem; font-weight: 400; position: relative; }
    .premium-header .product-badge { display: inline-block; background: rgba(var(--primary-rgb), 0.15); color: var(--primary-color); padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem; }
    
    .metric-card {
        background-color: var(--bg-card);
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08);
        margin-bottom: 0.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.3); border-color: var(--border-light); }
    .metric-card h4 { color: var(--text-muted); font-size: 0.75rem; margin-bottom: 0.5rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card h2 { color: var(--text-primary); font-size: 1.75rem; font-weight: 700; margin: 0; line-height: 1; }
    .metric-card .sub-metric { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem; font-weight: 500; }
    .metric-card.success h2 { color: var(--success-green); }
    .metric-card.danger h2 { color: var(--danger-red); }
    .metric-card.warning h2 { color: var(--warning-amber); }
    .metric-card.info h2 { color: var(--info-cyan); }
    .metric-card.neutral h2 { color: var(--neutral); }
    .metric-card.primary h2 { color: var(--primary-color); }
    
    .signal-card {
        background-color: var(--bg-card);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08);
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
    }
    
    .signal-card::before { content: ''; position: absolute; top: 0; left: 0; width: 4px; height: 100%; }
    .signal-card.buy::before { background: var(--success-green); }
    .signal-card.sell::before { background: var(--danger-red); }
    .signal-card-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem; }
    .signal-card-title { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); }
    
    .status-badge { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.4rem 0.8rem; border-radius: 20px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
    .status-badge.buy { background: rgba(16, 185, 129, 0.15); color: var(--success-green); border: 1px solid rgba(16, 185, 129, 0.3); }
    .status-badge.sell { background: rgba(239, 68, 68, 0.15); color: var(--danger-red); border: 1px solid rgba(239, 68, 68, 0.3); }
    .status-badge.oversold { background: rgba(6, 182, 212, 0.15); color: var(--info-cyan); border: 1px solid rgba(6, 182, 212, 0.3); }
    .status-badge.overbought { background: rgba(245, 158, 11, 0.15); color: var(--warning-amber); border: 1px solid rgba(245, 158, 11, 0.3); }
    .status-badge.neutral { background: rgba(136, 136, 136, 0.15); color: var(--neutral); border: 1px solid rgba(136, 136, 136, 0.3); }
    .status-badge.divergence { background: rgba(var(--primary-rgb), 0.15); color: var(--primary-color); border: 1px solid rgba(var(--primary-rgb), 0.3); }
    
    .info-box { background: var(--secondary-background-color); border: 1px solid var(--border-color); border-left: 0px solid var(--primary-color); padding: 1.25rem; border-radius: 12px; margin: 0.5rem 0; box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08); }
    .info-box h4 { color: var(--primary-color); margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 700; }
    .info-box p { color: var(--text-muted); margin: 0; font-size: 0.9rem; line-height: 1.6; }
    
    .stButton>button { border: 2px solid var(--primary-color); background: transparent; color: var(--primary-color); font-weight: 700; border-radius: 12px; padding: 0.75rem 2rem; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); text-transform: uppercase; letter-spacing: 0.5px; }
    .stButton>button:hover { box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.6); background: var(--primary-color); color: #1A1A1A; transform: translateY(-2px); }
    .stButton>button:active { transform: translateY(0); }
    
    .stTabs [data-baseweb="tab-list"] { gap: 24px; background: transparent; }
    .stTabs [data-baseweb="tab"] { color: var(--text-muted); border-bottom: 2px solid transparent; transition: color 0.3s, border-bottom 0.3s; background: transparent; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: var(--primary-color); border-bottom: 2px solid var(--primary-color); background: transparent !important; }
    
    .stPlotlyChart { border-radius: 12px; background-color: var(--secondary-background-color); padding: 10px; border: 1px solid var(--border-color); box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.1); }
    .stDataFrame { border-radius: 12px; background-color: var(--secondary-background-color); border: 1px solid var(--border-color); }
    .section-divider { height: 1px; background: linear-gradient(90deg, transparent 0%, var(--border-color) 50%, transparent 100%); margin: 1.5rem 0; }
    
    .symbol-row { display: flex; align-items: center; justify-content: space-between; padding: 0.75rem 1rem; border-radius: 8px; background: var(--bg-elevated); margin-bottom: 0.5rem; transition: all 0.2s ease; }
    .symbol-row:hover { background: var(--border-light); }
    .symbol-name { font-weight: 700; color: var(--text-primary); font-size: 0.9rem; }
    .symbol-price { color: var(--text-muted); font-size: 0.85rem; }
    .symbol-score { font-weight: 700; font-size: 0.9rem; }
    
    .sidebar-title { font-size: 0.75rem; font-weight: 700; color: var(--primary-color); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.75rem; }
    
    [data-testid="stSidebar"] { background: var(--secondary-background-color); border-right: 1px solid var(--border-color); }
    
    .stTextInput > div > div > input { background: var(--bg-elevated) !important; border: 1px solid var(--border-color) !important; border-radius: 8px !important; color: var(--text-primary) !important; }
    .stTextInput > div > div > input:focus { border-color: var(--primary-color) !important; box-shadow: 0 0 0 2px rgba(var(--primary-rgb), 0.2) !important; }
    
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--background-color); }
    ::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--border-light); }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & UNIVERSE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

INDEX_LIST = [
    "NIFTY 50", "NIFTY NEXT 50", "NIFTY 100", "NIFTY 200", "NIFTY 500",
    "NIFTY MIDCAP 50", "NIFTY MIDCAP 100", "NIFTY SMLCAP 100", "NIFTY BANK",
    "NIFTY AUTO", "NIFTY FIN SERVICE", "NIFTY FMCG", "NIFTY IT",
    "NIFTY MEDIA", "NIFTY METAL", "NIFTY PHARMA"
]

BASE_URL = "https://www.niftyindices.com/IndexConstituent/"
INDEX_URL_MAP = {
    "NIFTY 50": f"{BASE_URL}ind_nifty50list.csv",
    "NIFTY NEXT 50": f"{BASE_URL}ind_niftynext50list.csv",
    "NIFTY 100": f"{BASE_URL}ind_nifty100list.csv",
    "NIFTY 200": f"{BASE_URL}ind_nifty200list.csv",
    "NIFTY 500": f"{BASE_URL}ind_nifty500list.csv",
    "NIFTY MIDCAP 50": f"{BASE_URL}ind_niftymidcap50list.csv",
    "NIFTY MIDCAP 100": f"{BASE_URL}ind_niftymidcap100list.csv",
    "NIFTY SMLCAP 100": f"{BASE_URL}ind_niftysmallcap100list.csv",
    "NIFTY BANK": f"{BASE_URL}ind_niftybanklist.csv",
    "NIFTY AUTO": f"{BASE_URL}ind_niftyautolist.csv",
    "NIFTY FIN SERVICE": f"{BASE_URL}ind_niftyfinancelist.csv",
    "NIFTY FMCG": f"{BASE_URL}ind_niftyfmcglist.csv",
    "NIFTY IT": f"{BASE_URL}ind_niftyitlist.csv",
    "NIFTY MEDIA": f"{BASE_URL}ind_niftymedialist.csv",
    "NIFTY METAL": f"{BASE_URL}ind_niftymetallist.csv",
    "NIFTY PHARMA": f"{BASE_URL}ind_niftypharmalist.csv"
}

UNIVERSE_OPTIONS = ["F&O Stocks", "Index Constituents"]
TIMEFRAME_OPTIONS = ["Daily", "Weekly"]

# Macro symbols for MMR calculation
MACRO_SYMBOLS_STOOQ = {
    "India 10Y": "10YINY.B", "India 02Y": "2YINY.B",
    "US 30Y": "30YUSY.B", "US 10Y": "10YUSY.B", "US 05Y": "5YUSY.B", "US 02Y": "2YUSY.B",
    "UK 30Y": "30YUKY.B", "UK 10Y": "10YUKY.B", "UK 05Y": "5YUKY.B", "UK 02Y": "2YUKY.B",
    "EU (DE) 30Y": "30YDEY.B", "EU (DE) 10Y": "10YDEY.B", "EU (DE) 05Y": "5YDEY.B", "EU (DE) 02Y": "2YDEY.B",
    "China 10Y": "10YCNY.B", "China 02Y": "2YCNY.B",
    "Japan 30Y": "30YJPY.B", "Japan 10Y": "10YJPY.B", "Japan 02Y": "2YJPY.B",
    "Singapore 10Y": "10YSGY.B",
}

MACRO_SYMBOLS_YF = {
    "Dollar Index": "DX-Y.NYB", "Crude Oil": "CL=F", "Brent Crude": "BZ=F",
    "USD/INR": "INR=X", "GBP/INR": "GBPINR=X", "EUR/INR": "EURINR=X",
    "SGD/INR": "SGDINR=X", "JPY/INR": "JPYINR=X", "Gold": "GC=F", "Silver": "SI=F"
}

MACRO_SYMBOLS = {**MACRO_SYMBOLS_STOOQ, **MACRO_SYMBOLS_YF}

# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def get_fno_stock_list():
    """Fetch F&O stock list from NSE with multiple fallback methods"""
    
    # Method 1: Try NSE API directly
    try:
        url = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nseindia.com/market-data/live-equity-market?symbol=NIFTY%20FIN%20SERVICE',
        }
        
        session = requests.Session()
        # First hit the main page to get cookies
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
    
    # Method 2: Try nsepython library
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
    
    # Method 3: Fallback to NIFTY 500 as proxy (most F&O stocks are in NIFTY 500)
    try:
        url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
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


@st.cache_data(ttl=3600, show_spinner=False)
def get_index_stock_list(index):
    """Fetch index constituents from NSE Indices"""
    url = INDEX_URL_MAP.get(index)
    if not url:
        return None, f"No URL for {index}"
        
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, verify=False, timeout=10)
        response.raise_for_status()
        
        csv_file = io.StringIO(response.text)
        stock_df = pd.read_csv(csv_file)
        
        if 'Symbol' in stock_df.columns:
            symbols = stock_df['Symbol'].tolist()
            symbols_ns = [str(s) + ".NS" for s in symbols if s and str(s).strip()]
            return symbols_ns, f"✓ Fetched {len(symbols_ns)} constituents"
        else:
            return None, f"No Symbol column found"
            
    except Exception as e:
        return None, f"Error: {e}"


@st.cache_data(ttl=900, show_spinner=False)
def fetch_macro_data(days_back=100):
    """Fetch all macro data ONCE - to be reused across all stocks"""
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days_back + 365)
    
    stooq_df = pd.DataFrame()
    try:
        stooq_tickers = list(MACRO_SYMBOLS_STOOQ.values())
        stooq_raw = web.DataReader(stooq_tickers, "stooq", start=start_date, end=end_date)
        if isinstance(stooq_raw.columns, pd.MultiIndex):
            if 'Close' in stooq_raw.columns.get_level_values(0):
                stooq_df = stooq_raw['Close']
            elif 'Value' in stooq_raw.columns.get_level_values(0):
                stooq_df = stooq_raw['Value']
        else:
            stooq_df = stooq_raw
        stooq_df = stooq_df.sort_index()
    except Exception:
        pass

    yf_df = pd.DataFrame()
    try:
        yf_tickers = list(MACRO_SYMBOLS_YF.values())
        yf_raw = yf.download(yf_tickers, start=start_date, end=end_date, progress=False)
        if not yf_raw.empty:
            if isinstance(yf_raw.columns, pd.MultiIndex):
                if 'Close' in yf_raw.columns.get_level_values(0):
                    yf_df = yf_raw['Close']
                elif 'Adj Close' in yf_raw.columns.get_level_values(0):
                    yf_df = yf_raw['Adj Close']
            else:
                if 'Close' in yf_raw.columns:
                    yf_df = yf_raw[['Close']]
                else:
                    yf_df = yf_raw
            if yf_df.index.tz is not None:
                yf_df.index = yf_df.index.tz_localize(None)
            yf_df = yf_df.sort_index()
    except Exception:
        pass

    if not stooq_df.empty and not yf_df.empty:
        combined_macro = pd.concat([stooq_df, yf_df], axis=1).sort_index()
    elif not stooq_df.empty:
        combined_macro = stooq_df
    elif not yf_df.empty:
        combined_macro = yf_df
    else:
        return pd.DataFrame()
    return combined_macro.ffill()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_batch_data(stock_list, end_date=None, days_back=100, include_live=True):
    """Batch download for screener with optional live data for current day"""
    if end_date is None:
        end_date = datetime.date.today()
    
    # Add buffer for end date to ensure we get the requested date
    download_end = end_date + datetime.timedelta(days=5)
    start_date = end_date - datetime.timedelta(days=days_back + 365)
    
    try:
        all_data = yf.download(
            stock_list,
            start=start_date,
            end=download_end,
            progress=False,
            auto_adjust=True,
            group_by='ticker'
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
        
        # Fetch live data for today if requested and end_date is today
        if include_live and end_date == datetime.date.today() and data_dict:
            # Check if today's data is missing from at least one ticker
            sample_df = list(data_dict.values())[0]
            sample_df.index = pd.to_datetime(sample_df.index)
            if sample_df.index.tz is not None:
                sample_df.index = sample_df.index.tz_localize(None)
            
            has_today = any(idx.date() == datetime.date.today() for idx in sample_df.index)
            
            if not has_today:
                # Fetch live data for all tickers
                try:
                    live_data = yf.download(
                        list(data_dict.keys()),
                        period="1d",
                        progress=False,
                        auto_adjust=True,
                        group_by='ticker'
                    )
                    
                    if not live_data.empty:
                        if isinstance(live_data, pd.DataFrame) and isinstance(live_data.columns, pd.MultiIndex):
                            for ticker in data_dict.keys():
                                try:
                                    live_ticker = live_data.xs(ticker, level=0, axis=1)
                                    if not live_ticker.empty and not live_ticker['Close'].isnull().all():
                                        # Append live data to historical
                                        hist_df = data_dict[ticker]
                                        hist_df.index = pd.to_datetime(hist_df.index)
                                        if hist_df.index.tz is not None:
                                            hist_df.index = hist_df.index.tz_localize(None)
                                        
                                        live_ticker.index = pd.to_datetime(live_ticker.index)
                                        if live_ticker.index.tz is not None:
                                            live_ticker.index = live_ticker.index.tz_localize(None)
                                        
                                        # Only append if not already present
                                        new_dates = live_ticker.index.difference(hist_df.index)
                                        if len(new_dates) > 0:
                                            data_dict[ticker] = pd.concat([hist_df, live_ticker.loc[new_dates]]).sort_index()
                                except KeyError:
                                    pass
                        
                        return data_dict, f"✓ Downloaded {len(data_dict)} tickers (with live data)"
                except Exception:
                    pass  # Fall through to return historical data only
            
        return data_dict, f"✓ Downloaded {len(data_dict)} tickers"

    except Exception as e:
        return None, f"Download error: {e}"


def resample_to_weekly(df):
    """Resample daily OHLCV data to weekly candles"""
    if df is None or df.empty:
        return df
    
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    
    # Resample OHLCV to weekly (Week ending Friday)
    weekly = df.resample('W-FRI').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    return weekly


def resample_macro_to_weekly(macro_df):
    """Resample macro data to weekly (using last value of each week)"""
    if macro_df is None or macro_df.empty:
        return macro_df
    
    macro_df = macro_df.copy()
    macro_df.index = pd.to_datetime(macro_df.index)
    
    # Resample to weekly using last value
    weekly = macro_df.resample('W-FRI').last().dropna(how='all')
    
    return weekly

# ══════════════════════════════════════════════════════════════════════════════
# MSF + MMR INDICATOR CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def sigmoid(x, scale=1.0):
    return 2.0 / (1.0 + np.exp(-x / scale)) - 1.0


def zscore_clipped(series, window, clip=3.0):
    roll_mean = series.rolling(window=window).mean()
    roll_std = series.rolling(window=window).std()
    z = (series - roll_mean) / roll_std.replace(0, np.nan)
    return z.clip(-clip, clip).fillna(0)


def calculate_atr(df, length=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()


def calculate_msf(df, length=20, roc_len=14, clip=3.0):
    """
    Market Structure & Flow (MSF) Indicator
    Combines momentum, microstructure, and flow analysis
    """
    close = df['Close']
    
    # Momentum Component
    roc_raw = close.pct_change(roc_len, fill_method=None)
    roc_z = zscore_clipped(roc_raw, length, clip)
    momentum_norm = sigmoid(roc_z, 1.5)
    
    # Microstructure Component
    intrabar_dir = (df['High'] + df['Low']) / 2 - df['Open']
    vol_ma = df['Volume'].rolling(length).mean()
    vol_ratio = (df['Volume'] / vol_ma).fillna(1.0)
    
    vw_direction = (intrabar_dir * vol_ratio).rolling(length).mean()
    price_change_imp = close.diff(5)
    vw_impact = (price_change_imp * vol_ratio).rolling(length).mean()
    
    micro_raw = vw_direction - vw_impact
    micro_z = zscore_clipped(micro_raw, length, clip)
    micro_norm = sigmoid(micro_z, 1.5)
    
    # Composite Trend Component
    trend_fast = close.rolling(5).mean()
    trend_slow = close.rolling(length).mean()
    trend_diff_z = zscore_clipped(trend_fast - trend_slow, length, clip)
    
    mom_accel_raw = close.diff(5).diff(5)
    mom_accel_z = zscore_clipped(mom_accel_raw, length, clip)
    
    atr = calculate_atr(df, 14)
    vol_adj_mom_raw = close.diff(5) / atr
    vol_adj_mom_z = zscore_clipped(vol_adj_mom_raw, length, clip)
    
    mean_rev_z = zscore_clipped(close - trend_slow, length, clip)
    
    composite_trend_z = (trend_diff_z + mom_accel_z + vol_adj_mom_z + mean_rev_z) / np.sqrt(4.0)
    composite_trend_norm = sigmoid(composite_trend_z, 1.5)
    
    # Flow Component
    typical_price = (df['High'] + df['Low'] + close) / 3
    mf = typical_price * df['Volume']
    mf_pos = np.where(close > close.shift(1), mf, 0)
    mf_neg = np.where(close < close.shift(1), mf, 0)
    
    mf_pos_smooth = pd.Series(mf_pos, index=df.index).rolling(length).mean()
    mf_neg_smooth = pd.Series(mf_neg, index=df.index).rolling(length).mean()
    mf_total = mf_pos_smooth + mf_neg_smooth
    
    accum_ratio = mf_pos_smooth / mf_total.replace(0, np.nan)
    accum_ratio = accum_ratio.fillna(0.5)
    accum_norm = 2.0 * (accum_ratio - 0.5)
    
    # Regime Component
    pct_change = close.pct_change(fill_method=None)
    threshold = 0.0033
    regime_signals = np.select([pct_change > threshold, pct_change < -threshold], [1, -1], default=0)
    regime_count = pd.Series(regime_signals, index=df.index).cumsum()
    regime_raw = regime_count - regime_count.rolling(length).mean()
    regime_z = zscore_clipped(regime_raw, length, clip)
    regime_norm = sigmoid(regime_z, 1.5)
    
    # Combine Components
    osc_momentum = momentum_norm
    osc_structure = (micro_norm + composite_trend_norm) / np.sqrt(2.0)
    osc_flow = (accum_norm + regime_norm) / np.sqrt(2.0)
    
    msf_raw = (osc_momentum + osc_structure + osc_flow) / np.sqrt(3.0)
    msf_signal = sigmoid(msf_raw * np.sqrt(3.0), 1.0)
    
    return msf_signal, micro_norm, momentum_norm, accum_norm


def calculate_mmr(df, length=20, num_vars=5):
    """
    Macro Market Regression (MMR) Indicator
    Correlates price with macro factors
    """
    available_macros = [v for v in MACRO_SYMBOLS.values() if v in df.columns]
    target = df['Close']
    
    if len(df) < length + 10 or not available_macros:
        return pd.Series(0, index=df.index), pd.Series(0, index=df.index)

    correlations = df[available_macros].corrwith(target).abs().sort_values(ascending=False)
    top_drivers = correlations.head(num_vars).index.tolist()
    
    preds = []
    r2_sum = 0
    r2_sq_sum = 0
    y_mean = target.rolling(length).mean()
    y_std = target.rolling(length).std()

    for ticker in top_drivers:
        x = df[ticker]
        x_mean = x.rolling(length).mean()
        x_std = x.rolling(length).std()
        roll_corr = x.rolling(length).corr(target)
        slope = roll_corr * (y_std / x_std)
        intercept = y_mean - (slope * x_mean)
        
        pred = (slope * x) + intercept
        r2 = roll_corr ** 2
        
        preds.append(pred * r2)
        r2_sum += r2
        r2_sq_sum += r2 ** 2

    r2_sum = r2_sum.replace(0, np.nan)
    
    if len(preds) > 0:
        y_predicted = sum(preds) / r2_sum
    else:
        y_predicted = y_mean

    deviation = target - y_predicted
    mmr_z = zscore_clipped(deviation, length, 3.0)
    mmr_signal = sigmoid(mmr_z, 1.5)
    
    model_r2 = r2_sq_sum / r2_sum
    mmr_quality = np.sqrt(model_r2.fillna(0))
    
    return mmr_signal, mmr_quality


def run_full_analysis(df, length, roc_len, regime_sensitivity=1.5, base_weight=0.5):
    """Run full MSF + MMR analysis"""
    df['MSF'], df['Micro'], df['Momentum'], df['Flow'] = calculate_msf(df, length, roc_len)
    df['MMR'], df['MMR_Quality'] = calculate_mmr(df, length, num_vars=5)
    
    # Adaptive weighting
    msf_clarity = df['MSF'].abs()
    mmr_clarity = df['MMR'].abs()
    msf_clarity_scaled = msf_clarity.pow(regime_sensitivity)
    mmr_clarity_scaled = (mmr_clarity * df['MMR_Quality']).pow(regime_sensitivity)
    clarity_sum = msf_clarity_scaled + mmr_clarity_scaled + 0.001
    
    msf_w_adaptive = msf_clarity_scaled / clarity_sum
    mmr_w_adaptive = mmr_clarity_scaled / clarity_sum
    
    msf_w_final = 0.5 * base_weight + 0.5 * msf_w_adaptive
    mmr_w_final = 0.5 * (1.0 - base_weight) + 0.5 * mmr_w_adaptive
    w_sum = msf_w_final + mmr_w_final
    msf_w_norm = msf_w_final / w_sum
    mmr_w_norm = mmr_w_final / w_sum
    
    unified_signal = (msf_w_norm * df['MSF']) + (mmr_w_norm * df['MMR'])
    
    # Agreement multiplier
    agreement = df['MSF'] * df['MMR']
    agree_strength = agreement.abs()
    multiplier = np.where(agreement > 0, 1.0 + 0.2 * agree_strength, 1.0 - 0.1 * agree_strength)
    
    df['Unified'] = (unified_signal * multiplier).clip(-1.0, 1.0)
    df['Unified_Osc'] = df['Unified'] * 10
    df['MSF_Osc'] = df['MSF'] * 10
    df['MMR_Osc'] = df['MMR'] * 10
    df['MSF_Weight'] = msf_w_norm
    df['MMR_Weight'] = mmr_w_norm
    df['Agreement'] = agreement
    
    # Signals require strong agreement
    strong_agreement = agreement > 0.3
    df['Buy_Signal'] = strong_agreement & (df['Unified_Osc'] < -5)
    df['Sell_Signal'] = strong_agreement & (df['Unified_Osc'] > 5)
    
    # Divergences
    osc_rising = df['Unified_Osc'] > df['Unified_Osc'].shift(1)
    price_falling = df['Close'] < df['Close'].shift(1)
    osc_falling = df['Unified_Osc'] < df['Unified_Osc'].shift(1)
    price_rising = df['Close'] > df['Close'].shift(1)

    df['Bullish_Div'] = osc_rising & price_falling & (df['Unified_Osc'] < -5)
    df['Bearish_Div'] = osc_falling & price_rising & (df['Unified_Osc'] > 5)
    
    # Condition labels
    conditions = []
    for val in df['Unified_Osc']:
        if val < -5:
            conditions.append("Oversold")
        elif val > 5:
            conditions.append("Overbought")
        else:
            conditions.append("Neutral")
    df['Condition'] = conditions

    return df

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def create_distribution_chart(results_df):
    """Create signal distribution histogram"""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=results_df['Signal'], 
        nbinsx=20, 
        marker=dict(color='#FFC300', line=dict(color='#2A2A2A', width=1)), 
        opacity=0.8
    ))
    fig.add_vline(x=-5, line=dict(color='#10b981', width=2, dash='dash'))
    fig.add_vline(x=5, line=dict(color='#ef4444', width=2, dash='dash'))
    fig.add_vline(x=0, line=dict(color='#888888', width=1))
    fig.add_vrect(x0=-10, x1=-5, fillcolor='rgba(16,185,129,0.1)', line_width=0)
    fig.add_vrect(x0=5, x1=10, fillcolor='rgba(239,68,68,0.1)', line_width=0)
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', height=200,
        margin=dict(l=0, r=0, t=10, b=30),
        xaxis=dict(title=dict(text='Signal Value', font=dict(size=10, color='#888888')), showgrid=True, gridcolor='rgba(42,42,42,0.5)', range=[-12, 12]),
        yaxis=dict(title=dict(text='Count', font=dict(size=10, color='#888888')), showgrid=True, gridcolor='rgba(42,42,42,0.5)'),
        font=dict(family='Inter', color='#EAEAEA'), bargap=0.1
    )
    return fig


def create_ranking_chart(results_df, top_n=20):
    """Create horizontal bar chart of extreme signals"""
    sorted_df = results_df.sort_values('Signal')
    bottom = sorted_df.head(top_n//2)
    top = sorted_df.tail(top_n//2)
    combined = pd.concat([bottom, top])
    colors = ['#10b981' if v < 0 else '#ef4444' for v in combined['Signal']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=combined['DisplayName'], x=combined['Signal'], orientation='h',
        marker=dict(color=colors, line=dict(color='#2A2A2A', width=1)),
        text=[f"{v:.1f}" for v in combined['Signal']], textposition='outside', textfont=dict(size=10, color='#888888')
    ))
    fig.add_vline(x=0, line=dict(color='#FFC300', width=1))
    fig.add_vline(x=-5, line=dict(color='rgba(16,185,129,0.5)', width=1, dash='dash'))
    fig.add_vline(x=5, line=dict(color='rgba(239,68,68,0.5)', width=1, dash='dash'))
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', height=400,
        margin=dict(l=100, r=50, t=10, b=10),
        xaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)', range=[-12, 12], tickvals=[-10, -5, 0, 5, 10]),
        yaxis=dict(showgrid=False, tickfont=dict(size=10)),
        font=dict(family='Inter', color='#EAEAEA')
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def render_header():
    st.markdown("""
    <div class="premium-header">
        <h1>Sanket : Market Signal Screener</h1>
        <div class="tagline">MSF + MMR Quantitative Signal Scanner</div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
            <div style="font-size: 1.75rem; font-weight: 800; color: #FFC300;">Sanket</div>
            <div style="color: #888888; font-size: 0.75rem; margin-top: 0.25rem;">Signal Scanner</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Timeframe Selection (NEW)
        st.markdown('<div class="sidebar-title">⏱️ Timeframe</div>', unsafe_allow_html=True)
        timeframe = st.radio(
            "Select Timeframe",
            TIMEFRAME_OPTIONS,
            horizontal=True,
            help="Daily: Analyze daily candles | Weekly: Analyze weekly candles"
        )
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Universe Selection
        st.markdown('<div class="sidebar-title">🎯 Universe Selection</div>', unsafe_allow_html=True)
        universe = st.selectbox(
            "Analysis Universe",
            UNIVERSE_OPTIONS,
            help="Choose between F&O stocks or specific index constituents"
        )
        
        selected_index = None
        if universe == "Index Constituents":
            selected_index = st.selectbox(
                "Select Index",
                INDEX_LIST,
                index=INDEX_LIST.index("NIFTY 500"),
                help="Select the index for constituent analysis"
            )
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Analysis Date
        st.markdown('<div class="sidebar-title">📅 Analysis Date</div>', unsafe_allow_html=True)
        analysis_date = st.date_input(
            "Select Date",
            datetime.date.today(),
            max_value=datetime.date.today(),
            help="Select the date for signal analysis (defaults to today)"
        )
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Parameters
        st.markdown('<div class="sidebar-title">⚙️ Parameters</div>', unsafe_allow_html=True)
        with st.expander("MSF + MMR Settings", expanded=False):
            length = st.slider("Lookback Period", 10, 50, 20)
            roc_len = st.slider("ROC Length", 5, 30, 14)
            regime_sensitivity = st.slider("Regime Sensitivity", 0.5, 3.0, 1.5, 0.1)
            base_weight = st.slider("Base MSF Weight", 0.0, 1.0, 0.5, 0.05)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='info-box'>
            <p style='font-size: 0.8rem; margin: 0; color: var(--text-muted); line-height: 1.5;'>
                <strong>Version:</strong> {VERSION}<br>
                <strong>Engine:</strong> MSF + MMR Synthesis<br>
                <strong>Data:</strong> Live Market Feed
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        return universe, selected_index, analysis_date, length, roc_len, regime_sensitivity, base_weight, timeframe

# ══════════════════════════════════════════════════════════════════════════════
# MAIN SCREENER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_screener(universe, selected_index, analysis_date, length, roc_len, regime_sensitivity, base_weight, timeframe):
    """Main screener function with MMR and timeframe support"""
    
    # Format display
    analysis_date_str = analysis_date.strftime("%d %b %Y")
    is_today = analysis_date == datetime.date.today()
    
    universe_title = selected_index if universe == "Index Constituents" and selected_index else "F&O Stocks"
    timeframe_label = "Weekly" if timeframe == "Weekly" else "Daily"
    
    st.markdown(f"""
    <div class='info-box'>
        <h4>📊 Scanning {universe_title} ({timeframe_label})</h4>
        <p>MSF + MMR signal analysis across all securities.<br>
        <strong>Analysis Date:</strong> {analysis_date_str} {"(Today)" if is_today else ""} | <strong>Timeframe:</strong> {timeframe_label}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Validate analysis date
    if analysis_date > datetime.date.today():
        st.error("⚠️ Analysis date cannot be in the future.")
        return
    
    if st.button("◈ RUN SCREENER", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Fetch stock list
        status_text.markdown(f"**⏳ Fetching {universe_title} stock list...**")
        
        if universe == "F&O Stocks":
            stock_list, fetch_msg = get_fno_stock_list()
        else:
            stock_list, fetch_msg = get_index_stock_list(selected_index)
        
        if not stock_list:
            st.error(f"Failed to fetch stock list: {fetch_msg}")
            progress_bar.empty()
            status_text.empty()
            return
        
        st.toast(fetch_msg, icon="✅")
        total_stocks = len(stock_list)
        progress_bar.progress(0.05)
        
        # Step 2: Fetch macro data ONCE (optimized)
        status_text.markdown("**⏳ Fetching global macro data (one-time)...**")
        days_back_macro = 200 if timeframe == "Weekly" else 100
        macro_df = fetch_macro_data(days_back=days_back_macro + (datetime.date.today() - analysis_date).days)
        
        if macro_df.empty:
            st.warning("⚠️ Could not fetch macro data. Running with MSF-only mode.")
        else:
            # Resample macro data if weekly
            if timeframe == "Weekly":
                macro_df = resample_macro_to_weekly(macro_df)
            st.toast(f"✓ Loaded {len(macro_df.columns)} macro factors", icon="📊")
        
        progress_bar.progress(0.1)
        
        # Step 3: Batch download stock data
        status_text.markdown(f"**⏳ Downloading data for {total_stocks} stocks...**")
        
        days_back = 200 if timeframe == "Weekly" else 100
        data_dict, batch_msg = fetch_batch_data(stock_list, end_date=analysis_date, days_back=days_back)
        
        if data_dict is None:
            st.error(f"Failed to download data: {batch_msg}")
            progress_bar.empty()
            status_text.empty()
            return
        
        st.toast(batch_msg, icon="📥")
        progress_bar.progress(0.2)
        
        # Step 4: Process each stock
        results = []
        valid_tickers = list(data_dict.keys())
        total_valid = len(valid_tickers)
        
        for i, ticker in enumerate(valid_tickers):
            status_text.markdown(f"**⏳ Analyzing {ticker.replace('.NS', '')} ({i+1}/{total_valid})**")
            progress_bar.progress(0.2 + (0.8 * (i + 1) / total_valid))
            
            df = data_dict[ticker]
            
            if df is None or len(df) < length + 10:
                continue
                
            try:
                # Normalize index
                df.index = pd.to_datetime(df.index)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                
                # Resample to weekly if needed
                if timeframe == "Weekly":
                    df = resample_to_weekly(df)
                    if df is None or len(df) < length + 5:
                        continue
                
                # Join with macro data (reusing the pre-fetched macro_df)
                if not macro_df.empty:
                    df = df.join(macro_df, how='left').ffill()
                
                # Run full MSF + MMR analysis
                df = run_full_analysis(df, length, roc_len, regime_sensitivity, base_weight)
                
                # Find the row for the analysis date
                analysis_datetime = pd.Timestamp(analysis_date)
                
                if timeframe == "Weekly":
                    # For weekly, find the week containing the analysis date
                    valid_dates = df.index[df.index <= analysis_datetime]
                else:
                    valid_dates = df.index[df.index <= analysis_datetime]
                
                if len(valid_dates) == 0:
                    continue
                
                target_date = valid_dates[-1]
                target_idx = df.index.get_loc(target_date)
                
                if target_idx < 1:
                    continue
                
                last_row = df.iloc[target_idx]
                prev_row = df.iloc[target_idx - 1]
                price_change = ((last_row['Close'] - prev_row['Close']) / prev_row['Close']) * 100
                
                signal_str = "BUY" if last_row['Buy_Signal'] else "SELL" if last_row['Sell_Signal'] else "-"
                div_str = "BULL" if last_row['Bullish_Div'] else "BEAR" if last_row['Bearish_Div'] else "-"
                
                results.append({
                    "Symbol": ticker,
                    "DisplayName": ticker.replace(".NS", ""),
                    "Price": round(last_row['Close'], 2),
                    "Change": round(price_change, 2),
                    "Signal": round(last_row['Unified_Osc'], 2),
                    "MSF": round(last_row['MSF_Osc'], 2),
                    "MMR": round(last_row['MMR_Osc'], 2),
                    "Zone": last_row['Condition'],
                    "Trigger": signal_str,
                    "Divergence": div_str,
                    "Agreement": round(last_row['Agreement'], 3)
                })
            except Exception:
                pass
        
        progress_bar.empty()
        status_text.empty()
        
        if results:
            st.success(f"✅ Scan Complete! Analyzed {len(results)}/{total_stocks} stocks ({timeframe_label}) for {analysis_date_str}")
            results_df = pd.DataFrame(results)
            
            # Calculate summary stats
            n_oversold = len(results_df[results_df['Zone'] == 'Oversold'])
            n_overbought = len(results_df[results_df['Zone'] == 'Overbought'])
            n_neutral = len(results_df[results_df['Zone'] == 'Neutral'])
            n_buys = len(results_df[results_df['Trigger'] == 'BUY'])
            n_sells = len(results_df[results_df['Trigger'] == 'SELL'])
            avg_signal = results_df['Signal'].mean()
            
            regime = "BULLISH BIAS" if avg_signal < -2 else "BEARISH BIAS" if avg_signal > 2 else "NEUTRAL"
            regime_color = "success" if avg_signal < -2 else "danger" if avg_signal > 2 else "neutral"
            
            # Metrics row
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                st.markdown(f'<div class="metric-card info"><h4>Universe</h4><h2>{len(results)}</h2><div class="sub-metric">Stocks Analyzed</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card success"><h4>Oversold</h4><h2>{n_oversold}</h2><div class="sub-metric">Buy Zone</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card danger"><h4>Overbought</h4><h2>{n_overbought}</h2><div class="sub-metric">Sell Zone</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-card primary"><h4>Buy Signals</h4><h2>{n_buys}</h2><div class="sub-metric">Confirmed</div></div>', unsafe_allow_html=True)
            with c5:
                st.markdown(f'<div class="metric-card warning"><h4>Sell Signals</h4><h2>{n_sells}</h2><div class="sub-metric">Confirmed</div></div>', unsafe_allow_html=True)
            with c6:
                st.markdown(f'<div class="metric-card {regime_color}"><h4>Regime</h4><h2 style="font-size: 1.1rem;">{regime}</h2><div class="sub-metric">Avg: {avg_signal:.2f}</div></div>', unsafe_allow_html=True)
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["**📊 Signal Dashboard**", "**📈 Top Signals**", "**📉 Distribution**", "**📋 Full Data**"])
            
            with tab1:
                col_buy, col_sell = st.columns(2)
                
                with col_buy:
                    st.markdown('<div class="signal-card buy"><div class="signal-card-header"><span class="signal-card-title">🟢 Buy Opportunities</span></div>', unsafe_allow_html=True)
                    
                    confirmed_buys = results_df[results_df['Trigger'] == 'BUY'].sort_values('Signal').head(15)
                    if not confirmed_buys.empty:
                        st.markdown('<span class="status-badge buy">CONFIRMED BUY SIGNALS</span>', unsafe_allow_html=True)
                        for _, row in confirmed_buys.iterrows():
                            st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{row["DisplayName"]}</span><span class="symbol-price"> • ₹{row["Price"]:,.2f}</span></div><span class="symbol-score" style="color: #10b981;">{row["Signal"]:.1f}</span></div>', unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                    
                    oversold = results_df[(results_df['Zone'] == 'Oversold') & (results_df['Trigger'] != 'BUY')].sort_values('Signal').head(15)
                    if not oversold.empty:
                        st.markdown('<span class="status-badge oversold">OVERSOLD ZONE</span>', unsafe_allow_html=True)
                        for _, row in oversold.iterrows():
                            st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{row["DisplayName"]}</span><span class="symbol-price"> • ₹{row["Price"]:,.2f}</span></div><span class="symbol-score" style="color: #06b6d4;">{row["Signal"]:.1f}</span></div>', unsafe_allow_html=True)
                    
                    if confirmed_buys.empty and oversold.empty:
                        st.markdown('<p style="color: #888888; padding: 1rem;">No buy opportunities detected</p>', unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col_sell:
                    st.markdown('<div class="signal-card sell"><div class="signal-card-header"><span class="signal-card-title">🔴 Sell Opportunities</span></div>', unsafe_allow_html=True)
                    
                    confirmed_sells = results_df[results_df['Trigger'] == 'SELL'].sort_values('Signal', ascending=False).head(15)
                    if not confirmed_sells.empty:
                        st.markdown('<span class="status-badge sell">CONFIRMED SELL SIGNALS</span>', unsafe_allow_html=True)
                        for _, row in confirmed_sells.iterrows():
                            st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{row["DisplayName"]}</span><span class="symbol-price"> • ₹{row["Price"]:,.2f}</span></div><span class="symbol-score" style="color: #ef4444;">{row["Signal"]:.1f}</span></div>', unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                    
                    overbought = results_df[(results_df['Zone'] == 'Overbought') & (results_df['Trigger'] != 'SELL')].sort_values('Signal', ascending=False).head(15)
                    if not overbought.empty:
                        st.markdown('<span class="status-badge overbought">OVERBOUGHT ZONE</span>', unsafe_allow_html=True)
                        for _, row in overbought.iterrows():
                            st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{row["DisplayName"]}</span><span class="symbol-price"> • ₹{row["Price"]:,.2f}</span></div><span class="symbol-score" style="color: #f59e0b;">{row["Signal"]:.1f}</span></div>', unsafe_allow_html=True)
                    
                    if confirmed_sells.empty and overbought.empty:
                        st.markdown('<p style="color: #888888; padding: 1rem;">No sell opportunities detected</p>', unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Divergence alerts
                st.markdown("<br>", unsafe_allow_html=True)
                bull_divs = results_df[results_df['Divergence'] == 'BULL']
                bear_divs = results_df[results_df['Divergence'] == 'BEAR']
                
                if not bull_divs.empty or not bear_divs.empty:
                    st.markdown("##### 📊 Divergence Alerts")
                    div_cols = st.columns(2)
                    with div_cols[0]:
                        if not bull_divs.empty:
                            st.markdown('<span class="status-badge divergence">BULLISH DIVERGENCES</span>', unsafe_allow_html=True)
                            for _, row in bull_divs.head(10).iterrows():
                                st.markdown(f'<div class="symbol-row"><span class="symbol-name">{row["DisplayName"]}</span><span style="color: #FFC300;">Price ▼ | Signal ▲</span></div>', unsafe_allow_html=True)
                    with div_cols[1]:
                        if not bear_divs.empty:
                            st.markdown('<span class="status-badge divergence">BEARISH DIVERGENCES</span>', unsafe_allow_html=True)
                            for _, row in bear_divs.head(10).iterrows():
                                st.markdown(f'<div class="symbol-row"><span class="symbol-name">{row["DisplayName"]}</span><span style="color: #FFC300;">Price ▲ | Signal ▼</span></div>', unsafe_allow_html=True)
            
            with tab2:
                st.markdown("##### 🏆 Top 20 Most Oversold")
                top_oversold = results_df.nsmallest(20, 'Signal')
                cols_o = ['DisplayName', 'Price', 'Change', 'Signal', 'MSF', 'MMR', 'Zone', 'Trigger']
                st.dataframe(top_oversold[cols_o].rename(columns={'DisplayName': 'Symbol', 'Change': 'Chg %'}), width="stretch", hide_index=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("##### 🔻 Top 20 Most Overbought")
                top_overbought = results_df.nlargest(20, 'Signal')
                st.dataframe(top_overbought[cols_o].rename(columns={'DisplayName': 'Symbol', 'Change': 'Chg %'}), width="stretch", hide_index=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("##### 📊 Extreme Signals Chart")
                st.plotly_chart(create_ranking_chart(results_df, 20), width="stretch", config={'displayModeBar': False})
            
            with tab3:
                col_d1, col_d2 = st.columns(2)
                
                with col_d1:
                    st.markdown("##### Signal Distribution")
                    st.plotly_chart(create_distribution_chart(results_df), width="stretch", config={'displayModeBar': False})
                    
                    st.markdown("##### Zone Breakdown")
                    zone_data = {
                        "Zone": ["Oversold (< -5)", "Neutral (-5 to +5)", "Overbought (> +5)"],
                        "Count": [n_oversold, n_neutral, n_overbought],
                        "Percentage": [f"{n_oversold/len(results_df)*100:.1f}%", f"{n_neutral/len(results_df)*100:.1f}%", f"{n_overbought/len(results_df)*100:.1f}%"]
                    }
                    st.dataframe(pd.DataFrame(zone_data), width="stretch", hide_index=True)
                
                with col_d2:
                    st.markdown("##### Statistical Summary")
                    stats_data = {
                        "Metric": ["Total Stocks", "Mean Signal", "Median Signal", "Std Dev", "Min Signal", "Max Signal", "Buy/Sell Ratio"],
                        "Value": [
                            f"{len(results_df)}",
                            f"{results_df['Signal'].mean():.2f}",
                            f"{results_df['Signal'].median():.2f}",
                            f"{results_df['Signal'].std():.2f}",
                            f"{results_df['Signal'].min():.2f}",
                            f"{results_df['Signal'].max():.2f}",
                            f"{n_buys}:{n_sells}" if n_sells > 0 else f"{n_buys}:0"
                        ]
                    }
                    st.dataframe(pd.DataFrame(stats_data), width="stretch", hide_index=True)
                    
                    st.markdown("##### Top Gainers")
                    top_gainers = results_df.nlargest(10, 'Change')[['DisplayName', 'Price', 'Change', 'Signal']]
                    top_gainers.columns = ['Symbol', 'Price', 'Chg %', 'Signal']
                    st.dataframe(top_gainers, width="stretch", hide_index=True)
                    
                    st.markdown("##### Top Losers")
                    top_losers = results_df.nsmallest(10, 'Change')[['DisplayName', 'Price', 'Change', 'Signal']]
                    top_losers.columns = ['Symbol', 'Price', 'Chg %', 'Signal']
                    st.dataframe(top_losers, width="stretch", hide_index=True)
            
            with tab4:
                st.markdown(f"##### Complete Scan Results ({len(results_df)} stocks) - {analysis_date_str} ({timeframe_label})")
                
                # Filter options
                filter_col1, filter_col2, filter_col3 = st.columns(3)
                with filter_col1:
                    zone_filter = st.multiselect("Filter by Zone", ["Oversold", "Neutral", "Overbought"], default=["Oversold", "Neutral", "Overbought"])
                with filter_col2:
                    signal_filter = st.multiselect("Filter by Trigger", ["BUY", "SELL", "-"], default=["BUY", "SELL", "-"])
                with filter_col3:
                    sort_by = st.selectbox("Sort by", ["Signal", "Change", "Price", "DisplayName"], index=0)
                
                # Apply filters
                filtered_df = results_df[
                    (results_df['Zone'].isin(zone_filter)) & 
                    (results_df['Trigger'].isin(signal_filter))
                ].sort_values(sort_by, ascending=(sort_by == 'DisplayName'))
                
                display_cols = ['DisplayName', 'Price', 'Change', 'Signal', 'MSF', 'MMR', 'Zone', 'Trigger', 'Divergence']
                display_df = filtered_df[display_cols].copy()
                display_df.columns = ['Symbol', 'Price', 'Chg %', 'Signal', 'MSF', 'MMR', 'Zone', 'Trigger', 'Divergence']
                
                st.dataframe(display_df, width="stretch", hide_index=True, height=500)
                
                st.markdown("<br>", unsafe_allow_html=True)
                csv_data = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Report (CSV)",
                    data=csv_data,
                    file_name=f"sanket_{universe_title.replace(' ', '_')}_{timeframe_label}_{analysis_date.strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("No data retrieved. Please check your internet connection or try a different universe.")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    universe, selected_index, analysis_date, length, roc_len, regime_sensitivity, base_weight, timeframe = render_sidebar()
    render_header()
    
    # Signal interpretation guide
    with st.expander("📖 Signal Interpretation Guide", expanded=False):
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            st.markdown("""
            <div style='background: rgba(16, 185, 129, 0.1); border: 1px solid var(--success-green); border-radius: 12px; padding: 1rem;'>
                <h4 style='color: #10b981; margin-bottom: 0.5rem;'>🟢 Oversold Zone</h4>
                <p style='color: #888888; font-size: 0.85rem;'>Signal < -5</p>
                <p style='color: #EAEAEA; font-size: 0.85rem;'>
                    Potential buying opportunity. Look for MSF + MMR agreement.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_s2:
            st.markdown("""
            <div style='background: rgba(136, 136, 136, 0.1); border: 1px solid #888888; border-radius: 12px; padding: 1rem;'>
                <h4 style='color: #888888; margin-bottom: 0.5rem;'>⚪ Neutral Zone</h4>
                <p style='color: #888888; font-size: 0.85rem;'>Signal -5 to +5</p>
                <p style='color: #EAEAEA; font-size: 0.85rem;'>
                    No clear directional bias. Wait for breakout or use other factors.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_s3:
            st.markdown("""
            <div style='background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; border-radius: 12px; padding: 1rem;'>
                <h4 style='color: #ef4444; margin-bottom: 0.5rem;'>🔴 Overbought Zone</h4>
                <p style='color: #888888; font-size: 0.85rem;'>Signal > +5</p>
                <p style='color: #EAEAEA; font-size: 0.85rem;'>
                    Potential selling opportunity. Watch for bearish divergences.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("""
            <div style='background: rgba(255, 195, 0, 0.1); border: 1px solid #FFC300; border-radius: 12px; padding: 1rem;'>
                <h4 style='color: #FFC300; margin-bottom: 0.5rem;'>MSF - Market Structure & Flow</h4>
                <p style='color: #EAEAEA; font-size: 0.85rem;'>
                    Internal price-based indicator combining momentum, microstructure, and flow analysis.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m2:
            st.markdown("""
            <div style='background: rgba(6, 182, 212, 0.1); border: 1px solid #06b6d4; border-radius: 12px; padding: 1rem;'>
                <h4 style='color: #06b6d4; margin-bottom: 0.5rem;'>MMR - Macro Market Regression</h4>
                <p style='color: #EAEAEA; font-size: 0.85rem;'>
                    External macro correlation with bonds, currencies, and commodities.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Run screener
    run_screener(universe, selected_index, analysis_date, length, roc_len, regime_sensitivity, base_weight, timeframe)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption(f"© {datetime.datetime.now().year} Sanket | Hemrek Capital | {VERSION}")


if __name__ == "__main__":
    main()
