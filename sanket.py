"""
Sanket - Market Signal Screener | A Pragyam Product Family Member
UMA v3 Engine (MSF + MMR + Modulators) Quantitative Signal Scanner
"""

import streamlit as st
import pandas as pd
# pandas_datareader removed - using yfinance instead
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

VERSION = "v3.0.0 - UMA Engine"

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
    
    .info-box { background: var(--secondary-background-color); border: 1px solid var(--border-color); padding: 1.25rem; border-radius: 12px; margin: 0.5rem 0; box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08); }
    .info-box h4 { color: var(--primary-color); margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 700; }
    .info-box p { color: var(--text-muted); margin: 0; font-size: 0.9rem; line-height: 1.6; }
    
    .stButton>button { border: 2px solid var(--primary-color); background: transparent; color: var(--primary-color); font-weight: 700; border-radius: 12px; padding: 0.75rem 2rem; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); text-transform: uppercase; letter-spacing: 0.5px; }
    .stButton>button:hover { box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.6); background: var(--primary-color); color: #1A1A1A; transform: translateY(-2px); }
    
    .stPlotlyChart { border-radius: 12px; background-color: var(--secondary-background-color); padding: 10px; border: 1px solid var(--border-color); box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.1); }
    .stDataFrame { border-radius: 12px; background-color: var(--secondary-background-color); border: 1px solid var(--border-color); }
    .section-divider { height: 1px; background: linear-gradient(90deg, transparent 0%, var(--border-color) 50%, transparent 100%); margin: 1.5rem 0; }
    
    .symbol-row { display: flex; align-items: center; justify-content: space-between; padding: 0.75rem 1rem; border-radius: 8px; background: var(--bg-elevated); margin-bottom: 0.5rem; transition: all 0.2s ease; }
    .symbol-row:hover { background: var(--border-light); }
    .symbol-name { font-weight: 700; color: var(--text-primary); font-size: 0.9rem; }
    .symbol-price { color: var(--text-muted); font-size: 0.85rem; }
    .symbol-score { font-weight: 700; font-size: 0.9rem; }
    .signal-icon { font-size: 1.1rem; margin-right: 0.5rem; }
    
    .sidebar-title { font-size: 0.75rem; font-weight: 700; color: var(--primary-color); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.75rem; }
    [data-testid="stSidebar"] { background: var(--secondary-background-color); border-right: 1px solid var(--border-color); }
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
MACRO_SYMBOLS_YF_BONDS = {
    "India 10Y": "^INE10Y", "India 02Y": "^INE02Y",
    "US 30Y": "^TNX", "US 10Y": "^TNX", "US 05Y": "^FVX", "US 02Y": "^TVX",
    "UK 10Y": "^GXTG", "EU (DE) 10Y": "^DE10Y",
    "China 10Y": "^CN10Y", "Japan 10Y": "^JP10Y",
}

MACRO_SYMBOLS_YF = {
    "Dollar Index": "DX-Y.NYB", "Crude Oil": "CL=F", "Brent Crude": "BZ=F",
    "USD/INR": "INR=X", "GBP/INR": "GBPINR=X", "EUR/INR": "EURINR=X",
    "SGD/INR": "SGDINR=X", "JPY/INR": "JPYINR=X", "Gold": "GC=F", "Silver": "SI=F"
}

MACRO_SYMBOLS = {**MACRO_SYMBOLS_YF_BONDS, **MACRO_SYMBOLS_YF}

# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def get_fno_stock_list():
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
def fetch_macro_data(days_back=300):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days_back + 365)
    
    stooq_df = pd.DataFrame()
    try:
        stooq_tickers = list(MACRO_SYMBOLS_YF_BONDS.values())
        stooq_raw = yf.download(stooq_tickers, start=start_date, end=end_date, progress=False)
        if not stooq_raw.empty:
            if isinstance(stooq_raw.columns, pd.MultiIndex):
                if 'Close' in stooq_raw.columns.get_level_values(0):
                    stooq_df = stooq_raw['Close']
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
        
        if include_live and end_date == datetime.date.today() and data_dict:
            sample_df = list(data_dict.values())[0]
            sample_df.index = pd.to_datetime(sample_df.index)
            if sample_df.index.tz is not None:
                sample_df.index = sample_df.index.tz_localize(None)
            
            has_today = any(idx.date() == datetime.date.today() for idx in sample_df.index)
            
            if not has_today:
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
                                        hist_df = data_dict[ticker]
                                        hist_df.index = pd.to_datetime(hist_df.index)
                                        if hist_df.index.tz is not None:
                                            hist_df.index = hist_df.index.tz_localize(None)
                                        
                                        live_ticker.index = pd.to_datetime(live_ticker.index)
                                        if live_ticker.index.tz is not None:
                                            live_ticker.index = live_ticker.index.tz_localize(None)
                                        
                                        new_dates = live_ticker.index.difference(hist_df.index)
                                        if len(new_dates) > 0:
                                            data_dict[ticker] = pd.concat([hist_df, live_ticker.loc[new_dates]]).sort_index()
                                except KeyError:
                                    pass
                        
                        return data_dict, f"✓ Downloaded {len(data_dict)} tickers (with live data)"
                except Exception:
                    pass
            
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


def resample_macro_to_weekly(macro_df):
    if macro_df is None or macro_df.empty:
        return macro_df
    macro_df = macro_df.copy()
    macro_df.index = pd.to_datetime(macro_df.index)
    weekly = macro_df.resample('W-FRI').last().dropna(how='all')
    return weekly

# ══════════════════════════════════════════════════════════════════════════════
# UMA V3 ENGINE: MSF + MMR + MODULATORS CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def sigmoid(x, scale=1.0):
    return 2.0 / (1.0 + np.exp(-x / scale)) - 1.0


def zscore_clipped(series, window, clip=3.0):
    roll_mean = series.rolling(window=window).mean()
    roll_std = series.rolling(window=window).std(ddof=1)
    z = (series - roll_mean) / roll_std.replace(0, np.nan)
    return z.clip(-clip, clip).fillna(0)


def calculate_atr(df, length=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()


def calculate_rsi(series, length=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_wavetrend(df, length=20, wt_channel_len=10, wt_avg_len=21):
    ap = (df['High'] + df['Low'] + df['Close']) / 3.0
    esa = ap.ewm(span=wt_channel_len, adjust=False).mean()
    d = (ap - esa).abs().ewm(span=wt_channel_len, adjust=False).mean()
    ci = (ap - esa) / (0.015 * d).replace(0, np.nan)
    ci = ci.fillna(0)
    wt1 = ci.ewm(span=wt_avg_len, adjust=False).mean()
    wt2 = wt1.rolling(window=4).mean()
    wt_z = zscore_clipped(wt1, length, 3.0)
    wavetrend_norm = sigmoid(wt_z, 1.5)
    return wt1, wt2, wavetrend_norm


def calculate_entropy(df, length=20, lookback=50):
    d1 = df['Close'].diff()
    d_newest = d1
    d_middle = d1.shift(1)
    d_oldest = d1.shift(2)

    pe_p1 = ((d_oldest < d_middle) & (d_middle < d_newest)).astype(float)
    pe_p2 = ((d_oldest < d_newest) & (d_newest < d_middle)).astype(float)
    pe_p3 = ((d_middle < d_oldest) & (d_oldest < d_newest)).astype(float)
    pe_p4 = ((d_middle < d_newest) & (d_newest < d_oldest)).astype(float)
    pe_p5 = ((d_newest < d_oldest) & (d_oldest < d_middle)).astype(float)
    pe_p6 = ((d_newest < d_middle) & (d_middle < d_oldest)).astype(float)

    freq_1 = pe_p1.rolling(lookback).mean().fillna(0)
    freq_2 = pe_p2.rolling(lookback).mean().fillna(0)
    freq_3 = pe_p3.rolling(lookback).mean().fillna(0)
    freq_4 = pe_p4.rolling(lookback).mean().fillna(0)
    freq_5 = pe_p5.rolling(lookback).mean().fillna(0)
    freq_6 = pe_p6.rolling(lookback).mean().fillna(0)

    def safe_xlogx(x):
        return np.where(x > 1e-10, x * np.log(x), 0.0)
        
    h_raw = -(safe_xlogx(freq_1) + safe_xlogx(freq_2) + safe_xlogx(freq_3) + 
              safe_xlogx(freq_4) + safe_xlogx(freq_5) + safe_xlogx(freq_6))
    
    h_max = np.log(6)
    entropy_norm = pd.Series(h_raw / h_max, index=df.index)
    
    entropy_z = zscore_clipped(entropy_norm, length, 3.0)
    entropy_sigmoid = sigmoid(entropy_z, 1.5)
    entropy_mod = 1.0 - 0.25 * entropy_sigmoid
    return entropy_norm, entropy_mod


def calculate_hurst(df, short_len=10, long_len=50, sample_len=100):
    close = df['Close'].replace(0, np.nan)
    ret_short = np.log(close / close.shift(short_len))
    ret_long = np.log(close / close.shift(long_len))
    
    std_short = ret_short.rolling(sample_len).std(ddof=1).replace(0, np.nan)
    std_long = ret_long.rolling(sample_len).std(ddof=1).replace(0, np.nan)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        log_ratio_std = np.log(std_long / std_short)
        log_ratio_std = np.where(np.isfinite(log_ratio_std), log_ratio_std, np.nan)
    
    log_ratio_tau = np.log(long_len / short_len)
    
    hurst_raw = log_ratio_std / log_ratio_tau
    hurst_smooth = pd.Series(hurst_raw, index=df.index).ewm(span=10, adjust=False).mean()
    hurst_clipped = hurst_smooth.clip(0.1, 0.9)
    return hurst_clipped


def calculate_vol_structure(df, length=20):
    atr_s = calculate_atr(df, 5)
    atr_l = calculate_atr(df, 20)
    atr_ref = calculate_atr(df, 14)
    
    atr_ref_mean = atr_ref.rolling(length).mean()
    atr_ref_std = atr_ref.rolling(length).std()
    
    vov_raw = atr_ref_std / atr_ref_mean.replace(0, np.nan)
    vts_raw = (atr_s / atr_l.replace(0, np.nan)).fillna(1.0)
    
    vov_z = zscore_clipped(vov_raw.fillna(0), length, 3.0)
    vts_z = zscore_clipped(vts_raw.fillna(1.0), length, 3.0)
    
    vol_stress_z = (vov_z + vts_z) / np.sqrt(2.0)
    vol_stress_sigmoid = sigmoid(vol_stress_z, 1.5)
    
    vol_mod = 1.0 - 0.15 * np.maximum(vol_stress_sigmoid, 0.0)
    return vol_mod, vol_stress_sigmoid, vts_raw


def calculate_mmr(df, length=20, num_vars=5):
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
        x_std = x_std.replace(0, np.nan)
        y_std_safe = y_std.replace(0, np.nan)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            roll_corr = x.rolling(length).corr(target)
            slope = roll_corr * (y_std_safe / x_std)
            slope = slope.where(np.isfinite(slope), np.nan)
        
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
    """Run full UMA v3 Engine Analysis"""
    close = df['Close']
    
    # 1. Base MSF Families
    # Momentum
    roc_raw = close.pct_change(roc_len, fill_method=None)
    roc_z = zscore_clipped(roc_raw, length, 3.0)
    momentum_norm = sigmoid(roc_z, 1.5)
    
    # Structure
    intrabar_dir = (df['High'] + df['Low']) / 2 - df['Open']
    vol_ma = df['Volume'].rolling(length).mean()
    vol_ratio = (df['Volume'] / vol_ma).fillna(1.0)
    vw_direction = (intrabar_dir * vol_ratio).rolling(length).mean()
    price_change_imp = close - close.shift(5)
    vw_impact = (price_change_imp * vol_ratio).rolling(length).mean()
    micro_raw = vw_direction - vw_impact
    micro_norm = sigmoid(zscore_clipped(micro_raw, length, 3.0), 1.5)
    
    trend_fast = close.rolling(5).mean()
    trend_slow = close.rolling(length).mean()
    trend_diff_z = zscore_clipped(trend_fast - trend_slow, length, 3.0)
    mom_accel_z = zscore_clipped(close.diff(5).diff(5), length, 3.0)
    atr = calculate_atr(df, 14)
    vol_adj_mom_z = zscore_clipped(close.diff(5) / atr.replace(0, np.nan), length, 3.0)
    mean_rev_z = zscore_clipped(close - trend_slow, length, 3.0)
    composite_trend_norm = sigmoid((trend_diff_z + mom_accel_z + vol_adj_mom_z + mean_rev_z) / np.sqrt(4.0), 1.5)
    
    # Flow
    typical_price = (df['High'] + df['Low'] + close) / 3
    mf = typical_price * df['Volume']
    mf_pos = pd.Series(np.where(close > close.shift(1), mf, 0), index=df.index).rolling(length).mean()
    mf_neg = pd.Series(np.where(close < close.shift(1), mf, 0), index=df.index).rolling(length).mean()
    mf_total = mf_pos + mf_neg
    accum_norm = 2.0 * ((mf_pos / mf_total.replace(0, np.nan)).fillna(0.5) - 0.5)
    
    pct_change = close.pct_change(fill_method=None)
    regime_signals = np.select([pct_change > 0.0033, pct_change < -0.0033], [1, -1], default=0)
    regime_count = pd.Series(regime_signals, index=df.index).cumsum()
    regime_raw = regime_count - regime_count.rolling(length).mean()
    regime_norm = sigmoid(zscore_clipped(regime_raw, length, 3.0), 1.5)
    
    # Cycle
    wt1, wt2, wavetrend_norm = calculate_wavetrend(df, length)
    
    # RSI Component (add to structure)
    rsi_val = calculate_rsi(close, 14)
    rsi_norm = (rsi_val - 50) / 50
    
    # Price Position in Confidence Bands
    price_mean = close.rolling(length).mean()
    price_stdev = close.rolling(length).std(ddof=1)
    conf_mult = 1.96  # 95% confidence
    upper_bound = price_mean + conf_mult * price_stdev
    lower_bound = price_mean - conf_mult * price_stdev
    band_width = upper_bound - lower_bound
    price_position = np.where(band_width > 0, (close - lower_bound) / band_width * 2 - 1, 0.0)
    price_position_clipped = price_position.clip(-1.5, 1.5)
    
    # 2. Modulators
    entropy_norm, entropy_mod = calculate_entropy(df, length)
    hurst_clipped = calculate_hurst(df)
    vol_mod, vol_stress_sigmoid, vts_raw = calculate_vol_structure(df, length)
    
    # 3. Hurst Tilting & MSF Aggregation
    hurst_tilt = ((hurst_clipped - 0.5) * 2.5).clip(-1.0, 1.0)
    hurst_w_momentum = 1.0 + hurst_tilt * 0.3
    hurst_w_structure = 1.0 + hurst_tilt * 0.3
    hurst_w_flow = 1.0 - hurst_tilt * 0.15
    hurst_w_cycle = 1.0 - hurst_tilt * 0.3
    hurst_w_denom = np.sqrt(hurst_w_momentum**2 + hurst_w_structure**2 + hurst_w_flow**2 + hurst_w_cycle**2)
    
    osc_momentum = momentum_norm
    osc_structure = (micro_norm + composite_trend_norm) / np.sqrt(2.0)
    osc_flow = (accum_norm + regime_norm) / np.sqrt(2.0)
    osc_cycle = wavetrend_norm
    
    msf_raw_weighted = (hurst_w_momentum * osc_momentum + hurst_w_structure * osc_structure + 
                        hurst_w_flow * osc_flow + hurst_w_cycle * osc_cycle)
    
    msf_raw = msf_raw_weighted / hurst_w_denom.replace(0, 1.0)
    msf_pre_mod = sigmoid(msf_raw * 2.0, 1.0)
    msf_signal = (msf_pre_mod * entropy_mod).clip(-1.0, 1.0)
    
    # 4. MMR
    mmr_signal, mmr_quality = calculate_mmr(df, length, num_vars=5)
    
    # 5. Integration
    msf_clarity = msf_signal.abs()
    mmr_clarity = mmr_signal.abs()
    
    msf_clarity_scaled = msf_clarity.pow(regime_sensitivity)
    mmr_clarity_scaled = (mmr_clarity * mmr_quality).pow(regime_sensitivity)
    clarity_sum = msf_clarity_scaled + mmr_clarity_scaled + 0.001
    
    msf_w_adaptive = msf_clarity_scaled / clarity_sum
    mmr_w_adaptive = mmr_clarity_scaled / clarity_sum
    
    msf_w_final = 0.5 * base_weight + 0.5 * msf_w_adaptive
    mmr_w_final = 0.5 * (1.0 - base_weight) + 0.5 * mmr_w_adaptive
    w_sum = msf_w_final + mmr_w_final
    
    msf_w_norm = msf_w_final / w_sum
    mmr_w_norm = mmr_w_final / w_sum
    
    unified_signal = (msf_w_norm * msf_signal) + (mmr_w_norm * mmr_signal)
    
    signal_agreement = msf_signal * mmr_signal
    agreement_strength = signal_agreement.abs()
    
    wt_bull_cross = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    wt_bear_cross = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    
    cycle_confirm = np.where(wt_bull_cross, 1.0, np.where(wt_bear_cross, -1.0, 0.0))
    cycle_alignment = cycle_confirm * np.sign(unified_signal)
    
    base_agreement_mult = np.where(signal_agreement > 0, 1.0 + 0.2 * agreement_strength, 1.0 - 0.1 * agreement_strength)
    cycle_boost = np.where(cycle_alignment > 0, 1.05, np.where(cycle_alignment < 0, 0.95, 1.0))
    
    unified_pre_vol = unified_signal * base_agreement_mult * cycle_boost
    unified_final = (unified_pre_vol * vol_mod).clip(-1.0, 1.0)
    
    # Ensure scaling
    df['Unified_Osc'] = unified_final * 10.0
    df['MSF_Osc'] = msf_signal * 10.0
    df['MMR_Osc'] = mmr_signal * 10.0
    
    # 6. Bollinger Bands & RSI on oscillator
    bb_length = 20
    bb_mult = 2.0
    bb_basis = df['Unified_Osc'].rolling(bb_length).mean()
    bb_std = df['Unified_Osc'].rolling(bb_length).std()
    bb_upper = bb_basis + bb_mult * bb_std
    bb_lower = bb_basis - bb_mult * bb_std
    
    rsi_osc = calculate_rsi(df['Unified_Osc'], 14)
    rsi_lower = 40
    rsi_upper = 70
    
    # 7. Specific UMA v3 Signals Matching
    strong_agreement = signal_agreement > 0.3
    
    osc_rising = df['Unified_Osc'] > df['Unified_Osc'].shift(1)
    price_falling = close < close.shift(1)
    osc_falling = df['Unified_Osc'] < df['Unified_Osc'].shift(1)
    price_rising = close > close.shift(1)
    
    # Basic OB/OS conditions
    is_oversold = (df['Unified_Osc'] < bb_lower) & (rsi_osc < rsi_lower)
    is_overbought = (df['Unified_Osc'] > bb_upper) & (rsi_osc > rsi_upper)
    
    # Deep OB/OS with WaveTrend confirmation
    wt_at_oversold = wt1 < -53
    wt_at_overbought = wt1 > 53
    is_deep_oversold = is_oversold & wt_at_oversold
    is_deep_overbought = is_overbought & wt_at_overbought
    
    # Tier 3: Maximum conviction signals
    entropy_avg = entropy_norm.rolling(length).mean()
    entropy_favorable = entropy_norm < entropy_avg
    vol_calm = vol_stress_sigmoid < 0
    hurst_mean_revert = hurst_clipped < 0.48
    hurst_trending = hurst_clipped > 0.52
    
    is_tier3_buy = is_deep_oversold & entropy_favorable & vol_calm & hurst_mean_revert
    is_tier3_sell = is_deep_overbought & entropy_favorable & vol_calm & hurst_trending
    
    # VTS Regime (using vts_raw from vol_structure)
    vts_regime = np.select(
        [vts_raw > 1.15, vts_raw > 1.0, vts_raw < 0.85, vts_raw < 1.0],
        [2, 1, -2, -1],
        default=0
    )
    
    # Triangle = Divergence Signals
    df['Triangle_Buy'] = osc_rising & price_falling & (df['Unified_Osc'] < -5)
    df['Triangle_Sell'] = osc_falling & price_rising & (df['Unified_Osc'] > 5)
    
    # Circle = Confirmed OB/OS Strong Agreement Signals
    df['Circle_Buy'] = strong_agreement & (df['Unified_Osc'] < -5)
    df['Circle_Sell'] = strong_agreement & (df['Unified_Osc'] > 5)
    
    # Diamond = WT Cross in Extreme Zones
    df['Diamond_Buy'] = wt_bull_cross & (wt1 < -53)
    df['Diamond_Sell'] = wt_bear_cross & (wt1 > 53)
    
    # Star = Tier 3 signals
    df['Star_Buy'] = is_tier3_buy
    df['Star_Sell'] = is_tier3_sell
    
    # Store additional metrics
    df['RSI_Osc'] = rsi_osc
    df['BB_Upper'] = bb_upper
    df['BB_Lower'] = bb_lower
    df['Price_Position'] = price_position_clipped
    df['Entropy_Norm'] = entropy_norm
    df['Hurst'] = hurst_clipped
    df['Vol_Stress'] = vol_stress_sigmoid
    df['VTS_Regime'] = vts_regime
    df['WT1'] = wt1
    df['WT2'] = wt2
    df['RSI'] = rsi_val
    
    df['Condition'] = np.select(
        [is_tier3_buy, is_deep_oversold, is_oversold, is_tier3_sell, is_deep_overbought, is_overbought],
        ['Tier3 Buy', 'Deep Oversold', 'Oversold', 'Tier3 Sell', 'Deep Overbought', 'Overbought'],
        default='Neutral'
    )

    return df

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def create_distribution_chart(results_df):
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
    st.markdown(f"""
    <div class="premium-header">
        <h1>Sanket : Market Signal Screener</h1>
        <div class="tagline">UMA v3 Quantitative Signal Scanner (MSF + MMR + Ent/Hurst/Vol)</div>
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
        
        st.markdown('<div class="sidebar-title">⏱️ Timeframe</div>', unsafe_allow_html=True)
        timeframe = st.radio(
            "Select Timeframe",
            TIMEFRAME_OPTIONS,
            horizontal=True,
            help="Daily: Analyze daily candles | Weekly: Analyze weekly candles"
        )
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
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
        
        st.markdown('<div class="sidebar-title">📅 Analysis Date</div>', unsafe_allow_html=True)
        analysis_mode = st.radio(
            "Analysis Mode",
            ["Single Date", "Date Range (Time Series)"],
            horizontal=True,
            help="Single Date: Analyze one day | Date Range: Analyze historical period for time series"
        )
        
        if analysis_mode == "Single Date":
            analysis_date = st.date_input(
                "Select Date",
                datetime.date.today(),
                max_value=datetime.date.today(),
                help="Select the date for signal analysis"
            )
            start_date_hist = None
            end_date_hist = None
        else:
            analysis_date = None  # Not used in range mode
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                start_date_hist = st.date_input(
                    "Start Date",
                    datetime.date.today() - datetime.timedelta(days=30),
                    max_value=datetime.date.today() - 1,
                    help="Start date for time series analysis"
                )
            with col_date2:
                end_date_hist = st.date_input(
                    "End Date",
                    datetime.date.today(),
                    max_value=datetime.date.today(),
                    help="End date for time series analysis"
                )
            analysis_date = end_date_hist  # For single analysis fallback
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-title">⚙️ Parameters</div>', unsafe_allow_html=True)
        with st.expander("UMA Settings", expanded=False):
            length = st.slider("Lookback Period", 10, 50, 20)
            roc_len = st.slider("ROC Length", 5, 30, 14)
            regime_sensitivity = st.slider("Regime Sensitivity", 0.5, 3.0, 1.5, 0.1)
            base_weight = st.slider("Base MSF Weight", 0.0, 1.0, 0.5, 0.05)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='info-box'>
            <p style='font-size: 0.8rem; margin: 0; color: var(--text-muted); line-height: 1.5;'>
                <strong>Version:</strong> {VERSION}<br>
                <strong>Engine:</strong> UMA v3 Synthesis<br>
                <strong>Data:</strong> Live Market Feed
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        return universe, selected_index, analysis_date, length, roc_len, regime_sensitivity, base_weight, timeframe, analysis_mode, start_date_hist, end_date_hist

# ══════════════════════════════════════════════════════════════════════════════
# MAIN SCREENER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_screener(universe, selected_index, analysis_date, length, roc_len, regime_sensitivity, base_weight, timeframe):
    analysis_date_str = analysis_date.strftime("%d %b %Y")
    is_today = analysis_date == datetime.date.today()
    
    universe_title = selected_index if universe == "Index Constituents" and selected_index else "F&O Stocks"
    timeframe_label = "Weekly" if timeframe == "Weekly" else "Daily"
    
    st.markdown(f"""
    <div class='info-box'>
        <h4>📊 Scanning {universe_title} ({timeframe_label})</h4>
        <p>UMA v3 signal analysis across all securities.<br>
        <strong>Analysis Date:</strong> {analysis_date_str} {"(Today)" if is_today else ""} | <strong>Timeframe:</strong> {timeframe_label}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if analysis_date is not None and analysis_date > datetime.date.today():
        st.error("⚠️ Analysis date cannot be in the future.")
        return
    
    if analysis_date is not None:
        button_key = f"run_screener_{st.session_state.get('screener_click_count', 0)}"
        if st.button("◈ RUN SCREENER", type="primary", key=button_key):
            st.session_state.screener_click_count = st.session_state.get('screener_click_count', 0) + 1
            st.session_state.screener_run = True
        
        if st.session_state.get('screener_run', False):
            
            # 1. Fetch stock list
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
            
            # 2. Fetch macro data 
            status_text.markdown("**⏳ Fetching global macro data (one-time)...**")
            days_back_macro = 400 if timeframe == "Weekly" else 300
            macro_df = fetch_macro_data(days_back=days_back_macro + (datetime.date.today() - analysis_date).days)
            
            if macro_df.empty:
                st.warning("⚠️ Could not fetch macro data. Running with internal MSF mode only.")
            else:
                if timeframe == "Weekly":
                    macro_df = resample_macro_to_weekly(macro_df)
                st.toast(f"✓ Loaded {len(macro_df.columns)} macro factors", icon="📊")
            
            progress_bar.progress(0.1)
            
            # 3. Batch download stock data
            status_text.markdown(f"**⏳ Downloading data for {total_stocks} stocks...**")
            days_back = 500 if timeframe == "Weekly" else 300
            data_dict, batch_msg = fetch_batch_data(stock_list, end_date=analysis_date, days_back=days_back)
            
            if data_dict is None:
                st.error(f"Failed to download data: {batch_msg}")
                progress_bar.empty()
                status_text.empty()
                return
            
            st.toast(batch_msg, icon="📥")
            progress_bar.progress(0.2)
            
            # 4. Process each stock
            results = []
            valid_tickers = list(data_dict.keys())
            total_valid = len(valid_tickers)
            
            for i, ticker in enumerate(valid_tickers):
                status_text.markdown(f"**⏳ Analyzing {ticker.replace('.NS', '')} ({i+1}/{total_valid})**")
                progress_bar.progress(0.2 + (0.8 * (i + 1) / total_valid))
                
                df = data_dict[ticker]
                
                if df is None or len(df) < length + 100:
                    continue
                    
                try:
                    df.index = pd.to_datetime(df.index)
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    
                    if timeframe == "Weekly":
                        df = resample_to_weekly(df)
                        if df is None or len(df) < length + 50:
                            continue
                    
                    if not macro_df.empty:
                        df = df.join(macro_df, how='left').ffill()
                    
                    df = run_full_analysis(df, length, roc_len, regime_sensitivity, base_weight)
                    
                    analysis_datetime = pd.Timestamp(analysis_date)
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
                    
                    signals = []
                    if last_row.get('Star_Buy', False): signals.append("⭐ T3 BUY")
                    if last_row.get('Star_Sell', False): signals.append("⭐ T3 SELL")
                    if last_row['Diamond_Buy']: signals.append("💎 BUY")
                    if last_row['Diamond_Sell']: signals.append("🔶 SELL")
                    if last_row['Circle_Buy']: signals.append("🟢 BUY")
                    if last_row['Circle_Sell']: signals.append("🔴 SELL")
                    if last_row['Triangle_Buy']: signals.append("🔺 DIV")
                    if last_row['Triangle_Sell']: signals.append("🔻 DIV")
                    
                    trigger_str = " | ".join(signals) if signals else "-"
                    
                    if last_row.get('Star_Buy', False) or last_row.get('Star_Sell', False):
                        broad_class = "T3 BUY" if last_row.get('Star_Buy', False) else "T3 SELL"
                    elif last_row['Diamond_Buy'] or last_row['Diamond_Sell']:
                        broad_class = "BUY" if last_row['Diamond_Buy'] else "SELL"
                    elif last_row['Circle_Buy'] or last_row['Circle_Sell']:
                        broad_class = "BUY" if last_row['Circle_Buy'] else "SELL"
                    elif last_row['Triangle_Buy'] or last_row['Triangle_Sell']:
                        broad_class = "BUY" if last_row['Triangle_Buy'] else "SELL"
                    else:
                        broad_class = "-"
                    
                    results.append({
                        "Symbol": ticker,
                        "DisplayName": ticker.replace(".NS", ""),
                        "Price": round(last_row['Close'], 2),
                        "Change": round(price_change, 2),
                        "Signal": round(last_row['Unified_Osc'], 2),
                        "MSF": round(last_row['MSF_Osc'], 2),
                        "MMR": round(last_row['MMR_Osc'], 2),
                        "Zone": last_row['Condition'],
                        "Detailed Trigger": trigger_str,
                        "Trigger": broad_class,
                        "Has Diamond": last_row['Diamond_Buy'] or last_row['Diamond_Sell'],
                        "Has Circle": last_row['Circle_Buy'] or last_row['Circle_Sell'],
                        "Has Triangle": last_row['Triangle_Buy'] or last_row['Triangle_Sell'],
                    })
                except Exception:
                    pass
            
            progress_bar.empty()
            status_text.empty()
            
            if results:
                st.success(f"✅ Scan Complete! Analyzed {len(results)}/{total_stocks} stocks ({timeframe_label}) for {analysis_date_str}")
                results_df = pd.DataFrame(results)
                
                n_tier3_buy = len(results_df[results_df['Zone'] == 'Tier3 Buy'])
                n_tier3_sell = len(results_df[results_df['Zone'] == 'Tier3 Sell'])
                n_deep_oversold = len(results_df[results_df['Zone'] == 'Deep Oversold'])
                n_deep_overbought = len(results_df[results_df['Zone'] == 'Deep Overbought'])
                n_oversold = len(results_df[results_df['Zone'] == 'Oversold'])
                n_overbought = len(results_df[results_df['Zone'] == 'Overbought'])
                n_buys = len(results_df[results_df['Trigger'].str.contains('BUY', na=False)])
                n_sells = len(results_df[results_df['Trigger'].str.contains('SELL', na=False)])
                n_tier3 = n_tier3_buy + n_tier3_sell
                
                avg_signal = results_df['Signal'].mean()
                regime = "BULLISH BIAS" if avg_signal < -2 else "BEARISH BIAS" if avg_signal > 2 else "NEUTRAL"
                regime_color = "success" if avg_signal < -2 else "danger" if avg_signal > 2 else "neutral"
                
                st.markdown("<br>", unsafe_allow_html=True)
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                with c1:
                    st.markdown(f'<div class="metric-card info"><h4>Universe</h4><h2>{len(results)}</h2><div class="sub-metric">Stocks Analyzed</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="metric-card" style="border-color: #fbbf24;"><h4>Tier 3</h4><h2>{n_tier3}</h2><div class="sub-metric">Max Conviction</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="metric-card success"><h4>Oversold</h4><h2>{n_oversold + n_deep_oversold}</h2><div class="sub-metric">Buy Zone</div></div>', unsafe_allow_html=True)
                with c4:
                    st.markdown(f'<div class="metric-card danger"><h4>Overbought</h4><h2>{n_overbought + n_deep_overbought}</h2><div class="sub-metric">Sell Zone</div></div>', unsafe_allow_html=True)
                with c5:
                    st.markdown(f'<div class="metric-card primary"><h4>Buy Signals</h4><h2>{n_buys}</h2><div class="sub-metric">Confirmed</div></div>', unsafe_allow_html=True)
                with c6:
                    st.markdown(f'<div class="metric-card warning"><h4>Sell Signals</h4><h2>{n_sells}</h2><div class="sub-metric">Confirmed</div></div>', unsafe_allow_html=True)
                
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                
                tab1, tab2, tab3, tab4 = st.tabs(["**📊 Signal Dashboard**", "**📈 Top Signals**", "**📉 Distribution**", "**📋 Full Data**"])
                
                with tab1:
                    col_buy, col_sell = st.columns(2)
                    
                    with col_buy:
                        st.markdown('<div class="signal-card buy"><div class="signal-card-header"><span class="signal-card-title">🟢 Buy Opportunities</span></div>', unsafe_allow_html=True)
                        
                        stars_buy = results_df[results_df['Detailed Trigger'].str.contains("⭐")].sort_values('Signal')
                        if not stars_buy.empty:
                            st.markdown('<span class="status-badge buy" style="background: #fbbf24; color: #000;">⭐ TIER 3 - MAXIMUM CONVICTION</span>', unsafe_allow_html=True)
                            for _, row in stars_buy.head(10).iterrows():
                                st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{row["DisplayName"]}</span><span class="symbol-price"> • ₹{row["Price"]:,.2f}</span></div><span class="symbol-score" style="color: #fbbf24;">{row["Signal"]:.1f}</span></div>', unsafe_allow_html=True)
                            st.markdown("<br>", unsafe_allow_html=True)
                        
                        diamonds_buy = results_df[results_df['Detailed Trigger'].str.contains("💎")].sort_values('Signal')
                        if not diamonds_buy.empty:
                            st.markdown('<span class="status-badge buy">💎 WT CYCLE CROSSED</span>', unsafe_allow_html=True)
                            for _, row in diamonds_buy.head(10).iterrows():
                                st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{row["DisplayName"]}</span><span class="symbol-price"> • ₹{row["Price"]:,.2f}</span></div><span class="symbol-score" style="color: #10b981;">{row["Signal"]:.1f}</span></div>', unsafe_allow_html=True)
                            st.markdown("<br>", unsafe_allow_html=True)

                        circles_buy = results_df[results_df['Detailed Trigger'].str.contains("🟢")].sort_values('Signal')
                        if not circles_buy.empty:
                            st.markdown('<span class="status-badge oversold">🟢 STRONG AGREEMENT BUY</span>', unsafe_allow_html=True)
                            for _, row in circles_buy.head(10).iterrows():
                                st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{row["DisplayName"]}</span><span class="symbol-price"> • ₹{row["Price"]:,.2f}</span></div><span class="symbol-score" style="color: #06b6d4;">{row["Signal"]:.1f}</span></div>', unsafe_allow_html=True)

                        if stars_buy.empty and diamonds_buy.empty and circles_buy.empty:
                            st.markdown('<p style="color: #888888; padding: 1rem;">No buy opportunities detected</p>', unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col_sell:
                        st.markdown('<div class="signal-card sell"><div class="signal-card-header"><span class="signal-card-title">🔴 Sell Opportunities</span></div>', unsafe_allow_html=True)
                        
                        stars_sell = results_df[results_df['Detailed Trigger'].str.contains("⭐")].sort_values('Signal', ascending=False)
                        if not stars_sell.empty:
                            st.markdown('<span class="status-badge sell" style="background: #fbbf24; color: #000;">⭐ TIER 3 - MAXIMUM CONVICTION</span>', unsafe_allow_html=True)
                            for _, row in stars_sell.head(10).iterrows():
                                st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{row["DisplayName"]}</span><span class="symbol-price"> • ₹{row["Price"]:,.2f}</span></div><span class="symbol-score" style="color: #fbbf24;">{row["Signal"]:.1f}</span></div>', unsafe_allow_html=True)
                            st.markdown("<br>", unsafe_allow_html=True)
                        
                        diamonds_sell = results_df[results_df['Detailed Trigger'].str.contains("🔶")].sort_values('Signal', ascending=False)
                        if not diamonds_sell.empty:
                            st.markdown('<span class="status-badge sell">🔶 WT CYCLE CROSSED</span>', unsafe_allow_html=True)
                            for _, row in diamonds_sell.head(10).iterrows():
                                st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{row["DisplayName"]}</span><span class="symbol-price"> • ₹{row["Price"]:,.2f}</span></div><span class="symbol-score" style="color: #ef4444;">{row["Signal"]:.1f}</span></div>', unsafe_allow_html=True)
                            st.markdown("<br>", unsafe_allow_html=True)

                        circles_sell = results_df[results_df['Detailed Trigger'].str.contains("🔴")].sort_values('Signal', ascending=False)
                        if not circles_sell.empty:
                            st.markdown('<span class="status-badge overbought">🔴 STRONG AGREEMENT SELL</span>', unsafe_allow_html=True)
                            for _, row in circles_sell.head(10).iterrows():
                                st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{row["DisplayName"]}</span><span class="symbol-price"> • ₹{row["Price"]:,.2f}</span></div><span class="symbol-score" style="color: #f59e0b;">{row["Signal"]:.1f}</span></div>', unsafe_allow_html=True)

                        if stars_sell.empty and diamonds_sell.empty and circles_sell.empty:
                            st.markdown('<p style="color: #888888; padding: 1rem;">No sell opportunities detected</p>', unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Divergence Alerts
                    div_cols = st.columns(2)
                    with div_cols[0]:
                        bull_divs = results_df[results_df['Detailed Trigger'].str.contains("🔺")]
                        if not bull_divs.empty:
                            st.markdown('<span class="status-badge divergence">🔺 BULLISH DIVERGENCES</span>', unsafe_allow_html=True)
                            for _, row in bull_divs.head(10).iterrows():
                                st.markdown(f'<div class="symbol-row"><span class="symbol-name">{row["DisplayName"]}</span><span style="color: #FFC300;">Price ▼ | Signal ▲</span></div>', unsafe_allow_html=True)
                    with div_cols[1]:
                        bear_divs = results_df[results_df['Detailed Trigger'].str.contains("🔻")]
                        if not bear_divs.empty:
                            st.markdown('<span class="status-badge divergence">🔻 BEARISH DIVERGENCES</span>', unsafe_allow_html=True)
                            for _, row in bear_divs.head(10).iterrows():
                                st.markdown(f'<div class="symbol-row"><span class="symbol-name">{row["DisplayName"]}</span><span style="color: #FFC300;">Price ▲ | Signal ▼</span></div>', unsafe_allow_html=True)
                
                with tab2:
                    st.markdown("##### 🏆 Top 20 Most Oversold")
                    top_oversold = results_df.nsmallest(20, 'Signal')
                    cols_o = ['DisplayName', 'Price', 'Change', 'Signal', 'MSF', 'MMR', 'Zone', 'Detailed Trigger']
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
                        n_neutral = len(results_df[results_df['Zone'] == 'Neutral'])
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
                
                with tab4:
                    st.markdown(f"##### Complete Scan Results ({len(results_df)} stocks) - {analysis_date_str} ({timeframe_label})")
                    
                    filter_col1, filter_col2, filter_col3 = st.columns(3)
                    with filter_col1:
                        zone_filter = st.multiselect("Filter by Zone", ["Oversold", "Neutral", "Overbought"], default=["Oversold", "Neutral", "Overbought"])
                    with filter_col2:
                        signal_filter = st.multiselect("Filter by Direction", ["BUY", "SELL", "-"], default=["BUY", "SELL", "-"])
                    with filter_col3:
                        sort_by = st.selectbox("Sort by", ["Signal", "Change", "Price", "DisplayName"], index=0)
                    
                    filtered_df = results_df[
                        (results_df['Zone'].isin(zone_filter)) & 
                        (results_df['Trigger'].isin(signal_filter))
                    ].sort_values(sort_by, ascending=(sort_by == 'DisplayName'))
                    
                    display_cols = ['DisplayName', 'Price', 'Change', 'Signal', 'MSF', 'MMR', 'Zone', 'Detailed Trigger']
                    display_df = filtered_df[display_cols].copy()
                    display_df.columns = ['Symbol', 'Price', 'Chg %', 'Signal', 'MSF', 'MMR', 'Zone', 'UMA Triggers']
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    csv_data = display_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Full Report (CSV)",
                        data=csv_data,
                        file_name=f"sanket_{universe_title.replace(' ', '_')}_{timeframe_label}_{analysis_date.strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )


def run_timeseries_analysis(universe, selected_index, start_date, end_date, length, roc_len, regime_sensitivity, base_weight, timeframe):
    """Run time series analysis across a date range"""
    
    universe_title = selected_index if universe == "Index Constituents" and selected_index else "F&O Stocks"
    timeframe_label = "Weekly" if timeframe == "Weekly" else "Daily"
    
    # Generate date range
    if timeframe == "Weekly":
        date_range = pd.bdate_range(start=start_date, end=end_date, freq='W-FRI')
    else:
        date_range = pd.bdate_range(start=start_date, end=end_date)
    
    if len(date_range) == 0:
        st.error("No valid trading dates in the selected range.")
        return
    
    st.markdown(f"""
    <div class='info-box'>
        <h4>📈 Time Series Analysis ({universe_title})</h4>
        <p>UMA v3 signal analysis across {len(date_range)} {timeframe_label.lower()} periods.<br>
        <strong>Date Range:</strong> {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("◈ RUN TIME SERIES ANALYSIS", type="primary", key="run_timeseries_btn"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Get stock list
        status_text.markdown(f"**⏳ Fetching {universe_title} stock list...**")
        if universe == "F&O Stocks":
            stock_list, fetch_msg = get_fno_stock_list()
        else:
            stock_list, fetch_msg = get_index_stock_list(selected_index)
        
        if not stock_list:
            st.error(f"Failed to fetch stock list: {fetch_msg}")
            return
        
        # Limit for performance (use top 100 by default for time series)
        stock_list = stock_list[:100]
        total_stocks = len(stock_list)
        
        # Fetch macro data once for the entire range
        days_back_macro = 400 if timeframe == "Weekly" else 300
        macro_df = fetch_macro_data(days_back=days_back_macro + (end_date - start_date).days)
        if timeframe == "Weekly":
            macro_df = macro_df.resample('W-FRI').last().dropna(how='all')
        
        # Initialize results storage
        all_daily_results = []
        
        # Process each date
        for date_idx, analysis_date in enumerate(date_range):
            progress = (date_idx + 1) / len(date_range)
            progress_bar.progress(progress)
            status_text.markdown(f"**⏳ Processing {analysis_date.strftime('%d %b %Y')}... ({date_idx+1}/{len(date_range)})**")
            
            # Fetch data for this date
            days_back = 500 if timeframe == "Weekly" else 300
            data_dict, batch_msg = fetch_batch_data(stock_list, end_date=analysis_date, days_back=days_back)
            
            if not data_dict:
                continue
            
            # Process each stock
            date_results = []
            for ticker in data_dict:
                try:
                    df = data_dict[ticker].copy()
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    
                    # Get target date index
                    analysis_datetime = pd.Timestamp(analysis_date)
                    valid_dates = df.index[df.index <= analysis_datetime]
                    
                    if len(valid_dates) < 50:
                        continue
                    
                    target_idx = len(valid_dates) - 1
                    if target_idx < 0:
                        continue
                    
                    df = df.iloc[:target_idx+1]
                    
                    # Run analysis
                    df = run_full_analysis(df, length, roc_len, regime_sensitivity, base_weight)
                    
                    # Get metrics for this date
                    last_row = df.iloc[-1]
                    
                    # Count signals
                    signals = []
                    has_tier3 = last_row.get('Star_Buy', False) or last_row.get('Star_Sell', False)
                    has_diamond = last_row['Diamond_Buy'] or last_row['Diamond_Sell']
                    has_circle = last_row['Circle_Buy'] or last_row['Circle_Sell']
                    has_triangle = last_row['Triangle_Buy'] or last_row['Triangle_Sell']
                    
                    zone = last_row['Condition']
                    
                    date_results.append({
                        'Date': analysis_date,
                        'Zone': zone,
                        'Tier3': has_tier3,
                        'Diamond': has_diamond,
                        'Circle': has_circle,
                        'Triangle': has_triangle,
                        'Signal': last_row['Unified_Osc'],
                        'MSF': last_row['MSF_Osc'],
                        'MMR': last_row['MMR_Osc'],
                        'Stock': ticker
                    })
                except:
                    pass
            
            if date_results:
                all_daily_results.extend(date_results)
        
        progress_bar.empty()
        status_text.empty()
        
        if not all_daily_results:
            st.warning("No data retrieved. Please check your internet connection or try a different date range.")
            return
        
        # Create time series DataFrame
        ts_df = pd.DataFrame(all_daily_results)
        
        # Aggregate by date
        daily_agg = ts_df.groupby('Date').agg({
            'Signal': 'mean',
            'MSF': 'mean',
            'MMR': 'mean',
            'Tier3': 'sum',
            'Diamond': 'sum',
            'Circle': 'sum',
            'Triangle': 'sum',
            'Stock': 'count'
        }).rename(columns={'Stock': 'Total'})
        
        # Count zones
        zone_counts = ts_df.groupby(['Date', 'Zone']).size().unstack(fill_value=0)
        daily_agg = daily_agg.join(zone_counts)
        
        # Display results
        st.success(f"✅ Time Series Analysis Complete! Analyzed {len(daily_agg)} {timeframe_label.lower()} periods")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ===== Time Series Charts =====
        st.markdown("### 📊 Signal Count Time Series")
        
        # Signal counts by zone
        if 'Zone' in ts_df.columns:
            fig_zones = go.Figure()
            
            # Stacked bar for zones
            zone_cols = [col for col in daily_agg.columns if col in ['Tier3 Buy', 'Tier3 Sell', 'Deep Oversold', 'Deep Overbought', 'Oversold', 'Overbought', 'Neutral']]
            
            for zone_col in zone_cols:
                if zone_col in daily_agg.columns:
                    fig_zones.add_trace(go.Bar(
                        x=daily_agg.index,
                        y=daily_agg[zone_col],
                        name=zone_col,
                        stack='stack'
                    ))
            
            fig_zones.update_layout(
                barmode='stack',
                title="Signal Counts by Zone Over Time",
                xaxis_title="Date",
                yaxis_title="Count",
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='#1A1A1A',
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_zones, use_container_width=True)
        
        # Signal types over time
        fig_types = go.Figure()
        fig_types.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Tier3'], name='Tier 3', line=dict(color='#fbbf24', width=2), fill='tozeroy', fillcolor='rgba(251, 191, 36, 0.2)'))
        fig_types.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Diamond'], name='Diamond', line=dict(color='#10b981', width=2)))
        fig_types.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Circle'], name='Circle', line=dict(color='#06b6d4', width=2)))
        fig_types.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Triangle'], name='Triangle', line=dict(color='#f59e0b', width=2)))
        
        fig_types.update_layout(
            title="Signal Types Over Time",
            xaxis_title="Date",
            yaxis_title="Count",
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#1A1A1A',
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_types, use_container_width=True)
        
        # Average signal over time
        fig_avg = go.Figure()
        fig_avg.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['Signal'], name='Unified Signal', line=dict(color='#FFC300', width=3), fill='tozeroy', fillcolor='rgba(255, 195, 0, 0.1)'))
        fig_avg.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['MSF'], name='MSF', line=dict(color='#06b6d4', width=2), opacity=0.7))
        fig_avg.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['MMR'], name='MMR', line=dict(color='#a78bfa', width=2), opacity=0.7))
        
        fig_avg.add_hline(y=0, line=dict(color='gray', width=1, dash='dash'))
        fig_avg.add_hline(y=-5, line=dict(color='green', width=1, dash='dash'))
        fig_avg.add_hline(y=5, line=dict(color='red', width=1, dash='dash'))
        
        fig_avg.update_layout(
            title="Average Signal Strength Over Time",
            xaxis_title="Date",
            yaxis_title="Signal",
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#1A1A1A',
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_avg, use_container_width=True)
        
        # Heatmap of zones
        if 'Zone' in ts_df.columns:
            zone_pivot = ts_df.pivot_table(index='Date', columns='Zone', values='Stock', aggfunc='count', fill_value=0)
            
            fig_heat = go.Figure(data=go.Heatmap(
                z=zone_pivot.values,
                x=zone_pivot.columns,
                y=zone_pivot.index,
                colorscale='RdYlGn_r',
                showscale=True
            ))
            fig_heat.update_layout(
                title="Zone Distribution Heatmap",
                xaxis_title="Zone",
                yaxis_title="Date",
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='#1A1A1A',
                height=400
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        
        # Summary stats
        st.markdown("### 📋 Time Series Summary")
        
        summary_data = {
            "Metric": ["Total Periods", "Avg Signals/Day", "Peak Tier3", "Peak Diamond", "Peak Circle", "Avg Unified Signal", "Best Day (Buy)", "Worst Day (Sell)"],
            "Value": [
                f"{len(daily_agg)}",
                f"{(daily_agg['Tier3'] + daily_agg['Diamond'] + daily_agg['Circle']).mean():.1f}",
                f"{daily_agg['Tier3'].max()} ({daily_agg['Tier3'].idxmax().strftime('%d %b %Y')})",
                f"{daily_agg['Diamond'].max()} ({daily_agg['Diamond'].idxmax().strftime('%d %b %Y')})",
                f"{daily_agg['Circle'].max()} ({daily_agg['Circle'].idxmax().strftime('%d %b %Y')})",
                f"{daily_agg['Signal'].mean():.2f}",
                f"{daily_agg['Signal'].min():.2f} ({daily_agg['Signal'].idxmin().strftime('%d %b %Y')})",
                f"{daily_agg['Signal'].max():.2f} ({daily_agg['Signal'].idxmax().strftime('%d %b %Y')})"
            ]
        }
        st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)
        
        # Data table
        st.markdown("### 📅 Daily Signal Data")
        st.dataframe(daily_agg.round(2), use_container_width=True)


def main():
    (universe, selected_index, analysis_date, length, roc_len, 
     regime_sensitivity, base_weight, timeframe, analysis_mode, 
     start_date_hist, end_date_hist) = render_sidebar()
    render_header()
    
    if 'screener_run' not in st.session_state:
        st.session_state.screener_run = False
    
    if analysis_mode == "Date Range (Time Series)":
        run_timeseries_analysis(universe, selected_index, start_date_hist, end_date_hist, 
                                 length, roc_len, regime_sensitivity, base_weight, timeframe)
    else:
        run_screener(universe, selected_index, analysis_date, length, roc_len, 
                     regime_sensitivity, base_weight, timeframe)
    
    with st.expander("📖 Signal Interpretation Guide (UMA v3)", expanded=False):
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            st.markdown("""
            <div style='background: rgba(16, 185, 129, 0.1); border: 1px solid var(--success-green); border-radius: 12px; padding: 1rem;'>
                <h4 style='color: #10b981; margin-bottom: 0.5rem;'>💎 Diamond (Confirmed)</h4>
                <p style='color: #888888; font-size: 0.85rem;'>WaveTrend Cross in Extremes</p>
                <p style='color: #EAEAEA; font-size: 0.85rem;'>
                    Marks a confirmed cycle turnaround in deep oversold or overbought territory.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_s2:
            st.markdown("""
            <div style='background: rgba(6, 182, 212, 0.1); border: 1px solid #06b6d4; border-radius: 12px; padding: 1rem;'>
                <h4 style='color: #06b6d4; margin-bottom: 0.5rem;'>⚪ Circle (Oversold/Bought)</h4>
                <p style='color: #888888; font-size: 0.85rem;'>Strong Agreement | Signal > |5|</p>
                <p style='color: #EAEAEA; font-size: 0.85rem;'>
                    Strong directional agreement between internal MSF conditions and Macro MMR factors.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_s3:
            st.markdown("""
            <div style='background: rgba(255, 195, 0, 0.1); border: 1px solid #FFC300; border-radius: 12px; padding: 1rem;'>
                <h4 style='color: #FFC300; margin-bottom: 0.5rem;'>🔺 Divergent (Triangle)</h4>
                <p style='color: #888888; font-size: 0.85rem;'>Oscillator vs Price Action</p>
                <p style='color: #EAEAEA; font-size: 0.85rem;'>
                    A reversal warning indicating momentum, macro or flow are diverging from price.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("""
            <div style='background: rgba(42, 42, 42, 0.4); border: 1px solid #3A3A3A; border-radius: 12px; padding: 1rem;'>
                <h4 style='color: #EAEAEA; margin-bottom: 0.5rem;'>MSF Modulators (Engine)</h4>
                <p style='color: #888888; font-size: 0.85rem;'>
                    <b>Entropy:</b> Scales confidence via signal disorder.<br>
                    <b>Hurst:</b> Tilts weight based on fractal memory (trend vs revert).<br>
                    <b>Vol Struct:</b> Dampens conviction dynamically under market stress.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m2:
            st.markdown("""
            <div style='background: rgba(42, 42, 42, 0.4); border: 1px solid #3A3A3A; border-radius: 12px; padding: 1rem;'>
                <h4 style='color: #EAEAEA; margin-bottom: 0.5rem;'>MMR - Macro Market Regression</h4>
                <p style='color: #888888; font-size: 0.85rem;'>
                    External macro correlation model with global bonds, currencies, and commodities acting as external guardrails.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    run_screener(universe, selected_index, analysis_date, length, roc_len, regime_sensitivity, base_weight, timeframe)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption(f"© {datetime.datetime.now().year} Sanket | Hemrek Capital | {VERSION}")

if __name__ == "__main__":
    main()
