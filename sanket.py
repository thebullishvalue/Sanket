import streamlit as st
import pandas as pd
import yfinance as yf
import time
from datetime import datetime, timedelta
import pandas_ta as ta
import numpy as np
import urllib3
from nsepython import nse_get_advances_declines
import requests
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import base64
from io import BytesIO
import os
import pickle

# --- System Configuration ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
pd.options.mode.chained_assignment = None
pd.set_option('future.no_silent_downcasting', True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Patch for nsepython library bug ---
# The nse_get_advances_declines() function seems to expect a global 'logger'
# This line makes one available for it to use.
logger = logging.getLogger(__name__)
# --- End of Patch ---


# --- Page Configuration ---
st.set_page_config(
    page_title="Sanket | Quantitative Signal Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
VERSION = "v3.2.4 (Match Count)" # <-- UPDATED VERSION
SECTOR_MAP_FILE = "sector_map.pkl"
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
ANALYSIS_UNIVERSE_OPTIONS = ["F&O Stocks", "Index Constituents"]

# --- Premium Professional CSS (Inspired by Pragati) ---
# This CSS from v.py will be used for both models.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        /* Pragati "Glowy Matte" Palette */
        --primary-color: #FFC300; /* Vibrant Gold/Yellow */
        --primary-rgb: 255, 195, 0; /* RGB for Gold/Yellow */
        --background-color: #0F0F0F; /* Near Black */
        --secondary-background-color: #1A1A1A; /* Charcoal */
        --bg-card: #1A1A1A; /* Card Background */
        --bg-elevated: #2A2A2A; /* Hover / Elevated */
        --text-primary: #EAEAEA; /* Light Grey */
        --text-secondary: #EAEAEA; /* Light Grey */
        --text-muted: #888888; /* Grey */
        --border-color: #2A2A2A; /* Darker Border */
        --border-light: #3A3A3A; /* Lighter Border */
        
        /* v.py Semantic Colors */
        --success-green: #10b981;
        --success-dark: #059669;
        --danger-red: #ef4444;
        --danger-dark: #dc2626;
        --warning-amber: #f59e0b;
        --info-cyan: #06b6d4;
        
        /* Signal Colors */
        --extreme-long: #10b981;
        --long: #34d399;
        --div-long: #6ee7b7;
        --extreme-short: #ef4444;
        --short: #f87171;
        --div-short: #fca5a5;
        --neutral: #888888;
    }
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Layout */
    .main, [data-testid="stSidebar"] {
        background-color: var(--background-color);
        color: var(--text-primary);
    }
    
    /* Hide Streamlit Header */
    .stApp > header {
        background-color: transparent;
    }
    
    .block-container {
        padding-top: 1rem; /* Reduced top padding */
        max-width: 1400px;
    }
    
    /* Premium Header */
    .premium-header {
        background: var(--secondary-background-color);
        padding: 1.25rem 2rem; /* Reduced padding */
        border-radius: 16px;
        margin-bottom: 1.5rem; /* Reduced margin */
        box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.1); /* MODIFIED: Reduced glow */
        border: 1px solid var(--border-color);
        position: relative;
        overflow: hidden;
        margin-top: 2.5rem; /* Added margin-top */
    }
    
    .premium-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 20% 50%, rgba(var(--primary-rgb),0.08) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .premium-header h1 {
        margin: 0;
        font-size: 2.50rem; /* Reduced font size */
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.50px;
        position: relative;
        text-shadow: 0 0 10px rgba(var(--primary-rgb), 0.00);
    }
    
    .premium-header .tagline {
        color: var(--text-muted);
        font-size: 1rem; /* Reduced font size */
        margin-top: 0.25rem; /* Reduced margin */
        font-weight: 400;
        position: relative;
    }
    
    /* Premium Metric Cards */
    .metric-card {
        background-color: var(--bg-card);
        padding: 1.25rem; /* Reduced padding */
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08); /* MODIFIED: Reduced glow */
        margin-bottom: 0.5rem; /* Reduced margin */
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
        border-color: var(--border-light);
    }
    
    .metric-card h4 {
        color: var(--text-muted);
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-card h2 {
        color: var(--text-primary);
        font-size: 2rem; /* Reduced font size */
        font-weight: 700;
        margin: 0;
        line-height: 1;
    }
    
    .metric-card .sub-metric {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* v.py Semantic Metric Cards */
    .metric-card.success h2 { color: var(--success-green); }
    .metric-card.danger h2 { color: var(--danger-red); }
    .metric-card.warning h2 { color: var(--warning-amber); }
    .metric-card.info h2 { color: var(--info-cyan); }
    .metric-card.neutral h2 { color: var(--neutral); }

    .metric-card[style*='border-left-color: var(--extreme-long)'] { border-left: 4px solid var(--extreme-long); }
    .metric-card[style*='border-left-color: var(--extreme-short)'] { border-left: 4px solid var(--extreme-short); }
    .metric-card[style*='border-left-color: var(--long)'] { border-left: 4px solid var(--long); }
    .metric-card[style*='border-left-color: var(--short)'] { border-left: 4px solid var(--short); }
    .metric-card[style*='border-left-color: var(--div-long)'] { border-left: 4px solid var(--div-long); }
    .metric-card[style*='border-left-color: var(--div-short)'] { border-left: 4px solid var(--div-short); }
    
    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-badge.success {
        background: linear-gradient(135deg, var(--success-green), var(--success-dark));
        color: white;
    }
    
    .status-badge.danger {
        background: linear-gradient(135deg, var(--danger-red), var(--danger-dark));
        color: white;
    }
    
    .status-badge.warning {
        background: var(--warning-amber);
        color: var(--bg-primary);
    }
    
    .status-badge.info {
        background: var(--info-cyan);
        color: white;
    }
    
    /* Info Boxes */
    .info-box {
        background: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--primary-color);
        padding: 1.25rem;
        border-radius: 12px;
        margin: 0.5rem 0; /* Reduced margin */
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08); /* MODIFIED: Reduced glow */
    }
    
    .info-box h4 {
        color: var(--primary-color);
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
        font-weight: 700;
    }
    
    .info-box p, .info-box ul {
        margin: 0;
        color: var(--text-secondary);
        line-height: 1.6;
    }
    
    /* Welcome Info Box (Tighter) */
    .info-box.welcome h4 {
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    .info-box.welcome p, .info-box.welcome ul {
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .info-box.welcome ul {
        padding-left: 20px;
        margin-top: 0.5rem;
    }
    .info-box.welcome li {
        margin-bottom: 0.25rem;
    }
    
    /* Enhanced Tables */
    .dataframe-container {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    
    /* Style raw HTML tables from .to_html() */
    .stMarkdown table {
        width: 100%;
        border-collapse: collapse;
        background: var(--bg-card);
        border-radius: 16px;
        overflow: hidden; /* To respect border-radius */
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    
    .stMarkdown table th,
    .stMarkdown table td {
        text-align: left !important;
        padding: 12px 10px;
        border-bottom: 1px solid var(--border-color);
    }
    
    .stMarkdown table th {
        background-color: var(--bg-elevated);
        font-size: 0.9rem;
        letter-spacing: 0.5px;
    }

    .stMarkdown table tr:last-child td {
        border-bottom: none;
    }
    
    .stMarkdown table tr:hover {
        background-color: var(--bg-elevated);
    }

    /* --- NEW: Highlight Row Style --- */
    .stMarkdown table tr.highlight-row {
        background-color: var(--primary-color) !important;
        color: var(--background-color) !important;
        font-weight: 700;
    }
    /* Ensure text and spans within the highlighted row are dark */
    .stMarkdown table tr.highlight-row td,
    .stMarkdown table tr.highlight-row td span {
        color: var(--background-color) !important;
    }
    /* --- END NEW --- */
    
    /* Tabs Enhancement */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: var(--background-color);
        padding: 0.5rem;
        border-radius: 16px;
        border: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        color: var(--text-muted);
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--secondary-background-color);
        color: var(--text-primary);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--secondary-background-color);
        color: var(--primary-color);
        border: 1px solid var(--border-color);
        box-shadow: 0 0 10px rgba(var(--primary-rgb), 0.08); /* MODIFIED: Reduced glow */
    }
    
    /* Sidebar Enhancement (Tighter) */
    [data-testid="stSidebar"] {
        background: var(--background-color);
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] .sidebar-content {
        padding: 1rem; /* Reduced padding */
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        margin-bottom: 0.5rem; /* Reduced margin */
        margin-top: 0.5rem; /* Reduced margin */
    }
    
    /* Buttons */
    .stButton>button {
        border: 2px solid var(--primary-color);
        background: transparent;
        color: var(--primary-color);
        font-weight: 700;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.6);
        background: var(--primary-color);
        color: #1A1A1A; /* Dark text on hover for contrast */
        transform: translateY(-2px);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Loading States */
    .stSpinner > div {
        border-top-color: var(--primary-color) !important;
    }
    
    /* Section Dividers (Tighter) */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, var(--border-color) 50%, transparent 100%);
        margin: 1rem 0; /* Reduced margin */
    }
    
    /* Download Links */
    .download-link {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1.5rem;
        border: 2px solid var(--primary-color);
        background: transparent;
        color: var(--primary-color);
        text-decoration: none;
        border-radius: 12px;
        font-weight: 700;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .download-link:hover {
        box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.6);
        background: var(--primary-color);
        color: #1A1A1A; /* Dark text on hover for contrast */
        transform: translateY(-2px);
    }
    
    /* Signal Colors in Tables */
    .signal-extreme-long { color: var(--extreme-long) !important; font-weight: 700; }
    .signal-long { color: var(--long) !important; font-weight: 600; }
    .signal-div-long { color: var(--div-long) !important; font-weight: 600; }
    .signal-extreme-short { color: var(--extreme-short) !important; font-weight: 700; }
    .signal-short { color: var(--short) !important; font-weight: 600; }
    .signal-div-short { color: var(--div-short) !important; font-weight: 600; }
    .signal-neutral { color: var(--neutral) !important; }
    .signal-error { color: var(--warning-amber) !important; }
    
    /* Percentage Colors */
    .pct-positive { color: var(--success-green) !important; font-weight: 600; }
    .pct-negative { color: var(--danger-red) !important; font-weight: 600; }
    .pct-neutral { color: var(--neutral) !important; }
</style>
""", unsafe_allow_html=True)

# --- Premium Header ---
st.markdown(f"""
<div class="premium-header">
    <h1>Sanket | Quantitative Signal Analytics</h1>
</div>
""", unsafe_allow_html=True)

# --- Stock List Functions ---
@st.cache_data(ttl=3600)
def get_fno_stock_list():
    """Fetches the list of F&O stocks."""
    try:
        stock_data = nse_get_advances_declines()
        if not isinstance(stock_data, pd.DataFrame):
            return None, f"API returned unexpected type: {type(stock_data)}"
        
        symbols = None
        if 'SYMBOL' in stock_data.columns:
            symbols = stock_data['SYMBOL'].tolist()
        elif 'symbol' in stock_data.columns:
            symbols = stock_data['symbol'].tolist()
        elif stock_data.index.name in ['SYMBOL', 'symbol']:
            symbols = stock_data.index.tolist()
        else:
            if isinstance(stock_data.index, pd.RangeIndex):
                return None, f"Could not find SYMBOL column"
            elif len(stock_data.index) > 0:
                symbols = stock_data.index.tolist()

        if symbols is None:
             return None, f"Could not extract symbols"
            
        symbols_ns = [str(s) + ".NS" for s in symbols if s and str(s).strip()]
        
        if not symbols_ns:
            return None, "Symbol list empty after cleaning"

        return symbols_ns, f"✓ Fetched {len(symbols_ns)} F&O securities"
            
    except Exception as e:
        return None, f"Error: {e}"

@st.cache_data(ttl=3600)
def get_index_stock_list(index):
    """Fetches the list of stocks for a given index."""
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

@st.cache_data(ttl=300, show_spinner=False)
def fetch_all_data(stock_list, end_date):
    """Downloads historical data for all stocks in the list."""
    # Use a longer buffer to support both models
    buffer_days = 250 
    start_date = end_date - timedelta(days=buffer_days)
    download_end_date = end_date + timedelta(days=1)
    
    try:
        all_data = yf.download(
            stock_list,
            start=start_date,
            end=download_end_date,
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
                        data_dict[ticker] = ticker_df
                except KeyError:
                    logging.warning(f"No data for {ticker}")
            return data_dict, f"✓ Downloaded {len(data_dict)} tickers"

        elif isinstance(all_data, dict):
            valid_data = {t:df for t,df in all_data.items() if not df.empty and not df['Close'].isnull().all()}
            return valid_data, f"✓ Downloaded {len(valid_data)} tickers"

        else:
             return None, "Unexpected data structure"

    except Exception as e:
        return None, f"Download error: {e}"

# --- Sector Map Functions (Identical in both, kept from v.py) ---
@st.cache_resource(show_spinner=False)
def load_sector_map():
    """Loads the persistent sector map from disk."""
    if os.path.exists(SECTOR_MAP_FILE):
        logging.info(f"Loading cached sector map from {SECTOR_MAP_FILE}")
        with open(SECTOR_MAP_FILE, 'rb') as f:
            return pickle.load(f)
    logging.info("No cached sector map found, starting with an empty map.")
    return {}

def save_sector_map(sector_map):
    """Saves the sector map to disk."""
    logging.info(f"Saving updated sector map ({len(sector_map)} entries) to {SECTOR_MAP_FILE}")
    with open(SECTOR_MAP_FILE, 'wb') as f:
        pickle.dump(sector_map, f)

def fetch_sectors_for_list(stock_list):
    """Fetches sector info ONLY for new tickers."""
    logging.info(f"Fetching sector info for {len(stock_list)} new tickers...")
    new_sectors = {}
    
    if not stock_list:
        return new_sectors
        
    tickers = yf.Tickers(stock_list)
    
    for i, ticker_symbol in enumerate(stock_list):
        try:
            info = tickers.tickers[ticker_symbol].info
            sector = info.get('sector')
            new_sectors[ticker_symbol] = sector if sector else "Other"
            
            if (i + 1) % 25 == 0:
                logging.info(f"Fetched {i+1}/{len(stock_list)} new tickers")
                
        except Exception as e:
            logging.warning(f"Could not fetch .info for {ticker_symbol}: {e}")
            new_sectors[ticker_symbol] = "Other"
            
    logging.info("Finished fetching new sectors.")
    return new_sectors

# --- MODEL 1: ILFO Signal Calculation (from v.py) ---
def compute_ilfo_signal(ticker, df, end_date):
    """
    Calculates the ILFO signal for a single ticker.
    FIXED VERSION - Robust handling for Windows/Mac cross-platform consistency.
    """
    
    # Parameters from Pine Script
    adaptiveLength = 21
    microLength = 9
    impactWindow = 5
    devMultiplier = 2.0
    signalSmooth = 5
    divLookback = 10
    volThreshold = 1.2
    
    # Consistent epsilon for division protection
    EPSILON = 1e-10

    def get_error_dict(signal, details, e=""):
        logging.error(f"Error for {ticker}: {details} - {e}")
        return {
            "ticker": ticker, "signal": signal, "details": details, "pct_change": np.nan,
            "ilfo_value": np.nan, "vol_surge": np.nan, "momentum_rsi": np.nan,
            "osc_momentum": np.nan, "osc_accel": np.nan, "volume_score": np.nan,
            "normalized_liq": np.nan
        }
    
    try:
        if df.empty or len(df) < adaptiveLength * 2:
            return get_error_dict("Insufficient Data", "N/A")
        
        # Data Preparation with robust cleaning
        df = df.copy()
        df = df.ffill().bfill()
        
        if df['Close'].isnull().all() or df['Volume'].isnull().all():
            return get_error_dict("Insufficient Data", "Missing main series")
        
        # Ensure positive volumes and handle zeros
        df['Volume'] = df['Volume'].fillna(0).replace(0, 1).clip(lower=1)
        
        # Validate price data is reasonable
        if (df['Close'] <= 0).any():
            return get_error_dict("Invalid Data", "Non-positive prices detected")
        
        # ═══════════════════════════════════════════════════════════
        # 1. MARKET MICROSTRUCTURE (Pine Lines 29-52)
        # ═══════════════════════════════════════════════════════════
        
        bodySize = (df['Close'] - df['Open']).abs()
        spreadProxy = (df['High'] + df['Low']) / 2 - df['Open']
        
        # Robust volume MA with guaranteed positive values
        volMa = ta.sma(df['Volume'], adaptiveLength)
        volMa = volMa.fillna(df['Volume'].mean()).replace(0, EPSILON).clip(lower=EPSILON)
        
        # Calculate spreads with division protection
        vwapSpread = ta.sma(spreadProxy * df['Volume'] / volMa, adaptiveLength)
        vwapSpread = vwapSpread.fillna(0)
        
        priceImpact = ta.sma((df['Close'] - df['Close'].shift(impactWindow)) * df['Volume'] / volMa, adaptiveLength)
        priceImpact = priceImpact.fillna(0)
        
        # Liquidity Score
        liquidityScore = vwapSpread - priceImpact
        liquidityScore = liquidityScore.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Normalized Liquidity with robust standardization
        liqMean = ta.sma(liquidityScore, adaptiveLength).fillna(0)
        liqStdev = ta.stdev(liquidityScore, adaptiveLength)
        liqStdev = liqStdev.fillna(1).replace(0, 1).clip(lower=EPSILON)
        
        normalizedLiq = (liquidityScore - liqMean) / liqStdev
        normalizedLiq = normalizedLiq.replace([np.inf, -np.inf], 0).fillna(0)
        # Clip extreme values to prevent calculation explosions
        normalizedLiq = normalizedLiq.clip(-10, 10)

        # ═══════════════════════════════════════════════════════════
        # 2. VOLUME FLOW ANALYSIS (Pine Lines 57-73)
        # ═══════════════════════════════════════════════════════════
        
        # Volume Surge with robust Z-score
        volStdev = ta.stdev(df['Volume'], microLength)
        volStdev = volStdev.fillna(df['Volume'].std()).replace(0, EPSILON).clip(lower=EPSILON)
        
        volZscore = (df['Volume'] - volMa) / volStdev
        volZscore = volZscore.replace([np.inf, -np.inf], 0).fillna(0).clip(-5, 5)
        
        volSurge = (50 + (volZscore * 20)).clip(0, 100).fillna(50)
        
        # Directional Volume Flow
        volDirection = np.where(df['Close'] > df['Open'], volSurge, -volSurge)
        
        # Accumulation/Distribution Flow
        typicalPrice = (df['High'] + df['Low'] + df['Close']) / 3
        moneyFlow = typicalPrice * df['Volume']
        moneyFlow = moneyFlow.replace([np.inf, -np.inf], 0).fillna(0)
        
        posFlow = ta.sma(moneyFlow * (df['Close'] > df['Close'].shift(1)), microLength)
        negFlow = ta.sma(moneyFlow * (df['Close'] < df['Close'].shift(1)), microLength)
        posFlow = posFlow.fillna(0)
        negFlow = negFlow.fillna(0)
        
        accumFlow = (posFlow - negFlow) / (posFlow + negFlow + EPSILON)
        accumFlow = accumFlow.replace([np.inf, -np.inf], 0).fillna(0).clip(-1, 1)
        
        # Composite Volume Score
        volumeScore = (volDirection / 100) * 0.5 + accumFlow * 0.5
        volumeScore = volumeScore.replace([np.inf, -np.inf], 0).fillna(0)

        # ═══════════════════════════════════════════════════════════
        # 3. MOMENTUM & CONVICTION (Pine Lines 78-98)
        # ═══════════════════════════════════════════════════════════
        
        # Body Conviction with safer rolling apply
        def safe_body_conviction(x):
            try:
                if len(x) < 2 or pd.isna(x.iloc[-1]):
                    return 0.0
                return float((x.iloc[-1] > x.iloc[:-1]).mean() * 100)
            except:
                return 0.0
        
        bodyConviction = bodySize.rolling(window=microLength + 1, min_periods=2).apply(
            safe_body_conviction, raw=False
        ).fillna(0)
        
        # Directional Conviction
        directionConviction = np.where(df['Close'] > df['Open'], bodyConviction, -bodyConviction)
        
        # Price Velocity with extreme value protection
        price_change = df['Close'].diff()
        price_base = df['Close'].shift(1).replace(0, EPSILON).clip(lower=EPSILON)
        
        priceVelocity = (price_change / price_base) * 10000
        priceVelocity = priceVelocity.replace([np.inf, -np.inf], 0).fillna(0)
        # Clip to prevent RSI calculation issues
        priceVelocity = priceVelocity.clip(-1000, 1000)
        
        momentumRsi = ta.rsi(priceVelocity, microLength)
        momentumRsi = momentumRsi.fillna(50).clip(0, 100)

        # ═══════════════════════════════════════════════════════════
        # 4. STATISTICAL BOUNDS (Pine Lines 103-134)
        # ═══════════════════════════════════════════════════════════
        
        priceMean = ta.sma(df['Close'], adaptiveLength).fillna(df['Close'])
        priceStdev = ta.stdev(df['Close'], adaptiveLength)
        priceStdev = priceStdev.fillna(df['Close'].std()).replace(0, EPSILON).clip(lower=EPSILON)
        
        # Statistical Bounds
        upperBound = priceMean + devMultiplier * priceStdev
        lowerBound = priceMean - devMultiplier * priceStdev

        # Zone Detection
        inOverbought = df['Close'] > upperBound
        inOversold = df['Close'] < lowerBound

        # ═══════════════════════════════════════════════════════════
        # 5. COMPOSITE OSCILLATOR (Pine Lines 141-157)
        # ═══════════════════════════════════════════════════════════
        
        # Weighted Component Assembly with clipping
        rawScore = (normalizedLiq * 0.30) + \
                   (volumeScore * 0.25) + \
                   (directionConviction / 100 * 0.25) + \
                   ((momentumRsi - 50) / 50 * 0.20)
        
        rawScore = rawScore.replace([np.inf, -np.inf], 0).fillna(0)
        # Critical: Clip rawScore to prevent oscillator explosion
        rawScore = rawScore.clip(-5, 5)
        
        # Normalize to -8 to +8 range
        oscillator = -(rawScore * 8)
        oscillator = oscillator.replace([np.inf, -np.inf], 0).fillna(0)
        # Final safety clip
        oscillator = oscillator.clip(-10, 10)
        
        signal = ta.sma(oscillator, signalSmooth)
        signal = signal.fillna(0).clip(-10, 10)
        
        # Momentum Metrics with robust diff
        oscMomentum = oscillator.diff(2).fillna(0)
        oscAccel = oscMomentum.diff().fillna(0)
        
        # Clip momentum metrics to reasonable ranges
        oscMomentum = oscMomentum.clip(-15, 15)
        oscAccel = oscAccel.clip(-15, 15)

        # ═══════════════════════════════════════════════════════════
        # 6. DIVERGENCE DETECTION (Pine Lines 160-190)
        # ═══════════════════════════════════════════════════════════
        
        # Pivot Detection with safer rolling window
        def safe_is_pivot_low(x):
            try:
                if len(x) < divLookback * 2 + 1:
                    return 0
                mid = divLookback
                return 1 if x.iloc[mid] == x.min() else 0
            except:
                return 0
        
        def safe_is_pivot_high(x):
            try:
                if len(x) < divLookback * 2 + 1:
                    return 0
                mid = divLookback
                return 1 if x.iloc[mid] == x.max() else 0
            except:
                return 0
        
        price_lows = df['Low'].rolling(window=divLookback*2+1, center=True, min_periods=divLookback+1).apply(
            safe_is_pivot_low, raw=False
        ).fillna(0)
        
        price_highs = df['High'].rolling(window=divLookback*2+1, center=True, min_periods=divLookback+1).apply(
            safe_is_pivot_high, raw=False
        ).fillna(0)
        
        pivot_lows_idx = df.index[price_lows > 0]
        pivot_highs_idx = df.index[price_highs > 0]
        
        # Track last pivot values with forward fill
        df['last_pivot_low_price'] = df.loc[pivot_lows_idx, 'Low'].reindex(df.index).ffill().fillna(df['Low'])
        df['last_pivot_low_osc'] = oscillator.loc[pivot_lows_idx].reindex(df.index).ffill().fillna(oscillator)
        
        df['last_pivot_high_price'] = df.loc[pivot_highs_idx, 'High'].reindex(df.index).ffill().fillna(df['High'])
        df['last_pivot_high_osc'] = oscillator.loc[pivot_highs_idx].reindex(df.index).ffill().fillna(oscillator)

        # Volume Confirmation
        volConfirm = df['Volume'] > volMa * 0.8
        
        # Bullish Divergence with safe comparisons
        priceLL = df['Low'] < (df['last_pivot_low_price'] * 0.998)
        oscHL = oscillator > (df['last_pivot_low_osc'] * 1.05)
        bullishDiv = priceLL & oscHL & inOversold & volConfirm
        
        # Bearish Divergence with safe comparisons
        priceHH = df['High'] > (df['last_pivot_high_price'] * 1.002)
        oscLH = oscillator < (df['last_pivot_high_osc'] * 0.95)
        bearishDiv = priceHH & oscLH & inOverbought & volConfirm

        # ═══════════════════════════════════════════════════════════
        # 7. SIGNAL GENERATION (Pine Lines 202-209)
        # ═══════════════════════════════════════════════════════════
        
        # Extreme Reversal Signals
        extremeLong = inOversold & (oscMomentum > 0) & (oscAccel > 0) & (volSurge > volThreshold * 50)
        extremeShort = inOverbought & (oscMomentum < 0) & (oscAccel < 0) & (volSurge > volThreshold * 50)

        # ═══════════════════════════════════════════════════════════
        # FINAL SIGNAL LOGIC & DATA EXTRACTION
        # ═══════════════════════════════════════════════════════════
        
        analysis_datetime = datetime.combine(end_date, datetime.max.time())
        df.index = pd.to_datetime(df.index)
        target_date = df.index.asof(analysis_datetime)
        
        if pd.isna(target_date):
            return get_error_dict("No Data", f"No data at {end_date.date()}")
        
        # Validate all critical values at target date
        try:
            test_values = [
                oscillator.loc[target_date],
                volSurge.loc[target_date],
                normalizedLiq.loc[target_date],
                momentumRsi.loc[target_date],
                oscMomentum.loc[target_date],
                oscAccel.loc[target_date],
                volumeScore.loc[target_date]
            ]
            
            # Check for invalid values
            for val in test_values:
                if pd.isna(val) or np.isinf(val):
                    return get_error_dict("No Data", "Invalid calculation result")
                    
        except (KeyError, IndexError):
            return get_error_dict("No Data", "Target date not in index")

        # Extract signal flags at target date
        isExtremeLong = bool(extremeLong.loc[target_date])
        isBullishDiv = bool(bullishDiv.loc[target_date])
        isExtremeShort = bool(extremeShort.loc[target_date])
        isBearishDiv = bool(bearishDiv.loc[target_date])

        # Signal hierarchy
        signal_text = "Neutral"
        if isExtremeLong and isBullishDiv:
            signal_text = "Extreme Long"
        elif isExtremeShort and isBearishDiv:
            signal_text = "Extreme Short"
        elif isExtremeLong:
            signal_text = "Long"
        elif isBullishDiv:
            signal_text = "Divergence Long"
        elif isExtremeShort:
            signal_text = "Short"
        elif isBearishDiv:
            signal_text = "Divergence Short"

        # Extract values for table display
        nl_value = float(normalizedLiq.loc[target_date])
        ilfo_value = float(oscillator.loc[target_date])
        vol_value = float(volSurge.loc[target_date])
        mom_rsi_val = float(momentumRsi.loc[target_date])
        osc_mom_val = float(oscMomentum.loc[target_date])
        osc_accel_val = float(oscAccel.loc[target_date])
        vol_score_val = float(volumeScore.loc[target_date])

        # Match count and highlighting logic
        match_count = 0
        
        ilfo_string = f'<span class="signal-neutral">{ilfo_value:.2f}</span>'
        vol_string = f'<span class="signal-neutral">{vol_value:.1f}</span>'
        mom_rsi_string = f'<span class="signal-neutral">{mom_rsi_val:.2f}</span>'
        osc_mom_string = f'<span class="signal-neutral">{osc_mom_val:.2f}</span>'
        osc_accel_string = f'<span class="signal-neutral">{osc_accel_val:.2f}</span>'
        vol_score_string = f'<span class="signal-neutral">{vol_score_val:.2f}</span>'

        if nl_value >= 0:
            nl_string = f'<span class="pct-positive">POS</span>'
        else:
            nl_string = f'<span class="pct-negative">NEG</span>'

        if "Long" in signal_text:
            if 3.10 <= ilfo_value <= 3.60:
                ilfo_string = f'<span class="pct-positive">{ilfo_value:.2f}</span>'
                match_count += 1
            if 73 <= vol_value <= 91:
                vol_string = f'<span class="pct-positive">{vol_value:.1f}</span>'
                match_count += 1
            if 0.000 <= mom_rsi_val <= 38.50:
                mom_rsi_string = f'<span class="pct-positive">{mom_rsi_val:.2f}</span>'
                match_count += 1
            if 1.77 <= osc_mom_val <= 3.48:
                osc_mom_string = f'<span class="pct-positive">{osc_mom_val:.2f}</span>'
                match_count += 1
            if -0.26 <= osc_accel_val <= 1.51:
                osc_accel_string = f'<span class="pct-positive">{osc_accel_val:.2f}</span>'
                match_count += 1
            if -0.46 <= vol_score_val <= 0.45:
                vol_score_string = f'<span class="pct-positive">{vol_score_val:.2f}</span>'
                match_count += 1
                
        elif "Short" in signal_text:
            if -10.35 <= ilfo_value <= -3.75:
                ilfo_string = f'<span class="pct-negative">{ilfo_value:.2f}</span>'
                match_count += 1
            if 79.0 <= vol_value <= 95.60:
                vol_string = f'<span class="pct-negative">{vol_value:.1f}</span>'
                match_count += 1
            if 53.30 <= mom_rsi_val <= 57.08:
                mom_rsi_string = f'<span class="pct-negative">{mom_rsi_val:.2f}</span>'
                match_count += 1
            if -9.77 <= osc_mom_val <= -4.13:
                osc_mom_string = f'<span class="pct-negative">{osc_mom_val:.2f}</span>'
                match_count += 1
            if -12.09 <= osc_accel_val <= -3.99:
                osc_accel_string = f'<span class="pct-negative">{osc_accel_val:.2f}</span>'
                match_count += 1
            if 0.77 <= vol_score_val <= 1.000:
                vol_score_string = f'<span class="pct-negative">{vol_score_val:.2f}</span>'
                match_count += 1

        details_text = (
            f"ILFO: {ilfo_string} | NL: {nl_string} | VolSurge: {vol_string} | MomRSI: {mom_rsi_string}<br>"
            f"OscMom: {osc_mom_string} | OscAcc: {osc_accel_string} | VolScore: {vol_score_string}"
        )
        
        # Calculate percentage change
        try:
            prev_close = df['Close'].shift(1).loc[target_date]
            curr_close = df['Close'].loc[target_date]
            if pd.notna(prev_close) and prev_close > 0:
                pct_change_val = ((curr_close / prev_close) - 1) * 100
            else:
                pct_change_val = np.nan
        except:
            pct_change_val = np.nan
            
        return {
            "ticker": ticker,
            "signal": signal_text,
            "details": details_text,
            "pct_change": pct_change_val,
            "match_count": match_count,
            "ilfo_value": ilfo_value,
            "vol_surge": vol_value,
            "momentum_rsi": mom_rsi_val,
            "osc_momentum": osc_mom_val,
            "osc_accel": osc_accel_val,
            "volume_score": vol_score_val,
            "normalized_liq": nl_value
        }

    except Exception as e:
        return get_error_dict("Error (Calc)", str(e), e)

# --- MODEL 2: ROC & BasisSlope Signal Calculation (from Sanket.py) ---
def compute_roc_slope_signal(ticker, df, end_date):
    """
    Original signal calculation logic from Sanket.py
    Returns: A standardized dictionary.
    """
    rocLength = 14
    length = 20
    delta = 0.95
    multiplier = 2.0
    impact_window = 5
    rsi_length = 14

    # --- NEW: Standardized return dictionary for errors ---
    def get_error_dict(signal, details, e=""):
        logging.error(f"Error for {ticker}: {details} - {e}")
        return {
            "ticker": ticker, "signal": signal, "details": details, "pct_change": np.nan,
            "ilfo_value": np.nan, "vol_surge": np.nan, "momentum_rsi": np.nan,
            "osc_momentum": np.nan, "osc_accel": np.nan, "volume_score": np.nan,
            "normalized_liq": np.nan
        }
    
    try:
        if df.empty or len(df) < length * 2:
            return get_error_dict("Insufficient Data", "N/A") # Standardized
            
        df['rsi'] = ta.rsi(df['Close'], length=rsi_length)
        df['roc'] = ta.roc(df['Close'], length=rocLength)
        
        rocMax = df['roc'].rolling(window=length).max()
        rocMin = df['roc'].rolling(window=length).min()
        scaledRoc = 16 * (df['roc'] - rocMin) / (rocMax - rocMin + 1e-9) - 8
        df['scaledRoc'] = scaledRoc.replace([np.inf, -np.inf], 0).fillna(0) if scaledRoc is not None else 0.0

        # *** MODIFICATION: Ensure NaN volumes are filled AND 0s replaced ***
        df['Volume'] = df['Volume'].fillna(0).replace(0, 1)
        spread = (df['High'] + df['Low']) / 2 - df['Open']
        
        vol_ma = ta.sma(df['Volume'], length)
        vol_ma = vol_ma.replace(0, 1e-9).fillna(1e-9) if vol_ma is not None else pd.Series(1e-9, index=df.index)
        
        vwap_spread = ta.sma((spread * df['Volume'] / vol_ma), length)
        price_impact = ta.sma(((df['Close'] - df['Close'].shift(impact_window)) * df['Volume'] / vol_ma), length)
        
        liquidity_score = vwap_spread.fillna(0) - price_impact.fillna(0)
        
        liq_sma = ta.sma(liquidity_score, length)
        liq_stdev = ta.stdev(liquidity_score, length)
        if liq_sma is None or liq_stdev is None:
            normalized_liq = pd.Series(0.0, index=df.index)
        else:
            normalized_liq = (liquidity_score - liq_sma) / (liq_stdev.replace(0, 1e-9) + 1e-9)

        vol_reg = ta.linreg(liquidity_score, length)
        
        if vol_reg is None or vol_reg.isnull().all():
            normalized_reg = pd.Series(0.0, index=df.index)
        else:
            reg_sma = ta.sma(vol_reg, length)
            reg_stdev = ta.stdev(vol_reg, length)
            if reg_sma is None or reg_stdev is None:
                normalized_reg = pd.Series(0.0, index=df.index)
            else:
                normalized_reg = (vol_reg - reg_sma) / (reg_stdev.replace(0, 1e-9) + 1e-9)

        combined_liquidity = normalized_liq.fillna(0) + normalized_reg.fillna(0)

        so_value = df['Close'] + liquidity_score
        low_value = so_value.rolling(window=length).min()
        high_value = so_value.rolling(window=length).max()
        
        oscillator = 16 * (so_value - low_value) / (high_value - low_value + 1e-9) - 8
        oscillator = oscillator.replace([np.inf, -np.inf], 0).fillna(0) if oscillator is not None else pd.Series(0.0, index=df.index)
        
        df['ad'] = ta.ad(df['High'], df['Low'], df['Close'], df['Volume']).fillna(0)
        
        nv = (df['Close'].diff() > 0).astype(int) * df['Volume'] - (df['Close'].diff() < 0).astype(int) * df['Volume']
        cnv = nv.cumsum()
        cnv_tb = cnv - ta.sma(cnv, length)
        cnv_tb = cnv_tb.fillna(0) if cnv_tb is not None else pd.Series(0.0, index=df.index)

        mean = ta.sma(df['Close'], length)
        stdev = ta.stdev(df['Close'], length)
        if mean is None or stdev is None:
            return get_error_dict("Error (Calc)", "SMA/STDEV failed") # Standardized
            
        non_conformity_score = (df['Close'] - mean).abs()
        threshold = non_conformity_score.rolling(window=length).quantile(delta)
        
        upper_bound = mean + multiplier * stdev
        lower_bound = mean - multiplier * stdev

        lowerBandSlope = lower_bound - lower_bound.shift(5)
        upperBandSlope = upper_bound - upper_bound.shift(5)
        basis = (lowerBandSlope + upperBandSlope) / 2
        basisSlope = basis - basis.shift(5)

        basisSlope_sma = ta.sma(basisSlope, 5).fillna(0)
        
        basisSlope_trend = (basisSlope > basisSlope_sma).astype(int) - (basisSlope < basisSlope_sma).astype(int)
        basisSlope_acceleration = basisSlope.diff(1)
        
        basis_stdev = ta.stdev(basisSlope, length).replace(0, 0.01).fillna(0.01)
        basisSlope_magnitude = basisSlope.abs() / (basis_stdev + 0.01)
        
        basisSlope_divergence = (df['Close'].diff(1) > 0).astype(int) & (basisSlope.diff(1) < 0).astype(int) * -1 + \
                                (df['Close'].diff(1) < 0).astype(int) & (basisSlope.diff(1) > 0).astype(int) * 1
        basisSlope_reversal_signal = (basisSlope > basisSlope_sma).astype(int) & (basisSlope.shift(1) < basisSlope_sma.shift(1)).astype(int) * 1 + \
                                     (basisSlope < basisSlope_sma).astype(int) & (basisSlope.shift(1) > basisSlope_sma.shift(1)).astype(int) * -1

        basisSlope_score = (basisSlope_trend * 0.3 + 
                            (basisSlope_acceleration / (basisSlope.abs().replace(0, 1e-9))) * 0.2 + 
                            basisSlope_magnitude * 0.2 + 
                            basisSlope_divergence * 0.2 + 
                            basisSlope_reversal_signal * 0.3)

        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        positive_money_flow = ta.sma(money_flow * (df['Close'] > df['Close'].shift(1)), 10).fillna(0)
        negative_money_flow = ta.sma(money_flow * (df['Close'] < df['Close'].shift(1)), 10).fillna(0)
        
        accumulation_score = positive_money_flow / (positive_money_flow + negative_money_flow + 1e-9)

        signal_score = ((df['rsi'] < 40).astype(int) - (df['rsi'] > 70).astype(int) +
                        (basisSlope_score > 0.75).astype(int) - (basisSlope_score < -0.75).astype(int) +
                        (accumulation_score > 0.65).astype(int) - (accumulation_score < 0.35).astype(int) +
                        (df['Close'] < lower_bound).astype(int) - (df['Close'] > upper_bound).astype(int) +
                        (normalized_liq > 0).astype(int) - (normalized_liq < 0).astype(int))

        liquidity = basisSlope_score - (combined_liquidity + signal_score)

        lowest_value = liquidity.rolling(window=length).min()
        highest_value = liquidity.rolling(window=length).max()
        
        liq_osc = 16 * (liquidity - lowest_value) / (highest_value - lowest_value + 1e-9) - 8
        liq_osc = liq_osc.replace([np.inf, -np.inf], 0).fillna(0) if liq_osc is not None else pd.Series(0.0, index=df.index)

        data_sell = df['Close'] > upper_bound
        data_buy = df['Close'] < lower_bound

        signal_direction = (
            (signal_score >= 1) & (liq_osc < -7.5) & data_buy
        ).astype(int) * 1 + (
            (signal_score <= -1) & (liq_osc > 7.5) & data_sell
        ).astype(int) * -1

        # --- Standardized Date & % Change Logic ---
        analysis_datetime = datetime.combine(end_date, datetime.max.time())
        df.index = pd.to_datetime(df.index)
        target_date = df.index.asof(analysis_datetime)
        
        if pd.isna(target_date):
             return get_error_dict("No Data", f"No data at {end_date.date()}")
        
        # Check for critical data at target_date
        critical_cols_roc = [
            df['rsi'], scaledRoc, df['ad'], df['Volume'], df['Close'], df['High'], df['Low'], df['Open'], 
            signal_score, liq_osc, signal_direction, df['Close'].shift(1)
        ]
        if any(pd.isna(col.loc[target_date]) for col in critical_cols_roc):
            return get_error_dict("No Data", "Incomplete calc data")

        final_signal_val = signal_direction.loc[target_date]
        
        if final_signal_val == 1:
            signal_text = "Buy"
        elif final_signal_val == -1:
            signal_text = "Sell"
        else:
            signal_text = "Neutral"
            
        # --- NEW: Get, format, and combine all parameters for Details ---
        rsi_val = df.loc[target_date, 'rsi']
        sig_score_val = signal_score.loc[target_date]
        liq_osc_val = liq_osc.loc[target_date]
        nl_val = normalized_liq.loc[target_date]

        # --- NEW: Conditional Highlighting Logic for ROC Model ---
        match_count = 0 # <-- Initialize match count
        
        # Default neutral strings
        # Note: Using .2f for nl_val as it's a float, not POS/NEG string
        rsi_string = f'<span class="signal-neutral">{rsi_val:.1f}</span>'
        sig_score_string = f'<span class="signal-neutral">{sig_score_val:.2f}</span>'
        liq_osc_string = f'<span class="signal-neutral">{liq_osc_val:.2f}</span>'
        # nl_string will be set below
        
        # --- NEW: Independent NL highlighting (mirroring ILFO - REVERTED) ---
        if nl_val >= 0: # Reverted: A positive value is POS
            nl_string = f'<span class="pct-positive">POS</span>'
        else: # nl_val < 0
            nl_string = f'<span class="pct-negative">NEG</span>'

        if signal_text == "Buy":
            # Apply GREEN highlighting if criteria are met (using mapped params)
            if 3.10 <= liq_osc_val <= 3.60: # liq_osc -> ilfo_value
                liq_osc_string = f'<span class="pct-positive">{liq_osc_val:.2f}</span>'; match_count += 1
            # if nl_val < 0: # <-- REMOVED FROM COUNT
            #     match_count += 1
            if 0.000 <= rsi_val <= 38.50: # rsi -> momentum_rsi
                rsi_string = f'<span class="pct-positive">{rsi_val:.1f}</span>'; match_count += 1
            if -0.46 <= sig_score_val <= 0.45: # sig_score -> volume_score
                sig_score_string = f'<span class="pct-positive">{sig_score_val:.2f}</span>'; match_count += 1
                
        elif signal_text == "Sell":
            # Apply RED highlighting if criteria are met (using mapped params)
            if -10.35 <= liq_osc_val <= -3.75: # liq_osc -> ilfo_value
                liq_osc_string = f'<span class="pct-negative">{liq_osc_val:.2f}</span>'; match_count += 1
            # if nl_val >= 0: # <-- REMOVED FROM COUNT
            #     match_count += 1
            if 53.30 <= rsi_val <= 57.08: # rsi -> momentum_rsi
                rsi_string = f'<span class="pct-negative">{rsi_val:.1f}</span>'; match_count += 1
            if 0.77 <= sig_score_val <= 1.000: # sig_score -> volume_score
                sig_score_string = f'<span class="pct-negative">{sig_score_val:.2f}</span>'; match_count += 1
        
        # --- END NEW Logic ---

        details_text = (
            f"RSI: {rsi_string} | SigScore: {sig_score_string} | LiqOsc: {liq_osc_string} | NL: {nl_string}"
        )
        
        pct_change = df.loc[target_date, 'Close'] / df['Close'].shift(1).loc[target_date] - 1
        pct_change_val = (pct_change * 100) if pd.notna(pct_change) else np.nan
            
        # --- NEW: Return full dictionary ---
        return {
            "ticker": ticker, 
            "signal": signal_text, 
            "details": details_text, 
            "pct_change": pct_change_val,
            "match_count": match_count, # <-- Add match count
            # Add placeholders for ILFO columns so table structure is consistent
            "ilfo_value": liq_osc.loc[target_date], # Use LiqOsc as a stand-in
            "vol_surge": np.nan, 
            "momentum_rsi": df.loc[target_date, 'rsi'], # Use RSI as a stand-in
            "osc_momentum": np.nan, 
            "osc_accel": np.nan, 
            "volume_score": signal_score.loc[target_date], # Use SigScore as a stand-in
            "normalized_liq": normalized_liq.loc[target_date] # Use normalized_liq
        }

    except Exception as e:
        return get_error_dict("Error (Calc)", str(e), e)

# --- UI & Utility Functions ---

def format_dataframe_for_display(df):
    """
    Format dataframe with proper HTML for colored display
    UPDATED: Now returns a DataFrame with HTML strings in cells, not a full HTML table.
    """
    if df.empty:
        return df
    
    # Create a copy to avoid modifying original
    display_df = df.copy()
    
    # Format Signal column with colors
    if 'Signal' in display_df.columns:
        def format_signal(val):
            if pd.isna(val):
                return '<span class="signal-neutral">N/A</span>'
            val_str = str(val)
            
            # ILFO Signals
            if val_str == "Extreme Long":
                return f'<span class="signal-extreme-long">{val_str}</span>'
            elif val_str == "Long":
                return f'<span class="signal-long">{val_str}</span>'
            elif val_str == "Divergence Long":
                return f'<span class="signal-div-long">{val_str}</span>'
            
            # ROC & BasisSlope Signals (Mapped to ILFO styles)
            elif val_str == "Buy":
                return f'<span class="signal-long">{val_str}</span>' # Use "Long" style
                
            # ILFO Signals
            elif val_str == "Extreme Short":
                return f'<span class="signal-extreme-short">{val_str}</span>'
            elif val_str == "Short":
                return f'<span class="signal-short">{val_str}</span>'
            elif val_str == "Divergence Short":
                return f'<span class="signal-div-short">{val_str}</span>'
                
            # ROC & BasisSlope Signals (Mapped to ILFO styles)
            elif val_str == "Sell":
                return f'<span class="signal-short">{val_str}</span>' # Use "Short" style

            # Common
            elif "Error" in val_str or "Data" in val_str:
                return f'<span class="signal-error">{val_str}</span>'
            return f'<span class="signal-neutral">{val_str}</span>'
        
        display_df['Signal'] = display_df['Signal'].apply(format_signal)
    
    # Format % Change column with colors
    if '% Change' in display_df.columns:
        def format_pct(val):
            if pd.isna(val):
                return '<span class="pct-neutral">N/A</span>'
            if val > 0:
                return f'<span class="pct-positive">+{val:.2f}%</span>'
            elif val < 0:
                return f'<span class="pct-negative">{val:.2f}%</span>'
            else:
                return f'<span class="pct-neutral">{val:.2f}%</span>'
        
        display_df['% Change'] = display_df['% Change'].apply(format_pct)
    
    # Reset index to make 'Ticker' a column
    display_df = display_df.reset_index()
    
    # --- UPDATED: Return DataFrame, not HTML ---
    return display_df

def create_export_link(df, filename):
    """Create downloadable CSV link"""
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-link">📥 Download CSV Report</a>'
    return href

def calculate_market_health(buy_signals, sell_signals, error_rate):
    """Calculate overall market health score"""
    if buy_signals + sell_signals == 0:
        return "Unknown", 0
    
    ratio = buy_signals / (sell_signals + 1)
    signal_quality = 100 - (error_rate * 100)
    
    if ratio > 1.5 and signal_quality > 80:
        return "Bullish", 85
    elif ratio > 1.2 and signal_quality > 70:
        return "Moderately Bullish", 70
    elif ratio > 0.8 and signal_quality > 60:
        return "Neutral", 50
    elif ratio > 0.5 and signal_quality > 60:
        return "Moderately Bearish", 35
    else:
        return "Bearish", 20

# --- Main Analysis Function (HEAVILY UPDATED) ---
def run_analysis(analysis_universe, selected_index, analysis_date, selected_model):
    """Main analysis orchestrator, now model-aware"""
    
    if analysis_universe == "F&O Stocks":
        analysis_title = "F&O Stocks"
        logging.info(f"🔍 Analyzing {analysis_title}...")
        logging.info("📡 Fetching F&O stock list from NSE...")
        stock_list, fetch_msg = get_fno_stock_list()
    else:
        analysis_title = selected_index
        logging.info(f"🔍 Analyzing {analysis_title}...")
        logging.info(f"📡 Fetching constituents for {selected_index}...")
        stock_list, fetch_msg = get_index_stock_list(selected_index)
    
    if not stock_list:
        st.error(f"Failed to fetch stock list: {fetch_msg}")
        st.stop()
        
    logging.info(fetch_msg)
    
    # --- Dynamic Model Setup ---
    if selected_model == "ILFO (Pine v6 Rebuild)":
        compute_function = compute_ilfo_signal
        signal_types = [
            "Extreme Long", "Long", "Divergence Long",
            "Extreme Short", "Short", "Divergence Short",
            "Neutral", "Error"
        ]
        # Lambda function to aggregate granular signals
        get_buy_sell_counts = lambda counts: (
            counts["Extreme Long"] + counts["Long"] + counts["Divergence Long"],
            counts["Extreme Short"] + counts["Short"] + counts["Divergence Short"]
        )
    elif selected_model == "ROC & BasisSlope":
        compute_function = compute_roc_slope_signal
        signal_types = ["Buy", "Sell", "Neutral", "Error"]
        # Lambda function for simple signals
        get_buy_sell_counts = lambda counts: (
            counts.get("Buy", 0),
            counts.get("Sell", 0)
        )
    # --- End Dynamic Model Setup ---

    # Sector Map Logic (Unchanged)
    logging.info(f"📡 Loading persistent sector map...")
    sector_map = load_sector_map()
    required_tickers = set(stock_list)
    cached_tickers = set(sector_map.keys())
    missing_tickers = list(required_tickers - cached_tickers)
    
    if missing_tickers:
        logging.info(f"New tickers found. Fetching sector data for {len(missing_tickers)} stocks...")
        new_sector_data = fetch_sectors_for_list(missing_tickers)
        sector_map.update(new_sector_data)
        save_sector_map(sector_map)
        logging.info(f"✓ Sector map updated and saved.")
    else:
        logging.info(f"✓ All sectors found in cache.")

    # Data Download (Unchanged)
    logging.info(f"⬇️ Downloading historical data for {len(stock_list)} stocks...")
    all_data_dict, batch_msg = fetch_all_data(stock_list, analysis_date)
    
    if all_data_dict is None:
        st.error(f"Failed to download data: {batch_msg}")
        st.stop()
    
    logging.info(batch_msg)
    
    # Calculate Signals
    total_stocks = len(stock_list)
    
    results = []
    
    # Use dynamic signal types
    signal_counts = {sig: 0 for sig in signal_types}
    
    sector_signals = {}
    
    valid_tickers = list(all_data_dict.keys())
    total_to_process = len(valid_tickers)
    
    for i, ticker in enumerate(valid_tickers):
        ticker_df = all_data_dict[ticker]
        
        # --- NEW: Use standardized dictionary output ---
        result_dict = compute_function(ticker, ticker_df, analysis_date)
        
        signal = result_dict["signal"]
        sector = sector_map.get(ticker, "Other") 

        if sector not in sector_signals:
            sector_signals[sector] = {sig: 0 for sig in signal_types}
        
        # Add sector to the dict and append
        result_dict["Sector"] = sector
        results.append(result_dict)
        # --- END NEW ---
        
        if signal in signal_counts:
            signal_counts[signal] += 1
            if signal not in sector_signals[sector]:
                 sector_signals[sector][signal] = 0
            sector_signals[sector][signal] += 1
        elif "Error" in signal or "Data" in signal:
            signal_counts["Error"] += 1
            if "Error" not in sector_signals[sector]:
                 sector_signals[sector]["Error"] = 0
            sector_signals[sector]["Error"] += 1
        else:
            signal_counts["Neutral"] += 1
            if "Neutral" not in sector_signals[sector]:
                 sector_signals[sector]["Neutral"] = 0
            sector_signals[sector]["Neutral"] += 1
        
        if (i + 1) % 50 == 0: # Log progress every 50 stocks
            logging.info(f"Analyzing {ticker} ({i+1}/{total_to_process})...")
        
    download_errors = total_stocks - total_to_process
    signal_counts["Error"] += download_errors

    logging.info("✅ Analysis Complete!")
    
    # Calculate Metrics
    # --- NEW: Create DataFrame from list of dicts ---
    results_df = pd.DataFrame(results)
    
    # Rename columns to match old schema
    results_df = results_df.rename(columns={
        "ticker": "Ticker", 
        "signal": "Signal", 
        "pct_change": "% Change", 
        "details": "Details",
        "match_count": "Count", # <-- Renamed
    })
    results_df = results_df.set_index("Ticker")
    
    # Define column order, putting new metrics at the end
    display_columns = ["Signal", "% Change", "Details", "Count"] # <-- Renamed
    criteria_columns = [
        "ilfo_value", "vol_surge", "momentum_rsi", 
        "osc_momentum", "osc_accel", "volume_score", "normalized_liq"
    ]
    all_columns = display_columns + criteria_columns
    
    # Ensure all columns exist (for safety)
    for col in all_columns:
        if col not in results_df.columns:
            results_df[col] = np.nan
            
    results_df = results_df[all_columns]
    # --- END NEW ---
    
    # --- Dynamic Signal Aggregation ---
    total_buy_signals, total_sell_signals = get_buy_sell_counts(signal_counts)
    total_neutral_signals = signal_counts["Neutral"]
    total_error_stocks = signal_counts["Error"]
    total_signals = total_buy_signals + total_sell_signals + total_neutral_signals
    # --- End Dynamic Aggregation ---

    try:
        ratio = total_buy_signals / total_sell_signals if total_sell_signals > 0 else float('inf')
        ratio_text = f"{ratio:.2f}" if ratio != float('inf') else "∞"
    except:
        ratio = 0
        ratio_text = "0.00"
        
    if ratio > 1.2: ratio_class = "success"
    elif ratio > 0.8: ratio_class = "neutral"
    else: ratio_class = "danger"

    error_rate = total_error_stocks / total_stocks if total_stocks > 0 else 0
    market_sentiment, health_score = calculate_market_health(total_buy_signals, total_sell_signals, error_rate)

    # Filter Dataframes (Updated to be more generic)
    buy_df = results_df[results_df['Signal'].str.contains("Long|Buy", na=False)].copy()
    sell_df = results_df[results_df['Signal'].str.contains("Short|Sell", na=False)].copy()
    errors_df = results_df[results_df['Signal'].fillna('').astype(str).str.contains("Error|Data")].copy()

    # --- NEW: Define styling function and hidden columns ---
    # These highlight functions are a 1:1 match for the 'tableSignal' logic
    # in the Pine Script (lines 247-264)
    def highlight_long_criteria(row):
        """Applies 'highlight-row' class if all LONG criteria are met."""
        try:
            criteria = (
                (3.062 <= row['ilfo_value'] <= 3.692) and
                (row['normalized_liq'] >= 0) and # <-- REVERTED LOGIC
                (72.350 <= row['vol_surge'] <= 90.373) and
                (39.047 <= row['momentum_rsi'] <= 42.789) and
                (1.363 <= row['osc_momentum'] <= 3.701) and
                (0.833 <= row['osc_accel'] <= 3.788) and
                (-0.589 <= row['volume_score'] <= -0.452) 
            )
            # Apply style to all cells in the row
            style = 'background-color: var(--primary-color); color: var(--background-color); font-weight: 700;'
            return [style if criteria else '' for _ in row]
        except:
            return ['' for _ in row] # Return no style if data is bad

    def highlight_short_criteria(row):
        """Applies 'highlight-row' class if all SHORT criteria are met."""
        try:
            criteria = (
                (-3.936 <= row['ilfo_value'] <= -3.220) and
                (row['normalized_liq'] < 0) and # <-- REVERTED LOGIC
                (77.883 <= row['vol_surge'] <= 96.083) and
                (56.678 <= row['momentum_rsi'] <= 60.686) and
                (-6.542 <= row['osc_momentum'] <= -4.154) and
                (-7.139 <= row['osc_accel'] <= -4.184) and
                (0.526 <= row['volume_score'] <= 0.655)
            )
            # Apply style to all cells in the row
            style = 'background-color: var(--danger-red); color: var(--text-primary); font-weight: 700;'
            return [style if criteria else '' for _ in row]
        except:
            return ['' for _ in row] # Return no style if data is bad

    # --- END NEW ---


    # --- UI DISPLAY (NOW DYNAMIC) ---
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Tabs
    tab_dash, tab_sector, tab_buy, tab_sell, tab_all, tab_errors = st.tabs([
        f"📊 Dashboard", 
        f"🏢 Sector Analysis",
        f"⬆️ All Long/Buy ({total_buy_signals})", 
        f"⬇️ All Short/Sell ({total_sell_signals})", 
        "📋 All Signals", 
        f"⚠️ Errors ({total_error_stocks})"
    ])

    with tab_dash:
        st.markdown("### Key Metrics")
        
        # This section is universal and works for both models
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"<div class='metric-card success'><h4>⬆️ Total Long/Buy</h4><h2>{total_buy_signals:,}</h2><div class='sub-metric'>All Bullish Signals</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card danger'><h4>⬇️ Total Short/Sell</h4><h2>{total_sell_signals:,}</h2><div class='sub-metric'>All Bearish Signals</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card neutral'><h4>➖ Neutral</h4><h2>{total_neutral_signals:,}</h2><div class='sub-metric'>No Clear Signal</div></div>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<div class='metric-card {ratio_class}'><h4>📈 Long/Short Ratio</h4><h2>{ratio_text}</h2><div class='sub-metric'>{'Bullish' if ratio > 1.2 else 'Bearish' if ratio < 0.8 else 'Balanced'}</div></div>", unsafe_allow_html=True)

        # Get theme colors with fallbacks
        try:
            bg_color = st.config.get_option('theme.backgroundColor') or 'rgba(0,0,0,0)'
            text_color = st.config.get_option('theme.textColor') or '#EAEAEA'
            border_color = st.config.get_option('theme.fadedText') or '#2A2A2A'
            grid_color = st.config.get_option('theme.fadedText') or '#2A2A2A'
        except RuntimeError:
            bg_color = 'rgba(0,0,0,0)'
            text_color = '#EAEAEA'
            border_color = '#2A2A2A'
            grid_color = '#2A2A2A'
            
        # --- Model-Specific Dashboard ---
        if selected_model == "ILFO (Pine v6 Rebuild)": # <-- FIX: Was "ILFO (Pine v6)"
            st.markdown("### Granular Signal Distribution")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"<div class='metric-card' style='border-left-color: var(--extreme-long);'><h4>🔥 Extreme Long</h4><h2 style='color: var(--extreme-long);'>{signal_counts['Extreme Long']:,}</h2></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card' style='border-left-color: var(--extreme-short);'><h4>📉 Extreme Short</h4><h2 style='color: var(--extreme-short);'>{signal_counts['Extreme Short']:,}</h2></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='metric-card' style='border-left-color: var(--long);'><h4>🟢 Long</h4><h2 style='color: var(--long);'>{signal_counts['Long']:,}</h2></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card' style='border-left-color: var(--short);'><h4>🔴 Short</h4><h2 style='color: var(--short);'>{signal_counts['Short']:,}</h2></div>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div class='metric-card' style='border-left-color: var(--div-long);'><h4>🔎 Div. Long</h4><h2 style='color: var(--div-long);'>{signal_counts['Divergence Long']:,}</h2></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card' style='border-left-color: var(--div-short);'><h4>🔎 Div. Short</h4><h2 style='color: var(--div-short);'>{signal_counts['Divergence Short']:,}</h2></div>", unsafe_allow_html=True)

            st.markdown("### Signal Breakdown")
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                labels = ['Total Long', 'Total Short', 'Neutral']
                values = [total_buy_signals, total_sell_signals, total_neutral_signals]
                colors = ['#10b981', '#ef4444', '#888888']
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=labels, values=values,
                    marker=dict(colors=colors, line=dict(color=border_color, width=3)),
                    hole=.4, textinfo='label+percent', textfont_size=14,
                    pull=[0.1 if l == 'Total Long' else 0.05 if l == 'Total Short' else 0 for l in labels]
                )])
                fig_pie.update_layout(
                    title="Aggregate Signal Breakdown", template="plotly_dark",
                    paper_bgcolor=bg_color, plot_bgcolor=bg_color,
                    height=350, showlegend=True, font=dict(color=text_color)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_chart2:
                labels = ["Ext. Long", "Long", "Div. Long", "Ext. Short", "Short", "Div. Short"]
                values = [
                    signal_counts["Extreme Long"], signal_counts["Long"], signal_counts["Divergence Long"],
                    signal_counts["Extreme Short"], signal_counts["Short"], signal_counts["Divergence Short"]
                ]
                colors = ['#10b981', '#34d399', '#6ee7b7', '#ef4444', '#f87171', '#fca5a5']
                
                fig_bar = go.Figure(data=[go.Bar(
                    x=labels, y=values, marker_color=colors,
                    text=values, textposition='outside', textfont=dict(size=14, color=text_color)
                )])
                fig_bar.update_layout(
                    title="Granular Signal Counts", template="plotly_dark",
                    paper_bgcolor=bg_color, plot_bgcolor=bg_color,
                    height=350, showlegend=False, font=dict(color=text_color),
                    yaxis=dict(title="Count", gridcolor=grid_color),
                    xaxis=dict(title="Signal Type")
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        elif selected_model == "ROC & BasisSlope":
            st.markdown("### Signal Breakdown")
            
            labels = ['Buy', 'Sell', 'Neutral']
            values = [total_buy_signals, total_sell_signals, total_neutral_signals]
            colors = ['#10b981', '#ef4444', '#888888'] # Using v.py colors
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=labels, values=values,
                marker=dict(colors=colors, line=dict(color=border_color, width=3)),
                hole=.4, textinfo='label+percent', textfont_size=14,
                pull=[0.1 if l == 'Buy' else 0.05 if l == 'Sell' else 0 for l in labels]
            )])
            fig_pie.update_layout(
                title="Aggregate Signal Breakdown", template="plotly_dark",
                paper_bgcolor=bg_color, plot_bgcolor=bg_color,
                height=400, showlegend=True, font=dict(color=text_color)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        # --- End Model-Specific Dashboard ---

    with tab_sector:
        st.markdown("### 🏢 Sector-wise Signal Distribution")
        
        sector_df = pd.DataFrame(sector_signals).T
        if "Other" in sector_df.index:
            other_row = sector_df.loc["Other"]
            sector_df = sector_df.drop("Other")
            sector_df.loc["Other"] = other_row
            
        sector_df['Total'] = sector_df.sum(axis=1)
        sector_df = sector_df.sort_values('Total', ascending=False)
        
        fig_sector = go.Figure()

        display_cols = [] # <-- FIX: Initialize variable to prevent UnboundLocalError

        # --- Dynamic Sector Chart ---
        if selected_model == "ILFO (Pine v6 Rebuild)":
            sector_df['Total Long'] = sector_df["Extreme Long"] + sector_df["Long"] + sector_df["Divergence Long"]
            sector_df['Total Short'] = sector_df["Extreme Short"] + sector_df["Short"] + sector_df["Divergence Short"]
            
            fig_sector.add_trace(go.Bar(
                name='Total Long', x=sector_df.index, y=sector_df['Total Long'], marker_color='#10b981'
            ))
            fig_sector.add_trace(go.Bar(
                name='Total Short', x=sector_df.index, y=sector_df['Total Short'], marker_color='#ef4444'
            ))
            fig_sector.add_trace(go.Bar(
                name='Neutral', x=sector_df.index, y=sector_df['Neutral'], marker_color='#888888'
            ))
            display_cols = [
                "Total Long", "Total Short", "Neutral",
                "Extreme Long", "Long", "Divergence Long",
                "Extreme Short", "Short", "Divergence Short",
                "Total"
            ]

        elif selected_model == "ROC & BasisSlope":
            fig_sector.add_trace(go.Bar(
                name='Buy', x=sector_df.index, y=sector_df['Buy'], marker_color='#10b981'
            ))
            fig_sector.add_trace(go.Bar(
                name='Sell', x=sector_df.index, y=sector_df['Sell'], marker_color='#ef4444'
            ))
            fig_sector.add_trace(go.Bar(
                name='Neutral', x=sector_df.index, y=sector_df['Neutral'], marker_color='#888888'
            ))
            display_cols = ["Buy", "Sell", "Neutral", "Total"]
        # --- End Dynamic Sector Chart ---

        fig_sector.update_layout(
            barmode='stack', title="Aggregate Signals by Sector", template="plotly_dark",
            paper_bgcolor=bg_color, plot_bgcolor=bg_color,
            height=500, font=dict(color=text_color),
            yaxis=dict(title="Signal Count", gridcolor=grid_color), xaxis=dict(title="Sector"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_sector, use_container_width=True)
        
        st.markdown("### 📋 Detailed Sector Breakdown")
        sector_display_cols = [col for col in display_cols if col in sector_df.columns]
        sector_display = sector_df[sector_display_cols].copy()
        st.dataframe(sector_display, use_container_width=True, height=400)

    # --- NEW: Define helper to render styled HTML ---
    def render_styled_html(df, highlight_func=None):
        """Applies formatting, styling, and renders HTML."""
        # 1. Apply cell-level HTML formatting
        formatted_df = format_dataframe_for_display(df)
        
        # 2. Start styling from the formatted DF
        styler = formatted_df.style
        
        # 3. Apply row-level highlighting if function is provided
        if highlight_func:
            # Pass subset to avoid styling the index if it were visible
            styler = styler.apply(highlight_func, axis=1, subset=formatted_df.columns)
            
        # 4. Hide the criteria columns (USER REQUEST: DISABLED HIDING)
        # We need to find which criteria columns actually exist in the formatted_df
        cols_to_hide = [col for col in criteria_columns if col in formatted_df.columns]
        if cols_to_hide:
            # --- FIX: Changed syntax from hide(columns=...) to hide(..., axis=...) ---
            styler = styler.hide(cols_to_hide, axis='columns')
            
        # 5. Set table class for CSS and hide index
        styler = styler.set_table_attributes('class="stMarkdown table"').hide(axis="index")
        
        # 6. Render to HTML
        # Formatting is now done within the 'Details' column string
        
        return styler.to_html(escape=False)

    with tab_buy:
        st.markdown(f"### ⬆️ All Long / Buy Signals ({total_buy_signals})")
        if not buy_df.empty:
            # --- NEW: Use styler to render ---
            buy_df_sorted = buy_df.sort_values("Signal")
            # Pass the highlight function *only* to this tab
            html_buy = render_styled_html(buy_df_sorted, highlight_long_criteria)
            st.markdown(html_buy, unsafe_allow_html=True)
            # --- END NEW ---
            
            st.markdown("")
            st.markdown(create_export_link(buy_df, f"{analysis_title}_{analysis_date}_all_long.csv"), unsafe_allow_html=True)
        else:
            st.info("No long/buy signals generated for this analysis period.")

    with tab_sell:
        st.markdown(f"### ⬇️ All Short / Sell Signals ({total_sell_signals})")
        if not sell_df.empty:
            # --- NEW: Use styler to render (with SHORT highlight) ---
            sell_df_sorted = sell_df.sort_values("Signal")
            html_sell = render_styled_html(sell_df_sorted, highlight_short_criteria)
            st.markdown(html_sell, unsafe_allow_html=True)
            # --- END NEW ---
            
            st.markdown("")
            st.markdown(create_export_link(sell_df, f"{analysis_title}_{analysis_date}_all_short.csv"), unsafe_allow_html=True)
        else:
            st.info("No short/sell signals generated for this analysis period.")

    with tab_all:
        st.markdown("### 📋 Complete Signal Report")
        # --- NEW: Use styler to render (no highlight) ---
        html_all = render_styled_html(results_df, highlight_func=None)
        st.markdown(html_all, unsafe_allow_html=True)
        # --- END NEW ---
        
        st.markdown("")
        st.markdown(create_export_link(results_df, f"{analysis_title}_{analysis_date}_complete.csv"), unsafe_allow_html=True)

    with tab_errors:
        st.markdown(f"### ⚠️ Analysis Errors ({total_error_stocks})")
        if not errors_df.empty:
            # --- NEW: Use styler to render (no highlight) ---
            html_errors = render_styled_html(errors_df, highlight_func=None)
            st.markdown(html_errors, unsafe_allow_html=True)
            # --- END NEW ---
            st.warning(f"⚠️ {total_error_stocks} stocks encountered errors during analysis. This may affect overall signal accuracy.")
        else:
            st.success("✅ No errors encountered during analysis!")

# --- SIDEBAR CONFIGURATION (UPDATED) ---
with st.sidebar:
    st.markdown("# ⚙️ Configuration")
    
    # --- NEW: Model Selection ---
    st.markdown("### 🔬 Model Selection")
    selected_model = st.selectbox(
        "Select Analysis Model",
        ["ILFO (Pine v6 Rebuild)", "ROC & BasisSlope"], # <-- Renamed
        help="Choose the quantitative model for signal generation. Highlighting is only available for ILFO."
    )
    # --- END NEW ---
    
    st.markdown("### 🎯 Universe Selection")
    analysis_universe = st.selectbox(
        "Analysis Universe",
        ANALYSIS_UNIVERSE_OPTIONS,
        help="Choose between F&O stocks or specific index constituents"
    )
    
    selected_index = None
    if analysis_universe == "Index Constituents":
        selected_index = st.selectbox(
            "Select Index",
            INDEX_LIST,
            index=INDEX_LIST.index("NIFTY 500"),
            help="Select the index for constituent analysis"
        )
    
    st.markdown("### 📅 Time Period")
    analysis_date = st.date_input(
        "Analysis Date",
        datetime.today().date(),
        help="Select the date for signal analysis"
    )
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    submit_button = st.button(
        label="Run Analysis",
        use_container_width=True,
        type="primary"
    )
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # --- NEW: Dynamic Info Sections ---
    if selected_model == "ILFO (Pine v6 Rebuild)":
        st.markdown("### ℹ️ Platform Info")
        st.markdown(f"""
        <div class='info-box'>
            <p style='font-size: 0.85rem; margin: 0; color: var(--text-muted); line-height: 1.6;'>
                <strong>Version:</strong> {VERSION}<br>
                <strong>Model:</strong> ILFO (Pine v6 Rebuild)<br> 
                <strong>Data:</strong> yfinance
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📖 Quick Guide")
        with st.expander("💡 How to Use"):
            st.markdown("""
            **Step-by-Step:**
            1. Select your analysis universe (F&O or Index)
            2. Choose the analysis date
            3. Click "Run Analysis"
            4. Review signals in dashboard tabs
            5. Export data using CSV download links
            
            **Tips:**
            - Dashboard tab shows overall market health
            - Sector tab reveals industry trends
            - Long/Short tabs filter specific signals
            - **Rows in 'All Long/Buy' (yellow) or 'All Short/Sell' (red) meet your criteria.**
            """)
        
        with st.expander("🎯 Signal Guide (ILFO)"):
            st.markdown("""
            **Long Signals:**
            - 🔥 **Extreme Long**: Strongest buy signal (oversold + divergence)
            - 🟢 **Long**: Strong oversold reversal
            - 🔎 **Div. Long**: Bullish divergence detected
            
            **Short Signals:**
            - 📉 **Extreme Short**: Strongest sell signal (overbought + divergence)
            - 🔴 **Short**: Strong overbought reversal
            - 🔎 **Div. Short**: Bearish divergence detected
            
            **Others:**
            - ➖ **Neutral**: No clear signal
            - ⚠️ **Error**: Data processing issue
            """)
        
        with st.expander("🔬 ILFO Methodology"):
            st.markdown("""
            **Intelligent Liquidity Flow Oscillator** combines:
            - Market microstructure analysis
            - Volume flow patterns
            - Momentum & conviction metrics
            - Statistical overbought/oversold zones
            - Advanced divergence detection
            
            The model identifies high-probability reversal points with multi-factor confirmation.
            """)
            
    elif selected_model == "ROC & BasisSlope":
        st.markdown("### ℹ️ Platform Info")
        st.markdown(f"""
        <div class='info-box'>
            <p style='font-size: 0.85rem; margin: 0; color: var(--text-muted); line-height: 1.6;'>
                <strong>Version:</strong> {VERSION}<br>
                <strong>Model:</strong> ROC & BasisSlope<br>
                <strong>Data:</strong> yfinance
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📖 Quick Guide")
        with st.expander("💡 How to Use"):
            st.markdown("""
            **Step-by-Step:**
            1. Select your analysis universe (F&O or Index)
            2. Choose the analysis date
            3. Click "Run Analysis"
            4. Review signals in dashboard tabs
            5. Export data using CSV download links
            """)
        
        with st.expander("🎯 Signal Guide (ROC)"):
            st.markdown("""
            - **Buy**: Strong bullish indicators
            - **Sell**: Strong bearish indicators  
            - **Neutral**: No clear directional bias
            - ⚠️ **Error**: Data processing issue
            """)
    # --- END Dynamic Info ---

# --- MAIN APP BODY (UPDATED) ---
if submit_button:
    if analysis_date > datetime.today().date():
        st.error("⚠️ Analysis date cannot be in the future.")
    else:
        # Pass the selected_model to the analysis function
        run_analysis(analysis_universe, selected_index, analysis_date, selected_model)
else:
    st.markdown("""
    <div class='info-box welcome'>
        <h4>👋 Welcome to Sanket | Quantitative Signal Analytics</h4>
        <p>
            Experience professional-grade quantitative signal analysis.
            Use the sidebar to select your desired model (ILFO or ROC & BasisSlope), 
            configure your parameters, and click <strong>"Run Analysis"</strong> to begin.
        </p>
        <ul>
            <li><strong>Multi-Model:</strong> Choose between ILFO (Pine v6 Rebuild) or ROC & BasisSlope.</li>
            <li><strong>Sector Intelligence:</strong> Industry-wide trend analysis.</li>
            <li><strong>Real-Time Data:</strong> Live market data integration.</li>
            <li><strong>Export Ready:</strong> Professional CSV reports.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card info'>
            <h4>🎯 PRECISION</h4>
            <h2>Multi-Factor</h2>
            <div class='sub-metric'>Confirmation System</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card success'>
            <h4>⚡ SPEED</h4>
            <h2>Real-Time</h2>
            <div class='sub-metric'>Market Analysis</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card warning'>
            <h4>📊 INSIGHTS</h4>
            <h2>Sector-Level</h2>
            <div class='sub-metric'>Deep Analytics</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption(f"© 2025 Sanket | Quantitative Signal Analytics | {VERSION} | Last Updated: {time.strftime('%Y-%m-%d %H:%M:%S IST')}")