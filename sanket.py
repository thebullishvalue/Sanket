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

logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title="Sanket | Quantitative Signal Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
VERSION = "v3.3.0" # UPDATED VERSION
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

# --- NEW: Backtested Optimal Ranges Configuration ---
OPTIMAL_RANGES = {
    "Long": {
        'ilfo_value': {'min': 3.062, 'max': 3.692, 'weight': 0.25, 'importance': 'critical'},
        'vol_surge': {'min': 72.350, 'max': 90.373, 'weight': 0.20, 'importance': 'high'},
        'momentum_rsi': {'min': 39.047, 'max': 42.789, 'weight': 0.20, 'importance': 'high'},
        'osc_momentum': {'min': 1.363, 'max': 3.701, 'weight': 0.15, 'importance': 'medium'},
        'osc_accel': {'min': 0.833, 'max': 3.788, 'weight': 0.10, 'importance': 'medium'},
        'volume_score': {'min': -0.589, 'max': -0.452, 'weight': 0.10, 'importance': 'low'}
    },
    "Short": {
        'ilfo_value': {'min': -3.936, 'max': -3.220, 'weight': 0.25, 'importance': 'critical'},
        'vol_surge': {'min': 77.883, 'max': 96.083, 'weight': 0.20, 'importance': 'high'},
        'momentum_rsi': {'min': 56.678, 'max': 60.686, 'weight': 0.20, 'importance': 'high'},
        'osc_momentum': {'min': -6.542, 'max': -4.154, 'weight': 0.15, 'importance': 'medium'},
        'osc_accel': {'min': -7.139, 'max': -4.184, 'weight': 0.10, 'importance': 'medium'},
        'volume_score': {'min': 0.526, 'max': 0.655, 'weight': 0.10, 'importance': 'low'}
    },
    "Buy": {
        'liq_osc': {'min': 3.10, 'max': 3.60, 'weight': 0.35, 'importance': 'critical'},
        'momentum_rsi': {'min': 0.000, 'max': 38.50, 'weight': 0.35, 'importance': 'critical'},
        'volume_score': {'min': -0.46, 'max': 0.45, 'weight': 0.30, 'importance': 'high'}
    },
    "Sell": {
        'liq_osc': {'min': -10.35, 'max': -3.75, 'weight': 0.35, 'importance': 'critical'},
        'momentum_rsi': {'min': 53.30, 'max': 57.08, 'weight': 0.35, 'importance': 'critical'},
        'volume_score': {'min': 0.77, 'max': 1.000, 'weight': 0.30, 'importance': 'high'}
    }
}

# --- NEW: Advanced Confidence Scoring Function ---
def calculate_weighted_confidence_score(values_dict, signal_type):
    """
    Calculate sophisticated confidence score based on:
    1. Proximity to optimal range center (closer = better)
    2. Weighted contribution of each parameter
    3. Penalties for values outside optimal range
    4. Bonus for multiple strong signals
    
    Returns: confidence_score (0-100), detailed_breakdown (dict)
    """
    if signal_type not in OPTIMAL_RANGES:
        return 0.0, {}
    
    ranges = OPTIMAL_RANGES[signal_type]
    total_score = 0.0
    max_possible_score = 0.0
    breakdown = {}
    critical_hits = 0
    high_hits = 0
    
    for param, config in ranges.items():
        if param not in values_dict or pd.isna(values_dict[param]):
            breakdown[param] = {
                'value': None,
                'status': 'missing',
                'contribution': 0.0,
                'max_contribution': config['weight'] * 100
            }
            max_possible_score += config['weight'] * 100
            continue
        
        value = values_dict[param]
        min_val = config['min']
        max_val = config['max']
        weight = config['weight']
        importance = config['importance']
        
        range_center = (min_val + max_val) / 2
        range_width = max_val - min_val
        
        # Calculate contribution based on position
        if min_val <= value <= max_val:
            # Inside optimal range
            distance_from_center = abs(value - range_center)
            normalized_distance = distance_from_center / (range_width / 2)
            
            # Proximity score: 1.0 at center, decays to 0.5 at edges
            proximity_factor = 1.0 - (normalized_distance * 0.5)
            
            # Base contribution
            contribution = weight * proximity_factor * 100
            
            # Bonus for being very close to center (within 25% of range)
            if normalized_distance < 0.25:
                contribution *= 1.15  # 15% bonus
                
            status = 'optimal'
            
            # Track importance hits
            if importance == 'critical':
                critical_hits += 1
            elif importance == 'high':
                high_hits += 1
                
        else:
            # Outside optimal range - calculate penalty
            if value < min_val:
                overshoot = (min_val - value) / range_width
            else:
                overshoot = (value - max_val) / range_width
            
            # Exponential decay: closer misses get more credit
            if overshoot < 0.5:
                # Within 50% of range width - partial credit
                decay_factor = np.exp(-2 * overshoot)
                contribution = weight * decay_factor * 100 * 0.4  # Max 40% credit
                status = 'near_miss'
            elif overshoot < 1.0:
                # Within 100% of range width - minimal credit
                contribution = weight * 100 * 0.15  # Max 15% credit
                status = 'far_miss'
            else:
                # Very far - almost no credit
                contribution = weight * 100 * 0.05  # Max 5% credit
                status = 'very_far'
        
        breakdown[param] = {
            'value': value,
            'optimal_range': (min_val, max_val),
            'status': status,
            'contribution': contribution,
            'max_contribution': weight * 100,
            'importance': importance
        }
        
        total_score += contribution
        max_possible_score += weight * 100
    
    # Apply synergy bonus if multiple critical/high criteria are met
    synergy_bonus = 0
    if critical_hits >= 2:
        synergy_bonus = 5  # 5 point bonus for 2+ critical hits
    if critical_hits >= 1 and high_hits >= 2:
        synergy_bonus = max(synergy_bonus, 3)  # 3 point bonus for mixed strong signals
    
    total_score = min(total_score + synergy_bonus, 100)
    
    # Calculate final confidence percentage
    confidence_score = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0
    
    breakdown['summary'] = {
        'raw_score': total_score,
        'max_possible': max_possible_score,
        'synergy_bonus': synergy_bonus,
        'critical_hits': critical_hits,
        'high_hits': high_hits,
        'confidence_percentage': confidence_score
    }
    
    return confidence_score, breakdown

def get_confidence_grade(score):
    """Convert confidence score to letter grade"""
    if score >= 90:
        return 'A+', 'exceptional'
    elif score >= 80:
        return 'A', 'excellent'
    elif score >= 70:
        return 'B+', 'good'
    elif score >= 60:
        return 'B', 'acceptable'
    elif score >= 50:
        return 'C+', 'marginal'
    elif score >= 40:
        return 'C', 'weak'
    else:
        return 'D', 'poor'

# --- Premium Professional CSS ---
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
        --success-dark: #059669;
        --danger-red: #ef4444;
        --danger-dark: #dc2626;
        --warning-amber: #f59e0b;
        --info-cyan: #06b6d4;
        
        --extreme-long: #10b981;
        --long: #34d399;
        --div-long: #6ee7b7;
        --extreme-short: #ef4444;
        --short: #f87171;
        --div-short: #fca5a5;
        --neutral: #888888;
        
        --grade-a-plus: #10b981;
        --grade-a: #34d399;
        --grade-b-plus: #6ee7b7;
        --grade-b: #fbbf24;
        --grade-c: #f59e0b;
        --grade-d: #ef4444;
    }
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main, [data-testid="stSidebar"] {
        background-color: var(--background-color);
        color: var(--text-primary);
    }
    
    .stApp > header {
        background-color: transparent;
    }
    
    .block-container {
        padding-top: 1rem;
        max-width: 1400px;
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
        margin-top: 2.5rem;
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
        font-size: 2.50rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.50px;
        position: relative;
    }
    
    .premium-header .tagline {
        color: var(--text-muted);
        font-size: 1rem;
        margin-top: 0.25rem;
        font-weight: 400;
        position: relative;
    }
    
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
        font-size: 2rem;
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
    
    .metric-card.success h2 { color: var(--success-green); }
    .metric-card.danger h2 { color: var(--danger-red); }
    .metric-card.warning h2 { color: var(--warning-amber); }
    .metric-card.info h2 { color: var(--info-cyan); }
    .metric-card.neutral h2 { color: var(--neutral); }
    .metric-card.primary h2 { color: var(--primary-color); } /* New class for 6th color */
    .metric-card.white h2 { color: var(--text-primary); } /* New class for white text */
    
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
    
    .info-box {
        background: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--primary-color);
        padding: 1.25rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08);
    }
    
    .info-box h4 {
        color: var(--primary-color);
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
        font-weight: 700;
    }

    /* --- START: Button CSS from sanket.py --- */
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
    /* --- END: Button CSS from sanket.py --- */
    
    .stMarkdown table {
        width: 100%;
        border-collapse: collapse;
        background: var(--bg-card);
        border-radius: 16px;
        overflow: hidden;
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
    
    .signal-extreme-long { color: var(--extreme-long) !important; font-weight: 700; }
    .signal-long { color: var(--long) !important; font-weight: 600; }
    .signal-div-long { color: var(--div-long) !important; font-weight: 600; }
    .signal-extreme-short { color: var(--extreme-short) !important; font-weight: 700; }
    .signal-short { color: var(--short) !important; font-weight: 600; }
    .signal-div-short { color: var(--div-short) !important; font-weight: 600; }
    .signal-neutral { color: var(--neutral) !important; }
    .signal-error { color: var(--warning-amber) !important; }
    
    .pct-positive { color: var(--success-green) !important; font-weight: 600; }
    .pct-negative { color: var(--danger-red) !important; font-weight: 600; }
    .pct-neutral { color: var(--neutral) !important; }
    
    .grade-a-plus { color: var(--grade-a-plus) !important; font-weight: 700; }
    .grade-a { color: var(--grade-a) !important; font-weight: 700; }
    .grade-b-plus { color: var(--grade-b-plus) !important; font-weight: 600; }
    .grade-b { color: var(--grade-b) !important; font-weight: 600; }
    .grade-c { color: var(--grade-c) !important; font-weight: 600; }
    .grade-d { color: var(--grade-d) !important; font-weight: 600; }
    
    .confidence-high { background: linear-gradient(135deg, var(--success-green), var(--success-dark)); color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-weight: 700; }
    .confidence-medium { background: var(--warning-amber); color: var(--background-color); padding: 0.25rem 0.75rem; border-radius: 12px; font-weight: 700; }
    .confidence-low { background: var(--danger-red); color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-weight: 700; }
    
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, var(--border-color) 50%, transparent 100%);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="premium-header">
    <h1>Sanket | Quantitative Signal Analytics</h1>
</div>
""", unsafe_allow_html=True)

# --- Stock List Functions (Keep existing) ---
@st.cache_data(ttl=3600)
def get_fno_stock_list():
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

@st.cache_resource(show_spinner=False)
def load_sector_map():
    if os.path.exists(SECTOR_MAP_FILE):
        logging.info(f"Loading cached sector map from {SECTOR_MAP_FILE}")
        with open(SECTOR_MAP_FILE, 'rb') as f:
            return pickle.load(f)
    logging.info("No cached sector map found, starting with an empty map.")
    return {}

def save_sector_map(sector_map):
    logging.info(f"Saving updated sector map ({len(sector_map)} entries) to {SECTOR_MAP_FILE}")
    with open(SECTOR_MAP_FILE, 'wb') as f:
        pickle.dump(sector_map, f)

def fetch_sectors_for_list(stock_list):
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

# --- ILFO Signal Calculation (Keep existing, add confidence scoring) ---
def compute_ilfo_signal(ticker, df, end_date):
    """ILFO signal with enhanced confidence scoring"""
    
    adaptiveLength = 21
    microLength = 9
    impactWindow = 5
    devMultiplier = 2.0
    signalSmooth = 5
    divLookback = 10
    volThreshold = 1.2
    
    EPSILON = 1e-10

    def get_error_dict(signal, details, e=""):
        logging.error(f"Error for {ticker}: {details} - {e}")
        return {
            "ticker": ticker, "signal": signal, "details": details, "pct_change": np.nan,
            "confidence_score": 0.0, "confidence_grade": "D", "confidence_class": "poor",
            "ilfo_value": np.nan, "vol_surge": np.nan, "momentum_rsi": np.nan,
            "osc_momentum": np.nan, "osc_accel": np.nan, "volume_score": np.nan,
            "normalized_liq": np.nan
        }
    
    try:
        if df.empty or len(df) < adaptiveLength * 2:
            return get_error_dict("Insufficient Data", "N/A")
        
        df = df.copy()
        df = df.ffill().bfill()
        
        if df['Close'].isnull().all() or df['Volume'].isnull().all():
            return get_error_dict("Insufficient Data", "Missing main series")
        
        df['Volume'] = df['Volume'].fillna(0).replace(0, 1).clip(lower=1)
        
        if (df['Close'] <= 0).any():
            return get_error_dict("Invalid Data", "Non-positive prices detected")
        
        # [Keep all existing ILFO calculation logic - lines 29-209 from original]
        # Market Microstructure
        bodySize = (df['Close'] - df['Open']).abs()
        spreadProxy = (df['High'] + df['Low']) / 2 - df['Open']
        
        volMa = ta.sma(df['Volume'], adaptiveLength)
        volMa = volMa.fillna(df['Volume'].mean()).replace(0, EPSILON).clip(lower=EPSILON)
        
        vwapSpread = ta.sma(spreadProxy * df['Volume'] / volMa, adaptiveLength)
        vwapSpread = vwapSpread.fillna(0)
        
        priceImpact = ta.sma((df['Close'] - df['Close'].shift(impactWindow)) * df['Volume'] / volMa, adaptiveLength)
        priceImpact = priceImpact.fillna(0)
        
        liquidityScore = vwapSpread - priceImpact
        liquidityScore = liquidityScore.replace([np.inf, -np.inf], 0).fillna(0)
        
        liqMean = ta.sma(liquidityScore, adaptiveLength).fillna(0)
        liqStdev = ta.stdev(liquidityScore, adaptiveLength)
        liqStdev = liqStdev.fillna(1).replace(0, 1).clip(lower=EPSILON)
        
        normalizedLiq = (liquidityScore - liqMean) / liqStdev
        normalizedLiq = normalizedLiq.replace([np.inf, -np.inf], 0).fillna(0).clip(-10, 10)

        # Volume Flow Analysis
        volStdev = ta.stdev(df['Volume'], microLength)
        volStdev = volStdev.fillna(df['Volume'].std()).replace(0, EPSILON).clip(lower=EPSILON)
        
        volZscore = (df['Volume'] - volMa) / volStdev
        volZscore = volZscore.replace([np.inf, -np.inf], 0).fillna(0).clip(-5, 5)
        
        volSurge = (50 + (volZscore * 20)).clip(0, 100).fillna(50)
        
        volDirection = np.where(df['Close'] > df['Open'], volSurge, -volSurge)
        
        typicalPrice = (df['High'] + df['Low'] + df['Close']) / 3
        moneyFlow = typicalPrice * df['Volume']
        moneyFlow = moneyFlow.replace([np.inf, -np.inf], 0).fillna(0)
        
        posFlow = ta.sma(moneyFlow * (df['Close'] > df['Close'].shift(1)), microLength)
        negFlow = ta.sma(moneyFlow * (df['Close'] < df['Close'].shift(1)), microLength)
        posFlow = posFlow.fillna(0)
        negFlow = negFlow.fillna(0)
        
        accumFlow = (posFlow - negFlow) / (posFlow + negFlow + EPSILON)
        accumFlow = accumFlow.replace([np.inf, -np.inf], 0).fillna(0).clip(-1, 1)
        
        volumeScore = (volDirection / 100) * 0.5 + accumFlow * 0.5
        volumeScore = volumeScore.replace([np.inf, -np.inf], 0).fillna(0)

        # Momentum & Conviction
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
        
        directionConviction = np.where(df['Close'] > df['Open'], bodyConviction, -bodyConviction)
        
        price_change = df['Close'].diff()
        price_base = df['Close'].shift(1).replace(0, EPSILON).clip(lower=EPSILON)
        
        priceVelocity = (price_change / price_base) * 10000
        priceVelocity = priceVelocity.replace([np.inf, -np.inf], 0).fillna(0).clip(-1000, 1000)
        
        momentumRsi = ta.rsi(priceVelocity, microLength)
        momentumRsi = momentumRsi.fillna(50).clip(0, 100)

        # Statistical Bounds
        priceMean = ta.sma(df['Close'], adaptiveLength).fillna(df['Close'])
        priceStdev = ta.stdev(df['Close'], adaptiveLength)
        priceStdev = priceStdev.fillna(df['Close'].std()).replace(0, EPSILON).clip(lower=EPSILON)
        
        upperBound = priceMean + devMultiplier * priceStdev
        lowerBound = priceMean - devMultiplier * priceStdev

        inOverbought = df['Close'] > upperBound
        inOversold = df['Close'] < lowerBound

        # Composite Oscillator
        rawScore = (normalizedLiq * 0.30) + \
                   (volumeScore * 0.25) + \
                   (directionConviction / 100 * 0.25) + \
                   ((momentumRsi - 50) / 50 * 0.20)
        
        rawScore = rawScore.replace([np.inf, -np.inf], 0).fillna(0).clip(-5, 5)
        
        oscillator = -(rawScore * 8)
        oscillator = oscillator.replace([np.inf, -np.inf], 0).fillna(0).clip(-10, 10)
        
        signal = ta.sma(oscillator, signalSmooth)
        signal = signal.fillna(0).clip(-10, 10)
        
        oscMomentum = oscillator.diff(2).fillna(0).clip(-15, 15)
        oscAccel = oscMomentum.diff().fillna(0).clip(-15, 15)

        # Divergence Detection
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
        
        df['last_pivot_low_price'] = df.loc[pivot_lows_idx, 'Low'].reindex(df.index).ffill().fillna(df['Low'])
        df['last_pivot_low_osc'] = oscillator.loc[pivot_lows_idx].reindex(df.index).ffill().fillna(oscillator)
        
        df['last_pivot_high_price'] = df.loc[pivot_highs_idx, 'High'].reindex(df.index).ffill().fillna(df['High'])
        df['last_pivot_high_osc'] = oscillator.loc[pivot_highs_idx].reindex(df.index).ffill().fillna(oscillator)

        volConfirm = df['Volume'] > volMa * 0.8
        
        priceLL = df['Low'] < (df['last_pivot_low_price'] * 0.998)
        oscHL = oscillator > (df['last_pivot_low_osc'] * 1.05)
        bullishDiv = priceLL & oscHL & inOversold & volConfirm
        
        priceHH = df['High'] > (df['last_pivot_high_price'] * 1.002)
        oscLH = oscillator < (df['last_pivot_high_osc'] * 0.95)
        bearishDiv = priceHH & oscLH & inOversold & volConfirm

        # Signal Generation
        extremeLong = inOversold & (oscMomentum > 0) & (oscAccel > 0) & (volSurge > volThreshold * 50)
        extremeShort = inOverbought & (oscMomentum < 0) & (oscAccel < 0) & (volSurge > volThreshold * 50)

        # Extract values at target date
        analysis_datetime = datetime.combine(end_date, datetime.max.time())
        df.index = pd.to_datetime(df.index)
        target_date = df.index.asof(analysis_datetime)
        
        if pd.isna(target_date):
            return get_error_dict("No Data", f"No data at {end_date.date()}")
        
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
            
            for val in test_values:
                if pd.isna(val) or np.isinf(val):
                    return get_error_dict("No Data", "Invalid calculation result")
                    
        except (KeyError, IndexError):
            return get_error_dict("No Data", "Target date not in index")

        isExtremeLong = bool(extremeLong.loc[target_date])
        isBullishDiv = bool(bullishDiv.loc[target_date])
        isExtremeShort = bool(extremeShort.loc[target_date])
        isBearishDiv = bool(bearishDiv.loc[target_date])

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

        nl_value = float(normalizedLiq.loc[target_date])
        ilfo_value = float(oscillator.loc[target_date])
        vol_value = float(volSurge.loc[target_date])
        mom_rsi_val = float(momentumRsi.loc[target_date])
        osc_mom_val = float(oscMomentum.loc[target_date])
        osc_accel_val = float(oscAccel.loc[target_date])
        vol_score_val = float(volumeScore.loc[target_date])

        # --- NEW: Calculate Confidence Score ---
        values_for_scoring = {
            'ilfo_value': ilfo_value,
            'vol_surge': vol_value,
            'momentum_rsi': mom_rsi_val,
            'osc_momentum': osc_mom_val,
            'osc_accel': osc_accel_val,
            'volume_score': vol_score_val
        }
        
        # Only calculate confidence for actionable signals
        if "Long" in signal_text or "Short" in signal_text:
            confidence_score, breakdown = calculate_weighted_confidence_score(
                values_for_scoring, 
                signal_text if signal_text in ["Long", "Short"] else signal_text.split()[-1]
            )
            grade, grade_class = get_confidence_grade(confidence_score)
        else:
            confidence_score = 0.0
            grade = "N/A"
            grade_class = "neutral"
        # --- END NEW ---

        # Build details string with formatted values
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

        # Highlight based on optimal ranges
        if "Long" in signal_text:
            ranges = OPTIMAL_RANGES["Long"]
            if ranges['ilfo_value']['min'] <= ilfo_value <= ranges['ilfo_value']['max']:
                ilfo_string = f'<span class="pct-positive">{ilfo_value:.2f}</span>'
            if ranges['vol_surge']['min'] <= vol_value <= ranges['vol_surge']['max']:
                vol_string = f'<span class="pct-positive">{vol_value:.1f}</span>'
            if ranges['momentum_rsi']['min'] <= mom_rsi_val <= ranges['momentum_rsi']['max']:
                mom_rsi_string = f'<span class="pct-positive">{mom_rsi_val:.2f}</span>'
            if ranges['osc_momentum']['min'] <= osc_mom_val <= ranges['osc_momentum']['max']:
                osc_mom_string = f'<span class="pct-positive">{osc_mom_val:.2f}</span>'
            if ranges['osc_accel']['min'] <= osc_accel_val <= ranges['osc_accel']['max']:
                osc_accel_string = f'<span class="pct-positive">{osc_accel_val:.2f}</span>'
            if ranges['volume_score']['min'] <= vol_score_val <= ranges['volume_score']['max']:
                vol_score_string = f'<span class="pct-positive">{vol_score_val:.2f}</span>'
                
        elif "Short" in signal_text:
            ranges = OPTIMAL_RANGES["Short"]
            if ranges['ilfo_value']['min'] <= ilfo_value <= ranges['ilfo_value']['max']:
                ilfo_string = f'<span class="pct-negative">{ilfo_value:.2f}</span>'
            if ranges['vol_surge']['min'] <= vol_value <= ranges['vol_surge']['max']:
                vol_string = f'<span class="pct-negative">{vol_value:.1f}</span>'
            if ranges['momentum_rsi']['min'] <= mom_rsi_val <= ranges['momentum_rsi']['max']:
                mom_rsi_string = f'<span class="pct-negative">{mom_rsi_val:.2f}</span>'
            if ranges['osc_momentum']['min'] <= osc_mom_val <= ranges['osc_momentum']['max']:
                osc_mom_string = f'<span class="pct-negative">{osc_mom_val:.2f}</span>'
            if ranges['osc_accel']['min'] <= osc_accel_val <= ranges['osc_accel']['max']:
                osc_accel_string = f'<span class="pct-negative">{osc_accel_val:.2f}</span>'
            if ranges['volume_score']['min'] <= vol_score_val <= ranges['volume_score']['max']:
                vol_score_string = f'<span class="pct-negative">{vol_score_val:.2f}</span>'

        details_text = (
            f"ILFO: {ilfo_string} | NL: {nl_string} | VolSurge: {vol_string} | MomRSI: {mom_rsi_string}<br>"
            f"OscMom: {osc_mom_string} | OscAcc: {osc_accel_string} | VolScore: {vol_score_string}"
        )
        
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
            "confidence_score": confidence_score,
            "confidence_grade": grade,
            "confidence_class": grade_class,
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

# --- ROC Signal Calculation (Keep existing, add confidence scoring) ---
def compute_roc_slope_signal(ticker, df, end_date):
    """ROC signal with enhanced confidence scoring"""
    
    rocLength = 14
    length = 20
    delta = 0.95
    multiplier = 2.0
    impact_window = 5
    rsi_length = 14

    def get_error_dict(signal, details, e=""):
        logging.error(f"Error for {ticker}: {details} - {e}")
        return {
            "ticker": ticker, "signal": signal, "details": details, "pct_change": np.nan,
            "confidence_score": 0.0, "confidence_grade": "D", "confidence_class": "poor",
            "ilfo_value": np.nan, "vol_surge": np.nan, "momentum_rsi": np.nan,
            "osc_momentum": np.nan, "osc_accel": np.nan, "volume_score": np.nan,
            "normalized_liq": np.nan
        }
    
    try:
        if df.empty or len(df) < length * 2:
            return get_error_dict("Insufficient Data", "N/A")
            
        df['rsi'] = ta.rsi(df['Close'], length=rsi_length)
        df['roc'] = ta.roc(df['Close'], length=rocLength)
        
        rocMax = df['roc'].rolling(window=length).max()
        rocMin = df['roc'].rolling(window=length).min()
        scaledRoc = 16 * (df['roc'] - rocMin) / (rocMax - rocMin + 1e-9) - 8
        df['scaledRoc'] = scaledRoc.replace([np.inf, -np.inf], 0).fillna(0) if scaledRoc is not None else 0.0

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
            return get_error_dict("Error (Calc)", "SMA/STDEV failed")
            
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

        analysis_datetime = datetime.combine(end_date, datetime.max.time())
        df.index = pd.to_datetime(df.index)
        target_date = df.index.asof(analysis_datetime)
        
        if pd.isna(target_date):
             return get_error_dict("No Data", f"No data at {end_date.date()}")
        
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
            
        rsi_val = df.loc[target_date, 'rsi']
        sig_score_val = signal_score.loc[target_date]
        liq_osc_val = liq_osc.loc[target_date]
        nl_val = normalized_liq.loc[target_date]

        # --- NEW: Calculate Confidence Score ---
        values_for_scoring = {
            'liq_osc': liq_osc_val,
            'momentum_rsi': rsi_val,
            'volume_score': sig_score_val
        }
        
        if signal_text in ["Buy", "Sell"]:
            confidence_score, breakdown = calculate_weighted_confidence_score(
                values_for_scoring, 
                signal_text
            )
            grade, grade_class = get_confidence_grade(confidence_score)
        else:
            confidence_score = 0.0
            grade = "N/A"
            grade_class = "neutral"
        # --- END NEW ---
        
        rsi_string = f'<span class="signal-neutral">{rsi_val:.1f}</span>'
        sig_score_string = f'<span class="signal-neutral">{sig_score_val:.2f}</span>'
        liq_osc_string = f'<span class="signal-neutral">{liq_osc_val:.2f}</span>'
        
        if nl_val >= 0:
            nl_string = f'<span class="pct-positive">POS</span>'
        else:
            nl_string = f'<span class="pct-negative">NEG</span>'

        if signal_text == "Buy":
            ranges = OPTIMAL_RANGES["Buy"]
            if ranges['liq_osc']['min'] <= liq_osc_val <= ranges['liq_osc']['max']:
                liq_osc_string = f'<span class="pct-positive">{liq_osc_val:.2f}</span>'
            if ranges['momentum_rsi']['min'] <= rsi_val <= ranges['momentum_rsi']['max']:
                rsi_string = f'<span class="pct-positive">{rsi_val:.1f}</span>'
            if ranges['volume_score']['min'] <= sig_score_val <= ranges['volume_score']['max']:
                sig_score_string = f'<span class="pct-positive">{sig_score_val:.2f}</span>'
                
        elif signal_text == "Sell":
            ranges = OPTIMAL_RANGES["Sell"]
            if ranges['liq_osc']['min'] <= liq_osc_val <= ranges['liq_osc']['max']:
                liq_osc_string = f'<span class="pct-negative">{liq_osc_val:.2f}</span>'
            if ranges['momentum_rsi']['min'] <= rsi_val <= ranges['momentum_rsi']['max']:
                rsi_string = f'<span class="pct-negative">{rsi_val:.1f}</span>'
            if ranges['volume_score']['min'] <= sig_score_val <= ranges['volume_score']['max']:
                sig_score_string = f'<span class="pct-negative">{sig_score_val:.2f}</span>'

        details_text = (
            f"RSI: {rsi_string} | SigScore: {sig_score_string} | LiqOsc: {liq_osc_string} | NL: {nl_string}"
        )
        
        pct_change = df.loc[target_date, 'Close'] / df['Close'].shift(1).loc[target_date] - 1
        pct_change_val = (pct_change * 100) if pd.notna(pct_change) else np.nan
            
        return {
            "ticker": ticker, 
            "signal": signal_text, 
            "details": details_text, 
            "pct_change": pct_change_val,
            "confidence_score": confidence_score,
            "confidence_grade": grade,
            "confidence_class": grade_class,
            "ilfo_value": liq_osc_val,
            "vol_surge": np.nan, 
            "momentum_rsi": rsi_val,
            "osc_momentum": np.nan, 
            "osc_accel": np.nan, 
            "volume_score": sig_score_val,
            "normalized_liq": nl_val
        }

    except Exception as e:
        return get_error_dict("Error (Calc)", str(e), e)

# --- UI Functions ---
def format_dataframe_for_display(df):
    """Format dataframe with proper HTML for colored display"""
    if df.empty:
        return df
    
    display_df = df.copy()
    
    if 'Signal' in display_df.columns:
        def format_signal(val):
            if pd.isna(val):
                return '<span class="signal-neutral">N/A</span>'
            val_str = str(val)
            
            if val_str == "Extreme Long":
                return f'<span class="signal-extreme-long">{val_str}</span>'
            elif val_str == "Long":
                return f'<span class="signal-long">{val_str}</span>'
            elif val_str == "Divergence Long":
                return f'<span class="signal-div-long">{val_str}</span>'
            elif val_str == "Buy":
                return f'<span class="signal-long">{val_str}</span>'
            elif val_str == "Extreme Short":
                return f'<span class="signal-extreme-short">{val_str}</span>'
            elif val_str == "Short":
                return f'<span class="signal-short">{val_str}</span>'
            elif val_str == "Divergence Short":
                return f'<span class="signal-div-short">{val_str}</span>'
            elif val_str == "Sell":
                return f'<span class="signal-short">{val_str}</span>'
            elif "Error" in val_str or "Data" in val_str:
                return f'<span class="signal-error">{val_str}</span>'
            return f'<span class="signal-neutral">{val_str}</span>'
        
        display_df['Signal'] = display_df['Signal'].apply(format_signal)
    
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
    
    # --- NEW: Format Confidence Score ---
    if 'Confidence' in display_df.columns:
        def format_confidence(val):
            if pd.isna(val) or val == 0:
                return '<span class="pct-neutral">N/A</span>'
            if val >= 80:
                return f'<span class="confidence-high">{val:.1f}%</span>'
            elif val >= 60:
                return f'<span class="confidence-medium">{val:.1f}%</span>'
            else:
                return f'<span class="confidence-low">{val:.1f}%</span>'
        
        display_df['Confidence'] = display_df['Confidence'].apply(format_confidence)
    
    # --- NEW: Format Grade ---
    if 'Grade' in display_df.columns:
        def format_grade(val):
            if pd.isna(val) or val == "N/A":
                return '<span class="signal-neutral">N/A</span>'
            val_str = str(val).replace('+', '-plus')
            class_name = f"grade-{val_str.lower()}"
            return f'<span class="{class_name}">{val}</span>'
        
        display_df['Grade'] = display_df['Grade'].apply(format_grade)
    # --- END NEW ---
    
    display_df = display_df.reset_index()
    return display_df

def create_export_link(df, filename):
    """Create downloadable CSV link"""
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-link">📥 Download CSV Report</a>'
    return href

# --- Main Analysis Function ---
def run_analysis(analysis_universe, selected_index, analysis_date, selected_model):
    """Main analysis orchestrator with confidence scoring"""
    
    if analysis_universe == "F&O Stocks":
        analysis_title = "F&O Stocks"
        logging.info(f"🔍 Analyzing {analysis_title}...")
        stock_list, fetch_msg = get_fno_stock_list()
    else:
        analysis_title = selected_index
        logging.info(f"🔍 Analyzing {analysis_title}...")
        stock_list, fetch_msg = get_index_stock_list(selected_index)
    
    if not stock_list:
        st.error(f"Failed to fetch stock list: {fetch_msg}")
        st.stop()
        
    logging.info(fetch_msg)
    
    if selected_model == "ILFO (Pine v6 Rebuild)":
        compute_function = compute_ilfo_signal
        signal_types = [
            "Extreme Long", "Long", "Divergence Long",
            "Extreme Short", "Short", "Divergence Short",
            "Neutral", "Error"
        ]
        get_buy_sell_counts = lambda counts: (
            counts["Extreme Long"] + counts["Long"] + counts["Divergence Long"],
            counts["Extreme Short"] + counts["Short"] + counts["Divergence Short"]
        )
    elif selected_model == "ROC & BasisSlope":
        compute_function = compute_roc_slope_signal
        signal_types = ["Buy", "Sell", "Neutral", "Error"]
        get_buy_sell_counts = lambda counts: (
            counts.get("Buy", 0),
            counts.get("Sell", 0)
        )

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

    logging.info(f"⬇️ Downloading historical data for {len(stock_list)} stocks...")
    all_data_dict, batch_msg = fetch_all_data(stock_list, analysis_date)
    
    if all_data_dict is None:
        st.error(f"Failed to download data: {batch_msg}")
        st.stop()
    
    logging.info(batch_msg)
    
    total_stocks = len(stock_list)
    results = []
    signal_counts = {sig: 0 for sig in signal_types}
    sector_signals = {}
    
    valid_tickers = list(all_data_dict.keys())
    total_to_process = len(valid_tickers)
    
    for i, ticker in enumerate(valid_tickers):
        ticker_df = all_data_dict[ticker]
        result_dict = compute_function(ticker, ticker_df, analysis_date)
        
        signal = result_dict["signal"]
        sector = sector_map.get(ticker, "Other") 

        if sector not in sector_signals:
            sector_signals[sector] = {sig: 0 for sig in signal_types}
        
        result_dict["Sector"] = sector
        results.append(result_dict)
        
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
        
        if (i + 1) % 50 == 0:
            logging.info(f"Analyzing {ticker} ({i+1}/{total_to_process})...")
        
    download_errors = total_stocks - total_to_process
    signal_counts["Error"] += download_errors

    logging.info("✅ Analysis Complete!")
    
    results_df = pd.DataFrame(results)
    
    results_df = results_df.rename(columns={
        "ticker": "Ticker", 
        "signal": "Signal", 
        "pct_change": "% Change", 
        "details": "Details",
        "confidence_score": "Confidence",
        "confidence_grade": "Grade"
    })
    results_df = results_df.set_index("Ticker")
    
    # --- NEW: Sort by Confidence Score (descending) for actionable signals ---
    actionable_mask = results_df['Signal'].str.contains('Long|Short|Buy|Sell', na=False)
    actionable_df = results_df[actionable_mask].copy()
    if not actionable_df.empty:
        actionable_df = actionable_df.sort_values('Confidence', ascending=False)
    
    neutral_df = results_df[~actionable_mask].copy()
    results_df = pd.concat([actionable_df, neutral_df])
    # --- END NEW ---
    
    display_columns = ["Signal", "% Change", "Confidence", "Grade", "Details"]
    criteria_columns = [
        "ilfo_value", "vol_surge", "momentum_rsi", 
        "osc_momentum", "osc_accel", "volume_score", "normalized_liq",
        "confidence_class"
    ]
    all_columns = display_columns + criteria_columns
    
    for col in all_columns:
        if col not in results_df.columns:
            results_df[col] = np.nan
            
    results_df = results_df[all_columns]
    
    total_buy_signals, total_sell_signals = get_buy_sell_counts(signal_counts)
    total_neutral_signals = signal_counts["Neutral"]
    total_error_stocks = signal_counts["Error"]

    try:
        ratio = total_buy_signals / total_sell_signals if total_sell_signals > 0 else float('inf')
        ratio_text = f"{ratio:.2f}" if ratio != float('inf') else "∞"
    except:
        ratio = 0
        ratio_text = "0.00"
        
    if ratio > 1.2: ratio_class = "success"
    elif ratio > 0.8: ratio_class = "neutral"
    else: ratio_class = "danger"

    buy_df = results_df[results_df['Signal'].str.contains("Long|Buy", na=False)].copy()
    sell_df = results_df[results_df['Signal'].str.contains("Short|Sell", na=False)].copy()
    errors_df = results_df[results_df['Signal'].fillna('').astype(str).str.contains("Error|Data")].copy()

    # --- NEW: Calculate Confidence Statistics ---
    high_confidence_count = len(results_df[results_df['Confidence'] >= 80])
    medium_confidence_count = len(results_df[(results_df['Confidence'] >= 60) & (results_df['Confidence'] < 80)])
    low_confidence_count = len(results_df[(results_df['Confidence'] > 0) & (results_df['Confidence'] < 60)])
    
    if not buy_df.empty and buy_df['Confidence'].notna().any():
        avg_buy_confidence = buy_df['Confidence'].mean()
    else:
        avg_buy_confidence = 0
        
    if not sell_df.empty and sell_df['Confidence'].notna().any():
        avg_sell_confidence = sell_df['Confidence'].mean()
    else:
        avg_sell_confidence = 0
    # --- END NEW ---

    # --- UI DISPLAY ---
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
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
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"<div class='metric-card success'><h4>⬆️ Total Long/Buy</h4><h2>{total_buy_signals:,}</h2><div class='sub-metric'>Avg Confidence: {avg_buy_confidence:.1f}%</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card danger'><h4>⬇️ Total Short/Sell</h4><h2>{total_sell_signals:,}</h2><div class='sub-metric'>Avg Confidence: {avg_sell_confidence:.1f}%</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card neutral'><h4>➖ Neutral</h4><h2>{total_neutral_signals:,}</h2><div class='sub-metric'>No Clear Signal</div></div>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<div class='metric-card {ratio_class}'><h4>📈 Long/Short Ratio</h4><h2>{ratio_text}</h2><div class='sub-metric'>{'Bullish' if ratio > 1.2 else 'Bearish' if ratio < 0.8 else 'Balanced'}</div></div>", unsafe_allow_html=True)

        # --- NEW: Confidence Quality Metrics ---
        st.markdown("### Confidence Quality Distribution")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='metric-card success'><h4>🎯 High Confidence</h4><h2>{high_confidence_count:,}</h2><div class='sub-metric'>≥80% Quality Score</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card warning'><h4>⚖️ Medium Confidence</h4><h2>{medium_confidence_count:,}</h2><div class='sub-metric'>60-79% Quality Score</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card danger'><h4>⚠️ Low Confidence</h4><h2>{low_confidence_count:,}</h2><div class='sub-metric'><60% Quality Score</div></div>", unsafe_allow_html=True)
        # --- END NEW ---

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

        if selected_model == "ILFO (Pine v6 Rebuild)":
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

        fig_sector.update_layout(
            barmode='stack', title="Aggregate Signals by Sector", template="plotly_dark",
            paper_bgcolor=bg_color, plot_bgcolor=bg_color,
            height=500, font=dict(color=text_color),
            yaxis=dict(title="Signal Count", gridcolor=grid_color), xaxis=dict(title="Sector"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_sector, width='stretch')
        
        st.markdown("### 📋 Detailed Sector Breakdown")
        sector_display_cols = [col for col in display_cols if col in sector_df.columns]
        sector_display = sector_df[sector_display_cols].copy()
        st.dataframe(sector_display, width='stretch', height=400)

    def render_styled_html(df):
        """Applies formatting and renders HTML"""
        formatted_df = format_dataframe_for_display(df)
        styler = formatted_df.style
        
        cols_to_hide = [col for col in criteria_columns if col in formatted_df.columns]
        if cols_to_hide:
            styler = styler.hide(cols_to_hide, axis='columns')
            
        styler = styler.set_table_attributes('class="stMarkdown table"').hide(axis="index")
        return styler.to_html(escape=False)

    with tab_buy:
        st.markdown(f"### ⬆️ All Long / Buy Signals ({total_buy_signals})")
        st.markdown(f"*Sorted by Confidence Score (Highest First)*")
        if not buy_df.empty:
            html_buy = render_styled_html(buy_df)
            st.markdown(html_buy, unsafe_allow_html=True)
            st.markdown("")
            st.markdown(create_export_link(buy_df, f"{analysis_title}_{analysis_date}_all_long.csv"), unsafe_allow_html=True)
        else:
            st.info("No long/buy signals generated for this analysis period.")

    with tab_sell:
        st.markdown(f"### ⬇️ All Short / Sell Signals ({total_sell_signals})")
        st.markdown(f"*Sorted by Confidence Score (Highest First)*")
        if not sell_df.empty:
            html_sell = render_styled_html(sell_df)
            st.markdown(html_sell, unsafe_allow_html=True)
            st.markdown("")
            st.markdown(create_export_link(sell_df, f"{analysis_title}_{analysis_date}_all_short.csv"), unsafe_allow_html=True)
        else:
            st.info("No short/sell signals generated for this analysis period.")

    with tab_all:
        st.markdown("### 📋 Complete Signal Report")
        st.markdown(f"*Actionable signals sorted by Confidence Score*")
        html_all = render_styled_html(results_df)
        st.markdown(html_all, unsafe_allow_html=True)
        st.markdown("")
        st.markdown(create_export_link(results_df, f"{analysis_title}_{analysis_date}_complete.csv"), unsafe_allow_html=True)

    with tab_errors:
        st.markdown(f"### ⚠️ Analysis Errors ({total_error_stocks})")
        if not errors_df.empty:
            html_errors = render_styled_html(errors_df)
            st.markdown(html_errors, unsafe_allow_html=True)
            st.warning(f"⚠️ {total_error_stocks} stocks encountered errors during analysis.")
        else:
            st.success("✅ No errors encountered during analysis!")

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("# ⚙️ Configuration")
    
    st.markdown("### 🔬 Model Selection")
    selected_model = st.selectbox(
        "Select Analysis Model",
        ["ILFO (Pine v6 Rebuild)", "ROC & BasisSlope"],
        help="Choose the quantitative model for signal generation with confidence scoring."
    )
    
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
        width='stretch',
        type="primary"
    )
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("### ℹ️ Platform Info")
    st.markdown(f"""
    <div class='info-box'>
        <p style='font-size: 0.85rem; margin: 0; color: var(--text-muted); line-height: 1.6;'>
            <strong>Version:</strong> {VERSION}<br>
            <strong>Model:</strong> {selected_model}<br> 
            <strong>Data:</strong> yfinance<br>
            <strong>Feature:</strong> Confidence Scoring
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📖 Confidence System")
    with st.expander("🎯 How It Works"):
        st.markdown("""
        **Advanced Weighted Scoring:**
        - Measures proximity to optimal backtested ranges
        - Closer to range center = higher score
        - Critical parameters weighted more heavily
        - Synergy bonus for multiple strong signals
        
        **Grade System:**
        - **A+ (90-100%)**: Exceptional - All criteria optimal
        - **A (80-89%)**: Excellent - Strong alignment
        - **B+ (70-79%)**: Good - Above average quality
        - **B (60-69%)**: Acceptable - Moderate quality
        - **C+ (50-59%)**: Marginal - Below average
        - **C (40-49%)**: Weak - Poor alignment
        - **D (<40%)**: Poor - Very weak signal
        
        **Sorting:**
        Results are automatically sorted by confidence score (highest first) for easy decision-making.
        """)

# --- MAIN APP ---
if submit_button:
    if analysis_date > datetime.today().date():
        st.error("⚠️ Analysis date cannot be in the future.")
    else:
        run_analysis(analysis_universe, selected_index, analysis_date, selected_model)
else:
    st.markdown("""
    <div class='info-box welcome'>
        <h4>👋 Welcome to Sanket | Advanced Confidence Scoring</h4>
        <p>
            Experience next-generation quantitative signal analysis with sophisticated confidence scoring.
            Our weighted proximity algorithm evaluates each signal against 10 years of backtested optimal ranges.
        </p>
        <ul>
            <li><strong>Confidence Scoring:</strong> 0-100% quality score based on proximity to optimal ranges</li>
            <li><strong>Letter Grades:</strong> A+ to D rating system for instant quality assessment</li>
            <li><strong>Smart Sorting:</strong> Signals automatically ranked by confidence (highest first)</li>
            <li><strong>Multi-Factor:</strong> Weighted evaluation of all critical parameters</li>
            <li><strong>Synergy Detection:</strong> Bonus scoring for multiple strong confirmations</li>
        </ul>
        <p style="margin-top: 1rem; font-weight: 600; color: var(--primary-color);">
            Configure your parameters in the sidebar and click "Run Analysis" to begin.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # --- MODIFIED: Added second row of cards ---
    # Feature highlights - Row 1 (from v.py)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card info'>
            <h4>🎯 PRECISION</h4>
            <h2>Weighted</h2>
            <div class='sub-metric'>Proximity Scoring</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card success'>
            <h4>📊 QUALITY</h4>
            <h2>A+ to D</h2>
            <div class='sub-metric'>Grade System</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card white'>
            <h4>🔬 VALIDATED</h4>
            <h2>10-Year</h2>
            <div class='sub-metric'>Backtested Ranges</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Feature highlights - Row 2 (from sanket.py)
    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("""
        <div class='metric-card danger'>
            <h4>🎯 PRECISION</h4>
            <h2>Multi-Factor</h2>
            <div class='sub-metric'>Confirmation System</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class='metric-card neutral'>
            <h4>⚡ SPEED</h4>
            <h2>Real-Time</h2>
            <div class='sub-metric'>Market Analysis</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown("""
        <div class='metric-card primary'>
            <h4>📊 INSIGHTS</h4>
            <h2>Sector-Level</h2>
            <div class='sub-metric'>Deep Analytics</div>
        </div>
        """, unsafe_allow_html=True)
    # --- END MODIFICATION ---
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # --- MODIFIED: Removed content below this line ---
    # Confidence Scoring Explanation
    
    # All content below this title (Proximity, Weighting, Synergy, Example) has been removed
    # as per the request.
    
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption(f"© 2025 Sanket | @thebullishvalue | {VERSION} | Last Updated: {time.strftime('%Y-%m-%d %H:%M:%S IST')}")
