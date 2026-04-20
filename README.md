# SANKET — Market Signal Screener
## Wave Trend Composite Index (WRCI) Quantitative Signal Detection System

**SANKET** identifies momentum signals and trend strength across Indian stocks, global indexes, commodities, currency pairs, and cryptocurrencies. Uses the WRCI (Wave Trend Composite Index) engine to detect bullish/bearish signals, rank by magnitude, and identify overbought/oversold zones with daily and weekly timeframe analysis.

### Key Features

- **Wave Trend Signal Detection**: WRCI momentum oscillations identify bullish (long) and bearish (short) signals
- **Multi-Universe Scanning**: F&O stocks, Indian indexes (NIFTY 50-500, sector indices), US indexes, commodities, currency, crypto
- **Signal Strength Ranking**: Rank signals by magnitude with diminishing returns above magnitude 50 for confidence weighting
- **Zone Detection**: Automatically identify overbought and oversold zones based on configurable thresholds
- **Trend Direction Tracking**: Secondary Trend oscillator shows directional strength and momentum confirmation
- **Dual Timeframe Analysis**: Daily and weekly signal detection for multi-timeframe confluence
- **Point & Range Study Modes**: Single-date screener or historical date-range analysis (500+ days)
- **Age-Based Signal Timeline**: Group signals by detection date (Today, 1d, 2d, 3d, 5d) to track emergence
- **Time-Series Visualization**: Track signal evolution across dates with interactive charts

---

## Tech Stack

- **Language**: Python 3.8+
- **Framework**: Streamlit (interactive web dashboard)
- **Data**: yfinance (market data), NSE API (Indian stock constituents), nsepython (F&O list)
- **Analysis**: NumPy, Pandas (data processing)
- **Charts**: Plotly (interactive WRCI/Trend visualization)
- **HTTP**: requests (web scraping), urllib3 (SSL handling)
- **UI**: Custom CSS (Obsidian Quant Terminal theme), SVG icons, HTML iframes

---

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Internet connection (for data fetching)
- Optional: NSE market data access (automatic via API)

---

## Getting Started

### 1. Clone or Download the Repository

```bash
# If in a git repository:
git clone <repository-url>
cd Sanket-final

# Or download the folder directly
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- streamlit >= 1.28
- pandas >= 1.5
- numpy >= 1.23
- yfinance >= 0.2.28
- plotly >= 5.14
- requests >= 2.28
- nsepython >= 0.2.16

### 3. Run the Application

```bash
streamlit run sanket.py
```

The app will open at `http://localhost:8501` in your browser.

---

## Architecture Overview

### Directory Structure

```
Sanket-final/
├── sanket.py                    # Main application (all logic + UI)
├── logger.py                    # Console logging utilities
├── requirements.txt             # Python dependencies
├── ui/
│   ├── __init__.py
│   ├── theme.py                # CSS injection, theme system
│   └── components.py           # Reusable UI components
├── README.md                    # This file
├── CHANGELOG.md                 # Version history
└── .gitignore
```

### Request Lifecycle

1. **User Configuration** → Sidebar controls select universe, timeframe, date range, WRCI parameters
2. **Run Screener Button** → Triggers analysis run, sets session state flags
3. **Data Fetching** → Fetch market data for selected universe symbols (yfinance for global, NSE for India)
4. **WRCI Analysis** → Compute Wave Trend + Trend oscillators for each symbol
5. **Signal Detection** → Identify bullish/bearish signals, calculate magnitude, detect zones
6. **Ranking & Display** → Sort by strength, group by age, render interactive tables
7. **Response** → Display signal cards, conviction tables, time-series charts

### Core Modules

**Data Fetching (`get_*_symbols()` functions)**
- `get_index_stock_list()`: Fetch NIFTY 50/NEXT 50/etc. constituents from NSE CSV archives
- `get_fno_stock_list()`: Fetch F&O eligible stocks with 3 fallback sources (NSE API → advances/declines → NIFTY 500)
- `get_us_index_symbols()`: Map S&P 500, DOW JONES, NASDAQ 100 to yfinance tickers
- `get_commodity_symbols()`: Commodity futures (Gold, Oil, Wheat, etc.)
- `get_currency_symbols()`: FX pairs (EUR/USD, USD/INR, GBP/JPY, etc.)
- `get_crypto_symbols()`: Cryptocurrencies (Bitcoin, Ethereum, etc.)

**Market Data Fetch**
- `fetch_batch_data()`: Download 1+ years of OHLCV data for symbols in parallel
- Handles API errors gracefully, skips unavailable symbols

**WRCI Engine (`run_screener_analysis()`, `run_timeseries_analysis()`)**
- Compute Wave Trend oscillator: smoothed price momentum
- Compute Trend oscillator: directional strength confirmation
- Detect crossovers, calculate signal magnitude (absolute value)
- Apply diminishing returns above magnitude 50 for strength scoring
- Flag symbols in overbought (>80 by default) or oversold (<20) zones

**Signal Bucketing & Display**
- `_bucket_signals_by_age()`: Group signals by detection date
- `_build_signal_table_html()`: Generate HTML table with age sections
- `_build_conviction_table_html()`: Ranked table of top signals
- `render_signal_detail_card()`: Detailed popup with RSI/price/zone confirmation

**UI Rendering**
- `render_sidebar()`: Configuration panel (universe, timeframe, date, WRCI parameters)
- `render_landing_page()`: System overview and instructions
- `render_footer()`: Copyright and version info
- SVG icons injected via CSS for tab labels

### Signal Strength Calculation

**Formula:**
```python
base_score = abs(Signal)
if base_score > 50:
    base_score = 50 + (base_score - 50) * 0.5
strength = min(100, base_score)
```

**Rationale:** Linear scoring for magnitude 0-50, then diminishing returns above 50 to prevent extreme values from dominating rankings.

**Quality Labels:**
- Strong: >= 65
- Moderate: 50-64
- Weak: 35-49
- Very Weak: < 35

---

## Configuration & Settings

### Sidebar Parameters

| Parameter | Type | Range | Description |
| --- | --- | --- | --- |
| **Universe** | Dropdown | 5 options | Market universe (India Indexes, US Indexes, Commodities, Currency, Crypto) |
| **Index/Universe** | Dropdown | 17 indices | NIFTY 50/NEXT 50/100/200/500, NIFTY sectors, F&O Stocks, US indexes, etc. |
| **Timeframe** | Dropdown | Daily, Weekly | Candle period for WRCI calculation |
| **Mode** | Radio | Single Date, Range | Single-date screener or historical date-range analysis |
| **Target Date** | Date picker | Any date | Analysis date for Single Date mode (defaults to today) |
| **Date Range** | Date range picker | Any range | Start-end dates for Range Study mode |
| **REG Length** | Slider | 10-50 | Regularization period for Wave Trend smoothing |
| **WT N1** | Slider | 5-20 | Fast EMA period for Wave Trend |
| **WT N2** | Slider | 20-50 | Slow EMA period for Wave Trend |
| **OB Level** | Slider | 50-100 | Overbought threshold |
| **OS Level** | Slider | -100 to -50 | Oversold threshold |

### WRCI Parameters Explained

- **REG Length**: Smoothing period; higher = smoother, more lag
- **WT N1**: Fast EMA controls responsiveness to recent price moves
- **WT N2**: Slow EMA provides trend context and stability
- **Default values** (10, 9, 21) tuned for daily timeframe

---

## Output & Interpretation

### Signal Columns

| Column | Meaning | Range | Interpretation |
| --- | --- | --- | --- |
| **Symbol** | Stock/commodity/crypto ticker | String | Identifier for the asset |
| **Price** | Current price in INR (or USD) | Float | Latest market price |
| **Signal** | Wave Trend oscillator value | -100 to +100 | Positive = bullish, negative = bearish |
| **Trend** | Trend oscillator value | -100 to +100 | Confirms direction; >30 strong up, <-30 strong down |
| **Zone** | Overbought/Oversold label | OB / OS / — | Indicates extreme conditions |
| **Age** | Signal detection date | Today, 1d, 2d, 3d, 5d+ | How long ago signal appeared |
| **Strength** | Magnitude-based score | 0-100 | Confidence/power of the signal |

### Reading the Screener

**"Today · 8 signals · Avg: +42.3"** = 8 bullish signals detected today with average magnitude 42.3

**"Strengthening" / "Weakening" / "Stable"** = Compares today's average signal magnitude vs. older signals. Strengthening = fresh, strong signals vs. aging ones.

**Rank 01 - NIFTY 50 · ₹22,150 · Signal: +67.8 · Trend: +45.2 · Zone: OB**
- 1st strongest signal (magnitude 67.8)
- In overbought zone (Signal > 80)
- Strong uptrend confirmation (Trend +45.2)

---

## Usage Guide

### Single Date Mode (Point Screener)

1. **Select Universe** (e.g., "India Indexes")
2. **Select Index** (e.g., "NIFTY 50")
3. **Select Timeframe** (Daily or Weekly)
4. **Set Date** (defaults to today)
5. **Adjust WRCI Parameters** if needed (optional)
6. **Click RUN SCREENER**
7. **View Results**:
   - **Action Dashboard**: Age-grouped signals with timeline
   - **Signal Strength Tab**: Ranked by magnitude
   - **Top Bullish/Bearish**: Highest conviction signals each side
   - **Time-Series**: WRCI evolution chart (last 100 candles)

### Range Study Mode (Time-Series)

1. Follow steps 1-3 above
2. **Select "Range" mode**
3. **Set Start & End dates** (up to 500 days back)
4. **Click RUN SCREENER**
5. **View Results**: Historical signal emergence across all dates
   - Each date shows signals that appeared that day
   - Charts show WRCI oscillations for ~20 sample symbols
   - Identify turning points and signal persistence

### Interpreting Results

**Strong Signal = High Conviction Entry?** Not necessarily. Strength = magnitude. Confirmation comes from:
- Trend > 30 (strong up) or Trend < -30 (strong down)
- Zone label (OB/OS indicates exhaustion risk)
- Signal age (fresher signals may be more tradeable)

**Overbought Signals**: Signals in OB zone (red) = high magnitude but exhaustion risk. Best for shorts.

**Oversold Signals**: Signals in OS zone (green) = high magnitude but bounce risk. Best for longs.

---

## Time-Series Analysis Details

### What is Range Study?

Analyzes signals across 500+ days to show:
- **Signal emergence pattern**: When does each symbol get signaled?
- **Signal persistence**: Does it stay signaled for multiple days?
- **Trend confirmation**: Which signals have strong trend support?

### Interpreting Timeline

**Axis X**: Date (from start to end date)
**Axis Y**: Symbol
**Color intensity**: Signal magnitude (brighter = stronger)

**Use case**: Find symbols with persistent, strengthening signals = higher conviction longs/shorts

---

## Troubleshooting

### "No data available for symbols"
- **Cause**: yfinance couldn't fetch data (network issue or invalid ticker)
- **Solution**: Check internet connection, retry

### "Failed to fetch F&O list from all sources"
- **Cause**: NSE API unreachable, advances/declines unavailable
- **Solution**: Try again in a few minutes, check NSE website status

### "Streamlit connection lost"
- **Cause**: App crashed or server restarted
- **Solution**: Refresh browser, verify terminal shows no errors

### Signals look weak or absent
- **Cause**: WRCI parameters tuned for specific volatility; different market conditions require adjustment
- **Solution**: Increase REG length (10→15) for noise reduction, or adjust N1/N2 for sensitivity

### Price data stale
- **Cause**: yfinance caching or market closed
- **Solution**: Wait for market to open or manually refresh page

---

## Development

### Adding a New Universe

1. **Define the universe** in constants (line 120+):
   ```python
   MY_UNIVERSE_MAP = { "Symbol1": "Ticker1", "Symbol2": "Ticker2" }
   MY_UNIVERSE_LIST = list(MY_UNIVERSE_MAP.keys())
   ```

2. **Add getter function**:
   ```python
   def get_my_universe_symbols(index):
       return list(MY_UNIVERSE_MAP.values()), f"✓ Fetched {len(...)} symbols"
   ```

3. **Wire in render_sidebar()** (line ~750):
   ```python
   elif universe == "My Universe":
       stock_list, msg = get_my_universe_symbols(selected_index)
   ```

4. **Update UI text** (line ~120):
   ```python
   UNIVERSE_OPTIONS = ["India Indexes", "US Indexes", ..., "My Universe"]
   ```

### Customizing Signal Strength Formula

Edit `get_conviction_score()` (line 637):
```python
def get_conviction_score(row):
    base_score = abs(row.get('Signal', 0))
    if base_score > 50:
        base_score = 50 + (base_score - 50) * 0.5  # ← Adjust multiplier here
    return min(100, base_score)
```

Lower multiplier (0.3) = less diminishing returns, higher multiplier (0.8) = more aggressive decay

### Adjusting Zone Thresholds

Defaults in sidebar (lines 750-800):
```python
obLevel1_default = 80  # Overbought boundary
osLevel1_default = -80  # Oversold boundary
```

Higher values = fewer zone alerts, lower = more sensitive

---

## Performance Considerations

- **Batch data fetch**: ~2-3 sec for 50 symbols, ~10-15 sec for 500
- **WRCI calculation**: Negligible (vectorized NumPy)
- **HTML rendering**: ~2-3 sec for large tables (500+ rows)
- **Time-series mode**: 15-30 sec for 500-day analysis on 100+ symbols

---

## Known Limitations

1. **Intraday data**: Only daily/weekly candles; no intraday analysis
2. **Real-time**: Requires manual re-run; no live updates
3. **Historical**: Limited to ~2+ years back (yfinance constraint)
4. **F&O list**: NSE API sometimes unavailable; fallbacks to NIFTY 500 approximation

---

## License

Built by **Antigravity** as member of Pragyam Product Family.

---

## Support & Feedback

For issues, feature requests, or questions:
- Check this README first
- Review CHANGELOG for known issues
- Test with default parameters if issues persist

---

**Version**: v2.0  
**Last Updated**: April 2026  
**Status**: Production Ready
