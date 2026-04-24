# SANKET — Market Signal Screener

> **संकेत** — *Signal. Indicator. Omen.*
> Wave-Regime Composite Index (WRCI) quantitative signal scanner built for Indian and global markets.

SANKET applies the WRCI momentum engine across your chosen universe — from individual NIFTY sector constituents to benchmark index instruments, global commodities, FX pairs, and crypto — and returns ranked bullish/bearish signals with age tracking, zone detection, and trend confirmation. It runs as a Streamlit web application with a custom Obsidian Quant Terminal dark UI.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Architecture](#architecture)
- [WRCI Engine](#wrci-engine)
- [Universes & Coverage](#universes--coverage)
- [Signal Output Reference](#signal-output-reference)
- [Analysis Modes](#analysis-modes)
- [Troubleshooting](#troubleshooting)
- [Development Guide](#development-guide)
- [License](#license)

---

## Features

- **Dual Signal Engine** — Two independent signal generation logics running in parallel:
  - **Threshold Set** — Composite oscillator crosses extreme levels (−40 / +40) with signal-line confirmation
  - **Crossover Set** — Composite oscillator crosses its signal line while already in extreme territory
  - **Momentum Set** — Pure crossover (no level filter) used exclusively by Range Study
- **WRCI Signal Detection** — Wave-Regime Composite Index identifies bullish and bearish momentum events by combining a smoothed Wave Trend oscillator with a volume-weighted trend directional count
- **Multi-Universe Scanning** — India Indexes (F&O, 26 NIFTY indices, Benchmark Instruments), US Indexes, NSE ETF Universe, Commodities, Currency, Crypto
- **Benchmark Indexes Mode** — Track the index instruments themselves (^NSEI, ^NSEBANK, ^INDIAVIX, ^BSESN, BSE-100, etc.) as an asset universe rather than their constituents
- **Signal Strength Ranking** — Magnitude-based scoring with diminishing returns above 50 to prevent extreme outliers from dominating
- **UMA v6 Intelligence Layer** — Integration of Market Signal Fusion (MSF) and Macro-Market Regime (MMR) ensemble for advanced trend and volatility analysis.
- **V4 Mathematical Parity** — Primary Signal and Trend oscillators aligned exactly with the v4 institutional baseline.
- **Regime-Aware States** — Confirmed Bullish/Bearish states driven by HMM (Hidden Markov Model) regime detection and macro-divergence analysis.
- **Asset Index Mode** — Treat global bond yields, commodities, and currencies as unified asset classes rather than isolated tickers.
- **Age-Based Timing Groups** — Signals bucketed into Today / 1 Day Ago / 2 Days Ago / 3 Days Ago / Within 5 Days.
- **Obsidian Quant Terminal UI** — Dark-first design with amber accents, light mode toggle, IBM Plex Mono + Space Grotesk typography.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.8+ |
| Web Framework | Streamlit |
| Market Data | yfinance (global), NSE API + nsepython (India) |
| Data Processing | pandas, NumPy (vectorized operations) |
| Charts | Plotly (interactive WRCI/Trend visualisation) |
| HTTP | requests, urllib3 |
| UI | Custom CSS (Obsidian Quant Terminal), inline SVG icons, HTML iframes |
| Logging | Custom ANSI console logger (`logger.py`) |

---

## Prerequisites

- **Python 3.8 or higher**
- **pip**
- **Internet access** — required for market data (yfinance) and NSE constituent lists

No database, no Redis, no message queue. Entirely stateless beyond Streamlit session state.

---

## Getting Started

### 1. Obtain the project

```bash
# From git:
git clone <repository-url>
cd Sanket-Final

# Or copy the folder directly
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Full dependency list:

| Package | Purpose |
|---|---|
| `streamlit` | Web app framework |
| `pandas` | DataFrame processing |
| `pandas-datareader` | Supplementary data fetching |
| `yfinance` | OHLCV market data (global + NSE) |
| `numpy` | Vectorized WRCI computation |
| `plotly` | Interactive oscillator charts |
| `requests` | NSE API / CSV fetching |
| `nsepython` | NSE advances/declines (F&O fallback) |
| `logger` | Bundled ANSI console logger |

### 4. Launch

```bash
streamlit run sanket.py
```

Opens at **http://localhost:8501**.

### 5. First run

1. Sidebar → **Universe**: `India Indexes` (default)
2. Sidebar → **Index**: `Benchmark Indexes` (default)
3. Sidebar → **Timeframe**: `Daily`
4. Sidebar → **Mode**: `Single Date` → today's date
5. Click **◈ RUN SCREENER**

Results appear across two tabs: **Action Dashboard** (signals by timing) and **Signal Strength Analysis**.

---

## Architecture

### Directory Structure

```
Sanket-Final/
├── sanket.py            # Entire application — constants, data fetch, WRCI engine, UI
├── logger.py            # ANSI console output system (ConsoleOutput class)
├── requirements.txt     # pip dependencies
├── ui/
│   ├── theme.py         # CSS injection, chart theming, progress bar helper
│   ├── theme.css        # Full Obsidian Quant Terminal stylesheet
│   └── components.py    # Reusable Streamlit components (metric card, section header, etc.)
├── README.md
├── CHANGELOG.md
└── LICENSE
```

### Application Flow

```
Browser Request
      │
      ▼
 inject_css()                ← Obsidian Quant Terminal stylesheet injected once
 render_theme_toggle()       ← Light/dark mode button
      │
      ▼
 render_sidebar()
   ├── Universe selector
   ├── Index selector
   ├── Timeframe (Daily / Weekly)
   ├── Mode (Single Date / Date Range)
   ├── Date picker(s)
   └── ◈ RUN SCREENER button
      │
      ▼ (on run_clicked)
 session_state["run_screener_flag"] = True → st.rerun()
      │
      ▼
 run_screener_analysis()
   ├── get_*_symbols()      ← Universe-specific fetcher
   ├── fetch_batch_data()   ← yfinance parallel download
   ├── resample_to_weekly() ← if Weekly timeframe
   ├── run_full_analysis()  ← WRCI engine per ticker
   └── returns results_df
      │
      ▼
 session_state["results_df"] = results_df → st.rerun()
      │
      ▼
 Render tabs
   ├── Tab 1: Action Dashboard
   │     ├── bull_tab: _build_signal_table_html(side='long')
   │     ├── bear_tab: _build_signal_table_html(side='short')
   │     └── Signal interpretation legend
   ├── Tab 2: Signal Strength Analysis
   │     ├── Metric cards
   │     ├── Top Bullish (ranked)
   │     └── Top Bearish (ranked)
   └── (Range Study) Inline time-series charts
```

### Session State Keys

| Key | Type | Purpose |
|---|---|---|
| `results_df` | `DataFrame \| None` | Cached screener output; `None` = landing page |
| `run_screener_flag` | `bool` | Triggers analysis on next render cycle |
| `run_timeseries_flag` | `bool` | Reserved for range study trigger |
| `timeseries_done` | `bool` | Suppresses re-render after range study completes |
| `run_error` | `str \| None` | Fetch error message; survives rerun, shown above landing page |

---

## WRCI Engine

The core algorithm lives in `run_full_analysis()` and combines two sub-oscillators.

### Sub-oscillator 1 — Wave Trend (WT1)

```
hlc3  = (High + Low + Close) / 3
hma_p = HMA(hlc3, 15)           # Hull Moving Average of price
esa   = EMA(hlc3, wt_n1)        # Default wt_n1 = 10
d     = EMA(|hlc3 - esa|, wt_n1)
ci    = (hlc3 - esa) / (0.015 × d)
wt1   = EMA(ci, wt_n2)          # Default wt_n2 = 21
wt2   = SMA(wt1, 4)             # 4-period simple moving average of wt1
```

### Sub-oscillator 2 — Normalised Trend Count

```
hma_p = HMA(hlc3, 15)
trend = Σ sign(hma_p[i] - hma_p[i-k]) for k=1..reg_len   # Default reg_len = 20
norm_trend = (trend × (10 / reg_len)) × 10
```

### Composite

```
composite_line   = (wt1 + norm_trend) / 2
composite_signal = rolling_mean(composite_line, 4)
```

### Signal Conditions — Three Independent Sets

The system computes three signal sets for different use-cases:

#### Set A — Threshold (Screener Primary)
Long when composite crosses **below −40** (oversold entry) with `composite_signal` > −40. Short when composite crosses **above +40** (overbought entry) with `composite_signal` < +40.

#### Set B — Crossover (Screener Secondary)
Long when `composite_line` crosses **below** `composite_signal` while already in oversold (`composite_line < −40`). Short when `composite_line` crosses **above** `composite_signal` while already in overbought (`composite_line > +40`).

#### Set C — Momentum (Range Study Only)
Pure crossover logic with no level filter: Long when `composite_line` crosses above `composite_signal`; Short when it crosses below.

### Zone Classification

| Zone | Criteria |
|---|---|
| OB Extreme | `composite_line > 80` |
| OB | `composite_line > 40` |
| Neutral | everything else |
| OS | `composite_line < -40` |
| OS Extreme | `composite_line < -80` |

### Signal Strength Scoring

```python
score = abs(Signal)
if score > 50:
    score = 50 + (score - 50) × 0.5   # diminishing returns above 50
score = min(100, score)
```

Quality bands: **Strong** ≥ 65 · **Moderate** 50–64 · **Weak** 35–49 · **Very Weak** < 35

### UMA State Detection

Unified Market Analysis overlays four highlight states on top of WRCI signals:

- **Bullish Divergence** — Oscillator rising while price falling, with WRCI < −5
- **Bearish Divergence** — Oscillator falling while price rising, with WRCI > +5
- **Confirmed Bullish** — MSF & MMR both bullish agreement (>40) + deep oversold WRCI (<10)
- **Confirmed Bearish** — MSF & MMR both bearish agreement (<-40) + deep overbought WRCI (>-10)

---

## Universes & Coverage

### India Indexes

Selected from a 29-item dropdown. Constituent lists fetched via 3-source cascade:

1. **NSE JSON API** — `https://www.nseindia.com/api/equity-stockIndices?index={NAME}` (primary; session-warmed)
2. **NSE Archive CSV** — `https://archives.nseindia.com/content/indices/ind_{name}list.csv` (fallback)
3. **Wikipedia** — index-specific page table (last resort, available for 5 major indices)

| Group | Entries |
|---|---|
| Special | F&O Stocks |
| Broad Market | NIFTY 50, NEXT 50, 100, 200, 500 |
| Midcap | NIFTY MIDCAP 50/100/150, NIFTY MID SELECT |
| Smallcap | NIFTY SMLCAP 50/100/250 |
| Banking | NIFTY BANK, NIFTY PRIVATE BANK, NIFTY PSU BANK |
| Sectoral | NIFTY FIN SERVICE, IT, AUTO, FMCG, PHARMA, METAL, ENERGY, INFRA, REALTY, MEDIA |
| Instruments | **Benchmark Indexes** (default) |

### Benchmark Indexes

Tracks the 34 index instruments themselves as the analysis universe — not their constituents.

| Group | Tickers |
|---|---|
| Broad NSE | `^NSEI` `^NSMIDCP` `NIFTY_100.NS` `NIFTY_200.NS` `NIFTY_500.NS` `^NSEMDCP50` `NIFTY_MIDCAP_100.NS` `NIFTY_MIDCAP_150.NS` `NIFTY_MID_SELECT.NS` `NIFTYSMLCAP50.NS` `NIFTY_SMALLCAP_100.NS` `NIFTY_SMALLCAP_250.NS` |
| Volatility | `^INDIAVIX` |
| Broad BSE | `^BSESN` `BSE-100.BO` `BSE-200.BO` `BSE-500.BO` |
| Sectoral NSE | `^NSEBANK` `^CNXFIN` `^CNXIT` `^CNXAUTO` `^CNXFMCG` `^CNXPHARMA` `^CNXMETAL` `^CNXREALTY` `^CNXENERGY` `^CNXINFRA` `^CNXPSUBANK` `NIFTY_PRIVATE_BANK.NS` `^CNXMEDIA` |

### ETF Index

30 NSE-listed ETFs covering FMCG, IT, Pharma, Auto, PSU Bank, Infra, Gold, Silver, Defence, EV, Smallcap, etc.

### US Indexes

S&P 500 (`^GSPC`), DOW JONES (`^DJI`), NASDAQ 100 (`^NDX`)

### Commodities

16 commodity futures via yfinance: Gold, Silver, Crude Oil (WTI + Brent), Natural Gas, Copper, Platinum, Palladium, Wheat, Corn, Soybeans, Coffee, Sugar, Cotton, Lumber, Cocoa.

### Currency

25 FX pairs: EUR/USD, GBP/USD, USD/JPY, USD/INR, USD/CHF, AUD/USD, USD/CAD, NZD/USD, EUR/GBP, EUR/JPY, GBP/JPY, and others.

### Crypto

21 cryptocurrencies: Bitcoin, Ethereum, Solana, BNB, XRP, Cardano, Dogecoin, Tron, Chainlink, Polkadot, Polygon, Litecoin, Bitcoin Cash, Shiba Inu, Avalanche, Near, Uniswap, Stellar, Ethereum Classic, Monero, Cosmos.

---

## Signal Output Reference

### Action Dashboard — Dual Signal Sets

The screener displays **two independent signal logics** side-by-side:

| Column Set | Description |
|---|---|
| **Threshold** (Set A) | Signals where composite crosses ±40 with signal-line confirmation |
| **Crossover** (Set B) | Signals where composite crosses its signal line within extreme zones |

Each set has its own timing table with nested tabs.

### Timing Tables Columns

| Column | Description |
|---|---|
| **Symbol** | Ticker or display name (e.g. `RELIANCE` or `BTC-USD (Bitcoin)`) |
| **Price (₹)** | Closing price at analysis date |
| **Signal** | WRCI composite value. Positive = bullish momentum, negative = bearish. Magnitude indicates strength. |
| **Trend** | Normalised trend count. Positive = uptrend, negative = downtrend. Aligns with Signal for high-conviction setups. |
| **UMA State** | Unified Market Analysis highlight: `Bullish Div` · `Bearish Div` · `Confirmed Bullish` · `Confirmed Bearish` · empty |
| **Zone** | Market regime: `OB Extreme` · `OB` · `Neutral` · `OS` · `OS Extreme`. OB/OS signals carry exhaustion risk. |

**Timing groups:**

| Group | Meaning |
|---|---|
| Today | Signal fired on the analysis date — highest urgency |
| 1 Day Ago | Signal fired one session prior — still fresh |
| 2 / 3 Days Ago | Ageing signal — watch for follow-through or fade |
| Within 5 Days | Any signal in the 5-day window not captured above |

### Signal Strength Tab

Two side-by-side sections:
- **Threshold Signals** — Top 10 longs and top 10 shorts from Set A
- **Crossover Signals** — Top 10 longs and top 10 shorts from Set B

Each table shows rank, symbol, price, signal magnitude, trend direction, UMA State, and zone.

### Range Study Mode

Uses **Set C (Momentum)** exclusively — pure composite/signal crossovers without threshold filters. Outputs `LongSignal` / `ShortSignal` columns in the time-series DataFrame.

---

## Analysis Modes

### Single Date (Point Screener)

1. Select Universe + Index + Timeframe (Daily / Weekly / Monthly)
2. Set analysis date (defaults to today)
3. Click **◈ RUN SCREENER**
4. View Action Dashboard (dual signal sets) and Signal Strength tabs

Data fetched covers ~665 days back from the target date. If today's candle is absent (market still open), a live 1-day quote is appended.

### Date Range (Range Study)

1. Select Universe + Index + Timeframe
2. Switch Mode to **Date Range**
3. Set Start and End dates
4. Click **◈ RUN SCREENER**

Runs a multi-date pass: for each date in the range, WRCI is sampled and signals recorded. Output includes an interactive heatmap and WRCI charts for ~20 representative symbols across the period. Uses **Set C (Momentum)** logic exclusively for consistent historical tracking.

---

## Troubleshooting

### "Failed to fetch constituents" error on landing page

**Cause:** All three NSE data sources (API, CSV, Wikipedia) failed for the selected index.

**Solutions:**
- NSE's API throttles automated requests; wait 60 seconds and retry
- Try a different index that uses the CSV fallback (e.g. NIFTY 50 is reliably served)
- Check https://www.nseindia.com for planned downtime

### "No data returned" / analysis produces no results

**Cause:** yfinance couldn't download price data for the symbol list.

**Solutions:**
- Verify internet connection
- Try a smaller index (e.g. NIFTY BANK instead of NIFTY 500) to isolate the issue
- yfinance occasionally returns empty frames during market hours — retry after a few minutes

### Tables cut off on mobile

This was a known issue (fixed in current version). Tables use `overflow-x: auto` with `min-width: 480px` so narrow screens scroll horizontally instead of squishing columns. If you see truncation, hard-refresh the browser to clear cached CSS.

### Signal strength appears low across the board

WRCI is tuned with defaults: `reg_len=20`, `wt_n1=10`, `wt_n2=21`. These defaults work well for daily Indian large-cap equities. For smaller or more volatile universes (crypto, smallcap), signals naturally carry higher magnitude. Changing timeframe to Weekly often produces cleaner, stronger signals.

### App shows landing page instead of results after run

**Cause:** The screener returned `None` (constituent fetch or data download failed). The error message is displayed above the landing page — look for the red error banner at the top.

### Streamlit "connection lost" in browser

The terminal running `streamlit run sanket.py` shows the actual exception. Common causes: a dependency version conflict, or a timeout on large universes (NIFTY 500 + 500 days).

---

## Development Guide

### Adding a New Universe

1. **Define the map and list** in the constants block (~line 140):
```python
MY_MAP = {"Gold ETF": "GOLDBEES.NS", "Silver ETF": "SILVERBEES.NS"}
MY_LIST = list(MY_MAP.keys())
```

2. **Add a getter function**:
```python
def get_my_symbols():
    return list(MY_MAP.values()), f"✓ Loaded {len(MY_MAP)} symbols"
```

3. **Wire into `render_sidebar()`**:
```python
elif universe == "My Universe":
    selected_index = "My Universe"
```

4. **Wire into `run_screener_analysis()` and `run_timeseries_analysis()`**:
```python
elif universe == "My Universe":
    stock_list, msg = get_my_symbols()
```

5. **Add to `UNIVERSE_OPTIONS`**:
```python
UNIVERSE_OPTIONS = ["India Indexes", ..., "My Universe"]
```

### Adding a New India Index

1. Add the display name to `INDEX_LIST`
2. Add the NSE archive CSV URL to `INDEX_URL_MAP`:
```python
"NIFTY CONSUMPTION": f"{BASE_URL}ind_niftyconsumptionlist.csv"
```
The NSE JSON API is tried first (using the display name directly), so the CSV entry is only a fallback.

### Adjusting WRCI Parameters

Parameters are currently hardcoded in `render_sidebar()`:
```python
reg_len, wt_n1, wt_n2 = 20, 10, 21
obLevel1, obLevel2, osLevel1, osLevel2 = 80, 40, -80, -40
```

To make them user-configurable, replace these with `st.slider()` calls and pass them through to `run_screener_analysis()`.

### Adjusting Zone Thresholds

Modify the four levels above. Higher absolute values = fewer zone alerts (less sensitive). The thresholds feed directly into `run_full_analysis()` → `np.select()` classification.

---

## Performance Benchmarks

| Universe | Symbol Count | Fetch Time | WRCI Compute | Total |
|---|---|---|---|---|
| NIFTY BANK | ~12 | ~2s | <1s | ~3s |
| NIFTY 50 | ~50 | ~4s | <1s | ~5s |
| NIFTY 100 | ~100 | ~6s | <1s | ~7s |
| NIFTY 500 | ~500 | ~15s | ~2s | ~17s |
| F&O Stocks | ~180 | ~8s | ~1s | ~9s |
| Benchmark Indexes | 34 | ~3s | <1s | ~4s |

Range Study (100 days, 50 symbols): ~25–35s

---

## Known Limitations

- **No intraday support** — Daily and Weekly candles only; no hourly or sub-hourly WRCI
- **No live auto-refresh** — Manual re-run required; there is no websocket or polling loop
- **Historical depth** — yfinance reliably provides ~2 years; requests beyond that may return incomplete data
- **NSE API availability** — NSE throttles or blocks automated requests intermittently; the 3-source cascade mitigates but does not eliminate this
- **Benchmark Indexes data** — Some `NIFTY_*.NS` and `BSE-*.BO` tickers have limited history on yfinance; they are silently skipped if data is insufficient

---

## License

Copyright © 2026 Antigravity. All rights reserved.

Built as part of the **Pragyam** product family. See [LICENSE](LICENSE) for terms.

---

*SANKET v1.2.0 · @thebullishvalue · Pragyam / Antigravity*
