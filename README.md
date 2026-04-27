# SANKET v2.1.0 — Unified Market Signal Screener

> **संकेत** — *Signal. Indicator. Omen.*
> Institutional-grade quantitative terminal for multi-asset momentum discovery and regime analysis.

SANKET is a production-ready quantitative signal screener built on the **UMA v6 (Unified Market Analytics)** engine. It combines a multi-component momentum oscillator, macro multiple regression, and hidden-Markov regime classification to surface high-conviction signals across global equity indexes, macro instruments, commodities, currencies, and crypto.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Signal Engine: UMA v6](#signal-engine-uma-v6)
  - [WRCI — Core Oscillator](#wrci--core-oscillator)
  - [Signal Sets A / B / C](#signal-sets-a--b--c)
  - [MSF — Momentum Structure Flow](#msf--momentum-structure-flow)
  - [MMR — Macro Multiple Regression](#mmr--macro-multiple-regression)
  - [Adaptive HMM — Regime Classification](#adaptive-hmm--regime-classification)
  - [UMA Flags](#uma-flags)
- [Universes & Coverage](#universes--coverage)
- [Signal Output Reference](#signal-output-reference)
- [Analysis Modes](#analysis-modes)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Development Guide](#development-guide)
- [License](#license)

---

## Features

- **UMA v6 Engine** — Three-pillar intelligence layer: MSF momentum, MMR macro context, and Adaptive HMM regime classification united under a single signal flag system.
- **WRCI Composite Oscillator** — Wave Trend cycle fused with an HMA-based normalised trend count into one composite oscillator + signal line.
- **Three Signal Sets** — Hierarchical signal classification (Set B → Set A → Set C → Zone) covering crossover-in-zone events, broad momentum crossovers, and threshold-entry signals.
- **MSF (Momentum Structure Flow)** — Six-component composite: ROC momentum, microstructure, composite trend, accumulation/distribution, permutation-entropy dampening, and Hurst-regime weighting.
- **MMR (Macro Multiple Regression)** — Gram-Schmidt orthogonalized top-3 macro factor regression for macro context scoring and R² tracking.
- **Adaptive HMM Regime Discovery** — Hidden Markov Model (Bullish / Neutral / Bearish) classifies signal reliability zone and filters noise.
- **8 Market Universes** — India Indexes (NIFTY suite + F&O), Global Indexes (56 country benchmarks), Global Macro (bond ETFs, yield curves, credit spreads), US Indexes, NSE ETFs, Commodities (24 futures), Currencies (24 FX pairs), and Crypto (21 assets).
- **3-Source India Constituent Fetch** — NSE JSON API → NSE Archive CSV → Wikipedia, with automatic fallthrough per index.
- **Smart Macro Context Reuse** — When screener universe overlaps macro context symbols, already-downloaded series are reused instead of re-fetched.
- **Intraday Quote Injection** — Live 1-day candle appended automatically when today's close is absent from the historical feed.
- **Obsidian Quant Terminal UI** — Dark-mode terminal design with amber accents, glassmorphism surfaces, IBM Plex Mono + Space Grotesk typography, and ANSI-colored console logger.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.8+ |
| Web Framework | Streamlit ≥ 1.30 |
| Market Data | yfinance (global) · NSE JSON API · nsepython (India) |
| Data Processing | pandas ≥ 2.1 · NumPy ≥ 1.24 (vectorized) |
| Statistics | SciPy · scikit-learn · statsmodels |
| Charts | Plotly ≥ 5.18 (institutional theming) |
| UI | Custom CSS — Obsidian Quant Terminal (CSS variables) |
| Logging | `logger.py` — ANSI console via stdout (colorama / VT100) |

---

## Signal Engine: UMA v6

### WRCI — Core Oscillator

The **Wave-Regime Composite Index** is the screener's primary oscillator:

1. **HLC3 HMA** — Hull Moving Average of the typical price and volume series, smoothing microstructure noise.
2. **Trend Count** — For each of the last `reg_len` bars, adds +1 if HMA is rising, −1 if falling. Result is normalised to a ±100 range (`Norm_Trend`).
3. **WaveTrend Cycle** — EMA-smoothed Channel Index (`WT1`) captures medium-frequency momentum swings.
4. **Composite Line** — `(WT1 + Norm_Trend) / 2` — the unified oscillator.
5. **Signal Line** — 4-period rolling mean of the composite line, used for crossover detection.

Zone thresholds (default): OB Extreme > 80, OB > 40, OS < −40, OS Extreme < −80.

---

### Signal Sets A / B / C

Three mutually-exclusive signal categories are evaluated in **priority order Set B → Set A → Set C → Zone**. The first matching condition wins.

| Priority | Set | Logic | Signal Labels |
|:---:|:---:|---|---|
| 1 | **Set B** — Crossover in Zone | Composite crosses signal line **inside** an OB/OS extreme zone (> ±40) | Long Crossover · Short Crossover |
| 2 | **Set A** — Momentum Crossover | Composite crosses signal line **anywhere** (broad directional shift) | Long Momentum · Short Momentum |
| 3 | **Set C** — Threshold Entry | Composite freshly crosses the ±40 threshold with signal-line validation | Long Threshold · Short Threshold |
| 4 | Zone | No signal condition active; exposes the current OB/OS/Neutral zone label | OB Extreme · OB · OS · OS Extreme · Neutral |

---

### MSF — Momentum Structure Flow

A six-component composite that extends the WRCI for the UMA flag layer (read-only context, not the primary screener ranking):

| Component | What it measures |
|---|---|
| **ROC Momentum** | Z-scored rate-of-change, sigmoid-mapped |
| **Microstructure** | Volume-weighted open-to-midpoint vs. 5-bar drift |
| **Composite Trend** | Four-sub-component normalised trend: MA spread, double-diff acceleration, ATR-normalised momentum, price-to-MA deviation |
| **Accumulation / Distribution** | Rolling money-flow partitioned by up/down closes |
| **Permutation Entropy** | Order-3 price-pattern entropy dampener — reduces signal weight in choppy, high-entropy regimes |
| **Hurst Regime Weighting** | Variance-ratio Hurst exponent tilts component weights toward trend-following or mean-reversion |
| **Volatility Structure Damper** | ATR Variance-of-Variance (VoV) + Volatility-Trend-Strength (VTS) reduce MSF amplitude during structurally noisy periods |

---

### MMR — Macro Multiple Regression

Quantifies how much of an asset's move is driven by systemic macro factors vs. idiosyncratic strength:

1. Correlates the asset's close series against all symbols in **Global Macro + Commodities + Currency** universes over the last 200 bars.
2. Selects the **top 3 highest-correlation** macro factors.
3. Gram-Schmidt orthogonalizes the 3 regressors to eliminate multicollinearity.
4. Fits a rolling OLS regression to produce a **Macro Context Score** and rolling **R²**.

A high R² means the asset is macro-driven; a low R² with strong MSF indicates idiosyncratic momentum.

---

### Adaptive HMM — Regime Classification

A Hidden Markov Model continuously re-evaluates the market environment:

| State | Label | Implication |
|:---:|:---:|---|
| 0 | **Bullish** | High-conviction trend-following signals; long bias appropriate |
| 1 | **Neutral** | Mean-reversion / range-trading logic prioritised |
| 2 | **Bearish** | Aggressive risk management; short-bias signals elevated |

---

### UMA Flags

Read-only context annotations generated by `compute_uma_flags()` for top-ranked symbols:

| Flag | Condition |
|---|---|
| **Conf Bull** | MSF signal and MMR score both positive and above threshold → high cross-pillar agreement |
| **Conf Bear** | MSF and MMR both negative and below threshold → high cross-pillar agreement |
| **Bull Div** | Price making lower lows while oscillator makes higher lows → bullish divergence |
| **Bear Div** | Price making higher highs while oscillator makes lower highs → bearish divergence |

---

## Universes & Coverage

| Universe | Instruments | Count |
|---|---|---|
| **India Indexes** | NIFTY 50, NEXT 50, 100/200/500, Midcap, Smallcap, Sectoral (Bank, IT, FMCG, Pharma, Auto, Metal, Energy, Infra, Realty, Media) + F&O Stocks + Benchmark Indexes | 26 index options |
| **Global Indexes** | Primary national equity benchmarks — Americas (10), Europe (20), Asia-Pacific (20+), Middle East & Africa (6) | 56 instruments |
| **Global Macro** | US Treasury curve (1M–30Y), TIPS, Aggregate/Corporate IG/HY bonds, Mortgage-Backed, Municipal, Emerging Market sovereign debt, India G-Sec, Eurozone / UK / Japan / Australia govt bonds | 65+ assets |
| **US Indexes** | S&P 500 constituents · Dow Jones 30 · NASDAQ 100 | 3 index options |
| **NSE ETFs** | Sectoral, factor, and broad-market NSE-listed ETFs | 30 assets |
| **Commodities** | Precious metals, energy complex, agricultural softs, livestock | 24 futures |
| **Currency** | G10 + major EM FX pairs vs. USD | 24 pairs |
| **Crypto** | Top 21 digital assets by market cap | 21 assets |

---

## Signal Output Reference

Every asset in the screener results table exposes the following fields:

| Field | Description |
|---|---|
| **Signal** | Unified Oscillator value (composite line). Higher absolute value = stronger momentum. |
| **Trend** | Normalised HMA trend count (Norm_Trend). Positive = uptrend, negative = downtrend. |
| **Wave** | Raw WaveTrend (WT1) value — the cyclical momentum component. |
| **Zone** | Current oscillator zone: OB Extreme · OB · Neutral · OS · OS Extreme. |
| **SignalType** | Active signal condition from the Set B → A → C priority hierarchy. |
| **UMA** | UMA v6 context flag (Conf Bull · Conf Bear · Bull Div · Bear Div · —). |
| **Set B (LB/SB)** | Crossover-in-zone signal history: Today · 1D · 2D · 3D · ≤5D. |
| **Set A (LA/SA)** | Momentum crossover signal history: Today · 1D · 2D · 3D · ≤5D. |
| **Set C (LC/SC)** | Threshold-entry signal history: Today · 1D · 2D · 3D · ≤5D. |
| **PctChange** | Day-over-day % change from previous close. |

---

## Analysis Modes

### Snapshot (Single Date)
Evaluates the WRCI composite at one point in time. Best for identifying fresh signals at today's market close. If today's candle is absent from yfinance (intraday), a live quote is appended automatically.

### Range Study (Historical Evolution)
Runs the full analysis pipeline across a user-defined date window (up to ~500 bars). Produces:
- Signal volume trend charts (Long / Short count over time)
- OB / OS zone distribution over time
- Regime Analysis tab (HMM Bullish % / Bearish % / Transition % over time)
- Transaction Dynamics tab (buy-sell signal counts and L/S ratio)
- Raw Data Terminal (full per-date results table)

---

## Getting Started

```bash
# 1. Clone repository
git clone <repository-url>
cd Sanket-main

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch application
streamlit run sanket.py
```

Python 3.8 or later required. On Windows, `colorama` is included in `requirements.txt` to enable ANSI color output in the console logger.

---

## Project Structure

```
Sanket-main/
├── sanket.py          # Main application — engine, data fetching, screener, UI entry
├── logger.py          # ANSI console logger (ConsoleOutput class)
├── requirements.txt   # Python dependencies
├── sector_map.pkl     # Cached sector/industry mapping for India equities
├── ui/
│   ├── theme.py       # CSS injection, chart theming, progress bar helper
│   ├── theme.css      # Obsidian Quant Terminal design system
│   └── components.py  # Reusable Streamlit UI components
├── README.md
├── CHANGELOG.md
└── LICENSE
```

---

## Development Guide

### Engine Parameters

Parameters are hardcoded in `render_sidebar()` and passed through to `run_screener_analysis()`:

| Parameter | Default | Description |
|---|---|---|
| `reg_len` | 20 | Trend count lookback for Norm_Trend and HMA window |
| `wt_n1` | 10 | WaveTrend EMA span (channel smoothing) |
| `wt_n2` | 21 | WaveTrend signal EMA span (average smoothing) |
| `obLevel1` | 80 | Overbought Extreme threshold |
| `obLevel2` | 40 | Overbought threshold (also used for Set B/C) |
| `osLevel1` | −80 | Oversold Extreme threshold |
| `osLevel2` | −40 | Oversold threshold (also used for Set B/C) |

### Adding a New Universe

1. Define a ticker map `MY_UNIVERSE_MAP = {"Name": "TICKER", ...}` in the constants block.
2. Add a helper `get_my_universe_symbols()` following the existing pattern.
3. Add the universe name to `UNIVERSE_OPTIONS`.
4. Wire the new option into `render_sidebar()`, `run_screener_analysis()`, and `run_timeseries_analysis()`.

### Signal Priority

Signal type is assigned in `run_screener_analysis()` at market-close sampling. Priority chain:

```
Set B (Crossover in Zone)  →  Set A (Momentum Crossover)  →  Set C (Threshold Entry)  →  Zone label
```

Conditions are defined in `run_full_analysis()` as boolean columns on the DataFrame.

---

## License

Copyright © 2026 Antigravity / Pragyam. All rights reserved.

---

*SANKET v2.1.0 · @thebullishvalue · Pragyam / Antigravity*
