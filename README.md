# SANKET v2.1.0 — Unified Market Signal Screener

> **संकेत** — *Signal. Indicator. Omen.*
> Institutional-grade quantitative terminal for multi-asset regime discovery and momentum analysis.

SANKET (v2.1.0) integrates the **UMA v6 (Unified Market Analytics)** engine, **Analog Engine v2** directional accuracy system, and **NIRNAY** regime intelligence features into a production-ready quantitative terminal. It applies multi-component momentum scoring, macro multiple regression, historical pattern matching with Mahalanobis distance, and hidden markov regime state discovery across global equity indexes, macro instruments, commodities, and crypto assets.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [UMA v6 Analysis Engine](#uma-v6-analysis-engine)
- [Universes & Coverage](#universes--coverage)
- [Signal Output Reference](#signal-output-reference)
- [Analysis Modes](#analysis-modes)
- [Getting Started](#getting-started)
- [Development Guide](#development-guide)
- [License](#license)

---

## Features

- **UMA v6 Analysis Engine** — A unified intelligence layer combining MSF momentum, MMR macro context, and Adaptive HMM regime classification.
- **MSF (Momentum Structure Flow)** — High-precision composite oscillator combining Wave Trend, volatility-adjusted ROC, and accumulation/distribution metrics.
- **MMR (Macro Multiple Regression)** — Real-time macro-context scoring using Gram-Schmidt orthogonalization against top-ranked global macro baskets.
- **Adaptive HMM Regime Discovery** — Hidden Markov Model state detection (Bullish/Neutral/Bearish) classifies signal reliability and filters noise.
- **Analog Engine v2** — Full Mahalanobis distance-based directional accuracy system:
    - **6-Dimensional Feature Space** — composite_line (WRCI), RSI, oscillator, MA alignment count, volume trend, mean reversion
    - **Gram-Schmidt Orthogonalization** — Produces 6 normalized orthogonal basis vectors for accurate pattern matching
    - **Temporal Decay Weighting** — Recent analogs weighted higher (250-bar half-life)
    - **Win Rate & Profit Factor** — Directional accuracy from 5-bar forward returns with risk-adjusted metrics
    - **Confidence Grading** — STRONG (≥70% accuracy + ≥1.5 PF), MODERATE (≥55% + ≥1.0 PF), WEAK
- **Multi-Universe Scanning** — India Indexes (NIFTY suite), Global Indexes (56 country benchmarks), Global Macro (Bond ETFs, Yields), US Indexes, NSE ETFs, Commodities, Currency, and Crypto.
- **Signal Interpretation Flags** — Advanced contextual signals combining UMA and Analog engines:
    - **UMA Flags**: Conf Bull/Bear (high signal agreement), Bull/Bear Div (oscillator divergence)
    - **Analog Flags**: ▲ BULL (>55% win rate), ● NEUTRAL (45-55%), ▼ BEAR (<45%)
- **Institutional UI (Obsidian Quant)** — Dark-mode terminal design with amber accents, glassmorphism surfaces, and precision typography (IBM Plex Mono + Space Grotesk).
- **3-Source Constituent Fetch** — High-reliability India index data: NSE JSON API → NSE Archive CSV → Wikipedia fallback.
- **Intraday Quote Injection** — Automated appending of live candles for same-day analysis when historical feeds are lagging.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.8+ |
| Web Framework | Streamlit |
| Market Data | yfinance (global), NSE API + nsepython (India) |
| Data Processing | pandas, NumPy (vectorized operations) |
| Statistics | Adaptive HMM, Gram-Schmidt Regression |
| Charts | Plotly (Institutional Theming) |
| UI | Custom CSS (Obsidian Quant Terminal), CSS Variables |
| Logging | Custom ANSI console logger (`logger.py`) |

---

## UMA v6 Analysis Engine

The core intelligence lives in `compute_uma_flags()` and integrates three primary quantitative pillars:

### 1. MSF (Momentum Structure Flow)
A composite oscillator that builds upon the legacy WRCI. It calculates price-volume variance and microstructure trend flow to identify momentum shifts before they appear in standard moving averages.

### 2. MMR (Macro Multiple Regression)
Calculates the relative strength of an asset against a macro-weighted index. By identifying how much of a stock's move is attributed to systemic macro factors vs. idiosyncratic strength, MMR assigns a "Macro Context Score."

### 3. Adaptive HMM (Hidden Markov Model)
A state-machine that constantly re-evaluates the market environment.
- **State 0 (Bullish)**: High-conviction trend following allowed.
- **State 1 (Neutral)**: Mean-reversion and range-trading logic prioritized.
- **State 2 (Bearish)**: Aggressive risk-management and short-bias signals.

---

## Universes & Coverage

| Universe | Description | Count |
|---|---|---|
| **India Indexes** | NIFTY 50, NIFTY Next 50, NIFTY 500, etc. | 26 Indices |
| **Global Indexes** | Primary benchmarks (S&P 500, DAX, Nikkei, etc.) | 56 Countries |
| **Global Macro** | Bond ETFs (TLT, AGG), Yields, TIPS, Credit Spreads | 40+ Assets |
| **US Indexes** | Major US Sector ETFs and Benchmarks | 30+ Assets |
| **Commodities** | Gold, Silver, Crude Oil, Natural Gas, etc. | 25+ Assets |
| **Currency** | Major G10 and EM Pairs | 35+ Pairs |
| **Crypto** | Top Market Cap Digital Assets | 20+ Assets |

---

## Signal Output Reference

The Screener Dashboard displays the following key metrics for every asset in the chosen universe:

| Column | Description |
|---|---|
| **Signal** | The Unified Oscillator value (-100 to +100). Marks momentum magnitude. |
| **Trend** | The underlying trend directionality score. |
| **Analog** | Directional accuracy from Analog Engine v2 (▲ BULL / ● NEUTRAL / ▼ BEAR · win_rate%). |
| **UMA** | High-conviction flags (Conf Bull, Bull Div, etc.) generated by UMA v6. |
| **Zone** | Qualitative classification (OB Extreme, OS, Neutral, etc.) |
| **Timing** | Age of the most recent signal (Today, 1D, 2D, 3D, <5D). |

---

## Analysis Modes

### 1. Single Date Mode
Provides a snapshot of the market at a specific point in time. Best for identifying fresh signals at the current market open/close.

### 2. Date Range Mode (Historical Evolution)
Analyzes how signals have evolved over a window of time. Identifies "sticky" signals and cluster events where multiple assets in a sector trigger simultaneously.

---

## Getting Started

```bash
# 1. Obtain project
git clone <repository-url>
cd Sanket-Final

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run application
streamlit run sanket.py
```

---

## Development Guide

### Adjusting Engine Parameters
Parameters are defined in `render_sidebar()` and passed to the analysis engine. 
- `reg_len`: Length for trend count (default: 20)
- `wt_n1 / wt_n2`: Wave trend smoothing factors

### License
Copyright © 2026 Antigravity. All rights reserved.
Built as part of the **Sanket** product family.

---

*SANKET v2.1.0 · @thebullishvalue · Pragyam / Antigravity*
