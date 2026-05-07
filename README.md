# SANKET — Institutional Market Signal Terminal
### Wave-Regime Composite Index · Obsidian Quant · Pragyam Family · `v3.1.0`

> **संकेत** *(Sanketa)* — Sanskrit for *Signal* · *Indicator* · *Forewarning*

Sanket is an institutional-grade quantitative market signal screener and analysis terminal built on the **Wave-Regime Composite Index (WRCI)**. It identifies abnormal market acceleration ("Pulse"), regime transitions, and high-conviction tactical setups across global asset classes — with verified **1:1 mathematical parity** between the Python screener and the TradingView Pine Script indicator.

Part of the **Pragyam Product Family** by [@thebullishvalue](https://github.com/thebullishvalue).

---

## Contents

- [What Sanket Does](#what-sanket-does)
- [Architecture Overview](#architecture-overview)
- [The WRCI Engine](#the-wrci-engine)
- [Signal Hierarchy (Sets A–D)](#signal-hierarchy-sets-ad)
- [Priority Engine](#priority-engine)
- [Intelligence — Self-Tuning Calibration](#intelligence--self-tuning-calibration)
- [Analysis Modes](#analysis-modes)
- [Asset Universe Coverage](#asset-universe-coverage)
- [UI System — Obsidian Quant](#ui-system--obsidian-quant)
- [Terminal Logging](#terminal-logging)
- [Project Structure](#project-structure)
- [Installation & Launch](#installation--launch)
- [Deployment](#deployment)
- [Configuration & Profiles](#configuration--profiles)
- [TradingView Pine Script Indicator](#tradingview-pine-script-indicator)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## What Sanket Does

Most screeners rank stocks on price momentum or RSI. Sanket does something different: it measures the **rate of change of momentum** — the acceleration signal — and combines it with regime-aware conviction scoring to rank assets not just by direction but by the quality and intensity of the move.

The core question Sanket answers: **which assets are experiencing abnormal regime acceleration right now, and how much conviction does each signal carry?**

Key capabilities:

- **Pulse detection** — identifies abnormal acceleration by modulating short-term velocity with long-term statistical intensity (Z-Score normalization)
- **Asymmetric prioritization** — long and short factor weights are learned independently; the system does not assume symmetry
- **Cross-asset breadth scanning** — runs simultaneously across NSE F&O, Indian indices, US equities, crypto, commodities, FX, and global macro ETFs
- **Self-tuning calibration** — Bayesian hyperparameter optimization (Optuna TPE) over a 21-dimensional asymmetric weight space
- **1:1 Pine Script parity** — every calculation in Python exactly mirrors the TradingView indicator, eliminating signal divergence between screener and chart
- **Historical regime intelligence** — bulk time-series analysis with rolling IC computation for signal quality validation

---

## Architecture Overview

```
sanket.py                  ← Main Streamlit entry point (UI, routing, analysis dispatch)
priority_engine.py         ← Asymmetric factor scoring + per-universe profile persistence
intelligence.py            ← Bayesian calibration engine (Optuna TPE, 21-dim search space)
logger.py                  ← Structured terminal logging (ANSI color, phase timing, run IDs)
wrci.pine                  ← TradingView Pine Script v6 (mathematical mirror of Python engine)
ui/
  theme.py                 ← CSS injection, Plotly Obsidian theme, progress cards
  theme.css                ← Full Obsidian Quant design system
  components.py            ← Reusable UI primitives (headers, metric cards, signal tables)
profiles/
  fno.json                 ← Sample calibrated profile for NSE F&O universe
requirements.txt           ← Pinned runtime dependencies
```

**Runtime artifact (not in repo):**

```
~/.sanket/profiles.json    ← Per-machine calibration profiles, one entry per (universe, index)
```

---

## The WRCI Engine

The **Wave-Regime Composite Index** is a multi-layer quantitative engine composed of three orthogonal components that feed into a unified signal hierarchy.

### 1. WaveTrend Core (WT1 / WT2)

The foundation is a normalized momentum oscillator computed from HLC3 (applied price):

```
ESA  = EMA(price, n1)                            # baseline
dist = EMA(|price − ESA|, n1)                    # volatility envelope
CI   = (price − ESA) / (0.015 × dist)            # composite index
WT1  = EMA(CI, n2)                               # signal line
WT2  = SMA(WT1, 4)                               # trigger line
```

Default periods: `n1 = 10`, `n2 = 21`. WT1 and WT2 crossovers define the primary momentum signal.

### 2. Conviction Engine

Conviction measures *quality* of the momentum regime across three orthogonal dimensions and combines them into a single score `[−100, +100]`:

| Component | Measurement | Timescale |
|:---|:---|:---|
| **Trend Strength** | HMA slope normalized by ATR | Slow · structural |
| **Momentum Quality** | WT1−WT2 separation | Medium · tactical |
| **Participation** | Volume Z-Score × price direction + RSI confluence | Fast · confirming |

```
Conviction = w1 × TrendStrength + w2 × MomentumQuality + w3 × Participation
           ∈ [−100, +100]
```

### 3. Pulse Engine (Abnormal Acceleration)

The Pulse Engine is the core innovation. It measures acceleration rather than velocity — detecting when momentum is building abnormally relative to its own recent statistical distribution.

```
velocity    = Conviction(t) − Conviction(t−3)         # 3-bar rate of change
baseline    = rolling mean of lagged conviction         # 30-bar, non-overlapping
σ_baseline  = rolling std of baseline
z           = (velocity − baseline) / (σ_baseline + ε) # Z-Score of velocity

vol_factor  = tanh(VolumeZScore / 2)                   # volume confirmation [−1, +1]
pa_factor   = Return / ATR_percentile                  # price-action normalization

Pulse = clip(z × vol_factor × pa_factor, −6, +6)
```

A Pulse reading of `±3` or greater represents a statistically significant acceleration event — the kind that precedes regime continuations or accelerated reversals.

### 4. Orthogonal Factors (F1–F2)

Two additional factors computed outside the WaveTrend framework:

- **F1 — Price Momentum**: Log return normalized by ATR percentile (trend-adjusted)
- **F2 — Volume Quality**: Volume Z-Score × price direction, smoothed by SMA — distinguishes conviction-backed moves from noise

### 5. Pulse Narrative Matrix

Each bar is classified into a 4×4 state matrix based on Pulse delta versus Conviction delta:

|  | **SURGE** | **FIRM** | **SOFT** | **CRUSH** |
|:---|:---|:---|:---|:---|
| **LEAD** | Peak acceleration building | Sustained trend | Deceleration warning | Trend weakening |
| **DEEP** | Strong base, accelerating | Deep trend continuation | Late cycle | Exhaustion |
| **LIGHT** | Shallow but explosive | Trend with light vol | Fading | Mean-reversion likely |
| **HOLLOW** | Volume-less move | Structural but weak | Divergence alert | Distribution |

These labels appear in the Pulse Narrative analysis mode and in the sidebar signal interpreter.

---

## Signal Hierarchy (Sets A–D)

Every market signal is classified into one of four non-redundant tiers. Tier multipliers (learned by the Intelligence engine) weight each signal class by historical effectiveness.

| Set | Type | Trigger Condition | Use Case |
|:---|:---|:---|:---|
| **A** | Momentum | WT1/WT2 crossover outside extreme zones | Tactical trend-following in liquid regimes |
| **B** | Contrarian | WT1/WT2 cross *inside* overbought/oversold zones | High-probability mean-reversion identification |
| **C** | Threshold | Zone entry/exit gates (±60 level crossings) | Volatility regime transition detection |
| **D** | Squeeze | Bollinger/Keltner volatility compression breakout | Positioning ahead of explosive expansion |

**Signal firing rules**:
- Sets A and B are mutually exclusive per bar (no double-counting crossovers)
- Set C fires when price crosses threshold levels but WT1/WT2 alignment is not yet confirmed
- Set D is independent — squeeze compression can co-exist with A/B/C

**Tier multipliers** are optimized per universe by the Intelligence engine. The sample F&O profile shows: `A=0.96, B=0.55, C=0.94, D=1.16`, meaning Set D squeeze signals are slightly overweighted relative to contrarian Set B in that universe.

---

## Priority Engine

The Priority Engine (`priority_engine.py`) translates raw WRCI outputs into a single ranked score per asset, independently for long and short sides.

### Factor Composition (F1–F6)

| Factor | Input | Description |
|:---|:---|:---|
| **F1** | Log return / ATR | Price momentum, volatility-adjusted |
| **F2** | Vol Z-Score × price direction | Volume quality signal |
| **F3** | Normalized conviction | Structural trend magnitude |
| **F4** | Pulse engine output | Abnormal acceleration |
| **F5** | HMM Bull − HMM Bear | Regime state differential |
| **F6** | Cross-sectional rank | Relative rank within universe |

### Asymmetric Weighting

Each factor has **separate long and short betas** (`beta_F4_pulse_long`, `beta_F4_pulse_short`). This is the key architectural choice: the system does not assume that what makes a good long signal makes a good short signal. In practice, regime factors tend to be more predictive on the short side, while pulse factors can have asymmetric magnitudes.

### Penalty Terms

Two penalty adjustments modify the raw score:

- **Reversion risk**: penalizes assets where Wave > +60 but conviction is declining (long) or Wave < −60 but conviction is rising (short)
- **Divergence penalty**: applied when price and indicator diverge at extremes — the classic warning sign of trend exhaustion

### Damping Factors

Applied equally to both sides:

| Damping Factor | Value |
|:---|:---|
| LOW volatility regime | ×1.2 |
| NORMAL volatility regime | ×1.0 |
| HIGH volatility regime | ×0.85 |
| EXTREME volatility regime | ×0.55 |
| Tier multiplier | Per signal class (learned) |
| Regime confidence | [0.6 – 1.0] |
| Change-point penalty | Applied on detected regime shifts |

**Output**: `Priority_Long` and `Priority_Short` in basis-point-equivalent units of forward return. Assets are ranked descending on `Priority_Long` for long screening.

---

## Intelligence — Self-Tuning Calibration

The Intelligence module (`intelligence.py`) uses Bayesian hyperparameter optimization to learn optimal factor weights from historical data.

### Algorithm

- **Optimizer**: Optuna Tree-structured Parzen Estimator (TPE) — Bayesian, not grid search
- **Search space**: 21-dimensional
  - 6 × (long beta + short beta) = 12 factor weights
  - 2 × reversion penalty (long + short) = 2 penalty weights
  - 4 tier multipliers (A, B, C, D)
  - Remaining: cross-sectional rank and damping parameters
- **Objective**: Maximize out-of-sample Information Ratio (Spearman IC averaged across hold periods)
- **Train/Val split**: 70% in-sample, 30% out-of-sample validation — results report **val IC only**
- **L2 regularization**: Prevents runaway weight inflation

### Pre-computed Dataset (50× Speed Boost)

The `_PrecomputedDataset` class computes weight-invariant arrays once before the optimization loop begins. Factor matrices, penalty arrays, tier indices, and return ranks are computed once and reused across all Optuna trials — delivering approximately 50× speedup versus naïve per-trial recalculation.

### Parameter Importance (fANOVA)

After optimization, Optuna's fANOVA analysis ranks each weight by its sensitivity contribution to the objective. The sample F&O profile shows:

```
Pulse_long:    25.2%   ← dominant long-side driver
Regime_short:  23.4%   ← dominant short-side driver
Tier_C:        14.7%   ← threshold signal quality matters significantly
```

### Profile Persistence

Calibrated weights are saved per (universe, selected_index) key in `~/.sanket/profiles.json`. When the user switches universe, the engine auto-loads the matching profile. The `_maybe_migrate_legacy_profile()` function handles v1 → v2 key migration for backwards compatibility.

---

## Analysis Modes

### 1. Single Date Screener

Fetches OHLCV data for all constituents of the selected universe on a specific date, runs the full WRCI + Conviction + Pulse pipeline, computes Priority scores, detects squeezes and divergences, and returns a ranked table sorted by `Priority_Long`.

**Output includes**:
- Priority_Long / Priority_Short scores
- Conviction, Pulse, WT1, WT2, RSI, Histogram
- Squeeze state (True/False)
- Signal tier classification (A/B/C/D)
- Divergence flag
- Regime confidence

### 2. Pulse Narrative Analysis

Runs the same pipeline as the screener but renders the **Pulse Narrative Matrix** for each asset — showing the current Pulse state (velocity × conviction quadrant) with interpretation text. Useful for understanding *why* an asset is ranked where it is.

### 3. Historical Range Analysis (Time-Series)

Bulk analysis over a user-specified date range. Aggregates daily signals, computes rolling metrics, and prepares the dataset for Intelligence calibration or export.

**Output includes**:
- Day-by-day signal history per asset
- Rolling IC (Spearman rank correlation of Priority vs. forward return)
- Signal count and hit-rate by tier
- Export to CSV/Excel

### 4. Intelligence (Self-Tuning)

Launches the Optuna calibration run on the historical dataset. Configurable trial count (default ~100). Results are displayed in the Intelligence Center with:
- Val IC achieved
- Best weights table
- Parameter importance chart (fANOVA)
- One-click save to `~/.sanket/profiles.json`

### 5. Correlation Analysis

Cross-asset correlation and confluence detection. Identifies assets with aligned or diverging WRCI signals — useful for building cross-hedged positions or confirming sector-wide regime shifts.

---

## Asset Universe Coverage

| Universe Group | Constituents |
|:---|:---|
| **NSE F&O** | All NSE F&O permitted stocks (dynamic, via nsepython) |
| **India Indices** | 28+ NIFTY indices: NIFTY 50, NIFTY 500, Bank, IT, Pharma, Midcap, Smallcap, sectoral |
| **US Equities** | S&P 500 constituents (Wikipedia scrape), NASDAQ symbols |
| **Global Indices** | International equity benchmark indices |
| **ETF Index** | 30 pre-defined Indian ETFs (gold, health, chemicals, infra, etc.) |
| **Commodities** | Gold, Silver, Crude Oil, Natural Gas futures |
| **Currencies** | INR/USD, EUR/USD, GBP/USD, JPY/USD |
| **Crypto** | Bitcoin, Ethereum, major caps |
| **Global Macro** | Bond ETFs, treasury instruments, global macro indicators |

**Data sources**:
- NSE India API via `nsepython` (F&O lists, index constituents)
- Yahoo Finance via `yfinance` (OHLCV + volume)
- Wikipedia (S&P 500 constituent list)
- NSE official CSV exports (index weightings)

---

## UI System — Obsidian Quant

The terminal runs in Streamlit with a fully custom design layer called **Obsidian Quant** — a precision-instrument aesthetic optimized for quantitative data density.

### Design Language

| Element | Specification |
|:---|:---|
| Background | `#1a1a1a` — dark obsidian |
| Primary text | `#F8FAFC` — bright white |
| Secondary text | `#94A3B8` — soft slate |
| Accent — bullish | `#22c55e` — emerald green |
| Accent — bearish | `#ef4444` — signal red |
| Accent — neutral | `#4a9eff` — quantum blue |
| Accent — amber | `#D4A853` — institutional gold |
| Display font | Syne, Space Grotesk |
| Monospace font | JetBrains Mono, IBM Plex Mono |

### Component Library (`ui/components.py`)

- `render_metric_card()` — animated metric display with staggered entrance
- `render_system_card()` — elevated surface card with fade-in
- `render_signal_item()` — signal strength indicator with SVG icons
- `render_conviction_signal()` — conviction-band display
- `render_signal_guide()` — Set A–D reference panel
- `render_interpretation_card()` — Pulse narrative + state matrix renderer
- `render_export_button_row()` — CSV/Excel download row
- `render_collapsible_section()` — expandable panel with close logic

### Chart Theming (`ui/theme.py`)

All Plotly charts use the Obsidian Quant theme:
- Font: JetBrains Mono (data), Syne (display)
- Grid: `rgba(255,255,255,0.035)` — barely-there
- Hover: unified X-axis mode, dark background tooltips
- Progress cards: animated fills with pulse dots

---

## Terminal Logging

Sanket uses a custom structured logging system (`logger.py`) that writes directly to `sys.stdout` — bypassing Python's `logging` module entirely for clean, grep-able terminal output.

### Log Format

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SANKET TERMINAL — Session Start  v3.1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Run ID     │  20260507_142301_a3f8
  Universe   │  NSE F&O
  Mode       │  Screener
  Date       │  2026-05-07
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ Fetched 182 symbols                    [  0.8s]
  ✓ WRCI computation complete              [  3.2s]
  ✓ Priority scoring complete              [  3.4s]
  ✓ Screener ranked — top 20 returned     [  3.6s]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Logger API

```python
console.header(title, version)          # Section banner
console.main_header(title, metadata)    # Analysis run start with metadata dict
console.item(label, value)              # Key-value pair
console.line()                          # Separator
```

**Colorama** provides Windows 10+ ANSI compatibility. All output uses UTF-8 with an ASCII fallback for restricted terminals.

---

## Project Structure

```
Sanket/
├── sanket.py               # Main entry — Streamlit UI + analysis dispatch (5,400 lines)
├── priority_engine.py      # Asymmetric priority engine + profile persistence (372 lines)
├── intelligence.py         # Self-tuning Bayesian calibration via Optuna (408 lines)
├── logger.py               # Structured terminal logging system (250+ lines)
├── wrci.pine               # TradingView Pine Script v6 — mathematical mirror (378 lines)
├── requirements.txt        # Pinned runtime dependencies
├── LICENSE                 # Proprietary institutional license
├── README.md               # This file
├── CHANGELOG.md            # Version history
├── .gitignore              # Excludes __pycache__, ~/.sanket, IDE metadata
├── profiles/
│   └── fno.json            # Sample calibrated profile for NSE F&O universe
├── ui/
│   ├── theme.py            # Obsidian Quant theme injection + chart styling
│   ├── theme.css           # Full design-system CSS (~100KB)
│   └── components.py       # Reusable UI primitives
└── .devcontainer/
    └── devcontainer.json   # VSCode dev container configuration
```

**Runtime artifact (not committed):**

```
~/.sanket/profiles.json     # Calibrated weight profiles, keyed by (universe, index)
```

---

## Installation & Launch

### Requirements

- Python 3.10 or higher
- Git

### Steps

```bash
# Clone the repository
git clone https://github.com/thebullishvalue/Sanket.git
cd Sanket

# Install dependencies
pip install -r requirements.txt

# Launch the terminal
streamlit run sanket.py
```

The app will open at `http://localhost:8501` in your browser.

### First Run

On first launch, no calibration profile exists. The Priority Engine will use default symmetric weights. To generate calibrated weights for your target universe:

1. Select a universe in the sidebar (e.g., NSE F&O)
2. Set a historical date range (90+ days recommended)
3. Run **Historical Range Analysis** to build the dataset
4. Switch to **Intelligence** mode and run calibration (100+ trials recommended)
5. Save the profile — it will auto-load on subsequent runs for that universe

---

## Deployment

### Streamlit Cloud

1. Push to GitHub (`.gitignore` excludes `__pycache__/`, `.sanket/`, IDE folders, OS artifacts)
2. Connect to [Streamlit Cloud](https://share.streamlit.io/)
3. Set entry point: `sanket.py`
4. Note: `~/.sanket/profiles.json` resets on container rebuild — use the **Export Profile** button in the Model Passport (sidebar) to download and re-import calibrations after redeployment

### Self-Hosted (VM / Container)

```bash
pip install -r requirements.txt
streamlit run sanket.py --server.port 8501 --server.headless true
```

The `~/.sanket/` directory is created automatically on first calibration save. No manual setup required.

### Dev Container

A `.devcontainer/devcontainer.json` is included for VSCode Remote Containers. Open the folder in VSCode and select "Reopen in Container" to get a pre-configured Python 3.10 environment.

---

## Configuration & Profiles

### Profile Storage

Calibrated profiles are stored at `~/.sanket/profiles.json` on the host machine. Each profile is keyed by a composite `(universe, selected_index)` string.

**Example profile structure:**

```json
{
  "NSE_FO__NIFTY50": {
    "weights": {
      "beta_F1_pricemom_long":  3.8,
      "beta_F1_pricemom_short": 2.1,
      "beta_F4_pulse_long":     1.35,
      "beta_F4_pulse_short":    7.11,
      "beta_F5_regime_long":    32.86,
      "beta_F5_regime_short":   45.38,
      "tier_multiplier_A":      0.96,
      "tier_multiplier_B":      0.55,
      "tier_multiplier_C":      0.94,
      "tier_multiplier_D":      1.16
    },
    "val_ic":   0.1556,
    "train_ic": 0.2198,
    "timestamp": "2026-05-07 12:35",
    "importance": {
      "beta_F4_pulse_long":    0.252,
      "beta_F5_regime_short":  0.234,
      "tier_multiplier_C":     0.147
    }
  }
}
```

### Model Passport

The **Model Passport** panel (sidebar) provides:
- View active profile weights for the current universe
- Export profile as JSON (for sharing or backup)
- Import a JSON profile
- Delete a profile to revert to default weights

---

## TradingView Pine Script Indicator

`wrci.pine` is the companion TradingView indicator — **WRCI [Sanket Core]** (`v3.1.0`). It implements the same mathematical pipeline as the Python engine:

| Calculation | Python | Pine Script |
|:---|:---|:---|
| Applied price | `(H+L+C)/3` | `hlc3` |
| ESA | `EMA(price, n1)` | `ta.ema(src, n1)` |
| Conviction | 3-component weighted | Identical weights |
| Pulse velocity | 3-bar diff | `conviction - conviction[3]` |
| Z-Score normalization | `(v − μ) / σ` | Rolling window, identical |
| Volume factor | `tanh(volZ / 2)` | `math.tanh(volZ / 2)` |

The indicator is Pine Script v6, published as an overlay-false indicator with precision 2 and `max_bars_back=500`.

**Color palette matches the Obsidian Quant UI**:
- Background: `#1a1a1a`
- WT1 bullish: `#22c55e`
- WT1 bearish: `#ef4444`
- Pulse positive: `#4a9eff`
- Pulse negative: `#E8555A`

---

## Tech Stack

| Layer | Technology |
|:---|:---|
| Language | Python 3.10+ |
| Web Framework | Streamlit 1.30+ |
| Numerical Computing | NumPy 1.24+, Pandas 2.1+ |
| Interactive Charts | Plotly 5.18+ |
| Bayesian Optimization | Optuna 3.5+ (TPE sampler) |
| NSE India Data | nsepython |
| US/Global Data | yfinance 0.2.31+ |
| HTML Parsing | BeautifulSoup4, lxml, html5lib |
| Excel Export | openpyxl 3.1+ |
| Terminal Colors | colorama 0.4.6+ |
| Pine Script | v6 (TradingView) |

---

## License

Proprietary — institutional usage only.

Copyright © 2026 [@thebullishvalue](https://github.com/thebullishvalue). All rights reserved.

This software, including the WRCI engine, signal detection logic, Pulse computation, priority scoring methodology, and UI design system, is the exclusive intellectual property of @thebullishvalue. Distribution, modification, sublicensing, and commercial exploitation are prohibited without prior written consent.

Signals produced by this system do not constitute financial advice. @thebullishvalue accepts no liability for trading or investment losses.

See [`LICENSE`](LICENSE) for full terms.

---

*Sanket v3.1.0 · Pragyam Family · Built by [@thebullishvalue](https://github.com/thebullishvalue)*
