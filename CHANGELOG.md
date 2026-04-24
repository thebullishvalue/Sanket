# CHANGELOG — SANKET Signal Screener

All notable changes to SANKET are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [v2.0.0] — April 2026

Major engine upgrade from legacy WRCI to the UMA v6 Unified Market Analytics framework.

### UMA v6 Analysis Engine

- **MSF (Momentum Structure Flow)** — Multi-component composite signal combining ROC momentum, microstructure (price-volume variance), composite trend (Hurst-adjusted), and accumulation/distribution metrics.
- **MMR (Macro Multiple Regression)** — Integrated macro context awareness using Gram-Schmidt orthogonalization across a 3-variable top-ranked macro basket (Global Macro, Commodities, Currency).
- **Adaptive HMM (Hidden Markov Model)** — Real-time regime state discovery (Bullish/Neutral/Bearish) for signal classification and noise reduction.
- **Enhanced Signal Flags** — New read-only context flags for top symbols:
    - **Conf Bull/Bear**: High signal agreement between MSF and MMR.
    - **Bull/Bear Div**: Momentum divergence against price action.
- **Volatility Structure Awareness** — Dynamic signal damping based on ATR-normalized volatility variance (VoV) and trend-structure ratios (VTS).

### UI & Performance

- **Institutional UI Refactor** — Absolute visual uniformity across all signal dashboard components.
- **Enhanced Terminal Aesthetics** — Refined glassmorphism and amber accent system.
- **Optimized Compute Pipeline** — Faster signal fusion and macro context caching.

---

## [v1.1.0] — April 2026

Expanded market coverage and constituent reliability improvements.

### Universes

- **Global Indexes** — Integrated 56 primary equity benchmark indexes across Americas, Europe, Asia-Pacific, and Middle East/Africa.
- **Global Macro** — Added 40+ macro instruments including US Treasuries (full curve), Direct Yield Indices, Inflation-Protected (TIPS), and Global Sovereign bonds.
- **Benchmark Indexes Mode** — Dedicated mode for tracking index instruments as assets rather than their constituents.

### Data & Reliability

- **3-Source Cascade Fetch** — Improved reliability of India index constituents: NSE JSON API → NSE Archive CSV → Wikipedia fallback.
- **Intraday Quote Injection** — Automated appending of live candles for same-day analysis when historical feeds are lagging.

---

## [v1.0.0] — April 2026

Initial production release of the SANKET WRCI Signal Screener.

### Core Engine

- **WRCI (Wave-Regime Composite Index)** — Combines Wave Trend oscillator (EMA-smoothed channel index) with a normalised HMA-based trend directional count into a single composite signal line
- Signal detection via composite line / signal line crossovers (Long Cross, Short Cross)
- Zone classification: OB Extreme (>80), OB (>40), OS (<-40), OS Extreme (<-80), Neutral
- Signal strength scoring with diminishing returns above magnitude 50 (prevents outliers from dominating rankings)
- Support for **Daily** and **Weekly** timeframes; weekly via Friday-close OHLCV resampling

### Universes

- **India Indexes** — 26 NIFTY indices (broad market, midcap, smallcap, sectoral) + F&O Stocks + Benchmark Indexes (34 instruments)
- **Benchmark Indexes** — ^NSEI, ^NSMIDCP, ^INDIAVIX, ^BSESN, BSE-100/200/500, ^NSEBANK, ^CNXFIN, ^CNXIT, ^CNXAUTO, ^CNXFMCG, ^CNXPHARMA, ^CNXMETAL, ^CNXREALTY, ^CNXENERGY, ^CNXINFRA, ^CNXPSUBANK, NIFTY_PRIVATE_BANK.NS, ^CNXMEDIA, and full NSE midcap/smallcap suite — set as default universe
- **ETF Index** — 30 NSE-listed ETFs
- **US Indexes** — S&P 500, DOW JONES, NASDAQ 100
- **Commodities** — 16 commodity futures (Gold, Silver, Crude WTI/Brent, Natural Gas, Copper, Wheat, Corn, etc.)
- **Currency** — 25 FX pairs
- **Crypto** — 21 cryptocurrencies

### India Index Constituent Fetching

3-source cascade with automatic fallthrough:
1. **NSE JSON API** — `nseindia.com/api/equity-stockIndices?index={NAME}` (session-warmed, same endpoint as F&O)
2. **NSE Archive CSV** — `archives.nseindia.com/content/indices/ind_{name}list.csv`
3. **Wikipedia fallback** — Index-specific page for NIFTY 50, NEXT 50, BANK, IT, FIN SERVICE

Sectoral indices (AUTO, FMCG, PHARMA, METAL, ENERGY, INFRA, REALTY, MEDIA, PRIVATE BANK, PSU BANK) previously failed silently because NSE archive CSV was the only source; now resolved by NSE API as primary.

### Analysis Modes

- **Single Date (Point Screener)** — WRCI sampled at a specific date; live intraday quote appended if today's candle is absent from the feed
- **Date Range (Range Study)** — Multi-date pass across a user-defined window; per-date results with heatmap and WRCI charts for representative symbols

### Signal Display

- **Action Dashboard** — Signals grouped by age: Today / 1 Day Ago / 2 Days Ago / 3 Days Ago / Within 5 Days; each group shows count and average magnitude
- **Signal Strength Tab** — Top 8 bullish + top 8 bearish ranked by absolute magnitude
- **Bullish / Bearish sub-tabs** — Separate tabs for long and short timing tables with per-tab accent colours (emerald / rose)
- **Signal interpretation legend** — Inline guide below timing tables explaining Signal, Trend, Zone, and Timing columns; styled with system glass card + amber accent

### Data Layer

- `fetch_batch_data()` — Parallel yfinance download with MultiIndex flattening; live 1-day append when today is absent
- `resample_to_weekly()` — Friday-close OHLCV aggregation
- NSE F&O list — 3-source fetch: NSE F&O API → nsepython advances/declines → NIFTY 500 CSV

### Error Handling

- Fetch failures stored in `session_state["run_error"]` and displayed above the landing page — survives the `st.rerun()` cycle that previously swallowed error messages silently
- `run_error` key initialised at startup; cleared on every new run

### UI & Design

- **Obsidian Quant Terminal** theme — dark glass panels, amber (#D4A853) accent system, IBM Plex Mono + Space Grotesk typefaces
- Light mode toggle via `render_theme_toggle()`
- Custom SVG icon system (CHECK, LONG, SHORT, DOT, UP, DOWN, ZAP, CHART, STRENGTH, SETTINGS)
- Mobile-responsive HTML tables — `overflow-x: auto` + `min-width: 480px` + `touch-action: auto` prevents bottom-row clipping on narrow viewports
- Dynamic iframe height formula: `120 + groups×46 + rows×44` (buffers for scrollbar chrome and section headers)
- System spec card in sidebar (version, universe, timeframe, mode)
- Landing page with three system cards: Signal Engine, Signal Types, Universe Coverage

### Console Logger

- `ConsoleOutput` class in `logger.py` — direct stdout with ANSI colours; bypasses Python's logging module
- Phase timers, section headers, run summaries, per-symbol detail lines
- Compatible with Windows 10+ (colorama or manual VT100 enable)
- Unique run ID per analysis: `YYYYMMDD_HHMMSS_uuid8`

---

## Version Numbering

`v[MAJOR].[MINOR].[PATCH]`

- **MAJOR** — New universe, breaking change to output schema, or significant engine change
- **MINOR** — New features, UI additions, additional index coverage
- **PATCH** — Bug fixes, fetch reliability improvements, copy/documentation corrections

---

## Roadmap

### Near-term
- [ ] Hourly / 4-hour timeframe support
- [ ] Auto-refresh mode (configurable polling interval)
- [ ] User-defined CSV import for custom universes
- [ ] Alert export (email / Telegram on new signals)

### Medium-term
- [ ] Signal backtesting module — historical hit-rate per zone type
- [ ] Sector rotation heatmap
- [ ] Correlation matrix for multi-asset confluence view

### Long-term
- [ ] REST API endpoint for external integrations
- [ ] Strategy builder with user-defined entry/exit rules

---

*Maintained by Antigravity / Pragyam · @thebullishvalue*
