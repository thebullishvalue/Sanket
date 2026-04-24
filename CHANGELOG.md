# CHANGELOG — SANKET Signal Screener

All notable changes to SANKET are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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

---

## [v1.2.0] — April 2026

### Added
#### UMA v6 Intelligence Layer
- **MMR (Macro-Market Regime)** — Multi-state HMM regime detector (Bullish/Bearish/Side) for US and India markets
- **MSF (Market Signal Fusion)** — Multi-factor signal aggregator combining trend, momentum acceleration, and mean reversion
- **Diagnostic v6 Columns** — `MSF_Osc`, `MMR_Osc`, `Entropy`, `Hurst`, `Vol_Stress`, and `MMR_Quality` now available in system data
- **Asset Index Mode** — Bond Yields, Commodities, Currencies, and Crypto now run as unified asset indexes by default (e.g. "Global Bond Yields")

#### Signal Refinements
- **UMA States 2.0** — Refined 'Confirmed Bullish' and 'Confirmed Bearish' states based on MSF/MMR ensemble agreement
- **Lookback Window** — Signals now visible for up to 5 days in the timing tables, ensuring high-conviction setups aren't prematurely aged out

### Changed
#### Mathematical Architecture
- **Restored V4 Core Parity** — Reverted primary `Signal` (Unified_Osc) and `Trend` (Norm_Trend) calculations to original v4 linear counting to ensure numerical parity with the baseline system
- **V6 Diagnostic Separation** — Moved v6 "smart" oscillators to diagnostic columns, separating institutional intelligence from core trigger logic

#### Range Study 2.0
- **Synchronized Signal Counts** — Historical long/short counts now use Threshold + Crossover triggers instead of pure momentum crossovers for screener consistency
- **State Awareness** — Added HMM regime distribution and Divergence persistence tracking to the Range Study analytics

### Fixed
- **Swapped Percentages** — Corrected a critical logic swap where Overbought (OB) counts were reported as Oversold % and vice-versa in the Range Study
- **Macro Data Fallback** — Fixed crashes caused by `NoneType` results from Stooq yields by implementing robust `yfinance` fallbacks for US Treasury drivers
- **Undefined Variables** — Resolved `hlc3`, `wt1`, and `wt2` reference errors in the analytical pipeline

---

## [v1.1.0] — April 2026

### Added

#### Dual-Signal Engine — Three Independent Signal Sets
- **Set A — Threshold** — Composite crosses extreme level (−40 / +40) with `composite_signal` confirmation
- **Set B — Crossover** — Composite crosses its signal line while already in extreme zone
- **Set C — Momentum** — Pure crossover (no level filter), used exclusively by Range Study

#### Nested Signal Tabs — Screener Dashboard
- **Bullish Signals by Timing** tab now contains nested `[Threshold]` / `[Crossover]` sub-tabs
- **Bearish Signals by Timing** tab mirrors the same dual-set structure
- Each sub-tab displays timed signals (Today, 1d, 2d, 3d, 5d) with UMA State and Zone columns

#### Split Signal Strength View
- Signal Strength tab split into two sections: Threshold and Crossover
- Each section shows side-by-side Top 10 Longs and Top 10 Shorts tables

#### Separate CSV Exports
- Export buttons grouped by signal set: Threshold (Bullish/Bearish) and Crossover (Bullish/Bearish)
- Full results DataFrame includes all signal sets with prefixed columns

#### UMA State Column
- New `UMA State` column displays: `Bullish Div`, `Bearish Div`, `Confirmed Bullish`, `Confirmed Bearish`, or blank
- Embedded in timing tables and strength tables with dedicated column

#### Monthly Timeframe Support
- Added `Monthly` to `TIMEFRAME_OPTIONS`
- Confirmation logic is timeframe-aware: Daily → Weekly, Weekly → Monthly, Monthly → Daily

### Changed

#### WRCI Conditions Refactored
- Previous single `long_cond` / `short_cond` pair replaced by three named condition sets
- Backward-compatible aliases: `long_cond` and `short_cond` now point to Set C (Momentum) for Range Study

#### `run_full_analysis()` Signature
- Added `timeframe` parameter to select primary computation frequency (Daily/Weekly/Monthly)
- Internal resampling: `primary_df` and `confirm_df` selected based on timeframe
- Computes Set A, B, C in all cases; downstream consumers pick relevant ones

#### `run_screener_analysis()` Output
- Historical columns now prefixed: `L_Thresh_*`, `S_Thresh_*`, `L_Comp_*`, `S_Comp_*`
- Boolean flags: `LongSignal_Thresh`, `ShortSignal_Thresh`, `LongSignal_Comp`, `ShortSignal_Comp`
- Old `L_*` / `S_*` names removed (replaced by set-prefixed variants)

#### `run_timeseries_analysis()` Output
- Uses Set C exclusively; `LongSignal` and `ShortSignal` now reflect Momentum-cross logic
- Column names unchanged (preserves backward compatibility with existing range-study expectations)

#### Table Widths
- Increased `min-width` from `480px` to `600px` (timing tables) and `720px` (strength tables) to accommodate `UMA State` column

### Fixed

- Zone detection now correctly uses `composite_signal` threshold condition as designed
- Historical tracking bug where shifted conditions could mis-report prior-day signals across resampled timeframes
- Consistent NaN handling with `.fillna(False)` across all condition expressions

---

## [v1.0.0] — April 2026

Initial production release of the SANKET WRCI Signal Screener.

### Core Engine

- **WRCI (Wave-Regime Composite Index)** — Combines Wave Trend oscillator (EMA-smoothed channel index) with a normalised HMA-based trend directional count into a single composite signal line
- Signal detection via composite line / signal line crossovers
- Zone classification: OB Extreme (>80), OB (>40), OS (<-40), OS Extreme (<-80), Neutral
- Signal strength scoring with diminishing returns above magnitude 50 (prevents outliers from dominating rankings)
- Support for **Daily**, **Weekly**, and **Monthly** timeframes

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
- **Bullish / Bearish sub-tabs** — Separate tabs for long and short timing tables with per-tab accent colours (emerald / rose)
- **Signal interpretation legend** — Inline guide below timing tables explaining Signal, Trend, Zone, and Timing columns; styled with system glass card + amber accent

### Data Layer

- `fetch_batch_data()` — Parallel yfinance download with MultiIndex flattening; live 1-day append when today is absent
- `resample_to_weekly()` — Friday-close OHLCV aggregation
- `resample_to_monthly()` — Month-end OHLCV aggregation
- NSE F&O list — 3-source fetch: NSE F&O API → nsepython advances/declines → NIFTY 500 CSV

### Error Handling

- Fetch failures stored in `session_state["run_error"]` and displayed above the landing page — survives the `st.rerun()` cycle that previously swallowed error messages silently
- `run_error` key initialised at startup; cleared on every new run

### UI & Design

- **Obsidian Quant Terminal** theme — dark glass panels, amber (#D4A853) accent system, IBM Plex Mono + Space Grotesk typefaces
- Light mode toggle via `render_theme_toggle()`
- Custom SVG icon system (CHECK, LONG, SHORT, DOT, UP, DOWN, ZAP, CHART, STRENGTH, SETTINGS)
- Mobile-responsive HTML tables — `overflow-x: auto` + `min-width` guards + `touch-action: auto` prevents bottom-row clipping on narrow viewports
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
