# CHANGELOG — SANKET Signal Screener

All notable changes to SANKET are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [v2.1.0] — April 2026

Signal classification priority correction and documentation overhaul.

### Changed

- **Signal String Priority Order** — Corrected the signal type evaluation chain in `run_screener_analysis()`. Order is now `Set B → Set A → Set C → Zone` (previously was `Set B → Set C → Set A → Zone`). This ensures broad momentum crossovers (Set A) take precedence over threshold-entry events (Set C), aligning displayed `SignalType` with analytical intent: a composite/signal-line crossover anywhere is a more significant event than freshly crossing the ±40 zone boundary.
  - **Set B** (Crossover in Zone): `long_cond_comp` / `short_cond_comp` — highest priority
  - **Set A** (Momentum Crossover): `long_cond` / `short_cond` — second priority *(moved up)*
  - **Set C** (Threshold Entry): `long_cond_wt` / `short_cond_wt` — third priority *(moved down)*
  - **Zone**: Fallback label when no signal condition is active

### Documentation

- **README** — Full rewrite from scratch: accurate engine descriptions, signal set priority table, MSF/MMR/HMM pillar details, universe coverage table, project structure, and development guide.
- **CHANGELOG** — Overhauled; all entries back-filled with precise technical detail.
- **Version Synchronisation** — `VERSION` constant in `sanket.py`, `logger.py` docstring, `requirements.txt` header, `README.md`, and `CHANGELOG.md` all unified to `v2.1.0`.

---

## [v2.0.0] — April 2026

Major engine upgrade from the legacy WRCI screener to the UMA v6 Unified Market Analytics framework.

### Added — UMA v6 Engine

- **MSF (Momentum Structure Flow)** — Six-component composite oscillator replacing single-oscillator WRCI for the UMA flag layer:
  - ROC Momentum (Z-scored, sigmoid-mapped rate of change)
  - Microstructure (volume-weighted open-to-midpoint vs. 5-bar drift)
  - Composite Trend (four-sub-component: MA spread, double-diff acceleration, ATR normalisation, price-to-MA)
  - Accumulation / Distribution (rolling money-flow split by up/down closes)
  - Permutation Entropy dampener (reduces weight in high-entropy / choppy regimes)
  - Hurst Regime Weighting (variance-ratio Hurst exponent tilts component weights)
  - Volatility Structure Damper (ATR VoV + VTS amplitude reduction)
- **MMR (Macro Multiple Regression)** — Gram-Schmidt orthogonalized top-3 macro factor regression using Global Macro + Commodities + Currency universes; outputs Macro Context Score and rolling R².
- **Adaptive HMM Regime Discovery** — Hidden Markov Model (Gaussian emissions, 3 states: Bullish / Neutral / Bearish) for real-time signal environment classification.
- **UMA Flags** — Read-only, per-symbol context annotations: `Conf Bull`, `Conf Bear`, `Bull Div`, `Bear Div`. Surfaced in the results table as the `UMAFlag` column.
- **Volatility Awareness** — ATR-normalised VoV and VTS dampen signals during structurally noisy or whipsaw periods.
- **`compute_uma_flags()`** — Entry point for the UMA v6 information layer; runs MSF + MMR + HMM together on a sliding window of up to 300 most recent bars per symbol.

### Changed

- **Macro Context Caching** — `_MACRO_SYM_ORDERED` and `_MACRO_SYM_SET` defined at module level; macro context is pre-fetched once per run and reused across all symbols.
- **Smart Context Reuse** — When the screener universe overlaps macro context symbols (Commodities, Currency, Global Macro), already-downloaded Close series are recycled from `data_dict`—up to 113 fewer re-fetches on overlapping universes.
- **`_fetch_remaining_macro_context()`** — Cached helper that fetches only missing macro symbols not already in `data_dict`.
- **Institutional UI Refactor** — Glassmorphism panels, amber accent system, and signal dashboard components visually unified.
- **Optimised Compute Pipeline** — Signal fusion and macro context assembly vectorised; cold-start time reduced significantly on large universes.

---

## [v1.1.0] — April 2026

Expanded universe coverage, fetch optimisation, and structured Phase 2 logging.

### Added — Universes

- **Global Indexes** — 56-instrument universe of primary national equity benchmarks:
  - Americas (10): S&P 500, Dow Jones, NASDAQ 100, Russell 2000, TSX, IPC, Bovespa, Merval, IPSA, COLCAP
  - Europe (20): FTSE 100, DAX, CAC 40, IBEX 35, FTSE MIB, AEX, SMI, OMX Stockholm/Copenhagen/Helsinki, ATX, BEL 20, WIG 20, BIST 100, PSI 20, ASE, PX, BUX, MOEX
  - Asia-Pacific (20+): Nikkei 225, TOPIX, Shanghai Composite, CSI 300, Hang Seng, KOSPI, KOSDAQ, TAIEX, Nifty 50, Sensex, ASX 200, All Ordinaries, STI, KLCI, SET, Jakarta Composite, PSEi, NZX 50, VN-Index, KSE 100
  - Middle East & Africa (6): TA-125, Tadawul, DFM General, QE Index, JSE All-Share, EGX 30
  - Futures proxies used where cash indexes are unavailable on Yahoo Finance.

### Added — Data Fetching

- **`_fetch_remaining_macro_context()`** — Fetches only macro symbols absent from the current `data_dict`; integrated with module-level `_MACRO_SYM_ORDERED` and `_MACRO_SYM_SET`.
- **ASSET_NAME_LOOKUP expansion** — `GLOBAL_INDEXES_MAP` reverse-mapped into `ASSET_NAME_LOOKUP` for friendly name display in results tables.

### Fixed

- **Duplicate `@st.cache_data` decorator** on `get_fno_stock_list()` — removed the redundant wrapping that caused double-caching through Streamlit's machinery.

### Changed — Logging

- **Phase 2 Structured Parameters Section** — Terminal now clearly shows: Timeframe, Regression Length, Wave Trend N1/N2, OB/OS Levels, and fetch success count (e.g., "487 of 500 fetched successfully").
- **UMA v6 Macro Context Reuse Statistics** — Console output reports exact symbol counts reused vs. freshly fetched per run.
- **Signal Analysis Section Header** — Replaces orphaned "Technical Diagnostics" label; now shows instrument count and timeframe context (e.g., "Signal Analysis — 487 daily instruments").

### Changed — UI

- **Global Indexes in Universe Dropdown** — Positioned between India Indexes and US Indexes.
- **Sidebar Symbol Count** — Spec card shows both total configured and successfully-fetched instrument counts.
- **Universe Dispatch** — `render_sidebar()`, `run_screener_analysis()`, and `run_timeseries_analysis()` updated for Global Indexes.

---

## [v1.0.0] — April 2026

Initial production release of the SANKET WRCI Signal Screener.

### Added — Core Engine

- **WRCI (Wave-Regime Composite Index)** — Composite oscillator combining:
  - WaveTrend cycle (EMA-smoothed Channel Index, `wt_n1=10`, `wt_n2=21`)
  - HMA-based normalised trend count over `reg_len=20` bars
  - Composite Line = `(WT1 + Norm_Trend) / 2`
  - Signal Line = 4-period rolling mean of Composite Line
- **Zone Classification** — OB Extreme (> 80), OB (> 40), OS (< −40), OS Extreme (< −80), Neutral
- **Signal Strength Scoring** — Magnitude-based score with diminishing returns above 50 (prevents outlier dominance in rankings)
- **Daily and Weekly Timeframes** — Weekly via Friday-close OHLCV resampling (`resample_to_weekly()`)

### Added — Signal Sets

- **Set A (Momentum Crossover)** — Composite line crosses signal line anywhere (`long_cond` / `short_cond`)
- **Set B (Crossover in Zone)** — Composite crosses signal line inside OB/OS extreme zone (`long_cond_comp` / `short_cond_comp`)
- **Set C (Threshold Entry)** — Composite freshly crosses ±40 threshold with signal-line validation (`long_cond_wt` / `short_cond_wt`)

### Added — Universes

- **India Indexes** — 26 NIFTY indices (broad market, midcap, smallcap, 13 sectoral) + F&O Stocks + Benchmark Indexes (34 instruments)
- **Benchmark Indexes** — ^NSEI, ^NSMIDCP, BSE/NSE broad indices, full sectoral NSE suite — set as default universe
- **ETF Index** — 30 NSE-listed ETFs (sectoral, factor, thematic)
- **US Indexes** — S&P 500 (Wikipedia), DOW JONES (Wikipedia + fallback hardcoded 30), NASDAQ 100 (Wikipedia)
- **Commodities** — 24 commodity futures (precious metals, energy complex, agricultural softs, livestock)
- **Currency** — 24 FX pairs (G10 + major EM)
- **Crypto** — 21 cryptocurrencies by market cap

### Added — India Constituent Fetching

3-source cascade with automatic fallthrough per index:
1. **NSE JSON API** (`nseindia.com/api/equity-stockIndices`) — session-warmed, same endpoint as F&O
2. **NSE Archive CSV** (`archives.nseindia.com/content/indices/ind_{name}list.csv`)
3. **Wikipedia fallback** — for NIFTY 50, NEXT 50, BANK, IT, FIN SERVICE

Resolves silent failures on all sectoral indices (AUTO, FMCG, PHARMA, METAL, ENERGY, INFRA, REALTY, MEDIA, PSU BANK, PRIVATE BANK) previously only sourced from the NSE Archive CSV.

### Added — Analysis Modes

- **Snapshot (Single Date)** — WRCI sampled at a specific date; live intraday quote appended if today's candle is absent
- **Range Study** — Multi-date pass across a user-defined window with per-date heatmap and signal trend charts

### Added — Signal Display

- **Action Dashboard** — Signals grouped by age: Today / 1D Ago / 2D Ago / 3D Ago / Within 5D; each group shows count and average magnitude
- **Signal Strength Tab** — Top 8 bullish + top 8 bearish ranked by absolute magnitude
- **Bullish / Bearish sub-tabs** — Separate tabs with emerald / rose accent colours respectively
- **Signal Interpretation Legend** — Inline guide explaining Signal, Trend, Zone, and Timing columns

### Added — Data Layer

- `fetch_batch_data()` — Parallel yfinance batch download with MultiIndex flattening; live 1-day append when today is absent
- `resample_to_weekly()` — Friday-close OHLCV aggregation
- NSE F&O list — 3-source fetch: NSE F&O API → nsepython advances/declines → NIFTY 500 CSV fallback

### Added — UI

- **Obsidian Quant Terminal** — Dark glass panels, amber (#D4A853) accent, IBM Plex Mono + Space Grotesk typefaces
- Light/dark mode toggle via `render_theme_toggle()`
- Custom SVG icon system (CHECK, LONG, SHORT, DOT, UP, DOWN, ZAP, CHART, STRENGTH, SETTINGS)
- Mobile-responsive HTML tables (`overflow-x: auto` + `touch-action: auto`)
- Dynamic iframe height formula: `120 + groups×46 + rows×44`
- System spec card in sidebar (version, universe, timeframe, mode)
- Landing page with three system cards: Signal Engine, Signal Types, Universe Coverage

### Added — Console Logger (`logger.py`)

- `ConsoleOutput` class — direct stdout with ANSI colours; bypasses Python's `logging` module
- Phase timers, section headers, per-symbol detail lines, run summaries
- Unique run ID per analysis: `YYYYMMDD_HHMMSS_uuid8`
- Windows 10+ ANSI compatibility via colorama or manual VT100 enable

### Added — Error Handling

- Fetch failures stored in `session_state["run_error"]` and displayed above the landing page — survives `st.rerun()` cycles that previously swallowed errors silently
- `run_error` key initialized at startup; cleared on every new run

---

## Version Numbering

`v[MAJOR].[MINOR].[PATCH]`

| Segment | Trigger |
|---|---|
| **MAJOR** | New universe type, breaking output schema change, or significant engine architecture change |
| **MINOR** | New features, signal logic changes, UI additions, additional index coverage |
| **PATCH** | Bug fixes, fetch reliability improvements, copy/documentation corrections |

---

## Roadmap

### Near-term
- [ ] Hourly / 4-hour timeframe support
- [ ] Auto-refresh mode (configurable polling interval)
- [ ] User-defined CSV import for custom universes
- [ ] Alert export (email / Telegram on new signals)

### Medium-term
- [ ] Signal backtesting module — historical hit-rate per zone and signal type
- [ ] Sector rotation heatmap
- [ ] Correlation matrix for multi-asset confluence view

### Long-term
- [ ] REST API endpoint for external integrations
- [ ] Strategy builder with user-defined entry/exit rules

---

*Maintained by Antigravity / Pragyam · @thebullishvalue*
