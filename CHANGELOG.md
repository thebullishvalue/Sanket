# CHANGELOG
### Sanket — Wave-Regime Composite Index Terminal

All notable changes to the **Sanket** platform are documented here. Sanket is part of the **Pragyam Product Family** by [@thebullishvalue](https://github.com/thebullishvalue).

Format: `[version] · date — release title`

---

## [v3.1.0] · 2026-05-07
### Documentation & Version Unification

**"Uniform Signal"**

Version alignment pass bringing all system components — main application, UI theme, logger, and Pine Script indicator — to a single canonical version string. Accompanied by a full documentation rewrite.

#### Changes
- **Version unification**: `sanket.py`, `ui/theme.py`, and `logger.py` all bumped to `v3.1.0`, matching the existing `wrci.pine` indicator version
- **README rewrite**: Complete documentation overhaul — architecture deep-dive, engine internals, factor math, signal hierarchy, intelligence calibration workflow, deployment guide, profile structure reference
- **CHANGELOG rewrite**: Full version history reconstructed from v2.0.0 to present with accurate release notes
- **LICENSE update**: Product version updated to `v3.1.0`

---

## [v3.0.0] · 2026-05-06
### Production Calibration Release

**"The Asymmetric Engine"**

First production-grade release of the asymmetric Priority Engine with Bayesian self-tuning. Separated long and short factor betas, introduced per-universe profile persistence, and shipped the Intelligence Center UI.

#### Features
- **Asymmetric Priority Engine**: Separate `beta_*_long` and `beta_*_short` weights for all six factors — the system no longer assumes long/short symmetry
- **Intelligence Center**: Full Optuna TPE calibration workflow surfaced in UI — trial count control, live progress, val IC display, parameter importance chart
- **Model Passport**: Sidebar panel for viewing, exporting, importing, and deleting calibration profiles per universe
- **fANOVA Sensitivity Analysis**: Post-calibration parameter importance ranking using Optuna's fANOVA implementation
- **Per-Universe Profile Auto-Loading**: `load_profile_for()` auto-selects the matching profile when the user switches universe
- **F&O Sample Profile**: Seed calibration (`profiles/fno.json`) with pre-optimized weights for NSE F&O universe (val IC: 0.1556)
- **Legacy Profile Migration**: `_maybe_migrate_legacy_profile()` handles v1 → v2 key format upgrade

#### Internals
- **`_PrecomputedDataset`**: Pre-computes weight-invariant arrays before Optuna trials begin — ~50× speedup over per-trial recalculation
- **L2 regularization**: Added to `_evaluate_ic()` to prevent weight inflation in low-signal regimes
- **Change-point penalty**: Regime shift detection now applies a damping multiplier at structural breaks

---

## [v2.2.0] · 2026-05-05
### Obsidian Quant Transformation & Mathematical Parity

**"The Obsidian Quant Transformation"**

Final synchronization pass between the Python screener and the TradingView indicator. Achieved 1:1 mathematical parity across all calculations. Applied the Obsidian Quant design system terminal-wide.

#### Features
- **1:1 Mathematical Parity**: Unified HMA, WMA, and Linear Regression endpoint calculations across `sanket.py` and `wrci.pine` — screener signals now match chart signals exactly
- **Pulse Engine (v3)**: Implemented abnormal acceleration detection — 3-bar velocity modulated by 20-bar volatility Z-Score, with volume factor (`tanh(volZ/2)`) and price-action factor
- **Obsidian Quant UI**: Applied the full Obsidian design language — `#1a1a1a` background, JetBrains Mono data font, Syne display font, glass-morphism metric cards, staggered entrance animations
- **Pulse Narrative Matrix**: 4×4 state grid (SURGE/FIRM/SOFT/CRUSH × LEAD/DEEP/LIGHT/HOLLOW) for per-bar signal interpretation
- **Global Macro ETFs**: Expanded asset coverage to include global bond ETFs and treasury instruments
- **Fractal MTF Anchoring**: Integrated daily and weekly macro-regime context for tactical signal filtering

#### Fixes & Optimizations
- **VWAP Accuracy**: Refactored to ratio-of-sums instead of discrete averaging — eliminates accumulation drift
- **Signal Gating**: Hardened mutual exclusivity rules for Sets A–D — no double-firing on same bar
- **Anti-Clustering**: Enhanced pattern matching with anti-clustering logic to prevent redundant analog matches
- **Colorama Compatibility**: Fixed ANSI color rendering on Windows 10+ terminals via explicit `colorama.init()`

---

## [v2.1.0] · 2026-04-30
### WRCI Foundation

**"Wave-Regime Composite Index Core"**

Initial release of the WRCI engine and its companion Pine Script indicator. Established the WaveTrend core, Conviction engine, and signal hierarchy framework.

#### Features
- **WRCI Core Engine**: WaveTrend oscillator (WT1/WT2) with normalized HLC3 applied price
- **Conviction Engine**: Three-component composite score — Trend Strength (HMA slope / ATR), Momentum Quality (WT separation), Participation (Volume Z-Score × price direction)
- **Signal Sets A–D**: Defined non-redundant signal classification framework — Momentum, Contrarian, Threshold, Squeeze
- **Squeeze Engine**: Bollinger Band / Keltner Channel compression detection for volatility breakout identification
- **Pine Script v6 Indicator**: First release of `wrci.pine` companion TradingView indicator
- **Analog Pattern Matcher**: Cosine similarity-based historical pattern matching engine (first iteration)
- **Overbought/Oversold Logic**: Zone definitions at `±60` (Threshold) and `±80` (Extreme)

---

## [v2.0.0] · 2026-04-15
### Pragyam Family Rebirth

**"Modular Architecture & Web Terminal"**

Architectural rebuild from standalone script to the Pragyam modular product framework. Launched the Streamlit web terminal.

#### Changes
- **Pragyam Architecture**: Decomposed monolithic script into `sanket.py`, `priority_engine.py`, `intelligence.py`, `logger.py`, and `ui/` module
- **Streamlit Terminal**: Replaced CLI with interactive Streamlit web UI — session state management, sidebar routing, mode switching
- **Multi-Universe Scraping**: Introduced constituent scraping for NSE F&O (nsepython), NASDAQ, and S&P 500 (Wikipedia)
- **Obsidian Quant Design System**: Initial `theme.css` and `components.py` — dark terminal aesthetic with Plotly chart theming
- **Structured Logging**: Replaced `print()` statements with `ConsoleOutput` class — ANSI-colored, phase-timed, run-ID-tagged terminal output

---

*Full technical specifications: [README.md](README.md)*
*Author: [@thebullishvalue](https://github.com/thebullishvalue) · Pragyam Family*
