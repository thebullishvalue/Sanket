# CHANGELOG
### Sanket — Wave-Regime Composite Index Terminal

All notable changes to the **Sanket** platform are documented here. Sanket is part of the **Pragyam Product Family** by [@thebullishvalue](https://github.com/thebullishvalue).

Format: `[version] · date — release title`

---

## [v3.2.1] · 2026-05-21
### Set A Δ-Polarity Gate & Signal Engine Refactor

**"Symmetric Conviction"**

Behavior-changing tightening of the Momentum signal gate plus a focused refactor of `sanket.py` to extract long-lived inline blocks and surface the sidebar return as a typed dataclass. Pine indicator (`wrci.pine`) updated in lockstep to preserve 1:1 parity.

#### Behavior Change — Set A
- **Δ-polarity gate added to Set A** (Momentum): WT1/WT2 crossings now require `Conviction Δ` and `Pulse Δ` to be the same sign as the trade direction (long: both > 0; short: both < 0). Brings Set A in line with the gate already used by Sets B and D. Net effect: fewer but better-confirmed Set A signals; historical Set A counts will drop. The opposite-side Set B veto (long A blocked when B-short fires, and vice versa) is retained on top of the new gate.
- **`wrci.pine` synchronized**: Pine `momentum_long` / `momentum_short` now carry the same Δ gate, preserving 1:1 mathematical parity between the Python screener and the TradingView indicator.

#### Refactor (No Behavior Change)
- **`compute_signal_sets` helper extracted**: Sets A/B/C/D logic and the zone `Condition` column moved out of `run_full_analysis` into a dedicated function with a docstring explaining each set's predicate, gating, and the load-bearing `np.select` ordering for the zone label.
- **`SidebarState` dataclass**: `render_sidebar()` now returns a typed dataclass instead of a 16-element positional tuple. The data flow from sidebar to `main()` is name-keyed; new inputs no longer require updating a tuple unpack.
- **Shared HTML-builder palette helpers**: `_side_palette`, `_signed_color`, `_delta_arrow`, and `_GREEN` / `_RED` constants dedupe the green/red and arrow ternaries that were repeated across `_build_confluence_table_html`, `_build_signal_table_html`, `_build_narrative_table_html`, and `_build_signal_strength_table_html`.
- **Redundant imports removed**: Four local `import html as html_module` lines deleted — the top-level `import html` covers all `html.escape` callsites. Dead `from io import BytesIO` removed (all callers go through `io.BytesIO()`).

#### Documentation
- **README Signal Hierarchy table rewritten**: Sets B, C, D descriptions now accurately reflect the regime-filter crossover, signal-line-validated zone entry, and regime-zero-cross triggers respectively (previous text described unrelated logic).
- **README search-space breakdown corrected**: Now correctly enumerates 12 betas + 4 gammas (reversion + divergence, each side) + 5 tier multipliers = 21 dimensions.
- **README profile JSON example fixed**: Uses real field names (`val_score`/`train_score`/`sensitivity`/`tier_A_mult`) and the actual `" · "`-joined composite key format.
- **README line counts refreshed**: `sanket.py` (5,928) and `wrci.pine` (412).
- **`ui/components.py` Signal Types Reference rewritten**: Sets A and B descriptions match the actual triggers; Set D card added (CSS class `.signal-type.squeeze` already existed in `theme.css`, but the HTML card was missing).
- **Mislabeled comments fixed**: Two `# Set C: Momentum` comments at `sanket.py:2402-2408` and `sanket.py:5147` corrected to `# Set A: Momentum` (these labelled the legacy `L_`/`S_` alias columns, which read `long_cond` / `short_cond` — Set A's columns, not Set C's).

---

## [v3.2.0] · 2026-05-09
### System Hardening, Fidelity & UI Polish

**"Precision Instrument"**

Comprehensive institutional-grade hardening pass across the full stack — data correctness, multi-session isolation, calibration reliability, and terminal UI smoothness. No surface-level UX changes; all improvements are under-the-hood quality and fidelity improvements.

#### System Architecture
- **Per-session weight isolation**: Active weights now stored in `st.session_state["active_weights"]` per session, eliminating cross-user bleed when the app is deployed for multiple users simultaneously
- **Smart data registry**: TTL-aware fetch cache keyed by universe + date — avoids redundant OHLCV fetches across mode switches (15 min during market hours, 90 min outside)
- **Registry DataFrame copies**: Registry stores `.copy()` of each DataFrame, preventing downstream mutations from silently corrupting cached data
- **Reproducible calibration**: Optuna TPE sampler seeded with `seed=42` — calibration results are now reproducible across identical inputs
- **`HOLD_HORIZONS` constant**: Fibonacci-spaced horizons `[2, 3, 5, 8, 13]` extracted to `priority_engine.py` as a single source of truth, imported by `sanket.py` and `intelligence.py` — previously hardcoded in five separate places

#### Correctness Fixes
- **L/S Ratio division guard**: Long/Short signal ratio now emits `NaN` instead of `Inf` when short signal count is zero
- **Divergence order scaled to timeframe**: `argrelextrema` order parameter set to `2` for weekly and `3` for daily data — was fixed at `3` regardless of timeframe, causing missed weekly divergences
- **Regime detector warmup**: HMM state estimator now runs a 20-bar warm-up period before recording signal history, preventing false regime transitions at the start of the analysis window
- **ymax NaN/Inf guard**: Bar chart y-axis maximum now guarded against `NaN`/`Inf` values that caused silent chart rendering failures on short date ranges
- **Confluence score clipped**: `Confluence_Score` clamped to `[0.0, 1.0]` on both calibrated-priority and fallback paths — previously could exceed 1.0 on wide-spread cross-sections
- **% Change Since sentinel**: Percentage-change-since-analysis field now uses `None` sentinel instead of `0.0` when the analysis date equals the latest available date — eliminates spurious 0.00% displays

#### Calibration Improvements
- **Overfit detection split**: Separated `low_ir` (Val IR ≤ 0) and `overfit` (Train IR >> Val IR) flags — both are now detected independently with distinct user messages
- **Quality Check card**: Fourth metric card added to Calibration Diagnostics showing `No Edge` / `Overfit` / `Quality OK` status with semantic color coding
- **Small universe warning**: Calibration warns when average symbols per date falls below 20 — IC-based ranking is statistically unreliable on sparse cross-sections
- **Exception logging**: All `except: pass` patterns replaced with typed exception logging via `console.detail()` — silent failures are now surfaced in the terminal

#### Display & Data Quality
- **Run stats header**: Results header now shows total universe size, symbols fetched, analyzed, and failed — provides context that was previously invisible
- **Column display rename**: Result table columns use human-readable names (`Priority_Long_pct` → `Long Priority %ile`, `F1_PriceMom` → `Price Momentum`, etc.)
- **Widget state persistence**: All sidebar widgets keyed with `sb_*` session-state keys — widget selections persist across reruns without unexpected resets
- **Timeframe-aware passport**: Calibration profile keys now include timeframe — daily and weekly calibrations are stored and loaded independently under the same universe

#### UI Smoothness (No Visual Changes)
- **Skeleton shimmer suppressed**: `[data-testid="stSkeleton"] { display: none !important }` eliminates the native Streamlit shimmer that briefly appears between a button click and the first progress bar render
- **CSS loading cached**: `@st.cache_resource` on `_load_theme_css()` — the 4,300-line `theme.css` is read from disk once per process; subsequent reruns pay zero I/O cost
- **Equal-height metric cards**: `:has(.metric-card)` CSS block extends the flex chain through `element-container` and `stMarkdownContainer` — metric card rows are now equal height on all pages (Single Date, Pulse Narrative, Calibration Diagnostics, Historical Range, Intelligence, Correlation) regardless of content length or screen size

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
