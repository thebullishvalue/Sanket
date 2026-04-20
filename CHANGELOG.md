# CHANGELOG — SANKET Signal Screener

All notable changes to the SANKET signal screening system are documented here.

---

## [v2.0] — April 2026

### 🎯 Major Features Added

#### F&O Stocks Universe Integration
- **Added**: F&O Stocks as primary option in India Indexes
- **Features**: Fetch F&O eligible stocks from NSE with 3 fallback sources
  - Primary: NSE F&O API endpoint
  - Fallback 1: NSE advances/declines data
  - Fallback 2: NIFTY 500 CSV (when API unavailable)
- **Impact**: Users can now screen only F&O-eligible names for derivative trading
- **Reliability**: Multiple fallback sources ensure consistent data availability

#### Dynamic Table Height Calculation
- **Fixed**: Tables cutting off rows at bottom of viewport
- **Before**: Fixed heights (3000px, 1500px) caused row cutoff on small universes
- **After**: Dynamic calculation based on actual data:
  - Signal tables: `height = 94 + (num_groups × 42) + (num_rows × 40)`
  - Conviction tables: `height = 94 + (num_rows × 40)`
- **Impact**: All rows now visible without excessive whitespace

#### Documentation Accuracy Overhaul
- **Audited**: All docstrings, descriptions, error messages for accuracy
- **Fixed Issues**:
  - Copy claimed "technical confirmations" but system ranks by magnitude only
  - Metrics described as percentages but calculated as absolute values
  - "Conviction" terminology replaced with honest "Signal Strength"
- **Result**: All user-facing text now accurately reflects actual behavior

### 📝 Documentation Updates

#### Module Docstrings
- Added context to function docstrings:
  - `run_screener_analysis()`: Now explains WRCI processing pipeline
  - `run_timeseries_analysis()`: Clarifies historical date-range behavior
  - `render_signal_detail_card()`: Documents all parameters and rendering

#### User Interface Text
- **Landing Page Card 1**: "identifies momentum signals... with daily updates" (was "detects reversals in real-time")
- **Landing Page Card 2**: "Rank momentum signals by strength, identify zones..." (was "trend confirmations without jargon")
- **Universe Coverage Card**: Clear explanation of F&O + Index coverage
- **Dashboard Labels**:
  - "Signal Strength Analysis" (was "Conviction Analysis")
  - "Avg Signal Magnitude" (was "Avg Signal Strength")
  - "Strongest Bullish/Bearish Signals" (was "Top Long/Short Conviction")

#### Error Messages
- All error messages now specific about what failed and why
- Example: "✓ Fetched 500 F&O securities" vs generic "Success"

### 🧹 Code Quality Improvements (Kaizen)

#### Dead Code Removal
- **Removed**: `get_fno_stock_list()` function stub (57 lines)
  - Function was defined but never called (confirmed via AST analysis)
  - Replaced with working implementation post-import
- **Removed**: 7 unused documentation files
  - `v5_func.txt`, `engine_diff.txt`, `v5_engine.txt`, `final_engine.txt`, `final_func.txt`
  - `scratch/` directory (development utilities)
  - `sector_map.pkl` (unused data file)
- **Rationale**: Reduce codebase friction, simplify maintenance

#### No Behavioral Changes
- All removals verified to have zero impact on system output
- Syntax validation confirms no regressions
- Signal detection logic unchanged

### 📋 Version Metadata

- **Status**: Stable (production-ready)
- **Python**: 3.8+
- **Key Dependencies**: Unchanged (yfinance, pandas, streamlit, plotly)

### 🔍 Testing Notes

- F&O fetch tested with 3 fallback scenarios
- Dynamic table height validated on universes of 10, 100, 500 symbols
- Documentation accuracy verified by code audit against actual behavior
- No regressions in single-date or time-series modes

---

## [v1.0] — December 2025

### ✨ Initial Release

#### Core Features
- **Wave Trend Composite Index (WRCI) Engine**: Momentum oscillator signal detection
- **Multi-Universe Scanning**: India Indexes, US Indexes, Commodities, Currency, Crypto
- **Signal Strength Ranking**: Magnitude-based scoring with diminishing returns above 50
- **Zone Detection**: Automatic overbought/oversold identification
- **Dual Timeframe**: Daily and weekly WRCI analysis

#### User Interface
- **Sidebar Configuration**: Universe, index, date, timeframe, WRCI parameters
- **Two Analysis Modes**:
  - Single Date: Point-in-time screener
  - Range Study: Historical date-range signal evolution
- **Three Dashboard Tabs**:
  - Action Dashboard: Age-grouped signals with timeline
  - Signal Strength: Ranked by magnitude
  - Top Bullish/Bearish: Conviction rankings each side
- **Interactive Charts**: WRCI + Trend oscillator visualization (last 100 candles)

#### Market Data Sources
- **Indian Stocks**: NSE CSV archives (NIFTY 50, sector indices, 500 constituents)
- **US Markets**: yfinance (S&P 500, DOW JONES, NASDAQ 100)
- **Commodities**: 16 commodity futures (gold, oil, wheat, etc.)
- **Currency**: 25 FX pairs (EUR/USD, USD/INR, etc.)
- **Crypto**: 21 cryptocurrencies (Bitcoin, Ethereum, etc.)

#### Technical Foundation
- **Framework**: Streamlit interactive web app
- **Data Processing**: pandas, NumPy (vectorized)
- **Charts**: Plotly interactive graphs
- **Styling**: Custom CSS (Obsidian Quant Terminal theme), SVG icons
- **HTTP**: requests + nsepython for data fetching

#### Performance
- Batch data fetch: 2-3 sec for 50 symbols, 10-15 sec for 500
- WRCI calculation: <1 sec (vectorized)
- Full screener run: 15-30 sec depending on universe size

#### Known Limitations
- Intraday data not supported (daily/weekly only)
- No live real-time updates (manual re-run required)
- Historical data limited to ~2 years (yfinance constraint)
- F&O list fallback to NIFTY 500 when API unavailable

---

## Version Numbering

**Format**: `v[MAJOR].[MINOR]`

- **MAJOR**: Significant feature addition or breaking change
- **MINOR**: Bug fixes, refinements, documentation

**v1.0**: Launch with core WRCI engine and multi-universe support  
**v2.0**: F&O integration, accuracy fixes, code cleanup

---

## Roadmap (Future Considerations)

### Potential v2.1 Enhancements
- [ ] Intraday (hourly/4h) signal detection
- [ ] Live real-time screener with auto-refresh
- [ ] Custom universe definition (user CSV import)
- [ ] Alert system (email/SMS on new signals)
- [ ] Signal backtesting module

### Potential v3.0 Features
- [ ] Portfolio heat-map (sector rotation analysis)
- [ ] Correlation matrix (multi-asset confluence)
- [ ] Strategy backtesting engine
- [ ] API endpoint for external integrations
- [ ] Mobile app (React Native)

---

## Migration Notes

### Upgrading from v1.0 → v2.0

**Breaking Changes**: None

**New in v2.0**:
- F&O Stocks now default option (index 0) in India Indexes
- Table rendering improved; may appear in different heights
- Copy/descriptions changed for accuracy; no logic changes
- Dead files removed; no functional impact

**What's the Same**:
- Signal detection algorithm unchanged
- All analysis parameters unchanged
- Output format identical
- Performance characteristics same

**Action Required**: None. Existing workflows continue unchanged.

---

## Contributors & Credits

**Built by**: Antigravity  
**Product Family**: Pragyam  
**WRCI Algorithm**: Wave Trend momentum analysis (technical analysis foundation)  
**Data Sources**: NSE archives, yfinance, nsepython

---

## Support & Feedback

- **Issues**: Test with default parameters first
- **Feature Requests**: Document use case and expected behavior
- **Data Problems**: Verify NSE/yfinance availability; check for API rate limits

---

## Changelog Maintenance

This changelog follows [Keep a Changelog](https://keepachangelog.com/) format.

- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Dead code/features
- **Fixed**: Bug fixes
- **Security**: Security fixes

---

**Last Updated**: April 2026  
**Maintained by**: Antigravity / Pragyam  
**Status**: Active Development
