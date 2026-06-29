# SANKET — Institutional Market Signal Terminal
### Cross-Sectional Reversion Ranker · Obsidian Quant · Pragyam Family · `v4.0.0`

> **संकेत** *(Sanketa)* — Sanskrit for *Signal* · *Indicator* · *Forewarning*

Sanket is a quantitative market-screening terminal that ranks a universe of stocks by a
**single, validated edge**: short-horizon **cross-sectional mean-reversion**. It produces a daily
ranked long/short shortlist with a conviction read and an honest, live measurement of *whether
its own edge is currently working*.

Part of the **Pragyam Product Family** by [@thebullishvalue](https://github.com/thebullishvalue).

> **Read this first.** Sanket is **decision-support**, not a turnkey trading strategy. The edge it
> exploits is real but modest (validated rank-IC ≈ +0.03) and, after realistic transaction costs,
> only survives at multi-day holding periods. The terminal ranks and contextualizes; *you* decide,
> size, and execute. Signals are not financial advice.

---

## Contents

- [What Sanket Does](#what-sanket-does)
- [The Thesis (and the evidence)](#the-thesis-and-the-evidence)
- [The Engine](#the-engine)
- [Alpha-Health Monitor](#alpha-health-monitor)
- [Outputs](#outputs)
- [Architecture Overview](#architecture-overview)
- [Analysis Modes](#analysis-modes)
- [Asset Universe Coverage](#asset-universe-coverage)
- [UI System — Obsidian Quant](#ui-system--obsidian-quant)
- [Installation & Launch](#installation--launch)
- [What Changed in v4.0.0](#what-changed-in-v400)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## What Sanket Does

Most screeners rank stocks by momentum, RSI, or a stack of overlapping indicators. Sanket does
the opposite of momentum, and it does **one** thing, deliberately: it ranks the cross-section by
**how overextended each name is relative to its own volatility**, and surfaces the oversold tail
as longs and the overbought tail as shorts — because on this universe, that is what predicts the
next few days of *relative* returns.

The core question Sanket answers: **which names are most likely to revert toward their peers over
the next 1–5 days, and how much should I trust that read right now?**

---

## The Thesis (and the evidence)

**Short-horizon cross-sectional reversion.** Names that have moved most relative to their own ATR
tend to *under*-perform peers over the next 1–5 days; names that have sold off tend to
out-perform. This is a well-documented equity anomaly, and it is the dominant, direction-correct
edge in the NSE F&O cross-section.

Validated on real data (147 NSE F&O names, 5 years, ~170k symbol-days, fetched via the app's own
universe + yfinance):

| Check | Result |
|:---|:---|
| Per-feature reversion rank-IC (1–5d fwd) | **+0.025 … +0.031, t up to +8.6** |
| Walk-forward by year (2021–2025) | **Positive every year** (+0.025 … +0.049) |
| Regime dependence | **+0.055 in HIGH vol**, ~+0.025 LOW/NORMAL, noise in EXTREME |
| Combined composite, cross-sectional | **+0.028 … +0.031, t ≈ +8** |

What was **rejected** (and why), all on the same data:

- The legacy **WRCI / Conviction / Pulse** momentum factor stack: naked Priority IC = **−0.023**
  (t −3.9) — it ranked *backwards*. The Optuna calibrator could not flip it (non-negative weight
  bounds), so it shrank factors to noise. Removed.
- The **3-layer self-tuning Intelligence stack**: no out-of-sample ranking edge over naked
  Priority; fragile. Removed.
- **Inferred order-flow** (delta / CVD / divergence / absorption) as a *ranking* factor: adds
  **zero** cross-sectional IC (the delta is reconstructed from candle shape, not real tape).
  Demoted to descriptive context.

Full detail, including the cost analysis, is in [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## The Engine

`engine.py` is the entire ranking core. It is **fixed and validated** — there are no weights to
calibrate, no Optuna, no per-symbol models.

### 1. Reversion score (cross-sectional, per date)
An equal blend of within-date **robustly z-scored** (median / MAD) and **sign-flipped** reversion
features, oriented so a higher score = a more attractive long:

```
score = mean of  -z(ret2), -z(ret5), -z(dist5), -z(dist10), -z(rng_pos10)
```

- `retk` — k-bar return ÷ ATR14 (the recent move to fade)
- `distw` — (Close − SMAw) ÷ ATR14 (distance from short MA = overextension)
- `rng_pos10` — position within the 10-bar high/low range

Equal-weighted on purpose: the features are collinear and individually validated, and a fitted
weight vector did not beat the equal blend out of sample (it just invites overfit).

### 2. Conviction
A single `[0, 1]` headline per name:

```
Conviction = tail_strength × alpha_health × regime_suitability × regime_confidence
```

- **tail_strength** — distance from the cross-sectional median (0 at the middle, → 1 at the tails)
- **alpha_health** — the live edge multiplier (see below)
- **regime_suitability** — vol-regime weight (reversion best in HIGH vol, damped in EXTREME)

### 3. Side
Top cross-sectional tail → **Long**, bottom tail → **Short**, the muddy middle → context-only.
Side is assigned by **rank** (not conviction), so the shortlist is never empty — on a dormant day
it simply carries low conviction.

---

## Alpha-Health Monitor

The feature that makes Sanket trustworthy. The system measures its **own realized edge in real
time**: the trailing ~60-day mean of the daily cross-sectional IC of its score vs forward returns.

- On **healthy** days (~78% of history) forward IC ≈ **+0.036**; on **dormant** days ≈ +0.012
  (noise). The monitor maps the trailing IC to a conviction multiplier in `[0.35, 1]`.
- When the edge is **off** — as it was in 2026 — Sanket **stands down**: it still ranks the
  universe, but conviction shrinks toward the floor and the dashboard says so plainly.

A flat tape produces a flat, low-conviction screen **by design**. The system will not scream
conviction into a dead regime.

---

## Outputs

Per name, on each run:

- `Rev_Score` — the cross-sectional reversion score (+ = long-attractive)
- `Rev_Rank_Pct` — standing within today's universe (0–100)
- `Conviction` — the headline `[0, 1]` (= `Intel_Confidence` in the tables)
- `Side` — Long / Short / — (context)
- `Meta_Score` / `Meta_Tier` — fused rank × conviction (0–3 tier)
- Risk context — `Vol_Regime`, `Regime_Confidence`, `Change_Point`, `ATR_Pct`
- Flow context (descriptive) — `Bar_Delta`, `CVD`, `VA_Pos`, absorption flags

---

## Architecture Overview

```
sanket.py            ← Streamlit entry point: UI, data fetch, per-symbol features, screen routing
engine.py            ← THE ranking engine: reversion score + conviction + alpha-health monitor
breadth_engine.py    ← Market & sector advance/decline breadth (regime/risk context)
logger.py            ← Structured terminal logging (ANSI color, phase timing, run IDs)
ARCHITECTURE.md      ← Thesis, validation, and design rationale (read this)
ui/
  theme.py           ← CSS injection, Plotly Obsidian theme, progress cards
  theme.css          ← Full Obsidian Quant design system
  components.py      ← Reusable UI primitives (headers, metric cards, signal tables)
```

The **regime engine** (Hidden Markov + GARCH + CUSUM) lives in `sanket.py` and feeds the
conviction model and per-name risk context. It is order-flow-agnostic. The **order-flow layer**
(inferred delta, CVD, volume profile, absorption) is computed for display only — it never enters
the rank.

---

## Analysis Modes

1. **Single Date Screener** — fetch the universe on a date, compute reversion + regime + flow
   context, measure live alpha-health, and return a ranked long/short shortlist with conviction.
   Tabs: Action Dashboard · Signal Strength · **Alpha-Health Monitor** · System Data.
2. **Historical Range** — bulk time-series harvest used both to display history and to measure the
   trailing realized IC (the alpha-health reading). Exportable.
3. **Correlation Analysis** — cross-asset correlation + confluence, weighted by reversion rank.
4. **Pulse Narrative** — a per-name narrative/strength view over the same ranked screen.

---

## Asset Universe Coverage

| Universe Group | Constituents |
|:---|:---|
| **NSE F&O** | NSE F&O permitted stocks (dynamic; NIFTY-500 superset fallback) |
| **India Indices** | 28+ NIFTY indices: NIFTY 50/500, Bank, IT, Pharma, Midcap, sectoral |
| **US / Global Indices** | S&P 500, NASDAQ, international benchmarks |
| **ETF · Commodities · Currencies · Crypto · Global Macro** | Gold/Silver/Crude/Gas, FX majors, BTC/ETH, bond/macro ETFs |

**Data sources**: NSE India API (`nsepython` / `NseKit`), Yahoo Finance (`yfinance`), Wikipedia
(S&P 500 list). Cross-sectional reversion needs a real universe per date — best on the equity
universes (F&O, indices, US equities); thin/zero-volume instruments degrade gracefully.

---

## UI System — Obsidian Quant

A fully custom Streamlit design layer — a precision-instrument aesthetic optimized for
quantitative data density. Unchanged in v4.0.0 (the engine was rebuilt under it; the vehicle's
design language is preserved).

| Element | Specification |
|:---|:---|
| Background | `#1a1a1a` — dark obsidian |
| Accent — long / bullish | `#22c55e` |
| Accent — short / bearish | `#ef4444` |
| Accent — neutral | `#4a9eff` |
| Accent — amber | `#D4A853` |
| Display / mono fonts | Syne · Space Grotesk / JetBrains Mono · IBM Plex Mono |

---

## Installation & Launch

```bash
git clone https://github.com/thebullishvalue/Sanket.git
cd Sanket
pip install -r requirements.txt
streamlit run sanket.py
```

Opens at `http://localhost:8501`. No calibration, profiles, or training step — the engine is
fixed and validated; the alpha-health reading is measured inline on each run (the first run of the
day harvests a lookback window, later runs reuse it).

---

## What Changed in v4.0.0

A **complete scoring-engine replacement** — see [`CHANGELOG.md`](CHANGELOG.md) for the full entry.
In short: the WRCI/Pulse/Intelligence/order-flow ranking machinery was validated on real data,
found to anti-predict or add no edge, and **removed**; the system was rebuilt around the one
edge that survived walk-forward + cost testing (cross-sectional reversion), with a live
alpha-health monitor that scales conviction by realized edge. `priority_engine.py`,
`intelligence.py`, and the three `.pine` indicators were deleted; `optuna` / `numba` / `filelock`
dropped. The UI identity is preserved; copy was made honest.

---

## Tech Stack

| Layer | Technology |
|:---|:---|
| Language | Python 3.10+ |
| Web Framework | Streamlit 1.30+ |
| Numerical | NumPy 1.24+, Pandas 2.1+ |
| Charts | Plotly 5.18+ |
| Data | yfinance, nsepython / NseKit |
| Parsing / Excel | BeautifulSoup4, lxml, html5lib, openpyxl |
| Terminal | colorama |

---

## License

Proprietary — institutional usage only. Copyright © 2026
[@thebullishvalue](https://github.com/thebullishvalue). Signals produced by this system do not
constitute financial advice; the author accepts no liability for trading or investment losses.
See [`LICENSE`](LICENSE) for full terms.

---

*Sanket v4.0.0 · Pragyam Family · Built by [@thebullishvalue](https://github.com/thebullishvalue)*
