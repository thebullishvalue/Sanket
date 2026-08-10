# SANKET — Institutional Market Signal Terminal
### SB v8 Close-Location Reversal · Obsidian Quant · Pragyam Family · `v6.0.0`

> **संकेत** *(Sanketa)* — Sanskrit for *Signal* · *Indicator* · *Forewarning*

Sanket is a quantitative market-screening terminal built on **one screening condition**: the
z-score of **where price closes inside its own bar range**. It fires two events — a **BUY** on a
weak close (green triangle) and a **SELL** on a strong close (yellow diamond) — ranks the whole
cross-section by that signal, and states the *measured out-of-sample expectancy for the asset class
you selected* on every screen.

The engine is a port of [`sb_v8.pine`](sb_v8.pine). That file's header is the primary evidence
document; [`ARCHITECTURE.md`](ARCHITECTURE.md) summarises it.

Part of the **Pragyam Product Family** by [@thebullishvalue](https://github.com/thebullishvalue).

> **Read this first.** Sanket is **decision-support**, not a turnkey strategy. Three things the
> system says about itself, plainly:
> 1. The **SELL side did not confirm out of sample** (holdout +0.0094, CI [−0.030, +0.052]). The
>    source indicator calls it CAUTION, not a short. Sanket surfaces it as a sell signal by
>    configuration and carries that caveat everywhere it appears.
> 2. **Scope is narrow.** Only US equity indices and US sectors are holdout-confirmed. On the other
>    six asset classes signals still fire — the measured expectancy behind them does not.
> 3. **A 0.13–0.43 net Sharpe is an overlay, not a system.** The edge is real, small, and decaying
>    (a quarter of its 1990s strength). There is **no intraday edge** here; none is claimed.
>
> Signals are not financial advice.

---

## Contents

- [What Sanket Does](#what-sanket-does)
- [The Signal (and the evidence)](#the-signal-and-the-evidence)
- [Why the event form](#why-the-event-form)
- [Scope — wired to your universe](#scope--wired-to-your-universe)
- [The Engine](#the-engine)
- [Outputs](#outputs)
- [Architecture Overview](#architecture-overview)
- [Analysis Modes](#analysis-modes)
- [Asset Universe Coverage](#asset-universe-coverage)
- [UI System — Obsidian Quant](#ui-system--obsidian-quant)
- [Installation & Launch](#installation--launch)
- [What Changed](#what-changed)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## What Sanket Does

Most screeners rank stocks by a stack of overlapping indicators. Sanket runs **one** condition,
across a universe, and tells you where it holds:

```
sb_clv = ((close - low) - (high - close)) / (high - low)     where the bar closed in its range
SB_Z   = z-score of sb_clv over the trailing 252 bars
```

The sign is the finding: **a strong close predicts WEAKNESS.** So the fade of a weak close is the
buy. `SB_Z < −1.5σ` fires **▲ BUY**; `SB_Z > +1.5σ` fires **◆ SELL**. Entry is the next session's
open; the measured horizon is 5–10 bars.

The core question Sanket answers: **which names just closed unusually weak or unusually strong
relative to their own history, how far past the trigger are they, and does the edge behind that
signal actually hold on this asset class?**

---

## The Signal (and the evidence)

Measured on **39 instruments, 251,200 daily bars, 1993–2026** — spanning dot-com, GFC, ZIRP, COVID
and the inflation cycle. 22 pre-declared features × 3 horizons = 66 tests, Bonferroni corrected,
block-bootstrapped by date. Discovery 1993–2013; holdout 2014–2026 sealed and opened exactly once.

| Check | Result |
|:---|:---|
| `sb_clv` discovery IC | **−0.0634**, z −7.96, p_bonf **1.2e-13** — strongest in the study by 4 orders of magnitude |
| Holdout (opened once) | h=1 IC −0.0251 (p 3.5e-03) · h=5 IC −0.0254 (p 4.6e-04) — **CONFIRMED** |
| Drift-free fade-**long** | discovery +0.0546 [+0.031,+0.078] · holdout +0.0534 [+0.028,+0.078] — **confirmed both eras** |
| Drift-free fade-**short** | discovery +0.0469 [+0.008,+0.085] · holdout +0.0094 [**−0.030,+0.052**] — **not confirmed** |

The drift-control test is the one that matters most: each instrument's own mean forward return is
removed within era, so "equities went up" cannot contribute.

**The edge is small and decaying.** IC by era: 1993–99 −0.113 → 2000–06 −0.065 → 2007–13 −0.056 →
holdout 2014–19 −0.025, 2020–26 −0.026. Four times weaker than the 1990s, now stable at roughly a
quarter of it.

What **died** on the same data: `sb_effort` (the literal SB core) at p_bonf = 1.00 at every horizon,
with its own 21d and 63d versions disagreeing on sign; `sb_breadth` likewise; all five momentum
variants; both MA-trend variants; overnight/intraday decomposition; volatility change; efficiency
ratio. Only `sb_clv` lived.

---

## Why the event form

This is the single most important design decision in the system.

Holding a **continuous** position on this signal turns over daily, costs 12%/yr at 3bp, and is
fatal: **net Sharpe −0.48**. Firing only on `|z| > 1.5` (9.3% of days) and holding ~10 days cuts
turnover **~35×**:

| Scope | Cost | Discovery | Holdout |
|:---|:---|:---|:---|
| All 39 instruments | 3bp | NET +0.124 | NET +0.132 |
| US equity indices + sectors | 3bp | — | NET **+0.430** |

Cost breakeven is ~7bp pooled; on US equity indices and sectors it stays positive past 10bp.
**Turnover, not signal strength, was the binding constraint.** The threshold is therefore not
decoration: `1.0` fires 44% of days and loses to costs, `2.0` fires 0.1% and failed holdout, `1.5`
fires 9.3% and is net-positive in both eras.

---

## Scope — wired to your universe

The indicator takes an *instrument class* input, because the edge does not hold everywhere. Sanket
derives it from your **universe selection**, so the expectancy shown always matches what is on
screen. This is honesty wiring, not a tunable parameter.

| Sanket universe | Instrument class | OOS edge | Hit | Established? |
|:---|:---|:---|:---|:---|
| US Indexes | US index / ETF | **+0.121** | 57.5% | **yes** |
| — | US sector ETF | **+0.068** | 54.9% | **yes** |
| India Indexes · ETF Index | India index | +0.089 | 51.9% | no — n=239, CI includes zero |
| Global Indexes | International equity | +0.003 | 53.2% | no |
| Commodities | Commodity | +0.028 | 51.0% | no |
| Currency | FX | +0.035 | 51.9% | no |
| Global Macro | Rates / Credit | −0.003 | 51.1% | no |
| Crypto | Other / unknown | 0.000 | 50.0% | no — study covered no digital assets |

*Established* = the block-bootstrap CI excluded zero after drift removal. The sidebar **Engine
Status** card states which case you are in; the Action Dashboard shows a **scope warning banner**
whenever the active class is not established. Believe them.

One caveat the mapping cannot resolve: the study measured *index- and ETF-level* instruments, so on
a constituent universe (individual stocks inside an index) the class edge is indicative of the asset
class, not a measurement on those names.

---

## The Engine

`engine.py` is the whole thing — ~350 lines, no fitted weights, no training step, no per-symbol
models.

### 1. Per-symbol signal — `add_sb_features(df, z_look, thr, horizon)`
```
SB_CLV       = ((C−L) − (H−C)) / (H−L)          in [−1, +1]
SB_Z         = (SB_CLV − SMA(SB_CLV, z_look)) / STDEV(SB_CLV, z_look)
Fade_Score   = −SB_Z                             positive = bullish
buy_cond     = SB_Z < −thr                       ▲ green triangle
sell_cond    = SB_Z > +thr                       ◆ yellow diamond
SB_Hold_Dir / SB_Hold_Age                        the hold window
SB_State     = WARMING UP / BUY / SELL / NEUTRAL
```
`STDEV` uses `ddof=0` — Pine's `ta.stdev` is the *population* standard deviation; the sample
version would drift the fire rate off the measured 9.3%. A symbol needs `z_look + 2` bars before
it can signal (the Pine's own warmup refusal); shorter histories are excluded with a "warming up"
count in the run stats.

**Weekly is an extrapolation.** The study was daily. `z_look` becomes 52 on Weekly (one year, the
closest structural analogue) and the Engine Status card labels it as extrapolated.

### 2. Cross-sectional ranking — `compute_ranking(df, iclass, cost_bps, thr)`
Ranks by `Fade_Score` (weakest closes first). `Side` is `Buy` / `Sell` / `—`, gated on ±`thr`:
**only a fired event is actionable.** Sub-threshold rows still appear in the ranking tables — the
score is continuous — but read `—`, because the measured edge is in the *event*.

```
Conviction = clip(0.30 + 0.70·clip(|z|/3, 0, 1)) × class_factor × cost_factor
class_factor: 1.00 established · 0.75 positive-but-CI-includes-zero · 0.55 nominally positive · 0.40 zero/negative
cost_factor:  1.00 within the class breakeven, else 0.50
```
Conviction is a **relative weighting, not a probability**, and is labelled that way in every
tooltip. Note what is deliberately absent: no per-name volatility factor, no regime factor, no
live-IC scaling. The Pine has no such terms and neither does this.

### 3. Bar convention — one deliberate difference from the Pine
The Pine reads `z[1]` so an *intraday* chart cannot repaint a daily signal. Sanket evaluates
completed bars directly, so the shift is unnecessary: a signal fires on the bar whose close
produced it, and entry is the next session's open — the same trade the Pine backtested. One
carry-over: **a signal on a session that has not closed yet is provisional until it does.**

### Everything else is context, and never a signal input
Inferred delta / CVD / `Delta_Z` / absorption / volume profile (OHLC proxies, validated three times
to add no cross-sectional edge), the flow zone, and the **regime engine** (HMM + GARCH + CUSUM,
per-name *risk context*). All displayed beside the signal, aggregated in the range charts, and
exported — none of it enters `SB_Z`, `Side`, or `Conviction`.

`Delta_Z` and `SB_Z` are cousins, not duplicates: `Delta_Z` z-scores the *volume-weighted* close
location, `SB_Z` the raw close location. Only the latter is the signal.

---

## Outputs

Per symbol, on each run:

| Column | Meaning |
|:---|:---|
| `SB_CLV` | close location in [−1, +1] |
| `SB_Z` | **the signal** — z-score of the close location |
| `Signal` / `Fade_Score` | `−SB_Z`; positive = bullish. What every table ranks on |
| `BUY_Today…BUY_5d` | ▲ green-triangle event, by age |
| `SELL_Today…SELL_5d` | ◆ yellow-diamond event, by age |
| `Side` | `Buy` / `Sell` / `—` (context only) |
| `Conviction` | `[0,1]` = \|z\| × class expectancy × cost gate |
| `SB_State` | WARMING UP / BUY / SELL / NEUTRAL |
| `SB_Hold_Dir` / `SB_Hold_Age` | hold-window direction and bars elapsed ("day 3/10") |
| `Signal_Reason` | plain-language read of the row, caveat included |
| Risk context | `Vol_Regime`, `Regime_Confidence`, `Change_Point`, `ATR_Pct` |
| Flow context | `Bar_Delta`, `CVD`, `Delta_Z`, `Buy_Share`, `Absorption_Score`, `VA_Pos` |

---

## Architecture Overview

```
sanket.py            ← Streamlit entry point: UI, data fetch, per-symbol features, screen routing
engine.py            ← THE signal engine: SB v8 close-location z + events + instrument class + conviction
sb_v8.pine           ← Source indicator and the primary evidence document (read its header)
research.py          ← LEGACY harness from the previous momentum engine; does not validate SB v8
logger.py            ← Structured terminal logging (ANSI color, phase timing, run IDs)
ARCHITECTURE.md      ← Signal, evidence, scope, and design rationale (read this)
ui/
  theme.py           ← CSS injection, Plotly Obsidian theme, progress cards
  theme.css          ← Full Obsidian Quant design system
  components.py      ← Reusable UI primitives (headers, metric cards, signal tables)
```

The **regime engine** (Hidden Markov + GARCH + CUSUM) lives in `sanket.py` and provides per-name
risk context only. The **order-flow layer** (inferred delta, CVD, volume profile, absorption) is
computed for display only. Neither enters the signal.

---

## Analysis Modes

1. **Single Date Screener** — fetch the universe on a date, compute the close-location z, and return
   the fired BUY / SELL signals bucketed by age plus the full ranking.
   Tabs: Action Dashboard · Signal Strength · System Data.
2. **Historical Range** — bulk harvest of the signal across a date range, with breadth charts,
   forward-return labels, and Excel export.
3. **Correlation Analysis** — cross-asset correlation + confluence, weighted by SB v8 signal
   strength and conviction.
4. **Pulse Narrative** — full-universe close-location state, ranked both ways.

---

## Asset Universe Coverage

| Universe Group | Constituents |
|:---|:---|
| **NSE F&O** | NSE F&O permitted stocks (dynamic; NIFTY-500 superset fallback) |
| **India Indices** | 28+ NIFTY indices: NIFTY 50/500, Bank, IT, Pharma, Midcap, sectoral |
| **US / Global Indices** | S&P 500, NASDAQ, DOW, international benchmarks |
| **ETF · Commodities · Currencies · Crypto · Global Macro** | Gold/Silver/Crude/Gas, FX majors, BTC/ETH, bond/macro ETFs |

**Data sources**: NSE India API (`nsepython` / `NseKit`), Yahoo Finance (`yfinance`), Wikipedia
(index constituent lists). SB v8 is a per-symbol signal, so it works on any instrument with ~254
bars of clean OHLC — but see the scope table above for *where the edge was actually measured*.

---

## UI System — Obsidian Quant

A fully custom Streamlit design layer — a precision-instrument aesthetic optimized for
quantitative data density. Signal colours now match the indicator's own markers so the app and a
TradingView chart read the same.

| Element | Specification |
|:---|:---|
| Background | `#1a1a1a` — dark obsidian |
| Accent — ▲ BUY (weak close) | `#00E676` |
| Accent — ◆ SELL (strong close) | `#FFA726` |
| Accent — neutral / sub-threshold | `#787B86` |
| Accent — amber (chrome) | `#D4A853` |
| Display / mono fonts | Syne · Space Grotesk / JetBrains Mono · IBM Plex Mono |

---

## Installation & Launch

```bash
git clone https://github.com/thebullishvalue/Sanket.git
cd Sanket
pip install -r requirements.txt
streamlit run sanket.py
```

Opens at `http://localhost:8501`. No calibration, profiles, training, or pre-screen measurement
pass — the signal is a rolling mean and standard deviation, so a run is a single pass over the
universe.

---

## What Changed

**v6.0.0 — one screening condition: SB v8 close-location reversal.** The system was refactored down
to a single signal. Removed: the 12-1 cross-sectional momentum ranker, the **Set A / Set B** entry
screeners, the **alpha-health monitor** (trailing-IC measurement, the Engine Status passport, and
the pre-screen harvest pass), and the whole **Intelligence** layer — the Intelligence tab, Layer-2
`Intel_Confidence`/`Intel_Stars`, Layer-3 `Meta_Score`/`Meta_Tier`, the Meta Filter, and the
Context/Entry signal-aging machinery. In their place: `SB_Z` (the z-score of the close location) as
the only condition, two events (▲ BUY on a weak close, ◆ SELL on a strong close), and conviction
gated on the instrument class's measured out-of-sample expectancy plus a cost gate. The universe
selector now drives the indicator's instrument-class input, so every screen states the expectancy
for the asset class in front of you. The regime engine and order-flow layer survive as displayed
context. Single progress pass per run; `research.py` is retained as a legacy harness that documents
the *previous* engine and does **not** validate this one.

**v5.1.0 — rebuilt entry screeners + data-calibrated intelligence.** Two long-only, edge-validated
screeners (Set A · Momentum Pullback-Resumption, Set B · Gap-and-Go Continuation) replaced the dead
delta-divergence/clamp-cross signals; `VOL_REGIME_MOM` was recalibrated to near-neutral. *(Both
retired in v6.0.0.)*

**v5.0.0 — thesis replacement driven by a reproducible harness.** [`research.py`](research.py)
showed the prior reversion core was a cost trap and found 12-1 cross-sectional momentum as the
cost-survivable edge. *(Retired in v6.0.0.)* See [`CHANGELOG.md`](CHANGELOG.md) for full entries.

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

*Sanket v5.1.0 · Pragyam Family · Built by [@thebullishvalue](https://github.com/thebullishvalue)*
