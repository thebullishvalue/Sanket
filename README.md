# SANKET — Institutional Market Signal Terminal
### Cross-Sectional Momentum Ranker · Obsidian Quant · Pragyam Family · `v5.0.0`

> **संकेत** *(Sanketa)* — Sanskrit for *Signal* · *Indicator* · *Forewarning*

Sanket is a quantitative market-screening terminal that ranks a universe of stocks by a
**single, reproducibly-validated edge**: **12-1 cross-sectional momentum** (long tilt). It produces
a daily ranked shortlist with a conviction read and an honest, live measurement of *whether its own
edge is currently working* — and it ships the harness ([`research.py`](research.py)) that
regenerates every claim below from live data.

Part of the **Pragyam Product Family** by [@thebullishvalue](https://github.com/thebullishvalue).

> **Read this first.** Sanket is **decision-support**, not a turnkey trading strategy. The edge is
> real but modest: ~**+6%/yr excess** over the universe at ~0.6 excess Sharpe — and *mostly the
> return is market beta, not skill*. It also **decays** (momentum was dormant 2024–2026, and the
> system says so itself). The terminal ranks and contextualizes; *you* decide, size, and execute.
> Signals are not financial advice.

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

Most screeners rank stocks by a stack of overlapping indicators. Sanket does **one** thing,
deliberately, and only because a cost-aware harness proved it survives: it ranks the cross-section
by **12-1 momentum** — 12-month return skipping the most recent month — and surfaces the strongest
trending names as longs, entered on short-horizon pullbacks. Reversion (its old core) is real but
net-negative after costs, so it was demoted to entry timing.

The core question Sanket answers: **which names carry the strongest relative trend right now, how
should I time entry, and how much should I trust that read given the edge's current health?**

---

## The Thesis (and the evidence)

**12-1 cross-sectional momentum (long tilt).** Names with the strongest 12-month return (skipping
the most recent month) tend to out-perform peers over the following weeks. It is the edge that
**survives realistic costs** — because momentum's predictive power *grows* with horizon, a monthly
book turns slowly and the gross edge clears fees.

Validated by the in-repo harness ([`research.py`](research.py)) on **100 NIFTY-100 names,
2016–2026 (~2,600 bars), corporate-action-adjusted** — reproduce it with `python research.py`:

| Check | Result |
|:---|:---|
| Momentum rank-IC vs fwd return | **+0.025 (5d) → +0.032 (21d) → +0.048 (63d)** — grows with horizon |
| Long-only top quintile, monthly, net 15 bps/side | **~+6%/yr excess** over the equal-weight universe |
| Net excess Sharpe · turnover | **~0.6 · ~21%** (cost-robust to 25 bps) |
| Shuffled-null control | IC ≈ 0 — the harness doesn't manufacture edge |

**Honest caveats, stated up front:** the ~30%/yr *absolute* is mostly market beta (absolute Sharpe
1.34 ≈ the benchmark's 1.29); the real skill is the ~+6% *excess*. And momentum **decayed 2024–2026**
(IC negative) — the alpha-health monitor exists precisely to stand the book down then.

What was **rejected / demoted** (reproducible on the same harness):

- **Reversion** (the old core): real predictor (IC +0.03, t ≈ 7) but **net-negative after cost**
  (≈ −23%/yr at 25 bps, ~80% turnover) — the edge lives at 1–2 days where costs are highest.
  **Demoted to an entry-timing overlay.**
- The legacy **WRCI / Conviction / Pulse / Intelligence** stacks: anti-predicted or added no
  out-of-sample edge. **Removed.**
- **Inferred order-flow** (delta / CVD / absorption) as a *ranking* factor: adds **zero**
  cross-sectional IC (delta is reconstructed from candle shape, not real tape). **Descriptive
  context only.**

Full detail, including the cost frontier, is in [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## The Engine

`engine.py` is the entire ranking core. It is **fixed and validated** — there are no weights to
calibrate, no Optuna, no per-symbol models. Every claim it makes is reproducible via `research.py`.

### 1. Momentum score (cross-sectional, per date)
Within-date **robust z-score** (median / MAD) of 12-1 momentum, oriented so higher = more attractive
long. 6-1 momentum is a coverage fallback for shorter histories:

```
mom   = Close[t-21] / Close[t-252] − 1        (12-month return, skip last month)
score = robust_z_within_date(mom)             (fallback to 6-1 where 12-1 is NaN)
```

No fitted weights: a risk-adjusted (mom/vol) variant and a 12+6 blend did **not** beat plain 12-1
on excess return out of sample.

### 2. Reversion entry overlay
`Entry_Timing` = within-date rank of `−z(ret2)` in `[0,1]` — high = the name has pulled back. It
nudges conviction (side-aware, ±10%) so a momentum long is preferred *on a dip*. It never enters the
rank — reversion is timing, not thesis.

### 3. Conviction
A single `[0, 1]` headline per name:

```
Conviction = tail_strength × alpha_health × regime_suitability × regime_confidence × entry_nudge
```

- **tail_strength** — distance from the cross-sectional median (0 at the middle, → 1 at the tails)
- **alpha_health** — the live edge multiplier (see below)
- **regime_suitability** — vol-regime weight (momentum damped in HIGH / EXTREME vol, where it crashes)
- **entry_nudge** — the `Entry_Timing` pullback bonus

### 4. Side
Top cross-sectional tail → **Long**, bottom tail → **Short** (underweight / F&O-only — NSE cash
can't short single names), the muddy middle → context-only. Side is assigned by **rank**, so the
shortlist is never empty — on a dormant day it simply carries low conviction.

---

## Alpha-Health Monitor

The feature that makes Sanket trustworthy. The system measures its **own realized edge in real
time**: the trailing ~60-day mean of the daily cross-sectional IC of the momentum score vs a 5-day
forward return, mapped to a conviction multiplier in `[0.35, 1]`.

- When momentum is **working**, the screen runs at full conviction; when it goes **dormant** —
  as it did in **2024–2026** — Sanket **stands down**: it still ranks the universe, but conviction
  shrinks toward the floor and the dashboard says so plainly. (On a live check in mid-2026 the
  monitor read trailing IC ≈ −0.005 and floored conviction at 0.35 — the feature working.)
- The significance haircut accounts for the forward-return overlap; **no p-value is claimed** — the
  multiplier is a smooth de-rating, not a hypothesis test.

A dormant factor produces a flat, low-conviction screen **by design**. The system will not scream
conviction into a dead regime.

---

## Outputs

Per name, on each run:

- `Rev_Score` — the cross-sectional **momentum** alpha score (retained column name; + = long-attractive)
- `Rev_Rank_Pct` — standing within today's universe (0–100)
- `Conviction` — the headline `[0, 1]` (= `Intel_Confidence` in the tables)
- `Side` — Long / Short (underweight, F&O-only) / — (context)
- `Entry_Timing` — `[0,1]` pullback score for entry timing
- `Meta_Score` / `Meta_Tier` — fused rank × conviction (0–3 tier)
- Risk context — `Vol_Regime`, `Regime_Confidence`, `Change_Point`, `ATR_Pct`
- Flow context (descriptive) — `Bar_Delta`, `CVD`, `Buy_Share`, `Absorption_Score`, `VA_Pos`

---

## Architecture Overview

```
sanket.py            ← Streamlit entry point: UI, data fetch, per-symbol features, screen routing
engine.py            ← THE ranking engine: momentum score + entry overlay + conviction + alpha-health
research.py          ← Reproducible point-in-time cost-aware harness (regenerates all evidence)
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

1. **Single Date Screener** — fetch the universe on a date, compute momentum + regime + flow
   context, measure live alpha-health, and return a ranked shortlist with conviction.
   Tabs: Action Dashboard · Signal Strength · **Alpha-Health Monitor** · System Data.
2. **Historical Range** — bulk time-series harvest used both to display history and to measure the
   trailing realized IC (the alpha-health reading). Exportable.
3. **Correlation Analysis** — cross-asset correlation + confluence, weighted by momentum rank.
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

## What Changed in v5.0.0

A **thesis replacement, driven by a new reproducible harness** — see [`CHANGELOG.md`](CHANGELOG.md)
for the full entry. In short: [`research.py`](research.py) (point-in-time, cost-aware) was built to
regenerate evidence on demand, and it showed the prior core — cross-sectional reversion — is a
**cost trap** (real IC, but net-negative after fees at its 1–2 day horizon). The same harness found
the edge that *survives* costs: **12-1 cross-sectional momentum**, long tilt, monthly. `engine.py`
was rebuilt around it (reversion demoted to an `Entry_Timing` overlay; `VOL_REGIME_MOM` damps
momentum in high vol; alpha-health retuned to momentum), and the data window was widened so 12-month
formation has runway. The output-column contract and UI identity are preserved; the copy was made
honest about beta-vs-alpha and the 2024–2026 decay.

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

*Sanket v5.0.0 · Pragyam Family · Built by [@thebullishvalue](https://github.com/thebullishvalue)*
