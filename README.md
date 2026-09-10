# SANKET — Institutional Market Signal Terminal
### Siddhi Conviction Oscillator · Obsidian Quant · Pragyam Family · `v7.0.0`

> **संकेत** *(Sanketa)* — Sanskrit for *Signal* · *Indicator* · *Forewarning*

Sanket is a quantitative market-screening terminal built on **one screening condition**: how much
of the market's **effort** actually converts into price **displacement**, measured against its own
signal line. It fires two events — a **BUY** where that conviction histogram crosses **above zero**
(green triangle) and a **SELL** where it crosses **below** (yellow diamond) — ranks the whole
cross-section by it, and states the *measured out-of-sample expectancy for the symbols you actually
put on screen* on every run.

The engine is the **Siddhi Conviction Oscillator**, ported from [`siddhi.pine`](siddhi.pine). Its
header is the primary source document; [`ARCHITECTURE.md`](ARCHITECTURE.md) summarises it.

Part of the **Pragyam Product Family** by [@thebullishvalue](https://github.com/thebullishvalue).

> **Read this first.** Sanket is **decision-support**, not a turnkey strategy. Three things the
> system says about itself, plainly:
> 1. **A bare zero-crossing is the source indicator's own weakest tested configuration.** It
>    measures the `k = 0` case at +0.0205R on the instruments it was fitted to and **+0.0015R,
>    t = 0.2**, on eight held-out ones. Sanket ships it because it is the condition asked for, and
>    marks the trigger `⚠ BARE` on the engine card so the caveat travels with it.
> 2. **Scope is earned per universe, never inherited.** Nothing about your symbols is hardcoded.
>    The built-in **Edge Study** measures expectancy on your own symbols on every run, and the
>    verdict is whatever *your* data supports — with the confidence interval, the effective
>    sample size, and the minimum detectable effect all on screen. That measurement, not any
>    number quoted from the source, is what applies to what you are looking at.
> 3. **Nothing in the source reaches statistical significance** once overlapping forward windows
>    are accounted for: its best case across 48 horizon/bracket cells was t = 1.9, and the
>    correlation between a configuration's fitted edge and its edge on unseen instruments is
>    approximately zero. Read every published number as a ranking, not a promise. There is **no
>    intraday claim** here.
>
> Signals are not financial advice.

---

## Contents

- [What Sanket Does](#what-sanket-does)
- [The Signal (and the evidence)](#the-signal-and-the-evidence)
- [Why the event form](#why-the-event-form)
- [Edge Study — expectancy measured on your universe](#edge-study--expectancy-measured-on-your-universe)
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
conviction     c = (close - close[1]) / TrueRange              bounded -1 … +1
participation  w = min(volume / EMA(volume, 20), 3.0)          true-range fallback, automatic
raw            = 100 · SMA(c·w, 20) / SMA(|c|·w, 20)           share of effort that went somewhere
osc            = EMA(100 · tanh(raw / 3σ), 3)                  adaptive self-normalisation
sig            = EMA(osc, 9)
SID_Hist       = osc - sig                                     THE SCREENING VARIABLE
```

`SID_Hist` crossing **above** zero fires **▲ BUY**; crossing **below** fires **◆ SELL**. Entry is
the next session's open; the declared hold is 10 bars.

Two properties make this different from a momentum oscillator. `c` is signed displacement divided
by **true range**, so gaps count and a wide, violent bar that closes where it opened scores **zero**
— nothing was accomplished. `w` weights each bar by how much of the market showed up for it, capped
so one expiry print cannot own the window. A rally into a higher high on heavy volume that closes
badly therefore adds a lot to the denominator and very little to the numerator: the oscillator
flattens while price rises. That is *effort without result*, and it is a statement about how a move
is being paid for rather than about how fast price moved.

The core question Sanket answers: **whose conviction just turned, how forcefully did it turn, and
does that event carry an edge on the universe in front of me?**

---

## The Signal (and what is actually claimed)

The condition is a **state change, not a level**. Nothing fires while the histogram merely sits on
one side of zero, and the two sides are symmetric — unlike the close-location engine this replaces,
where the sides meant different things and only one survived its own holdout.

What the source indicator measures about its own trigger, quoted at face value:

| Check | Result |
|:---|:---|
| Bare zero-crossing (`k = 0`, what ships here) | +0.0205R on the primary futures (t = 1.7) · **+0.0015R, t = 0.2** on eight held-out instruments |
| Fire rate at `k = 0` | ~113 per 1000 bars — roughly one every nine bars |
| Participation weighting | Earns its place: switching it **Off** is the worst available setting in all three signal architectures tested |
| Adaptive scaling calibration | Holds — measured occupancy beyond the outer zone is 3.2–4.3% against the 4% claimed |
| Any of it, corrected for overlap | **Nothing reaches significance.** Best case across 48 horizon/bracket cells: t = 1.9 |
| Fitted vs out-of-sample edge, across 900 configurations | Correlation ≈ **0** (−0.07 to +0.11) |

Read that table as a ranking, not a promise — which is exactly what its author says. The number
that applies to your screen is the one the **Edge Study** below measures on your symbols.

**Do not tune this to a backtest.** The optimiser's best-fitted settings lost 81% of their edge in
the same assets' later period and went negative on new ones. Every parameter Sanket ships is the
source indicator's own default, and none is adjustable in the UI for that reason.

---

## Why the event form

This is the single most important design decision in the system, and it survives the engine change
intact.

Two lines that both hug zero **cross constantly**. A continuous position on that separation turns
over every time they touch, and the turnover — not the signal — is what decides whether anything
is tradeable. Firing on the crossing and holding a declared horizon is what makes the rule costable
at all.

Cost is charged in the units the edge is measured in: `cost_bps / 1e4 / σ_h`. That is why the edge
dies on low-volatility instruments — 3bp against a 4% 10-day sigma costs 0.008 vol units, but
against a 1% sigma it costs 0.030. The source makes the same point in its own units: a round trip
costs about 0.02R on daily bars and 0.09R on 5-minute bars, against a best-case measured edge near
0.05R. **On 5-minute bars the cost is roughly double anything this construction has been shown to
produce.** Daily is the only timeframe with real headroom.

---

## Edge Study — expectancy measured on your universe

The edge does not hold everywhere — the source indicator's own results change sign between the
instruments it was calibrated on and the ones it held out. Earlier versions of this app hardcoded
a per-class expectancy table and applied it as a conviction multiplier. That was wrong on four
counts: it could not cover a universe the source never touched (NSE F&O single names, NSE
thematic ETFs, most of what this app screens), it applied an **asset-class** claim to
**instrument-level** decisions, it could not report that a component had stopped working, and you
could not check it against your own data.

So the app measures it. **`edge.py` runs an event study on your symbols**, at the pre-declared
parameters, with the methodology that makes the source numbers credible:

| # | Step | The failure it prevents |
|:--|:---|:---|
| 1 | Event study at the declared horizon (enter the bar after the signal, hold `h`) | Measuring the continuous form, whose turnover is set by how often two lines near zero touch — a question nobody trades |
| 2 | **Drift removal, within era** — subtract each symbol's own mean forward return | Every long signal in a bull market prints a profit; you'd have measured beta |
| 3 | Vol normalisation by the symbol's own σ | FX, bond ETFs and small-caps on incomparable scales |
| 4 | Sign folding, so both sides read "positive = right" | Reporting the two sides on opposite conventions |
| 5 | **Block bootstrap over dates** | Overlapping returns *and* a correlated cross-section both inflate significance |
| 6 | Cost charged in the same vol units (`bps/1e4 ÷ σ_h`) | Ignoring that the same bps costs 4× more on a low-vol instrument |
| 7 | **Power stated**: `n_eff`, and a minimum detectable effect from it | Reporting "no edge" from a test that could never have detected one |

The **confidence interval decides**, not a p-value hurdle. Verdicts:

| Verdict | Meaning |
|:---|:---|
| `CONFIRMED` | holdout CI excludes zero **and** survives costs |
| `GROSS ONLY` | holdout edge is real but costs consume it |
| `DISCOVERY ONLY` | discovery CI excludes zero, holdout does not |
| `NO EDGE` | CI straddles zero at adequate power |
| `ANTI-PREDICTS` | CI excludes zero on the wrong side |
| `UNDERPOWERED` | the MDE exceeds the largest effect ever measured for this signal — the test is vacuous, so no verdict is claimed |

That last row is the point: *"we could not detect an edge"* and *"there is no edge"* are
different statements, and conflating them is how underpowered studies get quoted as evidence
of absence.

### Two things the study refuses to do

- **It does not tune the signal.** Every oscillator parameter and the horizon stay pre-declared.
  Searching for the best lookback or magnitude gate per universe would fit noise and destroy the
  credibility the study exists to establish — the source measured that fitted-vs-out-of-sample
  correlation at approximately zero and said so.
- **It does not gate the signal.** The measurement is *reported*, never applied. Conviction is
  `crossing force × cost gate` with no expectancy term. A universe that measures no edge still
  fires at full conviction and says so — the alternative is a hidden multiplier you cannot audit.
  The cost gate is careful about this too: it keys off the measured *cost charge*, never the
  measured net, so a `NO EDGE` verdict cannot halve conviction through the back door.

### It runs on a 1 GB shared container

The study needs ~15 years of history (the power arithmetic: resolving an effect of `e` needs
`n_eff ≈ (1.96/e)²`, and `n_eff = (dates/horizon) × participation_ratio` — the screener's own
900-day window resolves only ~0.10, i.e. nothing but the single largest effect the source study
ever found). Fetching that naively for a large universe OOMs a Streamlit Community Cloud
container. Three choices avoid it:

1. **Lean** — the study computes the conviction oscillator and forward returns only; no volume
   profile, no regime engine, no order flow. It calls `engine.siddhi_oscillator` directly rather
   than re-deriving the rule, so the study can never drift from what the screener fires.
2. **Streaming** — symbols are fetched and reduced in chunks of 20, each chunk released before
   the next; what accumulates is event tuples at a ~11% fire rate.
3. **Sampled** — universes above 80 symbols are sampled with a fixed seed. Nearly free
   statistically, because the participation ratio saturates well below 80.

Measured `tracemalloc` peak on 80 symbols × 15 years: **31 MB**, and flat in universe size. The
naive alternative — holding the full analysed panel — measures 556 MB at 80 symbols (6.8 MB per
symbol), which projects to ~3.4 GB on NIFTY 500: a hard OOM.

### The reference prior

The source indicator's per-group numbers survive as a **labelled comparison row** — "the source
measured *Commodity* at +0.036 on gold, silver and crude, and −0.016 on eight held-out
instruments; here is what we measure on your universe." Nothing computes from them, and the app
states plainly that the source establishes **no** class. Two operative constants remain: a pooled
~7bp cost breakeven used *only* as the cost-gate fallback until a study exists (the UI reports
which basis it used, `measured` vs `pooled prior`), and `LARGEST_KNOWN_EFFECT = 0.036` — the most
this construction has ever been worth anywhere — used as the cost-gate ceiling and as the bar the
minimum detectable effect must clear before a verdict is claimed at all.

## The Engine

`engine.py` is the whole thing — no fitted weights, no training step, no per-symbol models.

### 1. Per-symbol signal — `add_siddhi_features(df, **settings)`
```
c            = (C − C[1]) / TrueRange              conviction, bounded [−1, +1]
w            = clip(V / EMA(V, 20), 0, 3.0)        participation; true-range fallback, automatic
SID_Raw      = 100 · SMA(c·w, 20) / SMA(|c|·w, 20) share of effort that became displacement
SID_Osc      = EMA(100 · tanh(SID_Raw / 3σ), 3)    adaptive self-normalisation, bounded ±100
SID_Sig      = EMA(SID_Osc, 9)
SID_Hist     = SID_Osc − SID_Sig                   THE SCREENING VARIABLE
SID_Hist_Z   = SID_Hist / σ(SID_Hist, 200)         in its own σ — comparable ACROSS symbols
SID_Impulse  = Δ SID_Hist / σ(SID_Hist, 200)       crossing force
buy_cond     = SID_Hist crosses ABOVE 0            ▲ green triangle
sell_cond    = SID_Hist crosses BELOW 0            ◆ yellow diamond
SID_Zone     = Extreme Bull / Bull / Neutral / Bear / Extreme Bear   (context, never a gate)
SID_State    = WARMING UP / DEGENERATE / BUY / SELL / NEUTRAL
```

Numerical fidelity to the Pine is deliberate: `ta.ema` is `ewm(span=n, adjust=False)`, `ta.stdev`
is the **population** standard deviation (`ddof=0`), `ta.tr(true)` includes the gap, and the
hollow-bar volume carry is reproduced so a holiday or thin overnight print cannot kill the
participation baseline for a whole averaging window.

**Warmup is additive, not a maximum** — `norm + length + vol_n + smooth + 2`, about **245 daily
bars** at the defaults. Taking a maximum would let the first adaptive σ be computed across
zero-filled bars, biasing it low and inflating the scaling exactly where the series begins.
Shorter histories are excluded with a "warming up" count in the run stats.

`SID_K` scales an optional magnitude gate — the histogram must cross `± k·σ(hist)` rather than
`± 0`. **It defaults to 0.0, which is exactly the zero-crossing above.** The knob exists so the
parameter can be *measured* by `edge.py` on a real universe rather than argued about.

### 2. Cross-sectional ranking — `compute_ranking(df, cost_bps, k, horizon, study)`

Scores on `SID_Hist_Z` and takes `Side` from whether the histogram **actually crossed on this
bar**. Ranking on the *level* while firing on the *crossing* is deliberate: the level says who is
currently in control, the crossing says when that changed, and the claim is only about the
crossing.

Priority is **banded**, and it has to be:

```
FIRED TODAY      2 + conviction          a crossing on this bar, strongest first
IN HOLD WINDOW   1 + remaining fraction  a crossing still inside its horizon
CONTEXT          tanh(SID_Hist_Z)        no crossing; just who is in control
```

A zero-crossing sits at **zero by construction**, so sorting the universe on the level alone would
bury every fresh signal in the middle of the list. The bands cannot overlap, so an actionable row
always outranks a merely bullish one — the same statement `Side` and `Signal_Reason` already make.

```
Conviction = clip(0.30 + 0.70·tanh(|SID_Impulse|)) × cost_factor
cost_factor: 1.00 if the cost gate passes, else 0.50
             — measured from the Edge Study when one exists, else the pooled ~7bp prior
```

Conviction is built from the **crossing force**, not the level, for the same reason: at the instant
a histogram crosses zero it *is* ~zero, so scaling conviction off `|hist|` would score every fresh
signal at nothing and every stale one high. What distinguishes crossings is how forcefully the gap
opened, and `SID_Impulse` is the only quantity available at fire time that separates them.

It remains a **relative weighting, not a probability**, and is labelled that way in every tooltip.
Note what is deliberately absent: no expectancy term (measured and *reported* by the Edge Study,
never folded into an unauditable number), no per-name volatility factor, no regime factor, no
live-IC scaling.

### 3. Bar convention — one deliberate difference from the Pine
The Pine gates every discrete object on `barstate.isconfirmed` so nothing is drawn on a forming bar
and then withdrawn. Sanket evaluates completed bars directly, so that gate is structural rather
than explicit: a signal fires on the bar whose close produced it, and entry is the next session's
open. One carry-over: **a signal on a session that has not closed yet is provisional until it
does.**

### Everything else is context, and never a signal input
Inferred delta / CVD / `Delta_Z` / absorption / volume profile (OHLC proxies, validated three times
to add no cross-sectional edge), the flow zone, and the **regime engine** (HMM + GARCH + CUSUM,
per-name *risk context*). All displayed beside the signal, aggregated in the range charts, and
exported — none of it enters `SID_Hist`, `Side`, or `Conviction`.

`Delta_Z` is a *close-location* proxy and is unrelated to the oscillator: it z-scores the
volume-weighted position of the close inside its bar, where Siddhi measures signed displacement
against true range. Only the latter is the signal.

---

## Outputs

Per symbol, on each run:

| Column | Meaning |
|:---|:---|
| `SID_Raw` | raw participation-weighted share of effort that became displacement |
| `SID_Osc` / `SID_Sig` | the oscillator (bounded ±100) and its signal line |
| `SID_Hist` | **the screening variable** — `SID_Osc − SID_Sig`. Its crossing of zero is the signal |
| `Signal` / `SID_Hist_Z` / `SID_Score` | the histogram in its own σ; what the universe is ranked on |
| `SID_Impulse` | crossing force — Δhistogram in σ. What separates one crossing from another |
| `SID_Zone` | where the oscillator sits vs the ±30 / ±60 zones. Context, never a gate |
| `BUY_Today…BUY_5d` | ▲ green-triangle event (crossed up), by age |
| `SELL_Today…SELL_5d` | ◆ yellow-diamond event (crossed down), by age |
| `Side` | `Buy` / `Sell` / `—` (no crossing on this bar — context only) |
| `Conviction` | `[0,1]` = `tanh(\|SID_Impulse\|)` × cost gate |
| `SID_State` | WARMING UP / DEGENERATE / BUY / SELL / NEUTRAL |
| `SID_Hold_Dir` / `SID_Hold_Age` | hold-window direction and bars elapsed ("day 3/10") |
| `Signal_Reason` | plain-language read of the row, caveat included |
| Risk context | `Vol_Regime`, `Regime_Confidence`, `Change_Point`, `ATR_Pct` |
| Flow context | `Bar_Delta`, `CVD`, `Delta_Z`, `Buy_Share`, `Absorption_Score`, `VA_Pos` |

---

## Architecture Overview

```
sanket.py            ← Streamlit entry point: UI, data fetch, per-symbol features, screen routing
engine.py            ← THE signal engine: conviction oscillator + zero-cross events + conviction
edge.py              ← Measured expectancy: event study, drift removal, block bootstrap, power
siddhi.pine          ← Source indicator and the primary source document (read its header)
research.py          ← LEGACY harness from an older momentum engine; does not validate Siddhi
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

1. **Single Date Screener** — fetch the universe on a date, build each symbol's conviction
   oscillator, and return the fired BUY / SELL crossings bucketed by age plus the full ranking.
   Tabs: Action Dashboard · Signal Strength · System Data (which carries the Edge Study readout).
2. **Historical Range** — bulk harvest of the signal across a date range, with breadth charts,
   forward-return labels, and Excel export.
3. **Correlation Analysis** — cross-asset correlation + confluence, weighted by Siddhi signal
   strength (fired crossing > open hold window > level) and conviction.
4. **Pulse Narrative** — full-universe conviction state, ranked both ways.

---

## Asset Universe Coverage

| Universe Group | Constituents |
|:---|:---|
| **NSE F&O** | NSE F&O permitted stocks (dynamic; NIFTY-500 superset fallback) |
| **India Indices** | 28+ NIFTY indices: NIFTY 50/500, Bank, IT, Pharma, Midcap, sectoral |
| **US / Global Indices** | S&P 500, NASDAQ, DOW, international benchmarks |
| **ETF · Commodities · Currencies · Crypto · Global Macro** | Gold/Silver/Crude/Gas, FX majors, BTC/ETH, bond/macro ETFs |

**Data sources**: NSE India API (`nsepython` / `NseKit`), Yahoo Finance (`yfinance`), Wikipedia
(index constituent lists). Siddhi is a per-symbol signal, so it fires on any instrument with
~245 bars of clean OHLC — volume is used where it exists and relative true range where it does
not, automatically, so index spot works without configuration. Whether it carries an *edge* on a
given universe is not assumed — run the Edge Study and read the verdict.

---

## UI System — Obsidian Quant

A fully custom Streamlit design layer — a precision-instrument aesthetic optimized for
quantitative data density. Signal colours now match the indicator's own markers so the app and a
TradingView chart read the same.

| Element | Specification |
|:---|:---|
| Background | `#1a1a1a` — dark obsidian |
| Accent — ▲ BUY (histogram crossed up) | `#00E676` |
| Accent — ◆ SELL (histogram crossed down) | `#FFA726` |
| Accent — neutral / no crossing | `#787B86` |
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

Opens at `http://localhost:8501`. There is nothing to configure and nothing to tick: the signal's
settings are fixed measured plateaus, and the **Edge Study runs on every run**. It fetches ~15
years the first time it sees a universe on a given day and reuses that measurement for the rest of
the day — within one calendar day the study reads identical data, so re-measuring would return a
bit-identical answer for a 15-year round trip. It re-measures automatically once the date rolls.

---

## What Changed

**v7.0.0 — one screening condition: the Siddhi conviction oscillator.** The close-location
reversal (CLR) engine was replaced wholesale. In its place, ported from
[`siddhi.pine`](siddhi.pine): a participation-weighted oscillator measuring how much of each bar's
effort converts into displacement (`c = ΔC / TrueRange`, weighted by capped relative volume, ratio
of sums over the lookback, rescaled through `100·tanh(raw / 3σ)`), its signal-line EMA, and the
**histogram between them**. The condition is that histogram's crossing of zero — **up is the BUY,
down is the SELL** — which makes the two sides symmetric for the first time.

Three consequences worked through the rest of the system rather than bolted on:

- **Priority had to be banded.** A crossing sits at zero by construction, so ranking the universe
  on the signal level alone would bury every fresh signal mid-list. Ranking is now
  `fired crossing > open hold window > histogram level`, per side.
- **Conviction had to change basis.** For the same reason, `|hist|` is useless at fire time.
  Conviction is now the **crossing force** — the one-bar change in the histogram in σ of its own
  distribution, through `tanh` — × the cost gate.
- **`edge.py` now calls the engine directly** instead of re-deriving the rule inline, so the
  measured study can never drift from what the screener fires.

Columns renamed `CLR_*` → `SID_*` throughout with the analysed-frame cache tag bumped to `sid1`,
which retires every frame the old engine cached. Warmup rose from 254 to ~245 bars but is now
*additive* across the whole chain (normalization + lookback + participation baseline + smoothing),
because taking a maximum would let the first adaptive σ be computed across zero-filled bars. The
regime engine and order-flow layer survive unchanged as displayed context. The trigger carries a
visible `⚠ BARE` mark: a bare zero-crossing is the source indicator's own weakest tested
configuration, and the app says so rather than burying it.

**v6.3.0 — the study runs on every run; nothing left to configure.** The Edge Study is no longer
opt-in behind an expander: expectancy on the universe in front of you is what tells you whether to
believe the signals, so it is measured as part of every run. Reuses a same-day measurement (within
one calendar day the inputs are identical, so the answer is too), re-measures when the date rolls,
and never blocks a run if it fails. The sidebar now has no controls at all. **Also fixed a real
leak**: the cost gate had begun reading the measured *net*, which failed the gate — and so halved
conviction — on any universe that measured no edge. That made the measurement a hidden multiplier,
the exact thing this design refuses. The gate now keys off the measured cost *charge* against the
largest effect this signal has ever shown anywhere.

**v6.2.0 — named for what it measures, and a quieter surface.** The engine is now
**Close-Location Reversal (CLR)** throughout — code, columns and UI — retiring the "SB v8"
family tag that belonged to a lineage this engine refutes. The parameter sliders were removed:
every setting is a measured plateau, so exposing them only invited fitting them to whatever
universe is on screen. The two verdict message boxes are gone; the Engine Status card was
consolidated from eleven rows to six and now repaints as soon as a study finishes, instead of
one interaction later.

**v6.1.0 — expectancy is measured, not hardcoded.** The eight-row per-class expectancy table is
gone from every operative path. `edge.py` now measures CLR's out-of-sample expectancy on the
user's own symbols: event study at the pre-declared parameters, each instrument's own drift
removed within era, vol-normalised, block-bootstrapped over dates, with the participation ratio,
effective sample size and minimum detectable effect all reported. Conviction dropped its
expectancy term entirely — the measurement is reported, never applied, so a `NO EDGE` verdict
does not suppress a single signal. Streaming + sampling keep the 15-year study inside 31 MB so it
runs on a 1 GB Streamlit Cloud container. The source study's numbers survive only as a labelled
comparison row.

**v6.0.0 — one screening condition: CLR close-location reversal.** The system was refactored down
to a single signal. Removed: the 12-1 cross-sectional momentum ranker, the **Set A / Set B** entry
screeners, the **alpha-health monitor** (trailing-IC measurement, the Engine Status passport, and
the pre-screen harvest pass), and the whole **Intelligence** layer — the Intelligence tab, Layer-2
`Intel_Confidence`/`Intel_Stars`, Layer-3 `Meta_Score`/`Meta_Tier`, the Meta Filter, and the
Context/Entry signal-aging machinery. In their place: `CLR_Z` (the z-score of the close location) as
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
