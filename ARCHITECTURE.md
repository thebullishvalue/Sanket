# Sanket — Engine Architecture & Research Basis

> This document records *why* the engine is built the way it is. Sanket runs **one screening
> condition**: the **Siddhi Conviction Oscillator**, ported from [`siddhi.pine`](siddhi.pine).
>
> Two kinds of number appear below, and the distinction matters. Numbers about **the source
> indicator** (eleven instruments across four timeframes) come from the Pine header — it is the
> primary source and this document summarises it; if they disagree, the Pine is right. Numbers
> about **your universe** are measured live by [`edge.py`](edge.py) and appear only in the app,
> never here — nothing about your symbols is hardcoded anywhere in this system.

## The signal

Two quantities per bar, multiplied, then summed over a window.

**Conviction** — how much of the bar's travel became net displacement:

```
c = (close - close[1]) / TrueRange              bounded -1 … +1
```

True range, so gaps are included and a gap-and-go scores as the strong bar it is. A bar that
opens at its low and closes at its high scores +1. A wide, violent bar that closes where it
started scores **0** — which is correct, because nothing was accomplished.

**Participation** — how much of the market showed up, relative to its own recent average:

```
w = min(volume / EMA(volume, 20), 3.0)
```

Capped, because expiry days, index rebalances and block prints produce volume spikes that would
otherwise dominate the window for its whole length. On symbols with no volume the identical
construction runs on **relative true range** instead, automatically, so index spot works without
configuration.

**The oscillator** is the participation-weighted share of effort that went somewhere:

```
raw  = 100 · SMA(c·w, 20) / SMA(|c|·w, 20)
osc  = EMA(100 · tanh(raw / 3σ(raw, 200)), 3)      adaptive self-normalisation, bounded ±100
sig  = EMA(osc, 9)
hist = osc - sig                                    THE SCREENING VARIABLE
```

### Why the rescaling exists

The raw share **cannot reach its own bounds**. ±100 would require nearly every bar in the window
to close at its extreme in the same direction; in practice the series lives inside roughly ±15.
Fixed thresholds against a range the series never visits are thresholds that never fire, and the
oscillator stays tangled with its own signal line near zero. Mapping through `100·tanh(raw / 3σ)`
is bounded, smooth, monotone and free of a clipping artifact, and it is calibrated so the ±30 zone
is occupied about a third of the time and the ±60 zone about 4%. Measured on real data the outer
zone runs 3.2–4.3%, so that calibration holds.

### Why divergence between effort and result means something here

Divergence on a momentum oscillator is frequently an artifact: RSI is a function of recent price
change, so when price decelerates the oscillator falls whether or not anything changed underneath.
The divergence is *arithmetic*.

Here the numerator and denominator move apart for a reason that is not. A rally into a higher high
on heavy volume and wide ranges that closes badly adds a **lot** to `Σ|c|·w` and very little to
`Σ c·w`. The oscillator flattens or falls while price rises. That is effort without result —
distribution in the Wyckoff sense — and it is a statement about how the advance is being *paid
for* rather than about how fast price moved.

### The two events (the only two signals in the system)

| Event | Condition | Status |
|:---|:---|:---|
| **▲ BUY** (green triangle) | `hist` crosses **above** zero | The oscillator has pulled above its own signal line: conviction is turning up |
| **◆ SELL** (yellow diamond) | `hist` crosses **below** zero | The oscillator has dropped under its signal line: conviction is turning down |

A **state change, not a level.** Nothing fires while the histogram merely sits on one side of
zero, and the two sides are symmetric — unlike the close-location engine this replaces, where the
sides meant different things and only one survived its own holdout.

`SID_K` scales an optional magnitude gate — the histogram must cross `± k·σ(hist)` rather than
`± 0`. **It defaults to 0.0, which is exactly the zero-crossing above.** The knob exists so the
parameter can be *measured* by `edge.py` rather than argued about; nothing in the shipped default
path uses a non-zero value.

## What the source actually measured

Quoted at face value, including the parts that do not flatter the shipped configuration.

| Check | Result |
|:---|:---|
| Bare zero-crossing (`k = 0`, what ships here) | +0.0205R on the primary futures (t = 1.7) · **+0.0015R, t = 0.2** on eight held-out instruments |
| Fire rate at `k = 0` | ~113 per 1000 bars — roughly one every nine bars |
| Participation weighting | Earns its place: switching it **Off** is the worst available setting in all three signal architectures tested |
| Adaptive scaling calibration | Holds — 3.2–4.3% occupancy beyond the outer zone against the 4% claimed |
| Regular divergence | The most consistent component; the only element positive on both the primary futures and the held-out instruments in nearly every configuration |
| Continuation trigger | +0.036R on gold/silver/crude, **−0.016R** on eight others — it changes sign off the primaries |
| Ungated reversal trigger | +0.0003R, **t = 0.01** — the loudest marker on the pane and empty |
| Hidden divergence | No edge on any universe |
| Any of it, corrected for overlapping forward windows | **Nothing reaches significance.** Best case across 48 horizon/bracket cells: t = 1.9 |

Method, for what those numbers mean: entry at the next bar's open, bracketed at two ATR of target
against one ATR of stop, twenty bars maximum, stop assumed to fill first when a bar spans both;
quoted as edge in R over a baseline applying the identical bracket to every bar in both directions,
so drift and bracket geometry are controlled for. Fitted on the first 60% of each series, tested on
the last 40%, re-tested on eight instruments never used in fitting. 26 years of daily data on GC,
SI, CL, HG, NG, ES, SPY, QQQ, TLT, 6E and BTC.

**Read every number above as a ranking, not a promise** — which is what its author says.

## Do not tune this to a backtest

900 randomised parameter sets were scored. The correlation between a configuration's fitted edge
and its edge on unseen instruments is **approximately zero** (−0.07 to +0.11 depending on
architecture). The optimiser's best-fitted settings lost **81%** of their edge in the same assets'
later period and went **negative** on new ones.

The two preferences that held across every architecture tested were a lookback at or above 20 and
an inner zone at or below 40 — both respected by the shipped defaults. Every other value Sanket
uses is the source indicator's own default, and none is adjustable in the UI for exactly this
reason.

## Why the EVENT form — the single most important design decision

Two lines that both hug zero **cross constantly**. A continuous position on that separation turns
over every time they touch, and the turnover — not the signal — decides whether anything is
tradeable at all. Firing on the crossing and holding a declared horizon is what makes the rule
costable.

Cost is charged in the units the edge is measured in: `cost_bps / 1e4 / σ_h`. That is why the edge
dies on low-volatility instruments — 3bp against a 4% 10-day sigma costs 0.008 vol units, but
against a 1% sigma it costs 0.030, a real drag on an edge of ~0.03. A per-class cost table cannot
express that; this can.

The source makes the same point in its own units: **a round trip costs about 0.02R on daily bars,
0.045R on hourly, 0.05R on 15m and 0.09R on 5m — against a best-case measured edge near 0.05R.**
On 5-minute bars the cost is roughly *double* anything this construction has been shown to produce.
Daily is the only timeframe with real headroom; crude at 1h and anything at 5m is cost-negative
before it is anything else.

## Scope — MEASURED on your universe, not inherited from a table

The source indicator's results change sign between the instruments it was calibrated on
(gold, silver, crude) and the eight it held out. Those are *its* eleven instruments. Earlier
versions of this app hardcoded a per-class expectancy table and applied it as a conviction
multiplier. That was indefensible:

- it could not cover a universe the source never touched (NSE F&O single names, NSE thematic
  ETFs, most of what this app screens);
- it made an **asset-class** claim and applied it to **instrument-level** decisions;
- it could not report that a component had stopped working — while the source's own headline is
  that *nothing it measured reaches significance*;
- and it was unfalsifiable in-product: you could not check it against your own data.

So the app measures it. [`edge.py`](edge.py) runs an event study on **your symbols**, at the
**pre-declared** parameters, using the methodology that makes the source numbers credible in
the first place. Seven steps, each of which kills one specific way of fooling yourself:

| # | Step | The failure it prevents |
|:--|:---|:---|
| 1 | **Event study at the declared horizon** — enter the bar after the signal closes, hold `horizon` (EXEC-B) | Measuring the continuous form, whose turnover is set by how often two lines near zero touch — a question nobody trades |
| 2 | **Drift removal, within era** — subtract each symbol's own mean forward return | Every long signal on an equity universe in a bull market prints a profit; you'd have measured beta |
| 3 | **Vol normalisation** — divide by the symbol's own forward σ | FX, bond ETFs and small-caps landing on incomparable scales |
| 4 | **Sign folding** — a buy scores + when it beat drift, a sell scores + when it fell short | Reporting the two sides on opposite conventions |
| 5 | **Block bootstrap over DATES** | h-bar returns overlap *and* one date's cross-section shares the market factor — both inflate significance. Blocks fix the first, whole dates the second |
| 6 | **Cost charged in the same units** — `cost_bps/1e4 ÷ σ_h` | Ignoring that 3bp against a 4% 10-day σ costs 0.008 vol units but against a 1% σ costs 0.030 |
| 7 | **Power stated** — `n_eff = (dates/horizon) × participation_ratio`, and an MDE from it | Reporting "no edge" from a test that could never have detected one |

The **confidence interval decides** — not a p-value hurdle. `CONFIRMED` means the holdout CI
excluded zero *and* the net survived costs.

### Two things the study deliberately refuses to do

- **It does not tune the signal.** Threshold and horizon stay pre-declared. Searching for the
  best threshold per universe, on a few hundred independent blocks, would fit noise and
  destroy the credibility the study exists to establish. It measures a fixed rule.
- **It does not gate the signal.** The measurement is *reported*, never applied. Conviction is
  `|z| × cost gate` and contains no expectancy term. A universe that measures no edge still
  fires at full conviction and says so — because the alternative is a hidden multiplier the
  reader cannot audit.

  The cost gate has to be careful about the same thing. It asks whether this universe's measured
  trading *cost charge* (`cost_bps/1e4 ÷ σ_h`) exceeds `LARGEST_KNOWN_EFFECT` — the most this
  signal has ever been worth on any asset class — and never compares the cost against the
  *measured* edge. Comparing against the measured edge would fail the gate on every no-edge
  universe, halving its conviction, and smuggle the expectancy back into the signal.

### Power is the binding constraint, and it is arithmetic

Vol-normalised scores have σ ≈ 1, so the CI half-width is ≈ `1.96/√n_eff`:

| Effect to resolve | `n_eff` needed |
|:---|:---|
| 0.12 (the largest the source study found) | ~270 |
| 0.07 | ~780 |
| 0.05 | ~1,540 |

And `n_eff = (n_dates / horizon) × participation_ratio`, where the participation ratio is the
eigenvalue-based effective number of independent names — `(Σλ)²/Σλ²` of the correlation
matrix, **measured, not assumed**. This is why a 500-name NSE universe does not carry 500
observations per date; the source study makes the same point (26 symbols → 7.2 independent).

| History | Usable dates | Blocks (h=10) | `n_eff` at PR≈10 | Resolves |
|:---|:---|:---|:---|:---|
| The screener's own 900-day pool | ~370 | 37 | ~370 | 0.10 — nothing but the largest effect |
| **~15 years (what the study fetches)** | ~3,530 | 353 | ~3,530 | **~0.033** |

Hence a **separate, deeper fetch** for the study, distinct from the screening fetch.

### Why it fits on Streamlit Community Cloud (~1 GB, shared vCPU)

The naive implementation — fetch 15 years for the whole universe, run the analysis pipeline,
hold the panel — is several hundred MB and OOMs. Three choices avoid it:

1. **Lean.** The study computes the conviction oscillator and forward returns *only*. No volume
   profile (a Python double loop, the app's slowest path), no regime engine, no order flow. It
   calls `engine.siddhi_oscillator` **directly** rather than re-deriving the rule inline, so
   every guard in the engine applies to the study by construction — the previous version
   re-derived it, which meant a guard added in one place had to be mirrored in the other or the
   study would silently measure a rule the screener does not fire.
2. **Streaming.** Symbols are fetched and reduced in chunks of 20; each chunk's frames are
   released before the next is fetched. What accumulates is event tuples at a ~11% fire rate.
3. **Sampled.** Universes above 80 symbols are sampled with a fixed seed (so the answer is
   reproducible, and not biased toward one alphabetical/sector slice). This costs almost
   nothing statistically, because the participation ratio saturates far below 80.

Measured, not asserted — `tracemalloc` peak on 80 symbols × 15 years:

| Approach | Peak | Projected for NIFTY 500 |
|:---|:---|:---|
| **Streaming + lean (shipped)** | **31 MB** | **~31 MB — flat in universe size** |
| Raw OHLCV frames held for the universe | 14 MB | ~90 MB |
| Full analysed panel (naive) | 556 MB (6.8 MB/symbol) | **~3,400 MB — hard OOM** |

Streaming being *flat in universe size* is the property that matters; the naive panel scales at
6.8 MB per symbol and cannot reach the universes this app supports.

### The reference prior

The source indicator's per-group numbers survive in `engine.CLASS_EDGE` / `CLASS_HIT` purely as
a **labelled comparison row**: "the source measured *Commodity* at +0.036 on gold, silver and
crude, and −0.016 on eight held-out instruments; here is what we measure on your universe."
`compute_ranking` does not read them, `instrument_class` exists only to choose which row to
display, and `ESTABLISHED_CLASSES` is deliberately **empty** — the source establishes none, and
the app says so rather than staying silent about it.

One operative constant remains: `POOLED_BREAKEVEN_BPS = 7.0`. Until a study exists there is
nothing to compare a cost against, so the cost gate falls back to the study's pooled
breakeven and **labels itself as doing so** (`engine.cost_basis` returns `measured` or
`pooled prior (~7bp)`). Once a study exists, the gate uses its measured net.

### Cadence

The study runs on **every run**, not on request. It reuses a same-day measurement: within one
calendar day it reads the same completed bars (it needs forward returns, so it excludes the
forming bar) and must return a bit-identical answer, making a re-measurement a 15-year fetch for a
result already held. It re-measures automatically once the date rolls — exactly when new bars can
change the answer. A failure is recorded for the day rather than retried on every click, and the
run proceeds on the last measurement or on "not measured".

### Verdict ladder

| Verdict | Meaning |
|:---|:---|
| `CONFIRMED` | holdout CI excludes zero **and** survives costs |
| `GROSS ONLY` | holdout edge is real but costs consume it |
| `DISCOVERY ONLY` | discovery CI excludes zero, holdout does not |
| `NO EDGE` | CI straddles zero at adequate power |
| `ANTI-PREDICTS` | CI excludes zero on the wrong side |
| `UNDERPOWERED` | MDE exceeds the largest effect ever measured for this signal — the test is vacuous, so no verdict is claimed |

`UNDERPOWERED` being distinct from `NO EDGE` is the point. "We could not detect an edge" and
"there is no edge" are different statements, and conflating them is how underpowered studies
get quoted as evidence of absence.

## How to trade it — measured, not asserted

- **Horizon** 10 bars — the lookback *is* the horizon, so a 20-bar window is a swing instrument
  on daily bars and a scalping one on 5-minute bars. There is **no intraday claim** here, and the
  cost arithmetic above says why there could not be.
- **Entry** the next session's open after the signal bar closes (EXEC-B), because entering at the
  signal close is not available to most traders and tests barely different anyway.
- **BUY** the histogram crossed above zero → conviction turned up.
- **SELL** the histogram crossed below zero → conviction turned down.
- **Read the level as context, never as the signal.** Above zero says participation-weighted
  effort is net upward; a zone reading says how one-sided the window is. Both are the *setting*
  for a signal, not a signal. Only the crossing is the event.

## The engine — [`engine.py`](engine.py)

### 1. Per-symbol signal · `add_siddhi_features(df, **settings)`
```
c            = (C − C[1]) / TrueRange              conviction, bounded [−1, +1]
w            = clip(V / EMA(V, 20), 0, 3.0)        participation; true-range fallback, automatic
SID_Raw      = 100 · SMA(c·w, 20) / SMA(|c|·w, 20)
SID_Osc      = EMA(100 · tanh(SID_Raw / 3σ), 3)    bounded ±100 under adaptive scaling
SID_Sig      = EMA(SID_Osc, 9)
SID_Hist     = SID_Osc − SID_Sig                   THE SCREENING VARIABLE
SID_Hist_Z   = SID_Hist / σ(SID_Hist, 200)         in its own σ — comparable ACROSS symbols
SID_Impulse  = Δ SID_Hist / σ(SID_Hist, 200)       crossing force
buy_cond     = SID_Hist crosses ABOVE 0            green triangle
sell_cond    = SID_Hist crosses BELOW 0            yellow diamond
SID_Hold_Dir / SID_Hold_Age                        the hold window
SID_Zone     = Extreme Bull / Bull / Neutral / Bear / Extreme Bear   (context, never a gate)
SID_State    = WARMING UP / DEGENERATE / BUY / SELL / NEUTRAL
```

**Numerical fidelity to the Pine is deliberate.** `ta.ema` seeds on its first value with
`alpha = 2/(n+1)`, which is `ewm(span=n, adjust=False)`. `ta.stdev` is the **population** standard
deviation, so `ddof=0` throughout. `ta.tr(true)` includes the gap and falls back to `high−low` on
the first bar. And the hollow-bar volume carry is reproduced: a holiday, half session or thin
overnight print would otherwise feed NaN into the participation EMA and kill it for a whole
averaging window, silently dropping the series back to range weighting — measured in the source,
volume weighting was live on only 73% of bars before that fix, and as low as 31% on 1h futures.

**Warmup is two nested normalizations, and it is counted rather than guessed**:

```
raw     pinned until its SMA window and the participation baseline fill   length + vol_n
rawSd   a stdev OF raw, so it needs a window free of those pinned bars    + norm
hist    inherits that, through smooth and signal
histSd  a stdev of the HISTOGRAM, needs its own full clean window         + norm + smooth + signal
```

which closes to `2·norm + length + vol_n + smooth + signal` — **452 daily bars** at the defaults.

The previous arithmetic covered the first stage only, omitted `signal`, and declared a symbol
ready at bar 245 while `histSd` was not clean until roughly 440. **That deflation matters more
here than it does on the chart.** The Pine spends `histSd` only on the impulse threshold, which
at `k = 0` is zero either way; Sanket divides by it *twice* — for `SID_Hist_Z`, the
cross-sectional ranking score, and for `SID_Impulse`, the conviction basis — so both were
inflated across the whole early region, on every symbol short enough to live there.

**Weekly runs a shorter normalization window.** `SID_NORM_WEEKLY = 60` gives a 172-bar warmup and
~14 months of context, the closest wall-clock analogue to what 200 daily bars gives the daily
screen, and it stays well above the source's own minimum of 30. At 200 a weekly symbol would need
452 *weekly* bars — 8.7 years each — before the screen showed anything. It is Sanket's number, not
the source's, and the engine card says so (`ADAPTED`) rather than presenting it as a measured
plateau. Weekly also fetches 1900 calendar days rather than 900; both changes are needed, neither
is sufficient alone. Shorter histories are excluded with a "warming up" count in the run stats.

`SID_Hist_Z` exists because the raw histogram is **not comparable across symbols** — dividing by
its own σ over the normalization window is what makes one instrument's reading rankable against
another's.

### 2. Cross-sectional ranking · `compute_ranking(df, cost_bps, k, horizon, study)`

Scores on `SID_Hist_Z`; `Side` comes from whether the histogram **actually crossed on this bar**.
Ranking on the *level* while firing on the *crossing* is deliberate: the level says who is
currently in control, the crossing says when that changed, and the claim is only about the
crossing. Rows that did not cross still appear in the tables — the score is continuous — but their
Side reads `—`, however extreme their level.

**Priority is banded, and it has to be:**

```
FIRED TODAY      2 + conviction          a crossing on this bar, strongest first
IN HOLD WINDOW   1 + remaining fraction  a crossing still inside its horizon
CONTEXT          tanh(SID_Hist_Z)        no crossing; just who is in control
```

The previous engine could sort the universe on its raw score because there the extreme readings
*were* the fired signals — a buy was the most negative z on the board. A zero-crossing is the
opposite: at the instant it fires the histogram is, by construction, ~0, so sorting on the level
would bury every fresh signal in the middle of the list. The bands cannot overlap, so an
actionable row always outranks a merely bullish one — which is the same statement `Side` and
`Signal_Reason` already make.

```
Conviction = clip(0.30 + 0.70·tanh(|SID_Impulse|)) × cost_factor
cost_factor: 1.00 if the cost gate passes, else 0.50
             — measured from `study` when one exists, else the pooled ~7bp prior
```

**Conviction is built from the crossing force, not the level**, for the same reason. Scaling it
off `|hist|` would score every fresh signal at ~0 and every stale one high, which is backwards.
What distinguishes crossings is how forcefully the gap opened, and the one-bar change in the
histogram — in σ of its own distribution, so one sigma is the natural unit and the transform needs
no fitted constant — is the only quantity available at fire time that separates them.

It is a *relative weighting*, not a probability, and it is labelled that way in every tooltip.
Note what is deliberately **absent**: no expectancy term (that is measured by `edge.py` and
*reported*, never folded into a number the reader cannot audit), no per-name volatility factor, no
regime factor, no live-IC scaling. Nothing has established that a larger crossing step predicts a
better outcome; it does not gate or scale any ranking beyond its own band.

### 3. Bar convention — one deliberate difference from the Pine
The Pine reads `z[1]` inside `request.security(..., "D", ...)` so an *intraday* chart cannot
repaint a daily signal. Sanket evaluates completed daily (or weekly) bars directly, so that shift
is unnecessary: a signal fires on the bar whose close produced it, and entry is the next session's
open — the same trade the Pine backtested. The one carry-over: **a signal on a session that has
not closed yet is provisional until it does.**

## What is context, and never a signal input

Everything else the app computes is descriptive. It is displayed beside the signal, aggregated in
the range-mode charts, and exported — but it does not enter `SID_Hist`, `Side`, or
`Conviction`:

- **Order flow** — inferred bar delta, CVD and its slope, `Delta_Z`, absorption, rolling buy
  share, volume profile (POC/VAH/VAL, `VA_Pos`), RVOL. OHLC proxies, validated three times to add
  no cross-sectional ranking edge.
- **Flow zone** (`Condition`) — where cumulative delta sits vs its 20-bar mean:
  Accumulation(+) / Distribution(+) / Neutral. Consumed by the Correlation setup classifier and
  the range-mode breadth charts.
- **Regime** — HMM bull/bear, GARCH vol regime, CUSUM change points. Per-name **risk context**.
  It informs the Regime / Vol columns and the range-mode Regime tab.
- **Forward returns** (`Ret_1b/5b/10b/21b`, Historical Range only) — evaluation **labels**.

`Delta_Z` is a *close-location* proxy and is unrelated to the oscillator: it z-scores the
volume-weighted position of the close inside its bar, where Siddhi measures signed displacement
against true range. Only the latter is the signal.

## Outputs (per symbol)

| Column | Meaning |
|:---|:---|
| `SID_Raw` | raw participation-weighted share of effort that became displacement |
| `SID_Osc` / `SID_Sig` | the oscillator (bounded ±100) and its signal line |
| `SID_Hist` | **the screening variable** — `SID_Osc − SID_Sig`; its crossing of zero is the signal |
| `Signal` / `SID_Hist_Z` / `SID_Score` | the histogram in its own σ. What the universe is ranked on |
| `SID_Impulse` | crossing force — Δhistogram in σ. What separates one crossing from another |
| `SID_Zone` | oscillator position vs the ±30 / ±60 zones. Context, never a gate |
| `buy_cond` / `BUY_Today…BUY_5d` | green-triangle event and its age |
| `sell_cond` / `SELL_Today…SELL_5d` | yellow-diamond event and its age |
| `Side` | `Buy` / `Sell` / `—` (context only) |
| `Conviction` | `[0,1]` = `tanh(\|SID_Impulse\|)` × cost gate. No expectancy term |
| `SID_State` | WARMING UP / DEGENERATE / BUY / SELL / NEUTRAL |
| `SID_Hold_Dir` / `SID_Hold_Age` | hold-window direction and bars elapsed |
| `SID_Rank_Pct`, `Priority_Long/Short(_pct)` | cross-sectional ordering keys (priority is banded) |
| `Signal_Reason` | plain-language read of the row, including the measured verdict for this universe |

## Known limitations (disclosed, not hidden)

- **The sell side was not confirmed out of sample** in the source study. It ships as a sell
  signal by configuration; the statistics say caution. The Edge Study measures both sides
  separately on your universe, so you can check whether that asymmetry reproduces on your data.
- **Scope must be earned per universe.** Nothing is inherited: until you run the Edge Study the
  app reports "not measured", and after you run it the verdict is whatever your symbols support.
  A `NO EDGE` verdict does not suppress signals — it is a measurement, not a filter.
- **One split, not a walk-forward.** The discovery/holdout split is by date, opened once. It is
  reported as one split and does not pretend to be more.
- **The study samples large universes** (80-symbol cap, fixed seed). Defensible because the
  participation ratio saturates, but it is a sample, and the app says so.
- **Nothing the source measured is significant.** Best case t = 1.9 across 48 horizon/bracket
  cells, and the bare zero-crossing that ships here is its *weakest* tested configuration
  (+0.0015R, t = 0.2, on held-out instruments). The app marks the trigger `⚠ BARE` rather than
  burying that.
- **The parameters are not transferable evidence.** Fitted and out-of-sample edge are
  approximately uncorrelated across 900 configurations, which is why nothing here is tunable.
- **This is an overlay, not a system.** Position sizing, risk limits and portfolio construction
  are all outside this tool.
- **Adaptive scaling is relative, not absolute.** A reading of +60 means conviction is extreme
  *for this instrument on this timeframe over the normalization window*. In a dead range the
  scaling will amplify small absolute imbalances into large readings — which is why `SID_Raw` is
  carried alongside, and worth a glance when the pane looks dramatic and the chart does not.
- **Costs decide everything.** Past the class breakeven the event form is net negative, and the
  cost gate halves conviction to say so — but it cannot make the trade profitable.
- **A live session is provisional.** Today's row can change until the close.
- **[`research.py`](research.py) is a legacy harness.** It documents the cross-sectional momentum
  study an older engine was built on; it does not validate Siddhi. For the source's own
  measurements read the [`siddhi.pine`](siddhi.pine) header; for evidence about *your*
  universe, run the Edge Study. Trust the app's measurement over this document, this document
  over the Pine header for how the app behaves, and neither over `research.py`.
