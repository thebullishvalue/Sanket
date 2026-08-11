# Sanket — Engine Architecture & Research Basis

> This document records *why* the engine is built the way it is. Sanket runs **one screening
> condition**: **Close-Location Reversal (CLR)**, ported from [`sb_v8.pine`](sb_v8.pine) —
> whose title's "CLR" half was a family tag for session-breadth indicators whose premise this
> engine refutes, so only the descriptive half carries over.
>
> Two kinds of number appear below, and the distinction matters. Numbers about **the source
> study** (39 instruments, 1993–2026) come from the Pine header — it is the primary source and
> this document summarises it; if they disagree, the Pine is right. Numbers about **your
> universe** are measured live by [`edge.py`](edge.py) and appear only in the app, never here —
> nothing about your symbols is hardcoded anywhere in this system.

## The signal

One measurement, on one variable:

```
sb_clv = ((close - low) - (high - close)) / (high - low)
```

**Where price closes inside its own daily range.** −1 = closed on the low, +1 = on the high.
Its z-score over a trailing window (252 daily bars) is the entire engine.

The sign is the finding: **a strong close predicts WEAKNESS.** IC −0.0634, z −7.96,
p_bonf 1.2e-13 — the strongest result in the study by four orders of magnitude, and negative.
So the fade of a *weak* close is the buy.

### The two events (the only two signals in the system)

| Event | Condition | Status |
|:---|:---|:---|
| **▲ BUY** (green triangle) | `z < −1.5σ` — a weak close | **Holdout-confirmed in both eras.** Drift-free discovery +0.0546 [+0.031,+0.078], holdout +0.0534 [+0.028,+0.078] |
| **◆ SELL** (yellow diamond) | `z > +1.5σ` — a strong close | **Did NOT confirm out of sample.** Drift-free holdout +0.0094, CI [−0.030,+0.052] |

The source indicator labels the sell side **CAUTION** rather than a short entry, for exactly
that reason. Sanket surfaces it as a sell signal (a configured product decision), and carries
the caveat with it everywhere it appears: the Action Dashboard tab description, the Signal
Reference cards, and the Excel legend.

## How it was validated

The search moved to daily data because intraday has no statistical power. Its predecessors
(v3/v4/v5, session breadth) were measured over 370,686 intraday bars on 82 datasets — but that
is only ~601 *independent* observations, where the smallest provable effect is 0.114 ATR, bigger
than most intraday edges ever get. None had forward directional edge.

So: **39 instruments, 251,200 daily bars, 1993–2026**, spanning dot-com, GFC, ZIRP, COVID and
the inflation cycle. 22 pre-declared features × 3 horizons = 66 tests, Bonferroni corrected,
block-bootstrapped by date. Discovery 1993–2013; holdout 2014–2026 sealed and opened once.

**What died:** `sb_effort` (the literal SB core) at p_bonf = 1.00 every horizon, with its own 21d
and 63d versions disagreeing on sign. `sb_breadth` likewise. All five momentum variants, both
MA-trend variants, overnight/intraday decomposition, volatility change, efficiency ratio.

**What lived:** `sb_clv` alone. Holdout: h=1 IC −0.0251 (p 3.5e-03), h=5 IC −0.0254 (p 4.6e-04).

## The edge is small and decaying — do not let the significance mislead you

| Era | IC | | Era | IC |
|:---|:---|---|:---|:---|
| 1993–1999 | −0.113 | | 2014–2019 | −0.025 ← holdout |
| 2000–2006 | −0.065 | | 2020–2026 | −0.026 ← holdout |
| 2007–2013 | −0.056 | | | |

Four times weaker than the 1990s, now stable at roughly a quarter of it. This is an overlay,
not a system.

## Why the EVENT form — the single most important design decision

Holding a **continuous** position on this signal turns over daily, costs 12%/yr at 3bp, and is
fatal: net Sharpe **−0.48**. Firing only on `|z| > 1.5` (9.3% of days) and holding ~10 days cuts
turnover **~35×**. Measured on the shipped logic:

| Scope | Cost | Discovery | Holdout |
|:---|:---|:---|:---|
| All 39 instruments | 3bp | NET +0.124 | NET +0.132 |
| US equity idx + sectors | 3bp | — | NET +0.430 |

Cost breakeven is ~7bp pooled; on US equity indices and sectors it stays positive past 10bp.
**Turnover, not signal strength, was the binding constraint** — and the event design is the one
thing that fixes it. This is why the threshold and the hold horizon are not decoration: at
`thr = 1.0` the signal fires 44% of days and loses to costs; at `2.0` it fires 0.1% and failed
holdout; `1.5` fires 9.3% and is net-positive in both eras.

## Scope — MEASURED on your universe, not inherited from a table

The source study's edge was drift-free and holdout-confirmed only on US equity indices and
US sectors. Those are *its* 39 instruments. Earlier versions of this app hardcoded its
eight per-class results and applied them as a conviction multiplier. That was indefensible:

- it could not cover a universe the study never touched (NSE F&O single names, NSE thematic
  ETFs, crypto);
- it made an **asset-class** claim and applied it to **instrument-level** decisions;
- it could not report that the edge had stopped working — while the study's own headline is
  that the edge decayed 4× since the 1990s;
- and it was unfalsifiable in-product: you could not check it against your own data.

So the app measures it. [`edge.py`](edge.py) runs an event study on **your symbols**, at the
**pre-declared** parameters, using the methodology that makes the source numbers credible in
the first place. Seven steps, each of which kills one specific way of fooling yourself:

| # | Step | The failure it prevents |
|:--|:---|:---|
| 1 | **Event study at the declared horizon** — enter the bar after the signal closes, hold `horizon` (the study's EXEC-B) | Measuring the continuous form, which turns over daily and nets −0.48 Sharpe — a question nobody trades |
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

1. **Lean.** The study computes the close-location z and forward returns *only*. No volume
   profile (a Python double loop, the app's slowest path), no regime engine, no order flow.
2. **Streaming.** Symbols are fetched and reduced in chunks of 20; each chunk's frames are
   released before the next is fetched. What accumulates is event tuples at a ~9% fire rate.
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

The source study's per-class numbers survive in `engine.CLASS_EDGE` / `CLASS_HIT` purely as a
**labelled comparison row**: "the source study measured *US index / ETF* at +0.121; here is
what we measure on your universe." `compute_ranking` does not read them, and
`instrument_class` exists only to choose which row to display.

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

- **Horizon** 5–10 trading days. There is **NO intraday edge** here. None was found, none is claimed.
- **Entry** the next session's open after the signal bar closes. Backtested exactly this way
  (EXEC-B), because entering at the signal close is not available to most traders and tests
  barely different anyway.
- **BUY** weak close (z < −1.5) → expect mean reversion up. The tradeable side.
- **SELL** strong close (z > +1.5) → the side that failed holdout confirmation. Stronger evidence
  for trimming longs than for initiating shorts.

## The engine — [`engine.py`](engine.py)

### 1. Per-symbol signal · `add_clr_features(df, z_look, thr, horizon)`
```
CLR_CLV       = ((C−L) − (H−C)) / (H−L)          in [−1, +1]
CLR_Z         = (CLR_CLV − SMA(CLR_CLV, z_look)) / STDEV(CLR_CLV, z_look)
Fade_Score   = −CLR_Z                             positive = bullish
buy_cond     = CLR_Z < −thr                       green triangle
sell_cond    = CLR_Z > +thr                       yellow diamond
CLR_Hold_Dir  / CLR_Hold_Age                       the hold window, Pine's sinceSig/sigDir
CLR_State     = WARMING UP / BUY / SELL / NEUTRAL
```
`STDEV` uses `ddof=0` — Pine's `ta.stdev` is the *population* standard deviation. Using the
sample stdev would shift every z and drift the fire rate off the measured 9.3%.

A symbol needs `z_look + 2` bars before it can signal (the Pine's own warmup refusal); shorter
histories are excluded from the screen with a "warming up" count surfaced in the run stats.

**Weekly is an extrapolation.** The study was daily. `z_look` becomes 52 on the Weekly timeframe
(one year, the closest structural analogue) and the Engine Status card labels it as extrapolated.

### 2. Cross-sectional ranking · `compute_ranking(df, cost_bps, thr, study)`
Ranks the universe by `Fade_Score` (weakest closes first). `Side` is `Buy` / `Sell` / `—`, gated
on ±`thr`: **only a fired event is actionable.** Sub-threshold rows still appear in the ranking
tables — the score is continuous — but their Side reads `—`, because the measured edge is in the
event, not in the continuous score.

```
Conviction = clip(0.30 + 0.70·clip(|z|/3, 0, 1)) × cost_factor
cost_factor: 1.00 if the cost gate passes, else 0.50
             — measured from `study` when one exists, else the pooled ~7bp prior
```
Conviction is a *relative weighting*, not a probability, and it is labelled that way in every
tooltip. Note what is deliberately **absent**: no expectancy term (that is measured by
`edge.py` and *reported*, never folded into a number the reader cannot audit), no per-name
volatility factor, no regime factor, no live-IC scaling.

### 3. Bar convention — one deliberate difference from the Pine
The Pine reads `z[1]` inside `request.security(..., "D", ...)` so an *intraday* chart cannot
repaint a daily signal. Sanket evaluates completed daily (or weekly) bars directly, so that shift
is unnecessary: a signal fires on the bar whose close produced it, and entry is the next session's
open — the same trade the Pine backtested. The one carry-over: **a signal on a session that has
not closed yet is provisional until it does.**

## What is context, and never a signal input

Everything else the app computes is descriptive. It is displayed beside the signal, aggregated in
the range-mode charts, and exported — but it does not enter `CLR_Z`, `Side`, or `Conviction`:

- **Order flow** — inferred bar delta, CVD and its slope, `Delta_Z`, absorption, rolling buy
  share, volume profile (POC/VAH/VAL, `VA_Pos`), RVOL. OHLC proxies, validated three times to add
  no cross-sectional ranking edge.
- **Flow zone** (`Condition`) — where cumulative delta sits vs its 20-bar mean:
  Accumulation(+) / Distribution(+) / Neutral. Consumed by the Correlation setup classifier and
  the range-mode breadth charts.
- **Regime** — HMM bull/bear, GARCH vol regime, CUSUM change points. Per-name **risk context**.
  It informs the Regime / Vol columns and the range-mode Regime tab.
- **Forward returns** (`Ret_1b/5b/10b/21b`, Historical Range only) — evaluation **labels**.

`Delta_Z` and `CLR_Z` are cousins, not duplicates: `Delta_Z` z-scores the *volume-weighted* close
location, `CLR_Z` the raw close location. Only the latter is the signal.

## Outputs (per symbol)

| Column | Meaning |
|:---|:---|
| `CLR_CLV` | close location in [−1, +1] |
| `CLR_Z` | **the signal** — z-score of the close location |
| `Signal` / `Fade_Score` / `CLR_Score` | `−CLR_Z`; positive = bullish. What every table ranks on |
| `buy_cond` / `BUY_Today…BUY_5d` | green-triangle event and its age |
| `sell_cond` / `SELL_Today…SELL_5d` | yellow-diamond event and its age |
| `Side` | `Buy` / `Sell` / `—` (context only) |
| `Conviction` | `[0,1]` = \|z\| × cost gate. No expectancy term — see below |
| `CLR_State` | WARMING UP / BUY / SELL / NEUTRAL |
| `CLR_Hold_Dir` / `CLR_Hold_Age` | hold-window direction and bars elapsed |
| `CLR_Rank_Pct`, `Priority_Long/Short(_pct)` | cross-sectional ordering keys |
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
- **The edge decays.** A quarter of its 1990s strength. Stable there for two holdout eras, but
  nothing guarantees the next one.
- **A 0.13–0.43 net Sharpe is an overlay, not a system.** Position sizing, risk limits and
  portfolio construction are all outside this tool.
- **Weekly is unvalidated.** The study was daily; the weekly variant is a structural analogue.
- **Costs decide everything.** Past the class breakeven the event form is net negative, and the
  cost gate halves conviction to say so — but it cannot make the trade profitable.
- **A live session is provisional.** Today's row can change until the close.
- **[`research.py`](research.py) is a legacy harness.** It documents the cross-sectional momentum
  study that the *previous* engine was built on; it does not validate CLR. For the source
  study's evidence read the [`sb_v8.pine`](sb_v8.pine) header; for evidence about *your*
  universe, run the Edge Study. Trust the app's measurement over this document, this document
  over the Pine header for how the app behaves, and neither over `research.py`.
