# Sanket — Engine Architecture & Research Basis

> This document records *why* the engine is built the way it is. Sanket runs **one screening
> condition**: SB v8 close-location reversal, ported from [`sb_v8.pine`](sb_v8.pine). Every
> number quoted below comes from that study; the Pine header is the primary source and this
> document is a summary of it. If the two disagree, the Pine is right.

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

## Scope — the instrument class is wired to your universe selection

The edge is drift-free and holdout-confirmed **only** on US equity indices and US sectors.
`engine.instrument_class` maps Sanket's universe selector to the Pine's instrument-class input,
so the expectancy the UI reports always matches the asset class on screen. This is honesty
wiring, not a tunable parameter.

| Sanket universe | Instrument class | OOS edge (vol) | Hit | Established? |
|:---|:---|:---|:---|:---|
| US Indexes | US index / ETF | **+0.121** | 57.5% | **yes** — CI excluded zero |
| — | US sector ETF | **+0.068** | 54.9% | **yes** — CI excluded zero |
| India Indexes · ETF Index | India index | +0.089 | 51.9% | no — n=239, CI includes zero |
| Global Indexes | International equity | +0.003 | 53.2% | no |
| Commodities | Commodity | +0.028 | 51.0% | no |
| Currency | FX | +0.035 | 51.9% | no |
| Global Macro | Rates / Credit | −0.003 | 51.1% | no |
| Crypto | Other / unknown | 0.000 | 50.0% | no — the study covered no digital assets |

Holdout 2014–2026, 10-day horizon, entry next open, each instrument's own mean forward return
removed within era (so "equities went up" cannot contribute). The sidebar Engine Status card
states which case you are in, and the Action Dashboard shows a scope warning banner whenever the
active class is not established. **Believe them.**

One caveat the mapping cannot resolve on its own: the study measured *index- and ETF-level*
instruments. On a constituent universe (individual stocks inside an index) the class edge is
indicative of the asset class, not a measurement on those names. The Signal Reference card says so.

## How to trade it — measured, not asserted

- **Horizon** 5–10 trading days. There is **NO intraday edge** here. None was found, none is claimed.
- **Entry** the next session's open after the signal bar closes. Backtested exactly this way
  (EXEC-B), because entering at the signal close is not available to most traders and tests
  barely different anyway.
- **BUY** weak close (z < −1.5) → expect mean reversion up. The tradeable side.
- **SELL** strong close (z > +1.5) → the side that failed holdout confirmation. Stronger evidence
  for trimming longs than for initiating shorts.

## The engine — [`engine.py`](engine.py)

### 1. Per-symbol signal · `add_sb_features(df, z_look, thr, horizon)`
```
SB_CLV       = ((C−L) − (H−C)) / (H−L)          in [−1, +1]
SB_Z         = (SB_CLV − SMA(SB_CLV, z_look)) / STDEV(SB_CLV, z_look)
Fade_Score   = −SB_Z                             positive = bullish
buy_cond     = SB_Z < −thr                       green triangle
sell_cond    = SB_Z > +thr                       yellow diamond
SB_Hold_Dir  / SB_Hold_Age                       the hold window, Pine's sinceSig/sigDir
SB_State     = WARMING UP / BUY / SELL / NEUTRAL
```
`STDEV` uses `ddof=0` — Pine's `ta.stdev` is the *population* standard deviation. Using the
sample stdev would shift every z and drift the fire rate off the measured 9.3%.

A symbol needs `z_look + 2` bars before it can signal (the Pine's own warmup refusal); shorter
histories are excluded from the screen with a "warming up" count surfaced in the run stats.

**Weekly is an extrapolation.** The study was daily. `z_look` becomes 52 on the Weekly timeframe
(one year, the closest structural analogue) and the Engine Status card labels it as extrapolated.

### 2. Cross-sectional ranking · `compute_ranking(df, iclass, cost_bps, thr)`
Ranks the universe by `Fade_Score` (weakest closes first). `Side` is `Buy` / `Sell` / `—`, gated
on ±`thr`: **only a fired event is actionable.** Sub-threshold rows still appear in the ranking
tables — the score is continuous — but their Side reads `—`, because the measured edge is in the
event, not in the continuous score.

```
Conviction = clip(0.30 + 0.70·clip(|z|/3, 0, 1)) × class_factor × cost_factor
class_factor:  1.00 established · 0.75 positive-but-CI-includes-zero · 0.55 nominally positive · 0.40 zero/negative
cost_factor:   1.00 if within the measured breakeven for the class, else 0.50
```
Conviction is a *relative weighting*, not a probability, and it is labelled that way in every
tooltip. Note what is deliberately **absent**: no per-name volatility factor, no regime factor,
no live-IC scaling. The Pine has no such terms and neither does this.

### 3. Bar convention — one deliberate difference from the Pine
The Pine reads `z[1]` inside `request.security(..., "D", ...)` so an *intraday* chart cannot
repaint a daily signal. Sanket evaluates completed daily (or weekly) bars directly, so that shift
is unnecessary: a signal fires on the bar whose close produced it, and entry is the next session's
open — the same trade the Pine backtested. The one carry-over: **a signal on a session that has
not closed yet is provisional until it does.**

## What is context, and never a signal input

Everything else the app computes is descriptive. It is displayed beside the signal, aggregated in
the range-mode charts, and exported — but it does not enter `SB_Z`, `Side`, or `Conviction`:

- **Order flow** — inferred bar delta, CVD and its slope, `Delta_Z`, absorption, rolling buy
  share, volume profile (POC/VAH/VAL, `VA_Pos`), RVOL. OHLC proxies, validated three times to add
  no cross-sectional ranking edge.
- **Flow zone** (`Condition`) — where cumulative delta sits vs its 20-bar mean:
  Accumulation(+) / Distribution(+) / Neutral. Consumed by the Correlation setup classifier and
  the range-mode breadth charts.
- **Regime** — HMM bull/bear, GARCH vol regime, CUSUM change points. Per-name **risk context**.
  It informs the Regime / Vol columns and the range-mode Regime tab.
- **Forward returns** (`Ret_1b/5b/10b/21b`, Historical Range only) — evaluation **labels**.

`Delta_Z` and `SB_Z` are cousins, not duplicates: `Delta_Z` z-scores the *volume-weighted* close
location, `SB_Z` the raw close location. Only the latter is the signal.

## Outputs (per symbol)

| Column | Meaning |
|:---|:---|
| `SB_CLV` | close location in [−1, +1] |
| `SB_Z` | **the signal** — z-score of the close location |
| `Signal` / `Fade_Score` / `SB_Score` | `−SB_Z`; positive = bullish. What every table ranks on |
| `buy_cond` / `BUY_Today…BUY_5d` | green-triangle event and its age |
| `sell_cond` / `SELL_Today…SELL_5d` | yellow-diamond event and its age |
| `Side` | `Buy` / `Sell` / `—` (context only) |
| `Conviction` | `[0,1]` = \|z\| × class expectancy × cost gate |
| `SB_State` | WARMING UP / BUY / SELL / NEUTRAL |
| `SB_Hold_Dir` / `SB_Hold_Age` | hold-window direction and bars elapsed |
| `SB_Rank_Pct`, `Priority_Long/Short(_pct)` | cross-sectional ordering keys |
| `Signal_Reason` | plain-language read of the row, caveat included |

## Known limitations (disclosed, not hidden)

- **The sell side is not confirmed out of sample.** It ships as a sell signal by configuration;
  the statistics say caution. This is the single most important caveat in the system.
- **Scope is narrow.** Two of eight asset classes are established. On the other six, signals still
  fire — the measured expectancy behind them does not.
- **The edge decays.** A quarter of its 1990s strength. Stable there for two holdout eras, but
  nothing guarantees the next one.
- **A 0.13–0.43 net Sharpe is an overlay, not a system.** Position sizing, risk limits and
  portfolio construction are all outside this tool.
- **Weekly is unvalidated.** The study was daily; the weekly variant is a structural analogue.
- **Costs decide everything.** Past the class breakeven the event form is net negative, and the
  cost gate halves conviction to say so — but it cannot make the trade profitable.
- **A live session is provisional.** Today's row can change until the close.
- **[`research.py`](research.py) is a legacy harness.** It documents the cross-sectional momentum
  study that the *previous* engine was built on; it does not validate SB v8. The SB v8 evidence
  lives in the [`sb_v8.pine`](sb_v8.pine) header. Trust the Pine over this document, and this
  document over `research.py`.
