# Sanket — Engine Architecture & Research Basis

> This document records *why* the engine is built the way it is. Sanket screens a universe with
> **Pragati** (`pragati.pine` v9.2, conviction × value) — the indicator Pragyam's Conviction-Value
> Grid reads — ported in `pragati.py`, `samanvaya.py` and `cvgrid.py`, and surfaced by `engine.py`.
>
> Two kinds of number appear below. Numbers about **the source indicator** come from the Pine's
> own evidence section; numbers about **Pragyam's grid** come from Pragyam's pre-registered study.
> Numbers about **your universe** are measured live by [`edge.py`](edge.py) and appear only in the
> app — nothing about your symbols is hardcoded anywhere in this system.

## Why the engine changed

Siddhi (v7) fired whenever its conviction oscillator crossed its own signal line — about 113 times
per 1,000 bars. Its source measured that bare crossing as its **weakest** configuration (+0.0205R
on the instruments it was fitted to, +0.0015R with t = 0.2 on eight held out). Sanket shipped it
because it was the condition asked for, and said so.

Pragati keeps Siddhi's measurement at its root — `c = ΔC / TR`, participation-weighted, how much of
a bar's travel became progress — and asks the other half of the question on the same bar: at what
**price** was that progress made? Samanvaya's founding result is that integrating position with
momentum beat either alone (information coefficient +0.0691 against +0.0637 and +0.0598 across 24
targets — directional, not decisive). Pragati does one level up what Samanvaya did inside itself.

## The stack

```
            conviction tape (MTF)                        value tape (MTF)
                  │  who controls                              │  where price stands
                  ▼                                            ▼
 OHLCV ──► conviction (chart) ──┐                 ┌── value (chart) ◄── OHLCV + macro drivers
                                ▼                 ▼
                          TRACE = conviction × value  (how far the move is stretched)
                                │
                          HISTOGRAM = trace − EMA9    (the trace's push)
                                │
        ┌───────────────────────┼─────────────────────────┐
        ▼                       ▼                         ▼
  ▲▼ (from the grid) / ◆   3 × 3 GRID STATE           edge.py
  (events)                 (where a name stands)      (measured on your universe)
```

### Conviction — Nishchaya v3, exactly

```
c    = (C − C[1]) / TR                        bounded −1 … +1; gaps count
w    = min(V / EMA(V, 20), 3)                 relative TR where there is no volume
raw  = 100 · SMA(c·w, 20) / SMA(|c|·w, 20)    the Agreement denominator (the measured one)
conv = EMA(100 · tanh(raw / 3σ(raw, 200)), 3)
```

A wide bar that closes where it opened scores zero; a heavy bar that goes nowhere adds to the
denominator and not the numerator. That is effort without result, and it is why divergence between
effort and result means something here rather than being arithmetic.

### Value — Samanvaya, carried whole

The name's per-bar log return is regressed on up to three macro factors, chosen by stepwise partial
correlation over a 250-period window read 12 periods in arrears (so the fit applied to any bar
ended 12 periods earlier), admitted past a Šidák-corrected Fisher floor sized to the sample
actually available, solved by ridge Gram-Schmidt in correlation space. **The hedge weighs
itself**: it is applied in proportion to its own out-of-sample skill over the last 104 periods, so
an instrument the drivers do not explain collapses to its own path. The running sum of the hedged
residual is a spread; its z over five timescales (8 … 55), each high-passed by a 400-bar EMA and
rescaled by 0.887 so θ keeps its single-window meaning, is the RV leg. Seven Market-Strength views
of the name's own price are the breadth leg. The legs are blended in z-space with their measured
correlation, variance restored, bounded once.

`samanvaya.py` is Pragyam's port with the selection and regression **vectorised across time** —
only the three hysteresis paths, which are genuinely path-dependent, run bar by bar. Checked:
bit-identical residuals and driver choices against Pragyam's loop; 5–6× faster, which matters on a
500-name screen and a 15-year study. Breadth runs on plain arrays for the same reason (identical to
1e-13).

The basket is Pragyam's **expanded** one — Brent beside WTI, copper, and the name's home equity
index — kept because Pragyam's pre-registered test found it raised median out-of-sample hedge skill
in every universe tested (ETF 0.01 → 0.45, Nifty 50 0.005 → 0.23, Dow 30 0.01 → 0.20). Driver
timing is the Pine's: a driver closing more than a third of a chart bar after the name is read at
its previous close.

### The trace and its histogram

```
z_c   = EMA(raw / σ, 3)                       conviction in σ
z_v   = Samanvaya's unified z                 value in σ
trace = 100 · softbound(0.5 · (z_c + z_v) / √(2 + 2ρ))     ρ measured over 200 bars
hist  = trace − EMA(trace, 9)
```

θ = 1.5σ lands at ±42.9 on the trace exactly as on the value tape. The histogram's push is read in
five levels from its three drawn channels — hue (side), lightness (impulse / pressing /
decelerating / turning) and saturation (a quiet regime, where the raw share's σ sits in the bottom
fifth of its history, gives no push).

### The tapes

Each ingredient is averaged across a ladder of timeframes in linear z and compressed once, with no
second normalization, so agreement across frames is **rarer** than any one frame's reading.

| Chart | Conviction tape | Value tape |
|:---|:---|:---|
| Daily | **Ladder down**: 1m·3m·5m·15m·30m·1h·4h inside · D, from yfinance's intraday history (`intraday.py`), each rung joining where it has calibrated; ↺ on older bars: W · D — the W rung **reconstructed**: the week's settled state as of its last close, completed with the week forming | W · D — the RV ensemble on the spread sampled at weekly closes, finished with today's spread; breadth at the last closed week |
| Weekly | the daily bars inside each week (participation-weighted), plus 1h and 4h where they exist · W | M · W — the monthly rung is its RV leg alone (100 months of breadth are never available) |

## The signals

**v9 reads the ▲▼ from the grid** (`pragati.v9_signals`; the Pine's section 8b).

**▲ CAPITULATION** — the first closed bar on which the grid stands in Buy · capitulation (the
conviction tape past −30, held there by conviction's own histogram; the value tape past −θ) with
the value momentum tape reverting (the 5 × 5 value phase +1). **▼ DISTRIBUTION** — the first bar in
Exit · distribution (sellers in control of a price past +θ). 10-bar cooldown per side; the last one
stands as the declaration. Measured (studies/pragati_v9_audit.md, three eras, daily and weekly, no
look-ahead): the ▲ +0.046 / +0.056 / +0.046σ at 10 bars; the distribution state followed by
underperformance in every era.

**v8's TURN** — the trace crossing back through θ, confirmed on its tapes inside 5 bars — measured
no edge after 2020 on daily bars and negative on weekly; v9.2 removed it.

**◆ RESUME** — the histogram dipped below zero inside 6 bars and crosses +0.5σ; chart conviction
is above zero; the conviction tape is at or past +30; the value tape is below +θ; effort is not
absorbed on the bar. Short mirrors. **Off by default since v8** (negative outside crypto; negative or
mixed again in v9).

A ▲▼ takes precedence over a ◆ on the same bar; ▲ and long ◆ share one 10-bar cooldown.

**Divergence is off by default in v9** (R about zero in every era, daily and weekly; H negative or
mixed), and when switched on it is found on **conviction's own pivots** (5/5,
separated by 5 … 60 bars), not on the trace — whose pivots would be a new, unmeasured object. It
counts only when zone-gated (the earlier pivot beyond ±30) and when value at the pivot was
stretched the right way. Price is sampled at the swing's true extreme.

**Why the grid, not the trace.** The trace blends two things, so its crossing alone cannot say
which of them moved. The grid keeps them apart — who controls across the ladder, where price
stands — and the audit found the edge exactly there: sellers still in control of a price already
cheap, with value starting to turn.

## What the source measured

Quoted from the Pine, at face value, including what does not flatter it:

| Element | Result |
|:---|:---|
| Regular divergence | **Ranked first** — the only element positive on both the primary futures and eight held-out instruments in nearly every configuration. Ranked, not established. |
| Participation weighting | Earns its place: Off is the worst setting in all three architectures |
| Adaptive scaling | Calibrated: 3.2–4.3% beyond the outer zone |
| Chart-only reversal trigger | +0.0003R, t = 0.01 — why no ▲▼ fires on conviction alone |
| Hidden divergence | No edge on any universe; retired |
| Continuation | +0.036R on the primaries, −0.016R elsewhere |
| Anything, overlap-corrected | **Nothing reaches significance**; best t = 1.9 of 48 cells |
| Fitted vs unseen edge, 900 configurations | correlation ≈ 0 |
| Trace, histogram, the ▲▼ / ◆, grid | unmeasured by the source — measured since by the v8 and v9 audits (`studies/`) |

Method: next-bar entry, 2 ATR target / 1 ATR stop, 20 bars max, edge in R over a matched
baseline; 60/40 split plus eight held-out instruments; 26 years daily, ~730 days hourly, 60 days
of 5m and 15m, on GC, SI, CL, HG, NG, ES, SPY, QQQ, TLT, 6E, BTC.

**Do not tune this to a backtest.** Every input Sanket uses is the Pine's own default, and none is
exposed in the UI.

## The grid, and what Pragyam measured about it

The conviction tape is the row (UP past +30, FAINT between, DOWN past −30), the value tape the
column (cheap past −θ, fair, rich past +θ). Conviction's own histogram decides when a row may
change — only while it confirms a push that way (on the move's side, not turning, not quiet);
otherwise the row is held. Units are graded inside each cell on the Pine's shading ramps, and the
5 × 5 phases halve an edge the faster view does not confirm. **Grid v8** is Pragati v5's 3 × 3
with four cells measured (`studies/pine_audit.md`, chosen before 2018, confirmed after):

```
            CHEAP                     FAIR                      RICH
UP          Buy · turned 3            Hold · building 1.5       Trim · paid 0.75
FAINT       Accumulate · basing 1.5   Wait · idle 1             Trim · stalling 0.75
DOWN        Buy · capitulation 4      Accumulate · washout 1.5  Exit · distribution 0.25
```

The seed had DOWN·cheap 1, DOWN·fair 0.5, UP·fair 3, UP·rich 1.5. Read as a position outside
crypto, the v8 units give +0.039σ / +0.039σ before / after 2018 at h 10, against −0.007σ / −0.027σ
for the seed and +0.016σ / +0.002σ for v7's 4 × 4. It is the final version for that reason. In
Pragyam's own allocator, re-measured before shipping (`research/cvg_reweight.py` there), v8 beats
the seed units in both eras on Nifty 50 (+0.42 / +0.98 %/yr) and Dow 30 (+0.55 / +0.89 %/yr) at
lower turnover. No single t clears 2.

For the record, the seed units through the same kind of allocator (monthly rebalances, every
name held) on Pragyam's ETF book, Nifty 50 and Dow 30:

```
             CVG − EW             gate (histogram)      grading
ETF book     −0.17%/yr (t −0.43)  −0.12%/yr (t −0.93)   +0.16%/yr (t +0.99)
Nifty 50     −0.32%/yr (t −0.62)  +0.31%/yr (t +2.17)   −0.07%/yr (t −0.29)
Dow 30       −0.41%/yr (t −0.78)  +0.28%/yr (t +1.47)   +0.28%/yr (t +0.99)
```

The seed units: within half a percent of equal weight everywhere, never above it. The histogram's gate earns its
place on single stocks. The two tapes are about +0.6 correlated (value's breadth leg is momentum),
so the corners that need them to disagree stay thin. **Sanket shows the grid as a name's state; it
neither fires on it nor, since the trace study, ranks by it.**

## Ranking

```
long side                                   short side
5 + s   ▲ on this bar                       5 + s'  ▼ on this bar
s       every other name, by stretch        s'      every other name, by stretch

s = −trace / 200 ∈ (−½, ½)   (stretched furthest DOWN leads the long side)   s' = −s
```

**Read as reversion — measured, not inherited.** v8.0.0 ranked by the grid's weight, banded
TURN > RESUME > hold window > open TURN window > grid state. `trace_study.py` measured that on five
NSE universes over ~15 years and it ran **backwards**: long-minus-short −0.022σ before 2021 and
−0.043σ after, clearly negative in 4 of 8 runs. Every ingredient — conviction, value, the trace,
the push, the grid weight — correlated *negatively* with the next 5–40 bars; the buyers-firm cells
the grid calls *Add* lagged, the sellers-firm cheap cells it calls *Watch* / *Reduce* led.

The replacement was chosen on the pre-2021 era only: stretch +0.058σ, TURN kept on top +0.059σ;
RESUME and the hold / open-window bands cost edge and no longer order the list. On the sealed
2021–2026 holdout it beat the grid ranking on average (Daily −0.007σ vs −0.043σ; Weekly
+0.087σ vs −0.040σ; rank IC clearly positive on NIFTY 50 / 100). Stated plainly: once a name's own
20-bar return is removed the trace carries ~0 information — on NSE equities this ranking **is**
short-term reversal, read through the indicator, and that effect has been weaker since 2021 on
mid and small caps. The grid, RESUME and the windows are still computed and shown; they describe
a name, they do not order the list. Reports: `studies/`.

Nothing measured on YOUR universe enters: the Edge Study is reported, never applied, and the
cost gate is a flag on the row.

## The Edge Study

`edge.py` measures the event set on the symbols on screen, through `engine.compute_frame` — the
exact call the screener makes — so warm-up, the stack gate, the basket gate and the cooldowns apply
to the study by construction. Six slices: Long · all and Short · all (the screen's sides), and each
kind alone (▲ capitulation, ▼ distribution, ◆ RESUME ↑, ◆ RESUME ↓).

| # | Step | The failure it prevents |
|:--|:---|:---|
| 1 | Event study: enter the bar after the signal, hold 10 | Measuring a continuous form nobody trades |
| 2 | Drift removal, **causal**: the mean of the 500 h-bar returns realised before the event (v9; was in-era, a look-ahead) | Beta read as edge |
| 3 | Vol normalisation by the symbol's own trailing σ | Incomparable instruments |
| 4 | Sign folding | Two sides on opposite conventions |
| 5 | Block bootstrap over dates | Overlap and a correlated cross-section inflating significance |
| 6 | Cost in the same vol units | The same bps costing 4× more on a low-vol name |
| 7 | Power stated: `n_eff = (dates / h) × participation ratio`, and an MDE | "No edge" from a test that could never have seen one |

`UNDERPOWERED` (the MDE exceeds 0.036) is distinct from `NO EDGE`. The ▲ is worth about +0.05σ, so
its slice will often be underpowered even over 15 years; the pooled sides less so.

**Memory.** Streaming in chunks of 20 symbols, lean (no volume profile, no regime engine),
universes above 80 sampled with a fixed seed. The macro drivers are one deep batch for the whole
study. Measured once per universe per day and reused until the date rolls.

## Adaptations, stated

| Adaptation | Why |
|:---|:---|
| Daily conviction ladder is Ladder down to the depth yfinance carries (1m 7 d … 1h ≈ 730 d), ↺ W · D before it | The Pine's default (v9.1) reads 1m … 4h intrabars; free intraday history is shallow, so older bars fall back as the Pine falls back when a direction has no frames |
| The Daily chart's W conviction rung normalises over 52 weeks | At 200 it needs four years of weekly history |
| Weekly normalises over 60 bars | Calibration costs two windows; at 200 a weekly name needs 8.7 years |
| Quiet regime ranks over the history available (≥ one window) | The panel cannot supply 4 × 200 bars of σ |
| A volume-less name's reconstructed parent rung calibrates on true range | The Pine requires a volume baseline there, which index spot and FX never have — under Ladder up every signal on them would pause for good |

**Warm-up.** The histogram calibrates after `length + vol_n + 2·norm + signal` = 450 daily bars.
The signal set also waits for both tapes; the conviction tape's weekly rung binds (92 weeks, then a
normalization window more), so the stack first judges near bar ~670. Weekly is bound by the monthly
value rung, about five years. Fetch depth follows: Daily 1,300 + 365 calendar days, Weekly
2,600 + 365.

## Repaint posture

Every column is computed from bars that exist. Higher frames enter as their last *settled* state
plus the forming parent bar aggregated from the chart; the Weekly chart's lower frame only as the
daily bars inside the week. Verified: every bar computed on the full history equals the same bar
computed on history cut at it (63 columns, including the grid and the hold windows). Signals are
committed on the bar's close; entry is the next session's open. **A live, unclosed session is
provisional until it closes.**

## What is context, and never a signal input

Order flow (inferred delta, CVD, `Delta_Z`, absorption, volume profile, RVOL), the flow zone, the
regime engine (HMM, GARCH, CUSUM) and forward returns. Displayed, aggregated and exported — none of
it enters the trace, the tapes, the signals or the grid. (`PRG_Absorbed` is Pragati's own
effort → result reading, not the order-flow absorption score.)

## Known limitations

- **The signal set is unmeasured in its source**, and the grid measured no better than equal
  weight. Read the Edge Study for your universe; read everything else as structure.
- **Momentum leads the default trace.** Samanvaya's value is half breadth (price momentum); blended
  with conviction at equal weight, about three quarters of the trace is momentum of some kind.
- **The tapes overlap the trace by construction** — each tape's chart rung is one of the trace's
  ingredients.
- **Adaptive scaling is relative.** A conviction reading is extreme *for this instrument* over the
  window; in a dead range small imbalances are amplified (the push then reads "quiet").
- **One split, not a walk-forward**; large universes are sampled for the study.
- **Costs decide the timeframe.** Daily is the only timeframe with real headroom; there is no
  intraday claim.
- **This is an overlay, not a system.** Sizing, risk limits and portfolio construction are outside
  it — that is Pragyam's job.
