# SANKET — Institutional Market Signal Terminal
### Pragati · Conviction × Value · Graphite · Pragyam Family · `v8.2.0`

> **संकेत** *(Sanketa)* — Sanskrit for *Signal* · *Indicator* · *Forewarning*

Sanket screens a universe with **Pragati** (`pragati.pine` v6) — the indicator Pragyam's
Conviction-Value Grid already reads — and asks of every name the indicator's own question:
**is the push paid for, and at what price?**

It reports three things, and keeps them apart:

- **Events.** ▲▼ **TURN** — a stretch releasing, each ingredient confirming on its own tape and
  the push that made the stretch shown to have failed. ◆ **RESUME** — a trend resuming from
  inside the zone. Bucketed by age, ranked, and measured.
- **State.** Every name's cell in the **4 × 4 conviction-value grid**, named as an action —
  Buy · Add · Accumulate · Hold · Wait · Watch · Trim · Reduce · Exit — with Pragyam's seed units
  as its weight. Where the name stands *between* events.
- **Evidence.** The out-of-sample expectancy of the event set **on the symbols you put on
  screen**, measured every day by the built-in Edge Study, with the interval and the power stated.

Part of the **Pragyam Product Family** by [@thebullishvalue](https://github.com/thebullishvalue).

> **Read this first.** Sanket is **decision-support**, not a turnkey strategy.
> 1. **The signal set is unmeasured in its source.** `pragati.pine`'s evidence section measured
>    conviction's *components* — regular divergence ranked first, participation weighting earns
>    its place, the scaling is calibrated — and nothing it measured reaches significance once
>    overlapping windows are counted (best t = 1.9 of 48 cells). The TURN / RESUME stack, the
>    trace, its histogram and the grid are new objects. So the app measures them, on your symbols.
> 2. **The grid is a weight, not a forecast.** Pragyam measured its 3 × 3 seed through a real
>    allocator on three universes: within half a percent a year of equal weight, never above it,
>    none of the gaps significant. Sanket ranks with it because it is the indicator's own reading
>    of where a name stands — not because it predicts.
> 3. **Scope is earned per universe, never inherited.** Nothing about your symbols is hardcoded;
>    until the Edge Study measures, the app says "not measured".
>
> Signals are not financial advice.

---

## Contents

- [What Sanket Does](#what-sanket-does)
- [The Stack](#the-stack)
- [The Signals](#the-signals)
- [The Grid](#the-grid)
- [Ranking](#ranking)
- [Edge Study — expectancy measured on your universe](#edge-study--expectancy-measured-on-your-universe)
- [What Is Adapted, and Why](#what-is-adapted-and-why)
- [Outputs](#outputs)
- [Architecture Overview](#architecture-overview)
- [Analysis Modes](#analysis-modes)
- [Asset Universe Coverage](#asset-universe-coverage)
- [UI System — Graphite](#ui-system--graphite)
- [Installation & Launch](#installation--launch)
- [What Changed](#what-changed)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## What Sanket Does

Siddhi, the engine Sanket shipped through v7, fired whenever conviction crossed its own signal
line — about 113 times per 1,000 bars, the source's weakest tested configuration. Pragati keeps
Siddhi's measurement at its root and asks the other half of the question on the same bar:

```
conviction   c = (C − C[1]) / TR,  w = min(V / EMA(V), 3)       how much of the travel became progress
             raw = 100 · Σ(c·w) / Σ(|c|·w),  100·tanh(raw / 3σ)  Nishchaya v3 exactly
value        Samanvaya: the name hedged against a macro basket, its    where price stands against
             spread's z over five timescales ⊕ seven breadth views    what the drivers explain
trace        100 · softbound(0.5 · (z_c + z_v) / √(2 + 2ρ))            how far the move is STRETCHED
histogram    trace − EMA(trace, 9)                                       the trace's own PUSH
tapes        each ingredient averaged across its ladder of timeframes   WHY — who controls, and
                                                                         where price stands
```

The trace says **how far**, the histogram says **which way it is going**, the tapes say **why**.

---

## The Stack

| Layer | Module | Carried from |
|:---|:---|:---|
| Value — rich or cheap against the macro drivers | `samanvaya.py` | Pragyam's port, vectorised; Weekly added |
| Conviction, the trace, the histogram, the signals | `pragati.py` | Pragyam's conviction port, extended to v6 |
| The 4 × 4 grid | `cvgrid.py` | the v6 Pine's section 10 (Pragyam's 3 × 3 seed, grown) |
| Settings, per-symbol features, snapshot, ranking | `engine.py` | Sanket |
| Measured expectancy | `edge.py` | Sanket, now measuring the new events |

**The value engine is Samanvaya, carried whole.** The name's per-bar return is regressed on up
to three macro factors — chosen by stepwise partial correlation over 250 periods read 12 in
arrears, admitted past a Šidák-corrected Fisher floor, solved by ridge Gram-Schmidt — and the
hedge is applied only as far as its own out-of-sample skill has earned. The basket is Pragyam's
*expanded* one: US yields, bond-ETF proxies for other 10-year yields, the dollar, energy (WTI and
Brent), precious and industrial metals, the INR crosses and the name's home equity index
(Nifty for NSE names, the S&P 500 otherwise). Drivers are fetched once per universe. A driver
that closes more than a third of a bar after the name is read at its previous close (US drivers
against NSE lag a day on Daily). If the drivers cannot be fetched, value runs unhedged — the
Pine's "Macro hedge: Off" — and the notice rail says so.

**Every number is causal.** A truncation test — every bar computed on the full history equals the
same bar computed on history cut off at it — passes at 1e-9 across the whole stack. The weekly
conviction rung is *reconstructed* from the forming week and lands on the settled weekly value
to 1e-13.

---

## The Signals

**▲ TURN** (▼ mirrors) — *a stretch releasing, the selling spent.* The trace crosses back up
through −θ (θ = ±42.9, Samanvaya's 1.5σ) with the value basket settled; that opens a 5-bar
window. Inside it, on one closed bar, all of:

| Layer | Condition |
|:---|:---|
| Value · its tape | reached −θ inside the last 20 bars (a dislocation every horizon saw), and not rich past +θ now |
| Conviction · its tape | above zero, or rising two bars running |
| The trace's push | histogram > 0 — the release still pushing |
| The push failed | inside the same 20 bars, effort was **absorbed** (bottom fifth of its history) **or** a regular bullish **divergence** formed on conviction's own pivots, zone-gated, at a price value called cheap |

A ▲ declares BUY, a ▼ SELL. A declaration stands until the opposite one; it has no exit.

**◆ RESUME** (long; short mirrors) — *a trend resuming from inside the zone.* The histogram dipped
below zero inside 6 bars and now crosses +k·σ (k = 0.5); the trace is inside ±θ; the conviction
tape is past +30 (control held across horizons); the value tape is short of +θ (room left); and
effort is not absorbed on the bar.

TURN takes precedence over RESUME on the same bar; ▲ and long ◆ share one 10-bar cooldown.
Nothing pauses silently: a name whose tapes are still calibrating is **paused**, and says which
layer it is waiting for.

**Why divergence and absorption are evidence, not signals.** Divergence was the one element the
source ranked first; hidden divergence and the chart-only reversal trigger measured nothing.
So v6 folds the ranked element into the TURN as evidence that the push failed, rather than
firing it alone.

---

## The Grid

The two tapes place every name in a 4 × 4, each split at its knee and at zero:

```
                 CHEAP              BELOW FAIR          ABOVE FAIR          RICH
buyers firm      Buy · turn 3       Add · trend 3       Add · strong 3      Hold · don't add 1.5
buyers edge      Accumulate ·       Accumulate ·        Wait · drifting 1   Trim · stalling 0.75
                 basing 1.5         early turn 1.5
sellers edge     Accumulate ·       Wait · no edge 1    Trim · rolling      Trim · topping 0.75
                 deep value 1.5                         over 0.75
sellers firm     Watch · still      Reduce ·            Reduce ·            Exit · distribution
                 falling 1          downtrend 0.5       breakdown 0.5       0.25
```

**The histogram runs the rows.** Columns move freely — price is where it is. A row moves only
with the push behind it: a push moves it one step toward the tape, an impulse all the way, no push
holds it (**held**, in gold). Before the histogram is calibrated the row follows the tape.

**The chart cell.** The trace's two ingredients on this chart alone place a second cell; when it
carries more units than the state, the chart **leads ↑**, fewer **↓**. Display only.

**Read the push against the action.** Buy, Add and Accumulate are best done on a push ↑; Trim,
Reduce and Exit now or into a push ↑; Hold, Wait and Watch change nothing.

---

## Ranking

Priority is **a TURN on this bar, then stretch**:

```
long side                                   short side
5 + s   ▲ TURN on this bar                  5 + s'  ▼ TURN on this bar
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
2021–2026 holdout it was never worse than the grid ranking (Daily −0.007σ vs −0.043σ; Weekly
+0.087σ vs −0.040σ; rank IC clearly positive on NIFTY 50 / 100). Stated plainly: once a name's own
20-bar return is removed the trace carries ~0 information — on NSE equities this ranking **is**
short-term reversal, read through the indicator, and that effect has been weaker since 2021 on
mid and small caps. The grid, RESUME and the windows are still computed and shown; they describe
a name, they do not order the list. Reports: `studies/`.

---

## Edge Study — expectancy measured on your universe

`edge.py` runs an event study on your symbols through **the exact engine call the screener
makes** (`engine.compute_frame`), so every guard — warm-up, the stack gate, the basket gate, the
cooldowns — applies to the study by construction. Six slices:

| Slice | What it is |
|:---|:---|
| Long · all / Short · all | the screen's two sides, TURN and RESUME pooled |
| ▲ TURN / ▼ TURN | the declarations alone |
| ◆ RESUME ↑ / ↓ | continuation alone |

The method is unchanged from v7 and each step kills one way of fooling yourself:

| # | Step | The failure it prevents |
|:--|:---|:---|
| 1 | Event study at the declared horizon (enter the bar after the signal, hold 10) | Measuring a continuous form nobody trades |
| 2 | **Drift removal, within era** | Every long signal in a bull market prints a profit — beta, not edge |
| 3 | Vol normalisation by the symbol's own σ | FX, bond ETFs and small-caps on incomparable scales |
| 4 | Sign folding | The two sides on opposite conventions |
| 5 | **Block bootstrap over dates** | Overlapping returns and a correlated cross-section inflating significance |
| 6 | Cost in the same vol units (`bps/1e4 ÷ σ_h`) | The same bps costing 4× more on a low-vol name |
| 7 | **Power stated**: `n_eff`, and a minimum detectable effect | "No edge" reported from a test that could never have seen one |

Verdicts: `CONFIRMED` · `GROSS ONLY` · `DISCOVERY ONLY` · `NO EDGE` · `ANTI-PREDICTS` ·
`UNDERPOWERED` (the MDE exceeds 0.036, the largest effect this indicator family has shown
anywhere). **Expect UNDERPOWERED often, especially for the TURN slices**: every layer must confirm
on one bar, so TURNs are far rarer than Siddhi's crossings were, and the app says "we could not
tell" rather than "there is no edge".

It fetches ~15 years once a day per universe (80-symbol fixed-seed sample above that), streams
in chunks of 20, and reuses the measurement until the date rolls.

---

## What Is Adapted, and Why

Every indicator input is `pragati.pine`'s own default. Five things differ, and each is stated
where it applies:

| Adaptation | Why |
|:---|:---|
| **Conviction ladder on Daily is W · D (Ladder up)**; the Pine's default is Ladder down (1m … 4h) | No free feed carries intraday history at depth. The Pine's own FALLBACK reads the other direction when one has no frames; Pragyam made the same choice |
| **The daily chart's W conviction rung normalises over 52 weeks**, not 200 | At 200 it needs four years of weekly history (Pragyam's adaptation) |
| **Weekly runs a 60-bar normalization window**, not 200 | Calibration costs two windows; at 200 a weekly name needs 8.7 years. Weekly's conviction ladder is the daily bars *inside* each week (Ladder down, the Pine's own fallback on Weekly) and its value ladder is M · W |
| **The quiet-regime test ranks over the history available** (≥ one normalization window) where the Pine asks for four | The panel cannot supply 800 bars of σ history |
| **A volume-less name's reconstructed parent rung calibrates on true range** | The Pine requires a volume baseline there, so on index spot or FX with Ladder up the conviction tape would never calibrate and every signal would stay paused |

**Warm-up.** The histogram is calibrated after 450 daily bars; the signal set needs both tapes as
well, and the conviction tape's weekly rung binds, so the stack first judges near bar ~670.
Daily therefore fetches 1,300 + 365 calendar days (~480 signal-bearing dates). Weekly is bound by
the value ladder's monthly rung (~5 years) and fetches 2,600 + 365 days.

---

## Outputs

Per symbol, on each run:

| Column | Meaning |
|:---|:---|
| `turn_buy` / `turn_sell`, `resume_long` / `resume_short` | the four events on this bar |
| `BUY_*` / `SELL_*` | the long / short event by age — ▲▼ a TURN, ◆ a RESUME, — none |
| `Side` / `Signal_Kind` / `PRG_Event` | an event on this bar, and which |
| `PRG_Armed` / `PRG_Armed_Age` | an open TURN window (the watchlist) and bars used of 5 |
| `PRG_Decl` / `PRG_Decl_Age` | the standing ▲/▼ declaration |
| `PRG_Trace` (`Signal`) | the trace, ±100 |
| `PRG_Hist` / `PRG_Hist_Z` / `PRG_Push` | the histogram, in its own σ, and in five push levels |
| `PRG_CTape` / `PRG_VTape` | the two MTF tapes — the grid's axes |
| `PRG_Conv` / `PRG_Value` | the trace's two ingredients on this chart |
| `PRG_Hedge` / `PRG_Drivers` | the macro hedge applied and the drivers selected |
| `PRG_Div_Seen_*` / `PRG_Abs_Seen` | TURN evidence inside the dislocation window |
| `PRG_Split` / `PRG_Quiet` / `PRG_Settling` | read-with-caution qualifiers |
| `PRG_Stack_OK` / `PRG_Why` | whether the signal set can judge, and if not why |
| `CVG_Action` / `CVG_Why` / `CVG_Units` / `CVG_Held` / `CVG_Lead` | the grid state |
| `Priority_Long` / `Priority_Short` | the ranking keys — a TURN on this bar, then stretch |
| `Signal_Reason` | a plain-language read of the row |
| Risk / flow context | `Vol_Regime`, `Regime_Confidence`, `Change_Point`, `Bar_Delta`, `CVD`, `Delta_Z`, `Buy_Share`, `Absorption_Score` — displayed, never an input |

---

## Architecture Overview

```
sanket.py            ← Streamlit entry point: UI, data + macro-driver fetch, screen routing
engine.py            ← settings, per-symbol features, the snapshot row, ranking, cost gate
pragati.py           ← pragati.pine v6: conviction, ladders, trace, histogram, TURN / RESUME
samanvaya.py         ← the value engine (Samanvaya, section 4c), carried from Pragyam
cvgrid.py            ← the 4 × 4 conviction-value grid
charts.py            ← chart builders: the conviction-value map, tone history, correlation heatmap
trace_study.py       ← backtest: the screener under all three trace settings, paired, holdout-sealed
edge.py              ← measured expectancy: event study, drift removal, block bootstrap, power
research.py          ← LEGACY harness from an older momentum engine; validates nothing here
logger.py            ← structured terminal logging
ARCHITECTURE.md      ← the stack, the evidence, and the design rationale
ui/                  ← theme.py · theme.css · components.py (the Graphite design system)
```

The **regime engine** (HMM + GARCH + CUSUM) and the **order-flow layer** (inferred delta, CVD,
volume profile, absorption) are unchanged and remain context only.

---

## Analysis Modes

1. **Single Date Screener** — Action Dashboard (events by age, each with its grid state, push,
   tapes and evidence) · **Grid** (the 4 × 4 census, the watchlist of open TURN windows, names by
   action) · Signal Strength (the ranking) · System Data (exports, raw frame, Edge Study).
2. **Historical Range** — event breadth by kind, open TURN windows, grid breadth (build vs cut),
   the universe-mean tapes, regime context, forward-return labels, Excel export.
3. **Correlation Analysis** — cross-asset correlation, with confluence = |correlation| × the
   normalised Pragati priority.
4. **Pulse Narrative** — every name's state, both sides, plus the Grid and Strength tabs.

---

## Asset Universe Coverage

| Universe Group | Constituents |
|:---|:---|
| **NSE F&O** | NSE F&O permitted stocks (dynamic; NIFTY-500 superset fallback) |
| **India Indices** | 28+ NIFTY indices: NIFTY 50/500, Bank, IT, Pharma, Midcap, sectoral |
| **US / Global Indices** | S&P 500, NASDAQ, DOW, international benchmarks |
| **ETF · Commodities · Currencies · Crypto · Global Macro** | Gold/Silver/Crude/Gas, FX majors, BTC/ETH, bond/macro ETFs |

**Data sources**: NSE India API (`nsepython` / `NseKit`), Yahoo Finance (`yfinance`, including the
macro drivers), Wikipedia (index constituent lists). Volume is used where it exists and relative
true range where it does not, automatically — index spot and FX work without configuration.
Whether the signal set carries an *edge* on a given universe is not assumed — read the Edge Study.

---

## UI System — Graphite

The design system is **Tattva's, adopted wholesale**. These two are the same product family,
and a reader moving between them should not have to relearn what a panel, a chip or a number
looks like. Everything below arrived with a reason attached, and the stylesheet keeps those
reasons as comments — they name the bug each rule fixes, which is what stops the next author
reverting one.

| Element | Specification |
|:---|:---|
| Ground | Graphite `#0A0C10` → `#1C212A` — near-achromatic, deliberately |
| Interactive | Cobalt `#4C7DF0` (Slate) / `#2B5FD9` (Paper) |
| Long / Short | `#2CA36B` / `#DD5A5A` (Slate) · `#0F7A54` / `#C0392F` (Paper) |
| Caution | Amber `#D79A3C` — **caution only, never brand** |
| Appearances | **Slate** (dark, default) and **Paper** (light, for reading and print) |
| Display / data fonts | Inter (prose) · JetBrains Mono (every figure, tabular numerals) |

**Amber is no longer the brand.** The previous system made amber-gold both the product accent
and its caution colour, so "this is Sanket" and "be careful" were the same signal. Amber now
means caution and nothing else; cobalt carries interaction.

**Paper is a token swap, not a second stylesheet.** `theme.css` defines the canonical dark
`:root`; every component rule reads `var(--token)` with nothing hardcoded outside that block, so
light mode is a second, smaller `:root` appended after it. The choice lives in a plain session
key (never a widget key — Streamlit garbage-collects widget state on any run that does not reach
the control) and the theme is resolved at the top of the script, so chrome and charts can never
disagree about which appearance is active.

**Streamlit is pinned exactly at 1.52.2.** It is a UI contract, not a dependency with a floor —
see the note in `requirements.txt`.

### The type contract

Two families, one job each. **Inter** carries prose and headings; **JetBrains Mono** carries
every figure, with tabular numerals on — a column of prices that shimmers as digits change
width is a column the eye cannot hold a baseline in. You should be able to tell data from
commentary with the page out of focus.

Nine tiers, and nothing may invent a tenth:

| tier | px | what it is for |
|:---|---:|:---|
| `--fs-3xs` | 9 | micro labels, eyebrows — **the floor**, nothing is smaller |
| `--fs-2xs` | 10 | table headers, chips, cells one tier below body |
| `--fs-xs` | 11 | dense metadata, table body |
| `--fs-sm` | 12 | secondary body, compact card values |
| `--fs-md` | 13 | **body** |
| `--fs-lg` | 15 | section titles, compact card headline |
| `--fs-xl` | 19 | card values |
| `--fs-2xl` | 24 | hero secondary |
| `--fs-3xl` | 32 | hero signal |

The ramp steps ~1.08–1.15 through the reading tiers and ~1.26–1.33 through display: it
tightens where sizes must be *told apart* at a glance and opens where they must not be
*confused*. Component tiers descend monotonically with importance — 24 → 19 → 15 → 13 → 12
→ 10 → 9 across masthead, card value, section title, body, description, context, label.

**Uppercase is for terse labels, not for small text.** Chips, card labels, rail group labels
and panel context are one or two words at 9–10px, where uppercase plus positive tracking
(0.08–0.18em) aids scanning. The 10px control-hint/note tier is *not* uppercase, because it
carries sentences and uppercase sentences read slower. The wordmark is the one exemption in
the other direction: a mark is allowed to be uppercase at display size because it is a mark.

**Two statements of the ramp, never three.** `theme.css` declares `--fs-*` for the app DOM;
`ui.components.FS` mirrors the same nine values for markup that renders into a
`components.v1.html` iframe, which cannot see a CSS variable. Using `var(--fs-*)` inside a
table cell applies *no size at all* — it fails silently and the cell falls back to the body
tier. `ui.components.MONO_STACK` exists for the same reason: an iframe that declares a face
it has not imported falls through to the system default, which is how tables end up as the
one surface rendering in a typeface the rest of the UI does not use.

### One anatomy for every framed thing

Charts, tables and embedded iframes all go through the same panel: header (title / context ·
meta · chip) · body · footer. There is no bare `st.dataframe`, `st.error`, `st.warning`,
`st.info` or `st.caption` anywhere in the app — each brings its own typeface, radius and ink
that the stylesheet cannot reach, and three of them on a page read as three different products.
Sanket's screener tables stay bespoke, because per-cell glyphs (▲▼ TURN / ◆ RESUME), grid
states, push levels and hold counters are not expressible as a DataFrame — but they draw their typeface,
row height, header and tokens from `ui.components.table_shell_css`, so the only thing that
differs from a generic table is the content of a cell.

---

## Installation & Launch

```bash
git clone https://github.com/thebullishvalue/Sanket.git
cd Sanket
pip install -r requirements.txt
streamlit run sanket.py
```

Opens at `http://localhost:8501`. There is nothing to configure and nothing to tick: every
indicator input is the Pine's own default, and the **Edge Study runs on every run**. It fetches
~15 years (and the macro drivers behind them) the first time it sees a universe on a given day and
reuses that measurement for the rest of the day — within one calendar day the study reads
identical data, so re-measuring would return a bit-identical answer. It re-measures once the date
rolls. `scipy` is required (the value engine's Šidák floor).

---

## What Changed

**v8.0.0 — Pragati: the screener re-envisioned on conviction × value.** The Siddhi zero-crossing
engine is replaced by the indicator Pragyam's Conviction-Value Grid reads, carried the rest of the
way to its signals. What changed and why:

- **The screening variable is a trace, not a histogram crossing.** Conviction (Siddhi's measure,
  unchanged at its root) is blended in σ with Samanvaya's macro-hedged value into one trace — how
  far a move is stretched — whose histogram is its push. Siddhi fired ~113 times per 1,000 bars on
  its source's weakest configuration; the new events need every layer on one bar.
- **Two signals.** ▲▼ TURN (a stretch releasing; declares) and ◆ RESUME (a trend resuming). The one
  element the source ranked first — regular divergence — is folded into TURN as evidence that the
  push failed, with effort absorption as the alternative.
- **One state for every name.** The 4 × 4 conviction-value grid, named as actions with Pragyam's
  units, is a new Grid tab (census, watchlist of open TURN windows, names by action) and a column
  in every table.
- **Ranking uses Pragyam's inference.** Banded — TURN > RESUME > hold window > open TURN window >
  grid state — and ordered inside every band by the grid's weight. The crossing-force "Conviction"
  column is gone.
- **The Edge Study measures the new events**, pooled per side and per kind, through the same
  engine call the screener makes.
- **Macro drivers** are fetched once per universe (cached an hour); a failed fetch degrades value
  to unhedged and says so in the notice rail.
- **Colours follow the Pine**: green up, red down, amber only for "read with caution". The yellow
  SELL diamond is gone.
- **Deeper fetch**: Daily 1,300 + 365 days, Weekly 2,600 + 365, because the tapes — not the
  histogram — bind the warm-up.
- **Fixed on the way**: a volume-less name's reconstructed weekly conviction rung never calibrated
  (the Pine demands a volume baseline there), which would have paused every index-spot and FX name
  for good under Ladder up. It now calibrates on true range, as the develop step already did.
- Validated offline on a synthetic market (Yahoo was unreachable from the build environment): the
  vectorised value fit is bit-identical to Pragyam's loop; the whole stack passes a truncation
  (no-look-ahead) test; all four modes run headless on Daily and Weekly.


**v7.1.5 — the rail stops repeating the command bar, and the tab boundary stops doubling.**

**The session readout is gone from the sidebar.** Version, Universe, Timeframe, Mode, Class —
five rows restating what the page already says. The command bar carries universe, timeframe and
as-of across the top of every loaded page; the mode is the control the reader set three inches
above it; the instrument class is a display label nothing computes from; and the version is in
the footer. A rail that repeats the command bar is not a readout, it is a second caption for the
same facts. The Engine readout stays, and is now the only one in the rail.

**The doubled rule under a tab bar is gone.** The tab list closes with a 1px rule — and that one
is load-bearing, since the 2px active-tab highlight runs along it — and then the first section
header inside the panel drew its *own* `border-top` about 24px below. Two horizontal rules with
nothing between them but whitespace read as one thick, badly-drawn boundary. A section rule
exists to separate a section from what came before it; first inside a tab panel there is nothing
before it, because the tab bar already made that boundary and made it more strongly.

The selector is scoped to the panel's own top-level block rather than to any first-child inside
it. That distinction is not pedantry: the correlation view opens two COLUMNS with section
headers, and those are `:first-child` too — a looser selector strips their rules as well.
Verified in a real browser against Streamlit's actual tab DOM: page-level header keeps its rule,
first-in-tab loses it, both column headers keep theirs, second-in-tab keeps its, and the tab
list keeps the rule its highlight runs along.

**v7.1.4 — the last of the retired palette, and the last of the dividers.**

**Colour.** An exhaustive comparison against Tattva across all six places colour lives —
`theme.css :root` (84 tokens), `LIGHT_TOKENS` (33), both chart palettes, `_CHART_THEME`, both
`config.toml` ramps, and the iframe table tokens — found the schema identical except for two
real divergences, now fixed:

- **23 `rgba()` literals survived the v7.1.0 hex sweep**, because `rgba(212,168,83,…)` is the
  same retired amber-gold brand as `#D4A853` spelled differently. Five of them were painting a
  header that still read "SIDDHI ENGINE". Beyond being the wrong colour, every one was fixed to
  the dark ground: a 0.12-alpha neon green over graphite is not that colour over Paper, and
  `rgba(255,255,255,0.015)` is nothing at all there. Plotly fills now go through
  `chart_rgba(name, alpha)`; tinted surfaces use the `--*-fill` / `--*-edge` pairs the system
  declares for exactly this.
- **One invented token.** I had written `#B4BCC9` for the dark table `ink_secondary`; the
  stylesheet says `#AEB8C7`. A currency cell was a slightly different grey from the same tier
  everywhere else.

**Three hand-rolled blocks went with them**, each the same mistake: a container invented at the
call site, tinted with a retired literal, separated from what follows by a rule the layout
already provides. The "How to Read" box (also a *plain* string carrying `{…}` placeholders, so
its colours rendered as literal braces and never applied at all) → `render_info_box`. The Signal
Reference cards, with a white tint and a 3px coloured left bar — the one container shape Tattva
deleted outright — → `panel`. The engine header → `render_sub_header` plus the note tier.

**Vertical hierarchy.** Every rule in the stylesheet is now a hairline (18 rules, one weight).
The one exception was a **2px** rule on the signal table's age-group row — the heaviest
horizontal line in the app, under its quietest content, and it was overriding the `.sect` class
that already styled that row with hairlines. Call sites emit no dividers at all: no `<hr>`, no
`section-divider`, no raw border values, nothing heavier than a hairline. The section rhythm is
stated once, in CSS.

`check_ui.py` grew two contracts — COLOUR (no raw literal outside the two mirror files; the
iframe tokens must equal the stylesheet) and VERTICAL (no call-site divider, no raw border, no
rule above 1px). Verified to fire: reintroducing a retired `rgba`, adding a 2px rule, and
drifting a table token each fail it.

**v7.1.3 — the Engine box stops arguing, and `config.toml`'s base stops mattering.**

The box lost its caveat note. Four readout rows, nothing else: Verdict, Edge, Trigger, Cost.
A rail states what the engine is doing; it is not where a reader goes to be argued with. The
two disclosures moved rather than vanished — the bare-crossing warning was already stated
twice in the body (the Signal Reference card and the SELL tab's own description), and the
adapted weekly normalization window, the one fact with nowhere else to live, is now a line in
the **notice rail**, which is the component for "something about THIS run you should know"
and which only appears on Weekly because that is the only timeframe it is true of.

**The residual limitation in `.streamlit/config.toml` is resolved, not merely restated.** That
file is static, so it cannot follow the in-app Slate/Paper toggle — anything Streamlit themes
natively tracks Streamlit's resolution instead, which on the wrong appearance means dark input
fields on a white page. Tattva documents this and prescribes the fix (override those controls
in `theme.css` so the file stops mattering); Sanket now does it.

Sanket renders six native widget types. Five were already painted from our tokens, including
the segmented control — which matters most, since it *is* the appearance switch, and a switch
that looks wrong in one of the two modes it offers is the worst thing on the page. The sixth,
`st.date_input`, appeared nowhere in the stylesheet at all: Tattva has no date inputs and
Sanket has four. The field now joins the existing input rule, and the BaseWeb calendar popover
— a separate element in a separate stacking context that inherits nothing — is painted from
our surfaces, ink and accent.

Worth noting for anyone comparing the two files: none of the five controls Tattva's config
comment names as its justification (radio/checkbox dots, toggle switches, links, code blocks)
are rendered by Sanket at all. The file still earns its place for two things CSS cannot reach —
the first paint, before the browser has parsed `theme.css`, and Streamlit's own chrome (the
hamburger menu, the "Running…" indicator, toasts) — and `base` must therefore still agree with
`APPEARANCES[0]`. The comment now says that instead of inheriting Tattva's reasoning.

**New: `check_ui.py`**, a runnable contract covering all of it — the type ramp, the family
split, the tier map, uppercase discipline, every native being claimed from our tokens, and
`base` agreeing with the default appearance. It exits non-zero on breach. Verified to actually
fire: flipping `base` to `light` and adding an unclaimed `st.radio` each fail it.

**v7.1.2 — the Engine box is a status line again.** It had become a report: a bespoke
`.metric-card` with five inline overrides wrapping an eight-cell grid of inline-styled divs,
in a rail otherwise made of widgets and one readout component. Every value was clipped to
~12 characters with `text-overflow: ellipsis`, and all eight cells carried their meaning in a
`title=` tooltip — the one thing this codebase has repeatedly decided not to do.

Six of the eight cells were edge-study internals (hit rate, minimum detectable effect,
sample, participation ratio, and the two per-side edge numbers) that **System Data ▸ Edge
Study** already reports in full, per era, with a glossary. A 145px column showing `≥0.048`
was not reporting the MDE, it was hinting at it.

It is now four rows in `.rail-readout` — the rail's own key/value grammar, the same component
the session readout below it uses — each carrying one fact: **Verdict** (for this universe,
or an honest "not measured"), **Edge** (the number behind that verdict, when there is one),
**Trigger**, and **Cost**, the only thing that gates conviction. Tone comes from the verdict
kind through one mapping rather than a conditional per cell.

The two caveats that must never be buried — a bare zero-crossing being the source's weakest
tested configuration, and the weekly normalization window being Sanket's number rather than
the source's — moved **out of tooltips and onto the page** as a visible note, which is the
entire reason they exist. The note also names where the full study lives.

Also fixed: the trigger read `hist × 0`, which parses as *histogram multiplied by zero* — the
one arithmetic statement this engine never makes. It now reads `crosses 0 · 10b`, from a named
`trigger_short` property rather than a string replacement at the call site. And the
"not measured" state told the reader to *"tick Measure edge in the sidebar"*, a control that
has not existed since v6.3.0, when the study became unconditional.

Zero bespoke HTML remains in the box. A test renders it unmeasured, measured, in both
appearances and on Weekly, asserting at most four rows, no card, no tooltip, no inline type,
no ellipsis, the bare-crossing caveat always present and the adapted caveat present only when
the window actually is adapted.

**v7.1.1 — the type system is a hierarchy, and it is now enforced.** An audit of every
size, family, weight and tracking value the app emits found the ramp was being declared and
then ignored: **47 font-size literals in `sanket.py`, only 3 of them on the nine-tier scale**
(0.52 / 0.62 / 0.65 / 0.66 / 0.68 / 0.7 / 0.72 / 0.78 / 0.8 / 1rem against a ramp containing
none of them). The 0.52rem one was 8.3px — below the system's own 9px floor, and the smallest
type in the app. All 47 now resolve from the ramp: `var(--fs-*)` in the app DOM,
`ui.components.FS` in iframe markup.

- **13 cells declared `IBM Plex Mono` inside iframes that import JetBrains only**, so the face
  never loaded and those cells fell through to the system default — the exact "tables are the
  one surface in a typeface the rest of the UI does not use" bug Tattva documents. The empty
  and section rows now carry `.empty` / `.sect`, which the shared shell already styles; the
  rest inherit from `body`.
- **Fixed a colour regression from v7.1.0**: the sweep that replaced hardcoded hexes with
  `_sid_buy()` did so inside *plain* string literals in `_side_cell`, `_hold_cell`,
  `_hist_cell` and `_conv_cell`, so the braces rendered literally and the browser dropped the
  declaration. Every em-dash cell, both Side cells and the expired-hold cell had been drawing
  with inherited ink. Found by walking the AST for non-f-string literals carrying a `{...}`
  placeholder, then confirmed against rendered output.
- **Three cells used `var(--fs-*)` inside an iframe**, which resolves to nothing — they were
  sized by fallback. Caught only by checking the *rendered* markup; the source looked correct.
- Off-system values removed: `1.25rem` in a responsive override (the only size in the
  stylesheet belonging to no tier, now `--fs-xl`), and `0.09em` tracking on a 9px card label
  (now 0.14em, matching every other 9px card label).
- `--ink-muted` was referenced but never defined, so disabled menu rows inherited the same
  ink as enabled ones — now `--ink-quaternary`.

Two audits now run against the files rather than by eye: one reads the stylesheet and asserts
the tier map descends monotonically with families and case assigned by role, the other renders
every bespoke table and cell and asserts the output carries only ramp sizes, the right face,
no uninterpolated placeholders, and colours that actually flip between Slate and Paper.

**v7.1.0 — the UI is Tattva's, adopted wholesale.** The "Obsidian Quant" layer (saturated navy
ground, amber-gold brand, `IBM Plex Mono` tables) is replaced by the Graphite design system from
Tattva, with the same component vocabulary: section headers, the panel anatomy, chips, metric
cards, KPI strips, empty states, the notice rail and one table primitive. Amber stops being the
brand and becomes caution only; cobalt carries interaction.

- **Two appearances.** Slate (dark, default) and Paper (light). Paper is a token swap over the
  canonical dark `:root`, and it reclaims Streamlit's own natives, which a static
  `.streamlit/config.toml` cannot follow at runtime. The choice lives in a plain session key —
  a widget key is garbage-collected on any run that does not reach the control, which is what
  makes a theme survive idle reruns and die on exactly the actions a user takes.
- **The page shell.** Cold start is masthead → lede → coverage KPIs → system panels → outcomes.
  A loaded page is command bar → notice rail → content: the thing being analysed is always the
  first element, and data-quality notices hang below what they qualify instead of pushing it
  below the fold.
- **Nothing Streamlit-native renders content any more.** 5 `st.dataframe`, 17 `st.error`,
  3 `st.warning`, 5 `st.info` and 2 `st.caption` are gone. The per-column help text the grids
  carried in hover tooltips was not dropped — it moved into panel footers as glossaries, which
  survive a screenshot.
- **Every colour resolves per render.** 67 hardcoded hex literals are gone from `sanket.py`;
  charts go through `chart_color()`, iframe cells through `table_tokens()`. A literal binds at
  import, when there is no session to read an appearance from, which is precisely how a UI ends
  up with its chrome in one theme and its cells in the other.
- **Every chart passes `PLOTLY_CONFIG`** and sits in panel chrome. Without it each one ships
  Plotly's stock toolbar, its logo and a link out to plotly.com.
- **18 `section-divider` spacers removed.** Streamlit wraps each in an element container that
  takes a full slot in the page column's flex gap, so the rule's own margin landed on top of a
  gap that already existed and identical-looking boundaries measured differently. Vertical
  rhythm is stated once, in CSS.
- **Streamlit pinned exactly at 1.52.2** (was `>=1.30.0`), with `starlette<1.0.0` as belt and
  braces. See `requirements.txt` for why a range is not safe here.

Two defects were found and fixed rather than copied: `--ink-muted` was referenced but never
defined, so disabled menu rows rendered in the same ink as enabled ones; and the correlation
lists were drawing on `.corr-row`/`.corr-bar-*` classes that no longer exist in any stylesheet,
with a bar whose length depended on its container rather than on the correlation. They now use
the system's own `.lookback-row` and `.conviction-bar`.

**v7.0.1 — tracking `siddhi.pine` v3·VP.** The source added a volume-profile overlay, which is
display-only on the price chart and changes nothing here. Two things it changed *do* reach the
screen:

- **Warmup is now counted, not guessed**, and it roughly doubles: `2·norm + length + vol_n +
  smooth + signal` = **452 bars**, against the old estimate of 245. The old figure covered the
  first of two nested normalizations and omitted `signal` entirely, so `histSd` — a stdev of the
  histogram — was being averaged across ~200 bars where the histogram is pinned near zero. **That
  mattered more here than on the chart**: the Pine only spends `histSd` on the impulse threshold,
  which at `k = 0` is zero either way, while Sanket divides by it twice, for the cross-sectional
  ranking score and the conviction basis. Both were inflated on every symbol short enough to live
  in that region.
- **The "Raw share" scaling option is gone.** Selecting it left the zones at ±30/±60 on a series
  that lives inside ±15, so nothing armed and three of four signal channels went silent with
  nothing on the pane to say why. A setting whose only effect is to break the engine is a trap,
  not a choice. `SID_Raw` still carries the unscaled reading, which is all it was ever wanted for.

**Weekly needed fixing as a consequence** — and was already quietly broken before it. Resampling
the daily pool yields ~180 weekly bars, short of the warmup on any setting, so every symbol read
WARMING UP forever and the screen came back empty with nothing to explain it. Weekly now runs a
60-bar normalization window (warmup 172) *and* fetches 1900 days instead of 900; both are needed,
neither is sufficient alone. Fetch depth is now part of the data-registry key, so a Daily run
cannot serve its shallower pool to a Weekly one.

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
which retires every frame the old engine cached. The regime engine and order-flow layer survive
unchanged as displayed context. The trigger carries a visible `⚠ BARE` mark: a bare zero-crossing
is the source indicator's own weakest tested configuration, and the app says so rather than
burying it.

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
| Web Framework | Streamlit 1.52.2 (pinned) |
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

*Sanket v8.2.0 · Pragyam Family · Built by [@thebullishvalue](https://github.com/thebullishvalue)*
