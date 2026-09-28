# SANKET — Institutional Market Signal Terminal
### Pragati · Conviction × Value · Graphite · Pragyam Family · `v9.3.0`

> **संकेत** *(Sanketa)* — Sanskrit for *Signal* · *Indicator* · *Forewarning*

Sanket screens a universe with **Pragati** (`pragati.pine` v9.3) — the indicator Pragyam's
Conviction-Value Grid already reads — and asks of every name the indicator's own question:
**is the push paid for, and at what price?**

It reports three things, and keeps them apart:

- **Events** (v9 — read from the grid). ▲ **CAPITULATION** — sellers in control across the ladder
  at a price cheap past θ, and value momentum already turned back toward fair: the one event the
  v9 audit found positive in every era, daily and weekly. ▼ **DISTRIBUTION** — sellers taking
  control of a rich price. ◆ **RESUME** is computed but off by default. Bucketed by age, ranked,
  and measured.
- **State.** Every name's cell in the **3 × 3 conviction-value grid** (v8), named as an action —
  Buy · Accumulate · Hold · Wait · Trim · Exit — with the measured v8 units, the same units
  Pragyam's book now sizes from. Where the name stands *between* events.
- **Evidence.** The out-of-sample expectancy of the event set **on the symbols you put on
  screen**, measured every day by the built-in Edge Study, with the interval and the power stated.

Part of the **Pragyam Product Family** by [@thebullishvalue](https://github.com/thebullishvalue).

> **Read this first.** Sanket is **decision-support**, not a turnkey strategy.
> 1. **The edge is small, and it lives in one place.** The v9 audit
>    ([`studies/pragati_v9_audit.md`](studies/pragati_v9_audit.md); 380 instruments, six classes,
>    three eras, daily and weekly, scored without look-ahead) found the readings — conviction,
>    value, the trace, its histogram, both tapes — carry almost no timing information of their
>    own. The capitulation state and its turn carry the one robust edge: about +0.05 to +0.08σ
>    over 10–20 bars in every era outside crypto. A lean, not a trade.
> 2. **The grid is a weight, not a forecast.** Its v8 units were chosen before 2018 and confirmed
>    after on 380 instruments, and in Pragyam's allocator they beat the seed units in both eras on
>    Nifty 50 and Dow 30 — but no single t clears 2 there. Sanket shows it as the indicator's own
>    reading of where a name stands; the screener does not rank by it.
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
| Conviction, the trace, the histogram, the signals | `pragati.py` | Pragyam's conviction port, extended; signals at v8 (= v5) |
| The 3 × 3 grid | `cvgrid.py` | the v8 Pine's CVG block, graded — bit for bit Pragyam's graded map |
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

v9 reads the ▲▼ **from the grid** (the Pine's section 8b; `pragati.signals` in the port):

**▲ CAPITULATION** — the first closed bar on which the grid stands in **Buy · capitulation**
(the conviction tape past −30 and held there by conviction's own histogram — sellers in control
across the ladder — and the value tape cheap past −θ) with the value momentum tape
**reverting**: the fast end of value has already turned back toward fair — and, since v9.3, not
while conviction's regime is **quiet** (its σ in the bottom fifth of its own history, where a small
imbalance scales into a large reading).

**▼ DISTRIBUTION** — the first closed bar on which sellers hold control of a price rich past +θ
(**Exit · distribution**).

A 10-bar cooldown per side; the last event stands as the declaration, with no exit. Measured
(v9 audit): the ▲ was positive in 2006–13, 2014–19 and 2020–26, on daily and on weekly bars, in
four or five of the five non-crypto classes each time (+0.046 / +0.056 / +0.046σ at 10 bars,
time-series); the distribution *state* was followed by underperformance in every era, though its
entry is too rare to measure alone.

**Why not v8's TURN.** v8's ▲▼ — the trace crossing back through θ, then its value tape, conviction
tape, push and absorption confirming inside 5 bars — faded to nothing after 2020 on daily bars,
read *negative* on weekly bars, and on daily worked only when a capitulation came first. v9.2
removed it from the Pine and the port.

**◆ RESUME** (long; short mirrors) — *a trend resuming.* The histogram dipped below zero inside
6 bars and now crosses +k·σ (k = 0.5); chart conviction is above zero; the conviction tape is past
+30 (control held across horizons); the value tape is short of +θ (room left); and effort is not
absorbed on the bar. **Off by default since v8** (`Params.resume`): negative outside crypto in
the v8 audit and negative or mixed again in v9.

A ▲▼ takes precedence over a ◆ on the same bar; ▲ and long ◆ share one 10-bar cooldown.
Nothing pauses silently: a name whose tapes are still calibrating is **paused**, and says which
layer it is waiting for.

**Divergence is off by default.** The Pine's R (regular) and H (hidden) marks remain available.
Re-measured, R read about zero in every era on daily and weekly bars and H was negative or mixed;
the source's claim that R "ranked first" did not reproduce.

---

## The Grid

The two tapes place every name in a 3 × 3 — conviction at its inner zone (±30), value at θ (±42.9):

```
                 CHEAP                     FAIR                      RICH
buyers           Buy · turned 3            Hold · building 1.5       Trim · paid 0.75
undecided        Accumulate · basing 1.5   Wait · idle 1             Trim · stalling 0.75
sellers          Buy · capitulation 4      Accumulate · washout 1.5  Exit · distribution 0.25
```

**Grid v8 — Pragati v5's grid, with four cells measured.** `pine_audit.py` ran the port on 380
instruments in six asset classes over ~20 years, split before / after 2018. Sellers in control at
a cheap or fair price was followed by gains in *both* eras on every class but crypto, and adding
where buyers hold a fair price earned nothing. The units were chosen on the pre-2018 era alone and
confirmed once after it — outside crypto, the graded units read as a position:

```
                      h = 10 before / after 2018     h = 20 before / after 2018
v5 seed units         −0.007σ / −0.027σ              −0.003σ / −0.043σ
v7 (4 × 4)            +0.016σ / +0.002σ              +0.019σ / +0.003σ
v8 units              +0.039σ / +0.039σ              +0.044σ / +0.051σ
```

v8 − v7, paired by date: +0.017σ (h 10) and +0.020σ (h 20) after 2018, both significant. That is
why the final version is v5's 3 × 3 rather than v7's 4 × 4. **Crypto is the stated limit** — it
trends. Full report: [`studies/pine_audit.md`](studies/pine_audit.md).

**The histogram runs the rows.** Columns move freely — price is where it is. A row moves to its
tape only while conviction's own histogram confirms a push that way: on the move's side, not
turning, not quiet. Otherwise it is **held**, in gold. Before the histogram is calibrated the row
follows the tape. **Graded**: the units are read at the name's shaded position inside its cell on
the Pine's own ramps, and the Pine's 5 × 5 phases halve an edge the faster view does not confirm.

**The chart cell.** The trace's two ingredients on this chart alone place a second cell; when it
carries more units than the state, the chart **leads ↑**, fewer **↓**. Display only.

**Read the push against the action.** The push says which way a row may move next. Measured:
inside capitulation its direction made no consistent difference — the state carried the edge —
so a Buy there does not wait for a push ↑. Hold and Wait change nothing. **The grid is also where
v9's ▲▼ come from** (above): ▲ when *Buy · capitulation* turns, ▼ on entering *Exit ·
distribution*.

---

## Ranking

Priority is **a ▲▼ on this bar, then stretch**:

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

**Re-checked in v9** without look-ahead (that study removed each name's drift inside the era): the
cross-sectional rank IC of the stretch agrees with reading it as reversion on NSE before 2020
(t ≈ +2 to +4) and reads about zero since. The band on top is now v9's ▲ — the capitulation turn,
measured positive in every era.

---

## Edge Study — expectancy measured on your universe

`edge.py` runs an event study on your symbols through **the exact engine call the screener
makes** (`engine.compute_frame`), so every guard — warm-up, the stack gate, the basket gate, the
cooldowns — applies to the study by construction. Six slices:

| Slice | What it is |
|:---|:---|
| Long · all / Short · all | the screen's two sides, ▲▼ and ◆ pooled |
| ▲ capitulation / ▼ distribution | v9's events alone |
| ◆ RESUME ↑ / ↓ | continuation alone |

Each step kills one way of fooling yourself (step 2 changed in v9):

| # | Step | The failure it prevents |
|:--|:---|:---|
| 1 | Event study at the declared horizon (enter the bar after the signal, hold 10) | Measuring a continuous form nobody trades |
| 2 | **Drift removal, causal**: the symbol's mean h-bar return over the 500 returns realised before the event | Every long signal in a bull market prints a profit — beta, not edge. (Up to v8.4 the mean was taken inside the era — a look-ahead that flatters reversal events) |
| 3 | Vol normalisation by the symbol's own trailing σ | FX, bond ETFs and small-caps on incomparable scales |
| 4 | Sign folding | The two sides on opposite conventions |
| 5 | **Block bootstrap over dates** | Overlapping returns and a correlated cross-section inflating significance |
| 6 | Cost in the same vol units (`bps/1e4 ÷ σ_h`) | The same bps costing 4× more on a low-vol name |
| 7 | **Power stated**: `n_eff`, and a minimum detectable effect | "No edge" reported from a test that could never have seen one |

Verdicts: `CONFIRMED` · `GROSS ONLY` · `DISCOVERY ONLY` · `NO EDGE` · `ANTI-PREDICTS` ·
`UNDERPOWERED` (the MDE exceeds 0.036, the largest effect this indicator family has shown
anywhere). **Expect UNDERPOWERED often**: the ▲ is worth about +0.05σ, which one universe's
history can rarely resolve, and the app says "we could not tell" rather than "there is no edge".

It fetches ~15 years once a day per universe (80-symbol fixed-seed sample above that), streams
in chunks of 20, and reuses the measurement until the date rolls.

---

## What Is Adapted, and Why

Every indicator input is `pragati.pine`'s own default. Five things differ, and each is stated
where it applies:

| Adaptation | Why |
|:---|:---|
| **Conviction ladder is Ladder down from yfinance's intraday history** (1m 7 d · 5m/15m/30m 60 d · 1h ≈ 730 d; 3m and 4h built), falling back to W · D (↺) on bars older than it | The Pine reads 1m … 4h intrabars at whatever depth TradingView carries; the free feed carries these depths. Each rung joins where it has calibrated, as in the Pine; `PRG_Ladder` says which ladder a bar read |
| **The daily chart's W conviction rung** (the ↺ fallback) **normalises over 52 weeks**, not 200 | At 200 it needs four years of weekly history (Pragyam's adaptation) |
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
| `BUY_*` / `SELL_*` | the long / short event by age — ▲ capitulation / ▼ distribution, ◆ a RESUME, — none |
| `Side` / `Signal_Kind` / `PRG_Event` | an event on this bar, and which |
| `PRG_Armed` / `PRG_Armed_Age` | the watchlist — in capitulation with value still cheapening — and bars in the cell |
| `PRG_Decl` / `PRG_Decl_Age` | the standing ▲/▼ declaration |
| `PRG_Trace` (`Signal`) | the trace, ±100 |
| `PRG_Hist` / `PRG_Hist_Z` / `PRG_Push` | the histogram, in its own σ, and in five push levels |
| `PRG_CTape` / `PRG_VTape` | the two MTF tapes — the grid's axes |
| `PRG_Conv` / `PRG_Value` | the trace's two ingredients on this chart |
| `PRG_Hedge` / `PRG_Drivers` | the macro hedge applied and the drivers selected |
| `PRG_Div_Seen_*` / `PRG_Abs_Seen` | a divergence / absorbed effort in the last 20 bars (context; no signal reads it) |
| `PRG_Split` / `PRG_Quiet` / `PRG_Settling` | read-with-caution qualifiers |
| `PRG_Stack_OK` / `PRG_Why` | whether the signal set can judge, and if not why |
| `CVG_Action` / `CVG_Why` / `CVG_Units` / `CVG_Held` / `CVG_Lead` | the grid state |
| `Priority_Long` / `Priority_Short` | the ranking keys — a ▲▼ on this bar, then stretch |
| `Signal_Reason` | a plain-language read of the row |
| Risk / flow context | `Vol_Regime`, `Regime_Confidence`, `Change_Point`, `Bar_Delta`, `CVD`, `Delta_Z`, `Buy_Share`, `Absorption_Score` — displayed, never an input |

---

## Architecture Overview

```
sanket.py            ← Streamlit entry point: UI, data + macro-driver fetch, screen routing
engine.py            ← settings, per-symbol features, the snapshot row, ranking, cost gate
pragati.py           ← pragati.pine v9.2: conviction, ladders, trace, histogram, the ◆'s condition, signals() (▲▼ ◆)
samanvaya.py         ← the value engine (Samanvaya, section 4c), carried from Pragyam
cvgrid.py            ← the 3 × 3 conviction-value grid: the state engine, names, units, tones — the ▲▼'s source
pragati.pine         ← the indicator itself, v9.2 — the Pine the port mirrors (archive/: v5, v7, v8)
intraday.py          ← the conviction ladder's lower frames from yfinance (1m … 4h), batched and cached
studies/v9_lab/      ← the v9 audit: look-ahead-free scorers, feature caches, every experiment
charts.py            ← chart builders: the conviction-value map, tone history, correlation heatmap
edge.py              ← measured expectancy: event study, causal drift removal, block bootstrap, power
logger.py            ← structured terminal logging
ARCHITECTURE.md      ← the stack, the evidence, and the design rationale
ui/                  ← theme.py · theme.css · components.py (the Graphite design system)
```

The **regime engine** (HMM + GARCH + CUSUM) and the **order-flow layer** (inferred delta, CVD,
volume profile, absorption) are unchanged and remain context only.

---

## Analysis Modes

1. **Single Date Screener** — Action Dashboard (events by age, each with its grid state, push,
   tapes and evidence) · **Grid** (the 3 × 3 census, the watchlist of capitulations not yet turned, names by
   action) · Signal Strength (the ranking) · System Data (exports, raw frame, Edge Study).
2. **Historical Range** — event breadth by kind, the watchlist's size, grid breadth (build vs cut),
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
Sanket's screener tables stay bespoke, because per-cell glyphs (▲ capitulation / ▼ distribution / ◆), grid
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

**v9.3.0 — the ▲ skips conviction's quiet regime.** A three-round signal search (discovery
2006–19; 2020–26 sealed until the shortlist was fixed; then weekly bars and the Ladder-down window —
[`studies/v9_lab/SIGNAL_SEARCH.md`](studies/v9_lab/SIGNAL_SEARCH.md)) found one change that held
everywhere: fire the capitulation turn only when conviction's σ is not in the bottom fifth of its
history. Daily, sealed 2020–26: worst reading +0.052σ vs +0.046σ, better in 11 of 12 era × scorer
× horizon readings; weekly 2020–26 +0.081 vs +0.054; on Ladder down better on indices, commodities,
FX and stocks. About 5% fewer ▲. No short or exit event held in every reading, so the ▼ is
unchanged. The grid, its units and the ◆ are untouched (`cvgrid` now exposes `cvg_quiet`).

**v9.2.0 — the legacy goes; the Pine's settings say what they move.** v8's TURN is gone from the
Pine (its `▲ ▼ source` option and two windows) and from the port (`signal_source`, the confirm /
dislocation windows, the four TURN gates, the arm loop); `pragati.signals()` alone produces the
▲▼, ◆ and declaration — checked bar for bar identical to v9.1.0 on the defaults and with the ◆ on.
Modules the app no longer reached are removed (`pine_v5.py` — its grid engine moved into
`cvgrid.py` — `cvgrid4.py`, `pine_audit.py`, `research.py`, `trace_study.py`; all in git history,
their reports stay in `studies/`). The Pine's settings are regrouped (impulse threshold k with the
engine; ◆-only knobs marked; one Readout group; two redundant volume-profile switches removed), each
tape sits one point clear of its momentum band, and the tape names carry the reading — not the
frame list — coloured by what the tape says.

**v9.1.0 — the conviction ladder reads DOWN everywhere; Buy · capitulation 4 units.** Measured
head to head on the same daily charts (377 instruments, Nov 2024 – Sep 2026, 1h + 4h rungs): the
grid tied; the ▲ was as good or better on Ladder down in every non-crypto class, significantly on
indices, commodities and FX; tied on stocks. The Pine now defaults to Ladder down, and Sanket and
Pragyam read it from yfinance's intraday history (`intraday.py`), falling back to W · D (↺) on bars
older than it. A walkthrough of the grid on real data: Buy · capitulation and Exit · distribution
hold as stated in every era, the other cells read ≈ 0; a 200-day trend axis and neutralised units
were tested and rejected; **Buy · capitulation 3 → 4** beat 3 in every era in Pragyam's allocator
and in the lab, and is adopted. The weekly grid's momentum sign now follows its ladder.

**v9.0.0 — Pragati v9: a fresh audit, the measuring stick fixed, the signals moved to the edge.**
Everything in the indicator was re-audited from scratch — 380 instruments, three eras, daily and
weekly — with a rule that a change must hold in all three eras
([`studies/pragati_v9_audit.md`](studies/pragati_v9_audit.md)).

- **The v8 measuring stick was biased.** Demeaning returns by the era's own mean makes random-walk
  momentum score as reversion (up to −0.04σ at 40 bars) and flatters events that cluster in names
  whose era went badly. v9 scores causally — and so does the app's Edge Study now.
- **The readings describe; they barely forecast.** Conviction, value, the trace, its histogram and
  both tapes carry almost no timing information of their own. **The capitulation state carries the
  one robust edge** (+0.06 to +0.08σ over 10–20 bars, every era, daily and weekly, every engine
  variation tried).
- **▲ is now the capitulation turn; ▼ distribution** — both read from the grid. v8's TURN had
  faded to nothing after 2020, was negative on weekly bars and worked only after a capitulation;
  it stays as `signal_source="turn"`. **R divergence off by default.**
- Every claim in the Pine, the app and the docs rewritten to what was measured; nothing tuned —
  nine engine ablations kept the defaults.

**v8.4.0 — Pragati v8, the final version: v5's grid with measured units.** The user's v5 Pine
was audited the same way v7 was and the two compared head to head. Their signals tie; v5's
graded 3 × 3 with four cells re-weighted beats v7's 4 × 4 after 2018 (paired +0.017σ at h 10,
+0.020σ at h 20). v8 = v5 + those units, ◆ RESUME off by default, Ladder up by default, the OI
gold cast on (colour only). Pragyam's book adopts the same units after passing its own allocator
test in both eras. See the CHANGELOG for the full list.

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

**v5.0.0 — thesis replacement driven by a reproducible harness.** `research.py` (removed in v9.2; in git history)
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

*Sanket v9.1.0 · Pragyam Family · Built by [@thebullishvalue](https://github.com/thebullishvalue)*
