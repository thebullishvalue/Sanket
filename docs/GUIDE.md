# Sanket · Guide

How Sanket reads a name, what each signal set and the grid mean, how names are ranked and how
the Edge Study measures them. The [README](../README.md) is the overview; this is the detail.

## How Pragati Reads a Name

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

## The Stack

| Layer | Module | Carried from |
|:---|:---|:---|
| Value — rich or cheap against the macro drivers | `samanvaya.py` | Pragyam's port, vectorised; Weekly added |
| Conviction, the trace, the histogram, the tapes | `pragati.py` | Pragyam's conviction port, extended |
| The signals — proven-state triangles | `ledger.py` | Pragyam (`pragyam.pine`), ported bar for bar |
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

## The Signals

Three sets, all fired on every screen and shown side by side, all read from **Pragyam's engine** (`ledger.py`,
ported bar for bar from `pragyam.pine` and checked against its reference on 151 instruments:
identical states, z-scores, cones, ladder and signals on every bar). Pragati's own signal set — the
grid's capitulation / distribution, the ◆ RESUME and the watchlist — is not computed.

| Set | ▲ fires on one closed bar when… | ▼ | Per 1,000 bars (151 instruments) |
|:---|:---|:---|:---|
| **A · proven** | the name enters a state its own record has proven positive | proven negative | 0.2 |
| **B · flip** | the surprise column flips positive · the close is above the midline · risk is calm · risk is not rising · the stretch ladder leans up | mirrors | 25 |
| **C · flip + trend** | set B's ▲, and the trend tape up | set B's ▼, and the trend down | 14 |

Measured against same-day peers on the research set and 572 fresh stocks (`studies/v13_lab`), no
set picks names that beat their peers over 10 bars by a margin worth trading: B and C sit within
about ±0.06σ on every dataset, and A is too rare to judge (+0.09 to +0.12 on the research set,
−0.05 to −0.07 on the fresh stocks). v10.3 briefly reversed C and added a set D on the strength of the old
drift-free score; v13 showed that score produced the effect, and v10.4 rolled both back.

What each term means, exactly:

| Term | Definition |
|:---|:---|
| Surprise column | the 10-bar move over the forecast made 10 bars ago, in σ. It **flips** when it is below zero on one bar and at or above zero on the next (negative mirrors) |
| Midline | the middle of the landing band — the cone drawn 10 bars ago |
| Risk calm | risk's rank in its last 500 bars in the bottom third (the inner risk tape at its lightest) |
| Risk not rising | risk up no more than 10% over 10 bars (the outer risk tape grey) |
| Stretch ladder leans up / down | the mean, over the 5-, 10-, 20- and 40-bar horizons, of each surprise as a fraction of its own band is ≥ 0 / < 0 (the outer trend tape green / red, any shade) |
| Trend up / down | the move over 63, 126 and 252 bars in units of risk is ≥ 0 / < 0 (the inner trend tape green / red, any shade) |
| Proven (set A) | nine states (trend × where the close landed against the cone); each scored on the name's own next-10-bar outcomes, shrunk toward no edge, proven past two standard errors |

**READY** (sets B and C): every condition of a side in place except the trigger — the flip and
the close crossing the midline, which arrive together. The next flip that way fires. The State
column, the Pulse and the Historical Range show how many names are ready.

No set has a cooldown; a tape still warming blocks B and C. The last event stands as the
declaration. **Warm-up:** set A can prove a state from bar ~563; sets B and C fire from bar ~752,
when risk's rank has 500 readings. Daily (~1,150 bars) leaves about 1.5 years in which B and C can
fire; the Weekly fetch was deepened to ~880 bars for them.

**What was measured before shipping.** Set A: on random walks it calls a state proven on 0.4% of
bars, and pooled across names no state reached |t| = 1.3 outside crypto — a triangle says the
name's own past has proven its state. Sets B and C: the nearest tested versions (the Indicator
repo's `research/flips.txt`, which did not include the risk-not-rising condition) measured +0.7
and +0.9σ × 100 over 10 bars with t below 1, and changed sign between eras. The Edge Study measures
all three on your universe; read it before leaning on any of them.

**Divergence is off by default.** The Pine's R (regular) and H (hidden) marks remain available.
Re-measured, R read about zero in every era on daily and weekly bars and H was negative or mixed;
the source's claim that R "ranked first" did not reproduce.

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
trends. Full report: [`studies/pine_audit.md`](../studies/pine_audit.md).

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
mid and small caps. The grid and the hold window are still computed and shown; they describe
a name, they do not order the list. Reports: `studies/`.

**Re-checked in v9** without look-ahead (that study removed each name's drift inside the era): the
cross-sectional rank IC of the stretch agrees with reading it as reversion on NSE before 2020
(t ≈ +2 to +4) and reads about zero since. The band on top is now the ledger's ▲▼ (v10) — a state
proven on the name's own record; no study has placed it on top of the stretch.

## Edge Study: Expectancy Measured on Your Universe

`edge.py` runs an event study on your symbols through **the exact engine call the screener
makes** (`engine.compute_frame`), so every warm-up applies to the study by construction. Six
slices — sets A, B and C, each side alone — measured on the same symbols and dates, so they can be
compared directly. Every set carries its own verdict wherever it appears; none borrows another's.

Each step kills one way of fooling yourself (step 2 changed in v9):

| # | Step | The failure it prevents |
|:--|:---|:---|
| 1 | Event study at the declared horizon (enter the bar after the signal, hold 10) | Measuring a continuous form nobody trades |
| 2 | **Measured against peers** (v10.4): the same-day mean h-bar return of the other names on screen (at least 5) is subtracted; drift-free (the symbol's own trailing mean) is kept as a second column | Every long signal in a bull market prints a profit — beta, not edge. (Up to v8.4 the mean was taken inside the era — a look-ahead that flatters reversal events) |
| 3 | Vol normalisation by the symbol's own trailing σ | FX, bond ETFs and small-caps on incomparable scales |
| 4 | Sign folding | The two sides on opposite conventions |
| 5 | **Block bootstrap over dates** | Overlapping returns and a correlated cross-section inflating significance |
| 6 | Cost in the same vol units (`bps/1e4 ÷ σ_h`) | The same bps costing 4× more on a low-vol name |
| 7 | **Power stated**: `n_eff`, and a minimum detectable effect | "No edge" reported from a test that could never have seen one |

Verdicts: `CONFIRMED` · `GROSS ONLY` · `DISCOVERY ONLY` · `NO EDGE` · `ANTI-PREDICTS` ·
`UNDERPOWERED` (the MDE exceeds 0.036, the largest effect this indicator family has shown
anywhere). **Expect UNDERPOWERED often**, set A above all — it is rare — and the app says "we
could not tell" rather than "there is no edge".

It fetches ~15 years once a day per universe (80-symbol fixed-seed sample above that), streams
in chunks of 20, and reuses the measurement until the date rolls.

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

## Outputs

Per symbol, on each run:

| Column | Meaning |
|:---|:---|
| `turn_buy` / `turn_sell` | ▲ / ▼ of any set on this bar |
| `A_Buy` … `C_Sell` | each set's ▲ / ▼ on this bar |
| `PRG_Buy_Sets` / `PRG_Sell_Sets` | the sets that fired each side on this bar: `A`, `B`, `B·C`, `A·B·C` … |
| `PRG_Ready` / `PRG_Ready_C` | +1 / −1 when every condition of set B's (set C's) side is in place but the flip |
| `LG_U` / `LG_Flip` / `LG_Mid` / `LG_Above_Mid` | the surprise column, its flip, the midline, and the close against it |
| `LG_Risk_Rank` / `LG_Calm` / `LG_Risk_Mom` / `LG_Rising` | risk's rank, calm, its 10-bar change, and rising |
| `LG_Ladder` / `LG_Lad_Up` / `LG_Lad_Dn` | the stretch ladder's lean and the rungs outside |
| `BUY_*` / `SELL_*` | the long / short event by age — ▲ / ▼, — none |
| `Side` / `Signal_Kind` / `PRG_Event` | an event on this bar (kind PROVEN) and its label |
| `PRG_State` | set A: PROVEN ▲ / ▼, IN PROVEN + / −, LEARNING, UNPROVEN, WARMING UP · sets B, C: ▲ B … ▼ C, READY ▲ / ▼, WAITING, WARMING UP |
| `LG_State_Name` / `LG_N` / `LG_Edge` / `LG_Post` / `LG_Z` / `LG_Proven` | the state, its samples on this name, its edge raw and shrunk, its z, and whether it is proven |
| `LG_Sigma` / `LG_Trend` | the risk forecast per bar and the trend in σ |
| `PRG_Decl` / `PRG_Decl_Age` | the standing ▲/▼ declaration |
| `PRG_Trace` (`Signal`) | the trace, ±100 |
| `PRG_Hist` / `PRG_Hist_Z` / `PRG_Push` | the histogram, in its own σ, and in five push levels |
| `PRG_CTape` / `PRG_VTape` | the two MTF tapes — the grid's axes |
| `PRG_Conv` / `PRG_Value` | the trace's two ingredients on this chart |
| `PRG_Hedge` / `PRG_Drivers` | the macro hedge applied and the drivers selected |
| `PRG_Div_Seen_*` / `PRG_Abs_Seen` | a divergence / absorbed effort in the last 20 bars (context; no signal reads it) |
| `PRG_Split` / `PRG_Quiet` / `PRG_Settling` | read-with-caution qualifiers |
| `PRG_Stack_OK` / `PRG_Why` | whether both tapes are ready for the grid, and if not why |
| `CVG_Action` / `CVG_Why` / `CVG_Units` / `CVG_Held` / `CVG_Lead` | the grid state |
| `Priority_Long` / `Priority_Short` | the ranking keys — a ▲▼ on this bar, then stretch |
| `Signal_Reason` | a plain-language read of the row |
| Risk / flow context | `Vol_Regime`, `Regime_Confidence`, `Change_Point`, `Bar_Delta`, `CVD`, `Delta_Z`, `Buy_Share`, `Absorption_Score` — displayed, never an input |

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

## UI System

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
| Appearances | **Paper** (light, default, for reading and print) and **Slate** (dark) |
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
Sanket's screener tables stay bespoke, because per-cell glyphs (▲ / ▼ and the ledger's state), grid
states, push levels and hold counters are not expressible as a DataFrame — but they draw their typeface,
row height, header and tokens from `ui.components.table_shell_css`, so the only thing that
differs from a generic table is the content of a cell.

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
