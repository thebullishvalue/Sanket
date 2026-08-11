# CHANGELOG
### Sanket — Close-Location Reversal (CLR)

All notable changes to the **Sanket** platform are documented here. Sanket is part of the **Pragyam Product Family** by [@thebullishvalue](https://github.com/thebullishvalue).

Format: `[version] · date — release title`

---

## [v6.3.0] · 2026-08-11
### The Study Runs On Every Run · Nothing Left To Configure

**The Edge Study is no longer opt-in.** Whether the rule carries an edge on the universe in front
of you is not an optional extra — it is the thing that tells you whether to believe the signals —
so it is measured as part of every run rather than hidden behind a checkbox. The
"🔬 Edge Study" expander and its "Measure edge on the next run" control are gone; the sidebar now
has **no controls at all**.

**Cadence: measured on every run, computed once a day.** `ensure_edge_study` reuses a same-day
measurement and re-measures once the date rolls. This is not a compromise on "every run" — the
study reads completed bars and needs forward returns, so it excludes the forming bar. Two runs on
the same calendar day therefore read identical data and must produce a bit-identical answer;
re-measuring inside a day is a 15-year fetch for a result already held, and on a shared cloud IP a
good way to earn a yfinance rate-limit mid-screen. The date rolling is exactly when new bars can
change the answer, and that is when it re-measures.

A failure is recorded for the day rather than retried on every click, and the run proceeds on the
last measurement if there is one, or "not measured" if there is not.

**One progress bar per click.** When the study actually measures it takes the head of the bar
(0→35%) and the analysis renders into the tail of the *same* bar; when the measurement is reused
the analysis owns all of it. Correlation keeps its own multi-phase bar rather than nesting two
offset schemes.

**Fixed: the measured edge was leaking into conviction through the cost gate.** `cost_ok` had
begun answering from the study's measured *net* (gross minus cost), which fails on any universe
that measures no edge — so a `NO EDGE` verdict silently halved conviction. That is precisely the
hidden multiplier this design exists to refuse. The gate now asks a question about **cost only**:
is this universe's measured cost *charge* (`cost_bps/1e4 ÷ σ_h`) smaller than
`LARGEST_KNOWN_EFFECT`, the most this signal has ever been worth on any asset class? If the cost
exceeds that ceiling no plausible version of the edge survives it; if it does not, the verdict is
irrelevant to conviction. Regression-tested: a measured `NO EDGE` study now leaves conviction
**bit-identical** to the unmeasured case, while a prohibitive cost charge still gates.
`LARGEST_KNOWN_EFFECT` moved to `engine.py` as the single source of truth (`edge.py` aliases it).

---

## [v6.2.0] · 2026-08-11
### Named For What It Measures · Quieter Surface

**The engine has a real name.** It is **Close-Location Reversal (CLR)** — in the code, the
column names and the UI. The source indicator titles itself "SB v8 — CLOSE-LOCATION REVERSAL";
only the descriptive half was ever meaningful. "SB v8" was a family tag for a lineage of
session-breadth indicators whose core measure tested flat at p_bonf = 1.00 and whose surviving
variable turned out to have the opposite sign — a label naming a premise this engine refutes.
It is retained nowhere but the source filename.

Renamed throughout: `CLR_Z`, `CLR_CLV`, `CLR_State`, `CLR_Hold_Dir/Age`, `CLR_Score`,
`CLR_Rank_Pct`, `CLR_THRESHOLD/HORIZON/COST_BPS/Z_LOOK_*`, `add_clr_features`, `CLRSettings`.
The engine name lives in one place (`ENGINE_NAME` / `ENGINE_CODE`) so it cannot drift between
screens. The analysed-frame cache tag moved `sbv8` → `clr1`, since the column schema changed.

**Parameter sliders removed.** The threshold, hold horizon and round-trip cost are no longer
exposed. Every one is a measured plateau from the source study, so a slider only invited fitting
them to whatever universe was on screen — precisely what would destroy the credibility of the
measurement beneath it. The values still appear, as one read-only `Setup` row, because a reader
needs to know what fired. Defaults are unchanged.

**Two message boxes removed.** The sidebar's "Measured on your data: …" panel and the Action
Dashboard's measured-expectancy banner are gone. The verdict lives in the Engine Status card and,
in full, in **System Data ▸ Edge Study** — one place for the status line, one for the report.

**Engine Status card consolidated: eleven rows → six.** Buy interval, sell interval, power
(`n_eff` + minimum detectable effect), sample (symbols · years · independent names), setup, cost
gate. Dropped: the redundant sell-verdict row, the separate trigger and z-lookback rows (merged),
the fire rate and the static "entry: next session open" (both belong in the report, not the
status line).

**Fixed: the card lagged a measurement by one interaction.** The sidebar renders before the
analysis executes (single-pass render), so a study measured on a click did not appear until the
next one. The card is now painted into a placeholder and repainted when the study completes —
the same pattern the retired alpha-health passport used, reinstated for the right reason.

**Fixed (edge.py): `verdict()` could report a confirmed holdout as UNDERPOWERED.** It keyed the
underpower check off the full-era result alone, so if that slice dropped out — every symbol
failing the in-era baseline minimum, say — a holdout whose CI excluded zero was still reported as
underpowered. It now requires *every* measured era to be underpowered, and picks its reference
from whichever era exists rather than dereferencing a possibly-absent one.

**Tests.** The two AppTest suites now use separate disk-cache directories: the study cache is
shared across processes by design, so suites running concurrently were seeing each other's
studies. Assertions that targeted whole-page text now target the specific card, and the
threshold-invalidation test moved from driving a (now absent) slider to asserting the cache-key
contract directly.

---

## [v6.1.0] · 2026-08-10
### Expectancy Is Measured, Not Hardcoded

v6.0.0 shipped the SB v8 signal with the source study's per-class expectancy wired in as a
lookup of eight frozen constants, used to scale conviction. That was the weakest part of the
design and it is now gone from every operative path.

**Why it had to go.** Eight numbers from someone else's 39 instruments could not cover a
universe the study never touched (NSE F&O single names, NSE thematic ETFs, crypto); they applied
an *asset-class* claim to *instrument-level* decisions; they could not report that the edge had
stopped working, while the study's own headline is that it decayed 4x since the 1990s; and they
were unfalsifiable in-product — you could not check them against your own data. A screen that
announces "this asset class is unproven" on authority it never earned is not institutional
fidelity, it is inherited assertion.

**New module: `edge.py`.** Measures SB v8's out-of-sample expectancy on the user's own symbols,
at the **pre-declared** parameters, using the methodology that makes the source numbers
credible. Seven steps, each killing one specific way of fooling yourself:

1. **Event study at the declared horizon** — enter the bar after the signal closes, hold
   `horizon` (the study's EXEC-B). Not a continuous IC: a continuous position on this signal
   nets -0.48 Sharpe, so measuring that form answers a question nobody trades.
2. **Drift removal, within era** — subtract each symbol's own mean forward return, computed
   inside the era being measured. Without it every long signal on an equity universe in a bull
   market prints a profit and you have measured beta. Computing it *within* era also stops the
   discovery period's drift leaking into the holdout.
3. **Volatility normalisation** — divide by the symbol's own forward sigma, so FX, bond ETFs and
   small-cap equities land on one scale.
4. **Sign folding** — a buy scores positive when it beat the symbol's drift, a sell when it fell
   short, so both sides read "positive = the signal was right".
5. **Block bootstrap over DATES** — resampling contiguous blocks of whole dates handles the
   h-bar forward-return overlap (blocks) and the within-date cross-sectional correlation (whole
   dates) in one move. Vectorised via per-date sums, so 2000 resamples is milliseconds.
6. **Costs charged in the same units** — `cost_bps/1e4 / sigma_h`. This is why the edge dies on
   low-volatility instruments: 3bp against a 4% 10-day sigma is 0.008 vol units, against a 1%
   sigma it is 0.030. A per-class cost table could not express that.
7. **Power stated, never assumed** — `n_eff = (n_dates / horizon) x participation_ratio`, where
   the participation ratio is the eigenvalue-based effective number of independent names
   (`(sum L)^2 / sum L^2` of the correlation matrix), **measured from the data**. A minimum
   detectable effect follows from it.

**The CI decides, not a p-value.** Verdict ladder: `CONFIRMED` (holdout CI excludes zero and
survives costs) · `GROSS ONLY` · `DISCOVERY ONLY` · `NO EDGE` · `ANTI-PREDICTS` ·
`UNDERPOWERED`. That last rung is deliberate: when the MDE exceeds the largest effect ever
measured for this signal the test is vacuous, so no verdict is claimed. "We could not detect an
edge" and "there is no edge" are different statements.

**Conviction lost its expectancy term.** Now `|z| magnitude x cost gate`, and nothing else:

```
Conviction = clip(0.30 + 0.70·clip(|z|/3, 0, 1)) x cost_factor
```

The measurement is **reported, never applied** — a universe that measures `NO EDGE` still fires
every signal at full conviction and says so on screen. `engine.compute_ranking` no longer takes
an `iclass` parameter at all. The cost gate reads the study's measured net when one exists and
falls back to the source study's pooled ~7bp breakeven otherwise, reporting which basis it used
(`engine.cost_basis` → `measured` / `pooled prior (~7bp)`).

**Built for a 1 GB shared container.** The study needs ~15 years of history — the power
arithmetic is unforgiving: resolving an effect of `e` needs `n_eff ~ (1.96/e)^2`, so the
screener's own 900-day window resolves only ~0.10, i.e. nothing but the single largest effect
the source study ever found. Fetching 15y naively for a large universe OOMs. Three choices avoid
it, and the numbers are measured, not asserted:

| Approach | Peak memory, 80 syms x 15y | Projected, NIFTY 500 |
|:---|:---|:---|
| **Streaming + lean (shipped)** | **31 MB** | **~31 MB — flat in universe size** |
| Raw OHLCV frames held for the universe | 14 MB | ~90 MB |
| Full analysed panel (the naive version) | 556 MB (6.8 MB/symbol) | **~3,400 MB — hard OOM** |

The middle row is why "just hold the raw data" is not the answer either: it is affordable at 80
symbols but the *analysed* panel is what a naive implementation actually builds, and that scales
at 6.8 MB per symbol. Streaming is flat in universe size, which is the property that matters.

  * **Lean** — close-location z and forward returns only. No volume profile (a Python double
    loop, the app's slowest path), no regime engine, no order flow.
  * **Streaming** — chunks of 20 symbols fetched, reduced to event tuples, then released. What
    accumulates is ~44k event rows, not a panel.
  * **Sampled** — universes above 80 symbols are sampled with a seed derived from the symbol set
    (reproducible, and not biased toward one alphabetical/sector slice). Nearly free
    statistically because the participation ratio saturates well below 80. Reported, not hidden.

**Opt-in, then cached.** A checkbox in sidebar ▸ Edge Study runs the measurement as part of the
next RUN; the result caches to session state and, best-effort, to disk (`.sanket_cache/`, which
is ephemeral on Streamlit Cloud — a miss there is normal, never an error). Deliberately not
automatic: it is a deep fetch, and an unprompted one from a shared cloud IP is a good way to get
rate-limited mid-screen. A failed study is never fatal — the screen runs and the UI reads "not
measured".

**Changing a parameter invalidates the study.** The cache key includes `(z_look, thr, horizon)`,
so a study measured at 1.5σ is never served for a 2.5σ screen.

**UI: verdicts replace assertions.**
- Sidebar Engine Status shows the **measured** buy/sell edge with intervals, the power (`n_eff`
  and MDE), coverage, the participation ratio and the fire rate — or an honest "not measured
  yet" with the control to fix it.
- The Action Dashboard banner states the measured verdict for both sides, and when the verdict
  is not `CONFIRMED` it adds, in as many words, that signals still fire because this is a
  measurement and not a filter.
- New **Edge Study** panel in System Data: every side x era row with edge, CI, net, hit rate,
  events, dates, `n_eff` and MDE, plus the method and the labelled reference prior.
- The old `Class Edge` metric cards and the hardcoded scope warning are gone.

**Statistical validation** (`scratchpad/stats_test.py`, six properties):
- **Null calibration** — 12/12 trials cover zero on pure noise; zero false "edge" verdicts.
- **Signal recovery** — planted edges of 0.05 / 0.10 / 0.20 recovered as 0.058 / 0.105 / 0.197,
  each inside its CI.
- **Drift immunity** — with +16%/yr drift and zero planted signal the measurement reads +0.005
  (`NO EDGE`), while the *same events* without drift removal read **+0.178, 33x larger**. Step 2
  is load-bearing.
- **Participation ratio** — 39.1/40 for independent names, collapsing to 1.5/40 at beta 0.9.
- **Overlap honesty** — the block-bootstrap interval is **2.3x wider** than a naive iid
  resample, i.e. the dependence correction genuinely binds.
- **Underpower honesty** — a thin sample returns `UNDERPOWERED` with the MDE explanation rather
  than guessing.

**Also:** the universe→symbols dispatch, previously duplicated across the screener, the range
harvest and correlation, is now one `resolve_universe` function — the study must screen exactly
the symbols the screen shows, or the measurement would describe a different set.

---

## [v6.0.0] · 2026-08-10
### One Screening Condition — SB v8 Close-Location Reversal

A deliberate subtraction. The system now runs **one** signal, ported from
[`sb_v8.pine`](sb_v8.pine): the z-score of where price closes inside its own bar range. Two
events, and nothing else.

```
sb_clv = ((close - low) - (high - close)) / (high - low)
SB_Z   = z-score of sb_clv over the trailing 252 bars (52 weekly)

SB_Z < -1.5σ  →  ▲ BUY   (green triangle · weak close · fade up)
SB_Z > +1.5σ  →  ◆ SELL  (yellow diamond · strong close)
```

The sign is the finding: **a strong close predicts weakness**, so the fade of a weak close is the
buy. Measured on 39 instruments / 251,200 daily bars / 1993-2026, 66 Bonferroni-corrected tests,
holdout 2014-2026 opened exactly once. Discovery IC -0.0634 (z -7.96, p_bonf 1.2e-13) — the
strongest result in that study by four orders of magnitude, and negative.

**Removed — the momentum engine.** The 12-1 cross-sectional momentum ranker, the reversion
entry-timing overlay, `VOL_REGIME_MOM`, and the whole `_mom_*` / `_rev_*` feature family.
`engine.py` was rewritten from scratch around SB v8.

**Removed — Set A / Set B.** Both entry screeners are gone, along with `compute_signal_sets`, the
`long_cond`/`short_cond`/`*_comp` booleans, and the `LA_`/`SA_`/`LB_`/`SB_` age columns. They are
replaced by `BUY_Today…BUY_5d` and `SELL_Today…SELL_5d` — the same age-bucketing UI, now driven by
the single condition.

**Removed — the entire Intelligence layer.**
- The **Intelligence tab** (Alpha-Health Monitor) and the signal-aging reference.
- The **alpha-health monitor** itself: `_measure_trailing_ic`, `_ensure_alpha_health`,
  `_edge_state`, the Engine Status passport (`_passport_status_html` / `_refresh_passport` /
  `_render_model_passport_sidebar`), `engine.alpha_health`, `engine.cross_sectional_ic`, and the
  `alpha_health_*` / `opt_results` session state. Conviction no longer scales by a live IC reading.
- **Layer-2 Signal Intelligence** — `Intel_Confidence`, `Intel_Stars`, `Intel_Source`, and the
  ◆/◇ Intel cells on every table.
- **Layer-3 Meta Intelligence** — `Meta_Score`, `Meta_Tier`, `Meta_Source`, `Meta_Reason` and the
  **Meta Filter** (Off/Dim/Hide + threshold slider).
- The **Context / Entry** aging machinery: `_context_status`, `_cached_conf_series`,
  `_fire_bar_metrics`, `_active_model_sig`, and the `intel_windows` / `intel_fire_cache` session
  caches. (An `_entry_status` move-exhaustion read survives, rebuilt to key off the z that fired.)

**Consequence:** the pre-screen harvest pass is gone. A Single-Date run is now a single pass over
the universe with one progress bar, instead of a 0→40% edge-measurement harvest followed by a
40→100% screen.

**The instrument class is wired to the universe selector.** The Pine exposes instrument class as an
input because the edge does not hold everywhere; `engine.instrument_class` derives it from the
selected universe, so the expectancy the UI reports always matches the asset class on screen:

| Universe | Class | OOS edge | Hit | Established? |
|:---|:---|:---|:---|:---|
| US Indexes | US index / ETF | **+0.121** | 57.5% | **yes** |
| — | US sector ETF | **+0.068** | 54.9% | **yes** |
| India Indexes · ETF Index | India index | +0.089 | 51.9% | no (n=239, CI includes zero) |
| Global Indexes | International equity | +0.003 | 53.2% | no |
| Commodities | Commodity | +0.028 | 51.0% | no |
| Currency | FX | +0.035 | 51.9% | no |
| Global Macro | Rates / Credit | -0.003 | 51.1% | no |
| Crypto | Other / unknown | 0.000 | 50.0% | no |

The sidebar Engine Status card shows the class, its edge and hit rate, the trigger, the z-lookback
and the cost gate; a **scope warning banner** appears on the Action Dashboard whenever the active
class is not holdout-confirmed.

**New conviction model.** `|z| magnitude × class expectancy × cost gate` — and nothing else:

```
Conviction = clip(0.30 + 0.70·clip(|z|/3, 0, 1)) × class_factor × cost_factor
class_factor: 1.00 established · 0.75 CI-includes-zero · 0.55 nominally positive · 0.40 zero/negative
cost_factor:  1.00 within the class breakeven (~7bp pooled, ~10bp established), else 0.50
```

Deliberately absent: no per-name vol factor, no regime factor, no live-IC scaling. The Pine has no
such terms. Conviction is labelled a relative weighting, not a probability, in every tooltip.

**The SELL side ships with its caveat attached.** The source indicator labels the strong-close side
CAUTION rather than a short entry: its drift-free holdout was +0.0094 with a CI of
[-0.030, +0.052], i.e. it did **not** confirm out of sample. Sanket surfaces it as a sell signal as
configured, and that caveat travels with it into the Action Dashboard tab description, the Signal
Reference cards, the Excel legend, and `Signal_Reason`.

**Warmup is a refusal, not a fudge.** A symbol needs `z_look + 2` bars before it can carry a signal
(the Pine's own `dBars < zLook + 2` guard). Shorter histories are excluded from the screen with a
"warming up" count surfaced in the run stats and a specific empty-state message when the whole
universe is too short.

**Four Pine inputs exposed** in a sidebar expander with the measured rationale in each tooltip:
signal threshold (σ), hold horizon (bars), round-trip cost (bps). The z-lookback follows the
timeframe (252 daily / 52 weekly) and the instrument class follows the universe — neither is a knob.
All three parameters are folded into the analyzed-frame cache signature, so changing the threshold
invalidates cached frames rather than serving stale conditions.

**UI rebuilt around the two events.**
- Signal colours now match the indicator's own markers (`#00E676` buy triangle, `#FFA726` sell
  diamond), so the app and a TradingView chart read the same.
- Tables gained **Close-Loc z** (coloured by SB v8 meaning, not price direction), **Side** (▲/◆/—),
  **Conv**, and **Hold** ("day 3/10" through the measured window); they lost Intel, Meta and Context.
- Aged signals report **the z that fired them**, not today's — via a per-symbol `Z_Hist`.
- Action Dashboard: `▲ BUY Signals by Timing` / `◆ SELL Signals by Timing`, each with a
  strengthening/weakening trend read computed on |z|.
- Historical Range: buy/sell **breadth** charts and an `Avg Fired |z|` metric replace the L/S counts;
  forward-return labels moved to the SB v8 horizons (`Ret_1b/5b/10b/21b`).
- Correlation: `Trade Intelligence` → **Confluence Setups**; confluence is now
  `|Corr| × normalised |fade score| × (0.5 + 0.5·Conviction)`.
- Excel legend split into **THE SIGNAL** and **CONTEXT ONLY** blocks — the distinction matters more
  than the ordering.

**Weekly is flagged as an extrapolation.** The study was daily; `z_look` becomes 52 on Weekly (one
year, the closest structural analogue) and the Engine Status card says so.

**Retained as displayed context, never as signal inputs:** the regime engine (HMM + GARCH + CUSUM,
per-name risk context — the "Regime Intelligence Engine" header was renamed to stop implying
otherwise), the order-flow layer (inferred delta, CVD, `Delta_Z`, absorption, volume profile), and
the flow zone. `Delta_Z` and `SB_Z` are cousins, not duplicates: the former z-scores the
*volume-weighted* close location, the latter the raw close location. Only the latter is the signal.

**`research.py` is now legacy.** It documents the cross-sectional momentum study the *previous*
engine was built on and does not validate SB v8. The evidence for this engine is the
[`sb_v8.pine`](sb_v8.pine) header.

---

## [v5.1.0] · 2026-07-23
### Rebuilt Entry Screeners (Set A/B) + Data-Calibrated Intelligence Engine

Follows v5.0.0's momentum thesis with a ground-up rebuild of the two live signal sets and an
empirical recalibration of the conviction engine — every change driven by the `research.py`
harness, not priors.

**New Set A / Set B — long-only, edge-validated entry screeners.** The old delta-divergence
(Set A) and clamp-cross (Set B) signals carried no tradeable edge and are retired. Across two
out-of-sample sweeps (~130 candidate conditions, ranked by train[≤2021]/test[≥2022] consistency +
per-year stability + beat-the-momentum-baseline), two survivors ship:
- **Set A · Momentum Pullback-Resumption** — an uptrend (Close>SMA200, 12-1 momentum>10%) that
  dips below its SMA20 and closes back above it. +0.25% vs universe @5d, positive in both OOS halves.
- **Set B · Gap-and-Go Continuation** — an uptrend gaps up ≥1.5%, holds it (Close>Open), finishes
  near its 20-day high. **+0.99% vs universe @5d, t~2.2, positive in 9 of 11 years** — the strongest
  signal in the system, near-orthogonal to Set A. Upgraded from a weaker volume-surge candidate.

Both are **long-only** (the short side of every tested event anti-predicted) and framed honestly as
*entry-odds screeners*, not standalone portfolio alpha. Two structural findings from the sweeps:
momentum-*ignition* (buy strength) beat *accumulation* (buy weakness), and **every inferred-delta
condition failed** — confirming, a third time, that the OHLC-proxy delta stays pure context.

**Intelligence engine recalibrated from data.** `VOL_REGIME_MOM` was a prior that damped high-vol
names. A calibration study showed per-name high-vol momentum names return *more* (that's beta, not
edge) and per-name vol does not predict the cross-sectional edge — so the weights were **flattened
to near-neutral** (`{1.0, 1.0, 1.0, 0.85}`, keeping only a mild extreme-vol risk trim). The real
edge-timing is market-wide (momentum IC +0.056 calm vs +0.007 turbulent) and is already owned by the
**alpha-health monitor**. HMM / GARCH-regime / CUSUM remain display context and never touch the rank.

**System-wide copy + version alignment.** Terminal logging, progress labels, headers, footer, system
cards, tooltips, and legends updated from "reversion" to the momentum + Set A/B language. `engine.py`
untouched at the rank (momentum IC re-confirmed +0.025/+0.032/+0.048). Version → **v5.1.0**.

---

## [v5.0.0] · 2026-07-23
### Thesis Replacement — Reversion → 12-1 Momentum, and a Reproducible Research Harness

**The core edge changed, because a new harness proved the old one wasn't tradeable.** The prior
core (cross-sectional reversion) was accepted on a docstring; nothing in the repo could reproduce
its claimed IC. This release makes evidence a first-class artifact and follows it wherever it led.

**Added — `research.py`, a reproducible point-in-time research harness.** Pulls corporate-action-
adjusted OHLCV, builds candidate cross-sectional signals under a strict no-lookahead contract, and
reports rank-IC + t-stat, IC decay across horizons, cost-aware **non-overlapping** quantile
backtests, turnover, per-year stability, and a **shuffled-null control** (which returns IC ≈ 0,
confirming the harness doesn't manufacture edge). Run `python research.py` to regenerate every
number in the docs. This is the permanent spine — no more edges-in-docstrings.

**Finding — reversion is a cost trap.** On 100 NIFTY-100 names, 2016–2026 (adjusted): reversion
rank-IC is real (+0.029…+0.031 @1–2d, t ≈ 7, positive every year) but its edge lives at a 1–2 day
horizon with ~80% turnover, so after realistic costs the L/S book is **net negative** (≈ −23%/yr at
25 bps). It predicts; it cannot be harvested.

**Finding — 12-1 momentum survives costs.** The same harness found the edge: momentum IC *grows*
with horizon (+0.025 @5d → +0.032 @21d → +0.048 @63d), so a monthly long-only book turns slowly
(~21%) and clears costs — **~+6%/yr excess over the equal-weight universe, excess Sharpe ~0.6**,
robust to 25 bps. Caveats stated everywhere: the ~30% absolute is mostly beta (abs Sharpe 1.34 ≈
benchmark 1.29), momentum **decayed 2024–2026**, and the current-constituent universe inflates it
(survivorship). The old engine had deleted momentum for "anti-predicting" — a horizon error; it was
measured at short horizons where reversion dominates.

**Changed — `engine.py` rebuilt around momentum.** `add_alpha_features` (12-1 + 6-1 momentum;
`add_reversion_features` kept as an alias) and a `compute_ranking` that ranks on the robust-z of
momentum. Reversion is demoted to an `Entry_Timing` overlay (favour momentum longs that have pulled
back; ±10% conviction nudge, never the rank). `VOL_REGIME_MOM` **damps** momentum in HIGH/EXTREME
vol (where it crashes) — the inverse of the old reversion regime map. The **output-column contract
is unchanged**, so the UI renders without edits (`Rev_Score` now carries the momentum score).

**Changed — `sanket.py` wiring.** `_MAX_DAYS_BACK` 500 → 900 (12-1 momentum needs ~273 bars of
warmup); the `_mom_12_1` / `_mom_6_1` features flow through the live results dict **and** the
alpha-health harvest panel; `HOLD_HORIZONS` → `[5,10,21,42,63]`; the alpha-health monitor now reads
`Ret_5b` with a horizon-correct overlap haircut. All user-facing copy (Alpha-Health tab, engine
card, tooltips) updated from reversion to momentum.

**Validated end-to-end on real data.** Engine `Priority_Long` IC vs forward return: +0.025 (5d) /
+0.032 (21d) / +0.048 (63d). Alpha-health monitor functional and, on a mid-2026 check, correctly
read trailing IC ≈ −0.005 → floored conviction at 0.35, detecting the live momentum dormancy.

**Docs.** `ARCHITECTURE.md` and `README.md` rewritten to the momentum thesis using only
harness-reproducible numbers; version → v5.0.0.

---

## [v4.0.5] · 2026-07-07
### Removed the Dead Breadth Engine

**Removed `breadth_engine.py`.** It ran on every screener and harvest — fetching NSE sector-index
membership (a network round-trip per sector) and attaching three columns (`Universe_Breadth`,
`Breadth_Momentum`, `Sector_Rel_Breadth`) to every symbol — but **nothing consumed its output**.
The ranking engine (`engine.py`) never referenced any breadth column, and no table or chart
displayed them; the columns were carried into the results/harvest dicts and dropped. Its old
consumers — the "Path-A/B/C" tuner factor (F8) and the Layer-2 `breadth_align` feature — were
deleted with the WRCI/intelligence stack in v4.0.0, orphaning the computation. Removed the module,
its import, both build/attach call sites, the three columns from the live + harvest row dicts, the
`Universe_Breadth` entry in the fire-bar feature window, and the market-breadth console line.
No behavior change (nothing read it); saves a per-run sector-map fetch and per-symbol attach. The
flow-zone "Distribution / Accumulation Breadth" charts are unrelated (they count the `Condition`
column) and are unaffected.

---

## [v4.0.4] · 2026-07-07
### Set A = Mean-Axis-Confirmed Divergence · Unified Progress Bar · UI Polish

**Changed — Set A is now a delta divergence confirmed by the thrust's side of the mean axis.**
Dropped the three-way trigger (exhaustion + delta divergence + weekly 80% rule) in favor of a
single, cleaner condition: **bullish = rawBull** (inferred_delta.pine — close down on positive
inferred delta at a 3-bar low) **AND vwm > vwm_mean** (net thrust above the clamp.pine mean
axis, force turning up); **bearish = rawBear AND vwm < vwm_mean**. The mean-axis gate keeps
only divergences the thrust is confirming. Fires ~1.5% of bars per side; verified bit-identical
to an independent inline reference over 40 seeds. Removed the now-dead exhaustion, cooldown, and
80%-rule / prior-week-VA machinery (~90 lines) and the `cool_bars` param; `compute_signal_sets`
no longer takes `opn`. Cache tag `rev4`→`rev5`.

**Fixed — one continuous progress bar per run.** A Single-Date run previously showed TWO
sequential bars: the alpha-health harvest ran its own 0→100% bar, then the screener ran a second
0→100% bar. Now the harvest renders into the first 40% ("Measuring Live Edge") and the screener
into 40→100% ("Screening") of a SINGLE shared bar (cached-edge runs give the screener the full
0→100%). Same fix applied to Correlation mode (the harvest was spawning a second bar mid-run).
`run_timeseries_analysis` / `_ensure_alpha_health` gained `external_progress_slot` + offset/scale
so a caller can own the bar. All progress headers are now Title Case.

**UI/UX polish (no fidelity or thesis change).**
- **Motion grammar unified.** Replaced all 28 `transition: all` declarations — which animated
  every property including layout (width/padding), causing reflow jank and unintended tweens —
  with an explicit compositor-safe property set (`--motion-props`) on one easing curve
  (`--ease`) and three duration tiers (`--dur-fast/base/slow`). Visible animations unchanged.
- **Keyboard focus rings.** Added `:focus-visible` outlines on buttons, tabs, radios,
  checkboxes, and expanders (previously no visible keyboard focus at all). Shows only for
  keyboard nav, never on mouse click — resting/pointer visuals unchanged.
- **Ultra-wide cap.** `.block-container` max-width is now `min(98%, 2100px)` so data tables stop
  sprawling edge-to-edge on ultra-wide monitors; typical displays (≤~2140px) are unaffected.

**Docs / copy.** Purged remaining stale signal vocabulary (Momentum/Crossover/triangle-diamond
comments, "three signal classes A·B·C", "Correlation × Momentum" confluence subtitle) to reflect
the Delta-Divergence (A) / Clamp-Cross (B) sets.

---

## [v4.0.3] · 2026-07-07
### Set B = Clamp Cross · Honest Edge Badge

**Changed — Set B is now the VWM clamp cross.** Replaced the 6-filter confluence with a
single, precise event from `clamp.pine`: **bullish when VWM thrust crosses UP through the
lower clamp** (a selling-thrust impact absorbed back into the band), **bearish when it crosses
DOWN through the upper clamp** (buying-thrust absorbed). Same direction convention as the pine
(below-lower = selling = bullish) — it's the Set A exhaustion re-entry without the
price-still-pushing gate or the cooldown. Fires ~1.5% of bars per side (vs the previous
selective-confluence ~11% and the original loose ~73%). Verified bit-identical to an
independent inline clamp reference over 40 seeds. `SignalType` priority unchanged (A > B).
Removed the now-dead confluence machinery (squeeze percentrank + `_percent_rank`, absorption
B4, thrust-σ B1, CVD-band B5, RVOL B6, and the `conf_min`/`thr_z_min`/`cvd_dev_k`/`rvol_min`/
`sq_*`/`abs_*` params); `compute_signal_sets` no longer takes `rel_delta`/`rel_range`/
`cvd_slope`/`rvol`/`vah`/`val`. Cache tag `rev3`→`rev4`.

**Fixed — the alpha-health "EDGE ACTIVE" badge was knife-edge and oversold noise.** The badge
flipped ACTIVE↔WEAK at a single `trailing_ic > 0.01` cutoff, so a live reading of **+0.0102
(t≈1.0, statistically indistinguishable from zero)** rendered a confident green "EDGE ACTIVE".
An empirical study (210 real F&O names, 4.5y, leak-free) confirmed the current regime is
marginal/dormant, so the confident badge was misleading — and was the main source of the
"ranking feels wrong" experience (the contrarian worst-first ranking is *correct* — reversion
IC positive every full year 2021-2025, momentum backwards — but a barely-positive edge was
being labeled as strong). Fixes:
- `_measure_trailing_ic` now also returns a **t-stat** with a 3-bar-overlap significance
  haircut (Newey-West-lite), threaded into session + `opt_results`.
- New shared `_edge_state(ic, t)` classifier: "EDGE ACTIVE" requires IC ≥ 0.015 **and**
  t ≥ 1.5; a positive-but-insignificant IC reads **"MARGINAL — not significant"**, not green.
  Hysteresis via a wide neutral band so a boundary reading doesn't oscillate. Both the
  Intelligence tab and sidebar Engine Status card show the IC **with its t-stat**.

The reversion ranking direction and the alpha-health *multiplier* math are unchanged — this
is a truthfulness fix on how the edge is *labeled*, plus the Set B swap.

---

## [v4.0.2] · 2026-07-07
### Set B Selectivity — Confluence That Actually Filters

**Fixed:** Set B fired on ~73% of the universe (screenshot showed 205/207 long, 199/203 short
candidates flagged), making it useless as a "confluence" read. Root cause was structural, not a
mere threshold: 3 of the 6 filters were direction-labeled coin-flips — B1 tested only the thrust
*sign* (~50% of bars), B5 was a 50/50 `cvd>ma OR slope>0` OR (~66%), B6 was `RVOL>1` (~50%). With
three ~50-66% filters the expected count already cleared the `≥2-of-6` gate on most bars.

Made the loose filters selective and raised the gate to `≥3-of-6`:
- **B1** now requires thrust beyond ±0.5σ (`thr_z_min`), not just a sign.
- **B5** now requires CVD to deviate ≥0.5 band-widths (`cvd_dev_k`) from its 20-bar mean in the
  signal's direction — the same deviation band the flow `Condition` column uses — replacing the
  50/50 OR.
- **B6** now requires `RVOL > 1.3` (`rvol_min`), not `> 1.0`.

Measured effect on synthetic daily panels: per-side Set B fire rate **73% → ~11% of bars**
(~20% of names on any given day); Set A unchanged (~6%). Cache tag bumped `rev2`→`rev3` so
frames computed under the old loose Set B invalidate. All knobs are keyword args on
`compute_signal_sets` (loosen `conf_min`/`thr_z_min`/`rvol_min` to widen it again).

---

## [v4.0.1] · 2026-07-07
### Fidelity Pass — Regime-Engine Corrections, Pine-Faithful Signal Sets, Honest Copy

A full-system audit (every module read, statistical claims verified by simulation) followed by
a dependency-ordered fix pass. No scoring-engine (`engine.py`) changes — the reversion ranker
and alpha-health monitor were audited and found sound (no look-ahead: forward returns are NaN
until realized; null-input simulation earns no health; skilled-input simulation earns full).

#### Fixed — statistical / correctness
- **GARCH vol-regime inversion under sustained high volatility.** The variance recursion was
  clipped at 1.0 while the long-term mean tracked *unclipped* realized shock variance, so the
  current/long-term ratio collapsed and sustained high-vol stretches read **"LOW"** (verified:
  a simulated σ 0.5→2.5 jump read LOW on 100% of bars after ~60 bars). Since `Vol_Regime`
  scales conviction (HIGH 1.15 / EXTREME 0.55 / LOW 0.90), this inverted the regime tilt in
  exactly the regime where the reversion edge is strongest. The cap is now a pure numerical
  guard (25.0). Post-fix, the same simulation reads EXTREME/HIGH right after the jump and
  re-baselines to NORMAL (the detector is relative-to-own-norm by design). NOTE: the
  `VOL_REGIME_REV` weights were validated against the old labels — flagged in ARCHITECTURE.md
  for re-validation.
- **HMM label switching.** The online adaptation of the 3-state Gaussian emissions had no
  ordering constraint, so the "BULL" state's mean could drift below "BEAR"'s (31/200 simulated
  paths fully inverted; 127/200 broke ordering at least once), semantically flipping every
  Regime / HMM_Bull / HMM_Bear output thereafter. An identifiability projection now re-sorts
  states by emission mean (permuting stds, beliefs, the transition matrix, and recorded state
  labels consistently). Post-fix: 0/200 violations.
- **Correlation "Trade Intelligence" setups were unreachable.** The classifier matched Zone
  against the WRCI-era `OB/OS` names; the flow Condition emits `Accumulation*/Distribution*`,
  so LAGGARD/RUNAWAY/CONTRA could never fire (the tab always showed 0 setups). Re-mapped to
  the real zone vocabulary — and fixed a second latent defect exposed by the resurrection:
  the divergence-sign conditions contradicted the classifier's own rationale (a "laggard"
  required div > +2, i.e. *out*-performance). Signs now match the stated semantics.
- **Expected % / Div % now use beta, not bare correlation.** `expected = corr × target_move`
  silently assumed every symbol has the target's volatility, inflating implied moves for
  low-vol names ~2× in simulation. Now `beta = corr × (σ_sym/σ_tgt)` over the same lookback.
- **Meta Filter "Hide" was inert.** Hiding was gated on a `Meta_Active` column that the
  reversion engine never emits, so Hide mode never hid anything. The gate is removed — the
  engine's Meta score is always the live conviction fusion and may hide.
- **"Aggregate Signal Momentum" chart was a dead flatline.** It plotted the daily mean of
  Delta_Z (which concentrates within ±0.25) on a ±80 axis with ±20 "extreme" bands sized for
  the removed WT1 ±100 oscillator; the strengthening/weakening trend badge used a ±5 threshold
  on the same scale and was pinned to "Stable". Rescaled (±0.25 bands, autoscaled axis,
  0.5 trend threshold) and retitled "Aggregate Flow Skew".
- **Volatility Dynamics chart:** the High-Vol % trace was never assigned to the labeled right
  axis (`yaxis2`), crushing the change-point bars on a shared axis. Assigned.
- **"TOP x%" percentile cell** could display >100% for bottom-ranked names. Clamped.

#### Changed — signal sets (two same-day steps; the second is what ships)
- **Step 1 (superseded within the day):** Set A / Set B were first replaced with exact Pine
  ports of the *structural* divergences — Set A = structural CVD divergence
  (`inferred_delta.pine` `rawSBull`/`rawSBear`), Set B = structural thrust divergence
  (`clamp.pine` `rawDivBull`/`rawDivBear`) — verified bit-identical to a bar-by-bar
  Pine-runtime simulator (25 seeds × 500 bars). These fire at pivot *confirmation*, i.e.
  10–20 bars after the swing, which made "Today" mean "confirmed today", not "happened today".
- **Step 2 (final): LIVE same-bar Trigger/Confluence sets.** Set A (`long_cond`/`short_cond`)
  = TRIGGERS, any one fires: (1) thrust EXHAUSTION (`clamp.pine` `rawExhSell`/`rawExhBuy`,
  Relative mode, adaptive clamp width, incl. the 5-bar same-side cooldown — part of the pine
  signal definition), (2) bar-level DELTA DIVERGENCE (`inferred_delta.pine` `rawBull`/
  `rawBear`), (3) the weekly 80% RULE (open outside the prior week's re-binned value area,
  two closes back inside; once per week; self-disables on weekly-resampled frames, matching
  pine's `sessionable` gate). Set B (`*_comp`) = CONFLUENCE, fires when ≥2 of 6
  direction-matched context filters agree: thrust sign, clamp squeeze (width percentile ≤15
  or just released), value-area edge (rolling or prior-week), one-sided absorption
  (relDelta>1.8, relRange<0.6), CVD agreement, RVOL>1. `SignalType` priority is now
  A (trigger) > B (context). The order-flow ATR feeding `Rel_Range` switched from SMA to
  pine's RMA (`ta.atr`) for absorption parity; the analyzed-frame cache tag bumped
  (`rev1`→`rev2`) so frames computed under the old sets invalidate. Verified bit-identical
  to a literal bar-by-bar pine-style reference (12 seeds × 520 bars × 4 signal streams).
  Observed firing rates on synthetic daily data: Set A ≈ 5% of bars per side; Set B ≈ 65%
  per side at the ≥2-of-6 gate (a deliberate, loose context read — raise `conf_min` to 3
  in `compute_signal_sets` for tighter confluence).

#### Removed — dead weight
- The theme toggle: it rendered in a 0-height component iframe (invisible, unstyled) and its
  JS set `data-theme` on the iframe's own document — it never switched anything. Its CSS rules
  went with it (the `[data-theme="light"]` palette is retained but currently unreachable).
- Ten zero-call-site UI components (~300 lines) from prior engine generations, two unused
  `_zone_colors` dicts, unused legacy signal frames, four never-called `reset()` methods, a
  dead `"tuned"` rerun branch, unused imports/locals, and the never-rendered "Intel Flags"
  column config.
- Duplicate CSS `@keyframes pulse`/`shimmer` definitions — the later bolt-on copies silently
  overrode the design-system originals for every consumer (the progress dot had lost its scale
  pulse; the skeleton sweep ran backwards).
- Duplicate `VERSION`/`PRODUCT_NAME`/`COMPANY` constants in `ui/theme.py` (zero importers);
  `sanket.VERSION` is the single source of truth.

#### Docs / copy honesty
- Purged stale WRCI/momentum/absorption/self-tuning vocabulary from every user-facing surface:
  landing tagline ("WRCI Engine"), tab names, tooltips (incl. an "Intel Conf" tooltip that
  described a removed calibrated-probability model), the Meta-Filter banner's pointer to a
  sidebar section that no longer exists, and export/legend copy. The historical-dashboard
  breadth series are now labeled what they are (Accumulation % / Distribution %).
- `ARCHITECTURE.md`: new "Known limitations" section (alpha-health survivorship, trailing-IC
  null behavior, vol-regime weight re-validation, relative nature of the GARCH regime).

---

## [v4.0.0] · 2026-06-29
### The Honest Rebuild — Cross-Sectional Reversion, Evidence-First

**"Trade what the data says, not what the indicator says"**

A **complete replacement of the scoring engine**. Every prior ranking subsystem (the WRCI
oscillator, Conviction/Pulse, the HCI count, the AutoTune filter, the order-flow signal sets,
the asymmetric Priority Engine, and the 3-layer self-tuning Intelligence stack) was validated
on real NSE F&O data — and **removed**, because it did not predict forward returns. The system
was rebuilt around the one edge that *did* survive walk-forward, cost-aware testing:
**short-horizon cross-sectional mean-reversion.** The full thesis, validation, and design
rationale live in the new `ARCHITECTURE.md`.

#### Why (the evidence)
- On daily NSE F&O (147 names, 5y, ~170k symbol-days), the old momentum factor stack had a
  **negative** cross-sectional rank-IC (naked Priority IC ≈ −0.023, t ≈ −3.9). The Optuna
  calibrator **could not fix it** — its factor-weight bounds are non-negative, so against
  anti-predictive factors the best it can do is shrink to noise. The 3 Intelligence layers
  added no out-of-sample edge over naked Priority and were fragile.
- A simple **reversion** composite scores **IC ≈ +0.031 (t ≈ +8.5)**, positive every year
  2021–2025, strongest in HIGH-vol regimes. The factors carried real information all along —
  the old engine was simply pointed the wrong way.
- After realistic costs the raw signal only survives at multi-day holding, so the product is
  honestly framed as a **decision-support ranker**, not a costless high-turnover strategy.

#### New engine (`engine.py`)
- **Cross-sectional reversion score**: an equal blend of within-date robustly-z-scored
  (median/MAD), sign-flipped reversion features — ATR-normalized 2/5-bar returns, distance
  from the 5/10-bar MA, and 10-bar range position. Equal-weighted on purpose (a fitted weight
  vector did not beat it out of sample and invites overfit).
- **Live alpha-health monitor**: the system measures its *own* realized edge — the trailing
  cross-sectional IC of its score vs forward returns — and scales a global Conviction
  multiplier in `[0.35, 1]`. When reversion is dormant (as it was in 2026) the screen
  **stands down**: it still ranks, but at honestly low conviction. This is surfaced, not hidden.
- **Conviction** = tail-strength × alpha-health × vol-regime suitability; **Side** is the
  cross-sectional tail (oversold → Long, overbought → Short), so the shortlist is never empty.
- Emits the prior UI column contract (`Priority_*`, `Intel_Confidence`, `Meta_*`) under new
  reversion semantics, so every table and card renders unchanged.

#### Removed
- **Deleted modules**: `priority_engine.py`, `intelligence.py` (Optuna calibration, per-set
  logistic, Meta fusion, asymmetric factor scoring, profile persistence). Removed the entire
  WRCI math library from `sanket.py` (EMA/HMA/WMA/VWMA/ALMA/RMA, `f_smooth`, linreg, RSI, the
  Ehlers AutoTune band-pass + its Numba JIT shim) and the WRCI-oscillator divergence detector.
- **Deleted indicators**: `wrci.pine`, `count.pine`, `Order Flow.pine` — they implemented the
  removed architecture and have no parity with the new (inherently cross-sectional) engine.
- **Dropped deps**: `optuna`, `numba`, `filelock` (all served calibration / JIT / profiles).

#### Retained
- The **regime engine** (HMM / GARCH / CUSUM) and **breadth engine** as order-flow-agnostic
  risk/regime context that conditions conviction.
- **Order-flow** (inferred delta / CVD / POC / value-area / absorption) as **descriptive UI
  context only** — validated to add zero cross-sectional ranking IC, so it informs the trader
  but never the rank.

#### UI / UX (identity preserved, copy made honest)
- "Intelligence Center" tab → **Alpha-Health Monitor** (trailing IC, edge state, conviction
  scaling). "Model Passport" sidebar → **Engine Status**. Signal guide, landing cards, table
  tooltips, terminal phases (`REVERSION RANKING`, `EDGE MEASUREMENT`), and progress labels all
  updated to reversion semantics. Layout, CSS, color system, and information hierarchy unchanged.

#### Documentation
- New `ARCHITECTURE.md` (thesis + validation + honest limits). README fully rewritten.

---

## [v3.5.0] · 2026-06-02
### Breadth Intelligence — Market & Sector Advance/Decline as a Three-Axis Edge

**"The Tape Joins the Engine"**

A feature release that folds advance/decline **breadth** into the ranking + intelligence stack along **three orthogonal axes**, from a single shared engine. Breadth is derived from the universe close panel the screener already holds (`get_universe_data`), so there is **zero new data dependency** — the same advances/declines the market shows are now read by the engine. A universe-wide breadth value is a *timing/regime* signal (one number per date), not a stock-discriminating one, so it deliberately enters in three different places rather than as a naïve cross-sectional factor (which would be inert — identical for every stock on a date → zero cross-sectional IC).

#### Features
- **Breadth Engine** (`breadth_engine.py`, new): ports the Hemrek "Relative Breadth" oscillator — EMA(10)-smoothed `A/(A+D)` blended with six Fibonacci-period SMAs (~[0,1], 0.40/0.50 oversold/overbought bands) — and adds `Breadth_Momentum` (3-bar) plus **sector-relative breadth** (`Sector_Rel_Breadth = sector_breadth − universe_breadth`, de-meaned so it's orthogonal to the market level). Built once per run from `data_dict`; attached identically in the live-screener and calibration-harvest paths so train and apply features match bar-for-bar. Self-tested.
- **Path A · Market-breadth regime tilt** (`priority_engine.py :: _breadth_tilt`, in `compute_priority`): a **bounded per-side multiplier** on final priority — `long ×(1+α·b)`, `short ×(1−α·b)`, where `b∈[-1,1]` blends breadth level + momentum and `α=0.20`. Breadth is uniform within a date, so this rescales **long-vs-short exposure** without reordering either side (within-side rank is invariant to the tilt — verified). Tilt is bounded to `[0.80, 1.20]` so it can never dominate the calibrated factors. Not IC-calibratable (zero within-date variance) → fixed, not searched.
- **Path B · Breadth confidence feature** (Layer 2): new `breadth_align = dir·(Universe_Breadth − 0.45)` in `CONF_FEATURES` / `signal_conf_features` — a long fired into an advancing tape scores positive, a short into a strong tape negative. A legitimate *temporal* feature (market-wide is fine for a per-signal classifier). Flows through `calibrate_signal_confidence` automatically (persisted in `feature_names`; name-aligned at predict, so old models still score).
- **Path C · F8 sector-relative breadth factor** (cross-sectional): `beta_F8_breadth_long/short` added to the inner score, the IC tuner kernel (`_PrecomputedDataset.M` → 7 factors, `_evaluate_ic`), and the Optuna search space. Because F8 is de-meaned against the market it varies across the cross-section (stocks in out-participating sectors rank up) and **can earn real IC** — unlike F7, it's searched by default (`enable_f8=True`), with the gate retained so it can be pinned to 0 if validation shows no edge. Sector map built from NSE sectoral-index membership (India universes only; fail-fast + cached, degrades to Path-C-off elsewhere so the screener is never stalled by breadth).
- **Intelligence-tab UI**: the **Active Weights** table now lists **F8 · Breadth** (a live calibrated factor, no probation tag) alongside F1–F7; the Optuna fANOVA importance chart surfaces F8's searched params automatically. A new **Fixed · Structural** section shows the Path-A **Breadth Tilt α** (long/short) with its implied `×[0.80, 1.20]` exposure band, explicitly flagged as *not calibrated* so it isn't mistaken for a tuned weight.

#### Design discipline
- **No double-counting**: Path C uses *sector minus universe* breadth, keeping it orthogonal to Path A's market-level tilt; Path A and Path B both condition on the same tape but on different stages (exposure scaling vs per-signal confidence), and Path B must prove marginal AUC under the existing probation gate. The self-regulating mechanisms (Optuna IC search for F8, Layer-2 calibration for `breadth_align`) shrink breadth toward zero automatically if it has no edge — so "default-on" is safe by construction.
- **Pine parity unaffected**: breadth is a market-wide / cross-sectional concept with no single-symbol equivalent, so — like Layer 3 — it is Python-only and `wrci.pine` carries the version stamp only. 1:1 indicator parity is preserved.

#### Versioning
- **Unification to `v3.5.0`** across `sanket.py`, `ui/theme.py`, `logger.py`, `breadth_engine.py`, `wrci.pine`, `count.pine` (stamp), `README.md`, and `LICENSE`.

#### Documentation
- **README**: new `breadth_engine.py` in the project structure; line counts refreshed — `sanket.py` (6,910), `priority_engine.py` (1,041), `intelligence.py` (807), `breadth_engine.py` (292).

---

## [v3.4.1] · 2026-06-02
### WaveTrend Parity Patch — `ci` Denominator Guard

**"1:1, To The Letter"**

A parity patch that brings `wrci.pine` into exact textual agreement with the `sanket.py` engine. A full line-by-line audit of the indicator against `run_full_analysis` / `compute_signal_sets` confirmed both sides are already in 1:1 functional parity across every engine — WaveTrend, Liquidity, AT Filter (Ehlers AutoTune), Conviction, Pulse, the Hemrek Count (HCI) trend gate, and all three signal sets (A · Momentum, B · Crossover, C · Threshold) with identical gates, parameters, and defaults. The audit surfaced a single source-level divergence: the WaveTrend channel-index (`ci`) zero-deviation guard was floored differently on each side. Output is unchanged on every real-data bar; the patch only closes a divergence that could appear on a pathologically flat series.

#### Fixes
- **WaveTrend `ci` guard aligned** (`wrci.pine §3A`): `(ap − esa) / (0.015 * math.max(d, 1e-9))` → `(ap − esa) / math.max(0.015 * d, 1e-6)`, matching `sanket.py :: run_full_analysis` exactly. The Pine form floored the raw deviation `d` (effective denominator floor ≈ 1.5e-11); the Python form floors the whole denominator at 1e-6. On real bars `0.015·d` dominates both guards, so screener and chart values are identical — the change only removes a ~6× divergence in the regime where `0.015·d < 1e-6` (near-flat / synthetic series). No engine, signal-logic, gate, parameter, or API change otherwise.

#### Versioning
- **Unification to `v3.4.1`** across `sanket.py`, `ui/theme.py`, `logger.py`, `wrci.pine`, `count.pine`, `README.md`, and `LICENSE`.

#### Documentation
- **README**: version strings refreshed to `v3.4.1`; project-structure line counts corrected to the current files — `sanket.py` (6,876), `priority_engine.py` (976), and `wrci.pine` (529, up from the stale 461 figure that predated the HCI / AT-Filter trace additions already present in the indicator).

---

## [v3.4.0] · 2026-06-02
### Meta Intelligence — The Final Intelligence Layer

**"The Final Layer"**

A feature release that upgrades **Layer 3** from a user-set threshold on a single confidence scalar into a calibrated, walk-forward-validated **meta intelligence** model. Layer 3 now *fuses* the two informationally-orthogonal views the earlier layers keep separate — the **cross-sectional Priority rank** (`compute_priority`) and the **per-signal Intel confidence** (Layers 1/2, per-symbol) — into a single `Meta_Score`, a 0–3 tier, and a human reason. Like the rest of the stack it is **probation-gated**: it may reorder/filter only when it has demonstrated out-of-sample edge, otherwise it stays advisory. `wrci.pine` carries the version stamp only — Layer 3 is a Python-only post-ranking layer with no Pine equivalent, so 1:1 indicator parity is unaffected.

#### Features
- **Layer 3 · Meta Intelligence** (`priority_engine.py`): `META_FEATURES` (rank percentile, confidence, their interaction, is-calibrated), `meta_conf_features`, `predict_meta_intel`, `set/get_active_meta_model`, and `compute_meta` → adds `Meta_Score` (0–1), `Meta_Tier` (0–3, fixed bands), `Meta_Source` (`meta`/`fallback`), `Meta_Active`, `Meta_Reason`, `Meta_Spread`. With no active model it falls back to `rank × confidence` (advisory).
- **Meta Intelligence calibrator** (`intelligence.py :: calibrate_meta`): materializes cross-sectional Priority on the harvested panel via a per-date `compute_priority` pass (the panel carries Intel confidence but not rank), fits a logistic on the same magnitude-aware directional-return-past-deadband label used by Layer 2, and reports out-of-sample diagnostics. New `_spearman_ir` helper computes the cross-sectional rank-IR (direction-signed, matching the Priority Engine's IC methodology).
- **Probation gate**: the model is `active` (allowed to reorder + Hide) **only if its OOS rank-IR beat naked Priority's** rank-IR and is positive; otherwise advisory (annotates, never hides). Same discipline that gates F7 and the Layer-2 filter.
- **Abstention**: when today's cross-section shows no spread in the Meta score, the screen falls back to the raw Priority order and labels it as such.
- **UI surfaces**: a new **`Meta`** column in the Action Dashboard, Priority Ranking, and Signal Strength tables (tier-banded fused score, ◆ calibrated / ◇ fallback); an **Intelligence-tab Layer-3 panel** reporting Meta-IR vs naked-Priority-IR, edge delta, AUC, and active/advisory status.

#### Behavior Changes
- **Layer-3 filter is now the Meta Filter**: the sidebar Off / Dim / Hide control and threshold act on the fused `Meta_Score` instead of `Intel_Confidence`. Today's fired signals filter by the Meta score; aged signals fall back to fire-bar Intel. An **advisory** meta model dims but **never hides** (probation guard); the threshold auto-seed prefers the meta AUC, then the Layer-2 AUC, then 0.45.
- **Profile artifacts**: each profile now persists a third model, `meta_intel`, alongside `weights` and `signal_conf`. Threaded through `save_profile` and every activation/import site via `set_active_meta_model`.
- **Calibration runner**: `run_priority_optimization` learns and logs the meta model (meta-IR vs priority-IR + active flag) right after Layer 2; `run_screener_analysis` applies `compute_meta` after `compute_signal_confidence`.

#### Versioning
- **Unification to `v3.4.0`** across `sanket.py`, `ui/theme.py`, `logger.py`, `wrci.pine` (stamp only), `README.md`, and `LICENSE`. (The prior `v3.3.0` changelog entry had shipped without the code version strings being bumped; this pass brings every component even.)

#### Documentation
- **README**: new **"The Intelligence Stack (Layers 1–3)"** section documenting all three layers, the probation/abstention discipline, and the Meta Filter; result-tabs and per-row output updated for the Intel + Meta columns; architecture blurbs and project-structure line counts refreshed.
- **LICENSE**: restriction §5 IP enumeration extended to the multi-layer intelligence stack (signal-confidence + meta intelligence calibration).
- Line counts: `sanket.py` (6,835), `priority_engine.py` (970), `intelligence.py` (793), `wrci.pine` (461), `logger.py` (226).

---

## [v3.3.0] · 2026-05-30
### Liquidity Engine, Inline Self-Tuning & Data-Source Hardening

**"Flow & Folded Intelligence"**

A feature release that adds a microstructure Liquidity engine and its per-set kinematic gates, folds the Self-Tuning calibration into the screener run (no separate mode), and hardens the F&O / index data sources. `wrci.pine` and `sanket.py` kept in 1:1 parity throughout.

#### Features
- **Liquidity Engine (microstructure flow)**: new ±100 oscillator (volume-weighted intrabar spread vs. multi-bar price impact → clipped z-score → sigmoid), with `liq_vel` (velocity) and `liq_accel` (acceleration). Added to both `wrci.pine` (§3B) and `sanket.py` (`run_full_analysis` → `Liquidity_Osc` / `Liq_Vel` / `Liq_Accel`), with a zero-volume divide guard on the Python side.
- **Inline, one-pass Self-Tuning**: the standalone "Intelligence (Self-Tuning)" analysis mode is **removed**; harvest + Optuna calibration now run inline on the **Single Date / Pulse** screener via `_ensure_intel_weights()`. Reuses a profile already calibrated **today** for the `(universe, index, timeframe)`; otherwise harvests a lookback (~2y daily / ~3y weekly) and calibrates, then ranks the screen with the tuned weights. Sidebar **Self-Tuning Intelligence** expander (below the Model Passport) carries trials / split / **Force recalibrate this run**.
- **Intelligence result tab**: Single-Date results gain an **Intelligence** tab (Train/Val IR, stability, factor-importance fANOVA chart, active-weights table) and a **Priority Rank** sub-tab listing the full universe by tuned priority (bull/bear aware).
- **NseKit F&O source**: F&O constituents now fetch via NseKit's official `underlying-information` endpoint as the primary source (survives datacenter-IP blocking), ahead of the legacy `equity-stockIndices` paths.

#### Behavior Changes
- **Per-set kinematic liquidity gates** (parity across `wrci.pine` + `sanket.py`): Set A & B require liquidity **level** (`Liquidity_Osc` same-signed); Set C requires liquidity **velocity** (`Liq_Vel`); Set D requires liquidity **level + acceleration** (`Liquidity_Osc` & `Liq_Accel`). Net effect: fewer, flow-confirmed signals.
- **Set C Δ-polarity gate**: Threshold now also requires `Conviction Δ` / `Pulse Δ` polarity (it previously omitted it), matching Sets A/B/D.
- **WT2 signal line = configurable MA, ALMA(20) default** (was SMA-4), plumbed through the sidebar/screener (`wt2_len`, `wt2_type`).
- **Profile key now includes timeframe**: profiles are keyed per `(universe, index, timeframe)` so daily and weekly weights no longer collide.

#### Fixes
- **F&O list correctness**: legacy `SECURITIES IN F&O` paths now skip the leading index-aggregate row (off-by-one phantom ticker) and de-duplicate; the NIFTY-500 fallback is flagged as a superset rather than reported as a clean F&O fetch.
- **Index-constituent resilience**: archive-CSV fallback tries both `archives.nseindia.com` and `nsearchives.nseindia.com`.
- **Model Passport refresh**: after a fresh inline calibration the Passport updated only on the next interaction; a guarded post-results `st.rerun()` now refreshes it in the same run (results are already persisted, so no recompute and no re-tune).
- **Streamlit deprecation**: all `use_container_width=True` replaced with `width='stretch'`.

#### Documentation
- **README**: added the Liquidity Engine core component + Micro Phase; corrected WT2 to configurable ALMA; reframed Intelligence as inline (no separate mode) with the new first-run workflow; renumbered Analysis Modes (Intelligence mode removed); fixed the profile key to `(universe, index, timeframe)`; Sets A–D table now lists each set's liquidity gate.
- **Docstrings/comments**: `compute_signal_sets` docstring documents the per-set liquidity gates; stale "Intelligence mode" references in the bulk-range comment and the legacy-profile import warning updated.
- Line counts: `sanket.py` (6,117), `wrci.pine` (523), `intelligence.py` (411), `priority_engine.py` (381).

---

## [v3.2.1] · 2026-05-21
### Set A Δ-Polarity Gate & Signal Engine Refactor

**"Symmetric Conviction"**

Behavior-changing tightening of the Momentum signal gate plus a focused refactor of `sanket.py` to extract long-lived inline blocks and surface the sidebar return as a typed dataclass. Pine indicator (`wrci.pine`) updated in lockstep to preserve 1:1 parity.

#### Behavior Change — Set A
- **Δ-polarity gate added to Set A** (Momentum): WT1/WT2 crossings now require `Conviction Δ` and `Pulse Δ` to be the same sign as the trade direction (long: both > 0; short: both < 0). Brings Set A in line with the gate already used by Sets B and D. Net effect: fewer but better-confirmed Set A signals; historical Set A counts will drop. The opposite-side Set B veto (long A blocked when B-short fires, and vice versa) is retained on top of the new gate.
- **`wrci.pine` synchronized**: Pine `momentum_long` / `momentum_short` now carry the same Δ gate, preserving 1:1 mathematical parity between the Python screener and the TradingView indicator.

#### Refactor (No Behavior Change)
- **`compute_signal_sets` helper extracted**: Sets A/B/C/D logic and the zone `Condition` column moved out of `run_full_analysis` into a dedicated function with a docstring explaining each set's predicate, gating, and the load-bearing `np.select` ordering for the zone label.
- **`SidebarState` dataclass**: `render_sidebar()` now returns a typed dataclass instead of a 16-element positional tuple. The data flow from sidebar to `main()` is name-keyed; new inputs no longer require updating a tuple unpack.
- **Shared HTML-builder palette helpers**: `_side_palette`, `_signed_color`, `_delta_arrow`, and `_GREEN` / `_RED` constants dedupe the green/red and arrow ternaries that were repeated across `_build_confluence_table_html`, `_build_signal_table_html`, `_build_narrative_table_html`, and `_build_signal_strength_table_html`.
- **Redundant imports removed**: Four local `import html as html_module` lines deleted — the top-level `import html` covers all `html.escape` callsites. Dead `from io import BytesIO` removed (all callers go through `io.BytesIO()`).

#### Documentation
- **README Signal Hierarchy table rewritten**: Sets B, C, D descriptions now accurately reflect the regime-filter crossover, signal-line-validated zone entry, and regime-zero-cross triggers respectively (previous text described unrelated logic).
- **README search-space breakdown corrected**: Now correctly enumerates 12 betas + 4 gammas (reversion + divergence, each side) + 5 tier multipliers = 21 dimensions.
- **README profile JSON example fixed**: Uses real field names (`val_score`/`train_score`/`sensitivity`/`tier_A_mult`) and the actual `" · "`-joined composite key format.
- **README line counts refreshed**: `sanket.py` (5,928) and `wrci.pine` (412).
- **`ui/components.py` Signal Types Reference rewritten**: Sets A and B descriptions match the actual triggers; Set D card added (CSS class `.signal-type.squeeze` already existed in `theme.css`, but the HTML card was missing).
- **Mislabeled comments fixed**: Two `# Set C: Momentum` comments at `sanket.py:2402-2408` and `sanket.py:5147` corrected to `# Set A: Momentum` (these labelled the legacy `L_`/`S_` alias columns, which read `long_cond` / `short_cond` — Set A's columns, not Set C's).

---

## [v3.2.0] · 2026-05-09
### System Hardening, Fidelity & UI Polish

**"Precision Instrument"**

Comprehensive institutional-grade hardening pass across the full stack — data correctness, multi-session isolation, calibration reliability, and terminal UI smoothness. No surface-level UX changes; all improvements are under-the-hood quality and fidelity improvements.

#### System Architecture
- **Per-session weight isolation**: Active weights now stored in `st.session_state["active_weights"]` per session, eliminating cross-user bleed when the app is deployed for multiple users simultaneously
- **Smart data registry**: TTL-aware fetch cache keyed by universe + date — avoids redundant OHLCV fetches across mode switches (15 min during market hours, 90 min outside)
- **Registry DataFrame copies**: Registry stores `.copy()` of each DataFrame, preventing downstream mutations from silently corrupting cached data
- **Reproducible calibration**: Optuna TPE sampler seeded with `seed=42` — calibration results are now reproducible across identical inputs
- **`HOLD_HORIZONS` constant**: Fibonacci-spaced horizons `[2, 3, 5, 8, 13]` extracted to `priority_engine.py` as a single source of truth, imported by `sanket.py` and `intelligence.py` — previously hardcoded in five separate places

#### Correctness Fixes
- **L/S Ratio division guard**: Long/Short signal ratio now emits `NaN` instead of `Inf` when short signal count is zero
- **Divergence order scaled to timeframe**: `argrelextrema` order parameter set to `2` for weekly and `3` for daily data — was fixed at `3` regardless of timeframe, causing missed weekly divergences
- **Regime detector warmup**: HMM state estimator now runs a 20-bar warm-up period before recording signal history, preventing false regime transitions at the start of the analysis window
- **ymax NaN/Inf guard**: Bar chart y-axis maximum now guarded against `NaN`/`Inf` values that caused silent chart rendering failures on short date ranges
- **Confluence score clipped**: `Confluence_Score` clamped to `[0.0, 1.0]` on both calibrated-priority and fallback paths — previously could exceed 1.0 on wide-spread cross-sections
- **% Change Since sentinel**: Percentage-change-since-analysis field now uses `None` sentinel instead of `0.0` when the analysis date equals the latest available date — eliminates spurious 0.00% displays

#### Calibration Improvements
- **Overfit detection split**: Separated `low_ir` (Val IR ≤ 0) and `overfit` (Train IR >> Val IR) flags — both are now detected independently with distinct user messages
- **Quality Check card**: Fourth metric card added to Calibration Diagnostics showing `No Edge` / `Overfit` / `Quality OK` status with semantic color coding
- **Small universe warning**: Calibration warns when average symbols per date falls below 20 — IC-based ranking is statistically unreliable on sparse cross-sections
- **Exception logging**: All `except: pass` patterns replaced with typed exception logging via `console.detail()` — silent failures are now surfaced in the terminal

#### Display & Data Quality
- **Run stats header**: Results header now shows total universe size, symbols fetched, analyzed, and failed — provides context that was previously invisible
- **Column display rename**: Result table columns use human-readable names (`Priority_Long_pct` → `Long Priority %ile`, `F1_PriceMom` → `Price Momentum`, etc.)
- **Widget state persistence**: All sidebar widgets keyed with `sb_*` session-state keys — widget selections persist across reruns without unexpected resets
- **Timeframe-aware passport**: Calibration profile keys now include timeframe — daily and weekly calibrations are stored and loaded independently under the same universe

#### UI Smoothness (No Visual Changes)
- **Skeleton shimmer suppressed**: `[data-testid="stSkeleton"] { display: none !important }` eliminates the native Streamlit shimmer that briefly appears between a button click and the first progress bar render
- **CSS loading cached**: `@st.cache_resource` on `_load_theme_css()` — the 4,300-line `theme.css` is read from disk once per process; subsequent reruns pay zero I/O cost
- **Equal-height metric cards**: `:has(.metric-card)` CSS block extends the flex chain through `element-container` and `stMarkdownContainer` — metric card rows are now equal height on all pages (Single Date, Pulse Narrative, Calibration Diagnostics, Historical Range, Intelligence, Correlation) regardless of content length or screen size

---

## [v3.1.0] · 2026-05-07
### Documentation & Version Unification

**"Uniform Signal"**

Version alignment pass bringing all system components — main application, UI theme, logger, and Pine Script indicator — to a single canonical version string. Accompanied by a full documentation rewrite.

#### Changes
- **Version unification**: `sanket.py`, `ui/theme.py`, and `logger.py` all bumped to `v3.1.0`, matching the existing `wrci.pine` indicator version
- **README rewrite**: Complete documentation overhaul — architecture deep-dive, engine internals, factor math, signal hierarchy, intelligence calibration workflow, deployment guide, profile structure reference
- **CHANGELOG rewrite**: Full version history reconstructed from v2.0.0 to present with accurate release notes
- **LICENSE update**: Product version updated to `v3.1.0`

---

## [v3.0.0] · 2026-05-06
### Production Calibration Release

**"The Asymmetric Engine"**

First production-grade release of the asymmetric Priority Engine with Bayesian self-tuning. Separated long and short factor betas, introduced per-universe profile persistence, and shipped the Intelligence Center UI.

#### Features
- **Asymmetric Priority Engine**: Separate `beta_*_long` and `beta_*_short` weights for all six factors — the system no longer assumes long/short symmetry
- **Intelligence Center**: Full Optuna TPE calibration workflow surfaced in UI — trial count control, live progress, val IC display, parameter importance chart
- **Model Passport**: Sidebar panel for viewing, exporting, importing, and deleting calibration profiles per universe
- **fANOVA Sensitivity Analysis**: Post-calibration parameter importance ranking using Optuna's fANOVA implementation
- **Per-Universe Profile Auto-Loading**: `load_profile_for()` auto-selects the matching profile when the user switches universe
- **F&O Sample Profile**: Seed calibration (`profiles/fno.json`) with pre-optimized weights for NSE F&O universe (val IC: 0.1556)
- **Legacy Profile Migration**: `_maybe_migrate_legacy_profile()` handles v1 → v2 key format upgrade

#### Internals
- **`_PrecomputedDataset`**: Pre-computes weight-invariant arrays before Optuna trials begin — ~50× speedup over per-trial recalculation
- **L2 regularization**: Added to `_evaluate_ic()` to prevent weight inflation in low-signal regimes
- **Change-point penalty**: Regime shift detection now applies a damping multiplier at structural breaks

---

## [v2.2.0] · 2026-05-05
### Obsidian Quant Transformation & Mathematical Parity

**"The Obsidian Quant Transformation"**

Final synchronization pass between the Python screener and the TradingView indicator. Achieved 1:1 mathematical parity across all calculations. Applied the Obsidian Quant design system terminal-wide.

#### Features
- **1:1 Mathematical Parity**: Unified HMA, WMA, and Linear Regression endpoint calculations across `sanket.py` and `wrci.pine` — screener signals now match chart signals exactly
- **Pulse Engine (v3)**: Implemented abnormal acceleration detection — 3-bar velocity modulated by 20-bar volatility Z-Score, with volume factor (`tanh(volZ/2)`) and price-action factor
- **Obsidian Quant UI**: Applied the full Obsidian design language — `#1a1a1a` background, JetBrains Mono data font, Syne display font, glass-morphism metric cards, staggered entrance animations
- **Pulse Narrative Matrix**: 4×4 state grid (SURGE/FIRM/SOFT/CRUSH × LEAD/DEEP/LIGHT/HOLLOW) for per-bar signal interpretation
- **Global Macro ETFs**: Expanded asset coverage to include global bond ETFs and treasury instruments
- **Fractal MTF Anchoring**: Integrated daily and weekly macro-regime context for tactical signal filtering

#### Fixes & Optimizations
- **VWAP Accuracy**: Refactored to ratio-of-sums instead of discrete averaging — eliminates accumulation drift
- **Signal Gating**: Hardened mutual exclusivity rules for Sets A–D — no double-firing on same bar
- **Anti-Clustering**: Enhanced pattern matching with anti-clustering logic to prevent redundant analog matches
- **Colorama Compatibility**: Fixed ANSI color rendering on Windows 10+ terminals via explicit `colorama.init()`

---

## [v2.1.0] · 2026-04-30
### WRCI Foundation

**"Wave-Regime Composite Index Core"**

Initial release of the WRCI engine and its companion Pine Script indicator. Established the WaveTrend core, Conviction engine, and signal hierarchy framework.

#### Features
- **WRCI Core Engine**: WaveTrend oscillator (WT1/WT2) with normalized HLC3 applied price
- **Conviction Engine**: Three-component composite score — Trend Strength (HMA slope / ATR), Momentum Quality (WT separation), Participation (Volume Z-Score × price direction)
- **Signal Sets A–D**: Defined non-redundant signal classification framework — Momentum, Contrarian, Threshold, Squeeze
- **Squeeze Engine**: Bollinger Band / Keltner Channel compression detection for volatility breakout identification
- **Pine Script v6 Indicator**: First release of `wrci.pine` companion TradingView indicator
- **Analog Pattern Matcher**: Cosine similarity-based historical pattern matching engine (first iteration)
- **Overbought/Oversold Logic**: Zone definitions at `±60` (Threshold) and `±80` (Extreme)

---

## [v2.0.0] · 2026-04-15
### Pragyam Family Rebirth

**"Modular Architecture & Web Terminal"**

Architectural rebuild from standalone script to the Pragyam modular product framework. Launched the Streamlit web terminal.

#### Changes
- **Pragyam Architecture**: Decomposed monolithic script into `sanket.py`, `priority_engine.py`, `intelligence.py`, `logger.py`, and `ui/` module
- **Streamlit Terminal**: Replaced CLI with interactive Streamlit web UI — session state management, sidebar routing, mode switching
- **Multi-Universe Scraping**: Introduced constituent scraping for NSE F&O (nsepython), NASDAQ, and S&P 500 (Wikipedia)
- **Obsidian Quant Design System**: Initial `theme.css` and `components.py` — dark terminal aesthetic with Plotly chart theming
- **Structured Logging**: Replaced `print()` statements with `ConsoleOutput` class — ANSI-colored, phase-timed, run-ID-tagged terminal output

---

*Full technical specifications: [README.md](README.md)*
*Author: [@thebullishvalue](https://github.com/thebullishvalue) · Pragyam Family*
