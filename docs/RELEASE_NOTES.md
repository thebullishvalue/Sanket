# Sanket · Release Notes

What changed in each major release, and why, newest first. Every release, with its full
entries, is in the [Changelog](../CHANGELOG.md).

**v10.4.0 — the Edge Study scores against peers; v10.3's set changes rolled back.** The v13 study
(`studies/v13_lab`) found the Edge Study's drift-free score — each event against the name's *own*
trailing average return — flatters any rule that fires in a downtrend, and that it alone produced
the "counter-trend works" result behind v10.3. The Edge Study now scores each event against the
**same-day average of the other names on screen**, with drift-free kept as a second column. Set C
is back to B with the trend agreeing, set D is removed, and the per-set power rule (built on the
inflated numbers) is withdrawn; the screen and copy work of v10.2.x stays.

**v10.3.1 — power judged per set.** A test is UNDERPOWERED when it cannot resolve the edge that
set has shown out of sample — 0.10σ for D, 0.08σ for C, 0.036σ for A and B — instead of holding
every set to 0.036σ, which made C and D read UNDERPOWERED on any normal universe.

**v10.3.0 — set C reversed, set D added.** The research in `studies/v10_turns`, `v11_lab` and
`v12_sets` found that in this system a flip *with* the trend is late and reverts, while a move
*against* a strong trend continues. **C** now asks for the trend tape in its opposing strong state
(down for ▲, up for ▼); on fresh data its ▼ went from +0.01 to +0.08σ daily and from 0.00 to +0.13σ
hourly, its ▲ from −0.05 to +0.03 daily and −0.06 to +0.11 hourly. **D** is new: a band break up in
a downtrend, long only, confirmed at +0.26σ on 572 stocks no study had used. Four set cards, four
slots per event, D on the rail and the Edge Study; the cache key moves to `prg7`.

**v10.2.5 — production copy.** Every page's text rewritten to read as a finished product: the
landing page says what Sanket is in a few lines; each tab carries a one-line subtitle and at most
one reading note; detail lives in hover tooltips and System Data; research history (audits, study
paths, source files) lives here, not on screen. Fixes: a past date no longer reads "Today's Signals"
or "fired today"; age buckets read "1 Session Before" off the latest date; the Historical Range
command bar shows its range, not today; a set tested on too few events shows "N events · too few"
instead of an edge; the landing page's history-per-run is ~3.5 years Daily (it said ~4.5).

**v10.2.4 — the Watchlist, verified.** Tested against history on 151 instruments
([`studies/v10_watch/`](../studies/v10_watch/README.md)): 85% of v10.2.0's "ready" names had a surprise
column already past zero and could not fire next (0% did). Ready now requires the column **armed**
(on the far side of zero): ready names fire B on the next bar ~19-27% of the time, ~50-60% within 10,
against ~0.3% / ~9% for names on no list. A **near** group (armed, calm, the ladder within 0.2 of
agreeing) catches the events the flip itself completes; ready and near together held ~80% of B
events the bar before they fired. Each group's measured rate is printed on the Watchlist.

**v10.2.0 — all three sets at once.** The Signal Set selector is gone: every set fires on every
screen. A signal board puts A, B and C side by side — each with its events in the last 5 bars, its
watch (ready for B / B·C, or in a proven state) and its own measured verdict per side. An event is
labelled by every set that fired it (▲ B·C), drawn as three fixed slots so a column of them reads
as a pattern. A Watchlist tab lists the names one flip from firing, nearest first. The Historical
Range compares the sets on the range, and the rail and the command bar summarise all six slices.
The Edge Study's measurement is unchanged, so its cache key stays `prg6`.

**v10.1.0 — three signal sets.** A (the proven-state triangles of v10.0) is joined by B — the
surprise column flips with the close past the midline, risk calm and not rising, the stretch ladder
agreeing — and C, which adds the trend. Chosen in the rail; all three are always computed and the
Edge Study measures them side by side (six slices, one study). `ledger.py` now carries Pragyam's
four-horizon stretch ladder, the midline and risk momentum, checked bar for bar on 151 instruments.
A READY state marks names with every B / C condition in place but the flip. Weekly fetch deepened to
~17 years so B and C can fire. Edge Study cache key `prg6`.

**v10.0.0 — the signals are Pragyam's proven-state triangles.** Pragati's signal set is stripped:
`pragati.signals()`, the grid's ▲ capitulation / ▼ distribution, the ◆ RESUME, its candidates and
settings, the watchlist and the stack gate on events are gone. `ledger.py` ports Pragyam's ledger
bar for bar (checked against `pragyam.pine`'s reference on 151 instruments: identical states,
z-scores and triangles) and `engine.compute_frame` reads the ▲▼ from it. The UI, the Historical
Range, the Edge Study (now two slices, ▲ and ▼) and the exports follow; the readings, the grid and
the stretch ranking are unchanged. Weekly fetch deepened to ~13.6 years so the ledger can prove a
state. The Edge Study's cache key moved to `prg5`.

**v9.3.0 — the ▲ skips conviction's quiet regime.** A three-round signal search (discovery
2006–19; 2020–26 sealed until the shortlist was fixed; then weekly bars and the Ladder-down window —
[`studies/v9_lab/SIGNAL_SEARCH.md`](../studies/v9_lab/SIGNAL_SEARCH.md)) found one change that held
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
([`studies/pragati_v9_audit.md`](../studies/pragati_v9_audit.md)).

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
`siddhi.pine`: a participation-weighted oscillator measuring how much of each bar's
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
cost-survivable edge. *(Retired in v6.0.0.)* See [`CHANGELOG.md`](../CHANGELOG.md) for full entries.
