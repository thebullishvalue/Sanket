# What the tapes, the column, VWAP and the volume profile read at turning points, and whether a signal built from them beats sets A, B and C

**The question:** when an instrument bottomed and turned up (or topped and turned down), what
did the pane and the price chart read? Can those readings tell a real turn from a failed one,
and does a signal built from them beat the existing sets?

**The data:** the Indicator repo's 151 instruments.

| | Daily (2003–2026) | Hourly (2024–2026) |
|---|---|---|
| Swing bottoms / tops | 3,953 / 3,977 | 4,394 / 4,375 |
| Major bottoms / tops | 630 / 671 | 780 / 796 |
| Instruments with volume | 136 | 118 |

| Script | What it does |
|---|---|
| `turns.py` | The pane: four tapes, the column, the midline and the cone, from `ledger.py`. |
| `turns_vp.py` | `pragyam.pine`'s session VWAP and volume profile, ported to bar data. |
| `signals_vs_sets.py` | Candidate signals against sets A, B and C, through Sanket's Edge Study. |

Run each with `python3 studies/v10_turns/<script> <Indicator/research> [data|data1h]`. Results
are in `out/`. Every reading uses only the bars up to that moment.

## Method

### Turns

Turns come from a zig-zag on closing prices, scaled to each instrument's own risk.

- **Swing:** a reversal of at least 2.5 × σ × √10. The median swing is 26% / 19% on daily and
  7% / 6% on hourly.
- **Major:** a reversal of at least 3 × σ × √40.

### Real or failed

The candidates are every new 20-bar closing low (or high). "New" means strictly beyond each of
the 19 closes before it.

- **Turned:** price reversed by the swing amount before making a new extreme.
- **Failed:** price made a new extreme first.

On daily, 10.8% of lows turned and 5.5% of highs. On hourly, 8.4% and 7.3%. Because every
candidate is already at an extreme, "price fell" is held equal across both groups.

"Strict" matters. With ties allowed, a run of stale identical closes counted as a low and a high
on every bar. Risk is near zero across such a run, so one bar's scaled return exploded. This
corrupted the first run of the signal test and is fixed.

### VWAP and the profile, as `pragyam.pine` draws them

- **Session:** a week on a daily chart, a day on an hourly chart.
- **VWAP:** volume-weighted average price since the session began. It falls back to equal
  weights where an instrument has volume on fewer than half its bars, as the Pine does.
- **Profile:** each bar's volume is spread across its range, in rows ATR(20)/4 wide.
  - The POC is the heaviest row.
  - The value area holds 70% of the volume, walked out from the POC.
  - The study also tracks the prior session's POC and value area, and naked POCs.
- **Relative volume:** volume over its 50-bar median.

## Findings: the pane

**The readings price forces.** At every swing bottom the column, the ladder and the midline are
red, and at every top they are green. This is true by construction: the lowest close comes after
a fall, so these readings only say that price fell.

**The readings that carry information:**

| At swing turns (daily) | Bottoms | Tops | All bars |
|---|---|---|---|
| Risk stressed | **51%** (major: 73%) | 26% | 31% |
| Risk rising | **40%** | 20% | 19% |
| Trend down | **50%** (major: 90%) | 1% | 15% |
| Trend up | 6% | **82%** | 44% |

Risk climbs into a bottom and eases within about 10 bars after it (`out/paths_daily.png`). Tops
form in calm risk, and risk rises after.

**Real or failed: no reading tells them apart.** At new 20-bar lows and highs, every single pane
reading with at least 1,000 cases has a lift between 0.89 and 1.21. None holds across asset
classes and eras.

The combinations don't either. The best combinations reach about 1.2–1.6 in the second half. The
first half's ranking predicts the second half poorly: correlation 0.25–0.29, and −0.53 for daily
highs.

**Reverse check.** The strongest reading is "closed below the band". A swing bottom falls within
±2 bars of it 10.9% of the time daily (4.6 times the base rate), so nine times in ten it isn't a
bottom.

**Timing after a swing bottom, in median bars (daily / hourly):**

| Reading | Daily | Hourly |
|---|---|---|
| Column turns green | 5 | 5 |
| Ladder turns green | 6 | 6 |
| Close above the midline | 6 | 5 |
| Trend turns green | 9 | 17 |

Set B fires near 37–42% of swing bottoms, at about bar +26. When it does, about 67% of the move
to the next turn is still ahead.

## Findings: VWAP and the volume profile

**At turns they mostly restate the move.** At daily swing bottoms:

- 96% close below VWAP (lift 2.0).
- 94% are below the prior week's value-area low (lift 3.5).
- 66% are below the session POC (lift 2.4).

Tops mirror this: 96% close above VWAP and 92% above the prior VAH.

**Volume is the one independent reading.**

- **At daily bottoms:** heavy volume (2× or more) is present at 17% of bottoms against 7% of all
  bars (lift 2.5), and above-usual volume at 36%. Volume builds over the 5 bars before a bottom
  and peaks at about 1.3× its median on the turn (`out/vp_paths_daily.png`).
- **At tops:** the rise is smaller, at about 1.15×.
- **Hourly:** the same pattern (heavy volume lift 2.1).

**Naked POCs are less likely near a turn.** A naked POC within 1 ATR has a lift of 0.3–0.6.

**Real or failed: no reading tells them apart here either.**

- **Single readings:** with at least 1,000 cases, VWAP and profile readings have lifts of
  0.78–1.20, and none holds on both timeframes.
- **Heavy volume at a 20-bar low:** 0.97 daily and 1.07 hourly. It marks capitulation at every
  low, not just the low that holds.
- **VWAP and profile crossed with every pane reading:** about 270 cells per side, with
  first-half/second-half correlation of −0.02 to 0.43. On daily the best first-half cells fell
  back to about 0.7–1.0.

**Timing.** VWAP is reclaimed 1–2 bars after a turn, faster than any tape, because the weekly
session VWAP resets often and price crosses it constantly.

## Does a signal built from this beat sets A, B and C?

**No.**

The candidates were declared in `signals_vs_sets.py` and committed before the VWAP results were
read. Each has a ▲ and a mirrored ▼, and fires at most once in 10 bars.

| | Rule (▲) |
|---|---|
| **T1** confirmed low | a new 20-bar low in the last 10 bars, and the column turns green |
| **T2** risk easing | T1, with risk stressed at the low and falling now |
| **T3** VWAP reclaim | a 20-bar low in the last 10 bars, and the close crosses back above session VWAP |
| **T4** back into value | a 20-bar low in the last 10 bars, and the close crosses back above the prior session's VAL |
| **T5** data-chosen | at the low, the VWAP/profile × pane combination with the best turned-vs-failed lift on the first half of the dates |

**How they were judged:** with Sanket's Edge Study, in **one** run with A, B and C, so all
signals share the same symbols, dates and discovery/holdout split.

- Entry is the bar after the signal, with a 10-bar hold.
- Trailing drift is removed and returns are volatility-normalised.
- Confidence intervals come from a block bootstrap over dates, with 3bp costs.
- The **developed signal** is the candidate side with the best discovery edge. It beats the sets
  only if its holdout edge is significant and above the best set's.

| Timeframe | Side | Developed signal | Discovery edge | Holdout edge (95% CI) | Best set, holdout | Beats the sets? |
|---|---|---|---|---|---|---|
| Daily | ▲ | T5 (light volume, normal risk) | +0.120σ (significant) | +0.022 (−0.07, +0.11) | A +0.277 | **No** |
| Daily | ▼ | T4 back into value | +0.033 | +0.021 (−0.03, +0.08) | B +0.057 | **No** |
| Hourly | ▲ | T2 risk easing | +0.089σ (significant) | +0.032 (−0.07, +0.12) | A +0.224 | **No** |
| Hourly | ▼ | T4 back into value | +0.041 (significant) | +0.032 (−0.01, +0.07) | A +0.047 | **No** |

**None of the 10 candidate sides is significant in the holdout, on either timeframe.**

- Every candidate that looked good in discovery shrank toward zero, which is the selection effect
  the protocol exists to catch.
- T1, T3 and T4 fire 20–36 times per 1,000 bars, with edges within ±0.05σ.
- T5's chosen rules collapsed out of sample: daily lows went from a 1.27 lift to 0.92, highs from
  1.79 to 0.80.

**A note on set A.** Set A ▲ is the only signal with a significant holdout edge here (daily
+0.277σ). It fires only once per 11,000 bars, so the Edge Study still rates it UNDERPOWERED.
Earlier work found it concentrated in very few names, so this is not a population edge either.

## What this means

1. **Turns have a recognisable look, but every failed low has it too.** Bottoms come with
   stressed, rising risk, a downtrend, price below VWAP and value, and a burst of volume.
   Nothing on the pane or the price chart, alone or combined, picks out the low that holds.
2. **The tapes confirm a turn, late and reliably.**
   - The column and ladder turn green about 5–6 bars after a swing bottom.
   - VWAP is reclaimed in 1–2 bars, but it is noisy.
   - The trend takes 10–20 bars.
3. **No signal built from these studies beats the existing sets.** Sets B and C have no edge
   either, so the honest summary is that nothing in the pane or the profile forecasts direction
   over 10 bars on this universe. What the system does forecast reliably is risk: the cone and the
   risk regime.

**Caveats:**

- Turns are labelled with hindsight; the readings never are.
- Instruments are correlated, so event counts overstate independent evidence. That only
  strengthens the null results.
- The Pine reads a developing profile from intraday bars where TradingView has them; this port
  uses each bar's own range, which is what the Pine does without intraday data.
