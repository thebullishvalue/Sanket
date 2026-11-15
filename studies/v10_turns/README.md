# What the tapes and the column read at turning points

**Question:** when an instrument bottomed and turned up (or topped and turned down), what did each
tape and the column read, and can those readings tell a real turn from a failed one?

**Data:** the Indicator repo's 151 instruments. Every reading comes from `ledger.py`, the same
code the Pragyam pane and Sanket run, and uses only bars up to that moment.

| Timeframe | History | Swing bottoms | Swing tops | Major bottoms | Major tops |
|---|---|---|---|---|---|
| Daily | 2003–2026 | 3,953 | 3,977 | 630 | 671 |
| Hourly | 2024–2026 | 4,394 | 4,375 | 780 | 796 |

To run it:
`python3 studies/v10_turns/turns.py <Indicator/research> [data|data1h]`

Results are in `out/`.

## Method

### Turns

Turns come from a zig-zag on closing prices, scaled to each instrument's own risk:

- **Swing turn:** a reversal of at least 2.5 × σ × √10. The median swing is 26% / 19% daily and
  7% / 6% hourly.
- **Major turn:** a reversal of at least 3 × σ × √40.

The turns are labelled with hindsight. The readings at each turn are what the chart showed live.

### Real or failed

This is the question a trader can act on. The candidates are every 20-bar low and high.

- **Turned:** price reversed by the swing amount before making a new extreme.
- **Failed:** price made a new extreme first.

Only 11% of daily 20-bar lows turned. Because every candidate is already at a 20-bar extreme,
the "price fell" part of the reading is held equal across both groups.

### Readings checked

- **Risk (inner tape):** calm, normal or stressed.
- **Risk (outer tape):** rising or not.
- **Trend (inner tape):** up, flat or down, and its colour.
- **Stretch ladder (outer tape):** its colour.
- **Column:** its colour.
- **Close vs midline:** above or below.
- **Cone:** below, inside or above.
- **Ledger:** the bar's proven state.

## Findings

### 1. The fingerprint at a turn is mostly price itself

At every swing bottom the column, the ladder and the midline are red. At every top they are
green. This is true by construction: at the lowest close, the 10-bar move is negative, so these
readings tell you nothing beyond "price fell".

The readings that are not tied to the price path:

| At swing turns (daily) | Bottoms | Tops | All bars |
|---|---|---|---|
| Risk stressed | **51%** (major bottoms: 73%) | 26% | 31% |
| Risk rising | **40%** | 20% | 19% |
| Trend down | **50%** (major bottoms: 90%) | 1% | 15% |
| Trend up | 6% | **82%** | 44% |

Bottoms tend to form in stressed, rising risk inside a downtrend. Tops tend to form in calm
risk inside an uptrend. Hourly data shows the same pattern, a little weaker.

### 2. No reading tells a real bottom from a failed one

This is the central result. At 20-bar lows, the share that turned is the same whatever the pane
shows:

- **Single readings:** every lift is between 0.93 and 1.06, daily and hourly, for both lows and
  highs. For example, risk stressed vs calm at a daily low: 10.7% vs 11.3% turned.
- **The one exception is unreliable:** a green column at a daily low turned 28% of the time.
  But it covers only 1.4% of lows, and it falls to a lift of 1.36 hourly, on just 88 cases.
- **Combinations:** the best mixes of all the colours reach lifts of only 1.1–1.3. The first
  half of the dates barely predicts the second. The correlation is 0.25 to 0.33 for lows, and
  negative (−0.39) for daily highs. That is selection noise, not a pattern.

So the stressed-risk, downtrend picture in Finding 1 describes **every** 20-bar low, not just the
one that holds.

### 3. The reverse check: the fingerprint is common, so it is weak evidence

The strongest single reading is "closed below the band". It is within ±2 bars of a swing bottom
10.9% of the time daily and 8.5% hourly, against a base of about 2.4%. That is 4.6 times more
likely, but roughly nine times in ten it is not a bottom.

The next strongest:

- **Trend down:** within ±2 bars of a bottom 7.1% of the time.
- **Risk rising:** 4.9%.

### 4. Timing: how late each tape confirms

| After a swing bottom | Daily, median bars (IQR) | Hourly, median bars (IQR) |
|---|---|---|
| Column turns green | 5 (3–7) | 5 (3–7) |
| Close crosses above the midline | 6 (4–8) | 5 (3–8) |
| Ladder turns green | 6 (4–10) | 6 (4–10) |
| Trend turns green | 9 (0–27); 23% were already green | 17 (3–37) |

Tops mirror these. The column and ladder turn in about 5 bars, while the trend takes about 20.

**Sets B and C at turns:**

- **Set B** fires near only 37–49% of swing turns, at a median of 18–27 bars after the turn.
  When it does, about **two thirds** of the move to the next turn is still ahead: 67–71%
  (IQR 48–83%).
- **Set C** fires near 15–27% of turns, later (+35 to +39 bars), with about half the move left
  (48–59%).

### 5. The shape around a turn

See `out/paths_daily.png` and `out/paths_hourly.png`.

**Around bottoms:**

- **Risk:** risk's rank climbs into the bottom (about 44 to 71 by one bar after). Risk momentum
  spikes at the turn, then falls below zero within about 10 bars. Risk peaks at the bottom and
  eases after it.
- **Trend:** the trend reading is at its low exactly at the turn.

**Around tops:**

- **Risk:** tops form in calm risk (rank about 33), and risk climbs after.
- **Trend:** the trend reading is at its high exactly at the turn.

**Tied to price:** the column and midline form a V shape with its tip 10 bars after the turn.
This is mechanical, because both measure a 10-bar move.

## What this means

- **The pane describes a turn but does not call it.** Bottoms happen in stressed, rising risk
  in a downtrend, but so does every failed low. Nothing on the pane, alone or combined, picks the
  low that holds. This agrees with the earlier Edge Study and flip study, which found no
  directional edge in any single-chart state.
- **The pane confirms a turn, at a known delay.** The column and ladder take the new side about
  5–6 bars after a swing turn, and the trend tape about 10–20. A trader who waits for the column
  and the ladder gives up about 5 bars. Set B, the stricter confirmation with calm risk, gives up
  about a third of the move and catches fewer than half of turns.
- **One lead worth testing on its own:** after bottoms, risk momentum falls fast. Risk easing
  after a spike is a different question from "is this the low". This study only shows it after
  the fact; it has not been tested as a forward signal.

**Caveats:**

- Turns are labelled with hindsight; the readings are not.
- Instruments are correlated, so the event counts overstate independent evidence. The real vs
  failed result is a null result, and that only makes it more secure.
- Swing size is sized to each instrument's own risk, so crypto and bonds are judged on the same
  scale.
