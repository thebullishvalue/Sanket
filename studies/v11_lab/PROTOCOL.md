# Signal lab v11: protocol, declared before any candidate was scored

**Goal:** find a signal that beats sets A, B and C and actually works. This file fixes, before
any result is seen, what is tested, how it is judged, and what "beats" and "works" mean. Nothing
here changes after the run.

## Universe and judge

**Data:** the Indicator repo's 151 instruments.

| Run | Bars | Role |
|---|---|---|
| Daily | 2003–2026 | The primary run. Every candidate is tested here. |
| Hourly | 2024–2026 | Replicates the K candidates. |
| Weekly | Resampled from daily | Replicates the P candidates. Pragati's engine runs only on daily and weekly charts. |

**Judge:** Sanket's Edge Study, `edge.measure`, unchanged.

- Entry on the bar after the signal, held for 10 bars.
- Each instrument's trailing drift is removed (causally), and returns are volatility-normalised.
- Intervals come from a block bootstrap over dates.
- Costs are 3 bp.
- One call measures sets A, B and C and every candidate together, so all share the same
  symbols, dates and discovery/holdout split (the first 60% of event dates, then the last 40%).

Every candidate fires at most once per 10 bars, on ▲ and on ▼.

## Candidates

### From the v9 audit (Pragati's conviction × value grid)

| | ▲ | ▼ |
|---|---|---|
| **P1** v9.3's shipped signal | The capitulation turn: the first bar in DOWN · cheap with value reverting (phase +1), not in conviction's quiet regime, stack ready | Entering DOWN · rich (distribution) |
| **P2** | Entering DOWN · cheap, stack ready | — |

### From the turning-point studies (pane, VWAP and profile)

| | ▲ | ▼ |
|---|---|---|
| **K1** capitulation | A new 20-bar low with relative volume ≥ 2, risk stressed (rank ≥ 66.7) and the close below the band | The mirror: a new 20-bar high, volume ≥ 2, stressed, above the band |
| **K2** capitulation, confirmed | K1 within the last 10 bars, and the column turns green now | Mirror |
| **K3** trend pullback | Trend up (T > 0.5), and the close enters below the band | Trend down (T < −0.5), and the close enters above the band |
| **K4** pullback, confirmed | Trend up, below the band within the last 10 bars, and the column turns green now | Mirror |
| **K5** calm breakout | Risk calm (rank < 33.3), T ≥ 0, relative volume ≥ 1.2, and the close crosses above the prior session's VAH | Calm, T < 0, volume ≥ 1.2, and the close crosses below the prior session's VAL |
| **K6** cross-sectional reversal | Among equities (US, NSE and indices; at least 30 names that date), the name enters the bottom decile of the surprise column | The name enters the top decile |
| **K7** strong trend entry | T crosses above +0.5 | T crosses below −0.5 |

## Pass rules

A candidate side must pass all four stages.

| Stage | Test |
|---|---|
| **S1 Discovery** | Daily discovery-era edge, with the lower 95% bound above 0. |
| **S2 Holdout** | The daily verdict is CONFIRMED: the holdout lower bound is above 0 and the net edge (after costs) is above 0. |
| **S3 Replication** | On the replication timeframe (hourly for K, weekly for P), the full-period lower bound is above 0, on the same side. |
| **S4 Breadth** | Daily holdout mean score above 0 in at least 2 of the 3 groups (equity, macro, crypto) with 30+ holdout events; **and** the daily full-period mean score above 0 in all three eras (2003–12, 2013–19, 2020–26). |

**"Works"** means it passes S1–S4. **"Beats all"** means it works and sets A, B and C do not.
The same four stages are applied to the sets. If several candidates pass, the one with the
highest daily holdout net edge is the signal.

There are 16 candidate sides. On their own, the holdout and replication tests would let roughly
one through by chance. The breadth stage across groups and eras is what guards against that.
If nothing passes, the answer is no, and it is reported as no.
