# Signal lab v11: a signal that beats sets A, B and C and holds on data it had never seen

## The signal

> **▲ When the trend tape reads down (T < −0.5) and the close breaks above the landing band, buy
> on the next bar and hold for 10 bars.**

On the Pragyam pane:

- **The trend condition:** the inner trend tape is in its down state.
- **The trigger:** the first close above the upper edge of the cone drawn 10 bars ago.

It fires at most once per 10 bars. This is a counter-trend break: a name in a downtrend makes a
10-bar move larger than its own risk forecast allowed for, and that move tends to continue.

It is long-only. The mirror short (a break below the band in an uptrend) was tested alongside it
and was not confirmed.

## Results on data the search never saw

572 stocks from today's S&P 500 and Nifty 200, minus every name used before. Daily, 2005–2026.
The tests used 98.3% intervals because three hypotheses were tested at once.

| | Edge, 10 bars (σ) | 98.3% interval | Net of 3 bp | Hit | US | NSE | 2005–12 | 2013–19 | 2020–26 |
|---|---|---|---|---|---|---|---|---|---|
| **H2 ▲ (the signal)** | **+0.259** | **[+0.097, +0.407]** | +0.254 | 59.3% | +0.260 | +0.254 | +0.247 | +0.257 | +0.275 |
| Set A ▲ | −0.074 | [−0.31, +0.10] | | 42% | | | | | |
| Set B ▲ | −0.048 | [−0.10, +0.00] | | 48% | | | | | |
| Set C ▲ | −0.053 | [−0.10, −0.01] | | 48% | | | | | |
| Control: buy any new 20-bar low | +0.009 | [−0.07, +0.09] | | 51% | | | | | |

**Scale:** 4,137 events on 558 names, about 2,000 independent observations.

**Raw returns:** a mean 10-bar return of **+1.70%** per trade, before removing drift.

## How it was found

Each step was committed before the next one ran, so none of the results could have steered the
step before it.

1. **The turning-point studies** (`studies/v10_turns/`). No reading on the pane, VWAP or the
   profile picks the low that holds.
2. **The v11 lab** (`PROTOCOL.md`, then `lab.py` and `evaluate.py`). 16 declared candidate sides
   faced four stages on the 151-instrument research set: discovery, holdout, replication on a
   second timeframe, and breadth across groups and eras.
   - **Nothing passed.**
   - **The near miss:** Pragati capitulation (P1) was positive everywhere but underpowered.
   - **The surprise:** "buy a pullback below the band in an uptrend" and "sell a rally above the
     band in a downtrend" were both significantly wrong, in discovery, in the holdout and on
     hourly data.
3. **The confirmation** (`CONFIRM.md`, written before any fresh data was fetched; then
   `fetch_fresh.py` and `confirm.py`). It tested three hypotheses on the fresh universe, with
   Bonferroni intervals and a survivorship control:
   - **H1**, Pragati capitulation: +0.121, not confirmed (lower bound −0.031).
   - **H2**, a band break up in a downtrend: **confirmed**.
   - **H3**, a band break down in an uptrend, sold: +0.082, not confirmed (lower bound −0.005; NSE
     only +0.010).

## Checks against the obvious objections

- **Survivorship.** Today's index members omit names that were delisted, and H2 buys beaten-down
  names. Three checks address this:
  - **The control:** buying any new 20-bar low under the same tailwind made only +0.009σ.
  - **Instruments that cannot be delisted:** on the research set's commodities, FX and bonds,
    H2 makes +0.120σ daily (t 2.0) and +0.133σ hourly (t 3.4). On crypto it makes +0.200σ hourly
    (t 3.9).
  - **Equities on the research set:** +0.189σ daily (t 2.7) and +0.122σ hourly (t 3.1).
- **Data-mining.** The direction was chosen after the lab showed its opposite failing, which is
  why the claim rests only on the fresh universe.
  - **Locked in advance:** the rule, the threshold (T < −0.5, Pragyam's own trend state) and the
    pass bar were all committed before the fresh data was fetched.
  - **Consistent everywhere:** it is positive in both markets and all three eras.
- **One era or a few names.** No. The eras range from +0.247 to +0.275, the markets from +0.254
  to +0.260, and 558 names fired it.
- **Costs.** About +0.26σ over 10 bars is roughly +1.5% for a typical stock, against a 3–20 bp
  round trip.

## What it is not

- **Not a short.** The mirror short failed confirmation.
- **Not a forecast for every downtrend name.** It fires about 1.4 times per 1,000 bars, roughly
  once every three years per name (1,307 events across 151 instruments in 23 years of daily data).
  About 40% of trades lose.
- **Not tested on intraday stops or sizing.** The measurement is a 10-bar hold from the next bar,
  as the Edge Study defines it.
