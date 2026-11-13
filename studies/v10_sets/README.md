# Signal Sets A / B / C: Edge Study (v10.1.0)

The question: does any of Sanket's three signal sets have an edge?

**Answer: no set has an edge you can trade.** Set A is the only one that ever reads as
significant, and it has too few events, spread over too few names, to count. Sets B and C fire
often enough to measure properly, and neither predicts anything. The C buy signal actually points
the wrong way.

## Data and method

- Universe: the Indicator repo's research data, 151 instruments: US and NSE stocks, indices,
  commodities, FX, bonds and crypto.
  - Daily: 945k bars, 2002–2026.
  - Hourly: 1.13M bars, 2024–2026.
- Signals come from `ledger.py`, the same code the screen uses to fire, run on each instrument's
  OHLC.
- Verdicts come from `edge.measure`, the same code as the app's Edge Study:
  - Entry on the bar after the signal, held 10 bars.
  - Each symbol's trailing drift is removed causally, and returns are vol-normalised.
  - Confidence intervals come from a block bootstrap over dates. The cost is 3 bp.
  - The first 60% of event dates are the discovery half; the last 40% are the holdout.
- On top of that, the study breaks results down by era and asset class, and runs each set as a
  10-bar holding rule against buy-and-hold.
- Run it with `python3 studies/v10_sets/sets_edge.py <Indicator/research>`. The full output is in
  `sets_edge.txt`.

## Verdicts

Edge is the gross, drift-free, vol-normalised 10-bar move in the signal's direction, in σ.

| Set · side | Daily edge (events) | Daily verdict | Hourly edge (events) | Hourly verdict |
|---|---|---|---|---|
| A ▲ proven + | +0.31 (84) | UNDERPOWERED: holdout +0.23, not significant | +0.20 (251) | UNDERPOWERED |
| A ▼ proven − | −0.15 (88) | UNDERPOWERED, anti | +0.01 (237) | UNDERPOWERED |
| B ▲ flip | −0.02 (11,044) | NO EDGE | −0.01 (13,147) | NO EDGE |
| B ▼ flip | +0.04 (12,444) | NO EDGE | +0.04 (13,874) | DISCOVERY ONLY: holdout +0.006 |
| C ▲ flip + trend | −0.03 (8,743) | ANTI-PREDICTS | −0.06 (8,354) | ANTI-PREDICTS in both halves |
| C ▼ flip + trend | −0.04 (4,334) | NO EDGE | −0.02 (7,717) | NO EDGE |

## What each set's result means

### Set A

Set A's positive number comes from one instrument at each timeframe:

- **Daily:** NTPC.NS accounts for 66 of the 84 buy events. Only 5 names ever fire a daily buy.
- **Hourly:** XRP-USD accounts for 212 of the 251 buy events.

The events also repeat: a name that sits in a proven state re-enters it many times. So the
+0.3σ describes NTPC and XRP, not the rule. The ledger proves states rarely and only on a few
names (about 0.1–0.2 events per 1,000 bars), and that is too thin to support a population-level
claim.

### Set B

Buy has no edge on either timeframe. The +0.04σ on sell held in the hourly discovery half
(+0.064) but fell to +0.006 in the holdout, and it is negative after costs.

### Set C

Adding the inner trend tape makes the buy side worse:

- Daily: −0.033σ, with the 2020–26 era at t = −2.0.
- Hourly: −0.058σ, t = −3.9, anti in both discovery and holdout.

This repeats what the Indicator flip study found: confirming a flip with trend means buying
short-term strength that then mean-reverts. The effect is small (about 0.05σ), so trading it in
reverse would not clear costs either.

### As trading rules (mean across instruments)

- **Daily, long-only:** every set has a Sharpe of about 0.12–0.16, against 0.48 for
  buy-and-hold. The sets beat holding on at most 20% of names. Their smaller drawdowns only
  reflect being out of the market 90%+ of the time.
- **Hourly:** B and C have negative Sharpe in every mode. A long-only has a positive Sharpe on
  every name it trades, but it trades only 3 names.

## Bottom line

None of the sets is a buy or sell edge. A, B and C describe chart states; they don't forecast
them. What the engine does measurably well is forecast risk: the cone and the vol regime are
calibrated. The trend and flip readings carry no directional information on their own.

What this means for the app:

- Keep Set A as the default. It is the only set built to fire only when the ledger has proven
  something, and it rarely fires.
- Treat B and C as descriptive filters, not entries.
- Do not use the C buy signal as a long entry.
