# v13: the search for upgrades found a flaw in the scoring, not a better signal

`PROTOCOL.md` set this round up as an exploration on the four datasets already used, followed by
one confirmation on an untouched universe. The exploration found that the Edge Study's own
score creates the "counter-trend works" pattern behind both v10.3 changes. Against same-day
peers, no set has an edge, and no reading separates good events from bad by a margin worth
trading.

## The flaw

The Edge Study's score is "drift-free": it subtracts each instrument's **trailing** 500-bar
average 10-bar return, then divides by the instrument's volatility.

A stock in a downtrend has a negative trailing average, so an ordinary next 10 bars scores as a
gain. A stock in an uptrend has a high trailing average, so an ordinary next 10 bars scores as a
loss. Any rule that fires in downtrends inherits a bonus it did not earn.

Two independent scorers were added (`scorers.py`, `xs.py`):

- **Raw:** the return in the event's direction, divided by volatility.
- **Peer-relative (xs):** the return minus the same-date average of the instrument's peer group
  (US stocks, NSE stocks, indices, commodities, FX, bonds, crypto), divided by volatility. It
  uses nothing from the instrument's own past.

### Flip ▲ events by trend, bottom to top fifth (research daily)

| Scorer | Bottom | | | | Top |
|---|---|---|---|---|---|
| Drift-free (the Edge Study) | **+0.090** | +0.019 | −0.034 | −0.058 | **−0.113** |
| Raw | +0.087 | +0.092 | +0.083 | +0.098 | +0.106 |
| Peer-relative | −0.018 | −0.024 | −0.039 | −0.048 | −0.045 |

The trend gradient exists only under the drift-free score.

### The current sets under all three scorers

Columns are research daily / research hourly / fresh daily / fresh hourly.

| | Drift-free | Raw | **Peer-relative** |
|---|---|---|---|
| D ▲ | +0.15 / +0.14 / +0.26 / +0.17 | +0.09 / +0.02 / +0.21 / +0.06 | **−0.031 / −0.023 / −0.009 / 0.000** |
| C ▲ (v10.3) | +0.09 / +0.10 / +0.03 / +0.11 | +0.08 / +0.02 / +0.04 / +0.04 | **+0.008 / −0.017 / −0.038 / −0.017** |
| C ▼ (v10.3) | +0.09 / +0.17 / +0.08 / +0.13 | −0.11 / +0.04 / −0.12 / −0.02 | **+0.058 / +0.006 / −0.009 / +0.002** |
| C ▲ (before v10.3) | −0.03 / −0.06 / −0.05 / −0.06 | | **−0.035 / −0.011 / −0.008 / −0.009** |
| C ▼ (before v10.3) | −0.04 / −0.02 / +0.01 / −0.00 | | **+0.024 / +0.005 / +0.024 / +0.017** |
| B ▲ | −0.02 / −0.01 / −0.05 / −0.01 | | **−0.037 / −0.013 / −0.012 / −0.011** |
| B ▼ | +0.04 / +0.04 / +0.05 / +0.05 | | **+0.044 / +0.001 / +0.007 / +0.008** |
| A ▲ | +0.31 / +0.20 / −0.07 / −0.17 (few events) | | +0.093 / +0.121 / −0.045 / −0.067 |

## What D really captures

D's raw gains come from days when many names fire together, which are broad market rebounds
after selloffs:

| D events by how many fired the same day | Raw | Peer-relative |
|---|---|---|
| Alone (fresh daily) | +0.045, below an ordinary upside band entry (+0.094) | −0.108 |
| More than 10 (fresh daily) | +0.426 | +0.050 |
| More than 10 (research daily, 58 events) | +0.644 | +0.117 |
| More than 10 (fresh hourly, the last two years) | **−0.054** | −0.056 |

So D is market timing on a handful of crash rebounds, not stock selection, and it failed in the
most recent window.

## The feature search, under the peer-relative score

All four datasets were searched: every pool (flips, band entries, state entries), every reading
(trend, ladder, column depth, stretch, midline, risk rank and momentum, ledger z, volume, VWAP,
value area, recency), and on daily data Pragati's grid, tapes and trace. The largest effect
consistent across all four datasets is a **0.03σ** gap between a reading's top and bottom fifths.
Pragati's trace and tapes do slightly better on the daily pair (0.03–0.04σ, stretched-up names
lagging), but that is still too small to trade. Under the drift-free score the same search finds
0.2σ gaps, all of which come from the trend artifact.

## Conclusions

1. **The v10.3 claims do not hold.** Reversing C and adding D both looked like improvements only
   because of the drift-free score. Against peers, neither the old nor the new C has an edge (the
   old one is marginally better on fresh data), and D is crash-rebound market timing.
   `studies/v11_lab` and `studies/v12_sets` used the drift-free score and carry correction notes
   pointing here.
2. **No upgrade was found to put to confirmation.** The untouched universe (S&P 400 and 600,
   Nifty 500 minus the Nifty 200; downloaded, never computed on) stays sealed for a future
   candidate.
3. **The Edge Study needs a better score.** For any universe with peers, the primary score should
   be peer-relative, with drift-free kept as a secondary reading. Otherwise it will keep calling
   counter-trend rules winners.
4. **The consistent picture across every study since v10:** the system forecasts risk well
   (the cone and the risk regime), and its readings carry little to no information about which
   stock will beat its peers over 10 bars.
