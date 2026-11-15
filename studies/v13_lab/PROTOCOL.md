# v13: a deeper search for upgrades to sets A, B, C and D

This protocol was declared before the exploration began.

## The constraint

| Dataset | Already used |
|---|---|
| The 151 research instruments | In every study since v10 |
| The 572 fresh daily stocks | Confirmed set D and the reversed set C |
| The 595 fresh hourly stocks | Confirmed set C |

Searching these four harder would find noise, so this round keeps exploration and confirmation
apart.

## Part 1: exploration, on the four used datasets only

The four datasets are research daily, research hourly, fresh daily and fresh hourly.

**Event pools.** Each set is studied through the whole family it belongs to, not just its
current firings:

| Pool | Events |
|---|---|
| FLIP ▲ / ▼ | Every bar the surprise column flips. Sets B and C live here. |
| BAND ▲ / ▼ | Every bar the close enters above (below) the landing band. Set D lives here. |
| STATE ▲ / ▼ | Every bar a name enters a ledger state with a z-score. Set A lives here. |

**Features on every event:**

- The trend T, the ladder and its rungs, the column and its depth over the prior 10 bars.
- The h-rung stretch, and the close against the midline.
- Risk: its rank and its momentum.
- The cone position, the ledger z.
- Volume, VWAP and the session value area.
- A new 20-bar extreme within 5 bars, and bars since the previous pool event.
- On daily only: Pragati's grid cell and its two tapes.

**Score.** Sanket's Edge Study score: the 10-bar return from the next bar, drift removed
(causally), volatility-normalised, in the event's direction.

**What counts as an upgrade.** A rule must:

- beat the set's current rule on its **worst** dataset (or the worse of the two daily datasets,
  for Pragati features);
- keep at least 0.3 events per 1,000 bars;
- be stated in the pane's own terms.

## Part 2: confirmation, on a universe never touched, fixed here before it is fetched

**The universe.** Today's constituents of the S&P MidCap 400 and S&P SmallCap 600 (Wikipedia),
and the Nifty 500 minus the Nifty 200 (niftyindices.com), minus every name in any dataset above.
It is fetched daily (2005–2026) and hourly (the last 730 days). Hourly is used only for rules that
need no Pragati reading.

**The test.** Each proposed upgrade (k of them in total) is measured against the current rule of
the same set and side, on the same data, by `edge.measure`, at a 1 − 0.05/k interval.

**Confirmation.** An upgrade is confirmed only if both hold:

- **Better:** its daily edge beats the current rule's daily edge, and its hourly edge does too
  (where hourly applies).
- **Works:** its daily lower bound is above 0.

**Partial results.** An upgrade that is better but not significant is reported as "better, not
proven", and is not shipped. Anything else is rejected.
