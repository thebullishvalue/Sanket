# Improving sets A, B and C: protocol, declared before any variant was scored

The question is whether the evidence gathered so far can make the existing sets better. Every
variant here comes from a finding already on record. The variants are chosen on the research set
and confirmed on fresh data where the sets have been measured but never tuned. This file is fixed
before any variant is scored.

## Evidence the variants rest on

- **E1. Trend agreement hurts.** C ▲ (B ▲ plus trend up) is worse than B ▲ on daily research,
  hourly research and the fresh stocks. C ▼ is worse than B ▼ on all three. Buying a pullback in
  an uptrend (K3 and K4 in v11) was significantly wrong. The confirmed band-break signal is a long
  *in a downtrend*.
- **E2. B ▼ is the most consistent set:** +0.036 daily, +0.040 hourly (significant) and +0.051 on
  the fresh stocks. B ▲ is near zero or negative.
- **E3. Turns happen in stressed, rising risk** (`studies/v10_turns`), while B fires only in calm,
  falling risk. Whether that filter helps has never been tested.

## Variants

Each variant changes **one** condition of the current rule. Every condition not named stays as it
is in `ledger.py`. All variants fire on the same flip bar, so the counts are directly comparable.

| | ▲ rule | ▼ rule | Rests on |
|---|---|---|---|
| **B0** (current) | Flip up · close above the midline · risk calm and not rising · ladder ≥ 0 | Mirror | — |
| **Bn** | B0 without the risk filter | Mirror | E3 |
| **C0** (current) | B0 with the trend tape agreeing (T ≥ 0) | B0 with T < 0 | — |
| **Cx** | B0 with the trend tape **opposing** (T < 0) | B0 with T ≥ 0 | E1 |
| **Cx5** | B0 with the trend tape in the opposing **state** (T < −0.5) | B0 with T > +0.5 | E1 (the confirmed signal's threshold) |
| **Cxn** | Cx without the risk filter | Mirror | E1 + E3 |

Set A stays as it is and is reported only, because no finding supports a specific change.

## Stage 1: choose, on the research set only

The research set is the 151 instruments, daily and hourly. For each set and side, one variant is
chosen:

- **New B:** from {B0, Bn}.
- **New C:** from {C0, Cx, Cx5, Cxn}.

The choice is the variant with the highest **minimum** of its daily and hourly full-period edges,
which rewards consistency over a single good run. A variant can only replace the current rule if
its minimum is positive. Otherwise the current rule stays.

## Stage 2: confirm, on fresh data only

Two fresh datasets:

- **Daily:** the 572 fresh stocks from v11, 2005–2026.
- **Hourly:** the same stocks, over the last 730 days, which is all yfinance serves. This data has
  never been fetched before.

Each chosen change, compared with the current rule on the same fresh data, needs:

- **A better edge:** the variant's edge beats the current rule's edge, on both fresh daily and
  fresh hourly.
- **Significance, for a "works" claim:** a fresh daily lower bound above 0. The interval level is
  1 − 0.05/k, where k is the number of changes put to confirmation.

How each outcome is reported:

| Outcome | Reported as |
|---|---|
| Better on both, and significant | Confirmed: it works and it is better |
| Better on both, not significant | Better, but not proven to work |
| Anything else | Not an improvement; the current rule stays |

## The judge

Sanket's Edge Study, unchanged except for the interval level: entry on the bar after the signal,
a 10-bar hold, drift removed, volatility-normalised, block bootstrap over dates, 3 bp costs.
