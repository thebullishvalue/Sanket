# Confirmation test, declared before any fresh data was fetched

The v11 lab (PROTOCOL.md) passed nothing, but two things in it need one clean confirmatory test.
Neither can be claimed from the lab itself.

1. **P1 ▲, Pragati's capitulation turn (v9.3).** It was positive in every era, group and
   timeframe: +0.14σ in discovery, +0.13σ in the holdout, and +0.11σ weekly. The v9 audit found
   the same on 380 other instruments. But its intervals touch zero, so it failed on power, not
   on sign.
2. **K3 / K4 inverted, band breaks against the trend continue.** Buying a pullback below the band
   in an uptrend, and selling a rally above it in a downtrend, were both significantly wrong: K3 ▲
   holdout −0.175σ with interval [−0.300, −0.052], and hourly agrees. The opposite trade is a
   hypothesis chosen **after** seeing that result, so only fresh data can test it.

## Fresh data

The constituents of the S&P 500 (Wikipedia) and the Nifty 200 (niftyindices.com) as of today,
**minus every name already in the research set**. Daily bars from 2005 to 2026, via yfinance.
None of this data has been used in any study here.

**Survivorship caveat.** These are today's members: names that were delisted are missing, which
flatters any buy-the-dip long. So a control shares that tailwind, and H1 has to beat the control,
not just zero.

## Hypotheses

Exactly three, plus a control. Every signal fires at most once per 10 bars.

| | Rule |
|---|---|
| **H1** P1 ▲ | The first bar in Pragati's DOWN · cheap cell with value reverting (phase +1), not in conviction's quiet regime, stack ready. As shipped in v9.3. |
| **H2** band break up in a downtrend, ▲ | T < −0.5 and the close enters above the landing band. |
| **H3** band break down in an uptrend, ▼ | T > +0.5 and the close enters below the landing band. |
| **C0** control | ▲ on any new 20-bar low; ▼ on any new 20-bar high. |

Sets A, B and C are measured alongside, for the record.

## Judge and pass rule

**Judge:** Sanket's Edge Study, unchanged except for the interval level. With three hypotheses,
each gets a **98.3%** interval (Bonferroni: 1 − 0.05/3). The whole fresh period counts as out of
sample, so the **full-period** result is the test.

A hypothesis is **confirmed** only if all of these hold:

- The full-period 98.3% lower bound is above 0.
- The net edge (after 3 bp) is above 0.
- Its mean score is positive in **both** markets, US and NSE.
- Its mean score is positive in at least 2 of 3 eras: 2005–12, 2013–19 and 2020–26.
- **For H1 only:** its lower bound is above the C0 ▲ point estimate, so it beats plain
  buy-the-dip under the same survivorship tailwind.

A confirmed hypothesis becomes the proposed signal. If none is confirmed, the answer is no.
