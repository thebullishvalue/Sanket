<div align="center">

# SANKET

**संकेत** · *signal, indicator, forewarning*

Market Signal Screener · Pragati · Conviction × Value

[![Version](https://img.shields.io/badge/version-10.4.10-2B5FD9)](CHANGELOG.md) [![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)](requirements.txt) [![Streamlit](https://img.shields.io/badge/streamlit-1.52.2-FF4B4B)](requirements.txt) [![License](https://img.shields.io/badge/license-proprietary-6B7280)](LICENSE) [![Pragyam Family](https://img.shields.io/badge/pragyam-family-0F1217)](#the-pragyam-family)

</div>

> What fired, what is about to, and whether it has paid: a universe screened with Pragati,
> with every signal set's out-of-sample record measured on the names you put on screen.

## Overview

Sanket screens a universe with **Pragati** (`pragati.pine` v9.3) — the indicator Pragyam's
Conviction-Value Grid already reads — and asks of every name the indicator's own question:
**is the push paid for, and at what price?**

It reports three things, and keeps them apart:

- **Events** (three signal sets, all on screen at once, all read from Pragyam's engine).
  **A** — entering a state the name's own record has proven (rare). **B** — the surprise column
  flips, the close past the midline, risk calm and not rising, the stretch ladder agreeing.
  **C** — B with the trend agreeing. A signal board shows each set's events, its watch and its
  own measured verdict side by side; each event row lights the sets that fired it.
- **State.** Every name's cell in the **3 × 3 conviction-value grid** (v8), named as an action —
  Buy · Accumulate · Hold · Wait · Trim · Exit — with the measured v8 units, the same units
  Pragyam's book now sizes from. Where the name stands *between* events.
- **Evidence.** The out-of-sample expectancy of the event set **on the symbols you put on
  screen**, measured every day by the built-in Edge Study, with the interval and the power stated.

> **Read this first.** Sanket is **decision-support**, not a turnkey strategy.
> 1. **A triangle is proven per name, not across names.** The v9 audit
>    ([`studies/pragati_v9_audit.md`](studies/pragati_v9_audit.md); 380 instruments) found the
>    readings — conviction, value, the trace, its histogram, both tapes — carry almost no timing
>    information of their own. The v10 events are Pragyam's ledger: a ▲ says this name's own past
>    has proven the state it just entered. Pooled across 151 instruments, no state reached
>    |t| = 1.3 outside crypto, and the rule fires about once per 5,500 bars. The Edge Study measures
>    whether per-name proofs carry forward on your symbols.
> 2. **The grid is a weight, not a forecast.** Its v8 units were chosen before 2018 and confirmed
>    after on 380 instruments, and in Pragyam's allocator they beat the seed units in both eras on
>    Nifty 50 and Dow 30 — but no single t clears 2 there. Sanket shows it as the indicator's own
>    reading of where a name stands; the screener does not rank by it.
> 3. **Scope is earned per universe, never inherited.** Nothing about your symbols is hardcoded;
>    until the Edge Study measures, the app says "not measured".

## What It Does

1. **Single Date Screener** — **Signals** (the signal board — sets A, B and C side by side, each
   with its events, its watch and its measured verdict — then every event of the last 5 bars by
   age, its sets lit, with its grid state, push, tapes and evidence) · **Watchlist** (names one flip
   from B or B·C, nearest first, and names in a proven state) · **Grid** (the 3 × 3 census, names by
   action) · Signal Strength (the ranking) · System Data (exports, raw frame, Edge Study).
2. **Historical Range** — the three sets compared on the range (what followed each set's events,
   beside its Edge Study verdict), events per set, breadth, the watch over time, grid breadth
   (build vs cut), the universe-mean tapes, regime context, forward-return labels, Excel export.
3. **Correlation Analysis** — cross-asset correlation, with confluence = |correlation| × the
   normalised Pragati priority.
4. **Pulse Narrative** — every name's state, both sides, the signal board, plus the Watchlist,
   Grid and Strength tabs.

## Quick Start

```bash
git clone https://github.com/thebullishvalue/Sanket.git
cd Sanket
pip install -r requirements.txt
streamlit run sanket.py
```

Opens at `http://localhost:8501`. There is nothing to configure and nothing to tick: every
indicator input is the Pine's own default, and the **Edge Study runs on every run**. It fetches
~15 years (and the macro drivers behind them) the first time it sees a universe on a given day and
reuses that measurement for the rest of the day — within one calendar day the study reads
identical data, so re-measuring would return a bit-identical answer. It re-measures once the date
rolls. `scipy` is required (the value engine's Šidák floor).

## Project Structure

```
sanket.py           Streamlit entry point: UI, data + macro-driver fetch, screen routing
engine.py           settings, per-symbol features, the snapshot row, ranking, cost gate
pragati.py          pragati.pine v9.3: conviction, ladders, trace, histogram, tapes (readings only)
ledger.py           the signals: Pragyam's engine (risk, trend, cones, ladder, ledger) and sets A / B / C
samanvaya.py        the value engine (Samanvaya, section 4c), carried from Pragyam
cvgrid.py           the 3 × 3 conviction-value grid: the state engine, names, units, tones
pragati.pine        the indicator itself, v9.3: the Pine the port mirrors (earlier versions are in git history)
intraday.py         the conviction ladder's lower frames from yfinance (1m … 4h), batched and cached
charts.py           chart builders: the conviction-value map, tone history, correlation heatmap
edge.py             measured expectancy: event study, scored against peers, block bootstrap, power
logger.py           the family console: structured terminal logging
ui/                 theme.py · theme.css · components.py (the family design system)
docs/               the guide, the architecture and its evidence, the release notes
studies/            the audits and labs behind every design decision (v9_lab … v13_lab)
```

The **regime engine** (HMM + GARCH + CUSUM) and the **order-flow layer** (inferred delta, CVD,
volume profile, absorption) are unchanged and remain context only.

## Documentation

| Document | What it covers |
|---|---|
| [Guide](docs/GUIDE.md) | How Pragati reads a name, the stack, the signals, the grid, ranking, the Edge Study, the outputs, coverage, the UI system |
| [Architecture](docs/ARCHITECTURE.md) | The stack, the evidence and the design rationale, with known limitations |
| [Release Notes](docs/RELEASE_NOTES.md) | What changed in each major release, and why, from v5 to v10.4 |
| [Pragati v9 audit](studies/pragati_v9_audit.md) | The 380-instrument audit the v10 signals rest on |
| [Pine audit](studies/pine_audit.md) | The port checked against the indicator itself |
| [Changelog](CHANGELOG.md) | Every release, newest first |

## The Pragyam Family

Eleven Streamlit applications built on one design system and one set of components, so a
panel, a table, a chart or a number reads the same in each.

| Product | What it is |
|---|---|
| [**Tattva**](https://github.com/thebullishvalue/Tattva) | Cross-asset fair value, with a call only where its own record supports one |
| [**Nishkarsh**](https://github.com/thebullishvalue/Nishkarsh) | Nifty 50 inference: a top-down forecast and bottom-up breadth, read as one |
| [**Arthagati**](https://github.com/thebullishvalue/Arthagati) | Valuation-anchored market sentiment, tested on data it never saw |
| [**Vyapti**](https://github.com/thebullishvalue/Vyapti) | Market breadth: where participation stands, which way it moves, and what that has meant |
| [**Pragyam**](https://github.com/thebullishvalue/Pragyam) | Portfolio curation that spreads risk rather than predicting returns |
| **Sanket** · *this repository* | A market signal screener: what fired, what is about to, and whether it has paid |
| [**Vaaydo**](https://github.com/thebullishvalue/Vaaydo) | An F&O trade screener on option positioning, with futures tickets |
| [**Anumaan**](https://github.com/thebullishvalue/Indicator) | A chart terminal: calibrated risk, stops and exposure, audited on the chart it is on |
| [**Nivesa**](https://github.com/thebullishvalue/Nivesa) | A bond portfolio ledger: cost, yield, duration and cashflows across accounts |
| [**Samhita**](https://github.com/thebullishvalue/Samhita) | A portfolio tracker: live prices, performance and risk against the benchmark |
| [**Swing**](https://github.com/thebullishvalue/Swing) | A portfolio tracker for the swing book, on the same engine as Samhita |

## License

Proprietary. © 2026 @thebullishvalue. All rights reserved. The source is visible here
for reference only: viewing it grants no right to use, copy, modify or distribute it.
See [LICENSE](LICENSE).

## Disclaimer

Sanket is a research and decision-support tool, not investment advice. Nothing it shows is
a recommendation to buy or sell any security, and past or measured results do not guarantee
future ones. Every decision made with it is your own.

---

<div align="center"><sub>© 2026 @thebullishvalue · Part of the Pragyam product family</sub></div>
