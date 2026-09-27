# Pragati v9 · a fresh audit, the measuring stick fixed, and the signals moved to the edge

**Scope.** Everything in `pragati.pine` v8: its purpose, the engines (conviction, Samanvaya's value, the trace and its histogram, the two ladder tapes and their momentum tapes), every signal (▲▼ TURN, ◆ continuation, R and H divergence, the standing declaration), the Conviction-Value Grid (cells, grading, 5 × 5 phases, the push-gated row), the open-interest readout and its gold cast, and the presentation (header, tooltips, panel, readout, alerts). Measured through the Python port (bar-for-bar checked) on **380 instruments** — 237 NSE stocks, 40 US large caps, 38 world indices, 23 commodities, 23 FX pairs, 19 cryptocurrencies — on ~20 years of daily bars, and again on **weekly** bars. OI on 187 NSE F&O stocks, 2019–2026.

**The rule for a change.** It must hold in **all three eras** — E1 2006–13, E2 2014–19, E3 2020–26 — and in most non-crypto classes, on both scorers below. v8's before/after-2018 holdout had been looked at several times; three eras is the stricter test.

Reproduce: `studies/v9_lab/` (see its README).

---

## 1 · The measuring stick was biased — fixed first

v8's audit (and the app's Edge Study up to v8.4) scored each reading against forward returns **minus the era's own mean return**. That mean is computed from the whole era, so it knows the future. On pure random walks (same instruments, same calendar, no predictability at all), that alone produces:

| Random-walk null, avg of US / indices / commodities / FX | h = 10 E1 / E2 / E3 | h = 40 E1 / E2 / E3 |
|---|---|---|
| 20-bar momentum, **v8 stick** (returns demeaned in era) | −0.020 / −0.008 / −0.005 | −0.040 / −0.021 / −0.004 |
| distance from the 200-day mean, **v8 stick** | −0.029 / −0.012 / −0.012 | −0.059 / −0.025 / −0.028 |
| distance from the 200-day mean, position demeaned in era | −0.054 / −0.040 / −0.039 | −0.102 / −0.083 / −0.086 |
| 20-bar momentum, **v9 stick** (causal) | +0.001 / +0.009 / +0.001 | +0.022 / +0.006 / −0.010 |
| distance from the 200-day mean, **v9 stick** | +0.003 / +0.003 / −0.010 | 0.000 / −0.001 / −0.019 |

So the v8 stick leaned toward "everything reverts" — a persistent momentum reading scored as reversion with no information in it — and events that cluster in names whose era went badly (capitulations) were flattered.

**The v9 stick.** Two scorers, both free of look-ahead:

* **Time-series (tc).** Position = the reading, centred on its **own trailing 250-bar mean** (removes a reading's persistent bias without seeing the future). Return = the h-bar forward return from the next open, divided by **trailing** 60-bar volatility. Not demeaned. Block bootstrap over dates, blocks ≥ 60 bars (persistent readings otherwise get intervals that are too narrow). Reads ≈ 0 on random walks (above).
* **Cross-sectional (xs).** Return net of the group's **same-date** mean return (market-neutral); for stocks also the **rank IC** across names per date.

Every number below is on the v9 stick. `*` = the 95% block-bootstrap interval excludes zero. "avg" = mean of the five non-crypto classes.

---

## 2 · What the readings know on their own (tc, h = 10, avg E1 / E2 / E3)

| Reading as a position | E1 | E2 | E3 |
|---|---|---|---|
| chart conviction | −0.031 | −0.029 | +0.005 |
| MTF conviction tape | −0.028 | −0.021 | −0.012 |
| MTF value tape | −0.009 | −0.031 | +0.006 |
| value on the chart | −0.000 | −0.025 | +0.006 |
| trace (conviction × value) | −0.015 | −0.030 | +0.008 |
| trace histogram (in σ) | +0.011 | −0.003 | +0.006 |
| **grid units** | **+0.025** | **+0.046** | **+0.025** |
| benchmark: 20-bar momentum | −0.016 | −0.035 | +0.020 |
| benchmark: RSI(14) | −0.010 | −0.032 | +0.014 |

At h = 40 the grid units read +0.022 / +0.033 / +0.038. **The readings describe; they barely forecast.** Only the grid's units are positive in every era.

**Cross-sectionally (rank IC, h = 10, E1 / E2 / E3).** NSE: value −0.038 / −0.034 / −0.005 (t −2.5 / −5.4 / −0.2) — short-term reversal, strong before 2020 and gone since; grid units +0.031 / +0.033 / +0.006 (t 1.9 / 4.4 / 0.6). US: grid units −0.007 / +0.021 / +0.021. Distance from the 200-day mean (long-term trend) is positive in every era on both markets (NSE +0.026 / +0.025 / +0.016; US −0.003 / +0.033 / +0.023).

---

## 3 · The grid, cell by cell (tc, h = 10, avg E1 / E2 / E3)

| Cell | E1 | E2 | E3 |
|---|---|---|---|
| **Buy · capitulation** (DOWN · cheap) | **+0.057** | **+0.064** | **+0.069** |
| Accumulate · washout (DOWN · fair) | +0.031 | −0.001 | +0.030 |
| **Exit · distribution** (DOWN · rich) | **−0.085** | **−0.050** | **−0.067** |
| Accumulate · basing (FAINT · cheap) | −0.008 | +0.018 | −0.040 |
| Wait · idle | −0.009 | −0.001 | −0.007 |
| Trim · stalling | +0.042 | −0.003 | +0.009 |
| Buy · turned (UP · cheap) | +0.150 | +0.086 | −0.028 |
| Hold · building | −0.018 | −0.009 | −0.001 |
| Trim · paid | +0.015 | −0.000 | −0.009 |

Market-neutral (xs) agrees: capitulation +0.063 / +0.064 / +0.074, significant on indices and NSE in every era; distribution negative in every era. **The capitulation finding survives the corrected stick** — smaller than v8 reported, still positive in every era. The other cells are near zero, and the v8 units (which were also re-tested in Pragyam's own allocator — portfolio returns, unaffected by the stick) are left as they are.

---

## 4 · The signals (tc, h = 10 unless stated)

| Signal | E1 | E2 | E3 |
|---|---|---|---|
| ▲▼ TURN (v8) | +0.041 | +0.015 | +0.005 |
| benchmark: value crossing back through θ | +0.007 | +0.042 | −0.005 |
| R divergence | +0.020 | +0.011 | −0.010 |
| H divergence | −0.034 | +0.002 | −0.003 |
| the standing declaration, held | +0.013 | −0.016 | +0.004 |
| benchmark: RSI(14) < 30 | +0.034 | +0.052 | **−0.032** |
| benchmark: 20-bar fall past 1.5σ | +0.043 | +0.048 | +0.001 |
| capitulation state | +0.057 | +0.064 | +0.069 |
| … with value momentum **reverting** (5 × 5 phase +1) | +0.061 | +0.073 | +0.071 |
| … with value momentum still widening | +0.041 | +0.027 | +0.040 |
| ▲ TURN within 20 bars of a capitulation | +0.066 | +0.027 | +0.064 |
| ▲ TURN with no capitulation before it | +0.055 | +0.060 | **−0.037** |

**Weekly bars** (horizon 4 weeks; E1 has no data after warm-up, E2 / E3): ▲▼ TURN **−0.071 / −0.064**; R −0.005 / +0.003; declaration +0.009 / −0.024; capitulation state +0.105 / +0.111; grid units +0.065 / +0.058.

So: v8's TURN faded to nothing after 2020 and reads negative on weekly bars; it worked on daily only when a capitulation came first. R, H, the ◆ (off) and the declaration carry nothing. Plain oversold stopped working after 2020; capitulation did not.

---

## 5 · v9's ▲ — the capitulation turn

**Definition.** The first closed bar on which the grid stands in DOWN · cheap (the conviction tape past −30 and held there by conviction's own histogram; the value tape past −θ) with the value momentum tape **reverting** (the grid's 5 × 5 value phase +1), with a 10-bar cooldown.

| Capitulation turn (10-bar cooldown) | E1 | E2 | E3 | classes + | sig + / − |
|---|---|---|---|---|---|
| daily, tc, h 10 | +0.046 | +0.056 | +0.046 | 5 / 4 / 4 | 3 / 0 |
| daily, tc, h 20 | +0.053 | +0.056 | +0.065 | 5 / 4 / 4 | 4 / 0 |
| daily, xs, h 10 | +0.045 | +0.066 | +0.064 | 4 / 4 / 4 | 4 / 0 |
| weekly (3-bar cooldown), tc, 4 w | — | +0.140 | +0.074 | – / 5 / 4 | 2 / 0 |

5,405 events on the non-crypto daily panel over 20 years. **Refinements tested and not adopted** (none beat the plain event in its worst era across both horizons and both scorers): the trace's histogram above zero; conviction's push above zero; the conviction tape rising two bars; effort absorbed in the window (ties, fewer events); widening to the washout cell; the raw tapes without the push gate. The plain event is the simplest and was kept.

**Crypto** reads −0.34 / +0.01 (E2 / E3) — the stated exception, as in v8.

## 6 · v9's ▼ — distribution

No short-side EVENT held up: ▼ TURN +0.010 / −0.006 / +0.008; "paid" or "stalling" turning −0.028 … +0.035, sign-unstable; entering distribution 337 events in 20 years (55 weekly) — too rare to judge alone. The distribution STATE, held short, was positive in every era: tc h 10 +0.085 / +0.050 / +0.067, h 20 +0.050 / +0.140 / +0.102; weekly +0.080 / +0.440. v9's ▼ marks the entry into that state, and says it is measured as a state.

---

## 7 · Does each engine component earn its place? (ablations — not tuning)

Each variant rebuilt all 380 instruments. Worst-era value, capitulation state / capitulation turn (tc, h 10):

| Variant | cap % of bars | cap state worst era | cap turn worst era | grid units worst era |
|---|---|---|---|---|
| **defaults** | 2.2 | **0.057** | **0.046** | **0.025** |
| participation off | 2.7 | 0.054 | 0.044 | 0.016 |
| Effort denominator | 2.2 | 0.053 | 0.042 | 0.022 |
| value = RV leg only | 3.8 | 0.055 | 0.024 | 0.022 |
| value = breadth leg only | 0.7 | 0.069 | 0.055 | 0.018 |
| macro hedge off | 2.3 | 0.060 | 0.051 | 0.024 |
| inner zone 20 | 3.1 | 0.049 | 0.024 | 0.019 |
| inner zone 40 | 1.3 | 0.033 | 0.040 | 0.021 |
| θ = 1.0 | 4.4 | 0.054 | 0.037 | 0.023 |
| θ = 2.0 | 0.8 | 0.052 | 0.035 | 0.022 |

The edge is broad — it survives every variation — and the defaults sit at or near the best worst-era value on every measure. **No input changed.** Two notes for the record: the macro hedge changes nothing measurable here (Auto often lands at zero skill); breadth-only value is sharper but four times rarer.

Grading and phases: the graded units beat flat cell units on tc (daily +0.025 / +0.046 / +0.025 vs +0.020 / +0.021 / +0.020; weekly +0.065 / +0.058 vs +0.043 / +0.046); cross-sectionally mixed. Kept. Inside capitulation, sellers running vs pausing (the conviction phase) was mixed across eras — no change.

## 8 · Open interest (187 NSE F&O stocks; xs, h = 10; 2019–22 / 2023–26)

| | 2019–22 | 2023–26 |
|---|---|---|
| push under the gold cast (exits paid for it) | −0.046 | −0.019 |
| push without the cast | +0.012 | +0.009 |
| short covering | −0.048* | −0.025* |
| long build-up | −0.030 | −0.020* |
| long unwinding | −0.001 | +0.002 |
| capitulation, same names | −0.021 | +0.081* |

The gold cast is borne out (on by default since v8, colour only). Short-covering rallies lagged in both halves — the readout's gold for covering stands. But build-ups did not predict continuation either: "genuine" in the readout means risk was committed, not that the move carries on; its tooltip now says so.

---

## 9 · What v9 changed, and what it did not

**Changed (Pine and port, in parity — 0 mismatches over 81,623 bars):**

1. **▲ = the capitulation turn; ▼ = distribution**, both read from the grid (the Pine's new section 8b; `pragati.v9_signals` in the port). v8's TURN stays on `▲ ▼ source: TURN (v8, legacy)` / `signal_source="turn"`.
2. **R divergence off by default** (H and the ◆ already were).
3. **Every claim rewritten to what was measured** — header, tooltips, the panel's declaration row, the readout, alerts; the stale v5/v7 text found in the read-through (Ladder down called the default; R "the measured element, drawn solid"; the gold cast "off by default"; "the stack is unmeasured"). R and H markers are drawn lighter; the ▲▼ solid.
4. **The app's Edge Study now scores causally** (trailing drift and σ from returns realised before each event) — the v8 in-era demeaning flattered exactly the kind of event v9 fires.

**Not changed:** every engine and every numeric input; the grid's units; the ranking (stretch read as reversion — the unbiased IC agrees with its sign before 2020 and reads ≈ 0 since, as v8.2 reported).

**Limits.** Effects are small — about +0.05σ over 10–20 bars: a lean, not a trade, and far below what one symbol's Edge Study can resolve (it will usually read UNDERPOWERED). Crypto is the exception. Intraday is unmeasured (no free intraday history at depth). The `.pine` file is not compiled here — load it in TradingView and check it compiles.
