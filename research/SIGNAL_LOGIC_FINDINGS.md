# Sanket — Signal Logic Findings (data-derived)

**Status:** research findings, validated out-of-sample by walk-forward. **Gross of
transaction costs.** F&O Stocks (NSE, 211 symbols), daily, ~5 years (2020-06 →
2026-06, 251,521 bars, ~30k fired signals). One universe / one frequency — re-run
the harness on others before generalizing.

This documents what the data says the signal logic **should** be, versus the
hand-coded gates currently in `compute_signal_sets` (sanket.py). It supersedes
intuition with measured, walk-forward-validated edge.

## How this was derived (reproducible)
1. `research_edge.py --lookback-days 1825` — harvest 5y panel (production analysis +
   new orthogonal features + forward returns/MFE/MAE), cached to parquet.
2. `signal_efficacy.py` — do the GATED triggers beat random entry? (event study)
3. `signal_anatomy.py` — take the NAKED crossings, learn which factor conditions
   sort winners (single 70/30 split). **Single-split numbers OVERSTATE edge.**
4. `walk_forward.py --horizon {2,3,5,8,13}` — re-learn each gate on rolling 252d
   windows, test on the next 63d, across 15 windows. **This is the verdict.** A gate
   must be positive in ≳65% of forward windows AND pick stable factors to be real.

## The core methodological lesson
**A single train/test split lies on financial data.** The anatomy study (one split)
showed Threshold Long +1.63%, Threshold Short +1.24%, Momentum Short +1.04% with
t-stats of 4–5. Walk-forward demolished all three (→ +0.05%, −0.04%, −0.16%) — they
were the optimizer fitting the last 30%. Only walk-forward across many windows, and
across horizons, separates real edge from curve-fit. Trust nothing validated on one
split.

---

## Per-signal verdict (walk-forward learned-gate μ% / % positive windows)

| Signal | 2b | 3b | 5b | 8b | 13b | Verdict |
|---|---|---|---|---|---|---|
| **Crossover Long** | +0.27/67% | +0.31/80% | +0.90/87% | +0.78/73% | +0.77/73% | ✅ ROBUST (all horizons) |
| Crossover Short | −0.17/27% | −0.13/40% | −0.27/40% | −0.36/47% | −1.41/27% | ❌ dead (negative all) |
| **Momentum Long** | +0.39/73% | +0.46/67% | +0.50/53% | +0.84/73% | +1.10/67% | ✅ HOLDS (positive all) |
| Momentum Short | −0.07/47% | −0.18/47% | −0.16/60% | −0.63/20% | −1.73/33% | ❌ dead (negative all) |
| **Threshold Long** | +0.03/50% | −0.04/57% | +0.05/57% | +1.29/88% | +0.51/67% | ⚠️ edge ONLY at 8b |
| Threshold Short | +0.13/62% | −0.17/50% | −0.04/57% | −0.27/50% | −1.08/40% | ❌ dead (2b marginal only) |

**Summary: edge is entirely LONG-side and concentrated. Three keepers, three dead shorts.**

---

## What the signal logic SHOULD be (data-derived gates)

Naked triggers (no gates) — the starting point:
```
crossover_long  = (LO > -75)  & (LO.shift(1) <= -75)
crossover_short = (LO <  75)  & (LO.shift(1) >=  75)
momentum_long   = bull_cross  & (~crossover_short)     # bull_cross = WT1 crosses up over Signal_Line
momentum_short  = bear_cross  & (~crossover_long)
threshold_long  = (WT1 < -40) & (WT1.shift(1) >= -40) & (Signal_Line > -40)
threshold_short = (WT1 >  40) & (WT1.shift(1) <=  40) & (Signal_Line <  40)
```

### ✅ Crossover Long — SHIP (the flagship)
- **Edge:** robust and stable at every horizon (best +0.90%/5b, positive 13/15 windows).
- **Learned gate (recurring across windows):** `cs_breadth < median` is the #1 factor
  (chosen 7/15 windows), supported by `Recent_Travel <`, `nf_rsi <`, `nf_rvpct ≥`.
- **Interpretation:** a WaveTrend/LO cross-up pays off best when **broad-market breadth
  is WEAK** (contrarian-strength) and the stock is **not already stretched** (low recent
  travel, lower RSI). It's a "leader emerging from a soft tape" setup.
- **vs current code:** current gate `(conv_d>0 & pulse_d>0)` is positive but leaves edge
  on the table (+0.64 vs learned +0.90 at 5b). The conv/pulse confirmation isn't wrong
  here, just incomplete — breadth is the missing ingredient.

### ✅ Momentum Long — SHIP (with a fixed gate)
- **Edge:** positive at every horizon (+0.39 → +1.10), stable at 2b/8b/13b. The 5b cell
  is its weakest point — do not judge it on 5b alone.
- **Learned gate:** `nf_rsi ≥`, `nf_distma ≥`, `Recent_Travel <` / `Liq_Vel <` (varies a
  bit by horizon — less rock-solid than Crossover Long's recipe).
- **CRITICAL — the current code gate is BACKWARDS here.** Current Momentum Long requires
  `conv_d>0 & pulse_d>0 & liq_osc>0` (strong confirmation) and that gate UNDERPERFORMS
  the naked signal (+0.32 vs naked +0.50 at 5b; −0.21 at 5b in the 2y study). The data
  says momentum crossings pay off when liquidity velocity / conviction are **cooling**,
  not surging. The "confirm with strength" heuristic selects the losing half.
- **Action:** replace the conv/pulse/liq-osc confirmation with the learned conditions.

### ⚠️ Threshold Long — SHIP ONLY AT 8b HORIZON
- **Edge:** flat at 2/3/5b, then +1.29%/8b positive in 7/8 windows (88%) — the single
  strongest cell in the study, but horizon-specific. It's a slow setup needing ~8 bars.
- **Learned gate:** `cs_breadth <`, `cs_conv_rank ≥`, `F1_PriceMom <` (relative-strength
  in a weak tape, similar flavor to Crossover Long).
- **vs current code:** current gate `(conv_d>0 & pulse_d>0 & liq_vel>0)` CUTS the edge
  (+0.59 vs learned +1.29 at 8b) — it filters out winners. Remove it; use the learned gate.
- **Caveat:** thin (≈320 test events). Promising, not proven to Crossover Long's degree.

### ❌ Crossover Short / Momentum Short / Threshold Short — DO NOT TRADE
- Negative at essentially every horizon, worsening as horizon extends (−1.4 to −1.7%/13b).
- Recipe stability is poor (top factors flip direction across windows = curve-fit on noise).
- Likely the structural long-drift of the index over this 5y window, but the data is
  unambiguous: **the short side of these triggers loses money on F&O 2020–2026.**
- Either drop the short signals, or only revisit on a universe/period with genuine
  two-sided movement (and re-validate by walk-forward first).

---

## Recurring theme: cross-sectional context > own-stock confirmation
The single most repeated factor in the winning gates is **`cs_breadth`** (market breadth)
and **`cs_conv_rank` / `cs_mom_rank`** (the stock's rank vs peers). The original gates only
looked at the stock's OWN `conv_d`/`pulse_d`. The data says **a signal's quality depends
heavily on where the stock sits relative to its peers and the broad tape** — information
the hand-coded gates ignore entirely. This is the real "intelligence" lever, applied at
the trigger (entry selection), NOT as a post-hoc confidence filter (which has AUC ~0.5 and
does not work — see real-data-validation memory).

## Hard caveats before any of this is tradeable
1. **Gross of costs.** A +0.3–1.3%/Hbar edge must clear F&O round-trip fees + slippage.
   Only Crossover Long (and maybe Momentum Long, Threshold Long @8b) are worth costing out.
2. **One universe, one 5y window** (≈one-and-a-half regime cycles, long-biased). Re-run
   `research_edge.py`/`walk_forward.py` on Crypto / US / other periods before generalizing.
3. **Median-split thresholds are coarse** — they prove WHICH factors and WHICH direction,
   not exact cut points. Refit precise thresholds on walk-forward before shipping.
4. **Not yet wired into the app.** These are findings, not code changes. `compute_signal_sets`
   still uses the old hand-coded gates. Changing it is a deliberate next step, not done here.

## OUT-OF-SAMPLE CONFIRMATION — NIFTY Smallcap 250 (2026-06)
Re-ran the FULL pipeline on NIFTY Smallcap 250 (5y), which is **92% disjoint from
F&O** (230/250 non-F&O names) — a genuine generalization test, gross of costs.

**Every directional conclusion reproduced:**
| Signal | F&O | Smallcap 250 (OOS) | Generalized |
|---|---|---|---|
| Crossover Long | +0.27→+0.90, 67-87% | +0.12→+1.33, 73-80% | ✅ yes |
| Momentum Long | +0.39→+1.10, all+ | +0.55→+1.85, 53-60% | ✅ yes (weakest) |
| Threshold Long | +1.29 @8b only | +0.69→+0.97, 70-75% @5/8/13b | ✅ yes (broader!) |
| Crossover/Momentum/Threshold Short | dead | dead (neg, ≤53%) | ✅ dead confirmed |

**Recipe also held:** Crossover Long again picks cs_breadth< / nf_rvpct≥ / Recent_Travel<;
Threshold Long picks cs_breadth< in 9-11 windows (its top factor, same as F&O). Two
disjoint universes independently selecting the SAME factors in the SAME direction =
strong evidence of a real structural effect, not curve-fit. Confidence-filter AUC
flat ~0.50 again (triple-confirmed dead: F&O 2y, F&O 5y, Smallcap 5y).

Notable: Threshold Long generalized BETTER on smallcaps (positive 5b/8b/13b, not just
8b) — plausibly because smallcaps are less efficient. Momentum Long is the weakest
of the three (positive everywhere but only 53-60% of windows).

VERDICT: the long-side gate logic is a REAL, GENERALIZABLE effect (gross of costs).
Not yet a tradeable strategy (costs deferred by user request).

## Concrete next steps (in priority order)
1. Cost-and-slippage model on Crossover Long (the one robust signal) → is it tradeable net?
2. Refit Crossover Long's exact gate thresholds on walk-forward (not median split).
3. Decide whether to rewire `compute_signal_sets`: drop/flag the 3 shorts, replace the
   Momentum Long gate (currently backwards), gate Threshold Long for 8b holds.
4. Re-run the whole battery on Crypto — do the shorts come alive in a two-sided market?
