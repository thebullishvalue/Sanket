# Signal search, 2026-09-28 — can the ▲ / ▼ be beaten? (nothing shipped)

Protocol fixed before scoring: discovery on 2006-19 only (E1, E2); 2020-26 (E3) sealed until a
shortlist of ≤ 7 was fixed, then opened once; then weekly bars and the Ladder-down window
(Nov 2024 - Sep 2026, 1h + 4h rungs). "Better" = the worst reading across eras × both scorers
(time-series `tc`, cross-sectional `xs`) × 10 / 20 bars beats the shipped ▲'s.

- `sig_r1.py` — 150 events (cell entries, all 72 transitions, phase turns per cell, absorption,
  divergence, held). Longs passing 8/8: capitulation-family only. **Shorts: none 8/8**; the
  shipped ▼ ranges −0.13 … +0.09 on 60-160 events/era.
- `sig_r2.py` — refinements. Best: ▲ OR "capitulation while the chart's own cell has left it"
  (worst +0.055 vs +0.045), ▲ & not quiet (+0.052), double-confirmed capitulation (mean +0.076).
- `sig_r3.py` — shortlist on all eras, daily + weekly. `sig_rL.py` — the Ladder-down window.

| | E3 worst (h10/20, daily) | vs ▲ in 12 era×scorer×h cells | weekly E3 | Ladder-down window |
|---|---|---|---|---|
| shipped ▲ | +0.046 | — | +0.054 | reference |
| **C1 ▲ & not quiet** | **+0.052** | ≥ in 11 / 12 | +0.081 | **≥ ▲ both groups, all 4 readings** |
| A2 (▲ or chart-turned) & not quiet | +0.056 | ≥ in 12 / 12 | +0.025 | **< ▲** (idx and stocks) |
| B2 double-confirmed & not quiet | +0.034 | fails holdout | fails | n/a |
| D1 washout→cap or A1 | +0.051 | mixed | +0.064 | not run |

Crypto: every capitulation variant reads ≤ 0 in the Ladder-down window (the stated exception).
