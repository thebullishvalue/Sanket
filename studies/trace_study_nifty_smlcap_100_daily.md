# Trace study · NIFTY SMLCAP 100 · Daily

70 names · Daily · 2011-11-04 → 2026-09-25 · holdout from 2020-10-22 (40%) · hold 10 bars · book = top 20% · participation ratio 15.0 · macro drivers on · measured 2026-09-27 12:51

**Decision:** no trace setting beats the default on the holdout long−short spread beyond noise — the default stays.

Scores are in σ of each name's own h-bar return (drift removed within era). Brackets are 95% block-bootstrap intervals.

## The screen · holdout

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | -0.017 [-0.056, +0.026] | -0.019 [-0.065, +0.027] | -0.009 [-0.045, +0.028] |
| Long book vs cross-section | -0.003 [-0.027, +0.022] | -0.012 [-0.038, +0.014] | -0.004 [-0.025, +0.015] |
| Short book vs cross-section | -0.014 [-0.036, +0.011] | -0.007 [-0.034, +0.021] | -0.005 [-0.025, +0.017] |
| IC · long priority | +0.010 [-0.005, +0.027] | +0.005 [-0.013, +0.022] | +0.011 [-0.003, +0.025] |
| IC · short priority | +0.011 [-0.005, +0.027] | +0.004 [-0.013, +0.021] | +0.011 [-0.003, +0.025] |
| IC · the trace level | -0.011 [-0.028, +0.004] | -0.005 [-0.022, +0.013] | -0.012 [-0.026, +0.003] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | -0.002 [-0.028, +0.022] | +0.008 [-0.018, +0.035] |
| Long book vs cross-section | -0.009 [-0.023, +0.006] | -0.001 [-0.017, +0.016] |
| Short book vs cross-section | +0.006 [-0.014, +0.027] | +0.009 [-0.008, +0.025] |
| IC · long priority | -0.006 [-0.014, +0.002] | +0.000 [-0.009, +0.010] |
| IC · short priority | -0.007 [-0.015, +0.001] | +0.001 [-0.009, +0.010] |

## The screen · discovery

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | +0.061 [+0.020, +0.102] | +0.051 [+0.005, +0.097] | +0.047 [+0.016, +0.082] |
| Long book vs cross-section | +0.033 [+0.010, +0.054] | +0.026 [+0.001, +0.051] | +0.011 [-0.008, +0.031] |
| Short book vs cross-section | +0.029 [+0.002, +0.053] | +0.025 [-0.002, +0.053] | +0.036 [+0.017, +0.057] |
| IC · long priority | +0.037 [+0.018, +0.054] | +0.029 [+0.010, +0.048] | +0.033 [+0.019, +0.048] |
| IC · short priority | +0.037 [+0.018, +0.054] | +0.030 [+0.011, +0.048] | +0.033 [+0.019, +0.048] |
| IC · the trace level | -0.037 [-0.054, -0.018] | -0.029 [-0.048, -0.010] | -0.033 [-0.048, -0.019] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | -0.010 [-0.036, +0.014] | -0.023 [-0.046, +0.000] |
| Long book vs cross-section | -0.007 [-0.024, +0.010] | -0.025 [-0.040, -0.010] |
| Short book vs cross-section | -0.003 [-0.022, +0.014] | +0.001 [-0.014, +0.017] |
| IC · long priority | -0.008 [-0.017, +0.002] | -0.005 [-0.015, +0.005] |
| IC · short priority | -0.007 [-0.017, +0.002] | -0.005 [-0.015, +0.005] |

## The events (Edge Study method)

| Slice | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long · all (holdout) | UNDERPOWERED · -0.067 [-0.191, +0.062] · n 852 | UNDERPOWERED · -0.081 [-0.216, +0.076] · n 454 | UNDERPOWERED · -0.050 [-0.145, +0.052] · n 1226 |
| Short · all (holdout) | UNDERPOWERED · -0.016 [-0.180, +0.117] · n 305 | UNDERPOWERED · -0.018 [-0.145, +0.096] · n 485 | UNDERPOWERED · -0.029 [-0.253, +0.166] · n 283 |
| ▲ TURN (holdout) | UNDERPOWERED · -0.143 [-0.342, +0.087] · n 150 | UNDERPOWERED · +0.029 [-0.137, +0.199] · n 154 | UNDERPOWERED · -0.185 [-0.339, -0.005] · n 198 |
| ▼ TURN (holdout) | UNDERPOWERED · -0.014 [-0.171, +0.133] · n 237 | UNDERPOWERED · -0.023 [-0.159, +0.102] · n 467 | UNDERPOWERED · -0.047 [-0.255, +0.145] · n 183 |
| ◆ RESUME ↑ (holdout) | UNDERPOWERED · -0.050 [-0.153, +0.074] · n 702 | UNDERPOWERED · -0.137 [-0.306, +0.031] · n 300 | UNDERPOWERED · -0.024 [-0.117, +0.085] · n 1028 |
| ◆ RESUME ↓ (holdout) | UNDERPOWERED · -0.021 [-0.130, +0.297] · n 68 | UNDERPOWERED · +0.105 [+nan, +nan] · n 18 | UNDERPOWERED · +0.003 [-0.301, +0.302] · n 100 |

Rule, fixed in advance: a mode replaces the default only if the holdout interval of its paired long−short difference excludes zero.
