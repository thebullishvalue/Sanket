# Trace study · NIFTY MIDCAP 100 · Daily

75 names · Daily · 2011-11-04 → 2026-09-25 · holdout from 2020-10-22 (40%) · hold 10 bars · book = top 20% · participation ratio 13.1 · macro drivers on · measured 2026-09-27 12:50

**Decision:** no trace setting beats the default on the holdout long−short spread beyond noise — the default stays.

Scores are in σ of each name's own h-bar return (drift removed within era). Brackets are 95% block-bootstrap intervals.

## The screen · holdout

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | -0.034 [-0.082, +0.016] | -0.043 [-0.096, +0.010] | -0.012 [-0.054, +0.030] |
| Long book vs cross-section | -0.014 [-0.041, +0.015] | -0.005 [-0.035, +0.026] | -0.017 [-0.039, +0.007] |
| Short book vs cross-section | -0.020 [-0.046, +0.006] | -0.038 [-0.067, -0.008] | +0.005 [-0.018, +0.028] |
| IC · long priority | -0.004 [-0.022, +0.017] | -0.002 [-0.023, +0.020] | -0.002 [-0.018, +0.017] |
| IC · short priority | -0.003 [-0.021, +0.017] | -0.001 [-0.022, +0.020] | -0.001 [-0.017, +0.017] |
| IC · the trace level | +0.003 [-0.017, +0.022] | +0.002 [-0.020, +0.023] | +0.001 [-0.017, +0.017] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | -0.009 [-0.035, +0.018] | +0.022 [-0.005, +0.046] |
| Long book vs cross-section | +0.009 [-0.005, +0.023] | -0.003 [-0.019, +0.011] |
| Short book vs cross-section | -0.018 [-0.037, +0.002] | +0.025 [+0.007, +0.041] |
| IC · long priority | +0.001 [-0.008, +0.011] | +0.002 [-0.008, +0.012] |
| IC · short priority | +0.002 [-0.008, +0.011] | +0.002 [-0.008, +0.012] |

## The screen · discovery

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | +0.030 [-0.012, +0.070] | +0.024 [-0.021, +0.066] | +0.050 [+0.019, +0.080] |
| Long book vs cross-section | +0.025 [+0.003, +0.045] | +0.026 [+0.003, +0.049] | +0.020 [+0.002, +0.037] |
| Short book vs cross-section | +0.006 [-0.019, +0.028] | -0.003 [-0.031, +0.023] | +0.030 [+0.012, +0.048] |
| IC · long priority | +0.013 [-0.003, +0.029] | +0.010 [-0.008, +0.026] | +0.024 [+0.011, +0.037] |
| IC · short priority | +0.013 [-0.002, +0.029] | +0.010 [-0.007, +0.026] | +0.024 [+0.011, +0.037] |
| IC · the trace level | -0.013 [-0.029, +0.003] | -0.010 [-0.026, +0.007] | -0.024 [-0.037, -0.011] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | -0.007 [-0.027, +0.014] | +0.005 [-0.019, +0.027] |
| Long book vs cross-section | +0.002 [-0.011, +0.016] | -0.012 [-0.025, +0.002] |
| Short book vs cross-section | -0.008 [-0.024, +0.006] | +0.016 [+0.001, +0.032] |
| IC · long priority | -0.003 [-0.011, +0.004] | +0.003 [-0.005, +0.011] |
| IC · short priority | -0.003 [-0.011, +0.004] | +0.003 [-0.005, +0.011] |

## The events (Edge Study method)

| Slice | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long · all (holdout) | UNDERPOWERED · -0.031 [-0.129, +0.070] · n 963 | UNDERPOWERED · -0.154 [-0.266, -0.035] · n 564 | UNDERPOWERED · -0.028 [-0.128, +0.076] · n 1382 |
| Short · all (holdout) | UNDERPOWERED · -0.007 [-0.162, +0.131] · n 363 | UNDERPOWERED · +0.032 [-0.082, +0.145] · n 570 | UNDERPOWERED · +0.047 [-0.083, +0.160] · n 419 |
| ▲ TURN (holdout) | UNDERPOWERED · -0.089 [-0.256, +0.078] · n 210 | UNDERPOWERED · -0.094 [-0.226, +0.049] · n 225 | UNDERPOWERED · -0.115 [-0.281, +0.047] · n 254 |
| ▼ TURN (holdout) | UNDERPOWERED · -0.006 [-0.140, +0.145] · n 261 | UNDERPOWERED · +0.025 [-0.095, +0.139] · n 533 | UNDERPOWERED · +0.073 [-0.095, +0.214] · n 243 |
| ◆ RESUME ↑ (holdout) | UNDERPOWERED · -0.015 [-0.138, +0.105] · n 753 | UNDERPOWERED · -0.194 [-0.340, -0.036] · n 339 | UNDERPOWERED · -0.008 [-0.127, +0.111] · n 1128 |
| ◆ RESUME ↓ (holdout) | UNDERPOWERED · -0.010 [-0.263, +0.140] · n 102 | UNDERPOWERED · +0.123 [-0.391, +0.354] · n 37 | UNDERPOWERED · +0.012 [-0.111, +0.142] · n 176 |

Rule, fixed in advance: a mode replaces the default only if the holdout interval of its paired long−short difference excludes zero.
