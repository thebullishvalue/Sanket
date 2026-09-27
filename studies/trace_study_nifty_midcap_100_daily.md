# Trace study · NIFTY MIDCAP 100 · Daily

75 names · Daily · 2011-11-04 → 2026-09-25 · holdout from 2020-10-22 (40%) · hold 10 bars · book = top 20% · participation ratio 13.1 · macro drivers on · measured 2026-09-27 12:41

**Decision:** no trace setting beats the default on the holdout long−short spread beyond noise — the default stays.

Scores are in σ of each name's own h-bar return (drift removed within era). Brackets are 95% block-bootstrap intervals.

## The screen · holdout

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | -0.027 [-0.067, +0.014] | -0.001 [-0.039, +0.037] | -0.027 [-0.067, +0.014] |
| Long book vs cross-section | -0.017 [-0.043, +0.009] | -0.013 [-0.039, +0.013] | -0.010 [-0.034, +0.014] |
| Short book vs cross-section | -0.011 [-0.033, +0.014] | +0.012 [-0.012, +0.036] | -0.017 [-0.042, +0.010] |
| IC · long priority | -0.017 [-0.034, -0.001] | -0.017 [-0.032, -0.003] | -0.018 [-0.035, -0.002] |
| IC · short priority | -0.012 [-0.028, +0.004] | -0.000 [-0.015, +0.013] | -0.012 [-0.031, +0.004] |
| IC · the trace level | +0.003 [-0.017, +0.022] | +0.002 [-0.020, +0.023] | +0.001 [-0.017, +0.017] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | +0.026 [-0.017, +0.070] | +0.000 [-0.026, +0.025] |
| Long book vs cross-section | +0.004 [-0.025, +0.032] | +0.007 [-0.009, +0.023] |
| Short book vs cross-section | +0.022 [-0.002, +0.048] | -0.007 [-0.023, +0.010] |
| IC · long priority | -0.000 [-0.011, +0.010] | -0.001 [-0.010, +0.008] |
| IC · short priority | +0.012 [-0.000, +0.024] | -0.000 [-0.009, +0.008] |

## The screen · discovery

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | -0.002 [-0.037, +0.035] | -0.003 [-0.041, +0.036] | -0.027 [-0.057, +0.002] |
| Long book vs cross-section | -0.010 [-0.034, +0.015] | -0.002 [-0.030, +0.026] | -0.015 [-0.035, +0.006] |
| Short book vs cross-section | -0.003 [-0.034, +0.025] | +0.001 [-0.021, +0.024] | -0.006 [-0.025, +0.014] |
| IC · long priority | -0.006 [-0.020, +0.009] | +0.004 [-0.012, +0.020] | -0.007 [-0.020, +0.006] |
| IC · short priority | -0.002 [-0.016, +0.012] | +0.003 [-0.010, +0.015] | -0.005 [-0.019, +0.009] |
| IC · the trace level | -0.013 [-0.029, +0.003] | -0.010 [-0.026, +0.007] | -0.024 [-0.037, -0.011] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | +0.009 [-0.028, +0.047] | -0.022 [-0.051, +0.002] |
| Long book vs cross-section | +0.012 [-0.015, +0.037] | -0.010 [-0.031, +0.010] |
| Short book vs cross-section | +0.000 [-0.026, +0.025] | -0.004 [-0.026, +0.018] |
| IC · long priority | +0.011 [+0.000, +0.020] | -0.002 [-0.011, +0.007] |
| IC · short priority | +0.004 [-0.008, +0.015] | -0.005 [-0.015, +0.003] |

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
