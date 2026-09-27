# Trace study · NIFTY 200 · Daily

77 names · Daily · 2011-11-04 → 2026-09-25 · holdout from 2020-10-22 (40%) · hold 10 bars · book = top 20% · participation ratio 12.4 · macro drivers on · measured 2026-09-27 12:38

**Decision:** no trace setting beats the default on the holdout long−short spread beyond noise — the default stays.

**Warning:** the screen ranks BACKWARDS on the holdout under Conviction × value, Value only — the long book trailed the short book beyond noise.

Scores are in σ of each name's own h-bar return (drift removed within era). Brackets are 95% block-bootstrap intervals.

## The screen · holdout

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | -0.063 [-0.096, -0.028] | -0.023 [-0.060, +0.013] | -0.037 [-0.073, -0.000] |
| Long book vs cross-section | -0.034 [-0.057, -0.010] | -0.014 [-0.040, +0.011] | -0.011 [-0.035, +0.013] |
| Short book vs cross-section | -0.029 [-0.050, -0.008] | -0.009 [-0.033, +0.016] | -0.026 [-0.050, -0.002] |
| IC · long priority | -0.025 [-0.039, -0.011] | -0.023 [-0.037, -0.009] | -0.023 [-0.038, -0.008] |
| IC · short priority | -0.020 [-0.034, -0.006] | -0.011 [-0.025, +0.001] | -0.016 [-0.031, -0.001] |
| IC · the trace level | +0.001 [-0.017, +0.020] | -0.001 [-0.020, +0.017] | +0.002 [-0.014, +0.018] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | +0.039 [-0.002, +0.081] | +0.025 [-0.004, +0.055] |
| Long book vs cross-section | +0.019 [-0.008, +0.048] | +0.022 [-0.000, +0.045] |
| Short book vs cross-section | +0.020 [-0.007, +0.047] | +0.003 [-0.012, +0.020] |
| IC · long priority | +0.002 [-0.007, +0.012] | +0.002 [-0.007, +0.011] |
| IC · short priority | +0.008 [-0.005, +0.022] | +0.004 [-0.005, +0.013] |

## The screen · discovery

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | -0.027 [-0.065, +0.013] | -0.044 [-0.084, -0.006] | -0.020 [-0.050, +0.014] |
| Long book vs cross-section | -0.012 [-0.038, +0.018] | -0.024 [-0.052, +0.001] | -0.011 [-0.030, +0.010] |
| Short book vs cross-section | -0.018 [-0.046, +0.006] | -0.024 [-0.052, +0.001] | -0.006 [-0.027, +0.016] |
| IC · long priority | -0.014 [-0.028, +0.003] | -0.012 [-0.027, +0.003] | -0.010 [-0.023, +0.003] |
| IC · short priority | -0.018 [-0.032, -0.003] | -0.014 [-0.028, +0.001] | -0.012 [-0.027, +0.002] |
| IC · the trace level | -0.026 [-0.040, -0.009] | -0.015 [-0.031, +0.003] | -0.032 [-0.045, -0.019] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | -0.007 [-0.044, +0.031] | -0.005 [-0.032, +0.020] |
| Long book vs cross-section | -0.005 [-0.034, +0.019] | -0.006 [-0.029, +0.016] |
| Short book vs cross-section | -0.002 [-0.030, +0.026] | +0.005 [-0.014, +0.027] |
| IC · long priority | +0.004 [-0.007, +0.014] | +0.000 [-0.009, +0.009] |
| IC · short priority | +0.004 [-0.008, +0.016] | +0.001 [-0.007, +0.010] |

## The events (Edge Study method)

| Slice | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long · all (holdout) | UNDERPOWERED · -0.011 [-0.105, +0.098] · n 1037 | UNDERPOWERED · -0.140 [-0.271, -0.011] · n 607 | UNDERPOWERED · +0.001 [-0.099, +0.112] · n 1448 |
| Short · all (holdout) | UNDERPOWERED · -0.043 [-0.154, +0.070] · n 394 | UNDERPOWERED · -0.061 [-0.210, +0.068] · n 604 | UNDERPOWERED · -0.048 [-0.199, +0.077] · n 403 |
| ▲ TURN (holdout) | UNDERPOWERED · -0.112 [-0.284, +0.036] · n 217 | UNDERPOWERED · -0.121 [-0.247, +0.031] · n 242 | UNDERPOWERED · -0.097 [-0.276, +0.056] · n 254 |
| ▼ TURN (holdout) | UNDERPOWERED · +0.037 [-0.120, +0.185] · n 283 | UNDERPOWERED · -0.067 [-0.219, +0.065] · n 557 | UNDERPOWERED · +0.062 [-0.113, +0.234] · n 238 |
| ◆ RESUME ↑ (holdout) | UNDERPOWERED · +0.016 [-0.091, +0.129] · n 820 | UNDERPOWERED · -0.152 [-0.258, -0.045] · n 365 | UNDERPOWERED · +0.023 [-0.072, +0.127] · n 1194 |
| ◆ RESUME ↓ (holdout) | UNDERPOWERED · -0.248 [-0.468, -0.050] · n 111 | UNDERPOWERED · -0.000 [-0.144, +0.210] · n 47 | UNDERPOWERED · -0.208 [-0.470, +0.021] · n 165 |

Rule, fixed in advance: a mode replaces the default only if the holdout interval of its paired long−short difference excludes zero.
