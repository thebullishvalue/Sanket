# Trace study · NIFTY 200 · Daily

77 names · Daily · 2011-11-04 → 2026-09-25 · holdout from 2020-10-22 (40%) · hold 10 bars · book = top 20% · participation ratio 12.4 · macro drivers on · measured 2026-09-27 12:50

**Decision:** no trace setting beats the default on the holdout long−short spread beyond noise — the default stays.

Scores are in σ of each name's own h-bar return (drift removed within era). Brackets are 95% block-bootstrap intervals.

## The screen · holdout

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | -0.027 [-0.073, +0.018] | -0.022 [-0.070, +0.028] | -0.013 [-0.054, +0.027] |
| Long book vs cross-section | -0.013 [-0.040, +0.013] | -0.010 [-0.039, +0.019] | -0.019 [-0.040, +0.002] |
| Short book vs cross-section | -0.014 [-0.039, +0.012] | -0.012 [-0.040, +0.017] | +0.006 [-0.017, +0.029] |
| IC · long priority | -0.002 [-0.020, +0.016] | +0.001 [-0.018, +0.020] | -0.002 [-0.018, +0.014] |
| IC · short priority | -0.001 [-0.019, +0.017] | +0.002 [-0.017, +0.021] | -0.001 [-0.017, +0.015] |
| IC · the trace level | +0.002 [-0.017, +0.020] | -0.001 [-0.020, +0.017] | +0.002 [-0.014, +0.018] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | +0.005 [-0.020, +0.030] | +0.014 [-0.010, +0.039] |
| Long book vs cross-section | +0.004 [-0.010, +0.017] | -0.005 [-0.019, +0.009] |
| Short book vs cross-section | +0.002 [-0.018, +0.021] | +0.020 [+0.003, +0.036] |
| IC · long priority | +0.003 [-0.006, +0.011] | -0.000 [-0.010, +0.009] |
| IC · short priority | +0.003 [-0.006, +0.011] | -0.000 [-0.010, +0.008] |

## The screen · discovery

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | +0.071 [+0.032, +0.107] | +0.047 [-0.001, +0.088] | +0.068 [+0.036, +0.100] |
| Long book vs cross-section | +0.026 [+0.002, +0.048] | +0.015 [-0.013, +0.039] | +0.029 [+0.013, +0.047] |
| Short book vs cross-section | +0.045 [+0.022, +0.064] | +0.032 [+0.005, +0.055] | +0.039 [+0.020, +0.057] |
| IC · long priority | +0.025 [+0.009, +0.039] | +0.015 [-0.003, +0.031] | +0.032 [+0.019, +0.045] |
| IC · short priority | +0.025 [+0.009, +0.039] | +0.015 [-0.003, +0.030] | +0.032 [+0.018, +0.045] |
| IC · the trace level | -0.026 [-0.040, -0.009] | -0.015 [-0.031, +0.003] | -0.032 [-0.045, -0.019] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | -0.025 [-0.047, -0.001] | -0.010 [-0.032, +0.014] |
| Long book vs cross-section | -0.012 [-0.025, +0.002] | -0.001 [-0.014, +0.013] |
| Short book vs cross-section | -0.013 [-0.028, +0.002] | -0.009 [-0.023, +0.006] |
| IC · long priority | -0.010 [-0.018, -0.002] | +0.004 [-0.004, +0.013] |
| IC · short priority | -0.010 [-0.018, -0.002] | +0.004 [-0.004, +0.013] |

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
