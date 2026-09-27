# Trace study · NIFTY 100 · Weekly

76 names · Weekly · 2012-03-05 → 2026-09-21 · holdout from 2020-11-30 (40%) · hold 10 bars · book = top 20% · participation ratio 11.4 · macro drivers on · measured 2026-09-27 12:38

**Decision:** no trace setting beats the default on the holdout long−short spread beyond noise — the default stays.

Scores are in σ of each name's own h-bar return (drift removed within era). Brackets are 95% block-bootstrap intervals.

## The screen · holdout

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | -0.031 [-0.104, +0.038] | -0.019 [-0.091, +0.052] | -0.008 [-0.082, +0.058] |
| Long book vs cross-section | -0.010 [-0.064, +0.041] | -0.003 [-0.058, +0.048] | -0.001 [-0.056, +0.050] |
| Short book vs cross-section | -0.021 [-0.065, +0.024] | -0.016 [-0.060, +0.026] | -0.007 [-0.048, +0.030] |
| IC · long priority | -0.031 [-0.061, -0.004] | -0.025 [-0.056, +0.006] | -0.039 [-0.073, -0.008] |
| IC · short priority | -0.052 [-0.078, -0.026] | -0.036 [-0.062, -0.010] | -0.045 [-0.073, -0.018] |
| IC · the trace level | -0.070 [-0.121, -0.020] | -0.072 [-0.124, -0.018] | -0.048 [-0.086, -0.009] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | +0.012 [-0.053, +0.085] | +0.023 [-0.033, +0.077] |
| Long book vs cross-section | +0.007 [-0.043, +0.062] | +0.010 [-0.032, +0.050] |
| Short book vs cross-section | +0.005 [-0.043, +0.056] | +0.014 [-0.026, +0.052] |
| IC · long priority | +0.006 [-0.019, +0.033] | -0.008 [-0.037, +0.019] |
| IC · short priority | +0.016 [-0.012, +0.043] | +0.008 [-0.020, +0.033] |

## The screen · discovery

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | -0.037 [-0.086, +0.010] | +0.022 [-0.059, +0.122] | -0.016 [-0.070, +0.040] |
| Long book vs cross-section | -0.036 [-0.077, +0.003] | +0.024 [-0.037, +0.089] | -0.027 [-0.079, +0.029] |
| Short book vs cross-section | +0.003 [-0.044, +0.044] | -0.003 [-0.037, +0.032] | +0.020 [-0.037, +0.073] |
| IC · long priority | -0.012 [-0.034, +0.008] | -0.006 [-0.030, +0.017] | -0.021 [-0.045, +0.002] |
| IC · short priority | +0.000 [-0.023, +0.021] | -0.011 [-0.038, +0.013] | -0.014 [-0.039, +0.010] |
| IC · the trace level | -0.045 [-0.081, -0.003] | -0.047 [-0.083, -0.001] | -0.029 [-0.059, -0.001] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | +0.050 [-0.024, +0.142] | +0.015 [-0.047, +0.077] |
| Long book vs cross-section | +0.050 [-0.009, +0.107] | -0.010 [-0.057, +0.037] |
| Short book vs cross-section | -0.006 [-0.059, +0.060] | +0.021 [-0.018, +0.070] |
| IC · long priority | +0.005 [-0.018, +0.026] | -0.013 [-0.036, +0.009] |
| IC · short priority | -0.012 [-0.034, +0.014] | -0.015 [-0.038, +0.010] |

## The events (Edge Study method)

| Slice | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long · all (holdout) | UNDERPOWERED · -0.003 [-0.221, +0.177] · n 118 | UNDERPOWERED · -0.006 [-0.209, +0.298] · n 102 | UNDERPOWERED · -0.067 [-0.308, +0.192] · n 255 |
| Short · all (holdout) | UNDERPOWERED · +0.206 [+0.034, +0.365] · n 74 | UNDERPOWERED · -0.052 [-0.431, +0.266] · n 114 | UNDERPOWERED · +0.087 [+0.053, +0.254] · n 80 |
| ▲ TURN (holdout) | UNDERPOWERED · +0.253 [+0.251, +0.574] · n 51 | UNDERPOWERED · +0.054 [-0.222, +0.673] · n 39 | UNDERPOWERED · +0.225 [-0.057, +0.515] · n 70 |
| ▼ TURN (holdout) | UNDERPOWERED · +0.170 [+0.007, +0.340] · n 72 | UNDERPOWERED · -0.063 [-0.441, +0.259] · n 111 | UNDERPOWERED · +0.145 [+0.031, +0.247] · n 72 |
| ◆ RESUME ↑ (holdout) | UNDERPOWERED · -0.197 [-0.360, +0.015] · n 67 | UNDERPOWERED · -0.043 [-0.249, +0.267] · n 63 | UNDERPOWERED · -0.178 [-0.396, +0.079] · n 185 |
| ◆ RESUME ↓ (holdout) | UNDERPOWERED · +1.507 [+nan, +nan] · n 2 | UNDERPOWERED · +0.354 [+nan, +nan] · n 3 | UNDERPOWERED · -0.432 [+nan, +nan] · n 8 |

Rule, fixed in advance: a mode replaces the default only if the holdout interval of its paired long−short difference excludes zero.
