# Trace study · NIFTY 100 · Weekly

76 names · Weekly · 2012-03-05 → 2026-09-21 · holdout from 2020-11-30 (40%) · hold 10 bars · book = top 20% · participation ratio 11.4 · macro drivers on · measured 2026-09-27 12:51

**Decision:** no trace setting beats the default on the holdout long−short spread beyond noise — the default stays.

Scores are in σ of each name's own h-bar return (drift removed within era). Brackets are 95% block-bootstrap intervals.

## The screen · holdout

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | +0.128 [-0.005, +0.254] | +0.143 [+0.001, +0.275] | +0.104 [+0.013, +0.192] |
| Long book vs cross-section | +0.080 [+0.015, +0.145] | +0.097 [+0.019, +0.168] | +0.062 [+0.012, +0.113] |
| Short book vs cross-section | +0.048 [-0.027, +0.123] | +0.046 [-0.031, +0.126] | +0.042 [-0.006, +0.088] |
| IC · long priority | +0.070 [+0.020, +0.121] | +0.072 [+0.018, +0.124] | +0.049 [+0.009, +0.086] |
| IC · short priority | +0.070 [+0.019, +0.121] | +0.071 [+0.017, +0.123] | +0.049 [+0.010, +0.086] |
| IC · the trace level | -0.070 [-0.121, -0.020] | -0.072 [-0.124, -0.018] | -0.048 [-0.086, -0.009] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | +0.015 [-0.032, +0.067] | -0.024 [-0.095, +0.049] |
| Long book vs cross-section | +0.016 [-0.010, +0.042] | -0.018 [-0.051, +0.015] |
| Short book vs cross-section | -0.002 [-0.033, +0.031] | -0.006 [-0.053, +0.041] |
| IC · long priority | +0.002 [-0.016, +0.020] | -0.021 [-0.049, +0.006] |
| IC · short priority | +0.001 [-0.017, +0.019] | -0.021 [-0.049, +0.006] |

## The screen · discovery

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | +0.089 [-0.006, +0.167] | +0.096 [-0.013, +0.178] | +0.041 [-0.024, +0.115] |
| Long book vs cross-section | +0.014 [-0.036, +0.062] | +0.026 [-0.038, +0.085] | +0.012 [-0.022, +0.052] |
| Short book vs cross-section | +0.075 [+0.018, +0.118] | +0.071 [+0.002, +0.120] | +0.029 [-0.014, +0.072] |
| IC · long priority | +0.045 [+0.002, +0.081] | +0.047 [+0.001, +0.083] | +0.029 [+0.001, +0.059] |
| IC · short priority | +0.045 [+0.002, +0.081] | +0.048 [+0.002, +0.084] | +0.029 [+0.001, +0.060] |
| IC · the trace level | -0.045 [-0.081, -0.003] | -0.047 [-0.083, -0.001] | -0.029 [-0.059, -0.001] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | +0.007 [-0.037, +0.040] | -0.049 [-0.086, +0.001] |
| Long book vs cross-section | +0.011 [-0.020, +0.038] | -0.012 [-0.033, +0.011] |
| Short book vs cross-section | -0.004 [-0.035, +0.020] | -0.037 [-0.065, +0.001] |
| IC · long priority | +0.002 [-0.011, +0.014] | -0.015 [-0.031, +0.006] |
| IC · short priority | +0.003 [-0.010, +0.014] | -0.014 [-0.030, +0.007] |

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
