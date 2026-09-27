# Trace study · NIFTY 50 · Weekly

49 names · Weekly · 2012-03-05 → 2026-09-21 · holdout from 2020-11-30 (40%) · hold 10 bars · book = top 20% · participation ratio 10.2 · macro drivers on · measured 2026-09-27 12:38

**Decision:** no trace setting beats the default on the holdout long−short spread beyond noise — the default stays.

**Warning:** the screen ranks BACKWARDS on the holdout under Conviction × value, Conviction only — the long book trailed the short book beyond noise.

Scores are in σ of each name's own h-bar return (drift removed within era). Brackets are 95% block-bootstrap intervals.

## The screen · holdout

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | -0.090 [-0.167, -0.025] | -0.099 [-0.177, -0.025] | -0.054 [-0.146, +0.030] |
| Long book vs cross-section | -0.062 [-0.115, -0.015] | -0.053 [-0.102, -0.004] | -0.042 [-0.102, +0.014] |
| Short book vs cross-section | -0.028 [-0.076, +0.020] | -0.046 [-0.106, +0.010] | -0.013 [-0.069, +0.040] |
| IC · long priority | -0.061 [-0.093, -0.031] | -0.053 [-0.086, -0.018] | -0.063 [-0.102, -0.026] |
| IC · short priority | -0.068 [-0.097, -0.037] | -0.058 [-0.092, -0.027] | -0.065 [-0.099, -0.032] |
| IC · the trace level | -0.096 [-0.148, -0.046] | -0.080 [-0.138, -0.021] | -0.090 [-0.126, -0.052] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | -0.009 [-0.086, +0.075] | +0.036 [-0.042, +0.113] |
| Long book vs cross-section | +0.009 [-0.034, +0.059] | +0.020 [-0.024, +0.065] |
| Short book vs cross-section | -0.018 [-0.083, +0.053] | +0.016 [-0.042, +0.072] |
| IC · long priority | +0.008 [-0.019, +0.038] | -0.003 [-0.038, +0.032] |
| IC · short priority | +0.010 [-0.024, +0.044] | +0.003 [-0.030, +0.034] |

## The screen · discovery

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | -0.016 [-0.109, +0.064] | +0.020 [-0.111, +0.122] | +0.010 [-0.051, +0.067] |
| Long book vs cross-section | -0.044 [-0.109, +0.024] | +0.019 [-0.069, +0.093] | -0.005 [-0.064, +0.055] |
| Short book vs cross-section | +0.034 [-0.021, +0.078] | -0.002 [-0.063, +0.042] | +0.035 [-0.024, +0.092] |
| IC · long priority | -0.027 [-0.061, +0.005] | -0.029 [-0.070, +0.003] | -0.014 [-0.047, +0.017] |
| IC · short priority | -0.001 [-0.031, +0.028] | -0.022 [-0.058, +0.007] | -0.006 [-0.038, +0.023] |
| IC · the trace level | -0.046 [-0.081, +0.001] | -0.052 [-0.094, +0.001] | -0.036 [-0.072, -0.005] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | +0.021 [-0.072, +0.111] | +0.021 [-0.058, +0.109] |
| Long book vs cross-section | +0.060 [-0.010, +0.129] | +0.002 [-0.056, +0.054] |
| Short book vs cross-section | -0.033 [-0.088, +0.017] | +0.009 [-0.046, +0.075] |
| IC · long priority | -0.004 [-0.038, +0.029] | +0.007 [-0.019, +0.036] |
| IC · short priority | -0.019 [-0.052, +0.006] | -0.006 [-0.033, +0.025] |

## The events (Edge Study method)

| Slice | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long · all (holdout) | UNDERPOWERED · +0.013 [-0.180, +0.266] · n 78 | UNDERPOWERED · +0.082 [-0.102, +0.548] · n 74 | UNDERPOWERED · -0.084 [-0.362, +0.208] · n 173 |
| Short · all (holdout) | UNDERPOWERED · +0.267 [+0.035, +0.399] · n 51 | UNDERPOWERED · -0.125 [-0.631, +0.365] · n 80 | UNDERPOWERED · +0.152 [+0.155, +0.478] · n 59 |
| ▲ TURN (holdout) | UNDERPOWERED · +0.150 [+0.244, +0.394] · n 37 | UNDERPOWERED · +0.242 [-0.383, +0.907] · n 33 | UNDERPOWERED · +0.141 [-0.071, +0.540] · n 50 |
| ▼ TURN (holdout) | UNDERPOWERED · +0.267 [+0.035, +0.399] · n 51 | UNDERPOWERED · -0.139 [-0.631, +0.365] · n 78 | UNDERPOWERED · +0.238 [+0.155, +0.478] · n 53 |
| ◆ RESUME ↑ (holdout) | UNDERPOWERED · -0.110 [-0.261, +0.051] · n 41 | UNDERPOWERED · -0.047 [-0.154, +0.277] · n 41 | UNDERPOWERED · -0.175 [-0.510, +0.185] · n 123 |
| ◆ RESUME ↓ (holdout) | UNDERPOWERED · no events | UNDERPOWERED · +0.388 [+nan, +nan] · n 2 | UNDERPOWERED · -0.610 [+nan, +nan] · n 6 |

Rule, fixed in advance: a mode replaces the default only if the holdout interval of its paired long−short difference excludes zero.
