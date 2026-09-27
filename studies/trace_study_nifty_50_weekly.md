# Trace study · NIFTY 50 · Weekly

49 names · Weekly · 2012-03-05 → 2026-09-21 · holdout from 2020-11-30 (40%) · hold 10 bars · book = top 20% · participation ratio 10.2 · macro drivers on · measured 2026-09-27 12:51

**Decision:** no trace setting beats the default on the holdout long−short spread beyond noise — the default stays.

Scores are in σ of each name's own h-bar return (drift removed within era). Brackets are 95% block-bootstrap intervals.

## The screen · holdout

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | +0.168 [+0.026, +0.305] | +0.159 [-0.008, +0.313] | +0.192 [+0.102, +0.282] |
| Long book vs cross-section | +0.078 [+0.007, +0.153] | +0.109 [+0.018, +0.193] | +0.087 [+0.036, +0.140] |
| Short book vs cross-section | +0.089 [+0.012, +0.160] | +0.051 [-0.035, +0.136] | +0.105 [+0.055, +0.154] |
| IC · long priority | +0.095 [+0.045, +0.147] | +0.080 [+0.020, +0.138] | +0.090 [+0.052, +0.126] |
| IC · short priority | +0.096 [+0.047, +0.148] | +0.078 [+0.019, +0.138] | +0.091 [+0.054, +0.126] |
| IC · the trace level | -0.096 [-0.148, -0.046] | -0.080 [-0.138, -0.021] | -0.090 [-0.126, -0.052] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | -0.008 [-0.062, +0.050] | +0.024 [-0.065, +0.120] |
| Long book vs cross-section | +0.030 [-0.012, +0.077] | +0.009 [-0.033, +0.051] |
| Short book vs cross-section | -0.039 [-0.067, -0.007] | +0.015 [-0.041, +0.080] |
| IC · long priority | -0.016 [-0.032, +0.004] | -0.005 [-0.037, +0.026] |
| IC · short priority | -0.018 [-0.035, +0.001] | -0.005 [-0.038, +0.026] |

## The screen · discovery

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | +0.080 [-0.026, +0.160] | +0.098 [-0.027, +0.190] | +0.070 [-0.001, +0.152] |
| Long book vs cross-section | +0.011 [-0.050, +0.063] | +0.037 [-0.037, +0.102] | +0.019 [-0.023, +0.067] |
| Short book vs cross-section | +0.069 [+0.013, +0.114] | +0.061 [-0.014, +0.122] | +0.051 [+0.008, +0.096] |
| IC · long priority | +0.046 [-0.001, +0.081] | +0.052 [-0.001, +0.093] | +0.036 [+0.004, +0.072] |
| IC · short priority | +0.045 [-0.002, +0.080] | +0.052 [-0.001, +0.093] | +0.036 [+0.004, +0.072] |
| IC · the trace level | -0.046 [-0.081, +0.001] | -0.052 [-0.094, +0.001] | -0.036 [-0.072, -0.005] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | +0.018 [-0.025, +0.051] | -0.030 [-0.081, +0.029] |
| Long book vs cross-section | +0.025 [+0.001, +0.050] | -0.012 [-0.047, +0.025] |
| Short book vs cross-section | -0.007 [-0.045, +0.021] | -0.018 [-0.051, +0.019] |
| IC · long priority | +0.006 [-0.010, +0.021] | -0.016 [-0.039, +0.012] |
| IC · short priority | +0.007 [-0.011, +0.022] | -0.015 [-0.039, +0.013] |

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
