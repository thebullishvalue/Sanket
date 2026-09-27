# Trace study · NIFTY MIDCAP 100 · Weekly

70 names · Weekly · 2012-03-05 → 2026-09-21 · holdout from 2020-11-30 (40%) · hold 10 bars · book = top 20% · participation ratio 12.9 · macro drivers on · measured 2026-09-27 12:51

**Decision:** no trace setting beats the default on the holdout long−short spread beyond noise — the default stays.

Scores are in σ of each name's own h-bar return (drift removed within era). Brackets are 95% block-bootstrap intervals.

## The screen · holdout

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | -0.035 [-0.141, +0.069] | -0.040 [-0.167, +0.086] | +0.015 [-0.062, +0.094] |
| Long book vs cross-section | -0.017 [-0.076, +0.039] | -0.006 [-0.072, +0.062] | +0.005 [-0.040, +0.052] |
| Short book vs cross-section | -0.018 [-0.084, +0.048] | -0.034 [-0.109, +0.043] | +0.010 [-0.031, +0.052] |
| IC · long priority | -0.003 [-0.044, +0.040] | -0.011 [-0.058, +0.037] | +0.010 [-0.022, +0.042] |
| IC · short priority | -0.005 [-0.046, +0.038] | -0.011 [-0.058, +0.037] | +0.008 [-0.022, +0.041] |
| IC · the trace level | +0.004 [-0.040, +0.044] | +0.011 [-0.038, +0.058] | -0.009 [-0.041, +0.021] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | -0.005 [-0.047, +0.039] | +0.051 [-0.020, +0.128] |
| Long book vs cross-section | +0.011 [-0.012, +0.038] | +0.023 [-0.014, +0.063] |
| Short book vs cross-section | -0.016 [-0.050, +0.020] | +0.028 [-0.024, +0.084] |
| IC · long priority | -0.008 [-0.024, +0.007] | +0.013 [-0.011, +0.039] |
| IC · short priority | -0.007 [-0.023, +0.009] | +0.013 [-0.011, +0.040] |

## The screen · discovery

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | -0.008 [-0.101, +0.069] | +0.018 [-0.068, +0.086] | -0.022 [-0.092, +0.046] |
| Long book vs cross-section | -0.001 [-0.051, +0.045] | +0.010 [-0.039, +0.054] | +0.006 [-0.025, +0.046] |
| Short book vs cross-section | -0.007 [-0.063, +0.037] | +0.008 [-0.047, +0.055] | -0.028 [-0.075, +0.007] |
| IC · long priority | -0.008 [-0.045, +0.025] | -0.005 [-0.045, +0.029] | -0.007 [-0.034, +0.021] |
| IC · short priority | -0.008 [-0.045, +0.025] | -0.005 [-0.045, +0.029] | -0.007 [-0.034, +0.021] |
| IC · the trace level | +0.007 [-0.026, +0.045] | +0.005 [-0.029, +0.045] | +0.006 [-0.022, +0.034] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | +0.026 [-0.012, +0.071] | -0.008 [-0.052, +0.036] |
| Long book vs cross-section | +0.012 [-0.016, +0.036] | +0.010 [-0.014, +0.037] |
| Short book vs cross-section | +0.015 [-0.014, +0.052] | -0.017 [-0.055, +0.011] |
| IC · long priority | +0.002 [-0.011, +0.018] | +0.002 [-0.019, +0.023] |
| IC · short priority | +0.003 [-0.011, +0.018] | +0.002 [-0.019, +0.023] |

## The events (Edge Study method)

| Slice | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long · all (holdout) | UNDERPOWERED · +0.095 [-0.209, +0.396] · n 103 | UNDERPOWERED · +0.044 [-0.273, +0.281] · n 87 | UNDERPOWERED · +0.037 [-0.277, +0.365] · n 183 |
| Short · all (holdout) | UNDERPOWERED · -0.006 [-0.534, +0.513] · n 86 | UNDERPOWERED · -0.049 [-0.422, +0.257] · n 120 | UNDERPOWERED · +0.019 [-0.320, +0.316] · n 82 |
| ▲ TURN (holdout) | UNDERPOWERED · +0.140 [-0.174, +0.453] · n 38 | UNDERPOWERED · +0.120 [+0.127, +0.218] · n 45 | UNDERPOWERED · +0.116 [-0.388, +0.603] · n 45 |
| ▼ TURN (holdout) | UNDERPOWERED · -0.011 [-0.545, +0.549] · n 81 | UNDERPOWERED · -0.055 [-0.429, +0.295] · n 118 | UNDERPOWERED · +0.037 [-0.305, +0.371] · n 75 |
| ◆ RESUME ↑ (holdout) | UNDERPOWERED · +0.068 [+0.023, +0.237] · n 65 | UNDERPOWERED · -0.038 [-0.369, +0.500] · n 42 | UNDERPOWERED · +0.012 [-0.311, +0.308] · n 138 |
| ◆ RESUME ↓ (holdout) | UNDERPOWERED · +0.077 [+nan, +nan] · n 5 | UNDERPOWERED · +0.276 [+nan, +nan] · n 2 | UNDERPOWERED · -0.166 [+nan, +nan] · n 7 |

Rule, fixed in advance: a mode replaces the default only if the holdout interval of its paired long−short difference excludes zero.
