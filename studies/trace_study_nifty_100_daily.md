# Trace study · NIFTY 100 · Daily

78 names · Daily · 2011-11-04 → 2026-09-25 · holdout from 2020-10-22 (40%) · hold 10 bars · book = top 20% · participation ratio 12.2 · macro drivers on · measured 2026-09-27 12:50

**Decision:** no trace setting beats the default on the holdout long−short spread beyond noise — the default stays.

Scores are in σ of each name's own h-bar return (drift removed within era). Brackets are 95% block-bootstrap intervals.

## The screen · holdout

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | +0.017 [-0.031, +0.068] | +0.038 [-0.016, +0.091] | +0.000 [-0.044, +0.046] |
| Long book vs cross-section | +0.010 [-0.019, +0.040] | +0.025 [-0.006, +0.054] | +0.000 [-0.024, +0.025] |
| Short book vs cross-section | +0.007 [-0.018, +0.034] | +0.013 [-0.016, +0.042] | +0.000 [-0.026, +0.026] |
| IC · long priority | +0.021 [+0.000, +0.041] | +0.026 [+0.005, +0.048] | +0.011 [-0.007, +0.029] |
| IC · short priority | +0.021 [+0.001, +0.041] | +0.026 [+0.005, +0.048] | +0.012 [-0.006, +0.030] |
| IC · the trace level | -0.021 [-0.041, -0.000] | -0.026 [-0.048, -0.005] | -0.012 [-0.029, +0.006] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | +0.021 [-0.006, +0.050] | -0.017 [-0.044, +0.011] |
| Long book vs cross-section | +0.014 [-0.001, +0.032] | -0.010 [-0.026, +0.004] |
| Short book vs cross-section | +0.006 [-0.013, +0.026] | -0.006 [-0.025, +0.012] |
| IC · long priority | +0.005 [-0.004, +0.015] | -0.009 [-0.020, +0.001] |
| IC · short priority | +0.005 [-0.004, +0.016] | -0.009 [-0.020, +0.001] |

## The screen · discovery

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | +0.064 [+0.023, +0.103] | +0.056 [+0.009, +0.098] | +0.061 [+0.030, +0.094] |
| Long book vs cross-section | +0.030 [+0.008, +0.052] | +0.024 [+0.001, +0.045] | +0.030 [+0.013, +0.048] |
| Short book vs cross-section | +0.035 [+0.010, +0.056] | +0.032 [+0.003, +0.060] | +0.031 [+0.014, +0.050] |
| IC · long priority | +0.032 [+0.014, +0.047] | +0.026 [+0.008, +0.042] | +0.032 [+0.020, +0.045] |
| IC · short priority | +0.031 [+0.014, +0.047] | +0.026 [+0.008, +0.042] | +0.031 [+0.019, +0.044] |
| IC · the trace level | -0.031 [-0.047, -0.014] | -0.026 [-0.042, -0.008] | -0.031 [-0.044, -0.019] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | -0.009 [-0.033, +0.014] | -0.010 [-0.029, +0.010] |
| Long book vs cross-section | -0.006 [-0.020, +0.007] | -0.003 [-0.014, +0.009] |
| Short book vs cross-section | -0.003 [-0.018, +0.013] | -0.007 [-0.021, +0.006] |
| IC · long priority | -0.005 [-0.013, +0.002] | -0.001 [-0.009, +0.006] |
| IC · short priority | -0.005 [-0.013, +0.002] | -0.001 [-0.009, +0.006] |

## The events (Edge Study method)

| Slice | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long · all (holdout) | UNDERPOWERED · -0.002 [-0.118, +0.117] · n 1123 | UNDERPOWERED · -0.058 [-0.176, +0.062] · n 671 | UNDERPOWERED · +0.017 [-0.088, +0.118] · n 1551 |
| Short · all (holdout) | UNDERPOWERED · -0.058 [-0.210, +0.086] · n 443 | UNDERPOWERED · -0.014 [-0.156, +0.116] · n 627 | UNDERPOWERED · -0.066 [-0.216, +0.075] · n 430 |
| ▲ TURN (holdout) | UNDERPOWERED · -0.055 [-0.212, +0.099] · n 254 | UNDERPOWERED · -0.054 [-0.243, +0.156] · n 276 | UNDERPOWERED · -0.031 [-0.189, +0.121] · n 296 |
| ▼ TURN (holdout) | UNDERPOWERED · -0.024 [-0.183, +0.117] · n 319 | UNDERPOWERED · -0.029 [-0.189, +0.104] · n 591 | UNDERPOWERED · +0.008 [-0.160, +0.163] · n 258 |
| ◆ RESUME ↑ (holdout) | UNDERPOWERED · +0.013 [-0.098, +0.139] · n 869 | UNDERPOWERED · -0.060 [-0.190, +0.073] · n 395 | UNDERPOWERED · +0.028 [-0.082, +0.151] · n 1255 |
| ◆ RESUME ↓ (holdout) | UNDERPOWERED · -0.147 [-0.290, -0.021] · n 124 | UNDERPOWERED · +0.235 [-0.140, +0.594] · n 36 | UNDERPOWERED · -0.176 [-0.369, +0.003] · n 172 |

Rule, fixed in advance: a mode replaces the default only if the holdout interval of its paired long−short difference excludes zero.
