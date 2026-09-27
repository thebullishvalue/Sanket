# Trace study · NIFTY 100 · Daily

78 names · Daily · 2011-11-04 → 2026-09-25 · holdout from 2020-10-22 (40%) · hold 10 bars · book = top 20% · participation ratio 12.2 · macro drivers on · measured 2026-09-27 12:40

**Decision:** no trace setting beats the default on the holdout long−short spread beyond noise — the default stays.

**Warning:** the screen ranks BACKWARDS on the holdout under Conviction × value, Value only — the long book trailed the short book beyond noise.

Scores are in σ of each name's own h-bar return (drift removed within era). Brackets are 95% block-bootstrap intervals.

## The screen · holdout

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | -0.047 [-0.080, -0.010] | -0.013 [-0.052, +0.026] | -0.055 [-0.093, -0.017] |
| Long book vs cross-section | -0.030 [-0.052, -0.006] | -0.019 [-0.045, +0.007] | -0.032 [-0.055, -0.009] |
| Short book vs cross-section | -0.017 [-0.039, +0.006] | +0.006 [-0.021, +0.032] | -0.022 [-0.046, +0.002] |
| IC · long priority | -0.036 [-0.052, -0.021] | -0.032 [-0.047, -0.017] | -0.043 [-0.059, -0.028] |
| IC · short priority | -0.025 [-0.040, -0.010] | -0.015 [-0.030, -0.002] | -0.028 [-0.044, -0.013] |
| IC · the trace level | -0.021 [-0.041, -0.000] | -0.026 [-0.048, -0.005] | -0.012 [-0.029, +0.006] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | +0.033 [-0.006, +0.070] | -0.008 [-0.036, +0.021] |
| Long book vs cross-section | +0.010 [-0.012, +0.035] | -0.003 [-0.021, +0.016] |
| Short book vs cross-section | +0.023 [-0.003, +0.049] | -0.005 [-0.022, +0.012] |
| IC · long priority | +0.005 [-0.005, +0.014] | -0.006 [-0.016, +0.003] |
| IC · short priority | +0.010 [-0.004, +0.023] | -0.003 [-0.013, +0.007] |

## The screen · discovery

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | -0.021 [-0.057, +0.016] | -0.045 [-0.082, -0.010] | -0.033 [-0.067, -0.002] |
| Long book vs cross-section | -0.013 [-0.036, +0.010] | -0.020 [-0.045, +0.003] | -0.021 [-0.039, -0.002] |
| Short book vs cross-section | -0.007 [-0.029, +0.017] | -0.024 [-0.049, -0.000] | -0.012 [-0.033, +0.009] |
| IC · long priority | -0.015 [-0.030, +0.000] | -0.014 [-0.029, -0.000] | -0.011 [-0.025, +0.002] |
| IC · short priority | -0.014 [-0.028, +0.001] | -0.011 [-0.025, +0.002] | -0.012 [-0.025, +0.003] |
| IC · the trace level | -0.031 [-0.047, -0.014] | -0.026 [-0.042, -0.008] | -0.031 [-0.044, -0.019] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | -0.013 [-0.050, +0.023] | -0.019 [-0.046, +0.007] |
| Long book vs cross-section | -0.003 [-0.028, +0.020] | -0.008 [-0.026, +0.010] |
| Short book vs cross-section | -0.017 [-0.044, +0.010] | -0.011 [-0.027, +0.005] |
| IC · long priority | +0.001 [-0.009, +0.011] | +0.004 [-0.004, +0.013] |
| IC · short priority | +0.002 [-0.009, +0.014] | -0.001 [-0.009, +0.008] |

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
