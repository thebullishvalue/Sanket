# Trace study · NIFTY 50 · Daily

50 names · Daily · 2011-11-04 → 2026-09-25 · holdout from 2020-10-22 (40%) · hold 10 bars · book = top 20% · participation ratio 11.0 · macro drivers on · measured 2026-09-27 12:37

**Decision:** Conviction only beats Conviction × value on the holdout long−short spread, paired, CI excluding zero, BUT its own screen shows no edge on the holdout — it loses less than the default, it does not pick winners; discovery does not confirm the difference.

**Warning:** the screen ranks BACKWARDS on the holdout under Conviction × value, Value only — the long book trailed the short book beyond noise.

Scores are in σ of each name's own h-bar return (drift removed within era). Brackets are 95% block-bootstrap intervals.

## The screen · holdout

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | -0.057 [-0.099, -0.014] | -0.005 [-0.046, +0.039] | -0.065 [-0.106, -0.023] |
| Long book vs cross-section | -0.037 [-0.065, -0.008] | -0.018 [-0.044, +0.009] | -0.036 [-0.065, -0.007] |
| Short book vs cross-section | -0.020 [-0.044, +0.005] | +0.013 [-0.014, +0.042] | -0.029 [-0.054, -0.003] |
| IC · long priority | -0.049 [-0.067, -0.032] | -0.042 [-0.060, -0.024] | -0.051 [-0.069, -0.034] |
| IC · short priority | -0.035 [-0.051, -0.018] | -0.016 [-0.033, +0.000] | -0.041 [-0.058, -0.025] |
| IC · the trace level | -0.029 [-0.052, -0.006] | -0.039 [-0.063, -0.015] | -0.014 [-0.035, +0.007] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | +0.051 [+0.005, +0.097] | -0.008 [-0.044, +0.026] |
| Long book vs cross-section | +0.018 [-0.014, +0.048] | +0.002 [-0.024, +0.027] |
| Short book vs cross-section | +0.032 [+0.007, +0.058] | -0.010 [-0.030, +0.010] |
| IC · long priority | +0.007 [-0.006, +0.019] | -0.002 [-0.014, +0.009] |
| IC · short priority | +0.019 [+0.004, +0.033] | -0.007 [-0.018, +0.004] |

## The screen · discovery

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | -0.021 [-0.070, +0.023] | -0.026 [-0.072, +0.018] | -0.028 [-0.068, +0.015] |
| Long book vs cross-section | -0.007 [-0.035, +0.019] | -0.006 [-0.038, +0.023] | -0.001 [-0.023, +0.021] |
| Short book vs cross-section | -0.014 [-0.043, +0.016] | -0.020 [-0.045, +0.005] | -0.027 [-0.052, -0.002] |
| IC · long priority | -0.004 [-0.021, +0.013] | -0.008 [-0.026, +0.009] | +0.000 [-0.014, +0.015] |
| IC · short priority | -0.008 [-0.025, +0.009] | -0.012 [-0.027, +0.004] | -0.011 [-0.027, +0.005] |
| IC · the trace level | -0.023 [-0.040, -0.004] | -0.018 [-0.036, +0.002] | -0.028 [-0.044, -0.013] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | +0.006 [-0.039, +0.050] | -0.012 [-0.047, +0.029] |
| Long book vs cross-section | +0.002 [-0.028, +0.031] | +0.001 [-0.020, +0.024] |
| Short book vs cross-section | -0.005 [-0.038, +0.028] | -0.013 [-0.035, +0.008] |
| IC · long priority | -0.002 [-0.015, +0.010] | +0.004 [-0.006, +0.015] |
| IC · short priority | -0.004 [-0.017, +0.008] | -0.004 [-0.015, +0.006] |

## The events (Edge Study method)

| Slice | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long · all (holdout) | UNDERPOWERED · +0.016 [-0.094, +0.136] · n 708 | UNDERPOWERED · -0.080 [-0.200, +0.050] · n 439 | UNDERPOWERED · +0.016 [-0.090, +0.121] · n 962 |
| Short · all (holdout) | UNDERPOWERED · -0.115 [-0.309, +0.063] · n 295 | UNDERPOWERED · +0.065 [-0.073, +0.203] · n 413 | UNDERPOWERED · -0.134 [-0.299, +0.052] · n 293 |
| ▲ TURN (holdout) | UNDERPOWERED · +0.016 [-0.144, +0.156] · n 163 | UNDERPOWERED · -0.046 [-0.249, +0.194] · n 191 | UNDERPOWERED · +0.040 [-0.094, +0.208] · n 187 |
| ▼ TURN (holdout) | UNDERPOWERED · -0.057 [-0.287, +0.134] · n 206 | UNDERPOWERED · +0.067 [-0.072, +0.210] · n 387 | UNDERPOWERED · -0.042 [-0.311, +0.232] · n 168 |
| ◆ RESUME ↑ (holdout) | UNDERPOWERED · +0.016 [-0.094, +0.142] · n 545 | UNDERPOWERED · -0.106 [-0.252, +0.078] · n 248 | UNDERPOWERED · +0.010 [-0.114, +0.150] · n 775 |
| ◆ RESUME ↓ (holdout) | UNDERPOWERED · -0.248 [-0.454, -0.114] · n 89 | UNDERPOWERED · +0.032 [+0.061, +0.064] · n 26 | UNDERPOWERED · -0.257 [-0.449, -0.111] · n 125 |

Rule, fixed in advance: a mode replaces the default only if the holdout interval of its paired long−short difference excludes zero.
