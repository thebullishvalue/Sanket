# Trace study · NIFTY 50 · Daily

50 names · Daily · 2011-11-04 → 2026-09-25 · holdout from 2020-10-22 (40%) · hold 10 bars · book = top 20% · participation ratio 11.0 · macro drivers on · measured 2026-09-27 12:50

**Decision:** Conviction only beats Conviction × value on the holdout long−short spread, paired, CI excluding zero, and its own screen works (holdout long−short CI above zero); discovery does not confirm the difference.

Scores are in σ of each name's own h-bar return (drift removed within era). Brackets are 95% block-bootstrap intervals.

## The screen · holdout

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | +0.028 [-0.029, +0.084] | +0.070 [+0.008, +0.129] | +0.008 [-0.046, +0.062] |
| Long book vs cross-section | +0.003 [-0.028, +0.034] | +0.032 [+0.001, +0.064] | -0.006 [-0.033, +0.022] |
| Short book vs cross-section | +0.025 [-0.008, +0.058] | +0.038 [+0.002, +0.073] | +0.013 [-0.018, +0.044] |
| IC · long priority | +0.029 [+0.005, +0.052] | +0.039 [+0.015, +0.064] | +0.014 [-0.007, +0.035] |
| IC · short priority | +0.029 [+0.006, +0.052] | +0.039 [+0.015, +0.064] | +0.014 [-0.007, +0.035] |
| IC · the trace level | -0.029 [-0.052, -0.006] | -0.039 [-0.063, -0.015] | -0.014 [-0.035, +0.007] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | +0.042 [+0.008, +0.077] | -0.020 [-0.049, +0.007] |
| Long book vs cross-section | +0.029 [+0.009, +0.049] | -0.009 [-0.026, +0.007] |
| Short book vs cross-section | +0.013 [-0.008, +0.036] | -0.011 [-0.030, +0.007] |
| IC · long priority | +0.010 [-0.000, +0.021] | -0.015 [-0.026, -0.004] |
| IC · short priority | +0.010 [+0.000, +0.022] | -0.015 [-0.026, -0.004] |

## The screen · discovery

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | +0.060 [+0.013, +0.101] | +0.055 [-0.000, +0.104] | +0.057 [+0.020, +0.096] |
| Long book vs cross-section | +0.026 [-0.000, +0.049] | +0.025 [-0.005, +0.050] | +0.029 [+0.008, +0.051] |
| Short book vs cross-section | +0.034 [+0.006, +0.060] | +0.030 [-0.006, +0.062] | +0.028 [+0.006, +0.050] |
| IC · long priority | +0.024 [+0.004, +0.040] | +0.018 [-0.002, +0.036] | +0.029 [+0.014, +0.044] |
| IC · short priority | +0.023 [+0.004, +0.040] | +0.018 [-0.003, +0.036] | +0.028 [+0.013, +0.044] |
| IC · the trace level | -0.023 [-0.040, -0.004] | -0.018 [-0.036, +0.002] | -0.028 [-0.044, -0.013] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | -0.005 [-0.032, +0.022] | -0.011 [-0.033, +0.015] |
| Long book vs cross-section | -0.001 [-0.019, +0.016] | -0.001 [-0.015, +0.015] |
| Short book vs cross-section | -0.004 [-0.023, +0.013] | -0.010 [-0.025, +0.007] |
| IC · long priority | -0.005 [-0.014, +0.003] | +0.003 [-0.005, +0.012] |
| IC · short priority | -0.005 [-0.014, +0.004] | +0.003 [-0.005, +0.012] |

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
