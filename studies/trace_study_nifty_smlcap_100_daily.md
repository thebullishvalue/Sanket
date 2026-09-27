# Trace study · NIFTY SMLCAP 100 · Daily

70 names · Daily · 2011-11-04 → 2026-09-25 · holdout from 2020-10-22 (40%) · hold 10 bars · book = top 20% · participation ratio 15.0 · macro drivers on · measured 2026-09-27 12:39

**Decision:** Conviction only beats Conviction × value on the holdout long−short spread, paired, CI excluding zero, BUT its own screen shows no edge on the holdout — it loses less than the default, it does not pick winners; discovery does not confirm the difference.

**Warning:** the screen ranks BACKWARDS on the holdout under Value only — the long book trailed the short book beyond noise.

Scores are in σ of each name's own h-bar return (drift removed within era). Brackets are 95% block-bootstrap intervals.

## The screen · holdout

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | -0.023 [-0.057, +0.014] | +0.021 [-0.012, +0.057] | -0.049 [-0.088, -0.010] |
| Long book vs cross-section | -0.017 [-0.039, +0.008] | +0.017 [-0.009, +0.044] | -0.022 [-0.046, +0.002] |
| Short book vs cross-section | -0.006 [-0.030, +0.018] | +0.004 [-0.017, +0.027] | -0.027 [-0.051, -0.003] |
| IC · long priority | -0.010 [-0.027, +0.007] | +0.005 [-0.010, +0.020] | -0.013 [-0.029, +0.002] |
| IC · short priority | -0.010 [-0.025, +0.005] | -0.002 [-0.016, +0.013] | -0.019 [-0.034, -0.004] |
| IC · the trace level | -0.011 [-0.028, +0.004] | -0.005 [-0.022, +0.013] | -0.012 [-0.026, +0.003] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | +0.044 [+0.003, +0.084] | -0.026 [-0.055, +0.001] |
| Long book vs cross-section | +0.033 [+0.008, +0.059] | -0.005 [-0.030, +0.017] |
| Short book vs cross-section | +0.011 [-0.017, +0.038] | -0.020 [-0.039, -0.002] |
| IC · long priority | +0.015 [+0.004, +0.025] | -0.003 [-0.013, +0.007] |
| IC · short priority | +0.008 [-0.005, +0.022] | -0.009 [-0.018, -0.000] |

## The screen · discovery

| Metric | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long − short spread | -0.033 [-0.080, +0.010] | -0.006 [-0.049, +0.034] | -0.026 [-0.068, +0.017] |
| Long book vs cross-section | -0.021 [-0.050, +0.007] | -0.016 [-0.047, +0.014] | -0.018 [-0.043, +0.007] |
| Short book vs cross-section | -0.012 [-0.041, +0.013] | +0.009 [-0.017, +0.034] | -0.010 [-0.034, +0.013] |
| IC · long priority | -0.006 [-0.024, +0.010] | +0.000 [-0.017, +0.016] | -0.010 [-0.028, +0.005] |
| IC · short priority | -0.006 [-0.022, +0.009] | +0.004 [-0.012, +0.018] | -0.013 [-0.029, +0.002] |
| IC · the trace level | -0.037 [-0.054, -0.018] | -0.029 [-0.048, -0.010] | -0.033 [-0.048, -0.019] |

**Paired against Conviction × value** (mode − default, same dates):

| Metric | Conviction only | Value only |
|---|---|---|
| Long − short spread | +0.032 [-0.008, +0.076] | +0.014 [-0.014, +0.043] |
| Long book vs cross-section | +0.011 [-0.017, +0.040] | +0.006 [-0.015, +0.025] |
| Short book vs cross-section | +0.023 [-0.005, +0.050] | +0.006 [-0.012, +0.023] |
| IC · long priority | +0.009 [-0.003, +0.021] | -0.003 [-0.013, +0.007] |
| IC · short priority | +0.009 [-0.005, +0.024] | -0.006 [-0.016, +0.004] |

## The events (Edge Study method)

| Slice | Conviction × value | Conviction only | Value only |
|---|---|---|---|
| Long · all (holdout) | UNDERPOWERED · -0.067 [-0.191, +0.062] · n 852 | UNDERPOWERED · -0.081 [-0.216, +0.076] · n 454 | UNDERPOWERED · -0.050 [-0.145, +0.052] · n 1226 |
| Short · all (holdout) | UNDERPOWERED · -0.016 [-0.180, +0.117] · n 305 | UNDERPOWERED · -0.017 [-0.144, +0.097] · n 486 | UNDERPOWERED · -0.029 [-0.253, +0.166] · n 283 |
| ▲ TURN (holdout) | UNDERPOWERED · -0.143 [-0.342, +0.087] · n 150 | UNDERPOWERED · +0.029 [-0.137, +0.199] · n 154 | UNDERPOWERED · -0.185 [-0.339, -0.005] · n 198 |
| ▼ TURN (holdout) | UNDERPOWERED · -0.014 [-0.171, +0.133] · n 237 | UNDERPOWERED · -0.022 [-0.158, +0.103] · n 468 | UNDERPOWERED · -0.047 [-0.255, +0.145] · n 183 |
| ◆ RESUME ↑ (holdout) | UNDERPOWERED · -0.050 [-0.153, +0.074] · n 702 | UNDERPOWERED · -0.137 [-0.306, +0.031] · n 300 | UNDERPOWERED · -0.024 [-0.117, +0.085] · n 1028 |
| ◆ RESUME ↓ (holdout) | UNDERPOWERED · -0.021 [-0.130, +0.297] · n 68 | UNDERPOWERED · +0.105 [+nan, +nan] · n 18 | UNDERPOWERED · +0.003 [-0.301, +0.302] · n 100 |

Rule, fixed in advance: a mode replaces the default only if the holdout interval of its paired long−short difference excludes zero.
