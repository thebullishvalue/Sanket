# Pragati · a from-scratch audit — v7, v5, and the final v8

**Scope.** Every output of `pragati.pine` v6, through its Python port (bar for bar with the Pine), measured on real data with no reliance on the script's own claims. 380 instruments: 237 NSE stocks, 40 US large caps, 38 world indices, 23 commodities, 23 FX pairs, 19 cryptocurrencies; daily bars back to 2006 where they exist. **Discovery** = before 2018-01-01, **holdout** = from it.

**Scoring.** Each output becomes a position p ∈ [−1, 1] (+ = the output's own "up"). Score = p × the forward return from the next open over h bars, minus the instrument's own mean and divided by its own σ inside the era — timing skill in σ, drift removed, so simply holding a rising asset scores zero. `**` = 95% block-bootstrap interval (blocks of whole dates) excludes zero.

Reproduce: `pine_audit.py` (`baseline`, `sweep`, `experiments`, `capitulation`, `oi`; `v5`, `oi5` for sections 9–10). Sections 1–8 audit v6 → v7; **sections 9–11 audit the user's v5, compare it with v7, and record the decision: v8.**

## 1 · The defaults, every output (h = 10 bars)

**Discovery**

| | NSE | US | Indices | Commod. | FX | Crypto |
|---|---|---|---|---|---|---|
| trace | +0.018 | -0.027 | -0.007 | -0.030 | +0.000 | +0.184 |
| hist_z | -0.006 | -0.020 | -0.012 | -0.011 | -0.000 | +0.042 |
| conv | +0.019 | -0.029 | -0.001 | -0.042** | +0.004 | +0.193 |
| value | +0.009 | -0.023 | -0.011 | -0.015 | -0.003 | +0.141** |
| c_tape | +0.023 | -0.024 | +0.033 | -0.030 | +0.016 | +0.218** |
| v_tape | +0.028 | -0.021 | +0.018 | -0.007 | +0.015 | +0.162 |
| turn | -0.055 | +0.028 | -0.011 | +0.117 | +0.078 | +1.014 |
| resume | -0.003 | -0.062** | -0.022 | -0.015 | +0.006 | +0.444 |
| div | -0.037 | +0.034 | -0.056 | -0.029 | -0.091 | -0.418 |
| decl | -0.009 | +0.009 | -0.002 | +0.024** | -0.007 | -0.461** |
| grid_side | +0.007 | -0.017 | +0.008 | -0.030 | +0.000 | +0.192 |
| grid_units | +0.007 | -0.026 | +0.003 | -0.058** | +0.006 | +0.357** |
| mom20 | +0.023 | -0.020 | +0.005 | -0.015 | +0.009 | +0.153** |
| rev20 | -0.023 | +0.020 | -0.005 | +0.015 | -0.009 | -0.153** |

**Holdout**

| | NSE | US | Indices | Commod. | FX | Crypto |
|---|---|---|---|---|---|---|
| trace | +0.019 | -0.040 | -0.032 | -0.015 | -0.028 | +0.071** |
| hist_z | +0.016 | +0.006 | +0.027 | -0.017 | -0.011 | +0.025 |
| conv | +0.017 | -0.053** | -0.057 | -0.021 | -0.036 | +0.069** |
| value | +0.015 | -0.019 | -0.007 | -0.011 | -0.016 | +0.060** |
| c_tape | +0.010 | -0.077** | -0.076** | -0.028 | -0.071** | +0.066 |
| v_tape | +0.016 | -0.041 | -0.035 | -0.013 | -0.025 | +0.080** |
| turn | -0.020 | +0.071 | +0.131** | -0.025 | +0.041 | -0.083 |
| resume | +0.017 | -0.083** | -0.056 | -0.019 | -0.100 | +0.043 |
| div | -0.002 | +0.039 | +0.081 | +0.002 | +0.016 | -0.049 |
| decl | -0.003 | +0.009 | +0.007 | +0.004 | +0.019 | +0.003 |
| grid_side | -0.009 | -0.047 | -0.049 | -0.003 | -0.068** | -0.003 |
| grid_units | +0.008 | -0.054** | -0.052 | +0.005 | -0.091** | +0.052 |
| mom20 | +0.018 | -0.035 | -0.024 | +0.005 | -0.026 | +0.056** |
| rev20 | -0.018 | +0.035 | +0.024 | -0.005 | +0.026 | -0.056** |

The continuous readings track plain 20-bar momentum (`mom20`) in sign and size, and their sign flipped between eras on US stocks, indices and FX. The ▲▼ declaration (`decl`, held until the opposite one) is ≈ 0 everywhere. No reading is a dependable timing edge on its own.

## 2 · Does tuning transfer? 161 random parameter sets

Every signal-affecting input moved at random (lookback, smoothing, normalization, participation, denominator, inner zone, trace mix, signal EMA, confirmation, dislocation, impulse k, pullback, effort, cooldown, pivots, θ). For each output: the Spearman correlation, across sets, of a set's discovery edge with its holdout edge (h = 10).

| Output | NSE | US | Indices | Commod. | FX | Crypto |
|---|---|---|---|---|---|---|
| trace | +0.07 | +0.68 | -0.18 | +0.02 | -0.45 | -0.09 |
| c_tape | +0.51 | +0.10 | -0.39 | +0.31 | +0.27 | -0.37 |
| hist_z | +0.50 | -0.06 | +0.02 | -0.41 | +0.05 | +0.74 |
| turn | +0.25 | -0.13 | +0.05 | -0.17 | +0.22 | -0.27 |
| resume | +0.10 | +0.28 | +0.22 | +0.01 | -0.04 | +0.04 |
| div | +0.13 | -0.05 | -0.14 | -0.07 | -0.09 | +0.09 |
| decl | +0.30 | -0.33 | -0.31 | +0.22 | +0.09 | -0.03 |
| grid_side | -0.30 | +0.77 | +0.58 | +0.40 | +0.34 | +0.60 |
| grid_units | +0.17 | +0.61 | +0.37 | +0.23 | +0.15 | +0.34 |

Mean ρ over every output × class × horizon: **+0.10**. Picking the top tenth of sets on discovery changed the holdout edge by **-0.0065σ** against the median set — tuning made it worse. Averaged over classes, every value of every input was positive in discovery and negative in the holdout for the grid and the conviction tape: a sign flip, which no parameter fixes. **All defaults kept.**

## 3 · TURN — do its four gates earn their place? (h = 10)

**Discovery**

| | NSE | US | Indices | Commod. | FX | Crypto |
|---|---|---|---|---|---|---|
| all gates (v6) | -0.055 | +0.028 | -0.011 | +0.103 | +0.078 | +1.014 |
| − value gate | -0.027 | +0.040 | +0.023 | +0.068 | +0.024 | +0.536 |
| − conviction gate | -0.034 | +0.025 | +0.001 | +0.039 | +0.071 | +0.236 |
| − push gate | -0.039 | +0.032 | -0.002 | +0.076 | +0.028 | +1.014 |
| − failed-push evidence | -0.037 | +0.036 | -0.044 | +0.088 | +0.056 | +1.382 |
| trace cross only | -0.005 | +0.037 | +0.022 | +0.057 | +0.034 | +0.429 |

**Holdout**

| | NSE | US | Indices | Commod. | FX | Crypto |
|---|---|---|---|---|---|---|
| all gates (v6) | -0.020 | +0.071 | +0.131** | -0.019 | +0.041 | -0.083 |
| − value gate | -0.017 | +0.064 | +0.087 | +0.010 | +0.014 | -0.073 |
| − conviction gate | -0.038 | +0.053 | +0.133 | +0.014 | +0.011 | -0.070 |
| − push gate | -0.029 | +0.062 | +0.119** | +0.000 | +0.056 | -0.095 |
| − failed-push evidence | -0.029 | +0.075 | +0.107** | -0.025 | +0.020 | -0.080 |
| trace cross only | -0.043 | +0.043 | +0.023 | -0.011 | +0.027 | -0.068 |

The full rule is the best or equal-best on the holdout where TURN works at all (indices, US, FX); the bare cross is weaker. Unchanged in v7. It is negative on NSE in both eras and flipped on crypto.

## 4 · The grid, cell by cell (h = 10)

**Discovery** — mean score while a name sits in the cell (long)

| | NSE | US | Indices | Commod. | FX | Crypto |
|---|---|---|---|---|---|---|
| Buy · capitulation *(v6: Watch · still falling)* | +0.104** | +0.166** | +0.266** | -0.022 | +0.132 | — |
| Accumulate · washout *(v6: Reduce · downtrend)* | +0.000 | +0.094 | +0.086 | +0.055 | +0.065 | — |
| Reduce · breakdown | -0.065 | +0.089 | +0.043 | -0.091 | +0.033 | — |
| Exit · distribution | -0.229 | +0.297** | -0.322 | -0.391** | +0.251 | — |
| Accumulate · deep value | -0.007 | +0.073 | +0.253** | +0.034 | +0.024 | — |
| Wait · no edge | -0.002 | +0.083** | +0.113 | +0.035 | -0.009 | +0.517 |
| Trim · rolling over | -0.012 | +0.069 | +0.025 | +0.024 | -0.031 | — |
| Trim · topping | -0.061 | +0.099 | +0.030 | +0.041 | +0.026 | — |
| Accumulate · basing | -0.001 | +0.164** | +0.057 | +0.054 | -0.039 | — |
| Accumulate · early turn | +0.004 | +0.048 | +0.028 | -0.005 | +0.019 | -0.014 |
| Wait · drifting | +0.026 | +0.033 | -0.021 | -0.031 | -0.032 | +0.554 |
| Trim · stalling | +0.028 | -0.009 | +0.120** | +0.036 | -0.015 | +1.287** |
| Buy · turn | +0.078 | +0.084 | +0.228** | +0.074 | +0.256 | +1.880 |
| Add · trend | +0.026 | -0.020 | +0.010 | -0.099 | -0.013 | +0.364** |
| Hold · extended *(v6: Add · strong trend)* | -0.001 | -0.044 | -0.014 | -0.086** | +0.017 | +0.361** |
| Hold · don't add | -0.024 | +0.030 | +0.048 | -0.067 | +0.036 | +0.639 |

**Holdout** — mean score while a name sits in the cell (long)

| | NSE | US | Indices | Commod. | FX | Crypto |
|---|---|---|---|---|---|---|
| Buy · capitulation *(v6: Watch · still falling)* | +0.043 | +0.185 | +0.398** | +0.171** | +0.277** | -0.000 |
| Accumulate · washout *(v6: Reduce · downtrend)* | +0.057 | +0.170** | +0.211** | +0.081** | +0.121** | +0.001 |
| Reduce · breakdown | +0.024 | +0.064 | +0.209 | -0.012 | +0.062 | +0.009 |
| Exit · distribution | -0.037 | -0.190 | +0.356 | -0.260** | -0.195 | -0.089 |
| Accumulate · deep value | -0.027 | +0.113 | +0.138 | -0.091 | -0.028 | +0.080 |
| Wait · no edge | -0.020 | +0.022 | +0.052 | -0.021 | +0.016 | -0.049 |
| Trim · rolling over | +0.016 | +0.036 | +0.050 | -0.021 | +0.049 | +0.028 |
| Trim · topping | +0.040 | +0.152 | +0.097 | -0.104 | -0.121 | +0.048 |
| Accumulate · basing | -0.084 | -0.099 | -0.237 | +0.055 | -0.039 | +0.114 |
| Accumulate · early turn | -0.046 | -0.015 | -0.011 | -0.028 | -0.013 | -0.095** |
| Wait · drifting | -0.004 | -0.003 | -0.005 | +0.010 | -0.017 | -0.022 |
| Trim · stalling | -0.028 | +0.017 | -0.006 | -0.007 | +0.093 | +0.169 |
| Buy · turn | -0.002 | +0.125 | -0.368 | +0.172 | +0.073 | +0.333** |
| Add · trend | +0.010 | -0.037 | -0.020 | +0.017 | -0.226** | -0.053 |
| Hold · extended *(v6: Add · strong trend)* | +0.020 | -0.069** | -0.053 | +0.017 | -0.081** | +0.113 |
| Hold · don't add | +0.018 | -0.101** | -0.061 | -0.069 | -0.078 | +0.275** |

## 5 · Capitulation — the robust finding (h = 10 / 20)

Sellers **firmly** in control (conviction tape ≤ −30, push-gated row) while price is **cheap or below fair** (value tape < 0). Benchmarks: plain oversold (20-bar return ≤ −1.5σ of its own year) and the same thresholds read off the raw tapes without the grid's push-gated row.

**Discovery · h = 10**

| | NSE | US | Indices | Commod. | FX | Crypto |
|---|---|---|---|---|---|---|
| in either cell | +0.033 | +0.114** | +0.152 | +0.035 | +0.079 | — |
| sellers firm · cheap | +0.104** | +0.166** | +0.266** | -0.022 | +0.132 | — |
| sellers firm · below fair | +0.000 | +0.094 | +0.086 | +0.055 | +0.065 | — |
| … and push ↑ | +0.041 | +0.052 | -0.019 | +0.082 | +0.078 | — |
| … and push ↓ | +0.044 | +0.153** | +0.273** | +0.032 | +0.049 | — |
| on entry only | +0.003 | +0.115 | +0.268** | +0.013 | +0.011 | — |
| raw tapes, no push-gated row | -0.064 | +0.041 | -0.187 | +0.008 | -0.018 | -0.294 |
| BENCH plain oversold | -0.004 | +0.132 | +0.030 | -0.054 | +0.044 | +0.386** |
| capitulation ∧ oversold | +0.069 | +0.132** | +0.146 | +0.007 | +0.141** | — |
| oversold ∧ NOT capitulation | -0.049 | +0.049 | -0.041 | -0.048 | -0.052 | +0.070 |
| short: buyers firm · above fair / rich | +0.005 | +0.032 | +0.006 | +0.082** | -0.022 | -0.429** |

**Holdout · h = 10**

| | NSE | US | Indices | Commod. | FX | Crypto |
|---|---|---|---|---|---|---|
| in either cell | +0.053 | +0.175** | +0.283** | +0.103** | +0.155** | +0.001 |
| sellers firm · cheap | +0.043 | +0.185 | +0.398** | +0.171** | +0.277** | -0.000 |
| sellers firm · below fair | +0.057 | +0.170** | +0.211** | +0.081** | +0.121** | +0.001 |
| … and push ↑ | +0.113 | +0.200** | +0.368** | +0.132** | +0.104 | -0.032 |
| … and push ↓ | +0.019 | +0.176** | +0.212 | +0.155** | +0.170** | +0.002 |
| on entry only | +0.001 | +0.107 | +0.067 | +0.058 | +0.182** | -0.014 |
| raw tapes, no push-gated row | +0.038 | +0.190** | +0.290** | +0.124** | +0.136** | -0.015 |
| BENCH plain oversold | +0.011 | +0.073 | +0.120 | +0.084 | +0.079 | -0.022 |
| capitulation ∧ oversold | +0.079 | +0.174 | +0.374** | +0.164** | +0.199** | -0.044 |
| oversold ∧ NOT capitulation | -0.042 | +0.019 | -0.010 | -0.005 | -0.016 | -0.073 |
| short: buyers firm · above fair / rich | -0.020 | +0.074** | +0.054 | +0.003 | +0.080 | -0.151** |

**Discovery · h = 20**

| | NSE | US | Indices | Commod. | FX | Crypto |
|---|---|---|---|---|---|---|
| in either cell | +0.045 | +0.132 | +0.164 | +0.025 | +0.089 | — |
| sellers firm · cheap | +0.100 | +0.170** | +0.220 | -0.001 | +0.130 | — |
| sellers firm · below fair | +0.019 | +0.117 | +0.132 | +0.033 | +0.079 | — |
| … and push ↑ | +0.066 | +0.048 | +0.060 | +0.058 | +0.068 | — |
| … and push ↓ | +0.051 | +0.180 | +0.222 | +0.026 | +0.074 | — |
| on entry only | -0.008 | +0.178** | +0.219 | -0.017 | +0.071 | — |
| raw tapes, no push-gated row | -0.054 | +0.006 | -0.293 | +0.009 | -0.017 | -0.246 |
| BENCH plain oversold | -0.027 | +0.079 | +0.007 | -0.080 | +0.021 | +0.345 |
| capitulation ∧ oversold | +0.066 | +0.137 | +0.179 | +0.058 | +0.121** | — |
| oversold ∧ NOT capitulation | -0.060 | +0.033 | -0.043 | -0.087 | -0.063 | +0.069 |
| short: buyers firm · above fair / rich | -0.005 | +0.027 | +0.007 | +0.090 | -0.028 | -0.534** |

**Holdout · h = 20**

| | NSE | US | Indices | Commod. | FX | Crypto |
|---|---|---|---|---|---|---|
| in either cell | +0.079 | +0.245** | +0.354** | +0.114 | +0.178** | -0.025 |
| sellers firm · cheap | +0.090 | +0.303** | +0.454** | +0.163 | +0.295** | -0.045 |
| sellers firm · below fair | +0.073 | +0.219** | +0.292** | +0.098 | +0.145** | -0.021 |
| … and push ↑ | +0.110 | +0.259** | +0.396** | +0.109 | +0.175** | -0.064 |
| … and push ↓ | +0.063 | +0.282** | +0.338** | +0.215** | +0.170** | -0.054 |
| on entry only | -0.002 | +0.160 | +0.156 | +0.055 | +0.183** | -0.022 |
| raw tapes, no push-gated row | +0.063 | +0.258** | +0.377** | +0.149** | +0.163** | -0.017 |
| BENCH plain oversold | +0.048 | +0.139 | +0.206 | +0.028 | +0.064 | -0.032 |
| capitulation ∧ oversold | +0.105 | +0.241** | +0.414** | +0.145 | +0.241** | -0.101 |
| oversold ∧ NOT capitulation | -0.019 | +0.058 | +0.062 | -0.029 | -0.027 | -0.112 |
| short: buyers firm · above fair / rich | -0.017 | +0.120** | +0.087 | +0.008 | +0.122** | -0.164 |

Positive in **both eras on every class but crypto**; significant on the holdout on US, indices, commodities and FX. It beats plain oversold, and oversold names *outside* these cells lagged — the stack carries information price alone does not. The grid's push-gated row matters (the raw-tape version is negative in discovery on NSE and indices). The push's direction inside the cells made no consistent difference. The short mirror worked only after 2018 and lost on crypto: not promoted.

## 6 · v7 — what changed and what it did

| Cell | v6 | v7 |
|---|---|---|
| sellers firm · cheap | Watch · still falling · 1u | **Buy · capitulation · 3u** |
| sellers firm · below fair | Reduce · downtrend · ½u | **Accumulate · washout · 1½u** |
| buyers firm · above fair | Add · strong trend · 3u | **Hold · extended · 1½u** |

The grid read as a position, before and after (h = 10):

**v6 · discovery**

| | NSE | US | Indices | Commod. | FX | Crypto |
|---|---|---|---|---|---|---|
| grid_units | +0.007 | -0.026 | +0.003 | -0.058** | +0.006 | +0.357** |
| grid_side | +0.007 | -0.017 | +0.008 | -0.030 | +0.000 | +0.192 |

**v6 · holdout**

| | NSE | US | Indices | Commod. | FX | Crypto |
|---|---|---|---|---|---|---|
| grid_units | +0.008 | -0.054** | -0.052 | +0.005 | -0.091** | +0.052 |
| grid_side | -0.009 | -0.047 | -0.049 | -0.003 | -0.068** | -0.003 |

**v7 · discovery**

| | NSE | US | Indices | Commod. | FX | Crypto |
|---|---|---|---|---|---|---|
| grid_units | +0.019 | +0.012 | +0.039 | -0.027 | +0.035 | +0.352** |
| grid_side | +0.015 | +0.024 | +0.044 | +0.002 | +0.029 | -0.129 |

**v7 · holdout**

| | NSE | US | Indices | Commod. | FX | Crypto |
|---|---|---|---|---|---|---|
| grid_units | +0.007 | -0.003 | -0.003 | +0.032 | -0.021 | +0.011 |
| grid_side | -0.011 | +0.014 | +0.008 | +0.022 | -0.003 | -0.043** |

Every significantly negative non-crypto cell of v6's holdout is gone. **Crypto is the stated limit**: it trends, the capitulation cells carry nothing there, and v7's grid side reads −0.043σ** on its holdout (v6 −0.003). A class switch was considered and not made — the crypto difference is not significant on the units read, and one map is simpler to trust.

## 7 · Open interest

240 NSE F&O stocks, total futures OI across expiries from the exchange's daily bhavcopy, 2019-01 → 2026-09, with the Pine's `f_oiState` ported line for line (10-day window, 4σ roll filter, 90th-percentile crowding). Discovery 2019–2022, holdout 2023 onward. Long position while in the state; for the pushes, the push's own direction.

| State (h = 10) | Discovery | Holdout |
|---|---|---|
| Long build-up | +0.017 | +0.007 |
| Short build-up | -0.038 | -0.012 |
| Short covering | -0.025 | -0.047 |
| Long unwinding | -0.035 | +0.030 |
| OI read: +LB / −SB | +0.029 | +0.010 |
| BENCH price direction, 10d | +0.007 | -0.013 |
| Crowded (OI top tenth of its year) | -0.060 | +0.018 |
| Crowded ∧ short build-up | -0.091 | +0.012 |
| Push made mostly by exits (the gold cast) | +0.037 | -0.035** |
| Push not made by exits | +0.025 | -0.036** |

No OI character and not 'crowded' keeps its sign across the two eras. The gold cast marks nothing: pushes made mostly by exits continued exactly as other pushes did. **v7:** the cast no longer vetoes a push from moving the grid's row (it had only its meaning to stand on); the OI readout and the cast stay as descriptions of who is trading.

## 8 · What v7 is, and is not

- **Is:** v6 with three grid cells re-weighted and one OI rule removed, each backed by evidence that
  held in both eras across asset classes. Every engine, signal, input and default is v6's.
- **Is not:** a tuned indicator. The parameter sweep says tuning this family does not transfer, so
  nothing was tuned.
- **Is not:** a timing system on its own. The capitulation state is active on ~6% of bars; as a
  standalone long-only rule its Sharpe (0.3–0.7 by class) stays below buy-and-hold. It is a tilt —
  where to lean — which is how the grid is meant to be read.
- **Limits:** crypto trends and the reversion reading fails there; NSE stocks show the weakest
  effects of the non-crypto classes; OI history covers 2019 onward only.

## 9 · v5 against v7 — the same audit, head to head

The user's `pragati.pine` v5 (kept at `archive/pragati_v5.pine`) was ported where it differs from v6/v7 (`pine_v5.py`: its ▲▼ with absorption as the only failed-push evidence, its ◆ without the in-zone requirement, its R / H divergence marks, its graded 3 × 3 grid with the 5 × 5 phases, its OI quadrant) and checked bar for bar: the port's signals match `pine_v5.signals` exactly on the test names, and the graded units match Pragyam's `cvgrid.graded_units` to 1e-15 wherever the phases do not halve an edge. Same 380 instruments, same scoring, same split.

**Signals (h = 10; avg = mean of the five non-crypto classes)**

| | disc avg | hold avg | holdout by class (NSE · US · Idx · Cmd · FX · Crypto) |
|---|---|---|---|
| v5 ▲▼ TURN | +0.032 | +0.033 | -0.024 · +0.066 · +0.125 · -0.030 · +0.027 · -0.094 |
| v7 ▲▼ TURN | +0.029 | +0.041 | -0.020 · +0.071 · +0.131** · -0.019 · +0.041 · -0.083 |
| v5 ◆ RESUME | -0.021 | -0.041 | +0.016 · -0.083** · -0.042 · -0.002 · -0.095 · +0.068 |
| v7 ◆ RESUME | -0.019 | -0.050 | +0.017 · -0.083** · -0.056 · -0.027 · -0.100 · +0.043 |
| v5 declaration held | +0.004 | +0.005 | |
| v7 declaration held | +0.003 | +0.007 | |
| v5 R divergence | -0.031 | +0.024 | |
| v5 H divergence | -0.049 | -0.089 | |

The signal sets **tie** — paired by date, v5 − v7 is ≈ 0 for TURN and for the declaration. ◆ RESUME is negative in both eras in both versions, significantly on US stocks in both. R has no stable sign; H is negative in both eras. The source's claim that regular divergence "ranked first" did not reproduce.

**The grid read as a position (graded units, centred on Wait; avg of non-crypto classes)**

| | h 10 disc | h 10 hold | h 20 disc | h 20 hold |
|---|---|---|---|---|
| v5, its own units (U0) | -0.007 | -0.027 | -0.003 | -0.043 |
| v7 4 × 4 | +0.016 | +0.002 | +0.019 | +0.003 |
| **v5 grid, units U4** | **+0.039** | **+0.039** | **+0.044** | **+0.051** |

U4 — DOWN·cheap 3, DOWN·fair 1½, UP·fair 1½, UP·rich ¾ — was chosen among five pre-registered unit sets (U0–U4) **on the discovery era alone**, from section 5's finding, and read once on the holdout. U4 by class on the holdout at h 10: NSE +0.006, US +0.050, indices +0.063, commodities +0.044, FX +0.029, crypto -0.003. Paired by date on the holdout: **U4 − v7 = +0.017σ** (h 10) and **+0.020σ** (h 20), both significant (discovery +0.007 / +0.009); U4 − U0 = +0.040σ / +0.052σ, significant.

## 10 · Open interest, v5's reading

v5 reads OI as a quadrant on the push and casts a gold histogram when a push is made mostly by exits. Same 240 F&O stocks, same split (h = 10):

| | Discovery | Holdout |
|---|---|---|
| Long build-up | -0.007 | -0.029 |
| Short build-up | -0.040 | -0.001 |
| Short covering | +0.004 | -0.021 |
| Long unwinding | +0.024 | +0.063 |
| Push with the gold cast | +0.027 | **-0.060\*\*** |
| Push without it | +0.041 | -0.018 |

No quadrant carries a stable edge of its own; long unwinding (a move against exiting longs) was followed by gains in both eras, but neither is significant. The cast does what a caution colour should: pushes made by exits did worse than other pushes **in both eras** (−0.014 and −0.042). As a colour it earns its place; as a veto on the grid (v6) it did not.

## 11 · The decision — Pragati v8

**v8 = v5, plus exactly what the data backed in both eras:**

1. **Grid units U4**, with the actions renamed to what was measured: DOWN·cheap *Buy · capitulation 3*, DOWN·fair *Accumulate · washout 1½*, UP·fair *Hold · building 1½*, UP·rich *Trim · paid ¾*.
2. **◆ RESUME off by default** — negative in both eras. It stays in the code and can be switched on.
3. **Conviction ladder default: Ladder up** (the ports have always run it; Ladder down needs intraday history no free feed carries, so it is unmeasured here).
4. **OI gold cast on by default**, colour only.
5. **R and H** keep their defaults and are drawn; their tooltips now say what the audit found.

Why v5 over v7: the signals tie, and v5's 3 × 3 grid with U4 beats v7's 4 × 4 on the holdout, paired and significant at both horizons. v7 is kept at `archive/pragati_v7.pine` and `cvgrid4.py` so this comparison stays reproducible.

**Confirmed in Pragyam's allocator** (`research/cvg_reweight.py` there; monthly, every name held, net of 10bp India / 3bp US costs, decided before 2018 and confirmed once after), %/yr:

| | v8 − seed, <2018 | v8 − seed, ≥2018 | v8 − EW, <2018 | v8 − EW, ≥2018 | turnover v8 / seed |
|---|---|---|---|---|---|
| Nifty 50 | +0.42 (t 0.6) | +0.98 (t 1.5) | +0.83 (t 2.1) | +0.47 (t 1.3) | 1.28x / 1.47x |
| Dow 30 | +0.55 (t 1.0) | +0.89 (t 1.0) | -0.23 | +0.90 (t 2.4) | 1.17x / 1.47x |

Positive in both eras on both panels at lower turnover, so Pragyam's units changed to match. No single t against the seed clears 2; it is shipped as consistent, not proven. The ETF book (1–27 funds, from 2012) is too thin to split and was not tested.

**Limits, restated for v8.** Crypto trends and the reversion reading fails there. NSE stocks show the weakest effects. Nothing was tuned — section 2 still holds. The `.pine` file cannot be compiled here: load it in TradingView and check it compiles before relying on it.
