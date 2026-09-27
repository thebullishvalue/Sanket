# Stack diagnosis · why the v8.0.0 ranking ran backwards, and what replaced it

Five NSE universes (fixed-seed sample of ≤ 80 names each), Daily, ~15 years of Yahoo Finance prices with the macro drivers on. Discovery = before 2020-10-22; holdout = after, sealed until the design was chosen. Scores are each name's forward return, drift removed and divided by its own σ inside the era. `*` = 95% block-bootstrap interval excludes zero.

## 1 · Every component points the same way — negative

Rank IC with the next 10 bars, discovery era:

| Component | NIFTY 50 | NIFTY 100 | NIFTY 200 | NIFTY MIDCAP 100 | NIFTY SMLCAP 100 |
|---|---|---|---|---|---|
| PRG_CTape | -0.020 | -0.025* | -0.019 | -0.010 | -0.027* |
| PRG_VTape | -0.028* | -0.033* | -0.027* | -0.016 | -0.036* |
| PRG_Conv | -0.020 | -0.029* | -0.021* | -0.017 | -0.041* |
| PRG_Value | -0.026* | -0.030* | -0.030* | -0.017* | -0.032* |
| PRG_Trace | -0.023* | -0.032* | -0.026* | -0.014 | -0.038* |
| PRG_Hist_Z | -0.012 | -0.016* | -0.021* | -0.006 | -0.017* |
| CVG_Units | -0.008 | -0.018 | -0.014 | -0.000 | -0.011 |

The same sign holds at 5, 20 and 40 bars and on Weekly; on the holdout it holds for the large-cap universes and fades toward zero on NIFTY 200, Midcap and Smallcap.

## 2 · The grid's cells, excess score vs the date's cross-section (10 bars)

| Cell | units | disc · 50 | disc · 100 | disc · 200 | disc · MIDCAP_100 | disc · SMLCAP_100 | hold · 50 | hold · 100 | hold · 200 | hold · MIDCAP_100 | hold · SMLCAP_100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Watch · still falling | 1 | +0.051 | +0.005 | +0.129* | +0.211* | +0.001 | +0.133* | +0.110* | +0.089* | +0.038 | +0.089 |
| Reduce · downtrend | 0.5 | +0.078* | +0.082* | +0.048 | +0.010 | +0.067 | +0.089* | +0.105* | +0.071 | +0.102* | -0.067 |
| Reduce · breakdown | 0.5 | -0.117 | -0.086 | -0.084 | -0.050 | +0.019 | +0.097 | +0.058 | -0.016 | +0.007 | +0.047 |
| Exit · distribution | 0.25 | -0.143 | -0.226 | -0.017 | -0.073 | -0.326 | +0.250 | +0.140 | -0.169* | +0.097 | +0.001 |
| Accumulate · deep value | 1.5 | +0.020 | +0.038 | +0.037 | +0.029 | +0.098* | -0.027 | +0.021 | -0.020 | -0.040 | -0.007 |
| Wait · no edge | 1 | +0.021 | +0.031* | -0.025 | -0.008 | -0.024 | +0.052* | +0.034 | -0.001 | -0.008 | -0.022 |
| Trim · rolling over | 0.75 | +0.017 | +0.027 | +0.031 | -0.067* | -0.086* | +0.017 | -0.008 | +0.059* | -0.015 | +0.006 |
| Trim · topping | 0.75 | -0.088 | -0.125* | -0.057 | -0.150 | +0.072 | +0.078 | +0.039 | +0.113 | +0.143 | +0.035 |
| Accumulate · basing | 1.5 | -0.028 | +0.044 | -0.011 | -0.097* | +0.022 | -0.157* | -0.099* | -0.067 | -0.050 | -0.060 |
| Accumulate · early turn | 1.5 | +0.009 | +0.011 | -0.002 | +0.002 | +0.024 | -0.014 | -0.008 | -0.051* | -0.082* | -0.012 |
| Wait · drifting | 1 | -0.026 | -0.002 | +0.004 | +0.033 | -0.008 | +0.020 | +0.005 | -0.008 | -0.014 | -0.002 |
| Trim · stalling | 0.75 | -0.052 | -0.020 | -0.011 | +0.018 | -0.043 | -0.001 | -0.040 | +0.097* | +0.046 | -0.047 |
| Buy · turn | 3 | +0.041 | +0.038 | -0.065 | +0.095 | -0.127 | -0.072 | -0.038 | -0.018 | +0.001 | -0.100 |
| Add · trend | 3 | +0.037 | +0.040 | +0.001 | -0.008 | -0.014 | -0.019 | -0.036 | -0.059* | -0.037 | -0.015 |
| Add · strong trend | 3 | -0.050* | -0.044* | -0.046* | -0.018 | -0.037 | -0.063* | -0.026* | -0.010 | +0.002 | +0.011 |
| Hold · don't add | 1.5 | -0.046 | -0.076* | -0.052 | -0.048 | +0.016 | -0.060 | -0.008 | -0.043 | +0.009 | +0.010 |

## 3 · Designs, long − short spread (10 bars)

**Discovery — the design was chosen here**

| Design | NIFTY 50 | NIFTY 100 | NIFTY 200 | NIFTY MIDCAP 100 | NIFTY SMLCAP 100 | avg |
|---|---|---|---|---|---|---|
| D0 current | -0.022 | -0.021 | -0.029 | -0.002 | -0.036 | -0.022 |
| D2 stretch only | +0.058* | +0.065* | +0.072* | +0.032 | +0.064* | +0.058 |
| D9 TURN today on top, then stretch | +0.060* | +0.065* | +0.072* | +0.032 | +0.064* | +0.059 |
| D10 TURN+RESUME today on top | +0.057* | +0.061* | +0.073* | +0.030 | +0.060* | +0.056 |
| D11 all bands (hold/armed) + stretch | +0.024 | +0.032 | +0.045* | +0.020 | +0.040* | +0.032 |
| B  -ret20 | +0.052 | +0.073* | +0.083* | +0.049* | +0.062* | +0.064 |
| D2 -trace | +0.058* | +0.065* | +0.072* | +0.032 | +0.064* | +0.058 |
| D7 rank avg(-ret20,-trace) | +0.063* | +0.076* | +0.078* | +0.042 | +0.071* | +0.066 |
| D5 -trace, push-confirmed | +0.017 | +0.013 | -0.011 | -0.012 | +0.031 | +0.008 |
| D8 -ret20, push-confirmed | +0.013 | +0.017 | -0.009 | -0.013 | +0.041* | +0.010 |
| R  trace ⟂ ret20 (resid) | -0.004 | +0.009 | -0.001 | -0.004 | -0.003 | -0.001 |
| R  ctape ⟂ ret20 (resid) | +0.017 | +0.010 | -0.005 | +0.010 | +0.007 | +0.008 |
| R  vtape ⟂ ret20 (resid) | +0.014 | +0.019 | -0.008 | -0.007 | -0.014 | +0.001 |
| R  histz ⟂ ret20 (resid) | -0.004 | -0.015 | -0.027 | -0.003 | -0.001 | -0.010 |

**Holdout — looked at once, after**

| Design | NIFTY 50 | NIFTY 100 | NIFTY 200 | NIFTY MIDCAP 100 | NIFTY SMLCAP 100 | avg |
|---|---|---|---|---|---|---|
| D0 current | -0.057* | -0.046* | -0.063* | -0.027 | -0.022 | -0.043 |
| D2 stretch only | +0.029 | +0.016 | -0.029 | -0.038 | -0.017 | -0.008 |
| D9 TURN today on top, then stretch | +0.029 | +0.017 | -0.027 | -0.036 | -0.018 | -0.007 |
| D10 TURN+RESUME today on top | +0.028 | +0.016 | -0.026 | -0.035 | -0.017 | -0.007 |
| D11 all bands (hold/armed) + stretch | +0.010 | +0.010 | -0.027 | -0.012 | -0.020 | -0.008 |
| B  -ret20 | +0.045 | +0.025 | -0.026 | -0.040 | -0.019 | -0.003 |
| D2 -trace | +0.029 | +0.016 | -0.029 | -0.038 | -0.017 | -0.008 |
| D7 rank avg(-ret20,-trace) | +0.041 | +0.027 | -0.031 | -0.040 | -0.008 | -0.002 |
| D5 -trace, push-confirmed | +0.056* | +0.034 | -0.008 | -0.004 | -0.037* | +0.008 |
| D8 -ret20, push-confirmed | +0.056* | +0.029 | -0.011 | -0.015 | -0.029 | +0.006 |
| R  trace ⟂ ret20 (resid) | +0.020 | +0.040 | +0.009 | -0.019 | -0.007 | +0.008 |
| R  ctape ⟂ ret20 (resid) | +0.092* | +0.100* | +0.045 | -0.018 | -0.005 | +0.043 |
| R  vtape ⟂ ret20 (resid) | +0.028 | +0.035 | +0.006 | -0.018 | -0.028 | +0.005 |
| R  histz ⟂ ret20 (resid) | +0.047* | +0.016 | +0.011 | +0.012 | -0.024 | +0.012 |

Chosen: **D9 — a TURN on this bar first, then stretch** (`engine.priorities`). The `R … ⟂ ret20` rows are each component with the name's own 20-bar return regressed out: ≈ 0 everywhere, so the ranking's information is short-term reversal read through the indicator. `push-confirmed` waits for the push to turn before ranking a stretch — it removes the edge.

The per-universe trace-setting comparisons under the new ranking are the `trace_study_*` reports beside this file.