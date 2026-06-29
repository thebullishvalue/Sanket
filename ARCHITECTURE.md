# Sanket — Engine Architecture & Research Basis

> This document records *why* the engine is built the way it is. Every design choice below
> is backed by validation on real NSE F&O data (147 names, 5 years, ~170k symbol-days,
> fetched via the app's own universe + yfinance). Nothing here is decorative.

## The thesis (what edge we exploit)

**Short-horizon cross-sectional reversion.** On daily NSE data, names that have moved most
relative to their own volatility tend to *under*-perform their peers over the next 1–5 days,
and names that have sold off tend to out-perform. This is the dominant, robust, and
*tradable-direction-correct* anomaly in this universe.

### Evidence
- Per-feature reversion rank-IC vs 1–5 day forward returns: **+0.025 to +0.031, t up to +8.6**
  (strongest: distance-from-MA `dist5`/`dist10`, and 2–5 day ATR-normalized returns).
- A combined reversion composite holds **walk-forward, positive every year 2021–2025**
  (IC +0.025 … +0.049). It went **dormant in 2026** (IC −0.01, t−0.9) — *flat, not inverted*
  (momentum was also flat), so this is an on/off anomaly, not a regime flip.
- It is **regime-dependent**: IC **+0.055 in HIGH-vol** regimes vs ~+0.025 LOW/NORMAL, noise
  in EXTREME. Reversion pays most when volatility is elevated but not chaotic.

### What we explicitly rejected (and why)
- **The old WRCI / Conviction / Pulse momentum factor stack** anti-predicts on this data:
  naked Priority IC = −0.023 (t−3.9); the Optuna calibrator *cannot* fix it because its
  factor-weight bounds are non-negative, so with anti-predictive factors the best it can do
  is shrink to noise. **Removed entirely.**
- **The 3-layer Intelligence stack** (per-set logistic + Meta fusion) added no out-of-sample
  ranking edge over naked priority and was fragile (silent −100 sentinel, `val_score=None`).
  **Removed.**
- **Inferred order-flow (delta/CVD/divergence/absorption) as a ranking factor**: adds *nothing*
  to cross-sectional IC (identical to 4 dp with/without). The delta is reconstructed from
  candle shape, not real tape — too weak and too sparse to rank a universe. **Demoted to
  descriptive UI context only; not in the score.**

## Cost reality (why this is a ranker, not a high-turnover strategy)

The reversion signal mean-reverts fast → high turnover. After realistic costs (10–15 bps/side):
- Daily-rebalanced L/S: **net Sharpe negative** (−1.3). Costs eat the entire gross edge.
- ~5-day holding period: **net ≈ break-even to slightly positive** at ≤7–10 bps.
- Concentration into the extreme tails *dilutes* edge — the alpha is a **broad, diffuse tilt**
  across the cross-section, not a few screaming names.

**Conclusion:** this is a **decision-support ranker** (a daily long/short shortlist with
conviction + risk context for a human who holds days, amortizing cost), not a standalone
high-frequency L/S book. The product surfaces ranked conviction; it does not pretend to be a
costless alpha.

## The engine (what we build)

### 1. Reversion score (cross-sectional, per date)
A blend of robustly-z-scored (median/MAD, within-date) reversion features, oriented so a
HIGHER score = more attractive LONG:
```
score = mean of  -z(ret2), -z(ret5), -z(dist5), -z(dist10), -z(rng_pos10)
```
(`retk` = k-bar return / ATR14; `distw` = (Close − SMAw)/ATR14; `rng_pos10` = position in the
10-bar high/low range.) Equal-weighted: the features are collinear and individually validated;
a fitted weight vector did not beat the equal blend out of sample and risks overfit.

### 2. Live alpha-health monitor (the part that makes it trustworthy)
The system measures its **own realized edge in real time** — the trailing 60-day mean of the
daily cross-sectional IC of the score vs realized forward return. When trailing IC is healthy
(> ~0.005, ~78% of days) forward IC is +0.036; when dormant it is +0.012 (noise). The monitor
scales a global **Conviction** multiplier in [0,1] so the system *stands down when its edge is
not working* (e.g. 2026). This is surfaced honestly in the UI, never hidden.

### 3. Regime / risk context (per name + universe)
Retained from the old regime engine (it is order-flow-agnostic and useful): HMM bull/bear,
GARCH vol-regime, CUSUM change-points. Used to (a) condition conviction (reversion is best in
HIGH vol, damped in EXTREME), and (b) provide per-name risk context. Vol-regime also drives
position-risk scaling in the displayed conviction.

### 4. Order-flow & profile (UI context, not score)
Inferred delta, CVD, POC/value-area, absorption — kept as **descriptive columns and chart
context** so the trader sees flow/structure, but they do **not** enter the ranking score.

## Outputs (per name)
- `Rev_Score` — raw cross-sectional reversion score (signed; + = long-attractive)
- `Rev_Rank_Pct` — within-date percentile
- `Conviction` — [0,1], = rank strength × alpha-health × regime factor (the headline number)
- `Side` — Long / Short / — (top/bottom tail with sufficient conviction)
- Risk context: `Vol_Regime`, `Regime_Confidence`, `Change_Point`, `ATR%`
- Flow context (descriptive): `Bar_Delta`, `CVD`, `VA_Pos`, absorption flags

## What "good" means / how we keep it honest
- Ranking quality = cross-sectional IC and top-vs-bottom spread, **after costs**.
- The alpha-health monitor is shown, not buried: when the edge is dormant the dashboard says
  so and conviction shrinks. A flat tape produces a flat, low-conviction screen — by design.
