# Sanket — Engine Architecture & Research Basis

> This document records *why* the engine is built the way it is. Every claim below is
> **reproducible from this repo**: run `python research.py` to regenerate the evidence on live,
> corporate-action-adjusted data. Nothing here is decorative, and no number is quoted that the
> harness cannot reproduce.

## How we know anything: the research harness

The core discipline of this system is that **no signal is an edge until the harness says so.**
[`research.py`](research.py) is a point-in-time, cost-aware evaluation harness that:

- pulls **split/dividend-adjusted** OHLCV (so corporate actions can't manufacture fake signal),
- builds candidate cross-sectional signals under a strict **no-lookahead** contract (features at
  date *t* use only data ≤ *t*; forward returns are labels only),
- reports **rank-IC + t-stat, IC decay across horizons, cost-aware non-overlapping quantile
  backtests, turnover, per-year stability, and a shuffled-null control**.

Everything in this document was produced by that harness on **100 NIFTY-100 names, 2016–2026
(~2,600 bars)**. The shuffled-null returns IC ≈ 0 (t < 2), so the harness does not manufacture
edge — the positive results below are real, not artifacts.

## The thesis (what edge we exploit)

**12-1 cross-sectional momentum, long tilt.** Names with the strongest 12-month return (skipping
the most recent month) tend to out-perform peers over the following weeks. On the NSE large-cap
cross-section this is the edge that **survives realistic transaction costs** — the one property
that makes a signal tradeable rather than merely predictive.

### Evidence (reproducible)

| Check | Result |
|:---|:---|
| Momentum rank-IC vs forward return | **+0.025 (5d) → +0.032 (21d) → +0.048 (63d)** — *grows* with horizon |
| Long-only top-quintile, monthly, net of 15 bps/side | **~+6%/yr EXCESS** over the equal-weight universe |
| Net excess Sharpe · turnover | **~0.6 · ~21%** — cost-robust (still 0.55 at 25 bps) |
| Worst excess year (2016–2026) | **2018: −2%** |
| Shuffled-null control | IC ≈ 0 — harness is calibrated |

Momentum's IC **grows** with horizon, so a monthly book turns slowly (~21%) and the gross edge
clears costs. This is the mirror image of reversion (below), whose edge lives at 1–2 days where
turnover — and therefore cost — is highest.

### Why reversion is NOT the core (it was, and the harness demoted it)

Short-horizon cross-sectional reversion is a **real predictor** (rank-IC +0.029…+0.031 @1–2d,
**t ≈ 7**, positive every year 2018–2026). But it is a **cost trap**: at a 2-day rebalance the
long/short book runs ~80% turnover, so after realistic costs it is **net NEGATIVE** (≈ −23%/yr at
25 bps/side; gross +13% is entirely consumed). It predicts; you cannot harvest it. Reversion is
therefore **demoted to an entry-timing overlay** — among high-momentum names it favours those that
have pulled back (`Entry_Timing`), refining *when* to enter, never *what* to hold.

### Why the *old* engine was wrong about momentum

The prior engine deleted momentum for "anti-predicting" (IC −0.023). That was a **horizon error**:
it measured momentum at *short* horizons (≤10 days) where reversion dominates and momentum
genuinely loses. At the correct **12-1 monthly** horizon, momentum is the survivor and reversion
is the cost trap. The lesson is baked into the harness: always evaluate a signal at the horizon its
turnover can afford.

## Honest limits (disclosed, not hidden)

- **Mostly beta, not a money-printer.** The top quintile's ~30%/yr absolute return is largely
  market beta — its absolute Sharpe (1.34) barely beats the equal-weight benchmark's (1.29). The
  genuine skill is the **~+6% excess** (excess Sharpe ~0.6). This engine reports the *tilt*; it
  never quotes the 30%.
- **Momentum decays.** Momentum IC went **negative in 2024–2026** (momentum crashes cluster at
  high-vol turning points). This is expected and handled by the alpha-health monitor, which stands
  the book down when the factor is off — the mechanism validated by the calm-vs-turbulent IC split
  (+0.056 vs +0.007).
- **Survivorship.** The evidence uses *current* index constituents over history, which inflates
  momentum (winners that stayed in the index). The true point-in-time number is lower; killing this
  bias needs historical index membership (a data-sourcing task).

## Cost reality (why this is a ranker, not an HFT book)

Even the winning edge is a **weekly-to-monthly, long-tilt decision-support ranker**, not a costless
high-frequency strategy. It surfaces a daily ranked shortlist with conviction and risk context for
a human who holds for weeks and amortizes cost. On NSE cash, single-name shorting is impractical, so
the bottom-momentum tail is surfaced as **underweight/avoid** (executable only in F&O), not as an
actionable short.

## The engine (what we build) — [`engine.py`](engine.py)

### 1. Momentum score (cross-sectional, per date)
Within-date **robust z-score** (median / MAD) of 12-1 momentum, oriented so higher = more
attractive long. 6-1 momentum is a coverage fallback for names with short histories:
```
mom      = Close[t-21] / Close[t-252] − 1        (12-month return, skip last month)
score    = robust_z_within_date(mom)             (fallback to 6-1 where 12-1 is NaN)
```
No fitted weights: a risk-adjusted (mom/vol) variant and a 12+6 blend did **not** beat plain 12-1
on excess return out of sample.

### 2. Reversion entry overlay (timing, not thesis)
`Entry_Timing = within-date rank of −z(ret2)` in [0,1] — high = the name has pulled back. It nudges
conviction (side-aware, kept small, ±10%) so a momentum long is preferred on a dip. It never enters
the rank.

### 3. Live alpha-health monitor (what makes it trustworthy)
The system measures its **own realized edge in real time**: the trailing ~60-day mean of the daily
cross-sectional IC of the momentum score vs a 5-day forward return. It maps that reading to a global
**Conviction** multiplier in `[0.35, 1]`, so the system *stands down when its edge is off* (as it is
in the 2024–2026 momentum dormancy). Surfaced honestly in the UI, never hidden. The significance
haircut accounts for the forward-return overlap; no p-value is claimed from it.

### 4. Regime / risk context (per name + universe)
HMM bull/bear, GARCH vol-regime, CUSUM change-points (order-flow-agnostic). `VOL_REGIME_MOM` is now
**data-calibrated to near-neutral** (`{LOW 1.0, NORMAL 1.0, HIGH 1.0, EXTREME 0.85}`): a study of
momentum-top-tercile forward returns by the name's own vol regime showed high-vol names return
*more*, not less — that is **beta, not edge**, and per-name vol does not predict the cross-sectional
momentum edge, so this weight barely tilts conviction (only a mild EXTREME trim for blow-up risk).
The real edge-timing is **market-wide** (momentum IC +0.056 on calm dates vs +0.007 turbulent) and
is owned by the alpha-health monitor, not this per-name weight. HMM/CUSUM are per-name risk *context*
and never enter the rank.

### 5. Entry screeners — Set A / Set B (long-only)
Two live, same-bar entry timers surfaced alongside the rank (never *in* it), each validated on an
out-of-sample condition sweep as a better-than-baseline entry on already-trending names:
- **Set A · Momentum Pullback-Resumption** — uptrend (Close>SMA200, 12-1 mom>10%) dips below SMA20,
  closes back above. +0.25% vs universe @5d, both OOS halves positive.
- **Set B · Gap-and-Go Continuation** — uptrend gaps up ≥1.5%, holds it (Close>Open), finishes near
  its 20-day high. **+0.99% vs universe @5d, t~2.2, 9/11 years** — strongest signal, orthogonal to A.

Both long-only (the short side of every tested event anti-predicted). They time *when* to enter, not
*what* to hold; not standalone portfolio alpha. Chosen from ~130 candidate conditions across two
sweeps — momentum-ignition beat accumulation, and every inferred-delta condition failed.

### 6. Order-flow & profile (UI context, not score)
Inferred delta, CVD, POC/value-area, absorption (`Buy_Share`, `Absorption_Score`) — descriptive
columns and chart context only. Validated (three times) to add **no** ranking or entry edge; never
in the score.

## Outputs (per name)
- `Rev_Score` — the cross-sectional **momentum** alpha score (retained column name; + = long-attractive)
- `Rev_Rank_Pct` — within-date percentile
- `Conviction` — `[0,1]` = tail strength × alpha-health × regime × confidence × entry nudge
- `Side` — Long / Short (underweight, F&O-only) / — (context)
- `Entry_Timing` — `[0,1]` pullback score for entry timing
- Risk context: `Vol_Regime`, `Regime_Confidence`, `Change_Point`, `ATR_Pct`
- Flow context (descriptive): `Bar_Delta`, `CVD`, `Buy_Share`, `Absorption_Score`, `VA_Pos`

## What "good" means / how we keep it honest
- Ranking quality = cross-sectional IC and top-vs-bottom spread, **after costs**, reproducible via
  `python research.py`.
- Alpha (excess) is reported separately from beta; the 30% absolute is never quoted as skill.
- The alpha-health monitor is shown, not buried: when momentum is dormant the dashboard says so and
  conviction shrinks toward the floor — by design.

## Known limitations (disclosed, not hidden)
- **Survivorship** in both the backtest and the alpha-health harvest (today's constituents applied
  over the lookback). Read results as "the edge on names we can trade today," not survivorship-free.
- **Momentum-crash risk.** The edge is time-varying and can invert sharply; the health monitor
  de-rates but does not eliminate this.
- **`VOL_REGIME_MOM` is calibrated but coarse** — set near-neutral from the vol-regime return study
  (per-name vol ≈ beta, not edge). It intentionally does little; the alpha-health monitor carries the
  real regime-timing. The mild EXTREME trim is risk control, not a fitted alpha weight.
- **The Set A/B entry edges are modest and screener-level** — ~+0.25–0.99% per signal, validated as
  entry odds, *not* portfolio alpha (overweighting a book that already holds the names doesn't help).
- **Docs vs. reproducibility:** trust `research.py` over prose. If a number here and the harness
  disagree, the harness is right and this document is stale.
