"""
edge.py — measured out-of-sample expectancy for the Siddhi signal, per universe.

Why this module exists
----------------------
The signal (``engine.py``) is a fixed, pre-declared rule. The question this module answers
is separate and empirical: **does that rule carry an edge on the universe actually on
screen, and can we prove it from data the app can fetch?**

Before this existed, the answer was a hardcoded lookup of eight numbers copied from the
source study's 39 instruments. That is indefensible for three reasons: it cannot cover a
universe the study never touched (NSE F&O single names, NSE thematic ETFs, crypto), it
makes an asset-class claim and applies it to instrument-level decisions, and it cannot
report that the edge has stopped working — while the source study's own headline is that
the edge decayed 4x since the 1990s.

So: measure it. On your symbols, at the pre-declared parameters, with the same methodology
that makes the source study's numbers credible in the first place.

Method (each step exists to kill a specific way of fooling yourself)
-------------------------------------------------------------------
1. **Event study at the declared horizon.** Enter at the bar AFTER the signal bar closes,
   exit ``horizon`` bars later (the source study's EXEC-B convention). Not a continuous
   IC: a continuous position on this signal turns over daily and nets -0.48 Sharpe, so
   measuring the continuous form would answer a question nobody trades.

2. **Drift removal, within era.** Subtract each symbol's own mean forward return, computed
   inside the era being measured. Without this, every long signal on an equity universe in
   a bull market prints a profit and you have measured beta, not edge. Computing the mean
   *within* era also stops the discovery period's drift leaking into the holdout.

3. **Volatility normalisation.** Divide by the symbol's own forward-return sigma, so the
   result is in vol units and an FX pair, a bond ETF and a small-cap equity are on one
   scale. This is also what makes the cost charge meaningful (step 6).

4. **Sign folding.** A buy event scores positive when the return beat the symbol's drift;
   a sell event scores positive when it fell short. Both sides are then "positive = the
   signal was right", which is how the source study reports fade-long and fade-short.

5. **Block bootstrap over DATES.** Two dependencies would otherwise inflate significance:
   an h-bar forward return overlaps its neighbours, and every symbol on one date shares
   the market factor. Resampling contiguous *blocks of whole dates* handles both at once —
   blocks for the serial overlap, whole dates for the cross-sectional correlation. The
   confidence interval, not a p-value, decides whether an edge is claimed.

6. **Costs charged in the same units.** ``cost_bps / 1e4 / sigma_h``. This is why the edge
   dies on low-volatility instruments: 3bp against a 4% 10-day sigma is 0.008 vol units,
   but against a 1% sigma it is 0.030 — a real drag on a small edge. A hardcoded class
   table cannot express that; this does.

7. **Power stated, never assumed.** Effective sample size is
   ``(n_dates / horizon) x participation_ratio``, where the participation ratio is the
   eigenvalue-based count of genuinely independent names in the cross-section — measured,
   not guessed. From it comes a minimum detectable effect. When the MDE is larger than the
   biggest effect the source study ever found, the test is vacuous and says so instead of
   reporting a verdict. This is the source study's own opening lesson: 370,686 intraday
   bars were only ~601 independent observations.

What this module deliberately does NOT do
-----------------------------------------
* **It does not tune the signal.** Every oscillator parameter and the horizon stay
  pre-declared. With a few hundred independent blocks, searching for the best lookback or
  magnitude gate per universe would fit noise and destroy the very credibility this module
  exists to establish — the source indicator measured that correlation between fitted and
  out-of-sample edge at approximately zero, and said so. It measures the expectancy of a
  fixed rule; it does not search for a better rule.
* **It does not gate the signal.** The measurement is reported, not applied. Conviction in
  ``engine.compute_ranking`` derives from the crossing impulse and the cost gate only. A
  universe that measures no edge still fires its signals at full conviction — the number is
  information for the person reading the screen, not a hidden multiplier.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

import engine as eng

# ── Bootstrap / power configuration ──────────────────────────────────────────────────────
N_BOOTSTRAP = 2000     # percentile CI resamples. Vectorised, so this is milliseconds.
CI_LEVEL = 0.95

# The largest edge the source indicator found on any instrument group. Defined in
# engine.py (it is a claim about the signal) and aliased here: if our minimum detectable effect
# exceeds it, the test cannot resolve even the best case ever observed for this signal — so it
# is vacuous, and reporting "no edge" would be an unsupported claim rather than a finding.
LARGEST_KNOWN_EFFECT = eng.LARGEST_KNOWN_EFFECT

# Minimum events per side before a point estimate is worth printing at all.
MIN_EVENTS = 30


# ════════════════════════════════════════════════════════════════════════════════════════
# EVENT EXTRACTION  (per symbol; the caller streams symbols so nothing accumulates)
# ════════════════════════════════════════════════════════════════════════════════════════
def symbol_events(close: pd.Series, high: pd.Series, low: pd.Series,
                  volume: pd.Series | None = None, *,
                  length: int = eng.SID_LENGTH, k: float = eng.SID_K,
                  horizon: int = eng.SID_HORIZON,
                  smooth: int = eng.SID_SMOOTH, signal: int = eng.SID_SIGNAL,
                  norm: int = eng.SID_NORM, vol_n: int = eng.SID_VOL_N,
                  cap: float = eng.SID_CAP,
                  participation: str = eng.SID_PARTICIPATION) -> pd.DataFrame:
    """Extract Siddhi crossing events for one symbol as a compact (date, side, fwd) table.

    Returns a frame with columns ``date``, ``side`` (+1 buy / -1 sell) and ``fwd`` (the
    raw h-bar forward return from the next bar's open-proxy). Drift removal and vol
    normalisation happen later, in :func:`measure`, because they must be computed *within
    era* — doing them here would leak across the discovery/holdout boundary.

    THE EVENTS COME FROM THE ENGINE ITSELF. The previous version of this function
    re-derived the signal inline, which meant every guard added to the engine had to be
    mirrored here or the study would silently measure a rule the screener does not fire.
    :func:`engine.siddhi_oscillator` is vectorised and touches only OHLCV, so calling it
    costs nothing and removes that whole class of drift by construction.

    Still deliberately lean: it does NOT compute the volume profile, the regime engine or
    the order-flow layer — the study does not need them, and on a shared vCPU those would
    dominate the runtime.
    """
    empty = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"),
                          "side": pd.Series(dtype=float),
                          "fwd": pd.Series(dtype=float)})
    n = len(close)
    warm = eng.warmup_bars(length, norm, vol_n, smooth, signal)
    if n < warm + int(horizon) + 3:
        return empty

    o = eng.siddhi_oscillator(high, low, close, volume,
                              length=length, smooth=smooth, signal=signal, norm=norm,
                              vol_n=vol_n, cap=cap, participation=participation)
    hist, hist_sd = o["hist"], o["hist_sd"]
    thr = float(k) * hist_sd.fillna(0.0)

    # ta.crossover / ta.crossunder against ±thr. At the shipped k = 0 this is exactly
    # "the histogram crossed zero", which is the screening condition.
    up = (hist > thr) & (hist.shift(1) <= thr.shift(1))
    dn = (hist < -thr) & (hist.shift(1) >= -thr.shift(1))

    ready = pd.Series(np.arange(n) >= warm, index=close.index)
    healthy = hist.notna() & hist.shift(1).notna() & (hist_sd > 1e-12)

    # EXEC-B: the signal bar closes, we enter on the NEXT bar and hold `horizon` bars.
    # Using next-bar close as the open proxy (the app's frames are OHLC; entering at the
    # signal close vs the next open tests barely different).
    entry = close.shift(-1)
    exit_ = close.shift(-1 - int(horizon))
    fwd = exit_ / entry - 1.0

    fires = (up | dn) & ready & healthy & fwd.notna()
    if not fires.any():
        return empty

    side = np.where(up[fires], 1.0, -1.0)   # cross up -> buy, cross down -> sell
    return pd.DataFrame({
        "date": pd.to_datetime(close.index[fires]),
        "side": side,
        "fwd": fwd[fires].to_numpy(dtype=float),
    })


def symbol_baseline(close: pd.Series, horizon: int) -> pd.Series:
    """Per-bar h-bar forward return for one symbol — the drift/vol baseline.

    Returned as a dated series so :func:`measure` can slice it per era and compute the
    mean and sigma *inside* each era, which is what keeps the holdout clean.
    """
    entry = close.shift(-1)
    exit_ = close.shift(-1 - int(horizon))
    fwd = exit_ / entry - 1.0
    fwd.index = pd.to_datetime(fwd.index)
    return fwd.dropna()


# ════════════════════════════════════════════════════════════════════════════════════════
# POWER  (how many genuinely independent observations do we have?)
# ════════════════════════════════════════════════════════════════════════════════════════
def participation_ratio(returns: pd.DataFrame) -> float:
    """Effective number of independent names in a cross-section.

    ``PR = (sum lambda)^2 / sum lambda^2`` over the eigenvalues of the correlation matrix
    — the standard "effective number of bets". For a correlation matrix ``sum lambda = N``,
    so this reduces to ``N^2 / sum lambda^2``: it equals N for a perfectly uncorrelated
    set and collapses toward 1 as everything moves together.

    This is why a 500-name NSE universe does not carry 500 observations per date. The
    source study makes the same point: 26 symbols carried a participation ratio of 7.2.
    """
    if returns is None or returns.shape[1] < 2:
        return float(max(returns.shape[1], 1)) if returns is not None else 1.0
    r = returns.dropna(axis=1, how="all")
    # Need enough overlapping rows for a stable correlation matrix.
    r = r.loc[:, r.notna().sum() >= 30]
    if r.shape[1] < 2:
        return float(max(r.shape[1], 1))
    c = r.corr(min_periods=30).to_numpy(dtype=float)
    c = np.nan_to_num(c, nan=0.0)
    np.fill_diagonal(c, 1.0)
    try:
        lam = np.linalg.eigvalsh(c)
    except np.linalg.LinAlgError:
        return float(c.shape[0])
    lam = np.clip(lam, 0.0, None)
    denom = float((lam ** 2).sum())
    if denom <= 0:
        return float(c.shape[0])
    pr = float(lam.sum() ** 2 / denom)
    return float(np.clip(pr, 1.0, c.shape[0]))


def effective_n(n_dates: int, horizon: int, part_ratio: float) -> float:
    """Independent observations: date blocks (serial overlap) x independent names."""
    blocks = max(float(n_dates) / max(int(horizon), 1), 1.0)
    return float(max(blocks * max(part_ratio, 1.0), 1.0))


def min_detectable_effect(n_eff: float, sigma: float = 1.0) -> float:
    """Smallest effect a two-sided 95% interval could separate from zero at this power.

    Scores are vol-normalised so sigma ~ 1; the half-width is ``1.96 * sigma / sqrt(n_eff)``.
    Reported always, so the reader can see what the test was *capable* of resolving rather
    than having to infer it from a CI.
    """
    return float(1.96 * float(sigma) / np.sqrt(max(n_eff, 1.0)))


# ════════════════════════════════════════════════════════════════════════════════════════
# BLOCK BOOTSTRAP  (dates in contiguous blocks — the only correct unit here)
# ════════════════════════════════════════════════════════════════════════════════════════
def block_bootstrap_ci(scores: np.ndarray, date_codes: np.ndarray, n_dates: int,
                       block: int, n_boot: int = N_BOOTSTRAP,
                       level: float = CI_LEVEL, seed: int = 12345) -> tuple:
    """Percentile CI for the mean score, resampling contiguous blocks of whole dates.

    ``date_codes`` maps each score to a dense date index in [0, n_dates). Resampling whole
    dates preserves the within-date cross-sectional correlation; resampling them in blocks
    of ``block`` consecutive dates preserves the h-bar forward-return overlap.

    Vectorised via per-date sums: a bootstrap draw is a gather-and-add over ``n_blocks``
    precomputed block totals, not a re-scan of every event. That keeps 2000 resamples in
    the millisecond range, which is what makes this affordable on a shared vCPU.

    Returns ``(lo, hi)``, or ``(nan, nan)`` when the sample is too thin to resample.
    """
    if scores.size == 0 or n_dates < 2:
        return (float("nan"), float("nan"))
    block = max(int(block), 1)

    per_date_sum = np.bincount(date_codes, weights=scores, minlength=n_dates)
    per_date_cnt = np.bincount(date_codes, minlength=n_dates).astype(float)

    # Fewer than two whole blocks means there is nothing to resample — bail BEFORE the
    # reshape below, which would otherwise raise. `n_dates // block` is 0 whenever the
    # slice spans fewer dates than the horizon (a thin side, a short era, a sparse
    # bucket); the old `max(..., 1)` turned that into a claim of one block and then tried
    # to reshape n_dates values into a (1, block) frame. Reaching this with n_dates=5 and
    # block=10 raised ValueError and took the whole Edge Study down with it.
    n_blocks = n_dates // block
    if n_blocks < 2:
        return (float("nan"), float("nan"))
    trim = n_blocks * block
    bs = per_date_sum[:trim].reshape(n_blocks, block).sum(axis=1)
    bc = per_date_cnt[:trim].reshape(n_blocks, block).sum(axis=1)
    # Any dates past the last whole block are dropped from the resample rather than
    # forming a short block with different variance.
    if bc.sum() <= 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_blocks, size=(int(n_boot), n_blocks))
    tot = bs[idx].sum(axis=1)
    cnt = bc[idx].sum(axis=1)
    ok = cnt > 0
    if not ok.any():
        return (float("nan"), float("nan"))
    means = tot[ok] / cnt[ok]
    a = (1.0 - level) / 2.0 * 100.0
    return (float(np.percentile(means, a)), float(np.percentile(means, 100.0 - a)))


# ════════════════════════════════════════════════════════════════════════════════════════
# RESULT TYPES
# ════════════════════════════════════════════════════════════════════════════════════════
@dataclass
class SideResult:
    """One side (buy or sell) measured in one era."""
    side: str                 # 'buy' | 'sell'
    era: str                  # 'discovery' | 'holdout' | 'full'
    n_events: int
    n_dates: int
    n_eff: float
    edge: float               # mean drift-free, vol-normalised score (GROSS)
    ci_lo: float
    ci_hi: float
    hit: float                # % of events where the signal was right vs the symbol's drift
    net: float                # edge minus the cost charge, same units
    cost_charge: float
    mde: float                # smallest effect this power could resolve

    @property
    def significant(self) -> bool:
        """CI excludes zero on the positive side."""
        return bool(np.isfinite(self.ci_lo) and self.ci_lo > 0.0)

    @property
    def anti(self) -> bool:
        """CI excludes zero on the NEGATIVE side — the signal predicts backwards here."""
        return bool(np.isfinite(self.ci_hi) and self.ci_hi < 0.0)

    @property
    def underpowered(self) -> bool:
        return bool(self.n_events < MIN_EVENTS or self.mde > LARGEST_KNOWN_EFFECT)


# Verdict ladder. Order matters: the first matching rung wins.
VERDICTS = {
    "CONFIRMED":        ("success", "holdout CI excludes zero and survives costs"),
    "GROSS ONLY":       ("warning", "holdout edge is real but costs consume it"),
    "DISCOVERY ONLY":   ("warning", "discovery CI excludes zero, holdout does not"),
    "NO EDGE":          ("danger",  "CI straddles zero at adequate power"),
    "ANTI-PREDICTS":    ("danger",  "CI excludes zero on the wrong side"),
    "UNDERPOWERED":     ("neutral", "too few independent observations to claim either way"),
}


@dataclass
class EdgeStudy:
    """The full measurement for one (universe, timeframe, parameter) combination."""
    universe: str
    selected_index: str | None
    timeframe: str
    iclass: str                       # label only — used to show the source's prior
    length: int                       # oscillator lookback
    k: float                          # magnitude gate in σ of the histogram (0 = zero-cross)
    horizon: int
    cost_bps: float
    # Coverage
    n_symbols_universe: int
    n_symbols_studied: int
    n_bars_median: int
    start: str
    end: str
    part_ratio: float
    fire_rate: float                  # fraction of usable bars that fired either side
    split_date: str
    # Results, keyed 'buy'/'sell' -> era -> SideResult (as dicts for cache round-tripping)
    results: dict = field(default_factory=dict)
    measured_at: str = ""
    partial: bool = False             # some symbols failed to fetch
    note: str = ""

    # ── accessors ──
    def get(self, side: str, era: str) -> SideResult | None:
        d = (self.results.get(side) or {}).get(era)
        return SideResult(**d) if isinstance(d, dict) else d

    def verdict(self, side: str) -> tuple:
        """(label, css_kind, detail) for one side, from the measured intervals."""
        hold = self.get(side, "holdout")
        disc = self.get(side, "discovery")
        full = self.get(side, "full")

        present = [r for r in (hold, disc, full) if r is not None]
        if not present:
            return ("UNDERPOWERED", "neutral", "no events measured for this side")
        # Underpowered only if EVERY era we managed to measure is underpowered. Keying this
        # off `full` alone would report a confirmed holdout as UNDERPOWERED whenever the
        # full-era slice happened to drop out (e.g. too few in-era baseline bars).
        if all(r.underpowered for r in present):
            ref = max(present, key=lambda r: r.n_eff)
            return ("UNDERPOWERED", "neutral",
                    f"MDE {ref.mde:.3f} vs the largest effect ever measured for this "
                    f"signal ({LARGEST_KNOWN_EFFECT:.3f}) · {ref.n_events} events, "
                    f"n_eff {ref.n_eff:.0f}")
        if hold is not None and hold.significant:
            if hold.net > 0:
                return ("CONFIRMED", "success",
                        f"holdout {hold.edge:+.3f} [{hold.ci_lo:+.3f},{hold.ci_hi:+.3f}] "
                        f"· net {hold.net:+.3f} after {self.cost_bps:.1f}bp")
            return ("GROSS ONLY", "warning",
                    f"holdout {hold.edge:+.3f} gross but {hold.net:+.3f} net after "
                    f"{self.cost_bps:.1f}bp")
        if disc is not None and disc.significant:
            return ("DISCOVERY ONLY", "warning",
                    f"discovery {disc.edge:+.3f} [{disc.ci_lo:+.3f},{disc.ci_hi:+.3f}] "
                    f"· holdout " + (f"{hold.edge:+.3f} [{hold.ci_lo:+.3f},{hold.ci_hi:+.3f}]"
                                     if hold is not None else "n/a") + " did not confirm")
        # Report against the holdout when we have one — it is the era that counts — else
        # whichever era we did manage to measure. `present` is non-empty by the guard above,
        # so this cannot be None.
        ref = hold or full or present[0]
        if ref.anti:
            return ("ANTI-PREDICTS", "danger",
                    f"{ref.era} {ref.edge:+.3f} [{ref.ci_lo:+.3f},{ref.ci_hi:+.3f}] — "
                    f"the interval excludes zero on the wrong side")
        return ("NO EDGE", "danger",
                f"{ref.era} {ref.edge:+.3f} [{ref.ci_lo:+.3f},{ref.ci_hi:+.3f}] · "
                f"resolvable down to {ref.mde:.3f}")

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> "EdgeStudy":
        return EdgeStudy(**d)

    def prior(self) -> tuple:
        """The source study's published number for this instrument class — reference only.

        Shown beside the measurement so the two can be compared, never used to compute
        anything. Import is local to keep engine.py free of a dependency on this module.
        """
        import engine as eng
        return (eng.class_edge(self.iclass), eng.class_hit(self.iclass),
                eng.is_established(self.iclass))


# ════════════════════════════════════════════════════════════════════════════════════════
# THE MEASUREMENT
# ════════════════════════════════════════════════════════════════════════════════════════
def _score_events(ev: pd.DataFrame, baselines: dict, symbols: np.ndarray,
                  lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    """Drift-remove and vol-normalise events inside one era. Returns scored events.

    ``baselines[sym]`` is that symbol's dated forward-return series. The mean and sigma are
    taken over ``[lo, hi]`` only — the era being measured — so nothing leaks across the
    split. Symbols whose in-era sigma is undefined or zero are dropped rather than divided
    by, which would manufacture infinite scores on a flat instrument.
    """
    m = (ev["date"] >= lo) & (ev["date"] <= hi)
    sub = ev.loc[m]
    if sub.empty:
        return sub.assign(score=pd.Series(dtype=float), cost=pd.Series(dtype=float))

    stats = {}
    for sym in np.unique(sub["symbol"].to_numpy()):
        b = baselines.get(sym)
        if b is None or b.empty:
            continue
        w = b.loc[(b.index >= lo) & (b.index <= hi)]
        if len(w) < 30:
            continue
        sd = float(w.std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            continue
        stats[sym] = (float(w.mean()), sd)

    if not stats:
        return sub.iloc[0:0].assign(score=pd.Series(dtype=float), cost=pd.Series(dtype=float))

    keep = sub["symbol"].isin(stats.keys())
    sub = sub.loc[keep].copy()
    mu = sub["symbol"].map(lambda s: stats[s][0]).to_numpy(dtype=float)
    sg = sub["symbol"].map(lambda s: stats[s][1]).to_numpy(dtype=float)
    # Sign folding: positive score == the signal was right, for BOTH sides.
    sub["score"] = sub["side"].to_numpy(dtype=float) * (sub["fwd"].to_numpy(dtype=float) - mu) / sg
    # Cost in the same vol units as the score: a round trip of `cost_bps` against this
    # symbol's own h-bar sigma. Stored per event so the charge reflects the actual mix of
    # instruments that fired, not a universe average.
    sub["_sigma"] = sg
    return sub


def _measure_side(sub: pd.DataFrame, side_key: str, era: str, horizon: int,
                  part_ratio: float, cost_bps: float) -> SideResult | None:
    """Bootstrap one (side, era) slice into a SideResult."""
    want = 1.0 if side_key == "buy" else -1.0
    s = sub.loc[sub["side"] == want]
    if s.empty:
        return None
    scores = s["score"].to_numpy(dtype=float)
    ok = np.isfinite(scores)
    scores = scores[ok]
    if scores.size == 0:
        return None
    dates = pd.to_datetime(s["date"].to_numpy())[ok]
    uniq, codes = np.unique(dates, return_inverse=True)
    n_dates = int(uniq.size)

    n_eff = effective_n(n_dates, horizon, part_ratio)
    lo, hi = block_bootstrap_ci(scores, codes, n_dates, block=int(horizon))
    sig = s["_sigma"].to_numpy(dtype=float)[ok]
    sig = sig[np.isfinite(sig) & (sig > 0)]
    cost_charge = (float((cost_bps / 1e4) / np.mean(sig)) if sig.size else float("nan"))
    edge = float(scores.mean())
    return SideResult(
        side=side_key, era=era,
        n_events=int(scores.size), n_dates=n_dates, n_eff=n_eff,
        edge=edge, ci_lo=lo, ci_hi=hi,
        hit=float((scores > 0).mean() * 100.0),
        net=float(edge - cost_charge) if np.isfinite(cost_charge) else edge,
        cost_charge=cost_charge,
        mde=min_detectable_effect(n_eff, float(np.std(scores, ddof=1)) if scores.size > 1 else 1.0),
    )


def measure(events: pd.DataFrame, baselines: dict, ret_matrix: pd.DataFrame, *,
            universe: str, selected_index, timeframe: str, iclass: str,
            length: int, k: float, horizon: int, cost_bps: float,
            n_symbols_universe: int, n_bars_median: int,
            holdout_frac: float = 0.40, partial: bool = False,
            measured_at: str = "") -> EdgeStudy:
    """Turn streamed events into an :class:`EdgeStudy`.

    ``events``    long frame of (symbol, date, side, fwd) from :func:`symbol_events`.
    ``baselines`` {symbol: dated h-bar forward-return series} from :func:`symbol_baseline`.
    ``ret_matrix`` wide daily-return frame used only to measure the participation ratio.

    The discovery/holdout split is by DATE (never by symbol), with the most recent
    ``holdout_frac`` sealed off. One split, reported as one split — this is not a
    walk-forward and does not pretend to be.
    """
    if events is None or events.empty:
        return EdgeStudy(
            universe=universe, selected_index=selected_index, timeframe=timeframe,
            iclass=iclass, length=int(length), k=float(k), horizon=int(horizon),
            cost_bps=float(cost_bps), n_symbols_universe=int(n_symbols_universe),
            n_symbols_studied=0, n_bars_median=int(n_bars_median), start="", end="",
            part_ratio=1.0, fire_rate=0.0, split_date="", results={},
            measured_at=measured_at, partial=partial,
            note="no events fired in the studied history",
        )

    ev = events.copy()
    ev["date"] = pd.to_datetime(ev["date"])
    ev = ev.sort_values("date", kind="stable")
    dates = ev["date"]
    lo_all, hi_all = dates.min(), dates.max()

    # Split by date so both eras see the whole cross-section.
    uniq_dates = np.array(sorted(dates.unique()))
    cut_i = int(len(uniq_dates) * (1.0 - float(holdout_frac)))
    cut_i = int(np.clip(cut_i, 1, max(len(uniq_dates) - 1, 1)))
    split = pd.Timestamp(uniq_dates[cut_i])

    pr = participation_ratio(ret_matrix)

    # Total usable bars across studied symbols -> the fire rate the cost story hinges on.
    total_bars = sum(len(b) for b in baselines.values()) or 1
    fire_rate = float(len(ev) / total_bars)

    eras = {
        "full":      (lo_all, hi_all),
        "discovery": (lo_all, split - pd.Timedelta(days=1)),
        "holdout":   (split, hi_all),
    }
    results: dict = {"buy": {}, "sell": {}}
    for era, (a, b) in eras.items():
        scored = _score_events(ev, baselines, ev["symbol"].unique(), a, b)
        if scored.empty:
            continue
        for side_key in ("buy", "sell"):
            r = _measure_side(scored, side_key, era, int(horizon), pr, float(cost_bps))
            if r is not None:
                results[side_key][era] = asdict(r)

    return EdgeStudy(
        universe=universe, selected_index=selected_index, timeframe=timeframe,
        iclass=iclass, length=int(length), k=float(k), horizon=int(horizon),
        cost_bps=float(cost_bps),
        n_symbols_universe=int(n_symbols_universe),
        n_symbols_studied=int(ev["symbol"].nunique()),
        n_bars_median=int(n_bars_median),
        start=str(lo_all.date()), end=str(hi_all.date()),
        part_ratio=float(pr), fire_rate=fire_rate, split_date=str(split.date()),
        results=results, measured_at=measured_at, partial=partial,
    )
