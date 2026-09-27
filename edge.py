"""
edge.py — measured out-of-sample expectancy for the Pragati signal set, per universe.

Why this module exists
----------------------
The signal set (``engine.py``: the ▲▼ read from the grid and ◆ RESUME, pragati.pine v9) is a fixed,
pre-declared rule. The question this module answers is separate and empirical: **does
that rule carry an edge on the universe actually on screen, and can we prove it from data
the app can fetch?** The v9 audit measured the set across 380 instruments (the ▲ about
+0.05σ over 10-20 bars in every era outside crypto), but a number from other symbols is
not a number for yours. This measures it, on your symbols.

What is measured
----------------
Six slices, each with the full method below:

    buy / sell                   every long event (▲ + ◆ RESUME ↑) / every short one
    turn_buy / turn_sell         ▲ capitulation / ▼ distribution alone (v9's ▲▼)
    resume_long / resume_short   ◆ RESUME alone — continuation

The pooled sides are what the screen's two sides are; the per-kind slices say which of
the two situations is carrying (or costing) the pooled number. The ▲▼ are rare by
construction — a capitulation turning, sellers taking a rich price — so their slices will
often read UNDERPOWERED, and the app says so rather than quoting a number it cannot resolve.

Method (each step exists to kill a specific way of fooling yourself)
-------------------------------------------------------------------
1. **Event study at the declared horizon.** Enter at the bar AFTER the signal bar closes,
   exit ``horizon`` bars later (EXEC-B).

2. **Drift removal, CAUSAL.** Subtract each symbol's own mean h-bar forward return over
   the 500 returns fully REALISED before the event (ending h + 1 bars earlier). Without
   drift removal every long signal on an equity universe in a bull market prints a profit
   and you have measured beta, not edge. (Up to v8.4 the mean was taken inside the era being
   measured — a look-ahead the v9 audit showed flatters reversal signals: on random walks
   it alone scores a persistent momentum reading as reversion, and events that cluster in
   names whose era went badly look better than they were.)

3. **Volatility normalisation.** Divide by the symbol's own forward-return sigma over the
   same trailing window, so an FX pair, a bond ETF and a small-cap equity land on one
   scale — and the cost charge means the same thing on each.

4. **Sign folding.** A long event scores positive when the return beat the symbol's drift;
   a short event scores positive when it fell short.

5. **Block bootstrap over DATES.** h-bar forward returns overlap, and every symbol on one
   date shares the market factor. Resampling contiguous blocks of whole dates handles
   both. The confidence interval, not a p-value, decides whether an edge is claimed.

6. **Costs charged in the same units.** ``cost_bps / 1e4 / sigma_h``.

7. **Power stated, never assumed.** ``n_eff = (n_dates / horizon) x participation_ratio``,
   and a minimum detectable effect from it. When the MDE is larger than the biggest effect
   this family of indicators has ever shown anywhere, the test is vacuous and says so.

The events come from ``engine.compute_frame`` — the exact call the screener makes — so the
study can never measure a rule the screener does not fire. The macro drivers behind the
value ingredient are fetched once for the whole history.

What this module deliberately does NOT do
-----------------------------------------
* **It does not tune the signal.** Every indicator input and the horizon stay pre-declared.
  The source measured fitted-vs-unseen edge at approximately zero correlation across 900
  configurations; searching here would fit noise.
* **It does not gate the signal.** The measurement is reported, not applied. Ranking is
  the grid state and the event bands; a universe that measures no edge still fires, and
  says so.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

import engine as eng

# ── Bootstrap / power configuration ──────────────────────────────────────────────────────
N_BOOTSTRAP = 2000     # percentile CI resamples. Vectorised, so this is milliseconds.
CI_LEVEL = 0.95

# The largest edge this indicator family has found on any instrument group. Defined in
# engine.py (it is a claim about the signal) and aliased here: if our minimum detectable effect
# exceeds it, the test cannot resolve even the best case ever observed for this family — so it
# is vacuous, and reporting "no edge" would be an unsupported claim rather than a finding.
LARGEST_KNOWN_EFFECT = eng.LARGEST_KNOWN_EFFECT

# Minimum events per side before a point estimate is worth printing at all.
MIN_EVENTS = 30

# The slices measured: key -> (side, kind or None for both kinds).
SLICES = {
    "buy":          (1.0, None),
    "sell":         (-1.0, None),
    "turn_buy":     (1.0, "turn"),
    "turn_sell":    (-1.0, "turn"),
    "resume_long":  (1.0, "resume"),
    "resume_short": (-1.0, "resume"),
}
SLICE_LABEL = {"buy": "Long · all", "sell": "Short · all", "turn_buy": "▲ capitulation",
               "turn_sell": "▼ distribution", "resume_long": "◆ RESUME ↑", "resume_short": "◆ RESUME ↓"}


# ════════════════════════════════════════════════════════════════════════════════════════
# EVENT EXTRACTION  (per symbol; the caller streams symbols so nothing accumulates)
# ════════════════════════════════════════════════════════════════════════════════════════
def symbol_events(df: pd.DataFrame, drivers: pd.DataFrame | None, symbol: str,
                  settings: "eng.EngineSettings", daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """Extract the Pragati events for one symbol as a compact (date, side, kind, fwd) table.

    ``df`` is the chart's OHLCV (Title-case, ascending) — weekly bars on a weekly study,
    with ``daily`` the bars behind them. ``side`` is +1 long / -1 short, ``kind`` is
    'turn' (the ▲▼) or 'resume', ``fwd`` the raw h-bar forward return from the next bar.
    Drift removal and vol normalisation happen later, in :func:`measure`, causally.

    THE EVENTS COME FROM THE ENGINE ITSELF (``engine.compute_frame``), so every guard
    the screener applies — warm-up, the stack gate, the basket-settling gate, cooldowns —
    applies to the study by construction. Still lean: no volume profile, no regime
    engine, no order flow.
    """
    empty = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"),
                          "side": pd.Series(dtype=float), "kind": pd.Series(dtype=object),
                          "fwd": pd.Series(dtype=float)})
    horizon = int(settings.horizon)
    if df is None or len(df) < settings.min_bars + horizon + 3:
        return empty
    f = eng.compute_frame(df, drivers, symbol, settings, daily)
    close = pd.to_numeric(df["Close"], errors="coerce").reindex(f.index)

    # EXEC-B: the signal bar closes, we enter on the NEXT bar and hold `horizon` bars.
    # Next-bar close is the open proxy (the frames are OHLC; entering at the signal close
    # vs the next open tests barely different).
    entry = close.shift(-1)
    exit_ = close.shift(-1 - horizon)
    fwd = (exit_ / entry - 1.0).to_numpy()

    parts = []
    for col, side, kind in (("turn_buy", 1.0, "turn"), ("turn_sell", -1.0, "turn"),
                            ("resume_long", 1.0, "resume"), ("resume_short", -1.0, "resume")):
        m = f[col].fillna(False).astype(bool).to_numpy() & np.isfinite(fwd)
        if m.any():
            parts.append(pd.DataFrame({"date": pd.to_datetime(f.index[m]), "side": side,
                                       "kind": kind, "fwd": fwd[m]}))
    if not parts:
        return empty
    return pd.concat(parts, ignore_index=True).sort_values("date", kind="stable")


def symbol_baseline(close: pd.Series, horizon: int) -> pd.Series:
    """Per-bar h-bar forward return for one symbol — the drift/vol baseline.

    Returned as a dated series (dated by its entry bar) so :func:`measure` can take the
    trailing mean and sigma of the returns already realised at each event.
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
    """One slice (a side, or a side × kind) measured in one era."""
    side: str                 # a key of SLICES
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
    iclass: str                       # display label for the kind of universe
    length: int                       # conviction lookback
    trigger: str                      # the signal set measured, in prose
    horizon: int
    cost_bps: float
    # Coverage
    n_symbols_universe: int
    n_symbols_studied: int
    n_bars_median: int
    start: str
    end: str
    part_ratio: float
    fire_rate: float                  # fraction of usable bars that fired any event
    split_date: str
    # Results, keyed by SLICES -> era -> SideResult (as dicts for cache round-tripping)
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
                    f"MDE {ref.mde:.3f} vs the largest effect this indicator family has "
                    f"shown anywhere ({LARGEST_KNOWN_EFFECT:.3f}) · {ref.n_events} events, "
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

    def counts(self) -> dict:
        """Events per slice over the full history — how often each situation fires."""
        out = {}
        for key in SLICES:
            r = self.get(key, "full")
            out[key] = r.n_events if r is not None else 0
        return out


# ════════════════════════════════════════════════════════════════════════════════════════
# THE MEASUREMENT
# ════════════════════════════════════════════════════════════════════════════════════════
BASE_WINDOW = 500      # realised h-bar returns behind each event's drift and sigma
BASE_MIN = 100         # fewer than this and the symbol's event is not scored


def _causal_stats(b: pd.Series, horizon: int) -> tuple:
    """Trailing mean and sigma of a symbol's h-bar forward returns, as known at each date.

    A forward return dated t (entered t+1, exited t+1+h) is realised h + 1 bars later, so
    the statistics at t use only returns dated t − h − 1 and earlier. Under the null — a
    return independent of its past — the score is then exactly unbiased."""
    r = b.rolling(BASE_WINDOW, min_periods=BASE_MIN)
    return r.mean().shift(horizon + 1), r.std(ddof=1).shift(horizon + 1)


def _score_events(ev: pd.DataFrame, baselines: dict, symbols: np.ndarray,
                  lo: pd.Timestamp, hi: pd.Timestamp, horizon: int = 1) -> pd.DataFrame:
    """Drift-remove and vol-normalise the events dated inside ``[lo, hi]``. Returns scored events.

    ``baselines[sym]`` is that symbol's dated forward-return series. Each event's drift and
    sigma are the symbol's trailing ones as known at the event (:func:`_causal_stats`) —
    nothing from the event's future, inside or outside the era. Events whose sigma is
    undefined or zero are dropped rather than divided by, which would manufacture infinite
    scores on a flat instrument.
    """
    m = (ev["date"] >= lo) & (ev["date"] <= hi)
    sub = ev.loc[m]
    if sub.empty:
        return sub.assign(score=pd.Series(dtype=float), cost=pd.Series(dtype=float))

    mu = np.full(len(sub), np.nan)
    sg = np.full(len(sub), np.nan)
    syms = sub["symbol"].to_numpy()
    dts = pd.DatetimeIndex(sub["date"])
    for sym in np.unique(syms):
        b = baselines.get(sym)
        if b is None or b.empty:
            continue
        mc, sc = _causal_stats(b.sort_index(), int(horizon))
        k = syms == sym
        mu[k] = mc.reindex(dts[k], method="ffill").to_numpy(dtype=float)
        sg[k] = sc.reindex(dts[k], method="ffill").to_numpy(dtype=float)

    keep = np.isfinite(mu) & np.isfinite(sg) & (sg > 0)
    if not keep.any():
        return sub.iloc[0:0].assign(score=pd.Series(dtype=float), cost=pd.Series(dtype=float))
    sub = sub.loc[keep].copy()
    mu, sg = mu[keep], sg[keep]
    # Sign folding: positive score == the signal was right, for BOTH sides.
    sub["score"] = sub["side"].to_numpy(dtype=float) * (sub["fwd"].to_numpy(dtype=float) - mu) / sg
    # Cost in the same vol units as the score: a round trip of `cost_bps` against this
    # symbol's own h-bar sigma. Stored per event so the charge reflects the actual mix of
    # instruments that fired, not a universe average.
    sub["_sigma"] = sg
    return sub


def _measure_side(sub: pd.DataFrame, side_key: str, era: str, horizon: int,
                  part_ratio: float, cost_bps: float) -> SideResult | None:
    """Bootstrap one (slice, era) into a SideResult."""
    want, kind = SLICES[side_key]
    m = sub["side"] == want
    if kind is not None and "kind" in sub.columns:
        m &= sub["kind"] == kind
    s = sub.loc[m]
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
            length: int, trigger: str, horizon: int, cost_bps: float,
            n_symbols_universe: int, n_bars_median: int,
            holdout_frac: float = 0.40, partial: bool = False,
            measured_at: str = "") -> EdgeStudy:
    """Turn streamed events into an :class:`EdgeStudy`.

    ``events``    long frame of (symbol, date, side, kind, fwd) from :func:`symbol_events`.
    ``baselines`` {symbol: dated h-bar forward-return series} from :func:`symbol_baseline`.
    ``ret_matrix`` wide daily-return frame used only to measure the participation ratio.

    The discovery/holdout split is by DATE (never by symbol), with the most recent
    ``holdout_frac`` sealed off. One split, reported as one split — this is not a
    walk-forward and does not pretend to be.
    """
    if events is None or events.empty:
        return EdgeStudy(
            universe=universe, selected_index=selected_index, timeframe=timeframe,
            iclass=iclass, length=int(length), trigger=str(trigger), horizon=int(horizon),
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
    results: dict = {k: {} for k in SLICES}
    for era, (a, b) in eras.items():
        scored = _score_events(ev, baselines, ev["symbol"].unique(), a, b, int(horizon))
        if scored.empty:
            continue
        for side_key in SLICES:
            r = _measure_side(scored, side_key, era, int(horizon), pr, float(cost_bps))
            if r is not None:
                results[side_key][era] = asdict(r)

    return EdgeStudy(
        universe=universe, selected_index=selected_index, timeframe=timeframe,
        iclass=iclass, length=int(length), trigger=str(trigger), horizon=int(horizon),
        cost_bps=float(cost_bps),
        n_symbols_universe=int(n_symbols_universe),
        n_symbols_studied=int(ev["symbol"].nunique()),
        n_bars_median=int(n_bars_median),
        start=str(lo_all.date()), end=str(hi_all.date()),
        part_ratio=float(pr), fire_rate=fire_rate, split_date=str(split.date()),
        results=results, measured_at=measured_at, partial=partial,
    )
