"""
research.py — reproducible, point-in-time research harness for the Sanket system.

Why this module exists
----------------------
A system you would trade with real money must be able to REGENERATE its own evidence
from data on demand. The prior codebase asserted an edge (rank-IC +0.025…0.031, t≈8.6)
in a docstring but shipped no code to reproduce it. This module is the missing spine:
given a universe and a date range it pulls corporate-action-adjusted prices, builds
candidate cross-sectional signals under a strict no-lookahead contract, and reports the
only numbers that decide whether an edge is real —

    rank-IC and its t-stat · IC decay across horizons · cost-aware quantile spread ·
    turnover · PER-YEAR stability · a shuffled-null control.

Clean-slate stance
------------------
Nothing here is assumed to work. Reversion, momentum, short-term reversal and a random
null are tested side by side; the data ranks them. If reversion does not survive costs
out of sample, it does not go in the tradeable system — regardless of what any docstring
in this repo claims.

Point-in-time contract (enforced structurally)
---------------------------------------------
* Prices are ADJUSTED (Adj Close/Close factor applied to OHLC) so splits/dividends
  cannot manufacture fake reversion.
* A feature at date t uses ONLY rows with index <= t (causal rolling ops on each name's
  own series; cross-sectional standardization uses only that date's row).
* Forward returns r[t -> t+h] use ONLY data after t and appear solely as evaluation labels.
* The tradeable quantile backtest rebalances NON-overlapping at frequency h, so reported
  returns are independent across rebalances and costs are charged per actual turn.

Run:  python research.py            (uses the default liquid universe, ~7y)
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Universe ────────────────────────────────────────────────────────────────────────────
# Default: a liquid, large-cap NSE set. NOTE ON SURVIVORSHIP: this is a CURRENT-constituent
# list, so it carries mild survivorship bias (today's liquid names conditioned on surviving).
# A production system needs point-in-time index membership; this harness is honest about the
# gap rather than hiding it. Override by passing your own list to run_study().
DEFAULT_UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC", "SBIN",
    "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI",
    "HCLTECH", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "NESTLEIND", "POWERGRID",
    "NTPC", "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "ADANIPORTS", "GRASIM", "TECHM",
    "HDFCLIFE", "DIVISLAB", "DRREDDY", "CIPLA", "BAJAJFINSV", "BRITANNIA", "EICHERMOT",
    "HEROMOTOCO", "COALINDIA", "BPCL", "IOC", "ONGC", "HINDALCO", "SBILIFE", "M&M",
    "APOLLOHOSP", "INDUSINDBK", "TATACONSUM", "UPL", "BAJAJ-AUTO", "SHREECEM",
]

TRADING_DAYS = 252


# ── Data layer ──────────────────────────────────────────────────────────────────────────
def fetch_panel(tickers, start, end):
    """Return {ticker -> adjusted OHLCV DataFrame}. OHLC scaled by Adj Close/Close so every
    feature sees split/dividend-consistent prices. Silently drops names with no data."""
    import yfinance as yf

    yts = [t if t.endswith(".NS") else f"{t}.NS" for t in tickers]
    raw = yf.download(yts, start=start, end=end, interval="1d",
                      auto_adjust=False, progress=False, group_by="ticker", threads=True)

    panel = {}
    for t, yt in zip(tickers, yts):
        try:
            sub = raw[yt].dropna(how="all")
        except (KeyError, TypeError):
            continue
        if sub.empty or "Adj Close" not in sub or sub["Close"].dropna().empty:
            continue
        factor = (sub["Adj Close"] / sub["Close"]).replace([np.inf, -np.inf], np.nan)
        adj = pd.DataFrame({
            "Open":  sub["Open"] * factor,
            "High":  sub["High"] * factor,
            "Low":   sub["Low"] * factor,
            "Close": sub["Adj Close"],
            "Volume": sub["Volume"],
        }).dropna(how="any")
        if len(adj) > 60:
            panel[t] = adj
    return panel


def _wide(panel, field):
    """Stack one OHLCV field into a wide (dates x tickers) frame."""
    return pd.DataFrame({t: df[field] for t, df in panel.items()}).sort_index()


# ── Candidate signals (all point-in-time; each returns a wide dates x tickers score) ──────
def _atr(H, L, C, n=14):
    pc = C.shift(1)
    tr = np.maximum.reduce([(H - L).values, (H - pc).abs().values, (L - pc).abs().values])
    tr = pd.DataFrame(tr, index=C.index, columns=C.columns)
    return tr.rolling(n).mean()


def _xs_robust_z(row: pd.Series) -> pd.Series:
    med = row.median()
    mad = (row - med).abs().median()
    return ((row - med) / (1.4826 * mad + 1e-9)).clip(-4, 4)


def _xs_z_frame(feat: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional robust-z applied per date (row-wise). Causal: only uses that row."""
    return feat.apply(_xs_robust_z, axis=1)


def build_signals(panel) -> dict[str, pd.DataFrame]:
    """Construct competing cross-sectional signals. Higher score = stronger LONG conviction."""
    C, H, L = _wide(panel, "Close"), _wide(panel, "High"), _wide(panel, "Low")
    atr = _atr(H, L, C).clip(lower=1e-9)

    # raw per-name time-series features (all causal)
    ret2 = (C - C.shift(2)) / atr
    ret5 = (C - C.shift(5)) / atr
    dist5 = (C - C.rolling(5).mean()) / atr
    dist10 = (C - C.rolling(10).mean()) / atr
    rng_hi = H.rolling(10).max(); rng_lo = L.rolling(10).min()
    rngpos = (C - rng_lo) / (rng_hi - rng_lo).replace(0, np.nan)          # 0..1
    mom_63 = C.pct_change(63)                                             # 3-month momentum
    mom_126 = C.pct_change(126)                                           # 6-month momentum

    # cross-sectional standardization
    z = {k: _xs_z_frame(v) for k, v in dict(
        ret2=ret2, ret5=ret5, dist5=dist5, dist10=dist10,
        rngpos=rngpos, mom63=mom_63, mom126=mom_126).items()}

    # --- extra raw series for the low-turnover battery ---
    ret_1d = C.pct_change()
    vol60 = ret_1d.rolling(60).std()                                     # realized vol (persistent)
    mom_12_1 = C.shift(21) / C.shift(252) - 1.0                          # 12m return, skip last month
    mom_6_1 = C.shift(21) / C.shift(126) - 1.0                           # 6m return, skip last month
    dist21 = (C - C.rolling(21).mean()) / atr                            # longer-horizon overextension
    hi252 = C.rolling(252).max()                                         # 52-week-high proximity
    resid = ret_1d.sub(ret_1d.mean(axis=1), axis=0)                      # market(equal-wt)-demeaned return
    idio_cum5 = resid.rolling(5).sum()                                   # idiosyncratic 5d move

    signals = {}
    # --- Reversion family (fast, high-turnover — the known cost trap) ---
    signals["reversion"] = -(z["ret2"] + z["ret5"] + z["dist5"] + z["dist10"]
                             + (rngpos.sub(rngpos.mean(axis=1), axis=0))) / 5.0
    signals["reversal_2d"] = -z["ret2"]
    signals["rev_21d"] = -_xs_z_frame(dist21)                            # slower reversion
    signals["idio_rev5"] = -_xs_z_frame(idio_cum5)                       # beta-neutral reversion

    # --- Low-turnover / persistent family (the tradeability candidates) ---
    signals["lowvol"] = _xs_z_frame(-vol60)                             # long low realized-vol
    signals["mom_12_1"] = _xs_z_frame(mom_12_1)                         # classic 12-1 momentum
    signals["mom_6_1"] = _xs_z_frame(mom_6_1)
    signals["hilo_52w"] = _xs_z_frame(C / hi252)                        # near 52w-high = strong

    # --- Contrast + control ---
    signals["momentum_fast"] = (z["mom126"] + z["mom63"]) / 2.0        # short-horizon momo (anti-signal)
    rng = np.random.default_rng(0)
    signals["_null"] = pd.DataFrame(rng.standard_normal(C.shape), index=C.index, columns=C.columns)
    return signals, C


def forward_returns(C: pd.DataFrame, h: int) -> pd.DataFrame:
    """r[t -> t+h] from adjusted close. Strictly future; used only as labels."""
    return C.shift(-h) / C - 1.0


# ── Evaluation ──────────────────────────────────────────────────────────────────────────
def daily_ic(score: pd.DataFrame, fwd: pd.DataFrame, min_n=15) -> pd.Series:
    """Per-date cross-sectional Spearman rank-IC(score_t, fwd_t). NaN on thin cross-sections."""
    idx = score.index.intersection(fwd.index)
    out = {}
    s_rank = score.loc[idx].rank(axis=1)
    f_rank = fwd.loc[idx].rank(axis=1)
    for dt in idx:
        a, b = s_rank.loc[dt], f_rank.loc[dt]
        m = a.notna() & b.notna()
        if m.sum() < min_n or a[m].nunique() < 3:
            out[dt] = np.nan
        else:
            out[dt] = np.corrcoef(a[m], b[m])[0, 1]
    return pd.Series(out)


@dataclass
class ICReport:
    horizon: int
    mean_ic: float
    ic_ir: float          # mean/std of daily IC
    t_stat: float         # mean/std * sqrt(n_obs)
    hit_rate: float       # frac of days IC>0
    n_days: int


def summarize_ic(ic: pd.Series, horizon: int) -> ICReport:
    ic = ic.dropna()
    n = len(ic)
    mu, sd = ic.mean(), ic.std(ddof=1)
    return ICReport(horizon, mu, mu / (sd + 1e-12), mu / (sd + 1e-12) * np.sqrt(n),
                    float((ic > 0).mean()), n)


def quantile_backtest(score: pd.DataFrame, C: pd.DataFrame, h: int,
                      n_q=5, cost_bps=25.0):
    """NON-overlapping long-short quintile backtest at rebalance frequency h.

    Returns per-rebalance gross top-minus-bottom return, net of round-trip costs charged on
    realized turnover, plus annualized stats. cost_bps = one-way bps per name traded.
    """
    dates = score.index
    rebal = dates[::h]                      # non-overlapping so returns don't overlap
    fwd = forward_returns(C, h)
    prev_long, prev_short = set(), set()
    rows = []
    for dt in rebal:
        s = score.loc[dt].dropna()
        r = fwd.loc[dt] if dt in fwd.index else None
        if r is None or s.shape[0] < n_q * 3:
            continue
        s = s[s.index.isin(r.dropna().index)]
        if len(s) < n_q * 3:
            continue
        q = pd.qcut(s.rank(method="first"), n_q, labels=False)
        longs, shorts = set(s.index[q == n_q - 1]), set(s.index[q == 0])
        gross = r[list(longs)].mean() - r[list(shorts)].mean()
        # turnover vs previous rebalance (fraction of the 2*basket that changed), both legs
        turn = (len(longs ^ prev_long) + len(shorts ^ prev_short)) / (2 * (len(longs) + len(shorts)) + 1e-9)
        net = gross - turn * 2 * (cost_bps / 1e4)     # enter+exit ~ 2x one-way on turned names
        rows.append((dt, gross, net, turn))
        prev_long, prev_short = longs, shorts
    bt = pd.DataFrame(rows, columns=["date", "gross", "net", "turnover"]).set_index("date")
    if bt.empty:
        return bt, {}
    per_year = TRADING_DAYS / h
    ann = lambda x: x.mean() * per_year
    sharpe = lambda x: (x.mean() / (x.std(ddof=1) + 1e-12)) * np.sqrt(per_year)
    stats = {
        "rebalances": len(bt),
        "ann_gross": ann(bt["gross"]),
        "ann_net": ann(bt["net"]),
        "sharpe_net": sharpe(bt["net"]),
        "avg_turnover": bt["turnover"].mean(),
        "hit_rate_net": float((bt["net"] > 0).mean()),
    }
    return bt, stats


def ic_by_year(ic: pd.Series) -> pd.Series:
    ic = ic.dropna()
    return ic.groupby(ic.index.year).mean()


# ── Study driver ────────────────────────────────────────────────────────────────────────
def run_study(universe=None, start="2018-01-01", end=None,
              horizons=(1, 2, 3, 5, 10), cost_bps=25.0):
    universe = universe or DEFAULT_UNIVERSE
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    print(f"# fetching {len(universe)} names {start}..{end} (adjusted) …", flush=True)
    panel = fetch_panel(universe, start, end)
    print(f"# usable names: {len(panel)}", flush=True)
    if len(panel) < 20:
        print("! too few names fetched to run a cross-sectional study.")
        return

    signals, C = build_signals(panel)
    print(f"# dates: {C.index.min().date()}..{C.index.max().date()}  bars: {len(C)}\n")

    for name, score in signals.items():
        print(f"== signal: {name} " + "=" * (46 - len(name)))
        # IC across horizons
        print(f"  {'h':>3} {'meanIC':>8} {'IC_IR':>7} {'t':>7} {'hit%':>6} {'days':>6}")
        h_for_bt = None
        best = None
        for h in horizons:
            ic = daily_ic(score, forward_returns(C, h))
            rep = summarize_ic(ic, h)
            flag = ""
            if best is None or abs(rep.mean_ic) > abs(best.mean_ic):
                best = rep; h_for_bt = h; ic_best = ic
            print(f"  {h:>3} {rep.mean_ic:>8.4f} {rep.ic_ir:>7.3f} {rep.t_stat:>7.2f} "
                  f"{rep.hit_rate*100:>5.1f}% {rep.n_days:>6}{flag}")
        # per-year stability at the strongest horizon
        yr = ic_by_year(ic_best)
        print(f"  per-year meanIC @h={h_for_bt}: " +
              "  ".join(f"{y}:{v:+.3f}" for y, v in yr.items()))
        # cost-aware tradeable backtest at the strongest horizon
        if name != "_null":
            bt, st = quantile_backtest(score, C, h_for_bt, cost_bps=cost_bps)
            if st:
                print(f"  L/S quintile @h={h_for_bt} (cost {cost_bps:.0f}bps/side): "
                      f"ann_gross {st['ann_gross']*100:+.1f}%  ann_net {st['ann_net']*100:+.1f}%  "
                      f"Sharpe_net {st['sharpe_net']:+.2f}  turnover {st['avg_turnover']*100:.0f}%  "
                      f"hit {st['hit_rate_net']*100:.0f}%")
        print()


if __name__ == "__main__":
    args = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
    run_study(start=args.get("start", "2018-01-01"),
              end=args.get("end"),
              cost_bps=float(args.get("cost_bps", 25.0)))
