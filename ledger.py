"""
ledger.py — Sanket's signals: Pragyam's proven-state triangles, ported.

What a signal is now
--------------------
Every closed bar is put in one of NINE STATES — the trend (down / flat / up) crossed with
where the close landed against the cone drawn ``horizon`` bars ago (below / inside / above).
For each state the ledger keeps THIS NAME's own record: what the next ``horizon`` bars did
after every bar spent in it, from the next bar's open, in units of the risk forecast, against
the name's own average. That record is shrunk toward "no edge" — a real edge is assumed to be
about ``tau`` σ at most — and a state is PROVEN only when the shrunk evidence clears two
standard errors.

    ▲  the first closed bar of a state this name's history has proven POSITIVE
    ▼  the first closed bar of a state this name's history has proven NEGATIVE

There is no cooldown, no gate on another indicator, and no watchlist: the ledger is the whole
rule. Most bars fire nothing, and on most names most states never become proven. That is the
design, not a fault — on random walks the same rule calls a state proven on about 0.4% of
bars (Indicator repo, research/results.txt).

The engine, carried exactly from pragyam.pine (research/pragyam.py in the Indicator repo)
-----------------------------------------------------------------------------------------
    RISK    per-bar variance from the bar's range and its gap (Garman-Klass), blended
            0.25 · 5-bar + 0.40 · 22-bar + 0.35 · 252-bar EMA, rescaled to close-to-close;
            first reading after 252 bars.
    TREND   T = mean over 63 / 126 / 252 bars of (ln C − ln C[j]) / (σ √j); up past +0.5,
            down past −0.5.
    CONE    the close ``horizon`` bars ahead, ± q · σ √h. q is LEARNED: each settled forecast
            moves log q by η · (miss − target) — one term per risk regime (σ's rank in its
            last 500 bars, in thirds) and one per trend state, each side separately — so the
            cone holds its stated 80% on this name.
    LEDGER  outcome y = (ln C[t] − ln O[t−h+1]) / (σ √h) of the bar h bars back, clipped ±6;
            per-state Welford means; samples counted as independent only ≥ h bars apart;
            needs ≥ 5 independent samples in the state and > 30 · h settled forecasts overall;
            se² = var / n_eff, B = τ² / (τ² + se²), z = (mean − grand mean) · √(B / se²).

Only the parts the triangles read are ported (Pragyam's other horizon rungs, stops and exposure
dial do not feed the ledger and are left out). Checked bar for bar against research/pragyam.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statistics import NormalDist

WARM = 252            # bars before the first risk reading
RN = 500              # window for the risk regime's rank
COVER = 0.80          # the cone's stated coverage
ETA = 0.02            # learning speed of the cone's widths
TAU = 0.10            # the believable edge, σ over the horizon
STATE_NAMES = tuple(f"trend {t} · {c}" for t in ("down", "flat", "up")
                    for c in ("closed below the band", "inside the band", "closed above the band"))

COLUMNS = ("lg_sigma", "lg_trend", "lg_state", "lg_n", "lg_mean", "lg_grand", "lg_edge",
           "lg_post", "lg_z", "lg_proven", "turn_buy", "turn_sell", "decl", "decl_since")


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    """ta.ema: alpha 2 / (n + 1), seeded with the first value."""
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def _lag(x: np.ndarray, j: int) -> np.ndarray:
    return np.r_[np.full(j, np.nan), x[:-j]] if j > 0 else x


def compute(lo: pd.DataFrame, horizon: int = 10, tau: float = TAU) -> pd.DataFrame:
    """The ledger and its triangles for one name.

    ``lo``: lower-case OHLC on the chart's own bars (engine._chart_bars). Returns ``COLUMNS``
    on the same index:

      lg_sigma    the per-bar risk forecast (log return)
      lg_trend    T, in σ
      lg_state    0 … 8 (trend × where the close landed), −1 before the first settled forecast
      lg_n        independent samples behind this bar's state
      lg_mean     the state's mean outcome on this name, σ over the horizon
      lg_grand    the name's mean outcome over every state
      lg_edge     lg_mean − lg_grand
      lg_post     the edge after shrinkage — what the ledger believes
      lg_z        the shrunk edge in standard errors (NaN until the state has enough record)
      lg_proven   +1 / −1 when |z| ≥ 2, else 0
      turn_buy    ▲: the first bar of a state proven positive
      turn_sell   ▼: the first bar of a state proven negative
      decl / decl_since   the last triangle (+1 ▲ / −1 ▼, 0 none) and the bar it fired on
    """
    H = int(horizon)
    o, h, l, c = (lo[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    n = len(c)
    idx = lo.index
    if n == 0:
        return pd.DataFrame(columns=list(COLUMNS), index=idx)
    with np.errstate(divide="ignore", invalid="ignore"):
        lc, lop = np.log(c), np.log(o)
        r = np.r_[np.nan, np.diff(lc)]
        gp = lop - np.r_[np.nan, lc[:-1]]
        co = lc - lop
        hl = np.log(h) - np.log(l)
        gk = gp ** 2 + 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
        v = np.where((hl > 0) & (gk > 0), gk, r ** 2)
        S, M, L, rL = _ema(v, 5), _ema(v, 22), _ema(v, 252), _ema(r ** 2, 252)
        k = np.clip(rL / L, 0.5, 2.0)
        sig = np.sqrt(k * (0.25 * S + 0.40 * M + 0.35 * L))
    sig[:WARM] = np.nan
    sig[~(sig > 0)] = np.nan

    reg = np.ones(n, dtype=int)
    if n > RN:
        W = np.lib.stride_tricks.sliding_window_view(sig, RN + 1)
        p = (W[:, :-1] <= W[:, -1:]).sum(axis=1) / RN * 100.0
        bad = np.isnan(W).any(axis=1)
        reg[RN:] = np.where(bad, 1, np.where(p < 100 / 3, 0, np.where(p < 200 / 3, 1, 2)))

    with np.errstate(divide="ignore", invalid="ignore"):
        T = sum((lc - _lag(lc, j)) / (sig * np.sqrt(j)) for j in (63, 126, 252)) / 3
        U = (lc - _lag(lc, H)) / (_lag(sig, H) * np.sqrt(H))
        sH = sig * np.sqrt(H)
        y = np.clip((lc - _lag(lop, H - 1)) / _lag(sH, H), -6, 6)
    trs = np.where(T > 0.5, 2, np.where(T < -0.5, 0, 1))

    a = (1 - COVER) / 2
    lz = np.log(NormalDist().inv_cdf(1 - a))
    bU, bD = [lz] * 3, [lz] * 3                    # by risk regime
    tU, tD = [0.0] * 3, [0.0] * 3                  # by trend state
    QU, QD = np.full(n, np.nan), np.full(n, np.nan)
    st = np.full(n, -1)
    cnt, mean, neff = np.zeros(9), np.zeros(9), np.zeros(9)
    last = np.full(9, -10 ** 9)
    gn, gm, gq = 0, 0.0, 0.0
    lg_n, lg_mean, lg_grand = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    edge, post, zpost = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)

    for t in range(n):
        settled = t >= H and not np.isnan(U[t]) and not np.isnan(QU[t - H])
        mU = mD = False
        if settled:                                 # the forecast made H bars ago has been settled
            mU, mD = U[t] > QU[t - H], U[t] < -QD[t - H]
            jr, jt = reg[t - H], trs[t - H]
            eu, ed = ETA * (mU - a), ETA * (mD - a)
            bU[jr] += eu; bD[jr] += ed
            tU[jt] += eu; tD[jt] += ed
            s = st[t - H]
            if s >= 0 and not np.isnan(y[t]):
                cnt[s] += 1
                mean[s] += (y[t] - mean[s]) / cnt[s]
                gn += 1
                if t - H - last[s] >= H:            # independent samples: signal bars at least H apart
                    neff[s] += 1
                    last[s] = t - H
                dlt = y[t] - gm
                gm += dlt / gn
                gq += dlt * (y[t] - gm)
        if not np.isnan(sig[t]):
            jr, jt = reg[t], trs[t]
            QU[t], QD[t] = np.exp(bU[jr] + tU[jt]), np.exp(bD[jr] + tD[jt])
        if settled and not np.isnan(T[t]):
            s = 3 * (2 if T[t] > 0.5 else 0 if T[t] < -0.5 else 1) + (2 if mU else 0 if mD else 1)
            st[t] = s
            lg_n[t], lg_mean[t], lg_grand[t] = neff[s], mean[s], gm
            if neff[s] >= 5 and gn > 30 * H:
                se2 = (gq / (gn - 1)) / neff[s]
                B = tau ** 2 / (tau ** 2 + se2)
                dd = mean[s] - gm
                edge[t], post[t], zpost[t] = dd, B * dd, dd * np.sqrt(B / se2)

    proven = np.where(zpost >= 2, 1, np.where(zpost <= -2, -1, 0))
    enter = (st >= 0) & (st != np.r_[-1, st[:-1]])
    buy = enter & (proven == 1)
    sell = enter & (proven == -1)
    decl = np.zeros(n, dtype=int)
    since = np.full(n, -1, dtype=int)
    d_now, d_bar = 0, -1
    for t in range(n):
        if buy[t]:
            d_now, d_bar = 1, t
        elif sell[t]:
            d_now, d_bar = -1, t
        decl[t], since[t] = d_now, d_bar
    return pd.DataFrame({
        "lg_sigma": sig, "lg_trend": T, "lg_state": st, "lg_n": lg_n, "lg_mean": lg_mean,
        "lg_grand": lg_grand, "lg_edge": edge, "lg_post": post, "lg_z": zpost, "lg_proven": proven,
        "turn_buy": buy, "turn_sell": sell, "decl": decl, "decl_since": since,
    }, index=idx)[list(COLUMNS)]


def state_name(s) -> str:
    """'trend up · inside the band', or '' before the first settled forecast."""
    try:
        s = int(s)
    except (TypeError, ValueError):
        return ""
    return STATE_NAMES[s] if 0 <= s < 9 else ""


def ready_bars(horizon: int = 10) -> int:
    """Bars of history before a state CAN be proven: the risk forecast's warm-up, one
    horizon to settle the first forecast, and 30 horizons of settled record."""
    return WARM + int(horizon) + 30 * int(horizon) + 1


__all__ = ["COLUMNS", "STATE_NAMES", "TAU", "compute", "ready_bars", "state_name"]
