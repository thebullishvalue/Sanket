"""
ledger.py — Sanket's signals: three sets, all read from Pragyam's engine.

THE THREE SETS (all three fire on every screen, side by side; the Edge Study measures each)
-----------------------------------------------------------------------------------------
    A · PROVEN      ▲ the first closed bar of a state this name's own record has proven
                    POSITIVE; ▼ proven NEGATIVE. Pragyam's triangles - the ledger below.
    B · FLIP        ▲ on one closed bar, all of:
                      · the surprise column flips positive - below zero on the bar before,
                        at or above zero on this one
                      · the close is above the midline (the middle of the landing band)
                      · risk is CALM - the inner risk tape in its lightest third (σ's rank in
                        its last 500 bars below 33.3)
                      · risk is NOT RISING - the outer risk tape grey: σ has not risen more
                        than 10% over the last horizon
                      · the stretch ladder (the outer trend tape) leans up - green, any shade
                    ▼ mirrors: flips negative, close below the midline, risk calm and not
                    rising, the ladder leaning down.
    C · FLIP+TREND  B, and the trend tape (the inner trend tape) on the same side - green
                    (T ≥ 0) for ▲, red (T < 0) for ▼, any shade.
A tape with no reading yet blocks B and C. No set has a cooldown.

READY (B and C): every condition of a side in place except the trigger - the flip and the
close crossing the midline, which arrive together - AND the column armed: still below zero for
▲ (at or above for ▼), so its next crossing IS the flip. Measured on 151 instruments, daily: a
ready ▲ fires on the next bar about 20% of the time and within 10 bars about half, against 0.3%
and 9% for names on no list (studies/v10_watch/; hourly agrees).

NEAR (B): armed, risk calm and not rising, but the stretch ladder still on the wrong side by at most
0.2. The ladder averages the column with three other horizons, so the flip itself often tips it over:
of the B events that were not READY the bar before, 81-87% were short only the ladder. Near names
fire on the next bar about 9-12% of the time; READY and NEAR together held ~80% of B events the bar
before they fired. C's
ready is B's with the trend agreeing, so a name ready for C is ready for B as well.

An EVENT is any set's triangle; a bar can carry several (a C triangle is always a B one).

How often, and what to expect: A fires about once per 5,500 bars (151 instruments); B and C
fire far more often. The Indicator repo's flip study tested the nearest versions of B and C
(research/flips.txt): +0.7 and +0.9 σ × 100 over 10 bars, t below 1. The Edge Study measures all
three on your symbols - read it before leaning on any of them.

THE ENGINE, carried from pragyam.pine (research/pragyam.py in the Indicator repo)
---------------------------------------------------------------------------------
    RISK    per-bar variance from the bar's range and its gap (Garman-Klass), blended
            0.25 · 5-bar + 0.40 · 22-bar + 0.35 · 252-bar EMA, rescaled to close-to-close;
            first reading after 252 bars. Its rank in the last 500 bars is the risk tape.
    TREND   T = mean over 63 / 126 / 252 bars of (ln C − ln C[j]) / (σ √j).
    CONES   four horizons - h/2, h, 2h, 4h - each closing at ± q · σ √j with q LEARNED: each
            settled forecast moves log q by η · (miss − target), one term per risk regime and
            one per trend state, each side separately. The h cone is the landing band; the
            surprise is the h-bar move over its forecast; the STRETCH LADDER is the mean, over
            the four horizons, of each surprise as a fraction of its own band.
    LEDGER  outcome y = (ln C[t] − ln O[t−h+1]) / (σ √h) of the bar h bars back, clipped ±6;
            per-state Welford means; samples independent only ≥ h bars apart; needs ≥ 5 of them
            and > 30 · h settled forecasts; se² = var / n_eff, B = τ² / (τ² + se²),
            z = (mean − grand mean) · √(B / se²); proven at |z| ≥ 2.
Checked bar for bar against research/pragyam.py.
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

SETS = ("A", "B", "C")
SET_NAMES = {"A": "A · proven state", "B": "B · flip, calm, ladder", "C": "C · flip, calm, ladder + trend"}
RISE = 0.10           # risk momentum past this is "rising" (the outer risk tape turns gold)
NEAR = 0.20           # 'near': the ladder this close to agreeing - the flip itself often tips it
COLUMNS = ("lg_sigma", "lg_trend", "lg_state", "lg_n", "lg_mean", "lg_grand", "lg_edge",
           "lg_post", "lg_z", "lg_proven",
           "lg_u", "lg_band_up", "lg_band_dn", "lg_mid", "lg_prk", "lg_calm", "lg_rmom", "lg_rising",
           "lg_ladx", "lg_lad_up", "lg_lad_dn", "lg_flip",
           "a_buy", "a_sell", "b_buy", "b_sell", "c_buy", "c_sell",
           "ready_b_buy", "ready_b_sell", "ready_c_buy", "ready_c_sell", "near_b_buy", "near_b_sell",
           "turn_buy", "turn_sell", "decl", "decl_since")


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    """ta.ema: alpha 2 / (n + 1), seeded with the first value."""
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def _lag(x: np.ndarray, j: int) -> np.ndarray:
    return np.r_[np.full(j, np.nan), x[:-j]] if j > 0 else x


def compute(lo: pd.DataFrame, horizon: int = 10, tau: float = TAU) -> pd.DataFrame:
    """The engine, the ledger and the three signal sets for one name.

    ``lo``: lower-case OHLC on the chart's own bars (engine._chart_bars). Returns ``COLUMNS``
    on the same index:

      lg_sigma / lg_trend     the per-bar risk forecast, and T in σ
      lg_state / lg_n         0 … 8 (−1 warming), and independent samples behind it
      lg_mean / lg_grand / lg_edge / lg_post / lg_z / lg_proven   the state's record
      lg_u                    the surprise: the h-bar move over the forecast made h bars ago
      lg_band_up / lg_band_dn the landing band's learned edges, in σ (the pane's two lines)
      lg_mid                  the midline - the middle of the landing band, in price
      lg_prk / lg_calm        σ's rank in its last 500 bars; calm = the bottom third
      lg_rmom / lg_rising     ln(σ / σ[h]); rising = past +10% (the outer risk tape gold)
      lg_ladx / lg_lad_up / lg_lad_dn   the stretch ladder's lean, and rungs outside above / below
      lg_flip                 +1 the column flipped positive on this bar, −1 negative, 0 none
      a_* / b_* / c_*         each set's ▲ (buy) and ▼ (sell)
      ready_b_* / ready_c_*   set B's / set C's conditions in place bar the trigger, the column armed
      near_b_*                armed and calm, the ladder within NEAR of agreeing
      turn_buy / turn_sell    any set's ▲ / ▼ on this bar
      decl / decl_since       the last triangle of any set (+1 / −1, 0 none) and its bar
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
    prk = np.full(n, np.nan)
    if n > RN:
        W = np.lib.stride_tricks.sliding_window_view(sig, RN + 1)
        p = (W[:, :-1] <= W[:, -1:]).sum(axis=1) / RN * 100.0
        bad = np.isnan(W).any(axis=1)
        prk[RN:] = np.where(bad, np.nan, p)
        reg[RN:] = np.where(bad, 1, np.where(p < 100 / 3, 0, np.where(p < 200 / 3, 1, 2)))

    RG = [max(2, H // 2), H, 2 * H, 4 * H]                       # the stretch ladder's rungs
    NR = len(RG)
    with np.errstate(divide="ignore", invalid="ignore"):
        T = sum((lc - _lag(lc, j)) / (sig * np.sqrt(j)) for j in (63, 126, 252)) / 3
        U = [(lc - _lag(lc, j)) / (_lag(sig, j) * np.sqrt(j)) for j in RG]
        sH = sig * np.sqrt(H)
        y = np.clip((lc - _lag(lop, H - 1)) / _lag(sH, H), -6, 6)
    trs = np.where(T > 0.5, 2, np.where(T < -0.5, 0, 1))

    a = (1 - COVER) / 2
    lz = np.log(NormalDist().inv_cdf(1 - a))
    bU = [[lz] * 3 for _ in RG]; bD = [[lz] * 3 for _ in RG]          # by risk regime
    tU = [[0.0] * 3 for _ in RG]; tD = [[0.0] * 3 for _ in RG]        # by trend state
    QU = np.full((NR, n), np.nan); QD = np.full((NR, n), np.nan)
    MU = np.zeros((NR, n), dtype=bool); MD = np.zeros((NR, n), dtype=bool)
    OK = np.zeros((NR, n), dtype=bool); X = np.full((NR, n), np.nan)
    st = np.full(n, -1)
    cnt, mean, neff = np.zeros(9), np.zeros(9), np.zeros(9)
    last = np.full(9, -10 ** 9)
    gn, gm, gq = 0, 0.0, 0.0
    lg_n, lg_mean, lg_grand = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    edge, post, zpost = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)

    for t in range(n):
        for i, j in enumerate(RG):
            if t >= j and not np.isnan(U[i][t]) and not np.isnan(QU[i][t - j]):   # forecast made j bars ago, settled
                qu, qd, uu = QU[i][t - j], QD[i][t - j], U[i][t]
                MU[i][t], MD[i][t], OK[i][t] = uu > qu, uu < -qd, True
                X[i][t] = uu / qu if uu >= 0 else uu / qd
                jr, jt = reg[t - j], trs[t - j]
                eu, ed = ETA * (MU[i][t] - a), ETA * (MD[i][t] - a)
                bU[i][jr] += eu; bD[i][jr] += ed
                tU[i][jt] += eu; tD[i][jt] += ed
        if OK[1][t]:
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
            for i in range(NR):
                QU[i][t], QD[i][t] = np.exp(bU[i][jr] + tU[i][jt]), np.exp(bD[i][jr] + tD[i][jt])
        if OK[1][t] and not np.isnan(T[t]):
            s = 3 * (2 if T[t] > 0.5 else 0 if T[t] < -0.5 else 1) + (2 if MU[1][t] else 0 if MD[1][t] else 1)
            st[t] = s
            lg_n[t], lg_mean[t], lg_grand[t] = neff[s], mean[s], gm
            if neff[s] >= 5 and gn > 30 * H:
                se2 = (gq / (gn - 1)) / neff[s]
                B = tau ** 2 / (tau ** 2 + se2)
                dd = mean[s] - gm
                edge[t], post[t], zpost[t] = dd, B * dd, dd * np.sqrt(B / se2)

    # ── set A: entering a proven state ──
    proven = np.where(np.nan_to_num(zpost, nan=0.0) >= 2, 1, np.where(np.nan_to_num(zpost, nan=0.0) <= -2, -1, 0))
    enter = (st >= 0) & (st != np.r_[-1, st[:-1]])
    a_buy, a_sell = enter & (proven == 1), enter & (proven == -1)

    # ── the readings sets B and C read ──
    allok = OK.all(axis=0)
    ladx = np.where(allok, np.nanmean(np.where(allok, X, 0.0), axis=0), np.nan)
    lad_up = np.where(allok, MU.sum(axis=0), np.nan)
    lad_dn = np.where(allok, MD.sum(axis=0), np.nan)
    u = U[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        band_up_px, band_dn_px = c * np.exp(QU[1] * sH), c * np.exp(-QD[1] * sH)
        mid = (_lag(band_up_px, H) + _lag(band_dn_px, H)) / 2.0
        rmom = np.log(sig / _lag(sig, H))
        u_prev = _lag(u, 1)
        flip = np.where((u >= 0) & (u_prev < 0), 1, np.where((u < 0) & (u_prev >= 0), -1, 0))
        calm = (reg == 0) & np.isfinite(prk)
        rising = rmom > RISE
        quiet = calm & np.isfinite(rmom) & ~rising                 # inner tape calm, outer tape grey
        lad_ok, t_ok = np.isfinite(ladx), np.isfinite(T)
        set_b_buy = quiet & lad_ok & (ladx >= 0)                    # the slow conditions, per side
        set_b_sell = quiet & lad_ok & (ladx < 0)
        b_buy = (flip == 1) & (c > mid) & set_b_buy
        b_sell = (flip == -1) & (c < mid) & set_b_sell
        c_buy = b_buy & t_ok & (T >= 0)
        c_sell = b_sell & t_ok & (T < 0)
        # READY = the slow conditions AND the column ARMED - still on the far side of zero, so its
        # next crossing is the flip. Without the second half, 85% of 'ready' bars had a column
        # already past zero: 0% fired on the next bar (studies/v10_watch/watch_check.py).
        ready_b_buy = set_b_buy & (u < 0)
        ready_b_sell = set_b_sell & (u >= 0)
        ready_c_buy, ready_c_sell = ready_b_buy & t_ok & (T >= 0), ready_b_sell & t_ok & (T < 0)
        # NEAR = armed, risk calm and not rising, the ladder on the wrong side by at most NEAR. The
        # ladder averages the column with three other horizons, so the flip itself often tips it:
        # 81-87% of the B events that were not READY the bar before were short only the ladder.
        near_b_buy = quiet & lad_ok & (ladx < 0) & (ladx >= -NEAR) & (u < 0)
        near_b_sell = quiet & lad_ok & (ladx >= 0) & (ladx <= NEAR) & (u >= 0)

    buy, sell = a_buy | b_buy | c_buy, a_sell | b_sell | c_sell
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
        "lg_u": u, "lg_band_up": QU[1], "lg_band_dn": QD[1], "lg_mid": mid, "lg_prk": prk,
        "lg_calm": calm, "lg_rmom": rmom, "lg_rising": rising,
        "lg_ladx": ladx, "lg_lad_up": lad_up, "lg_lad_dn": lad_dn, "lg_flip": flip,
        "a_buy": a_buy, "a_sell": a_sell, "b_buy": b_buy, "b_sell": b_sell, "c_buy": c_buy, "c_sell": c_sell,
        "ready_b_buy": ready_b_buy, "ready_b_sell": ready_b_sell,
        "ready_c_buy": ready_c_buy, "ready_c_sell": ready_c_sell,
        "near_b_buy": near_b_buy, "near_b_sell": near_b_sell,
        "turn_buy": buy, "turn_sell": sell, "decl": decl, "decl_since": since,
    }, index=idx)[list(COLUMNS)]


def tags(a, b, c) -> np.ndarray:
    """Which sets fired, per bar, as one label: 'A', 'B', 'B·C', 'A·B·C' … or ''."""
    a, b, c = (np.asarray(x, dtype=bool) for x in (a, b, c))
    out = np.full(a.shape, "", dtype=object)
    for k, m in (("A", a), ("B", b), ("C", c)):
        out = np.where(m, np.where(out == "", k, out + "·" + k), out)
    return out


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


__all__ = ["COLUMNS", "NEAR", "RISE", "SETS", "SET_NAMES", "STATE_NAMES", "TAU", "compute", "ready_bars", "state_name", "tags"]
