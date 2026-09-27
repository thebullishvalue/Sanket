"""
pine_v5.py — the parts of pragati_v5.pine that differ from v6, on top of the shared port.

v5 and v6 share every engine: the conviction engine (Nishchaya v3), the parent ladder
rung, Samanvaya's value engine and value tape, the conviction × value trace and its
histogram (pragati.py computes all of them, bar for bar with the Pine). v5 differs in
what it READS from them, and that is what this module ports, rule by rule:

  SIGNALS   ▲▼ confirm on absorption alone (v6 also accepts a bullish/bearish
            divergence as the failed push); ◆ requires chart conviction on its side
            and does NOT require the trace inside ±θ (v6 does).
  R / H     regular and hidden divergence are SIGNALS in v5, found on conviction's
            pivots and confirmed by the ladder tapes at the pivot (v6 keeps them as
            evidence only).
  GRID      Pragyam's 3 × 3 (UP / FAINT / DOWN × cheap / fair / rich) with rows run
            by conviction's OWN histogram (tier and quiet), the 5 × 5 momentum
            phases, and the graded units read at the name's shaded position.
  OI        the per-bar quadrant with a zero-centred RMS scale and the roll rules.

MEASUREMENT CAVEAT. v5's default conviction ladder is Ladder DOWN — 1m … 4h intrabars
inside each daily bar. Twenty years of intraday history do not exist on any free feed,
so v5 is measured with its conviction ladder on Ladder UP (W · D) — the setting its own
header names as the one Pragyam reads. Its value ladder default is already Ladder up.

Author: @thebullishvalue
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import pragati as pg

#: Pragyam's 3 × 3 units — v5's f_gUnit. Keys (row, col): row −1 DOWN / 0 FAINT / +1 UP,
#: col 0 cheap / 1 fair / 2 rich.
UNITS_V5 = {(1, 0): 3.0, (1, 1): 3.0, (1, 2): 1.5,
            (0, 0): 1.5, (0, 1): 1.0, (0, 2): 0.75,
            (-1, 0): 1.0, (-1, 1): 0.5, (-1, 2): 0.25}
ACTIONS_V5 = {(1, 0): "Buy", (1, 1): "Add", (1, 2): "Hold",
              (0, 0): "Accumulate", (0, 1): "Wait", (0, 2): "Trim",
              (-1, 0): "Watch", (-1, 1): "Reduce", (-1, 2): "Exit"}


def _sma_min(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n)


def _stdev(x: pd.Series, n: int) -> pd.Series:
    """ta.stdev (population), na until a full window of values exists."""
    return x.rolling(n, min_periods=n).std(ddof=0)


def _cross_over(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a1 = np.concatenate([[np.nan], a[:-1]])
    b1 = np.concatenate([[np.nan], b[:-1]])
    return (a > b) & (a1 <= b1)


def _cross_under(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a1 = np.concatenate([[np.nan], a[:-1]])
    b1 = np.concatenate([[np.nan], b[:-1]])
    return (a < b) & (a1 >= b1)


# ════════════════════════════════════════════════════════════════════════════════════════
# SIGNALS · ▲▼ declaration and ◆ continuation, v5's rules          (Pine section 8)
# ════════════════════════════════════════════════════════════════════════════════════════
def signals(out: pd.DataFrame, cv: pd.Series, p: pg.Params) -> pd.DataFrame:
    T = len(out)
    th = float(p.theta)
    trace = out["trace"].to_numpy(dtype=float)
    hist = out["hist"]
    hv = hist.to_numpy(dtype=float)
    m_c = out["c_tape"].to_numpy(dtype=float)
    m_v = out["v_tape"].to_numpy(dtype=float)
    stack = out["stack_ok"].fillna(False).to_numpy(bool)
    tradable = ~out["settling"].fillna(True).to_numpy(bool)
    eff_abs = out["eff_abs"].fillna(False).to_numpy(bool)
    cvo = cv.to_numpy(dtype=float)

    # mCUp: rising two bars running, spelled out as the Pine spells it
    c1 = np.concatenate([[np.nan], m_c[:-1]])
    c2 = np.concatenate([[np.nan, np.nan], m_c[:-2]])
    c1n = np.where(np.isfinite(c1), c1, m_c)
    c2n = np.where(np.isfinite(c2), c2, np.nan_to_num(c1))
    up2 = (m_c > c1n) & (np.nan_to_num(c1) > c2n)
    dn2 = (m_c < c1n) & (np.nan_to_num(c1) < c2n)
    vs = pd.Series(m_v)
    v_lo = vs.rolling(p.disloc, min_periods=p.disloc).min().fillna(vs).to_numpy()
    v_hi = vs.rolling(p.disloc, min_periods=p.disloc).max().fillna(vs).to_numpy()
    abs_seen = pd.Series(eff_abs.astype(float)).rolling(p.disloc, min_periods=1).max().to_numpy() > 0

    t_buy = (v_lo <= -th) & (m_v < th) & (hv > 0) & ((m_c > 0) | up2) & ((not p.effort) | abs_seen)
    t_sell = (v_hi >= th) & (m_v > -th) & (hv < 0) & ((m_c < 0) | dn2) & ((not p.effort) | abs_seen)
    t_con_l = (m_c >= p.z1) & (m_v < th) & ((not p.effort) | ~eff_abs)
    t_con_s = (m_c <= -p.z1) & (m_v > -th) & ((not p.effort) | ~eff_abs)

    hs = hist
    pulled_up = (hs.rolling(p.pull, min_periods=p.pull).min() < 0).to_numpy()
    pulled_dn = (hs.rolling(p.pull, min_periods=p.pull).max() > 0).to_numpy()
    thr = (p.k * _stdev(hist, p.norm).fillna(0.0)).to_numpy()
    imp_up = _cross_over(hv, thr)
    imp_dn = _cross_under(hv, -thr)
    x_long = _cross_over(trace, np.full(T, -th))
    x_short = _cross_under(trace, np.full(T, th))

    buy = np.zeros(T, bool); sell = np.zeros(T, bool)
    con_l = np.zeros(T, bool); con_s = np.zeros(T, bool)
    decl = np.zeros(T, int)
    arm_b = arm_s = None
    last_l = last_s = None
    d_now = 0
    for t in range(T):
        cool_l = last_l is None or t - last_l >= p.cool
        cool_s = last_s is None or t - last_s >= p.cool
        if tradable[t] and x_long[t]:
            arm_b, arm_s = t, None
        if tradable[t] and x_short[t]:
            arm_s, arm_b = t, None
        if np.isfinite(trace[t]) and trace[t] <= -th:
            arm_b = None
        if np.isfinite(trace[t]) and trace[t] >= th:
            arm_s = None
        armed_b = arm_b is not None and t - arm_b <= p.confirm - 1
        armed_s = arm_s is not None and t - arm_s <= p.confirm - 1
        b = p.turn and stack[t] and tradable[t] and armed_b and t_buy[t] and cool_l
        s = p.turn and stack[t] and tradable[t] and armed_s and t_sell[t] and cool_s
        if b:
            arm_b = None
        if s:
            arm_s = None
        cl = (p.resume and stack[t] and not b and cvo[t] > 0 and pulled_up[t] and imp_up[t]
              and cool_l and t_con_l[t])
        cs = (p.resume and stack[t] and not s and cvo[t] < 0 and pulled_dn[t] and imp_dn[t]
              and cool_s and t_con_s[t])
        if b or cl:
            last_l = t
        if s or cs:
            last_s = t
        if b:
            d_now = 1
        if s:
            d_now = -1
        buy[t], sell[t], con_l[t], con_s[t], decl[t] = b, s, cl, cs, d_now
    return pd.DataFrame({"v5_buy": buy, "v5_sell": sell, "v5_con_l": con_l, "v5_con_s": con_s,
                         "v5_decl": decl}, index=out.index)


# ════════════════════════════════════════════════════════════════════════════════════════
# DIVERGENCE · R and H as signals                                     (Pine section 7)
# ════════════════════════════════════════════════════════════════════════════════════════
def divergence(lo: pd.DataFrame, out: pd.DataFrame, cv: pd.Series, cv_ready: np.ndarray,
               p: pg.Params) -> pd.DataFrame:
    T = len(out)
    th = float(p.theta)
    cvo = cv.to_numpy(dtype=float)
    lad = out["c_tape"].to_numpy(dtype=float)
    tp = out["v_tape"].to_numpy(dtype=float)
    stack = out["stack_ok"].fillna(False).to_numpy(bool)
    span = p.pl + p.pr + 1
    swing_hi = lo["high"].rolling(span, min_periods=span).max().to_numpy()
    swing_lo = lo["low"].rolling(span, min_periods=span).min().to_numpy()
    pv_h = pg._pivots(cvo, p.pl, p.pr, True)
    pv_l = pg._pivots(cvo, p.pl, p.pr, False)
    r_bear = np.zeros(T, bool); r_bull = np.zeros(T, bool)
    h_bear = np.zeros(T, bool); h_bull = np.zeros(T, bool)
    for is_high, pv, swing in ((True, pv_h, swing_hi), (False, pv_l, swing_lo)):
        last_v = last_b = last_p = last_c = None
        for t in np.flatnonzero(np.isfinite(pv) & cv_ready):
            b = t - p.pr
            pp = float(swing[t])
            c_piv = lad[t - p.pr] if t - p.pr >= 0 else np.nan
            v_piv = tp[t - p.pr] if t - p.pr >= 0 else np.nan
            gap = 0 if last_b is None else b - last_b
            if last_v is not None and p.gap_min <= gap <= p.gap_max:
                if is_high:
                    r_ok = stack[t] and last_c is not None and np.isfinite(last_c) and c_piv < last_c and v_piv > 0
                    h_ok = stack[t] and c_piv <= -p.z1 and v_piv > -th
                    if pv[t] < last_v and pp > last_p and (not p.zone_gate or last_v >= p.z1) and r_ok:
                        r_bear[t] = True
                    if pv[t] > last_v and pp < last_p and h_ok:
                        h_bear[t] = True
                else:
                    r_ok = stack[t] and last_c is not None and np.isfinite(last_c) and c_piv > last_c and v_piv < 0
                    h_ok = stack[t] and c_piv >= p.z1 and v_piv < th
                    if pv[t] > last_v and pp < last_p and (not p.zone_gate or last_v <= -p.z1) and r_ok:
                        r_bull[t] = True
                    if pv[t] < last_v and pp > last_p and h_ok:
                        h_bull[t] = True
            last_v, last_b, last_p, last_c = float(pv[t]), b, pp, c_piv
    return pd.DataFrame({"v5_r_bear": r_bear, "v5_r_bull": r_bull, "v5_h_bear": h_bear,
                         "v5_h_bull": h_bull}, index=out.index)


# ════════════════════════════════════════════════════════════════════════════════════════
# THE 3 × 3 GRID · Pragyam's style, read on the chart                (Pine "CVG" block)
# ════════════════════════════════════════════════════════════════════════════════════════
def _tape_ink(x, knee, solid):
    a = np.abs(x)
    bright = 0.65 + 0.35 * np.clip((a - knee) / max(solid - knee, 1e-9), 0.0, 1.0)
    faint = 0.12 + 0.33 * np.clip(a / max(knee, 1e-9), 0.0, 1.0)
    return np.where(a >= knee, bright, faint)


def _g_ink(m, lo, hi, t0, t1):
    rng = hi - lo
    f = np.where(rng > 1e-12, np.clip((m - lo) / np.where(rng > 1e-12, rng, 1.0), 0.0, 1.0), 0.0)
    return 1.0 - (t0 + (t1 - t0) * f) / 100.0


def _g_at(rp, cp, units: dict):
    """f_gAt: the nine units read at a position — bilinear between the cells."""
    r = min(1.0, max(-1.0, rp))
    c = min(1.0, max(-1.0, cp))
    r0 = -1 if r < 0 else 0
    c0 = -1 if c < 0 else 0
    wr, wc = r - r0, c - c0
    U = lambda rr, cc: units[(rr, cc)]                     # noqa: E731
    lo_ = (1.0 - wc) * U(r0, c0 + 1) + wc * U(r0, c0 + 2)
    hi_ = (1.0 - wc) * U(r0 + 1, c0 + 1) + wc * U(r0 + 1, c0 + 2)
    return (1.0 - wr) * lo_ + wr * hi_


def grid(out: pd.DataFrame, cv: pd.Series, raw_sd: pd.Series, cv_ready: np.ndarray,
         p: pg.Params, units: dict | None = None) -> pd.DataFrame:
    units = units or UNITS_V5
    T = len(out)
    th = float(p.theta)
    # conviction's own histogram and its tiers — the gate Pragyam reads
    cv_hist = cv - pg._ema(cv, p.signal)
    d_cv = cv - cv.shift(1).fillna(cv)
    cv_db = 0.1 * _stdev(d_cv, p.norm).fillna(0.0)
    cv_hsd = _stdev(cv_hist, p.norm)
    cv_thr = p.k * cv_hsd.fillna(0.0)
    cv_mag = cv_hist.abs()
    above = (cv_hist >= 0).to_numpy()
    h1 = cv_hist.shift(1).fillna(0.0)
    expand = np.where(above, cv_hist > h1, cv_hist < h1)
    with_ = np.where(above, d_cv > cv_db, d_cv < -cv_db)
    imp = (cv_mag >= cv_thr).to_numpy()
    cv_hi = np.maximum.reduce([2.0 * cv_hsd.fillna(0.0).to_numpy(), 1.5 * cv_thr.to_numpy(),
                               np.full(T, 1e-9)])
    raw_pr = pg._percentrank(raw_sd.to_numpy(dtype=float), min(4 * p.norm, 4999))
    quiet = cv_ready & np.isfinite(raw_pr) & (np.nan_to_num(raw_pr, nan=100.0) < 20.0)
    tier = np.where(expand & imp, 0, np.where(expand, 1, np.where(with_, 2, 3)))
    gate = np.where(~cv_ready, np.nan,
                    np.where((tier != 3) & ~quiet, np.where(above, 1.0, -1.0), 0.0))
    m = cv_mag.to_numpy()
    thr_ = cv_thr.to_numpy()
    ink = np.select([tier == 0, tier == 1, tier == 2],
                    [_g_ink(m, thr_, cv_hi, 18.0, 0.0),
                     _g_ink(m, np.zeros(T), np.maximum(thr_, 1e-9), 60.0, 38.0),
                     _g_ink(m, np.zeros(T), cv_hi, 70.0, 50.0)],
                    _g_ink(m, np.zeros(T), cv_hi, 88.0, 72.0))
    g_push = np.where(~cv_ready, np.nan, np.where(above, 1.0, -1.0) * ink * np.where(quiet, 0.45, 1.0))

    # the momentum tapes (ladder up on both instruments: chart − tape)
    m_c = out["c_tape"]
    m_v = out["v_tape"]
    c_ready = out["c_ready"].fillna(False).to_numpy(bool)
    v_ready = out["v_ready"].fillna(False).to_numpy(bool)
    lad_mom = cv - m_c
    lm_ready = c_ready & cv_ready
    lm_sd = lad_mom.where(lm_ready).rolling(p.norm, min_periods=p.norm).std(ddof=0)
    lm_thr = (p.k * lm_sd.fillna(0.0)).to_numpy()
    tp_mom = out["value"] - m_v
    tp_thr = (0.5 * _stdev(tp_mom, 200).fillna(0.0)).to_numpy()
    lmv = lad_mom.to_numpy(dtype=float)
    tpv = tp_mom.to_numpy(dtype=float)
    mc = m_c.to_numpy(dtype=float)
    mv = m_v.to_numpy(dtype=float)
    read = c_ready & v_ready

    def side(prev, x, knee):
        if not np.isfinite(x):
            return 0.0
        if not np.isfinite(knee) or knee <= 0:
            return float(np.sign(x))
        if x >= knee:
            return 1.0
        if x <= -knee:
            return -1.0
        return prev if prev != 0.0 else float(np.sign(x))

    row = np.full(T, np.nan); col = np.full(T, np.nan); held = np.zeros(T, bool)
    u = np.full(T, np.nan); cell_flat = np.full(T, np.nan)
    c_ph = np.zeros(T, int); v_ph = np.zeros(T, int)
    g_row = None
    sc = sv_ = 0.0
    ink_c = _tape_ink(mc, p.z1, p.z2)
    ink_v = _tape_ink(mv, th, 70.0)
    for t in range(T):
        if not read[t]:
            g_row, sc, sv_ = None, 0.0, 0.0
            continue
        tgt = 1 if mc[t] >= p.z1 else (-1 if mc[t] <= -p.z1 else 0)
        gc = 0 if mv[t] <= -th else (2 if mv[t] >= th else 1)
        gg = gate[t]
        if g_row is None or not np.isfinite(gg):
            g_row = tgt
        elif tgt > g_row and gg > 0:
            g_row = tgt
        elif tgt < g_row and gg < 0:
            g_row = tgt
        sc = side(sc, lmv[t] if lm_ready[t] else np.nan, lm_thr[t])
        sv_ = side(sv_, tpv[t] if v_ready[t] else np.nan, tp_thr[t])
        hd = g_row != tgt
        cph = 0 if (g_row == 0 or sc == 0.0) else (1 if sc == g_row else -1)
        vph = 0 if (gc == 1 or sv_ == 0.0) else (1 if sv_ == (1.0 if gc == 0 else -1.0) else -1)
        if hd:
            hs = 1.0 if g_row > tgt else -1.0
            hh = 0.0 if not np.isfinite(g_push[t]) else min(1.0, max(0.0, g_push[t] * hs))
            nb = g_row + (1 if tgt > g_row else -1)
            pos_r = g_row * (0.5 + 0.5 * hh) + nb * (0.5 - 0.5 * hh)
        elif g_row != 0:
            pos_r = g_row * ink_c[t]
        else:
            pos_r = (1.0 if mc[t] > 0 else -1.0) * (ink_c[t] - 0.12)
        pos_c = (gc - 1) * ink_v[t] if gc != 1 else (1.0 if mv[t] > 0 else -1.0) * (ink_v[t] - 0.12)
        if g_row != 0 and cph == -1:
            pos_r *= 0.5
        if gc != 1 and vph == -1:
            pos_c *= 0.5
        row[t], col[t], held[t], c_ph[t], v_ph[t] = g_row, gc, hd, cph, vph
        u[t] = _g_at(pos_r, pos_c, units)
        cell_flat[t] = units[(g_row, gc)]
    return pd.DataFrame({"v5_row": row, "v5_col": col, "v5_held": held, "v5_units": u,
                         "v5_cell_units": cell_flat, "v5_cph": c_ph, "v5_vph": v_ph,
                         "v5_push": g_push, "v5_gate": gate}, index=out.index)


# ════════════════════════════════════════════════════════════════════════════════════════
# OPEN INTEREST · the per-bar quadrant                                (Pine OI block)
# ════════════════════════════════════════════════════════════════════════════════════════
OI_BASE, OI_ROLL, OI_ROLL_ABS, OI_STRU, OI_ACT, OI_DEAD, OI_PCT = 50, 3.0, 0.20, 0.50, 0.20, 0.50, 250


def oi_quadrant(close: pd.Series, oi: pd.Series, sig_len: int = 9) -> pd.DataFrame:
    """v5's OI character on one name: LONG/SHORT BUILDUP, SHORT COVERING, LONG UNWINDING,
    balanced, or EXPIRY/ROLLOVER; plus the book's percentile and the windowed exit counts
    the gold cast reads."""
    T = len(close)
    o = oi.to_numpy(dtype=float)
    moved = np.r_[0.0, (np.diff(o) != 0) & np.isfinite(np.diff(o))].astype(float)
    act = pd.Series(moved).rolling(OI_BASE, min_periods=1).mean().to_numpy()
    seen = np.maximum.accumulate(np.isfinite(o) & (o > 0))
    has = seen & np.isfinite(o) & (o > 0) & (act >= OI_ACT)
    prev = np.r_[np.nan, o[:-1]]
    pair = np.isfinite(o) & (o > 0) & np.isfinite(prev) & (prev > 0)
    lg = np.where(pair, np.log(np.where(pair, o / np.where(pair, prev, 1.0), 1.0)), 0.0)
    share = np.where(pair, np.abs(o - prev) / np.where(pair, prev, 1.0), 0.0)
    sq = []
    z = np.zeros(T)
    roll = np.zeros(T, bool)
    for t in range(T):
        rms = np.sqrt(np.mean(sq)) if len(sq) >= 10 else np.nan
        z[t] = lg[t] / rms if np.isfinite(rms) and rms > 1e-12 else 0.0
        roll[t] = has[t] and pair[t] and (share[t] > OI_STRU or (abs(z[t]) > OI_ROLL and share[t] > OI_ROLL_ABS))
        if has[t] and pair[t] and not roll[t]:
            sq.append(lg[t] ** 2)
            if len(sq) > OI_BASE:
                sq.pop(0)
    dpx = close.diff().fillna(0.0).to_numpy()
    char = np.where(~has, "—", np.where(roll, "ROLL",
           np.where((z > OI_DEAD) & (dpx > 0), "LONG BUILDUP",
           np.where((z > OI_DEAD) & (dpx < 0), "SHORT BUILDUP",
           np.where((z < -OI_DEAD) & (dpx > 0), "SHORT COVERING",
           np.where((z < -OI_DEAD) & (dpx < 0), "LONG UNWINDING", "balanced"))))))
    rank = pd.Series(o).rolling(OI_PCT + 1, min_periods=OI_PCT + 1).apply(
        lambda w: 100.0 * (w[:-1] <= w[-1]).sum() / OI_PCT, raw=True).to_numpy()
    cnt = lambda name: pd.Series((char == name).astype(float)).rolling(sig_len, min_periods=1).sum().to_numpy()  # noqa: E731
    return pd.DataFrame({"oi_char": char, "oi_z": z, "oi_rank": np.where(has, rank, np.nan),
                         "oi_exit_up": cnt("SHORT COVERING"), "oi_build_up": cnt("LONG BUILDUP"),
                         "oi_exit_dn": cnt("LONG UNWINDING"), "oi_build_dn": cnt("SHORT BUILDUP")},
                        index=close.index)


# ════════════════════════════════════════════════════════════════════════════════════════
def compute_v5(lo: pd.DataFrame, out: pd.DataFrame, p: pg.Params, units: dict | None = None) -> pd.DataFrame:
    """Everything v5 reads differently, on the port's shared outputs."""
    ch = pg.chart_conviction(lo, p)
    cv = ch["osc"]
    sd_ok = ch["sd_ok"].to_numpy(bool)
    cv_ready = np.cumsum(sd_ok) > p.norm + p.smooth + p.signal
    s = signals(out, cv, p)
    d = divergence(lo, out, cv, cv_ready, p)
    g = grid(out, cv, ch["raw_sd"], cv_ready, p, units)
    return pd.concat([s, d, g], axis=1)
