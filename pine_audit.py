"""
pine_audit.py — a from-scratch audit of pragati.pine, measured on real data.

Nothing the indicator says about itself is assumed: not its evidence section, not the
meaning it gives a signal, not the direction it calls "buy". Every output of the Python
port (bar-for-bar with the Pine) is turned into a POSITION and scored against what
price actually did next, per instrument, on six asset classes.

Scoring
-------
For instrument i, bar t, horizon h:  fwd = Open[t+1+h] / Open[t+1] − 1   (enter next open)
    z = (fwd − μ_i) / σ_i            μ, σ of that instrument's h-bar returns inside the era
    score = p_t · z                  p_t ∈ [−1, 1] the position the output implies
The mean score over active bars is timing skill in σ of the h-bar return, with the
instrument's own drift removed (so holding a rising asset earns nothing). Positive means
the output's own sign was right; NEGATIVE means the output works the other way round.

Eras: discovery before ``SPLIT``, holdout from it. Intervals are block-bootstrapped over
dates (blocks of h) on per-date pooled sums — overlapping windows and same-day
cross-correlation are both respected.

Positions tested (the indicator's sign convention: + = up / long)
    trace, hist_z, conv, value, c_tape, v_tape         continuous, scaled into [−1, 1]
    turn, resume, div                                   events, ±1 on the bar they fire
    decl                                                the TURN declaration, held every bar
    grid_side, grid_units                               the 4 × 4 grid as a position
    mom20, rev20                                        naive benchmarks, no indicator

Usage (research; reads the cache written by the fetch step):
    python pine_audit.py baseline
"""
from __future__ import annotations

import os
import pickle
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import cvgrid as cg          # noqa: E402
import engine as eng         # noqa: E402
import pragati as pg         # noqa: E402
import samanvaya as sv       # noqa: E402

CACHE = os.environ.get("PINE_AUDIT_CACHE", "cache")
GROUPS = ("nse", "us", "idx", "cmd", "fx", "crypto")
SPLIT = pd.Timestamp("2018-01-01")
HORIZONS = (1, 5, 10, 20)
ERAS = ("disc", "hold")
#: Round-trip cost per unit of position change, bps — for the strategy view only.
COST_BPS = {"nse": 10.0, "us": 3.0, "idx": 3.0, "cmd": 5.0, "fx": 2.0, "crypto": 15.0}

STRATS = ("trace", "hist_z", "conv", "value", "c_tape", "v_tape", "turn", "resume", "div",
          "decl", "grid_side", "grid_units", "mom20", "rev20")
CONT = {"trace", "hist_z", "conv", "value", "c_tape", "v_tape", "decl", "grid_side",
        "grid_units", "mom20", "rev20"}


# ════════════════════════════════════════════════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════════════════════════════════════════════════
def load_group(g: str) -> dict:
    return pickle.load(open(os.path.join(CACHE, f"multi_{g}.pkl"), "rb"))


def load_drivers():
    raw = pickle.load(open(os.path.join(CACHE, "drivers_raw20.pkl"), "rb"))
    return sv.prepare_drivers(raw, "D")


def calendar() -> pd.DatetimeIndex:
    idx = set()
    for g in GROUPS:
        for f in load_group(g).values():
            idx |= set(f.index)
    return pd.DatetimeIndex(sorted(idx))


# ════════════════════════════════════════════════════════════════════════════════════════
# POSITIONS
# ════════════════════════════════════════════════════════════════════════════════════════
def positions(out: pd.DataFrame, grid: pd.DataFrame, close: pd.Series) -> dict:
    """{strategy: position array} from one run of the port. NaN = inactive."""
    ok = out["stack_ok"].fillna(False).astype(bool).to_numpy()
    ready = out["trace_ok"].fillna(False).astype(bool).to_numpy() & np.isfinite(out["trace"].to_numpy())

    def cont(x, scale):
        v = np.clip(np.asarray(x, dtype=float) / scale, -1.0, 1.0)
        return np.where(ready & np.isfinite(v), v, np.nan)

    def ev(up, dn):
        u = out[up].fillna(False).astype(bool).to_numpy()
        d = out[dn].fillna(False).astype(bool).to_numpy()
        return np.where(u, 1.0, np.where(d, -1.0, np.nan))

    decl = out["decl"].fillna(0).to_numpy(dtype=float)
    cell = grid["cvg_cell"].fillna(cg.UNREAD).astype(int).to_numpy()
    side = np.array([cg.SIDES[c] for c in cell], dtype=float)
    units = np.array([cg.UNITS[c] for c in cell], dtype=float)
    read = cell != cg.UNREAD
    r20 = (close / close.shift(20) - 1.0).to_numpy()
    s20 = np.sign(r20)
    return {
        "trace": cont(out["trace"], 100.0),
        "hist_z": cont(out["hist_z"], 3.0),
        "conv": cont(out["conv"], 100.0),
        "value": cont(out["value"], 100.0),
        "c_tape": cont(out["c_tape"], 100.0),
        "v_tape": cont(out["v_tape"], 100.0),
        "turn": ev("turn_buy", "turn_sell"),
        "resume": ev("resume_long", "resume_short"),
        "div": ev("bull_div", "bear_div"),
        "decl": np.where(ok & (decl != 0), decl, np.nan),
        "grid_side": np.where(read & (side != 0), side, np.nan),
        # units centred on Wait (1u) and scaled so Buy / Add (3u) = +1, Exit (¼u) = −⅜
        "grid_units": np.where(read, (units - 1.0) / 2.0, np.nan),
        "mom20": np.where(ready, s20, np.nan),
        "rev20": np.where(ready, -s20, np.nan),
    }


def run_port(df: pd.DataFrame, val: pd.DataFrame, params: pg.Params):
    lo = eng._chart_bars(df)
    out = pg.compute(lo, val, params, chart="D")
    grid = cg.classify(out["c_tape"], out["v_tape"], out["push"], out["hist_ready"],
                       out["c_ready"] & out["v_ready"], out["conv"], out["value"],
                       out["conv_ready"] & out["value_built"], params.z1, params.theta,
                       index=out.index)
    return lo, out, grid


# ════════════════════════════════════════════════════════════════════════════════════════
# PER-INSTRUMENT SCORING  (returns per-date sums on the global calendar)
# ════════════════════════════════════════════════════════════════════════════════════════
def score_instrument(lo: pd.DataFrame, pos: dict, cal_pos: np.ndarray, n_cal: int,
                     strats=STRATS, horizons=HORIZONS, by_date: bool = True) -> dict:
    """{(strat, h, era): (sum_pz, sum_|p|, n, sum_pz², [per-date sum_pz, per-date sum_|p|])}."""
    op = lo["open"].where(lo["open"] > 0, lo["close"]).to_numpy(dtype=float)
    dates = lo.index
    era_mask = {"disc": np.asarray(dates < SPLIT), "hold": np.asarray(dates >= SPLIT)}
    res = {}
    for h in horizons:
        entry = np.roll(op, -1)
        exit_ = np.roll(op, -1 - h)
        fwd = exit_ / entry - 1.0
        fwd[-(h + 1):] = np.nan
        for era, m in era_mask.items():
            w = fwd[m]
            w = w[np.isfinite(w)]
            if w.size < 100:
                continue
            mu, sd = float(w.mean()), float(w.std(ddof=1))
            if not np.isfinite(sd) or sd <= 0:
                continue
            z = np.where(m, (fwd - mu) / sd, np.nan)
            for s in strats:
                p = pos[s]
                act = np.isfinite(p) & np.isfinite(z) & (p != 0)
                if not act.any():
                    continue
                pz = p[act] * z[act]
                ap = np.abs(p[act])
                item = [float(pz.sum()), float(ap.sum()), int(act.sum()), float((pz * pz).sum())]
                if by_date:
                    ci = cal_pos[act]
                    item.append(np.bincount(ci, weights=pz, minlength=n_cal))
                    item.append(np.bincount(ci, weights=ap, minlength=n_cal))
                res[(s, h, era)] = item
    return res


def strategy_returns(lo: pd.DataFrame, p: np.ndarray, cost_bps: float) -> pd.Series:
    """Daily open-to-open return of holding position p_t from the next open, net of costs."""
    op = lo["open"].where(lo["open"] > 0, lo["close"])
    r = (op.shift(-2) / op.shift(-1) - 1.0).to_numpy()
    pp = np.nan_to_num(p)
    turn = np.abs(np.diff(np.concatenate([[0.0], pp])))
    net = pp * r - turn * cost_bps / 1e4
    return pd.Series(net, index=lo.index)


# ════════════════════════════════════════════════════════════════════════════════════════
# WORKERS
# ════════════════════════════════════════════════════════════════════════════════════════
_W = {}


def _init(cal):
    _W["cal"] = cal
    _W["drv"] = load_drivers()


def _baseline_job(args):
    g, tkr, df, params = args
    cal = _W["cal"]
    try:
        lo = eng._chart_bars(df)
        val = sv.compute_value(lo, _W["drv"], tkr, chart="D")
        lo, out, grid = run_port(df, val, params)
    except Exception as e:                              # a bad series never ends the audit
        return g, tkr, None, f"{type(e).__name__}: {e}"
    pos = positions(out, grid, lo["close"])
    cal_pos = cal.get_indexer(lo.index)
    sc = score_instrument(lo, pos, cal_pos, len(cal))
    # the strategy view for the standing positions
    strat = {}
    for s in ("decl", "grid_side", "grid_units", "mom20", "rev20"):
        strat[s] = strategy_returns(lo, pos[s], COST_BPS[g])
    strat["hold"] = strategy_returns(lo, np.ones(len(lo)), 0.0)
    return g, tkr, (sc, strat), None


# ════════════════════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ════════════════════════════════════════════════════════════════════════════════════════
def boot_ci(sum_pz: np.ndarray, sum_ap: np.ndarray, block: int, n_boot: int = 1000,
            seed: int = 7) -> tuple:
    keep = sum_ap > 0
    s, a = sum_pz[keep], sum_ap[keep]
    n = s.size
    block = max(int(block), 1)
    nb = n // block
    if nb < 4:
        return (np.nan, np.nan)
    bs = s[: nb * block].reshape(nb, block).sum(1)
    ba = a[: nb * block].reshape(nb, block).sum(1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nb, size=(n_boot, nb))
    means = bs[idx].sum(1) / ba[idx].sum(1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


class Accumulator:
    """Pools per-instrument scores into per-group, per-date sums as they arrive."""

    def __init__(self, n_cal: int):
        self.n_cal = n_cal
        self.acc = {}

    def add(self, g: str, sc: dict):
        for (s, h, era), item in sc.items():
            k = (g, s, h, era)
            a = self.acc.get(k)
            if a is None:
                a = self.acc[k] = [0.0, 0.0, 0, 0.0, np.zeros(self.n_cal), np.zeros(self.n_cal), 0]
            a[0] += item[0]; a[1] += item[1]; a[2] += item[2]; a[3] += item[3]
            if len(item) > 4:
                a[4] += item[4]; a[5] += item[5]
            a[6] += 1

    def table(self, ci: bool = True) -> pd.DataFrame:
        rows = []
        for (g, s, h, era), a in self.acc.items():
            edge = a[0] / a[1] if a[1] > 0 else np.nan
            lo_, hi_ = boot_ci(a[4], a[5], block=max(h, 5)) if ci else (np.nan, np.nan)
            rows.append(dict(group=g, strat=s, h=h, era=era, edge=edge, lo=lo_, hi=hi_,
                             n=a[2], names=a[6]))
        return pd.DataFrame(rows)


def main_baseline(params: pg.Params | None = None, tag: str = "baseline"):
    params = params or eng.settings_for(None, None, "Daily").params
    cal = calendar()
    jobs = [(g, t, df, params) for g in GROUPS for t, df in load_group(g).items()]
    accum, strat_rets, errs, n_ok = Accumulator(len(cal)), {}, [], 0
    with Pool(4, initializer=_init, initargs=(cal,)) as pool:
        for i, (g, tkr, payload, err) in enumerate(pool.imap_unordered(_baseline_job, jobs, chunksize=4)):
            if err:
                errs.append((g, tkr, err)); continue
            sc, strat = payload
            accum.add(g, sc)
            n_ok += 1
            strat_rets[(g, tkr)] = strat
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    tab = accum.table()
    pickle.dump({"table": tab, "strat": strat_rets, "errors": errs},
                open(os.path.join(CACHE, f"audit_{tag}.pkl"), "wb"))
    print(f"done · {n_ok} instruments · {len(errs)} errors")
    return tab


if __name__ == "__main__":
    if sys.argv[1:2] == ["baseline"]:
        main_baseline()


# ════════════════════════════════════════════════════════════════════════════════════════
# PARAMETER SWEEP  (does a set's discovery score predict its holdout score?)
# ════════════════════════════════════════════════════════════════════════════════════════
from dataclasses import dataclass as _dc, field as _field   # noqa: E402


@_dc(frozen=True)
class TParams(pg.Params):
    """Params with θ as a field, so the sweep can move it."""
    theta_: float = _field(default=sv.THETA_OSC)

    @property
    def theta(self) -> float:          # noqa: D401
        return float(self.theta_)


SWEEP_STRATS = ("trace", "c_tape", "v_tape", "hist_z", "turn", "resume", "div", "decl",
                "grid_side", "grid_units")
SWEEP_H = (1, 10)


def random_params(n: int, seed: int = 11) -> list:
    rng = np.random.default_rng(seed)
    out = [TParams()]
    for _ in range(n):
        out.append(TParams(
            length=int(rng.choice([8, 10, 14, 20, 30, 40, 60])),
            smooth=int(rng.choice([1, 2, 3, 5, 8])),
            norm=int(rng.choice([100, 150, 200, 300, 500])),
            participation=str(rng.choice(["Auto", "Volume", "True range", "Off"])),
            denominator=str(rng.choice(["Agreement", "Effort"])),
            z1=float(rng.choice([20.0, 30.0, 40.0, 50.0])),
            mix=float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0])),
            signal=int(rng.choice([3, 5, 9, 14, 21])),
            confirm=int(rng.choice([2, 3, 5, 8, 12])),
            disloc=int(rng.choice([10, 20, 40, 60])),
            k=float(rng.choice([0.0, 0.25, 0.5, 1.0, 1.5])),
            pull=int(rng.choice([3, 6, 10])),
            effort=bool(rng.choice([True, False])),
            cool=int(rng.choice([0, 5, 10, 20])),
            pl=int(rng.choice([3, 5, 8])), pr=int(rng.choice([3, 5, 8])),
            theta_=float(rng.choice([25.0, 35.0, sv.THETA_OSC, 55.0, 65.0])),
        ))
    return out


def _sweep_job(args):
    g, tkr, df, plist = args
    cal = _W["cal"]
    try:
        lo0 = eng._chart_bars(df)
        val = sv.compute_value(lo0, _W["drv"], tkr, chart="D")
    except Exception as e:
        return g, tkr, None, f"value {type(e).__name__}: {e}"
    res = []
    for p in plist:
        try:
            lo, out, grid = run_port(df, val, p)
            pos = positions(out, grid, lo["close"])
            res.append(score_instrument(lo, pos, cal.get_indexer(lo.index), len(cal),
                                        strats=SWEEP_STRATS, horizons=SWEEP_H, by_date=False))
        except Exception:
            res.append({})
    return g, tkr, res, None


def main_sweep(n: int = 160, tag: str = "sweep"):
    cal = calendar()
    plist = random_params(n)
    jobs = [(g, t, df, plist) for g in GROUPS for t, df in load_group(g).items()]
    # acc[k][(g, s, h, era)] = [sum_pz, sum_ap, n, sum_pz2]
    acc = [dict() for _ in plist]
    errs = []
    with Pool(4, initializer=_init, initargs=(cal,)) as pool:
        for i, (g, tkr, res, err) in enumerate(pool.imap_unordered(_sweep_job, jobs, chunksize=1)):
            if err:
                errs.append((g, tkr, err)); continue
            for k, sc in enumerate(res):
                for (s, h, era), item in sc.items():
                    a = acc[k].setdefault((g, s, h, era), [0.0, 0.0, 0, 0.0])
                    a[0] += item[0]; a[1] += item[1]; a[2] += item[2]; a[3] += item[3]
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    rows = []
    for k, d in enumerate(acc):
        for (g, s, h, era), a in d.items():
            rows.append(dict(pset=k, group=g, strat=s, h=h, era=era,
                             edge=a[0] / a[1] if a[1] > 0 else np.nan, n=a[2]))
    tab = pd.DataFrame(rows)
    pickle.dump({"table": tab, "params": plist, "errors": errs},
                open(os.path.join(CACHE, f"audit_{tag}.pkl"), "wb"))
    print(f"done · {len(plist)} parameter sets · {len(errs)} errors")
    return tab


if __name__ == "__main__" and sys.argv[1:2] == ["sweep"]:
    main_sweep(int(sys.argv[2]) if len(sys.argv) > 2 else 160)


# ════════════════════════════════════════════════════════════════════════════════════════
# STRUCTURAL EXPERIMENTS  (E1 TURN ablations · E2 adaptive sign · E3 the grid per cell)
# ════════════════════════════════════════════════════════════════════════════════════════
TURN_VARIANTS = {
    "turn_all":     {},
    "turn_no_val":  {"gate_value": False},
    "turn_no_conv": {"gate_conv": False},
    "turn_no_push": {"gate_push": False},
    "turn_no_fail": {"effort": False},
    "turn_cross":   {"gate_value": False, "gate_conv": False, "gate_push": False, "effort": False},
}
ADAPT_W = 500           # bars of trailing evidence the adaptive sign reads


def adaptive_sign(x: pd.Series, lo: pd.DataFrame, h: int, w: int = ADAPT_W) -> np.ndarray:
    """Sign of the trailing correlation between x and its own completed h-bar outcomes.

    Causal: at bar t it only uses pairs (x_s, fwd_s) whose forward window closed by t —
    s ≤ t − h − 1. A market where the reading has lately pointed the right way keeps its
    sign; one where it has pointed the wrong way has it flipped.
    """
    op = lo["open"].where(lo["open"] > 0, lo["close"])
    fwd = op.shift(-1 - h) / op.shift(-1) - 1.0
    xs, ys = x.shift(h + 1), fwd.shift(h + 1)
    rho = xs.rolling(w, min_periods=w // 2).corr(ys)
    return np.sign(rho).to_numpy()


def _exp_job(args):
    g, tkr, df = args
    cal = _W["cal"]
    try:
        lo0 = eng._chart_bars(df)
        val = sv.compute_value(lo0, _W["drv"], tkr, chart="D")
    except Exception as e:
        return g, tkr, None, f"{type(e).__name__}: {e}"
    base = eng.settings_for(None, None, "Daily").params
    pos, cells = {}, None
    lo = None
    for name, over in TURN_VARIANTS.items():
        lo, out, grid = run_port(df, val, replace_params(base, over))
        u = out["turn_buy"].fillna(False).to_numpy(bool)
        d = out["turn_sell"].fillna(False).to_numpy(bool)
        pos[name] = np.where(u, 1.0, np.where(d, -1.0, np.nan))
        if name == "turn_all":
            full = positions(out, grid, lo["close"])
            cells = grid["cvg_cell"].fillna(cg.UNREAD).astype(int).to_numpy()
            tr = out["trace"]
            pos["trace"] = full["trace"]
            pos["neg_trace"] = -full["trace"]
            for h in (5, 10):
                sgn = adaptive_sign(tr, lo, h)
                pos[f"adapt_trace_h{h}"] = np.where(np.isfinite(sgn) & (sgn != 0), sgn * full["trace"], np.nan)
            m20 = full["mom20"]
            sgn = adaptive_sign(pd.Series(m20, index=lo.index), lo, 10)
            pos["adapt_mom20"] = np.where(np.isfinite(sgn) & (sgn != 0), sgn * m20, np.nan)
            pos["mom20"] = m20
    # E3 · the grid, one indicator position per cell (+1 on bars in that cell)
    for k in range(16):
        pos[f"cell{k:02d}"] = np.where(cells == k, 1.0, np.nan)
    sc = score_instrument(lo, pos, cal.get_indexer(lo.index), len(cal),
                          strats=tuple(pos), horizons=(1, 5, 10, 20))
    return g, tkr, sc, None


def replace_params(p, over: dict):
    from dataclasses import replace as _r
    return _r(p, **over) if over else p


def main_experiments(tag: str = "exp"):
    cal = calendar()
    jobs = [(g, t, df) for g in GROUPS for t, df in load_group(g).items()]
    accum, errs = Accumulator(len(cal)), []
    with Pool(4, initializer=_init, initargs=(cal,)) as pool:
        for i, (g, tkr, sc, err) in enumerate(pool.imap_unordered(_exp_job, jobs, chunksize=2)):
            if err:
                errs.append((g, tkr, err)); continue
            accum.add(g, sc)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    tab = accum.table()
    pickle.dump({"table": tab, "errors": errs}, open(os.path.join(CACHE, f"audit_{tag}.pkl"), "wb"))
    print(f"done · {len(jobs) - len(errs)} instruments · {len(errs)} errors")
    return tab


if __name__ == "__main__" and sys.argv[1:2] == ["experiments"]:
    main_experiments()
