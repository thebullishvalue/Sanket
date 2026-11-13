"""Does the Watchlist mean what it says? On the Indicator repo's 151-instrument data.

    python3 studies/v10_watch/watch_check.py <Indicator/research> [data|data1h]

For every bar a name would sit on the Watchlist (ready B·C / ready B / near / in A ±, each side), how
often does the event it promises actually fire over the next 1 / 5 / 10 / 20 bars, against the
same rate for names NOT on the list - and, armed (the column still on the far side of zero) vs
not. Then coverage: of the events that fired, how many were on the list the bar before.
Signals from ledger.py - the exact code the screen runs.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
RESEARCH = os.path.abspath(sys.argv[1])
sys.path.insert(0, RESEARCH)
import numpy as np, pandas as pd
import ledger
os.chdir(RESEARCH)
from lib import load                                          # noqa: E402
root = sys.argv[2] if len(sys.argv) > 2 else "data"
H, KS = 10, (1, 5, 10, 20)


def fwd_any(x, k):
    """True at t when x fires on any of t+1 … t+k."""
    s = pd.Series(x.astype(float))
    return (s[::-1].rolling(k, min_periods=1).max()[::-1].shift(-1).fillna(0) > 0).to_numpy()


rows, cover = [], []
D = load(root, 1500 if root == "data" else 2500)
for sym, (cls, d) in D.items():
    lo = d[["open", "high", "low", "close"]]
    lo = lo[lo.close.notna() & lo.high.notna() & lo.low.notna() & (lo.close > 0)]
    L = ledger.compute(lo, H)
    u = L.lg_u.to_numpy(float)
    live = np.isfinite(L.lg_prk.to_numpy(float)) & np.isfinite(L.lg_ladx.to_numpy(float))
    for side, sg in (("buy", 1), ("sell", -1)):
        b, c, a = (L[f"{k}_{side}"].to_numpy(bool) for k in "bca")
        rb, rc = L[f"ready_b_{side}"].to_numpy(bool), L[f"ready_c_{side}"].to_numpy(bool)
        nr = L[f"near_b_{side}"].to_numpy(bool)
        pv = (L.lg_proven.to_numpy() * sg) > 0
        armed = (u < 0) if sg > 0 else (u >= 0)
        groups = {"ready B·C": rc, "ready B (not C)": rb & ~rc, "near": nr & ~rb, "in A": pv & ~rb & ~nr,
                  "not on the list": live & ~rb & ~nr & ~pv}
        fut = {k: (fwd_any(b, k), fwd_any(c, k), fwd_any(a, k)) for k in KS}
        for g, m in groups.items():
            for arm_lbl, am in (("all", np.ones_like(m)), ("armed", armed), ("not armed", ~armed)):
                mm = m & am & live
                if g != "in A" and arm_lbl != "all":
                    continue
                if not mm.any():
                    continue
                r = dict(side=side, group=g, armed=arm_lbl, bars=int(mm.sum()))
                for k in KS:
                    fb, fc, fa = fut[k]
                    r[f"B in {k}"] = fb[mm].sum(); r[f"C in {k}"] = fc[mm].sum(); r[f"A in {k}"] = fa[mm].sum()
                rows.append(r)
        # coverage: was the name on the list on the bar before it fired?
        prev = lambda x: np.r_[False, x[:-1]]
        for k, ev, rdy in (("B", b, rb), ("C", c, rc)):
            cover.append(dict(side=side, set=k, events=int(ev.sum()), ready_before=int((ev & prev(rdy)).sum()),
                              ready_or_near_before=int((ev & prev(rdy | nr)).sum())))

R = pd.DataFrame(rows).groupby(["side", "group", "armed"], sort=False).sum().reset_index()
out = R[["side", "group", "armed", "bars"]].copy()
for k in KS:
    for s in "BCA":
        out[f"{s}≤{k}"] = (R[f"{s} in {k}"] / R.bars * 100).round(1)
pd.set_option("display.width", 250)
print(f"{root} · {len(D)} instruments · % of bars on which the event fires within the next k bars\n")
for side in ("buy", "sell"):
    print(("▲ LONG" if side == "buy" else "▼ SHORT") + " WATCH")
    print(out[out.side == side].drop(columns="side")[["group", "armed", "bars"] + [f"{s}≤{k}" for s in "BCA" for k in KS]]
          .to_string(index=False))
    print()
C = pd.DataFrame(cover).groupby(["side", "set"]).sum()
C["% ready bar before"] = (C.ready_before / C.events * 100).round(1)
C["% ready or near"] = (C.ready_or_near_before / C.events * 100).round(1)
print("COVERAGE · of the events that fired, how many were on the watchlist the bar before\n" + C.to_string())
