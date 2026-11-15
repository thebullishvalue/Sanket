"""v13 lab - the same-date peer-group mean of the 10-bar forward return, per dataset (for the cross-sectional scorer).

    python3 studies/v13_lab/xs.py <Indicator/research> RD|RH|FD|FH [fresh daily cache] [fresh hourly cache]

Peer group = the instrument's class (us, nse, index, cmdty, fx, bond, crypto); where fewer than 8
class members have a forward return on that date, the whole dataset's mean stands in.
Writes out/xs_<ds>.pkl: index (date, cls) -> peer mean.
"""
import os, sys, glob
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH, DS = os.path.abspath(sys.argv[1]), sys.argv[2]
FD, FH = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else (None, None)
H = 10


def load():
    if DS in ("RD", "RH"):
        sys.path.insert(0, RESEARCH); os.chdir(RESEARCH)
        from lib import load as lload
        return lload("data1h" if DS == "RH" else "data", 2500 if DS == "RH" else 1500)
    out = {}
    for fn in sorted(glob.glob(os.path.join(FH if DS == "FH" else FD, "*.csv"))):
        m, t = os.path.basename(fn)[:-4].split("__", 1)
        d = pd.read_csv(fn, index_col=0, parse_dates=True)
        d = d[(d.close > 0) & (d.high >= d.low)]
        d = d[d.close.pct_change().abs().fillna(0) < 0.6]
        if len(d) >= 1500:
            out[t] = (m, d)
    return out


rows = []
for sym, (cls, d) in load().items():
    c = d.close[d.close > 0]
    fwd = (c.shift(-1 - H) / c.shift(-1) - 1.0).dropna()
    rows.append(pd.DataFrame({"date": fwd.index, "cls": cls, "fwd": fwd.to_numpy()}))
A = pd.concat(rows, ignore_index=True)
g = A.groupby(["date", "cls"]).fwd.agg(["mean", "size"])
allm = A.groupby("date").fwd.mean()
peer = np.where(g["size"] >= 8, g["mean"], allm.reindex(g.index.get_level_values(0)).to_numpy())
pd.Series(peer, index=g.index, name="xs").to_pickle(os.path.join(HERE, "out", f"xs_{DS}.pkl"))
print(DS, len(g), "date x class cells")
