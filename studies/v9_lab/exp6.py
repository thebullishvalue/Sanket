import pickle, numpy as np, pandas as pd, lab_core as L, warnings; warnings.filterwarnings("ignore")
P = pickle.load(open("lab_P.pkl", "rb")); cal = L.calendar(P)
for h in (10, 20, 40): L.xs_demean(P, h)
c = lambda f: f.cvg_cell
def cool(ev, n=10):
    ev = ev.fillna(False).to_numpy(bool); out = np.zeros(len(ev), bool); last = -10**9
    for i in np.flatnonzero(ev):
        if i - last >= n: out[i] = True; last = i
    return out
def sevt(cond): return lambda f: np.where(f.live, np.where(np.asarray(cond(f), bool), -1.0, 0.0), np.nan)
def levt(cond): return lambda f: np.where(f.live, np.where(np.asarray(cond(f), bool), 1.0, 0.0), np.nan)
rich_recent = lambda f: c(f).isin([5, 8]).astype(float).rolling(20, min_periods=1).max().astype(bool)
rev_cap = lambda f: (c(f) == 0) & (f.cvg_vph == 1) & ~((c(f).shift(1) == 0) & (f.cvg_vph.shift(1) == 1))
S = {
 "▼ TURN sell only (short)": sevt(lambda f: f.turn_sell.astype(bool)),
 "▼ TURN after UP/FAINT·rich (short)": sevt(lambda f: f.turn_sell.astype(bool) & rich_recent(f)),
 "short: paid turns (cell 8, value reverting)": sevt(lambda f: (c(f) == 8) & (f.cvg_vph == 1) & ~((c(f).shift(1) == 8) & (f.cvg_vph.shift(1) == 1))),
 "short: stalling turns (cell 5, value reverting)": sevt(lambda f: (c(f) == 5) & (f.cvg_vph == 1) & ~((c(f).shift(1) == 5) & (f.cvg_vph.shift(1) == 1))),
 "short: leave UP·rich to lower row": sevt(lambda f: (c(f).shift(1) == 8) & c(f).isin([5, 2])),
 "short STATE distribution": sevt(lambda f: c(f) == 2),
 "short STATE paid": sevt(lambda f: c(f) == 8),
 "long: cap-turn, 10-bar cooldown": levt(lambda f: cool(rev_cap(f), 10)),
 "long: cap-turn, 20-bar cooldown": levt(lambda f: cool(rev_cap(f), 20)),
}
rows = []
for h in (10, 20, 40):
    for name, fn in S.items():
        for mode in ("tc", "xs"):
            t = L.score_c(P, fn, h=h, mode=mode, cal=cal)
            e = t.pivot(index="group", columns="era", values="edge")
            r = dict(h=h, sig=name, mode=mode)
            for er in ("E1", "E2", "E3"): r[er] = round(e.loc[L.NC, er].mean(), 3)
            r["cls+"] = "/".join(str(int((e.loc[L.NC, er] > 0).sum())) for er in ("E1", "E2", "E3"))
            r["sig+"] = int(((t.lo > 0) & t.group.isin(L.NC)).sum()); r["sig−"] = int(((t.hi < 0) & t.group.isin(L.NC)).sum())
            rows.append(r)
R = pd.DataFrame(rows)
cnt = {name: sum(int(np.nansum(np.abs(fn(f)))) for g in L.NC for f in P[g].values()) for name, fn in S.items() if "STATE" not in name}
print("counts:", cnt)
for mode in ("tc", "xs"):
    X = R[R["mode"] == mode]
    print(f"\n==== {mode}"); print(X.pivot_table(index="sig", columns="h", values=["E1", "E2", "E3", "cls+", "sig+", "sig−"], aggfunc="first").reindex(list(S)).to_string())
