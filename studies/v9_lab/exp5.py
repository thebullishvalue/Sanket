import pickle, numpy as np, pandas as pd, lab_core as L, warnings; warnings.filterwarnings("ignore")
P = pickle.load(open("lab_P.pkl", "rb")); cal = L.calendar(P)
for h in (5, 10, 20, 40): L.xs_demean(P, h)
def evt(cond): return lambda f: np.where(f.live, np.where(cond(f).fillna(False), 1.0, 0.0), np.nan)
def evt2(up, dn): return lambda f: np.where(f.live, np.where(up(f).fillna(False), 1.0, np.where(dn(f).fillna(False), -1.0, 0.0)), np.nan)
c = lambda f: f.cvg_cell
enter = lambda f, k: (c(f) == k) & (c(f).shift(1) != k)
rev = lambda f: (c(f) == 0) & (f.cvg_vph == 1) & ~((c(f).shift(1) == 0) & (f.cvg_vph.shift(1) == 1))
cap_recent = lambda f: (c(f) == 0).astype(float).rolling(20, min_periods=1).max().astype(bool)
S = {
 "EVENT enter capitulation": evt(lambda f: enter(f, 0)),
 "EVENT capitulation turns (value reverting in cell 0)": evt(rev),
 "EVENT enter distribution (short)": lambda f: np.where(f.live, np.where(enter(f, 2), -1.0, 0.0), np.nan),
 "EVENT cap-turn long / distribution-entry short": evt2(rev, lambda f: enter(f, 2)),
 "▲ TURN buy only": evt(lambda f: f.turn_buy.astype(bool)),
 "▲ TURN within 20 bars of capitulation": evt(lambda f: f.turn_buy.astype(bool) & cap_recent(f)),
 "▲ TURN not after capitulation": evt(lambda f: f.turn_buy.astype(bool) & ~cap_recent(f)),
 "STATE capitulation": evt(lambda f: c(f) == 0),
 "STATE capitulation reverting": evt(lambda f: (c(f) == 0) & (f.cvg_vph == 1)),
}
rows = []
for h in (5, 10, 20, 40):
    for name, fn in S.items():
        for mode in ("tc", "xs"):
            t = L.score_c(P, fn, h=h, mode=mode, cal=cal)
            e = t.pivot(index="group", columns="era", values="edge")
            r = dict(h=h, sig=name, mode=mode)
            for er in ("E1", "E2", "E3"): r[er] = round(e.loc[L.NC, er].mean(), 3)
            r["cls+"] = "/".join(str(int((e.loc[L.NC, er] > 0).sum())) for er in ("E1", "E2", "E3"))
            r["sig+"] = int(((t.lo > 0) & t.group.isin(L.NC)).sum()); r["sig−"] = int(((t.hi < 0) & t.group.isin(L.NC)).sum())
            r["crypto E3"] = round(e.loc["crypto", "E3"], 3)
            rows.append(r)
R = pd.DataFrame(rows); R.to_pickle("exp5.pkl")
# event counts
cnt = {}
for name, fn in S.items():
    if name.startswith("EVENT") or "TURN" in name:
        n = sum(int(np.nansum(np.abs(fn(f)))) for g in L.NC for f in P[g].values())
        cnt[name] = n
print("event counts (non-crypto, all eras):", cnt)
for mode in ("tc", "xs"):
    print(f"\n==== mode={mode}")
    X = R[R["mode"] == mode]
    print(X.pivot_table(index="sig", columns="h", values=["E1", "E2", "E3"], aggfunc="first").reindex(list(S)).round(3).to_string())
    print(X[X.h.isin([10, 20])].pivot_table(index="sig", columns="h", values=["cls+", "sig+", "sig−"], aggfunc="first").reindex(list(S)).to_string())
