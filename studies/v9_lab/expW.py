import pickle, glob, os, numpy as np, pandas as pd, lab_core as L, warnings; warnings.filterwarnings("ignore")
HS = (2, 4, 8)
P = {}
for g in L.GROUPS:
    P[g] = {}
    for fn in sorted(glob.glob(f"{L.SP}/featw/{g}__*.pkl")):
        f = pd.read_pickle(fn); f["era"] = L.era_of(f.index)
        f["live"] = f["stack_ok"].fillna(False).astype(bool) & f["cv_ready"].fillna(False).astype(bool)
        L.add_u(f, HS)
        lr = np.log(f.open.where(f.open > 0))
        for h in HS: f[f"r{h}"] = lr.shift(-(1 + h)) - lr.shift(-1)
        P[g][os.path.basename(fn)] = f
# the add_u vol uses 60 daily bars; on weekly that's 60 weeks — fine (causal)
for h in HS: L.xs_demean(P, h)
cal = L.calendar(P)
c = lambda f: f.cvg_cell
def cool(ev, n):
    ev = np.asarray(ev, bool); out = np.zeros(len(ev), bool); last = -10**9
    for i in np.flatnonzero(ev):
        if i - last >= n: out[i] = True; last = i
    return out
def first(x): x = x.fillna(False).astype(bool); return x & ~x.shift(1, fill_value=False)
def levt(cond, n=3): return lambda f: np.where(f.live, np.where(cool(cond(f).to_numpy(bool), n), 1.0, 0.0), np.nan)
def ev(u, d): return lambda f: np.where(f.live, np.where(f[u].fillna(False), 1.0, np.where(f[d].fillna(False), -1.0, 0.0)), np.nan)
st = lambda cond: (lambda f: np.where(f.live, np.where(cond(f), 1.0, 0.0), np.nan))
S = {
 "grid units": lambda f: np.where(f.live & (c(f) < 9), (f.cvg_units - 1) / 2, np.nan),
 "STATE capitulation": st(lambda f: c(f) == 0),
 "STATE distribution (short)": lambda f: np.where(f.live, np.where(c(f) == 2, -1.0, 0.0), np.nan),
 "EVENT cap-turn (cooldown 3w)": levt(lambda f: first((c(f) == 0) & (f.cvg_vph == 1))),
 "▲▼ TURN (v8)": ev("turn_buy", "turn_sell"),
 "▲ TURN only": st(lambda f: f.turn_buy.astype(bool)),
 "R divergence": ev("r_bull", "r_bear"),
 "declaration": lambda f: np.where(f.live, f.v5_decl.astype(float), np.nan),
 "BENCH mom20z": lambda f: np.where(f.live, np.clip(np.log(f.close / f.close.shift(4)) / (np.log(f.close).diff().rolling(52).std() * 2) / 2, -1, 1), np.nan),
}
rows = []
for h in HS:
    for name, fn in S.items():
        for mode in ("tc", "xs"):
            t = L.score_c(P, fn, h=h, mode=mode, cal=cal)
            e = t.pivot(index="group", columns="era", values="edge")
            r = dict(h=h, sig=name, mode=mode)
            for er in ("E1", "E2", "E3"): r[er] = round(e.loc[L.NC, er].mean(), 3)
            r["cls+"] = "/".join(str(int((e.loc[L.NC, er] > 0).sum())) for er in ("E1", "E2", "E3"))
            r["sig+/−"] = f"{int(((t.lo > 0) & t.group.isin(L.NC)).sum())}/{int(((t.hi < 0) & t.group.isin(L.NC)).sum())}"
            r["n"] = int(t[t.group.isin(L.NC)].n.sum())
            rows.append(r)
R = pd.DataFrame(rows)
cnt = {k: sum(int(np.nansum(np.abs(S[k](f)))) for g in L.NC for f in P[g].values()) for k in ("EVENT cap-turn (cooldown 3w)", "▲▼ TURN (v8)", "R divergence")}
print("WEEKLY bars. horizons in weeks. counts:", cnt)
live = {e: sum(int((f.live & (f.era == e)).sum()) for g in L.NC for f in P[g].values()) for e in ("E1", "E2", "E3")}
print("live name-weeks by era:", live)
for mode in ("tc", "xs"):
    print(f"\n==== {mode}"); print(R[R["mode"] == mode].drop(columns="mode").set_index(["h", "sig"]).to_string())
