import pickle, numpy as np, pandas as pd, lab_core as L, warnings; warnings.filterwarnings("ignore")
P = pickle.load(open("lab_P.pkl", "rb")); cal = L.calendar(P)
for h in (10, 20): L.xs_demean(P, h)
c = lambda f: f.cvg_cell
def cool(ev, n=10):
    ev = np.asarray(ev, bool); out = np.zeros(len(ev), bool); last = -10**9
    for i in np.flatnonzero(ev):
        if i - last >= n: out[i] = True; last = i
    return out
def first(cond):
    x = cond.fillna(False).astype(bool); return x & ~x.shift(1, fill_value=False)
def levt(cond): return lambda f: np.where(f.live, np.where(cool(cond(f).fillna(False).to_numpy(bool) if hasattr(cond(f), "fillna") else cond(f), 10), 1.0, 0.0), np.nan)
capR = lambda f: (c(f) == 0) & (f.cvg_vph == 1)
S = {
 "A cap-turn (base)": levt(lambda f: first(capR(f))),
 "B cap-turn ∧ trace hist>0": levt(lambda f: first(capR(f) & (f["hist"] > 0))),
 "C cap-turn ∧ conviction push>0": levt(lambda f: first(capR(f) & (f.cvg_push > 0))),
 "D cap-turn ∧ c_tape rising 2 bars": levt(lambda f: first(capR(f) & (f.c_tape > f.c_tape.shift(1)) & (f.c_tape.shift(1) > f.c_tape.shift(2)))),
 "E cap-turn ∧ absorbed in 20 bars": levt(lambda f: first(capR(f) & f.abs_seen.astype(bool))),
 "F cap or washout turn (DOWN·cheap/fair, value rev.)": levt(lambda f: first(c(f).isin([0]) & (f.cvg_vph == 1) | ((c(f) == 1) & (f.v_tape.diff() > 0)))),
 "G cheap∧DOWN raw tapes, value mom reverting": levt(lambda f: first((f.v_tape <= -42.857) & (f.c_tape <= -30) & (f.cvg_vph == 1))),
}
rows = []
for h in (10, 20):
    for name, fn in S.items():
        for mode in ("tc", "xs"):
            t = L.score_c(P, fn, h=h, mode=mode, cal=cal)
            e = t.pivot(index="group", columns="era", values="edge")
            r = dict(h=h, sig=name, mode=mode)
            for er in ("E1", "E2", "E3"): r[er] = round(e.loc[L.NC, er].mean(), 3)
            r["min era"] = min(r["E1"], r["E2"], r["E3"])
            r["cls+"] = "/".join(str(int((e.loc[L.NC, er] > 0).sum())) for er in ("E1", "E2", "E3"))
            r["sig+/−"] = f"{int(((t.lo > 0) & t.group.isin(L.NC)).sum())}/{int(((t.hi < 0) & t.group.isin(L.NC)).sum())}"
            r["crypto"] = f"{e.loc['crypto','E2']:+.2f}/{e.loc['crypto','E3']:+.2f}"
            rows.append(r)
R = pd.DataFrame(rows)
cnt = {name: sum(int(np.nansum(np.abs(fn(f)))) for g in L.NC for f in P[g].values()) for name, fn in S.items()}
print("counts:", cnt)
for mode in ("tc", "xs"):
    print(f"\n==== {mode}"); print(R[R["mode"] == mode].drop(columns="mode").set_index(["h", "sig"]).to_string())
