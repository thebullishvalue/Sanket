import pickle, glob, os, numpy as np, pandas as pd, lab_core as L, warnings; warnings.filterwarnings("ignore")
def cool(ev, n=10):
    ev = np.asarray(ev, bool); out = np.zeros(len(ev), bool); last = -10**9
    for i in np.flatnonzero(ev):
        if i - last >= n: out[i] = True; last = i
    return out
def first(x): x = x.fillna(False).astype(bool); return x & ~x.shift(1, fill_value=False)
FLAT = np.array([3, 1.5, .25, 1.5, 1, .75, 3, 1.5, .75, 1.0])
def run(P, hs, ncool, label):
    for h in hs: L.xs_demean(P, h)
    cal = L.calendar(P)
    S = {
     "▼ enter distribution": lambda f: np.where(f.live, -cool(first(f.cvg_cell == 2).to_numpy(bool), ncool).astype(float), np.nan),
     "▼ distribution turns (value reverting)": lambda f: np.where(f.live, -cool(first((f.cvg_cell == 2) & (f.cvg_vph == 1)).to_numpy(bool), ncool).astype(float), np.nan),
     "▼ STATE distribution": lambda f: np.where(f.live, -(f.cvg_cell == 2).astype(float), np.nan),
     "cap ∧ sellers running (cph+1)": lambda f: np.where(f.live, ((f.cvg_cell == 0) & (f.cvg_cph == 1)).astype(float), np.nan),
     "cap ∧ sellers pausing (cph−1)": lambda f: np.where(f.live, ((f.cvg_cell == 0) & (f.cvg_cph == -1)).astype(float), np.nan),
     "units: graded (5×5 phases)": lambda f: np.where(f.live & (f.cvg_cell < 9), (f.cvg_units - 1) / 2, np.nan),
     "units: flat 3×3 cell": lambda f: np.where(f.live & (f.cvg_cell < 9), (FLAT[f.cvg_cell.fillna(9).astype(int)] - 1) / 2, np.nan),
    }
    rows = []
    for h in hs:
        for name, fn in S.items():
            for mode in ("tc", "xs"):
                t = L.score_c(P, fn, h=h, mode=mode, cal=cal)
                e = t.pivot(index="group", columns="era", values="edge")
                r = dict(h=h, sig=name, mode=mode)
                for er in ("E1", "E2", "E3"): r[er] = round(e.loc[L.NC, er].mean(), 3)
                r["cls+"] = "/".join(str(int((e.loc[L.NC, er] > 0).sum())) for er in ("E1", "E2", "E3"))
                r["sig+/−"] = f"{int(((t.lo > 0) & t.group.isin(L.NC)).sum())}/{int(((t.hi < 0) & t.group.isin(L.NC)).sum())}"
                rows.append(r)
    R = pd.DataFrame(rows)
    cnt = {k: sum(int(np.nansum(np.abs(S[k](f)))) for g in L.NC for f in P[g].values()) for k in list(S)[:2]}
    print(f"\n######## {label} counts {cnt}")
    for mode in ("tc", "xs"):
        print(f"--- {mode}"); print(R[R["mode"] == mode].drop(columns="mode").set_index(["h", "sig"]).to_string())
P = pickle.load(open("lab_P.pkl", "rb"))
run(P, (10, 20), 10, "DAILY")
W = {}
for g in L.GROUPS:
    W[g] = {}
    for fn in sorted(glob.glob(f"{L.SP}/featw/{g}__*.pkl")):
        f = pd.read_pickle(fn); f["era"] = L.era_of(f.index)
        f["live"] = f["stack_ok"].fillna(False).astype(bool) & f["cv_ready"].fillna(False).astype(bool)
        L.add_u(f, (4, 8)); W[g][os.path.basename(fn)] = f
run(W, (4, 8), 3, "WEEKLY (E1 has no data)")
