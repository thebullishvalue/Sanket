import pickle, numpy as np, pandas as pd, lab_core as L, warnings; warnings.filterwarnings("ignore")
P = pickle.load(open("lab_P.pkl", "rb"))
Q = {g: P[g] for g in ("nse", "us")}
nz = lambda x: np.asarray(x, float)
S = {
 "cv": lambda f: nz(f.cv), "c_tape": lambda f: nz(f.c_tape), "v_tape": lambda f: nz(f.v_tape),
 "value": lambda f: nz(f.value), "rv leg": lambda f: nz(f.sv_rv_z), "breadth": lambda f: nz(f.sv_breadth_z),
 "trace": lambda f: nz(f.trace), "hist_z": lambda f: nz(f.hist_z),
 "grid units": lambda f: np.where(f.cvg_cell < 9, f.cvg_units, np.nan),
 "grid flat units": lambda f: np.where(f.cvg_cell < 9, np.array([1.5,1,.25,1.5,1,.75,3,1.5,.75,1])[f.cvg_cell.fillna(9).astype(int)] * 0 + np.array([3,1.5,.25,1.5,1,.75,3,1.5,.75,1])[f.cvg_cell.fillna(9).astype(int)], np.nan),
 "B mom5": lambda f: nz(f.mom5z), "B mom20": lambda f: nz(f.mom20z), "B mom60": lambda f: nz(f.mom60z),
 "B dist200": lambda f: nz(f.dist200), "B rsi14": lambda f: nz(f.rsi14),
}
rows = []
for h in (10, 20):
    for name, fn in S.items():
        for g in ("nse", "us"):
            s = L.xs_ic_u(Q, g, fn, h)
            er = L.era_of(s.index)
            r = {"h": h, "sig": name, "g": g}
            for e in ("E1", "E2", "E3"):
                x = s[er == e]; xs = x.iloc[::h]
                r[e] = f"{x.mean():+.3f} ({xs.mean()/(xs.std()/np.sqrt(len(xs))):+.1f})" if len(xs) > 5 else ""
            rows.append(r)
T = pd.DataFrame(rows)
for h in (10, 20):
    print(f"\n==== CROSS-SECTIONAL rank IC, h={h} (t from non-overlapping dates). + = reading ranks winners")
    print(T[T.h == h].pivot(index="sig", columns="g", values=["E1", "E2", "E3"]).reindex(list(S)).to_string())
