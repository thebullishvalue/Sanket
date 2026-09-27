import pickle, numpy as np, pandas as pd, lab_core as L, warnings; warnings.filterwarnings("ignore")
P = pickle.load(open("lab_P.pkl", "rb")); cal = L.calendar(P)
clip = lambda x, s: np.clip(np.asarray(x, float) / s, -1, 1)
live = lambda f, x: np.where(f["live"].to_numpy(), x, np.nan)
S = {
 "cv (chart conviction)": lambda f: live(f, clip(f.cv, 100)),
 "c_tape (MTF conviction)": lambda f: live(f, clip(f.c_tape, 100)),
 "v_tape (MTF value)": lambda f: live(f, clip(f.v_tape, 100)),
 "value (chart value)": lambda f: live(f, clip(f.value, 100)),
 "rv leg": lambda f: live(f, clip(f.sv_rv_z, 2)),
 "breadth leg": lambda f: live(f, clip(f.sv_breadth_z, 2)),
 "trace": lambda f: live(f, clip(f.trace, 100)),
 "hist_z": lambda f: live(f, clip(f.hist_z, 3)),
 "grid units": lambda f: live(f, np.where(f.cvg_cell < 9, (f.cvg_units - 1) / 2, np.nan)),
 "BENCH mom5z": lambda f: live(f, clip(f.mom5z, 2)),
 "BENCH mom20z": lambda f: live(f, clip(f.mom20z, 2)),
 "BENCH mom60z": lambda f: live(f, clip(f.mom60z, 2)),
 "BENCH dist200": lambda f: live(f, clip(f.dist200, .2)),
 "BENCH rsi14": lambda f: live(f, clip(f.rsi14 - 50, 30)),
}
out = {}
for h in (10, 40):
    rows = []
    for name, fn in S.items():
        t = L.score_c(P, fn, h=h, mode="tc", cal=cal)
        out[(name, h)] = t
        e = t.pivot(index="group", columns="era", values="edge"); sg = t.assign(s=(t.lo > 0) | (t.hi < 0)).pivot(index="group", columns="era", values="s")
        r = {"sig": name}
        for er in ("E1", "E2", "E3"): r[f"nc_{er}"] = e.loc[L.NC, er].mean()
        for g in L.GROUPS: r[g] = " ".join(f"{e.loc[g, er]:+.3f}{'*' if sg.loc[g, er] else ' '}" for er in ("E1", "E2", "E3"))
        rows.append(r)
    print(f"\n==== TIME-SERIES (causal-centred) h={h}. avg non-crypto by era; per class E1 E2 E3 (* = 95% block-bootstrap excludes 0)")
    print(pd.DataFrame(rows).set_index("sig").round(3).to_string())
pickle.dump(out, open("exp1b.pkl", "wb"))
