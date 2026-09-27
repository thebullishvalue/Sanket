import os, sys, pickle, numpy as np, pandas as pd, lab_core as L, warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
import cvgrid as N
warnings.filterwarnings("ignore")
P = pickle.load(open("lab_P.pkl", "rb")); cal = L.calendar(P)
NAMES = N.NAMES
for h in (10, 20):
    L.xs_demean(P, h)
res = []
for h in (10, 20):
    for c in range(9):
        fn = lambda f, c=c: np.where(f.live & (f.cvg_cell < 9), (f.cvg_cell == c).astype(float), np.nan)
        # time-series causal: in-cell indicator centred on its trailing frequency
        t = L.score_c(P, fn, h=h, mode="tc", cal=cal)
        # conditional mean market-neutral return while in the cell (xs), per era
        fn2 = lambda f, c=c: np.where(f.live & (f.cvg_cell == c), 1.0, np.nan)
        x = L.score_c(P, fn2, h=h, mode="xs", cal=cal)
        for T, kind in ((t, "tc"), (x, "xs")):
            for _, r in T.iterrows():
                res.append(dict(h=h, cell=NAMES[c], kind=kind, group=r.group, era=r.era, edge=r.edge, sig=(r.lo > 0) or (r.hi < 0), n=r.n))
R = pd.DataFrame(res); R.to_pickle("exp3.pkl")
for h in (10, 20):
    for kind in ("xs", "tc"):
        X = R[(R.h == h) & (R.kind == kind)]
        X = X.assign(s=X.apply(lambda r: f"{r.edge:+.3f}{'*' if r.sig else ' '}", axis=1))
        pv = X.pivot_table(index="cell", columns=["group", "era"], values="s", aggfunc="first")
        nc = X[X.group.isin(L.NC)].groupby(["cell", "era"]).edge.mean().unstack().round(3)
        share = X[X.era == "E3"].groupby("cell").n.sum() / X[X.era == "E3"].n.sum()
        print(f"\n==== h={h} {kind}: {'mean market-neutral return while in cell' if kind=='xs' else 'causal-centred timing of in-cell'} (σ)")
        print(pd.concat([nc.add_prefix("nc_"), (share*100).round(1).rename("E3 %bars")], axis=1).reindex(NAMES[:9]).to_string())
        print(pv.reindex(NAMES[:9]).to_string())
