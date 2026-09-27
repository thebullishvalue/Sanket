import pickle, numpy as np, pandas as pd, lab_core as L, warnings; warnings.filterwarnings("ignore")
P = pickle.load(open("lab_P.pkl", "rb")); cal = L.calendar(P)
for h in (10, 20): L.xs_demean(P, h)
th = 100 * 0.75 / 1.75
def ev(u, d): return lambda f: np.where(f.live, np.where(f[u].fillna(False), 1.0, np.where(f[d].fillna(False), -1.0, 0.0)), np.nan)
def lng(cond): return lambda f: np.where(f.live, np.where(cond(f), 1.0, 0.0), np.nan)
CAP = lambda f: (f.cvg_cell == 0)
vx_up = lambda f: (f.value > -th) & (f.value.shift(1) <= -th)
vx_dn = lambda f: (f.value < th) & (f.value.shift(1) >= th)
S = {
 "▲▼ TURN (v8)": ev("turn_buy", "turn_sell"),
 "BENCH value crosses back through θ": lambda f: np.where(f.live, np.where(vx_up(f), 1.0, np.where(vx_dn(f), -1.0, 0.0)), np.nan),
 "◆ continuation": ev("con_l", "con_s"),
 "R divergence": ev("r_bull", "r_bear"),
 "H divergence": ev("h_bull", "h_bear"),
 "declaration held": lambda f: np.where(f.live, f.v5_decl.astype(float), np.nan),
 "capitulation (cell 0)": lng(CAP),
 "cap ∧ above 200d avg": lng(lambda f: CAP(f) & (f.dist200 > 0)),
 "cap ∧ below 200d avg": lng(lambda f: CAP(f) & (f.dist200 <= 0)),
 "cap, first 5 bars in cell": lng(lambda f: CAP(f) & ((np.arange(len(f)) - f.cvg_since) < 5)),
 "cap, 5+ bars in cell": lng(lambda f: CAP(f) & ((np.arange(len(f)) - f.cvg_since) >= 5)),
 "cap, v_tape ≤ −60 (deep)": lng(lambda f: CAP(f) & (f.v_tape <= -60)),
 "cap, v_tape > −60": lng(lambda f: CAP(f) & (f.v_tape > -60)),
 "cap, value reverting (vph=+1)": lng(lambda f: CAP(f) & (f.cvg_vph == 1)),
 "cap, value widening (vph=−1)": lng(lambda f: CAP(f) & (f.cvg_vph == -1)),
 "cap, held row": lng(lambda f: CAP(f) & f.cvg_held.astype(bool)),
 "RAW cheap ∧ c_tape≤−30 (no push gate)": lng(lambda f: (f.v_tape <= -th) & (f.c_tape <= -30)),
 "cheap (v_tape≤−θ) any row": lng(lambda f: (f.v_tape <= -th)),
 "BENCH oversold rsi14<30": lng(lambda f: f.rsi14 < 30),
 "BENCH mom20z < −1.5": lng(lambda f: f.mom20z < -1.5),
}
rows = []
for h in (10, 20):
    for name, fn in S.items():
        for mode in ("tc", "xs"):
            t = L.score_c(P, fn, h=h, mode=mode, cal=cal)
            e = t.pivot(index="group", columns="era", values="edge"); sg = ((t.lo > 0) | (t.hi < 0)).groupby([t.group, t.era]).first().unstack()
            r = dict(h=h, sig=name, mode=mode)
            for er in ("E1", "E2", "E3"): r[er] = round(e.loc[L.NC, er].mean(), 3)
            r["classes+ E1/E2/E3"] = "/".join(str(int((e.loc[L.NC, er] > 0).sum())) for er in ("E1", "E2", "E3"))
            r["sig+ cells"] = int(((t.lo > 0) & t.group.isin(L.NC)).sum()); r["sig− cells"] = int(((t.hi < 0) & t.group.isin(L.NC)).sum())
            r["crypto E2/E3"] = f"{e.loc['crypto','E2']:+.3f}/{e.loc['crypto','E3']:+.3f}"
            r["n"] = int(t[t.group.isin(L.NC)].n.sum())
            rows.append(r)
R = pd.DataFrame(rows); R.to_pickle("exp4.pkl")
for h in (10, 20):
    for mode in ("tc", "xs"):
        print(f"\n==== h={h} mode={mode}: non-crypto average by era; how many of 5 classes positive; significant cells (of 15)")
        print(R[(R.h == h) & (R["mode"] == mode)].drop(columns=["h", "mode"]).set_index("sig").to_string())
