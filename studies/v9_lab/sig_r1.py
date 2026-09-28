"""ROUND 1 (discovery E1+E2 only; E3 sealed). Every event is scored as a LONG; a consistently
negative one is a SHORT candidate. Kept: n >= 150 events in each of E1 and E2, and the same
sign in all 8 readings (tc/xs × h10/h20 × E1/E2). Ranked by the worst of the 8 (its magnitude)."""
import os, pickle, numpy as np, pandas as pd, lab_core as L, sig_core as S, warnings; warnings.filterwarnings("ignore")
P = pickle.load(open("sig_P.pkl", "rb")) if os.path.exists("sig_P.pkl") else S.build("feat", out="sig_P.pkl")
cal = L.calendar(P)
NM = ["cap", "wash", "dist", "basing", "idle", "stall", "turned", "build", "paid"]
C = {}
C["REF ▲ shipped (cap & vph+1)"] = lambda f: S.first((f.cell == 0) & (f.cvg_vph == 1))
C["REF ▼ shipped (dist entry)"] = lambda f: S.first(f.cell == 2)
for k, nm in enumerate(NM):
    C[f"enter {nm}"] = lambda f, k=k: (f.cell == k) & (f.prev != k)
    C[f"{nm} & vph+1"] = lambda f, k=k: S.first((f.cell == k) & (f.cvg_vph == 1))
    C[f"{nm} & vph-1"] = lambda f, k=k: S.first((f.cell == k) & (f.cvg_vph == -1))
    C[f"{nm} & cph+1"] = lambda f, k=k: S.first((f.cell == k) & (f.cvg_cph == 1))
    C[f"{nm} & cph-1"] = lambda f, k=k: S.first((f.cell == k) & (f.cvg_cph == -1))
for a in range(9):
    for b in range(9):
        if a != b:
            C[f"{NM[a]}→{NM[b]}"] = lambda f, a=a, b=b: (f.prev == a) & (f.cell == b)
C["leave cap (any)"] = lambda f: (f.prev == 0) & (f.cell != 0) & (f.cell < 9)
C["leave dist (any)"] = lambda f: (f.prev == 2) & (f.cell != 2) & (f.cell < 9)
C["cap & absorbed"] = lambda f: S.first((f.cell == 0) & f.eff_abs.fillna(False).astype(bool))
C["wash & absorbed"] = lambda f: S.first((f.cell == 1) & f.eff_abs.fillna(False).astype(bool))
C["cap & R bull"] = lambda f: (f.cell == 0) & f.r_bull.fillna(False).astype(bool)
C["cap & held"] = lambda f: S.first((f.cell == 0) & f.cvg_held.fillna(False).astype(bool))
C["cap & chart not cap"] = lambda f: S.first((f.cell == 0) & (f.cvg_chart != 0))
rows = []
for name, cond in C.items():
    r, n = S.score(P, S.event(cond), cal=cal)
    v = np.array(list(r.values()))
    rows.append(dict(event=name, nE1=n["E1"], nE2=n["E2"], lo=np.nanmin(v), hi=np.nanmax(v), mean=np.nanmean(v),
                     **{f"{m}{h} {e}": r[(m, h, e)] for (m, h, e) in r}))
    print(f"{name:32s} n {n['E1']:6d} {n['E2']:6d}  min {np.nanmin(v):+.3f} max {np.nanmax(v):+.3f}", flush=True)
R = pd.DataFrame(rows).set_index("event"); R.to_pickle("sig_r1.pkl")
ok = R[(R.nE1 >= 150) & (R.nE2 >= 150)]
pd.set_option("display.width", 250)
print("\n== LONG candidates: all 8 readings > 0, ranked by the worst")
print(ok[ok.lo > 0].sort_values("lo", ascending=False)[["nE1", "nE2", "lo", "mean"]].round(3).head(25).to_string())
print("\n== SHORT candidates: all 8 readings < 0, ranked by the worst (as a short)")
s = ok[ok.hi < 0].copy(); s["short_lo"] = -s.hi; s["short_mean"] = -s["mean"]
print(s.sort_values("short_lo", ascending=False)[["nE1", "nE2", "short_lo", "short_mean"]].round(3).head(25).to_string())
