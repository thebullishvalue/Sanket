"""ROUND 2 (discovery E1+E2 only; E3 still sealed). Refinements of Round 1's survivors.
Same keep rule: n >= 150 per era, same sign in all 8 readings; ranked by the worst reading."""
import pickle, numpy as np, pandas as pd, lab_core as L, sig_core as S, warnings; warnings.filterwarnings("ignore")
P = pickle.load(open("sig_P.pkl", "rb")); cal = L.calendar(P)
B = lambda s: s.fillna(False).astype(bool)                       # noqa: E731
cap = lambda f: f.cell == 0                                        # noqa: E731
tri = lambda f: (f.cell == 0) & (f.cvg_vph == 1)                   # the shipped ▲ state
chr_ = lambda f: f.cvg_chart.fillna(9).astype(int)                 # noqa: E731
absr = lambda f: B(f.eff_abs).astype(float).rolling(20, min_periods=1).max() > 0   # noqa: E731
C = {}
C["REF ▲ shipped"] = lambda f: S.first(tri(f))
C["L1 cap & chart≠cap"] = lambda f: S.first(cap(f) & (chr_(f) != 0) & (chr_(f) < 9))
C["L1a cap & chart row FAINT/UP"] = lambda f: S.first(cap(f) & (chr_(f) // 3 >= 1) & (chr_(f) < 9))
C["L1b cap & chart row DOWN, col fair"] = lambda f: S.first(cap(f) & (chr_(f) == 1))
C["L1c cap & chart col fair/rich"] = lambda f: S.first(cap(f) & (chr_(f) % 3 >= 1) & (chr_(f) < 9))
C["L1d cap & chart cheap, not DOWN"] = lambda f: S.first(cap(f) & (chr_(f) % 3 == 0) & (chr_(f) // 3 >= 1) & (chr_(f) < 9))
C["L1e cap & chart≠cap & vph+1"] = lambda f: S.first(cap(f) & (chr_(f) != 0) & (chr_(f) < 9) & (f.cvg_vph == 1))
C["L2 ▲ OR L1"] = lambda f: S.first(tri(f)) | S.first(cap(f) & (chr_(f) != 0) & (chr_(f) < 9))
C["L3 ▲ & absorbed ≤20b"] = lambda f: S.first(tri(f) & absr(f))
C["L3b ▲ & hist>0"] = lambda f: S.first(tri(f) & (f["hist"] > 0))
C["L3c ▲ & not quiet"] = lambda f: S.first(tri(f) & ~B(f.quiet))
C["L3d ▲ & cph+1"] = lambda f: S.first(tri(f) & (f.cvg_cph == 1))
C["L3e ▲ & cap young (≤20b)"] = lambda f: S.first(tri(f) & ((np.arange(len(f)) - f.cvg_since.fillna(0).to_numpy()) <= 20))
C["L4 wash→cap"] = lambda f: (f.prev == 1) & (f.cell == 0)
C["L4b wash→cap & chart≠cap"] = lambda f: (f.prev == 1) & (f.cell == 0) & (chr_(f) != 0) & (chr_(f) < 9)
C["L5 cap & cph+1 & vph+1"] = lambda f: S.first(cap(f) & (f.cvg_cph == 1) & (f.cvg_vph == 1))
# SHORT side (scored as long: want all 8 < 0)
C["S1 stall & vph+1"] = lambda f: S.first((f.cell == 5) & (f.cvg_vph == 1))
C["S1b stall & vph+1 & hist<0"] = lambda f: S.first((f.cell == 5) & (f.cvg_vph == 1) & (f["hist"] < 0))
C["S1c stall/paid & vph+1"] = lambda f: S.first(f.cell.isin([5, 8]) & (f.cvg_vph == 1))
C["S1d rich col & vph+1"] = lambda f: S.first(f.cell.isin([2, 5, 8]) & (f.cvg_vph == 1))
C["S2 idle→build"] = lambda f: (f.prev == 4) & (f.cell == 7)
C["S2b idle→build & chart rich"] = lambda f: (f.prev == 4) & (f.cell == 7) & (chr_(f) % 3 == 2)
C["S2c enter build & hist<0"] = lambda f: (f.prev != 7) & (f.cell == 7) & (f["hist"] < 0)
C["S3 build→paid"] = lambda f: (f.prev == 7) & (f.cell == 8)
C["S3b build→paid & vph+1"] = lambda f: (f.prev == 7) & (f.cell == 8) & (f.cvg_vph == 1)
C["S4 dist & vph+1"] = lambda f: S.first((f.cell == 2) & (f.cvg_vph == 1))
C["S4b enter rich col from UP row"] = lambda f: f.cell.isin([2, 5]) & (f.prev == 8)
C["S5 paid & chart≠paid"] = lambda f: S.first((f.cell == 8) & (chr_(f) != 8) & (chr_(f) < 9))
C["S5b paid & chart row DOWN/FAINT"] = lambda f: S.first((f.cell == 8) & (chr_(f) // 3 <= 1) & (chr_(f) < 9))
rows = []
for name, cond in C.items():
    r, n = S.score(P, S.event(cond), cal=cal)
    v = np.array(list(r.values()))
    rows.append(dict(event=name, nE1=n["E1"], nE2=n["E2"], lo=np.nanmin(v), hi=np.nanmax(v), mean=np.nanmean(v),
                     npos=int((v > 0).sum()), **{f"{m}{h} {e}": r[(m, h, e)] for (m, h, e) in r}))
    print(f"{name:34s} n {n['E1']:6d} {n['E2']:6d}  min {np.nanmin(v):+.3f} max {np.nanmax(v):+.3f} mean {np.nanmean(v):+.3f} pos {int((v>0).sum())}/8", flush=True)
pd.DataFrame(rows).set_index("event").to_pickle("sig_r2.pkl")
