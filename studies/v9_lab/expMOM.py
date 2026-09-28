"""PRE-REGISTERED (before any number): can the grid be styled for MOMENTUM and beat the current
reversion units? Unit maps by cell (row 0 DOWN / 1 FAINT / 2 UP × cheap / fair / rich):
  C4  current      4 1.5 .25 | 1.5 1 .75 | 3 1.5 .75
  M1  trend        .25 .25 .25 | 1 1 1 | 3 3 3
  M2  strength     .25 .25 .5 | .5 1 1.5 | 1.5 3 4
  M3  pullback     .5 .25 .25 | 1 1 .75 | 4 3 1.5
  M4  confirmed    DOWN .25 | FAINT 1 | UP 4 if conviction momentum confirms the row else 2
  REF 12-1 month time-series momentum (sign), the classic factor, for context.
Scorers tc and xs (causal), h 10 / 20 / 40 / 60, eras E1 <2014, E2 2014-19, E3 >=2020, non-crypto
average (crypto separate). RULE: a momentum map is 'better' only if it beats C4 in all three eras
under both scorers at h 20 AND h 40 (non-crypto), and then passes Pragyam's allocator test."""
import sys, glob, os, numpy as np, pandas as pd, lab_core as L, warnings; warnings.filterwarnings("ignore")
D = sys.argv[1] if len(sys.argv) > 1 else "feat_cap4"
LAG = 252 if D != "featw" else 52; SKIP = 21 if D != "featw" else 4
HS = (10, 20, 40, 60) if D != "featw" else (4, 8, 13)
P = {}
for g in L.GROUPS:
    P[g] = {}
    for fn in sorted(glob.glob(f"{L.SP}/{D}/{g}__*.pkl")):
        f = pd.read_pickle(fn)[["open", "close", "stack_ok", "cv_ready", "cvg_cell", "cvg_cph", "cvg_units"]].copy()
        f["era"] = L.era_of(f.index)
        f["live"] = f["stack_ok"].fillna(False).astype(bool) & f["cv_ready"].fillna(False).astype(bool)
        L.add_u(f, HS); P[g][os.path.basename(fn)] = f
for h in HS: L.xs_demean(P, h)
cal = L.calendar(P)
MAPS = {"C4": [4, 1.5, .25, 1.5, 1, .75, 3, 1.5, .75],
        "M1": [.25, .25, .25, 1, 1, 1, 3, 3, 3],
        "M2": [.25, .25, .5, .5, 1, 1.5, 1.5, 3, 4],
        "M3": [.5, .25, .25, 1, 1, .75, 4, 3, 1.5]}
def cellmap(u):
    u = np.asarray(u + [np.nan], float)
    return lambda f: np.where(f.live & (f.cvg_cell < 9), (u[f.cvg_cell.fillna(9).astype(int).clip(0, 9)] - 1) / 2, np.nan)
F = {k: cellmap(v) for k, v in MAPS.items()}
def m4(f):
    c = f.cvg_cell.fillna(9).astype(int); row = c // 3
    u = np.where(row == 2, np.where(f.cvg_cph == 1, 4.0, 2.0), np.where(row == 1, 1.0, 0.25))
    return np.where(f.live & (c < 9), (u - 1) / 2, np.nan)
F["M4"] = m4
def ref(f):
    lc = np.log(f.close)
    return np.where(f.live, np.sign(lc.shift(SKIP) - lc.shift(LAG)), np.nan)
F["REF"] = ref
rows = []
for name, fn in F.items():
    for mode in ("tc", "xs"):
        for h in HS:
            t = L.score_c(P, fn, h=h, mode=mode, cal=cal, do_boot=False)
            pv = t.pivot(index="group", columns="era", values="edge")
            e = pv.loc[L.NC].mean(); c = pv.loc["crypto"]
            pos = (pv.loc[L.NC] > 0).sum()
            rows.append(dict(map=name, mode=mode, h=h, E1=e.E1, E2=e.E2, E3=e.E3,
                             pos=f"{pos.E1}{pos.E2}{pos.E3}", cE1=c.E1, cE2=c.E2, cE3=c.E3))
    print(name, "done", flush=True)
R = pd.DataFrame(rows)
pd.set_option("display.width", 220)
for mode in ("tc", "xs"):
    for h in HS:
        x = R[(R["mode"] == mode) & (R.h == h)].set_index("map")[["E1", "E2", "E3", "pos", "cE1", "cE2", "cE3"]]
        print(f"\n== {D} · {mode} h{h} · non-crypto avg edge by era (σ) · groups>0 of 5 · crypto")
        print(x.round(3).to_string())
R.to_pickle(f"expMOM_{D}.pkl")
