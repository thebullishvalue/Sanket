import pickle, glob, os, numpy as np, pandas as pd, lab_core as L, warnings; warnings.filterwarnings("ignore")
def loadv(d):
    P = {}
    for g in L.GROUPS:
        P[g] = {}
        for fn in sorted(glob.glob(f"{L.SP}/{d}/{g}__*.pkl")):
            f = pd.read_pickle(fn); f["era"] = L.era_of(f.index)
            f["live"] = f["stack_ok"].fillna(False).astype(bool) & f["cv_ready"].fillna(False).astype(bool)
            L.add_u(f, (10, 20)); P[g][os.path.basename(fn)] = f
    for h in (10, 20): L.xs_demean(P, h)
    return P
def cool(ev, n=10):
    ev = np.asarray(ev, bool); out = np.zeros(len(ev), bool); last = -10**9
    for i in np.flatnonzero(ev):
        if i - last >= n: out[i] = True; last = i
    return out
def first(x): x = x.fillna(False).astype(bool); return x & ~x.shift(1, fill_value=False)
S = {
 "cap state": lambda f: np.where(f.live, (f.cvg_cell == 0).astype(float), np.nan),
 "cap-turn event": lambda f: np.where(f.live, cool(first((f.cvg_cell == 0) & (f.cvg_vph == 1)).to_numpy(bool)).astype(float), np.nan),
 "grid units": lambda f: np.where(f.live & (f.cvg_cell < 9), (f.cvg_units - 1) / 2, np.nan),
}
rows = []
for d in ["feat", "feat_part_off", "feat_den_effort", "feat_leg_rv", "feat_leg_br", "feat_hedge_off", "feat_z1_20", "feat_z1_40", "feat_theta_10", "feat_theta_20"]:
    P = loadv(d); cal = L.calendar(P)
    occ = np.mean([np.nanmean(np.where(f.live, f.cvg_cell == 0, np.nan)) for g in L.NC for f in P[g].values()])
    r = {"variant": d.replace("feat_", "").replace("feat", "BASE"), "cap % bars": round(100 * occ, 1)}
    for name, fn in S.items():
        for mode, h in (("tc", 10), ("tc", 20), ("xs", 10)):
            t = L.score_c(P, fn, h=h, mode=mode, cal=cal, do_boot=False)
            e = t.pivot(index="group", columns="era", values="edge").loc[L.NC].mean()
            r[f"{name} {mode}{h}"] = f"{e.E1:+.3f} {e.E2:+.3f} {e.E3:+.3f}"
            r[f"{name} {mode}{h} min"] = round(e.min(), 3)
    rows.append(r); print(r["variant"], "done", flush=True)
R = pd.DataFrame(rows).set_index("variant")
pd.set_option("display.width", 250)
for name in S:
    cols = ["cap % bars"] + [c for c in R.columns if c.startswith(name)]
    print(f"\n==== {name}: non-crypto avg E1 E2 E3, and the worst era")
    print(R[cols].to_string())
