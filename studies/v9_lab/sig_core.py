"""Signal search lab: cached panel + pooled scorer. Discovery = E1 (<2014) + E2 (2014-19);
E3 (>=2020) is SEALED until the final shortlist."""
import glob, os, pickle, numpy as np, pandas as pd, lab_core as L, warnings; warnings.filterwarnings("ignore")
HS = (10, 20)
COLS = ["open", "close", "stack_ok", "cv_ready", "cvg_cell", "cvg_vph", "cvg_cph", "cvg_held", "cvg_since",
        "cvg_chart", "cvg_lead", "cvg_from", "push_tier", "quiet", "hist", "hist_z", "eff_abs", "eff_pct",
        "trace", "conv", "value", "c_tape", "v_tape", "r_bull", "r_bear", "h_bull", "h_bear", "cvg_push"]


def build(D="feat", hs=HS, out=None):
    P = {}
    for g in L.GROUPS:
        P[g] = {}
        for fn in sorted(glob.glob(f"{L.SP}/{D}/{g}__*.pkl")):
            f = pd.read_pickle(fn)
            f = f[[c for c in COLS if c in f.columns]].copy()
            f["era"] = L.era_of(f.index)
            f["live"] = f["stack_ok"].fillna(False).astype(bool) & f["cv_ready"].fillna(False).astype(bool)
            f["cell"] = f["cvg_cell"].fillna(9).astype(int)
            f["prev"] = f["cell"].shift(1).fillna(9).astype(int)
            L.add_u(f, hs); P[g][os.path.basename(fn)] = f
    for h in hs: L.xs_demean(P, h)
    if out: pickle.dump(P, open(out, "wb"))
    return P


def cool(ev, n=10):
    ev = np.asarray(ev, bool); o = np.zeros(len(ev), bool); last = -10**9
    for i in np.flatnonzero(ev):
        if i - last >= n: o[i] = True; last = i
    return o


def first(x):
    x = pd.Series(np.asarray(x, bool)); return (x & ~x.shift(1, fill_value=False)).to_numpy()


def event(cond, sign=1.0, n=10):
    """cond(f) -> bool array; a position fn: ±1 on the (cooled) event bar, 0 else, NaN not live."""
    def fn(f):
        e = np.asarray(cond(f), bool) & f["live"].to_numpy(bool)
        return np.where(f["live"], sign * cool(e, n).astype(float), np.nan)
    return fn


def score(P, fn, eras=("E1", "E2"), hs=HS, cal=None, groups=L.NC):
    """Non-crypto average edge per (mode, h, era), and event counts per era."""
    cal = cal if cal is not None else L.calendar(P)
    res, n = {}, {e: 0 for e in ("E1", "E2", "E3")}
    for g in groups:
        for f in P[g].values():
            v = np.nan_to_num(fn(f)) != 0; fe = f["era"].to_numpy()
            for e in n: n[e] += int((v & (fe == e)).sum())
    for mode in ("tc", "xs"):
        for h in hs:
            t = L.score_c(P, fn, h=h, mode=mode, groups=groups, cal=cal, do_boot=False)
            pv = t.pivot(index="group", columns="era", values="edge").reindex(groups)
            for e in eras: res[(mode, h, e)] = float(pv[e].mean())
    return res, n
