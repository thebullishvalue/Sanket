"""Ladder-down window (Nov 2024 - Sep 2026, 1h + 4h rungs): the shortlist read on Ladder DOWN vs
Ladder UP, same bars. B1/B2 need conviction phase per ladder (not in this set) — not checked here."""
import glob, os, numpy as np, pandas as pd, lab_core as L, sig_core as S, warnings; warnings.filterwarnings("ignore")
P = {}
for g in L.GROUPS:
    P[g] = {}
    for fn in sorted(glob.glob(f"{L.SP}/featL/{g}__*.pkl")):
        f = pd.read_pickle(fn)
        base = f"{L.SP}/feat/{os.path.basename(fn)}"
        if not os.path.exists(base):
            continue
        b = pd.read_pickle(base)[["cvg_chart", "quiet"]].reindex(f.index)
        f = f.join(b)
        f = f[f["ready_dn"].fillna(False).astype(bool)].copy()
        if len(f) < 120:
            continue
        f["era"] = "E3"
        f["live"] = f["stack_ok"].fillna(False).astype(bool) & f["cv_ready"].fillna(False).astype(bool)
        L.add_u(f, (10, 20)); P[g][os.path.basename(fn)] = f
for h in (10, 20): L.xs_demean(P, h)
cal = L.calendar(P)
B = lambda s: s.fillna(False).astype(bool)                       # noqa: E731
def fam(lad):
    cell = lambda f: f[f"cell_{lad}"].fillna(9).astype(int)       # noqa: E731
    vph = lambda f: f[f"vph_{lad}"]                                # noqa: E731
    chr_ = lambda f: f.cvg_chart.fillna(9).astype(int)             # noqa: E731
    tri = lambda f: S.first((cell(f) == 0) & (vph(f) == 1))        # noqa: E731
    a1 = lambda f: tri(f) | S.first((cell(f) == 0) & (chr_(f) != 0) & (chr_(f) < 9))   # noqa: E731
    nq = lambda f: ~B(f.quiet).to_numpy()                          # noqa: E731
    return {f"▲ shipped · {lad}": tri, f"A1 · {lad}": a1, f"A2 · {lad}": lambda f: a1(f) & nq(f),
            f"C1 · {lad}": lambda f: tri(f) & nq(f)}
rows = []
for lad in ("dn", "up"):
    for name, cond in fam(lad).items():
        for grp_name, groups in (("idx+cmd+fx", ["idx", "cmd", "fx"]), ("stocks", ["nse", "us"]), ("crypto", ["crypto"])):
            r, n = S.score(P, S.event(cond), eras=("E3",), cal=cal, groups=groups)
            rows.append(dict(event=name, groups=grp_name, n=n["E3"], **{f"{m}{h}": v for (m, h, e), v in r.items()}))
        print(name, "done", flush=True)
R = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(R.set_index(["groups", "event"]).sort_index().round(3).to_string())
