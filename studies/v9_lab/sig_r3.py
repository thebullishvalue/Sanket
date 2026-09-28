"""ROUND 3 — the shortlist, fixed from Rounds 1-2 on discovery only, then the SEALED era opened once.
Shortlist (+ the shipped ▲ as reference):
  A1  ▲ OR cap & chart≠cap                    (widened capitulation turn)
  A2  A1 & not quiet
  B1  cap & cph+1 & vph+1                     (double-confirmed capitulation turn)
  B2  B1 & not quiet
  C1  ▲ & not quiet
  D1  wash→cap OR A1
Daily E1/E2/E3 × tc/xs × h10/h20/h40 (non-crypto avg), crypto E3; weekly (featw) the same at h4/h8/h13."""
import os, sys, pickle, numpy as np, pandas as pd, lab_core as L, sig_core as S, warnings; warnings.filterwarnings("ignore")
B = lambda s: s.fillna(False).astype(bool)                       # noqa: E731
chr_ = lambda f: f.cvg_chart.fillna(9).astype(int)                 # noqa: E731
tri = lambda f: S.first((f.cell == 0) & (f.cvg_vph == 1))          # noqa: E731
l1 = lambda f: S.first((f.cell == 0) & (chr_(f) != 0) & (chr_(f) < 9))   # noqa: E731
A1 = lambda f: tri(f) | l1(f)                                      # noqa: E731
b1 = lambda f: S.first((f.cell == 0) & (f.cvg_cph == 1) & (f.cvg_vph == 1))  # noqa: E731
nq = lambda f: ~B(f.quiet).to_numpy()                              # noqa: E731
C = {"REF ▲ shipped": tri,
     "A1 ▲ or cap&chart≠cap": A1,
     "A2 A1 & not quiet": lambda f: A1(f) & nq(f),
     "B1 cap & cph+1 & vph+1": b1,
     "B2 B1 & not quiet": lambda f: b1(f) & nq(f),
     "C1 ▲ & not quiet": lambda f: tri(f) & nq(f),
     "D1 wash→cap or A1": lambda f: ((f.prev == 1) & (f.cell == 0)).to_numpy() | A1(f)}
D = sys.argv[1] if len(sys.argv) > 1 else "feat"
if D == "feat":
    P = pickle.load(open("sig_P.pkl", "rb")); HS3, COOL = (10, 20, 40), 10
    for g in P:
        for f in P[g].values():
            L.add_u(f, (40,))
    L.xs_demean(P, 40)
else:
    HS3, COOL = (4, 8, 13), 4
    P = S.build(D, hs=HS3)
cal = L.calendar(P)
rows = []
for name, cond in C.items():
    fn = S.event(cond, n=COOL)
    r, n = S.score(P, fn, eras=("E1", "E2", "E3"), hs=HS3, cal=cal)
    rc, _ = S.score(P, fn, eras=("E3",), hs=HS3, cal=cal, groups=["crypto"])
    row = dict(event=name, nE1=n["E1"], nE2=n["E2"], nE3=n["E3"])
    for (m, h, e), v in r.items(): row[f"{m}{h} {e}"] = v
    for (m, h, e), v in rc.items(): row[f"crypto {m}{h}"] = v
    rows.append(row); print(name, "done", flush=True)
R = pd.DataFrame(rows).set_index("event"); R.to_pickle(f"sig_r3_{D}.pkl")
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 60)
print("\nevents:"); print(R[["nE1", "nE2", "nE3"]].to_string())
for m in ("tc", "xs"):
    for h in HS3:
        c = [f"{m}{h} E1", f"{m}{h} E2", f"{m}{h} E3", f"crypto {m}{h}"]
        print(f"\n== {D} {m} h{h}: non-crypto E1 E2 E3 · crypto E3"); print(R[c].round(3).to_string())
ev = R[[c for c in R.columns if c[:2] in ("tc", "xs") and not c.endswith("crypto")]]
print("\n== worst / mean over every era × scorer × horizon (non-crypto)")
print(pd.DataFrame({"worst": ev.min(axis=1), "mean": ev.mean(axis=1),
                    "worst E3": ev[[c for c in ev.columns if c.endswith("E3")]].min(axis=1)}).round(3).to_string())
