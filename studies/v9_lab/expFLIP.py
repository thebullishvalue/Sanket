"""PRE-REGISTERED: 'state map + histogram flips green'. Events (first bar, cooldown, stack live):
FLIP any · FLIP|cell k (0..8) · FLIP|cheap col · FLIP|UP row · CFLIP (conviction's own hist,
the grid's push) |cell0 and |cheap · shorts: RED|DOWN·rich, RED|UP·rich, RED|rich col ·
reference: the shipped ▲ (cell 0 & value phase +1, first bar). Scorers tc / xs, eras, non-crypto
avg, crypto separate. RULE: positive in all 3 eras under both scorers at the two shorter
horizons, AND above both FLIP-any and the shipped ▲ — then Pragyam's allocator."""
import sys, glob, os, numpy as np, pandas as pd, lab_core as L, warnings; warnings.filterwarnings("ignore")
D = sys.argv[1] if len(sys.argv) > 1 else "feat"
W = D == "featw"; HS = (4, 8, 13) if W else (10, 20, 40); COOL = 4 if W else 10
P = {}
for g in L.GROUPS:
    P[g] = {}
    for fn in sorted(glob.glob(f"{L.SP}/{D}/{g}__*.pkl")):
        f = pd.read_pickle(fn)[["open", "close", "stack_ok", "cv_ready", "cvg_cell", "cvg_vph", "hist", "cvg_push"]].copy()
        f["era"] = L.era_of(f.index)
        f["live"] = f["stack_ok"].fillna(False).astype(bool) & f["cv_ready"].fillna(False).astype(bool)
        L.add_u(f, HS); P[g][os.path.basename(fn)] = f
for h in HS: L.xs_demean(P, h)
cal = L.calendar(P)
def cool(ev, n=COOL):
    ev = np.asarray(ev, bool); out = np.zeros(len(ev), bool); last = -10**9
    for i in np.flatnonzero(ev):
        if i - last >= n: out[i] = True; last = i
    return out
def up_flip(x):  x = x.astype(float); return (x > 0) & (x.shift(1) <= 0)
def dn_flip(x):  x = x.astype(float); return (x < 0) & (x.shift(1) >= 0)
def ev(cond_fn, sign=1.0):
    def fn(f):
        c = f.cvg_cell.fillna(9).astype(int)
        e = cond_fn(f, c) & f.live
        return np.where(f.live, sign * cool(e.fillna(False).to_numpy(bool)).astype(float), np.nan)
    return fn
E = {"FLIP any": ev(lambda f, c: up_flip(f["hist"]))}
NAMES = ["DOWN·cheap (capitulation)", "DOWN·fair (washout)", "DOWN·rich (distribution)",
         "FAINT·cheap (basing)", "FAINT·fair (idle)", "FAINT·rich (stalling)",
         "UP·cheap (turned)", "UP·fair (building)", "UP·rich (paid)"]
for k, nm in enumerate(NAMES):
    E[f"FLIP|{nm}"] = ev(lambda f, c, k=k: up_flip(f["hist"]) & (c == k))
E["FLIP|cheap col"] = ev(lambda f, c: up_flip(f["hist"]) & (c % 3 == 0) & (c < 9))
E["FLIP|UP row"] = ev(lambda f, c: up_flip(f["hist"]) & (c // 3 == 2) & (c < 9))
E["CFLIP|capitulation"] = ev(lambda f, c: up_flip(f.cvg_push) & (c == 0))
E["CFLIP|cheap col"] = ev(lambda f, c: up_flip(f.cvg_push) & (c % 3 == 0) & (c < 9))
E["shipped ▲"] = ev(lambda f, c: (c == 0) & (f.cvg_vph == 1) & ~((c == 0) & (f.cvg_vph == 1)).shift(1, fill_value=False))
E["RED|DOWN·rich (short)"] = ev(lambda f, c: dn_flip(f["hist"]) & (c == 2), -1.0)
E["RED|UP·rich (short)"] = ev(lambda f, c: dn_flip(f["hist"]) & (c == 8), -1.0)
E["RED|rich col (short)"] = ev(lambda f, c: dn_flip(f["hist"]) & (c % 3 == 2) & (c < 9), -1.0)
rows = []
for name, fn in E.items():
    n = {e: 0 for e in ("E1", "E2", "E3")}
    for g in L.NC:
        for f in P[g].values():
            v = fn(f); m = np.nan_to_num(v) != 0
            for e in n: n[e] += int((m & (f.era.to_numpy() == e)).sum())
    r = dict(event=name, nE1=n["E1"], nE2=n["E2"], nE3=n["E3"])
    for mode in ("tc", "xs"):
        for h in HS:
            t = L.score_c(P, fn, h=h, mode=mode, cal=cal, do_boot=False)
            pv = t.pivot(index="group", columns="era", values="edge")
            e = pv.loc[L.NC].mean()
            for era in ("E1", "E2", "E3"): r[f"{mode}{h} {era}"] = e[era]
            r[f"{mode}{h} crypto E3"] = pv.loc["crypto", "E3"]
    rows.append(r); print(name, "done", flush=True)
R = pd.DataFrame(rows).set_index("event")
R.to_pickle(f"expFLIP_{D}.pkl")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
print("\nevents (non-crypto):"); print(R[["nE1", "nE2", "nE3"]].to_string())
for mode in ("tc", "xs"):
    for h in HS:
        c = [f"{mode}{h} E1", f"{mode}{h} E2", f"{mode}{h} E3", f"{mode}{h} crypto E3"]
        print(f"\n== {D} · {mode} h{h} · non-crypto avg σ by era · crypto E3"); print(R[c].round(3).to_string())
