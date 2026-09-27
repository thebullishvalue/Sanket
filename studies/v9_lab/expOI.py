import pickle, os, sys, numpy as np, pandas as pd, lab_core as L, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
import pine_audit as pa, pine_v5 as p5
panel = pa.load_oi_panel(f"{L.SP}/cache/oi")
P = pickle.load(open("lab_P.pkl", "rb"))
Q = {"fno": {}}
for s in panel.columns:
    k = f"{s}.NS"
    if k not in P["nse"] or panel[s].notna().sum() < 300: continue
    f = P["nse"][k].copy()
    o = panel[s].reindex(f.index)
    have = o.notna().to_numpy()
    q = p5.oi_quadrant(f["close"], o.ffill(), 9)
    ch = q.oi_char.to_numpy(); hist = f["hist"].to_numpy(float); ab = hist >= 0
    cast = np.where(ab, (q.oi_exit_up > q.oi_build_up) & (q.oi_exit_up >= 3), (q.oi_exit_dn > q.oi_build_dn) & (q.oi_exit_dn >= 3)).astype(bool)
    f["oi_have"] = have; f["oi_char"] = ch; f["cast"] = cast; f["oi_rank"] = q.oi_rank.to_numpy()
    f["era"] = np.where(f.index < pd.Timestamp("2023-01-01"), "E2", "E3")   # two halves: 2019-22 → E2 slot, 2023-26 → E3 slot
    f.loc[f.index < pd.Timestamp("2019-01-01"), "era"] = "E1"
    Q["fno"][k] = f
print("F&O names with OI:", len(Q["fno"]))
for h in (5, 10): L.xs_demean(Q, h)
cal = L.calendar(Q)
push = lambda f: np.where(f.live & f.oi_have & np.isfinite(f["hist"]), np.sign(f["hist"]), np.nan)
S = {
 "push, gold cast": lambda f: np.where(f.cast, push(f), np.nan),
 "push, no cast": lambda f: np.where(~f.cast, push(f), np.nan),
 "long build-up (long)": lambda f: np.where(f.live & f.oi_have, (f.oi_char == "LONG BUILDUP").astype(float), np.nan),
 "short build-up (long)": lambda f: np.where(f.live & f.oi_have, (f.oi_char == "SHORT BUILDUP").astype(float), np.nan),
 "short covering (long)": lambda f: np.where(f.live & f.oi_have, (f.oi_char == "SHORT COVERING").astype(float), np.nan),
 "long unwinding (long)": lambda f: np.where(f.live & f.oi_have, (f.oi_char == "LONG UNWINDING").astype(float), np.nan),
 "crowded book ≥80 pct (long)": lambda f: np.where(f.live & f.oi_have, (np.nan_to_num(f.oi_rank) >= 80).astype(float), np.nan),
 "capitulation, F&O names (long)": lambda f: np.where(f.live & f.oi_have, (f.cvg_cell == 0).astype(float), np.nan),
}
rows = []
for h in (5, 10):
    for name, fn in S.items():
        for mode in ("tc", "xs"):
            t = L.score_c(Q, fn, h=h, mode=mode, groups=["fno"], cal=cal)
            t = t[t.era != "E1"].set_index("era")
            rows.append(dict(h=h, sig=name, mode=mode, **{f"{'2019-22' if e=='E2' else '2023-26'}": f"{t.loc[e,'edge']:+.3f}{'*' if (t.loc[e,'lo']>0 or t.loc[e,'hi']<0) else ' '} (n {t.loc[e,'n']})" for e in ("E2", "E3")}))
print(pd.DataFrame(rows).set_index(["mode", "h", "sig"]).sort_index().to_string())
