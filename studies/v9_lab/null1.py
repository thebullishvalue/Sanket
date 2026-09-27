import pickle, numpy as np, pandas as pd, lab_core as L
P = pickle.load(open("lab_P.pkl", "rb"))
rng = np.random.default_rng(1)
Q = {}
for g in ["us", "idx", "cmd", "fx"]:
    Q[g] = {}
    for t, f in P[g].items():
        f = f[["open", "close", "live", "era"]].copy()
        r = np.log(f.close).diff().std()
        rw = np.exp(np.cumsum(rng.normal(0.0003, r, len(f))))
        f["close"] = rw; f["open"] = rw
        lo = np.log(f.open)
        for h in (10, 40):
            fw = lo.shift(-(1 + h)) - lo.shift(-1); z = pd.Series(np.nan, index=f.index)
            for e in ("E1", "E2", "E3"):
                m = (f.era == e) & f.live & fw.notna()
                if m.sum() > 50: x = fw[m]; z[m] = (x - x.mean()) / x.std()
            f[f"z{h}"] = z
        c = f.close; vol = np.log(c).diff().rolling(60).std()
        f["mom20z"] = np.log(c / c.shift(20)) / (vol * np.sqrt(20)); f["mom60z"] = np.log(c / c.shift(60)) / (vol * np.sqrt(60))
        f["dist200"] = np.log(c / c.rolling(200).mean())
        Q[g][t] = f
cal = L.calendar(Q)
for name, fn in {"mom20z": lambda f: np.where(f.live, np.clip(f.mom20z / 2, -1, 1), np.nan),
                 "mom60z": lambda f: np.where(f.live, np.clip(f.mom60z / 2, -1, 1), np.nan),
                 "dist200": lambda f: np.where(f.live, np.clip(f.dist200 / .2, -1, 1), np.nan)}.items():
    for h in (10, 40):
        t = L.score(Q, fn, h=h, groups=["us", "idx", "cmd", "fx"], cal=cal, do_boot=False)
        print(name, h, t.pivot(index="group", columns="era", values="edge").mean().round(4).to_dict())
