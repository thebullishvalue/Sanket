import pickle, numpy as np, pandas as pd, lab_core as L
P = pickle.load(open("lab_P.pkl", "rb")); rng = np.random.default_rng(3); Q = {}
for g in ["us", "idx", "cmd", "fx"]:
    Q[g] = {}
    for t, f in P[g].items():
        f = f[["open", "close", "live", "era"]].copy()
        r = np.log(f.close).diff().std(); mu = 0.0004
        rw = np.exp(np.cumsum(rng.normal(mu, r, len(f))))
        f["close"] = rw; f["open"] = rw; L.add_u(f, (10, 40))
        c = f.close; vol = np.log(c).diff().rolling(60).std()
        f["mom20z"] = np.log(c / c.shift(20)) / (vol * np.sqrt(20)); f["dist200"] = np.log(c / c.rolling(200).mean())
        Q[g][t] = f
for h in (10, 40): L.xs_demean(Q, h)
cal = L.calendar(Q)
for name, fn in {"mom20z": lambda f: np.where(f.live, np.clip(f.mom20z / 2, -1, 1), np.nan),
                 "dist200": lambda f: np.where(f.live, np.clip(f.dist200 / .2, -1, 1), np.nan)}.items():
    for mode in ("xs", "tc"):
        for h in (10, 40):
            t = L.score_c(Q, fn, h=h, mode=mode, groups=["us", "idx", "cmd", "fx"], cal=cal)
            print(f"{name:8s} {mode} h{h}", t.pivot(index="group", columns="era", values="edge").mean().round(4).to_dict(),
                  "sig cells", int(((t.lo > 0) | (t.hi < 0)).sum()), "/", len(t))
