"""Fresh-audit lab: panel loader and scorers. Eras: E1 <2014, E2 2014-2019, E3 >=2020."""
import glob, os, numpy as np, pandas as pd
SP = os.environ.get("V9_LAB_DIR", "v9_work")
GROUPS = ["nse", "us", "idx", "cmd", "fx", "crypto"]
NC = ["nse", "us", "idx", "cmd", "fx"]
HS = (5, 10, 20, 40)
ERAS = (("E1", None, "2014-01-01"), ("E2", "2014-01-01", "2020-01-01"), ("E3", "2020-01-01", None))

def era_of(idx):
    e = np.full(len(idx), "", dtype=object)
    for name, a, b in ERAS:
        m = np.ones(len(idx), bool)
        if a: m &= idx >= pd.Timestamp(a)
        if b: m &= idx < pd.Timestamp(b)
        e[m] = name
    return e

def load(groups=GROUPS, extra=None):
    """dict group -> dict tkr -> frame with fwd z columns (drift-removed per era, σ-normalised)."""
    P = {}
    for g in groups:
        P[g] = {}
        for fn in sorted(glob.glob(f"{SP}/feat/{g}__*.pkl")):
            f = pd.read_pickle(fn)
            tkr = os.path.basename(fn)[len(g) + 2:-4]
            o = f["open"].where(f["open"] > 0)
            lo = np.log(o)
            f["era"] = era_of(f.index)
            ok = f["stack_ok"].fillna(False).astype(bool) & f["cv_ready"].fillna(False).astype(bool)
            f["live"] = ok
            for h in HS:
                fw = lo.shift(-(1 + h)) - lo.shift(-1)
                z = pd.Series(np.nan, index=f.index)
                for e in ("E1", "E2", "E3"):
                    m = (f["era"] == e) & ok & fw.notna()
                    if m.sum() > 50:
                        x = fw[m]; z[m] = (x - x.mean()) / x.std()
                f[f"z{h}"] = z
                f[f"r{h}"] = fw
            c = f["close"]
            f["mom20"] = np.log(c / c.shift(20))
            f["mom60"] = np.log(c / c.shift(60))
            f["mom5"] = np.log(c / c.shift(5))
            f["dist200"] = np.log(c / c.rolling(200).mean())
            vol = np.log(c).diff().rolling(60).std()
            f["mom20z"] = f["mom20"] / (vol * np.sqrt(20))
            f["mom60z"] = f["mom60"] / (vol * np.sqrt(60))
            f["mom5z"] = f["mom5"] / (vol * np.sqrt(5))
            d = c.diff(); up = d.clip(lower=0).ewm(alpha=1/14).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1/14).mean()
            f["rsi14"] = 100 - 100 / (1 + up / dn)
            if extra: extra(f)
            P[g][tkr] = f
    return P

def calendar(P):
    idx = set()
    for g in P:
        for f in P[g].values(): idx |= set(f.index)
    return pd.DatetimeIndex(sorted(idx))

def per_date(P, g, pos_fn, h, cal):
    """Per-date sums over instruments of p*z and |p| for group g. pos_fn(f) -> array of positions (nan inactive)."""
    pos = {d: i for i, d in enumerate(cal)}
    num = np.zeros(len(cal)); den = np.zeros(len(cal)); n = np.zeros(len(cal))
    for f in P[g].values():
        p = np.asarray(pos_fn(f), dtype=float)
        z = f[f"z{h}"].to_numpy()
        m = np.isfinite(p) & np.isfinite(z) & (p != 0)
        if not m.any(): continue
        ix = np.fromiter((pos[d] for d in f.index[m]), int)
        np.add.at(num, ix, p[m] * z[m]); np.add.at(den, ix, np.abs(p[m])); np.add.at(n, ix, 1)
    return num, den, n

def boot(num, den, block, reps=400, seed=0):
    T = len(num); rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block)); starts = rng.integers(0, max(T - block, 1), size=(reps, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :T]
    s = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return np.percentile(s, [2.5, 97.5])

def score(P, pos_fn, h=10, groups=GROUPS, cal=None, do_boot=True):
    cal = cal if cal is not None else calendar(P)
    era = era_of(cal)
    rows = []
    for g in groups:
        num, den, n = per_date(P, g, pos_fn, h, cal)
        for e in ("E1", "E2", "E3"):
            m = era == e
            D = den[m].sum()
            edge = num[m].sum() / D if D > 0 else np.nan
            lo, hi = boot(num[m], den[m], h) if (do_boot and D > 0) else (np.nan, np.nan)
            rows.append(dict(group=g, era=e, edge=edge, lo=lo, hi=hi, n=int(n[m].sum()), cov=float((n[m] > 0).mean())))
    return pd.DataFrame(rows)

def fmt(t, title=""):
    t = t.copy()
    t["s"] = t.apply(lambda r: "" if not np.isfinite(r.edge) else f"{r.edge:+.3f}{'*' if (r.lo > 0 or r.hi < 0) else ' '}", axis=1)
    pv = t.pivot(index="group", columns="era", values="s")
    e = t.pivot(index="group", columns="era", values="edge")
    pv.loc["avg_nc"] = e.loc[[g for g in NC if g in e.index]].mean().map(lambda v: f"{v:+.3f}")
    pv["n"] = t.groupby("group").n.sum().reindex(pv.index)
    return (title + "\n" if title else "") + pv.to_string()

def xs_ic(P, g, sig_col_fn, h=10, min_names=15):
    """Cross-sectional Spearman IC per date for group g; returns Series of IC by date."""
    frames = []
    for t, f in P[g].items():
        s = pd.Series(np.asarray(sig_col_fn(f), float), index=f.index)
        frames.append(pd.DataFrame({"t": t, "s": s, "r": f[f"r{h}"], "live": f["live"]}))
    D = pd.concat(frames); D = D[D.live & D.s.notna() & D.r.notna()]
    D["d"] = D.index
    def ic(x):
        return x.s.rank().corr(x.r.rank()) if len(x) >= min_names else np.nan
    return D.groupby("d").apply(ic).dropna()

def ic_table(P, fns, h=10, groups=("nse", "us")):
    rows = []
    for name, fn in fns.items():
        for g in groups:
            s = xs_ic(P, g, fn, h)
            er = era_of(s.index)
            for e in ("E1", "E2", "E3"):
                x = s[er == e]
                # non-overlapping sampling every h days for the t
                xs = x.iloc[::h]
                t = xs.mean() / (xs.std() / np.sqrt(len(xs))) if len(xs) > 5 else np.nan
                rows.append(dict(sig=name, group=g, era=e, ic=x.mean(), t=t, days=len(x)))
    return pd.DataFrame(rows)


# ── UNBIASED SCORING (the fresh audit's standard) ─────────────────────────────
# The old score demeaned forward returns by the era's own mean: on random walks that
# alone makes persistent momentum readings score as reversion (up to -0.04σ at h 40).
# Here returns are NOT demeaned. They are scaled by trailing (causal) volatility, and the
# POSITION is demeaned by its own per-instrument era mean, which removes drift/beta while
# staying exactly unbiased under the null (returns independent of the reading).
def add_u(f, hs=HS):
    lr = np.log(f["open"].where(f["open"] > 0))
    vol = np.log(f["close"]).diff().rolling(60, min_periods=40).std()
    for h in hs:
        fw = lr.shift(-(1 + h)) - lr.shift(-1)
        f[f"u{h}"] = fw / (vol * np.sqrt(h))

def score_u(P, pos_fn, h=10, groups=GROUPS, cal=None, do_boot=True, demean=True):
    cal = cal if cal is not None else calendar(P)
    era = era_of(cal); posi = {d: i for i, d in enumerate(cal)}
    rows = []
    for g in groups:
        num = np.zeros(len(cal)); den = np.zeros(len(cal)); n = np.zeros(len(cal))
        for f in P[g].values():
            p = np.asarray(pos_fn(f), dtype=float).copy()
            u = f[f"u{h}"].to_numpy()
            m = np.isfinite(p) & np.isfinite(u)
            if demean:
                fe = f["era"].to_numpy()
                for e in ("E1", "E2", "E3"):
                    k = m & (fe == e)
                    if k.sum() > 20: p[k] = p[k] - p[k].mean()
            m &= p != 0
            if not m.any(): continue
            ix = np.fromiter((posi[d] for d in f.index[m]), int)
            np.add.at(num, ix, p[m] * u[m]); np.add.at(den, ix, np.abs(p[m])); np.add.at(n, ix, 1)
        for e in ("E1", "E2", "E3"):
            mm = era == e; D = den[mm].sum()
            edge = num[mm].sum() / D if D > 0 else np.nan
            lo, hi = boot(num[mm], den[mm], max(h, 60)) if (do_boot and D > 0) else (np.nan, np.nan)
            rows.append(dict(group=g, era=e, edge=edge, lo=lo, hi=hi, n=int(n[mm].sum())))
    return pd.DataFrame(rows)

def xs_ic_u(P, g, fn, h=10, min_names=15):
    frames = []
    for t, f in P[g].items():
        frames.append(pd.DataFrame({"s": np.asarray(fn(f), float), "r": f[f"r{h}"].to_numpy(), "live": f["live"].to_numpy()}, index=f.index))
    D = pd.concat(frames); D = D[D.live & np.isfinite(D.s) & np.isfinite(D.r)]
    rs = D.groupby(level=0).s.rank(); rr = D.groupby(level=0).r.rank()
    cnt = D.groupby(level=0).s.transform("size")
    X = pd.DataFrame({"a": rs, "b": rr, "c": cnt}); X = X[X.c >= min_names]
    return X.groupby(level=0).apply(lambda x: x.a.corr(x.b)).dropna()


# ── CAUSAL SCORERS. mode "xs": returns net of the group's same-date mean (market-neutral);
#    mode "tc": position centred on its own trailing 250-bar mean (time-series, causal).
def xs_demean(P, h, groups=None):
    for g in (groups or P):
        frames = [f[f"u{h}"].rename(t) for t, f in P[g].items()]
        M = pd.concat(frames, axis=1)
        live = pd.concat([f["live"].rename(t) for t, f in P[g].items()], axis=1).astype("boolean").fillna(False).astype(bool)
        mu = M.where(live).mean(axis=1)
        for t, f in P[g].items():
            f[f"x{h}"] = f[f"u{h}"] - mu.reindex(f.index)

def score_c(P, pos_fn, h=10, mode="xs", groups=GROUPS, cal=None, do_boot=True, center=250):
    cal = cal if cal is not None else calendar(P)
    era = era_of(cal); posi = {d: i for i, d in enumerate(cal)}
    col = f"x{h}" if mode == "xs" else f"u{h}"
    rows = []
    for g in groups:
        num = np.zeros(len(cal)); den = np.zeros(len(cal)); n = np.zeros(len(cal))
        for f in P[g].values():
            p = pd.Series(np.asarray(pos_fn(f), dtype=float), index=f.index)
            if mode == "tc":
                p = p - p.rolling(center, min_periods=center // 2).mean().shift(1)
            p = p.to_numpy(); u = f[col].to_numpy()
            m = np.isfinite(p) & np.isfinite(u) & (p != 0)
            if not m.any(): continue
            ix = np.fromiter((posi[d] for d in f.index[m]), int)
            np.add.at(num, ix, p[m] * u[m]); np.add.at(den, ix, np.abs(p[m])); np.add.at(n, ix, 1)
        for e in ("E1", "E2", "E3"):
            mm = era == e; D = den[mm].sum()
            edge = num[mm].sum() / D if D > 0 else np.nan
            lo, hi = boot(num[mm], den[mm], max(h, 60)) if (do_boot and D > 0) else (np.nan, np.nan)
            rows.append(dict(group=g, era=e, edge=edge, lo=lo, hi=hi, n=int(n[mm].sum())))
    return pd.DataFrame(rows)
