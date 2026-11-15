"""v13 lab - harvest every pool event with every feature and its Edge Study score, on one dataset.

    python3 studies/v13_lab/harvest.py <Indicator/research> RD|RH|FD|FH [fresh daily cache] [fresh hourly cache]

RD / RH  research daily / hourly (151 instruments)      FD / FH  fresh daily / hourly (572 / 595 stocks)
Writes out/events_<ds>.pkl: one row per (pool, side, bar) with the features, the current sets' flags,
and `score` (edge._score_events: next-bar entry, 10-bar hold, causal drift removed, vol-normalised).
"""
import os, sys, glob, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
SANKET = os.path.abspath(os.path.join(HERE, "..", ".."))
RESEARCH, DS = os.path.abspath(sys.argv[1]), sys.argv[2]
FD_CACHE = sys.argv[3] if len(sys.argv) > 3 else None
FH_CACHE = sys.argv[4] if len(sys.argv) > 4 else None
sys.argv = [sys.argv[0], RESEARCH, "data1h" if DS in ("RH", "FH") else "data", os.path.join(HERE, "out")]
sys.path[:0] = [SANKET, os.path.join(SANKET, "studies", "v10_turns")]
import numpy as np, pandas as pd
import turns_vp as tv                                     # noqa: E402  (also chdirs to RESEARCH)
import ledger, edge, engine as eng, samanvaya as sv       # noqa: E402

H = 10
HOURLY = DS in ("RH", "FH")
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)


def yname(stem):
    s = stem
    if s.startswith("_"): s = "^" + s[1:]
    if s.endswith("_X"): s = s[:-2] + "=X"
    if s.endswith("_F"): s = s[:-2] + "=F"
    return s


def load():
    if DS in ("RD", "RH"):
        from lib import load as lload
        D = lload("data1h" if HOURLY else "data", 2500 if HOURLY else 1500)
        return {s: (c, d) for s, (c, d) in D.items()}
    out = {}
    for fn in sorted(glob.glob(os.path.join(FH_CACHE if HOURLY else FD_CACHE, "*.csv"))):
        m, t = os.path.basename(fn)[:-4].split("__", 1)
        d = pd.read_csv(fn, index_col=0, parse_dates=True)
        d = d[(d.close > 0) & (d.high >= d.low)]
        d = d[d.close.pct_change().abs().fillna(0) < 0.6]
        if len(d) >= 1500:
            out[t] = ("us" if m == "us" else "nse", d)
    return out


def features(sym, d, drv):
    lo = d[["open", "high", "low", "close"]]
    L = ledger.compute(lo, H)
    c = lo.close.to_numpy(float); lc = np.log(c)
    f = pd.DataFrame(index=d.index)
    T = L.lg_trend.to_numpy(float); u = L.lg_u.to_numpy(float)
    qu, qd = L.lg_band_up.to_numpy(float), L.lg_band_dn.to_numpy(float)
    st = L.lg_state.fillna(-1).astype(int).to_numpy()
    sig = L.lg_sigma.to_numpy(float)
    f["T"] = T; f["ladder"] = L.lg_ladx.to_numpy(float)
    f["rungs_up"] = L.lg_lad_up.to_numpy(float); f["rungs_dn"] = L.lg_lad_dn.to_numpy(float)
    f["u"] = u
    f["u_min10"] = pd.Series(u).shift(1).rolling(10, min_periods=3).min().to_numpy()
    f["u_max10"] = pd.Series(u).shift(1).rolling(10, min_periods=3).max().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        f["stretch"] = np.where(u >= 0, u / qu, u / qd)                 # the h-rung's band fraction
        f["vs_mid"] = (lc - np.log(L.lg_mid.to_numpy(float))) / (sig * np.sqrt(H))
    f["risk_rank"] = L.lg_prk.to_numpy(float); f["risk_mom"] = L.lg_rmom.to_numpy(float)
    f["cone"] = np.where(st >= 0, st % 3, -1)
    f["z"] = L.lg_z.to_numpy(float)
    s = pd.Series(lc)
    nlo = lc < s.shift(1).rolling(19).min().to_numpy(); nhi = lc > s.shift(1).rolling(19).max().to_numpy()
    f["new_low5"] = pd.Series(nlo.astype(float)).rolling(5, min_periods=1).max().to_numpy() > 0
    f["new_high5"] = pd.Series(nhi.astype(float)).rolling(5, min_periods=1).max().to_numpy() > 0
    f["ret5"] = (lc - s.shift(5).to_numpy()) / (sig * np.sqrt(5))
    vcat, vnum, tpo = tv.vp_readings(d, sig, "data1h" if HOURLY else "data")
    f["rel_vol"] = vnum["rel volume"].to_numpy(float)
    f["vs_vwap"] = vnum["vs vwap (σ)"].to_numpy(float)
    f["prior_session"] = pd.Categorical(vcat["prior session"].to_numpy())
    if not HOURLY:
        df = d.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]]
        g = eng.compute_frame(df, drv, yname(sym), eng.settings_for(None, None, "Daily"), intraday={}).reindex(d.index)
        f["cell"] = g["cvg_cell"].fillna(-1).astype(int).to_numpy()
        f["c_tape"] = pd.to_numeric(g["c_tape"], errors="coerce").to_numpy()
        f["v_tape"] = pd.to_numeric(g["v_tape"], errors="coerce").to_numpy()
        f["trace"] = pd.to_numeric(g["trace"], errors="coerce").to_numpy()
    for k in ("a", "b", "c", "d"):
        f[f"{k}_buy"] = L[f"{k}_buy"].to_numpy(bool); f[f"{k}_sell"] = L[f"{k}_sell"].to_numpy(bool)
    f["live"] = np.isfinite(T) & np.isfinite(f["ladder"]) & np.isfinite(f["risk_rank"]) & \
        np.isfinite(f["risk_mom"]) & (st >= 0) & np.isfinite(u)
    flip = L.lg_flip.fillna(0).astype(int).to_numpy()
    cone = f["cone"].to_numpy()
    above, below = cone == 2, cone == 0
    st_enter = (st >= 0) & (st != np.r_[-1, st[:-1]]) & np.isfinite(f["z"].to_numpy())
    pools = {("FLIP", 1): flip == 1, ("FLIP", -1): flip == -1,
             ("BAND", 1): above & ~np.r_[False, above[:-1]], ("BAND", -1): below & ~np.r_[False, below[:-1]],
             ("STATE", 1): st_enter, ("STATE", -1): st_enter}
    live = f["live"].to_numpy()
    fwd = (lo.close.shift(-1 - H) / lo.close.shift(-1) - 1.0).to_numpy()
    rows = []
    for (pool, side), m in pools.items():
        idx = np.where(m & live & np.isfinite(fwd))[0]
        if not len(idx):
            continue
        prev = np.r_[-10 ** 9, idx[:-1]]
        r = f.iloc[idx].copy()
        r["since_prev"] = idx - prev                     # bars since the previous event of this pool
        r["pool"], r["side"], r["fwd"], r["symbol"] = pool, float(side), fwd[idx], sym
        rows.append(r)
    return (pd.concat(rows) if rows else None), L, lo


def main():
    D = load()
    drv = None if HOURLY else sv.prepare_drivers(pd.read_pickle(os.path.join(SANKET, "studies", "v11_lab", "drivers_daily.pkl")), "D")
    parts, base, bars = [], {}, 0
    for i, (sym, (cls, d)) in enumerate(D.items()):
        d = d[d.close.notna() & d.high.notna() & d.low.notna() & (d.close > 0)]
        ev, L, lo = features(sym, d, drv)
        bars += len(d)
        if ev is not None:
            ev["cls"] = cls
            parts.append(ev)
        base[sym] = edge.symbol_baseline(lo.close, H)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(D)}", flush=True)
    E = pd.concat(parts)
    E["date"] = E.index; E = E.reset_index(drop=True)
    E["kind"] = E["pool"]
    sc = edge._score_events(E, base, E["symbol"].unique(), E["date"].min(), E["date"].max(), H)
    sc.attrs["bars"] = bars; sc.attrs["instruments"] = len(D)
    sc.to_pickle(os.path.join(OUT, f"events_{DS}.pkl"))
    print(DS, len(D), "instruments", bars, "bars", len(sc), "scored events")
    print(sc.groupby(["pool", "side"]).score.agg(["size", "mean"]).round(4).to_string())


if __name__ == "__main__":
    main()
