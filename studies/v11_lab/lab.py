"""Signal lab v11 - see PROTOCOL.md (committed before this ran).

    python3 studies/v11_lab/lab.py <Indicator/research> daily|hourly|weekly

daily  : sets A/B/C + P1, P2, K1-K7      hourly : sets + K1-K7      weekly : sets + P1, P2
Writes out/edge_<mode>.csv (every slice x era) and out/scores_<mode>.pkl (per-event scores for the
breadth stage).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.abspath(sys.argv[1]); MODE = sys.argv[2]
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
sys.argv = [sys.argv[0], RESEARCH, "data1h" if MODE == "hourly" else "data", OUT]      # for turns.py
SANKET = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [SANKET, os.path.join(SANKET, "studies", "v10_turns")]
import numpy as np, pandas as pd
import turns as tu, turns_vp as tv                       # noqa: E402  (chdir to RESEARCH happens here)
import ledger, edge, engine as eng, samanvaya as sv      # noqa: E402

H = 10
KS = ("K1", "K2", "K3", "K4", "K5", "K6", "K7")
PS = ("P1", "P2")
CANDS = {"daily": PS + KS, "hourly": KS, "weekly": PS}[MODE]
LABEL = {"P1": "Pragati capitulation turn (v9.3)", "P2": "entering capitulation", "K1": "capitulation volume",
         "K2": "capitulation, confirmed", "K3": "trend pullback", "K4": "pullback, confirmed",
         "K5": "calm breakout through value", "K6": "cross-sectional reversal", "K7": "strong trend entry"}
EQUITY = ("us", "nse", "index")


def yname(stem):
    s = stem
    if s.startswith("_"): s = "^" + s[1:]
    if s.endswith("_X"): s = s[:-2] + "=X"
    if s.endswith("_F"): s = s[:-2] + "=F"
    return s


def once(x, gap=H):
    x = np.asarray(x, bool); out = np.zeros_like(x); last = -10 ** 9
    for t in np.where(x)[0]:
        if t - last > gap:
            out[t] = True; last = t
    return out


def enter(m):
    m = np.asarray(m, bool)
    return m & ~np.r_[False, m[:-1]]


def recent(m, n=H):
    return pd.Series(np.asarray(m, float)).rolling(n, min_periods=1).max().to_numpy() > 0


def weekly(d):
    w = d.resample("W-FRI").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return w.dropna(subset=["close"])


def pragati(d, sym, drv, daily=None):
    """Pragati's grid on these bars (Sanket's own engine), as P1 / P2 masks."""
    df = d.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]]
    st = eng.settings_for(None, None, "Weekly" if MODE == "weekly" else "Daily")
    dd = daily.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]] if daily is not None else None
    f = eng.compute_frame(df, drv, yname(sym), st, daily=dd, intraday={}).reindex(d.index)
    cell = f["cvg_cell"].fillna(-1).astype(int).to_numpy()
    vph = f["cvg_vph"].fillna(0).astype(int).to_numpy()
    quiet = f["cvg_quiet"].fillna(False).astype(bool).to_numpy()
    ok = f["stack_ok"].fillna(False).astype(bool).to_numpy()
    cap = (cell == 0) & (vph == 1)
    return {"P1": (once(enter(cap) & ~quiet & ok), once(enter(cell == 2) & ok)),
            "P2": (once(enter(cell == 0) & ok), np.zeros(len(d), bool))}


def k_signals(d, L):
    c = d.close.to_numpy(float); lc = np.log(c)
    sig = L.lg_sigma.to_numpy(float)
    pcat, pnum, live = tu.readings(L, c)
    vcat, vnum, tpo = tv.vp_readings(d, sig, "data1h" if MODE == "hourly" else "data")
    prk, T, u = (L[k].to_numpy(float) for k in ("lg_prk", "lg_trend", "lg_u"))
    cone = L.lg_state.to_numpy(int) % 3                     # 0 below band, 1 inside, 2 above
    rv = vnum["rel volume"].to_numpy(float)
    nlo, nhi = tu.new_extremes(lc)
    stressed, calm = prk >= 200 / 3, prk < 100 / 3
    below, above = (cone == 0) & (L.lg_state.to_numpy(int) >= 0), cone == 2
    up = np.r_[False, (u[1:] >= 0) & (u[:-1] < 0)]; dn = np.r_[False, (u[1:] < 0) & (u[:-1] >= 0)]
    k1b = nlo & (rv >= 2) & stressed & below
    k1s = nhi & (rv >= 2) & stressed & above
    ps = vcat["prior session"].to_numpy()
    xup = np.r_[False, (ps[1:] == "above pVAH") & (ps[:-1] != "above pVAH")]
    xdn = np.r_[False, (ps[1:] == "below pVAL") & (ps[:-1] != "below pVAL")]
    sig_ = {
        "K1": (k1b, k1s),
        "K2": (recent(k1b) & up, recent(k1s) & dn),
        "K3": ((T > 0.5) & enter(below), (T < -0.5) & enter(above)),
        "K4": ((T > 0.5) & recent(below) & up, (T < -0.5) & recent(above) & dn),
        "K5": (calm & (T >= 0) & (rv >= 1.2) & xup, calm & (T < 0) & (rv >= 1.2) & xdn),
        "K7": (enter(T > 0.5), enter(T < -0.5)),
    }
    out = {k: (once(b & live), once(s & live)) for k, (b, s) in sig_.items()}
    return out, pd.Series(np.where(live, u, np.nan), index=d.index)


def main():
    D = tu.load("data1h" if MODE == "hourly" else "data", 2500 if MODE == "hourly" else 1500)
    drv = None
    if MODE in ("daily", "weekly"):
        drv = sv.prepare_drivers(pd.read_pickle(os.path.join(HERE, "drivers_daily.pkl")), "W" if MODE == "weekly" else "D")
    per, upanel = {}, {}
    for sym, (cls, d0) in D.items():
        d0 = d0[d0.close.notna() & d0.high.notna() & d0.low.notna() & (d0.close > 0)]
        d = weekly(d0) if MODE == "weekly" else d0
        lo = d[["open", "high", "low", "close"]]
        L = ledger.compute(lo, H)
        sets = {k: (L[f"{k.lower()}_buy"].to_numpy(bool), L[f"{k.lower()}_sell"].to_numpy(bool)) for k in "ABC"}
        if any(c in CANDS for c in KS):
            ks, useries = k_signals(d, L)
            sets.update({k: v for k, v in ks.items() if k in CANDS})
            if cls in EQUITY:
                upanel[sym] = useries
        if any(c in CANDS for c in PS):
            sets.update(pragati(d, sym, drv, daily=d0 if MODE == "weekly" else None))
        per[sym] = (cls, lo, sets)
    # K6: cross-sectional reversal among equities, ranked on each date
    if "K6" in CANDS and upanel:
        U = pd.DataFrame(upanel)
        cnt = U.notna().sum(axis=1)
        R = U.rank(axis=1, pct=True).where(cnt >= 30)
        for sym in U.columns:
            r = R[sym].reindex(per[sym][1].index).to_numpy()
            bot = np.nan_to_num(r, nan=0.5) <= 0.10
            top = np.nan_to_num(r, nan=0.5) >= 0.90
            per[sym][2]["K6"] = (once(enter(bot)), once(enter(top)))
    for k in CANDS:
        for side, sgn in (("buy", 1.0), ("sell", -1.0)):
            edge.SLICES[f"{k}_{side}"] = (sgn, k)
            edge.SLICE_LABEL[f"{k}_{side}"] = f"{k} · {'▲' if sgn > 0 else '▼'} {LABEL[k]}"
    evs, base, rets = [], {}, {}
    grp = lambda c: "crypto" if c == "crypto" else ("equity" if c in EQUITY else "macro")    # noqa: E731
    for sym, (cls, lo, sets) in per.items():
        close = lo.close
        fwd = (close.shift(-1 - H) / close.shift(-1) - 1.0).to_numpy()
        for k, (b, s) in sets.items():
            if k in CANDS or k in "ABC":
                for m0, sgn in ((b, 1.0), (s, -1.0)):
                    m = np.asarray(m0, bool) & np.isfinite(fwd)
                    if m.any():
                        evs.append(pd.DataFrame({"date": lo.index[m], "side": sgn, "kind": k, "fwd": fwd[m],
                                                 "symbol": sym, "cls": cls, "grp": grp(cls)}))
        base[sym] = edge.symbol_baseline(close, H)
        rets[sym] = close.pct_change()
    ev = pd.concat(evs, ignore_index=True)
    S = edge.measure(ev, base, pd.DataFrame(rets), universe="Indicator research universe", selected_index=None,
                     timeframe=MODE, iclass="mixed", length=20, trigger="v11 lab", horizon=H, cost_bps=eng.COST_BPS,
                     n_symbols_universe=len(per), n_bars_median=int(np.median([len(v[1]) for v in per.values()])))
    tot = sum(len(v[1]) for v in per.values())
    rows = []
    for key in edge.SLICES:
        k, side = key.split("_")
        if k not in CANDS and k not in "ABC":
            continue
        lbl = S.verdict(side, k)[0]
        for era in ("discovery", "holdout", "full"):
            r = S.get(key, era)
            if r is None:
                continue
            rows.append(dict(signal=k, side=side, era=era, events=r.n_events, edge=r.edge, ci_lo=r.ci_lo, ci_hi=r.ci_hi,
                             net=r.net, hit=r.hit, mde=r.mde, significant=r.significant, anti=r.anti, verdict=lbl,
                             per_1k=1000 * ((ev.kind == k) & (ev.side == (1 if side == "buy" else -1))).sum() / tot))
    E = pd.DataFrame(rows)
    E["split"] = S.split_date
    E.to_csv(os.path.join(OUT, f"edge_{MODE}.csv"), index=False)
    sc = edge._score_events(ev, base, ev["symbol"].unique(), ev["date"].min(), ev["date"].max(), H)
    sc["holdout"] = sc["date"] >= pd.Timestamp(S.split_date)
    sc[["date", "symbol", "side", "kind", "score", "holdout"]].assign(
        grp=sc["symbol"].map({s: grp(v[0]) for s, v in per.items()})).to_pickle(os.path.join(OUT, f"scores_{MODE}.pkl"))
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print(f"{MODE} · {len(per)} instruments · split {S.split_date}")
    print(E[E.era != "full"].pivot_table(index=["signal", "side"], columns="era", values=["edge", "ci_lo", "events"]).round(4).to_string())


if __name__ == "__main__":
    main()
