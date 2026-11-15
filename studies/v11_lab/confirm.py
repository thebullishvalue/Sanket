"""The confirmation test (CONFIRM.md, committed before the data was fetched).

    python3 studies/v11_lab/confirm.py <cache dir>
"""
import os, sys, glob, json, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
SANKET = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, SANKET)
import numpy as np, pandas as pd
import edge, engine as eng, samanvaya as sv                  # noqa: E402

CACHE = sys.argv[1]
H = 10
LEVEL = 1 - 0.05 / 3                                        # Bonferroni over the three hypotheses
nb, _lvl, seed = edge.block_bootstrap_ci.__defaults__
edge.block_bootstrap_ci.__defaults__ = (nb, LEVEL, seed)
KINDS = ("H1", "H2", "H3", "C0")
for k, (b, s) in {"H1": ("P1 ▲ Pragati capitulation turn", None), "H2": ("band break up in a downtrend", None),
                  "H3": (None, "band break down in an uptrend"), "C0": ("control: new 20-bar low", "control: new 20-bar high")}.items():
    for side, sgn, lab in (("buy", 1.0, b), ("sell", -1.0, s)):
        if lab:
            edge.SLICES[f"{k}_{side}"] = (sgn, k); edge.SLICE_LABEL[f"{k}_{side}"] = f"{k} · {'▲' if sgn > 0 else '▼'} {lab}"


def once(x, gap=H):
    x = np.asarray(x, bool); out = np.zeros_like(x); last = -10 ** 9
    for t in np.where(x)[0]:
        if t - last > gap:
            out[t] = True; last = t
    return out


def enter(m):
    m = np.asarray(m, bool)
    return m & ~np.r_[False, m[:-1]]


def main():
    drv = sv.prepare_drivers(pd.read_pickle(os.path.join(HERE, "drivers_daily.pkl")), "D")
    st = eng.settings_for(None, None, "Daily")
    evs, base, rets, mkt = [], {}, {}, {}
    files = sorted(glob.glob(os.path.join(CACHE, "*.csv")))
    for i, fn in enumerate(files):
        m, t = os.path.basename(fn)[:-4].split("__", 1)
        d = pd.read_csv(fn, index_col=0, parse_dates=True)
        d = d[(d.close > 0) & (d.high >= d.low)]
        d = d[d.close.pct_change().abs().fillna(0) < 0.6]                    # bad prints, as lib.load
        if len(d) < 1500:
            continue
        lo = d[["open", "high", "low", "close"]]
        f = eng.compute_frame(d.rename(columns=str.title), drv, t, st, intraday={}).reindex(d.index)
        lc = np.log(lo.close.to_numpy(float))
        T = f["lg_trend"].to_numpy(float); state = f["lg_state"].fillna(-1).astype(int).to_numpy()
        cone = np.where(state >= 0, state % 3, -1)
        cell = f["cvg_cell"].fillna(-1).astype(int).to_numpy(); vph = f["cvg_vph"].fillna(0).astype(int).to_numpy()
        quiet = f["cvg_quiet"].fillna(False).astype(bool).to_numpy(); ok = f["stack_ok"].fillna(False).astype(bool).to_numpy()
        s = pd.Series(lc)
        nlo = lc < s.shift(1).rolling(19).min().to_numpy(); nhi = lc > s.shift(1).rolling(19).max().to_numpy()
        live = np.isfinite(T) & (state >= 0)
        sig = {"H1": (once(enter((cell == 0) & (vph == 1)) & ~quiet & ok), None),
               "H2": (once(live & (T < -0.5) & enter(cone == 2)), None),
               "H3": (None, once(live & (T > 0.5) & enter(cone == 0))),
               "C0": (once(nlo & live), once(nhi & live))}
        for k in "abc":
            sig[k.upper()] = (f[f"{k}_buy"].fillna(False).astype(bool).to_numpy(), f[f"{k}_sell"].fillna(False).astype(bool).to_numpy())
        close = lo.close
        fwd = (close.shift(-1 - H) / close.shift(-1) - 1.0).to_numpy()
        for k, (b, sl) in sig.items():
            for m0, sgn in ((b, 1.0), (sl, -1.0)):
                if m0 is None:
                    continue
                mm = m0 & np.isfinite(fwd)
                if mm.any():
                    evs.append(pd.DataFrame({"date": lo.index[mm], "side": sgn, "kind": k, "fwd": fwd[mm], "symbol": t}))
        base[t] = edge.symbol_baseline(close, H); rets[t] = close.pct_change(); mkt[t] = m
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(files)}", flush=True)
    ev = pd.concat(evs, ignore_index=True)
    S = edge.measure(ev, base, pd.DataFrame(rets), universe="fresh S&P 500 + Nifty 200", selected_index=None,
                     timeframe="daily", iclass="equity", length=20, trigger="confirmation", horizon=H,
                     cost_bps=eng.COST_BPS, n_symbols_universe=len(base), n_bars_median=2500)
    sc = edge._score_events(ev, base, ev["symbol"].unique(), ev["date"].min(), ev["date"].max(), H)
    sc["market"] = sc["symbol"].map(mkt)
    eras = (("2005-12", None, "2013-01-01"), ("2013-19", "2013-01-01", "2020-01-01"), ("2020-26", "2020-01-01", None))
    rows = []
    for key in edge.SLICES:
        k, side = key.split("_")
        r = S.get(key, "full")
        if r is None:
            continue
        x = sc[(sc.kind == k) & (sc.side == (1 if side == "buy" else -1))]
        mk = x.groupby("market").score.mean()
        em = []
        for _, a, b in eras:
            q = x
            if a: q = q[q.date >= pd.Timestamp(a)]
            if b: q = q[q.date < pd.Timestamp(b)]
            em.append(q.score.mean() if len(q) >= 20 else np.nan)
        # raw (not drift-removed) mean 10-bar return in the signal's direction, for the trader's view
        raw = (x.side * x.fwd).mean() * 100 if "fwd" in x else np.nan
        rows.append(dict(signal=k, side="▲" if side == "buy" else "▼", events=r.n_events, names=x.symbol.nunique(),
                         edge=r.edge, ci_lo=r.ci_lo, ci_hi=r.ci_hi, net=r.net, hit=r.hit, n_eff=r.n_eff,
                         us=mk.get("us", np.nan), nse=mk.get("nse", np.nan),
                         **{f"era {n}": v for (n, _, _), v in zip(eras, em)}, raw_pct=raw))
    R = pd.DataFrame(rows)
    c0 = R[(R.signal == "C0") & (R.side == "▲")].edge.iloc[0]

    def verdict(r):
        if r.signal not in ("H1", "H2", "H3"):
            return ""
        eras_pos = sum(1 for c in R.columns if c.startswith("era") and np.isfinite(r[c]) and r[c] > 0)
        ok = r.ci_lo > 0 and r.net > 0 and r.us > 0 and r.nse > 0 and eras_pos >= 2
        if r.signal == "H1":
            ok = ok and r.ci_lo > c0
        return "CONFIRMED" if ok else "not confirmed"
    R["result"] = R.apply(verdict, axis=1)
    R.to_csv(os.path.join(HERE, "out", "confirm.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(f"fresh universe · {len(base)} names ({sum(v == 'us' for v in mkt.values())} US, "
          f"{sum(v == 'nse' for v in mkt.values())} NSE) · intervals at {LEVEL:.1%}")
    print(R.round(4).to_string(index=False))
    json.dump({"names": len(base), "level": LEVEL, "control_c0_buy_edge": float(c0),
               "results": {f"{a} {b}": c for a, b, c in zip(R.signal, R.side, R.result) if c}},
              open(os.path.join(HERE, "out", "confirm.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
