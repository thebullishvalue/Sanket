"""Edge Study of Sanket's three signal sets on the Indicator repo's 151-instrument research universe.

    python3 studies/v10_sets/sets_edge.py <path to Indicator/research> > studies/v10_sets/sets_edge.txt

The signals come from ledger.py - the exact code the screen fires - on each instrument's own OHLC
(the sets read nothing else). The verdicts come from edge.measure - the exact code the app's Edge
Study runs: entry the bar after the signal, a 10-bar hold, each symbol's trailing 500-return drift
removed causally, vol-normalised, a block bootstrap over dates, power stated. On top: eras, asset
classes, and the sets as rules against buy and hold, net of costs.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
RESEARCH = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "..", "Indicator", "research"))
sys.path.insert(0, RESEARCH)
import numpy as np, pandas as pd
import ledger, edge, engine as eng
os.chdir(RESEARCH)
from lib import load, CLASSES                                 # noqa: E402  (the Indicator repo's loader)
pd.set_option("display.width", 250)
H = eng.HORIZON
ERAS = [("2003", "2012"), ("2013", "2019"), ("2020", "2026")]
KEYS = [f"{k}_{s}" for k in "ABC" for s in ("buy", "sell")]


def harvest(D):
    """Events, baselines and returns exactly as edge.symbol_events / symbol_baseline build them."""
    evs, base, rets, frames = [], {}, {}, {}
    for sym, (cls, d) in D.items():
        lo = d[["open", "high", "low", "close"]]
        lo = lo[lo.close.notna() & lo.high.notna() & lo.low.notna() & (lo.close > 0)]
        L = ledger.compute(lo, H)
        close = lo["close"]
        fwd = (close.shift(-1 - H) / close.shift(-1) - 1.0).to_numpy()
        for k in "ABC":
            for s, sgn in (("buy", 1.0), ("sell", -1.0)):
                m = L[f"{k.lower()}_{s}"].to_numpy(bool) & np.isfinite(fwd)
                if m.any():
                    evs.append(pd.DataFrame({"date": lo.index[m], "side": sgn, "kind": k, "fwd": fwd[m],
                                             "symbol": sym, "cls": cls}))
        base[sym] = edge.symbol_baseline(close, H)
        rets[sym] = close.pct_change()
        frames[sym] = (cls, lo, L)
    ev = pd.concat(evs, ignore_index=True)
    return ev, base, pd.DataFrame(rets), frames


def study(ev, base, ret, D, label, cost):
    S = edge.measure(ev, base, ret, universe="Indicator research universe", selected_index=None,
                     timeframe=label, iclass="mixed", length=20, trigger="sets A / B / C", horizon=H,
                     cost_bps=cost, n_symbols_universe=len(D),
                     n_bars_median=int(np.median([len(f[1]) for f in D.values()])))
    rows = []
    for key in KEYS:
        lbl, _k, detail = S.verdict(key)
        for era in ("discovery", "holdout", "full"):
            r = S.get(key, era)
            if r is None:
                continue
            rows.append(dict(slice=edge.SLICE_LABEL[key], era=era, events=r.n_events, dates=r.n_dates, n_eff=round(r.n_eff),
                             edge=r.edge, ci=f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]", net=r.net, hit=r.hit, mde=r.mde,
                             sig="yes" if r.significant else ("ANTI" if r.anti else "no"), verdict=lbl if era == "full" else ""))
    return S, pd.DataFrame(rows)


def slices(ev, base, label, D):
    """Every set and side by era and by asset class, on the Edge Study's own scores."""
    sc = edge._score_events(ev, base, ev["symbol"].unique(), ev["date"].min(), ev["date"].max(), H)
    sc["key"] = sc["kind"] + "_" + np.where(sc["side"] > 0, "buy", "sell")
    sc["grp"] = np.where(sc["cls"].isin(["us", "nse", "index"]), "equity", np.where(sc["cls"] == "crypto", "crypto", "macro"))
    def tt(q):
        g = q.groupby("date")["score"]; m, n = g.mean(), g.size(); x = (m * n).values
        mu = x.sum() / n.sum(); e = x - mu * n.values; v = (e * e).sum()
        for L in range(1, H + 1):
            v += 2 * (1 - L / (H + 1)) * (e[L:] * e[:-L]).sum()
        return mu / (np.sqrt(v) / n.sum()) if v > 0 else np.nan
    print(f"\nB. {label} · BY SLICE · score = drift-free, vol-normalised 10-bar return in the signal's direction "
          f"(gross, σ). t clustered by date, Newey-West over 10 bars")
    g = sc.groupby("key")
    t = pd.DataFrame(dict(events=g.size(), names=g.symbol.nunique(), mean=g.score.mean(), t=g.apply(tt),
                          hit=g.score.apply(lambda x: (x > 0).mean())))
    if label == "DAILY":
        sc["era"] = pd.cut(sc.date.dt.year, [1999, 2012, 2019, 2030], labels=["03-12", "13-19", "20-26"]).astype(str)
        t = t.join(sc.groupby(["key", "era"], observed=True).score.mean().unstack())
        t = t.join(sc.groupby(["key", "era"], observed=True).apply(tt).unstack().add_prefix("t "))
    print(t.reindex(KEYS).round(3).to_string())
    print("\n   by asset class (mean score, σ; events in brackets)")
    m = sc.groupby(["key", "cls"]).score.mean().unstack(); n = sc.groupby(["key", "cls"]).size().unstack()
    print(pd.DataFrame({c: [f"{m.loc[k, c]:+.3f} ({int(n.loc[k, c])})" if k in m.index and c in m.columns and pd.notna(m.loc[k, c]) else "—"
                            for k in KEYS] for c in CLASSES}, index=KEYS).to_string())
    print("\n   ex crypto, by group")
    print(sc[sc.grp != "crypto"].groupby(["key", "grp"]).score.agg(["mean", "size"]).unstack().reindex(KEYS).round(3).to_string())
    return sc


def rules(frames, label, ann):
    """Each set as a rule: on a signal hold its side for 10 bars (a new signal restarts it), entry the
    next bar's open, net of 5 bp a side (10 crypto). Mean across instruments against buy and hold."""
    rows = []
    for sym, (cls, lo, L) in frames.items():
        o = lo["open"]; ro = (o.shift(-2) / o.shift(-1) - 1)
        ok = L["lg_sigma"].notna()
        hold = ro.where(ok).dropna()
        cost = 0.001 if cls == "crypto" else 0.0005
        for k in "ABC":
            sg = pd.Series(np.where(L[f"{k.lower()}_buy"], 1.0, np.where(L[f"{k.lower()}_sell"], -1.0, np.nan)), index=lo.index)
            last = sg.ffill(limit=H - 1)
            for mode in ("long only", "long / short"):
                p = (last.clip(lower=0) if mode == "long only" else last).fillna(0.0)
                pnl = (p * ro - cost * p.diff().abs().fillna(0)).where(ok).dropna()
                if len(pnl) > 250 and pnl.std() > 0 and hold.std() > 0:
                    eq = (1 + pnl).cumprod(); eh = (1 + hold).cumprod()
                    rows.append(dict(set=k, mode=mode, sym=sym, cls=cls, inmkt=(p[ok] != 0).mean(),
                                     sh=pnl.mean() / pnl.std() * np.sqrt(ann), h_sh=hold.mean() / hold.std() * np.sqrt(ann),
                                     dd=(eq / eq.cummax() - 1).min(), h_dd=(eh / eh.cummax() - 1).min(),
                                     ret=(eq.iloc[-1] ** (ann / len(pnl)) - 1)))
    X = pd.DataFrame(rows); g = X.groupby(["set", "mode"])
    print(f"\nC. {label} · AS RULES · hold the signal's side 10 bars, next-open entry, net of costs · mean across instruments")
    print(pd.DataFrame(dict(in_market=g.inmkt.mean(), sharpe=g.sh.mean(), hold_sharpe=g.h_sh.mean(), cagr=g.ret.mean(),
                            dd=g.dd.mean(), hold_dd=g.h_dd.mean(),
                            beats_hold=g.apply(lambda q: (q.sh > q.h_sh).mean()),
                            positive=g.apply(lambda q: (q.sh > 0).mean()))).round(3).to_string())


for label, root, minbars, ann in (("DAILY", "data", 1500, 252), ("HOURLY", "data1h", 2500, 252 * 7)):
    D = load(root, minbars)
    ev, base, ret, frames = harvest(D)
    tot = sum(len(f[1]) for f in frames.values())
    print(f"\n{'=' * 100}\n{label} · {len(D)} instruments · {tot:,} bars · {ev.date.min().date()} to {ev.date.max().date()}")
    print("events per 1,000 bars: " + " · ".join(f"{k} {((ev.kind == k[0]) & (ev.side == (1 if k[2:] == 'buy' else -1))).sum() / tot * 1000:.2f}"
                                                 for k in KEYS))
    S, T = study(ev, base, ret, D, label, eng.COST_BPS)
    print(f"\nA. {label} · SANKET'S EDGE STUDY (edge.measure) · discovery = first 60% of event dates, holdout = last 40% "
          f"(split {S.split_date}) · participation ratio {S.part_ratio:.1f} · cost {eng.COST_BPS:g} bp")
    print(T.round(4).to_string(index=False))
    slices(ev, base, label, D)
    rules(frames, label, ann)
