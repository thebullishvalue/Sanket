"""The set variants of PROTOCOL.md, built from ledger.compute's columns. B0 and C0 must equal ledger's own b_* / c_*.

    python3 studies/v12_sets/variants.py <Indicator/research> daily|hourly|freshD|freshH [cache dir]

research daily / hourly: stage 1 (choose). freshD / freshH: stage 2 (confirm), from the cache dir.
Writes out/<mode>.csv with every variant x side x era.
"""
import os, sys, glob, json, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
SANKET = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, SANKET)
import numpy as np, pandas as pd
import ledger, edge, engine as eng          # noqa: E402

RESEARCH, MODE = os.path.abspath(sys.argv[1]), sys.argv[2]
CACHE = sys.argv[3] if len(sys.argv) > 3 else None
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
H = 10
VARIANTS = ("B0", "Bn", "C0", "Cx", "Cx5", "Cxn")
LEVEL = float(os.environ.get("V12_LEVEL", "0.95"))
nb, _l, seed = edge.block_bootstrap_ci.__defaults__
edge.block_bootstrap_ci.__defaults__ = (nb, LEVEL, seed)
for k in VARIANTS + ("A",):
    for side, sgn in (("buy", 1.0), ("sell", -1.0)):
        edge.SLICES[f"{k}_{side}"] = (sgn, k); edge.SLICE_LABEL[f"{k}_{side}"] = f"{k} {'▲' if sgn > 0 else '▼'}"
for k in ("B", "C"):                         # measured through their variants B0 / C0
    for side in ("buy", "sell"):
        edge.SLICES.pop(f"{k}_{side}", None)


def variants(L, c):
    flip = L.lg_flip.fillna(0).to_numpy(int); mid = L.lg_mid.to_numpy(float)
    calm = L.lg_calm.fillna(False).astype(bool).to_numpy()
    rm = L.lg_rmom.to_numpy(float); rising = L.lg_rising.fillna(False).astype(bool).to_numpy()
    lad = L.lg_ladx.to_numpy(float); T = L.lg_trend.to_numpy(float)
    quiet = calm & np.isfinite(rm) & ~rising
    lad_ok, t_ok = np.isfinite(lad), np.isfinite(T)
    prk_ok = np.isfinite(L.lg_prk.to_numpy(float)) & np.isfinite(rm)    # the risk tapes have readings
    core_b = (flip == 1) & (c > mid) & lad_ok & (lad >= 0)
    core_s = (flip == -1) & (c < mid) & lad_ok & (lad < 0)
    v = {
        "B0": (core_b & quiet, core_s & quiet),
        "Bn": (core_b & prk_ok, core_s & prk_ok),
        "C0": (core_b & quiet & t_ok & (T >= 0), core_s & quiet & t_ok & (T < 0)),
        "Cx": (core_b & quiet & t_ok & (T < 0), core_s & quiet & t_ok & (T >= 0)),
        "Cx5": (core_b & quiet & t_ok & (T < -0.5), core_s & quiet & t_ok & (T > 0.5)),
        "Cxn": (core_b & prk_ok & t_ok & (T < 0), core_s & prk_ok & t_ok & (T >= 0)),
    }
    return v


def load():
    if MODE in ("daily", "hourly"):
        sys.path.insert(0, RESEARCH); cwd = os.getcwd(); os.chdir(RESEARCH)
        from lib import load as lload
        D = lload("data" if MODE == "daily" else "data1h", 1500 if MODE == "daily" else 2500)
        os.chdir(cwd)
        return {s: d for s, (_c, d) in D.items()}
    out = {}
    for fn in sorted(glob.glob(os.path.join(CACHE, "*.csv"))):
        d = pd.read_csv(fn, index_col=0, parse_dates=True)
        d = d[(d.close > 0) & (d.high >= d.low)]
        d = d[d.close.pct_change().abs().fillna(0) < 0.6]
        if len(d) >= 1500:
            out[os.path.basename(fn)[:-4].split("__", 1)[1]] = d
    return out


def main():
    D = load()
    evs, base, rets, checked = [], {}, {}, 0
    for sym, d in D.items():
        d = d[d.close.notna() & d.high.notna() & d.low.notna() & (d.close > 0)]
        lo = d[["open", "high", "low", "close"]]
        L = ledger.compute(lo, H)
        c = lo.close.to_numpy(float)
        V = variants(L, c)
        # the current rules, rebuilt, must be ledger's own - bar for bar
        assert (V["B0"][0] == L.b_buy.to_numpy(bool)).all() and (V["B0"][1] == L.b_sell.to_numpy(bool)).all(), sym
        assert (V["C0"][0] == L.c_buy.to_numpy(bool)).all() and (V["C0"][1] == L.c_sell.to_numpy(bool)).all(), sym
        checked += 1
        V["A"] = (L.a_buy.to_numpy(bool), L.a_sell.to_numpy(bool))
        fwd = (lo.close.shift(-1 - H) / lo.close.shift(-1) - 1.0).to_numpy()
        for k, (b, s) in V.items():
            for m0, sgn in ((b, 1.0), (s, -1.0)):
                m = m0 & np.isfinite(fwd)
                if m.any():
                    evs.append(pd.DataFrame({"date": lo.index[m], "side": sgn, "kind": k, "fwd": fwd[m], "symbol": sym}))
        base[sym] = edge.symbol_baseline(lo.close, H); rets[sym] = lo.close.pct_change()
    ev = pd.concat(evs, ignore_index=True)
    S = edge.measure(ev, base, pd.DataFrame(rets), universe=MODE, selected_index=None, timeframe=MODE, iclass="mixed",
                     length=20, trigger="v12 variants", horizon=H, cost_bps=eng.COST_BPS, n_symbols_universe=len(base),
                     n_bars_median=int(np.median([len(d) for d in D.values()])))
    tot = sum(len(d) for d in D.values())
    rows = []
    for key in edge.SLICES:
        k, side = key.split("_")
        for era in ("discovery", "holdout", "full"):
            r = S.get(key, era)
            if r is None:
                continue
            rows.append(dict(variant=k, side="▲" if side == "buy" else "▼", era=era, events=r.n_events,
                             per_1k=round(1000 * ((ev.kind == k) & (ev.side == (1 if side == "buy" else -1))).sum() / tot, 2),
                             edge=r.edge, ci_lo=r.ci_lo, ci_hi=r.ci_hi, net=r.net, hit=r.hit, n_eff=r.n_eff))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, f"{MODE}.csv"), index=False)
    pd.set_option("display.width", 250)
    print(json.dumps({"mode": MODE, "instruments": len(base), "rebuilt_equal_to_ledger": checked, "level": LEVEL,
                      "split": str(S.split_date)}))
    print(R[R.era == "full"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
