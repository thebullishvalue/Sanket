"""Does a signal built from the turning-point studies beat sets A / B / C?

    python3 studies/v10_turns/signals_vs_sets.py <Indicator/research> [data|data1h] [outdir]

PROTOCOL (declared before the VWAP / profile results were read)
    Candidates, each a ▲ with a mirrored ▼, firing once (no re-fire within 10 bars):
      T1 confirmed low     a 20-bar closing low in the last 10 bars, and the column turns green on this bar
      T2 risk easing       T1, and risk was stressed at that low and is falling now (ln sigma/sigma[10] < 0)
      T3 VWAP reclaim      a 20-bar low in the last 10 bars, and the close crosses back above session VWAP
      T4 back into value   a 20-bar low in the last 10 bars, and the close crosses back above last
                           session's value-area low
      T5 data-chosen       on the 20-bar low itself, the VWAP / profile x pane combination with the
                           highest turned-vs-failed lift on the FIRST HALF of the candidate dates
                           (studies/v10_turns/out/vp_crossed_*.csv) - chosen before the Edge Study runs
    Judge: Sanket's own Edge Study (edge.measure) - entry the bar after, a 10-bar hold, trailing drift
    removed, vol-normalised, block bootstrap over dates, 3 bp. ONE call measures sets A / B / C and the
    candidates together, so they share symbols, dates and the discovery / holdout split.
    The DEVELOPED signal is the candidate side with the best DISCOVERY edge. It beats the sets only if
    its HOLDOUT edge is significant and above every set's holdout edge on the same side.
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import turns as tu
import turns_vp as tv
import ledger
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import edge, engine as eng                              # noqa: E402

ROOT, OUT, H = tu.ROOT, tu.OUT, tu.H
TAG = "daily" if ROOT == "data" else "hourly"
CANDS = ("T1", "T2", "T3", "T4", "T5")
LABEL = {"T1": "confirmed low / high", "T2": "risk easing after the extreme", "T3": "VWAP reclaim",
         "T4": "back into value", "T5": "data-chosen combination"}


def t5_rules():
    """The first-half winner for lows and for highs, from the VWAP / profile study."""
    X = pd.read_csv(os.path.join(OUT, f"vp_crossed_{TAG}.csv"))
    out = {}
    for ext in ("low", "high"):
        r = X[X.extreme == ext].sort_values("lift_first", ascending=False).iloc[0]
        out[ext] = dict(vp=(r.vp_reading, r.vp_state), pane=(r.pane_reading, r.pane_state),
                        lift_first=round(float(r.lift_first), 3), lift_second=round(float(r.lift_second), 3))
    return out


def once(x, gap=H):
    """Keep a firing only if the same rule did not fire in the previous `gap` bars."""
    x = np.asarray(x, bool); out = np.zeros_like(x); last = -10 ** 9
    for t in np.where(x)[0]:
        if t - last > gap:
            out[t] = True; last = t
        # a suppressed firing does not extend the quiet window
    return out


def candidates(d, L, rules):
    c = d.close.to_numpy(float); lc = np.log(c)
    sig = L.lg_sigma.to_numpy(float)
    pcat, _pnum, live = tu.readings(L, c)
    vcat, vnum, _tpo = tv.vp_readings(d, sig, ROOT)
    u = L.lg_u.to_numpy(float); rm = L.lg_rmom.to_numpy(float); prk = L.lg_prk.to_numpy(float)
    lo20 = lc == pd.Series(lc).rolling(tu.LOOK).min().to_numpy()
    hi20 = lc == pd.Series(lc).rolling(tu.LOOK).max().to_numpy()
    rec = lambda m: pd.Series(m.astype(float)).rolling(H, min_periods=1).max().to_numpy() > 0   # noqa: E731
    lo_recent, hi_recent = rec(lo20), rec(hi20)
    # risk at the most recent extreme (forward-filled from the extreme bar)
    prk_lo = pd.Series(np.where(lo20, prk, np.nan)).ffill().to_numpy()
    prk_hi = pd.Series(np.where(hi20, prk, np.nan)).ffill().to_numpy()
    up = np.r_[False, (u[1:] >= 0) & (u[:-1] < 0)]
    dn = np.r_[False, (u[1:] < 0) & (u[:-1] >= 0)]
    vw = vnum["vs vwap (σ)"].to_numpy()
    vw_up = np.r_[False, (vw[1:] >= 0) & (vw[:-1] < 0)]
    vw_dn = np.r_[False, (vw[1:] < 0) & (vw[:-1] >= 0)]
    ps = vcat["prior session"].to_numpy()
    in_up = np.r_[False, (ps[1:] != "below pVAL") & (ps[:-1] == "below pVAL") & (ps[1:] != "none")]
    in_dn = np.r_[False, (ps[1:] != "above pVAH") & (ps[:-1] == "above pVAH") & (ps[1:] != "none")]
    stressed = lambda x: x >= 200 / 3                                                            # noqa: E731
    allc = pd.concat([pcat, vcat], axis=1)

    def t5(ext, at):
        r = rules[ext]
        return at & (allc[r["vp"][0]].to_numpy() == r["vp"][1]) & (allc[r["pane"][0]].to_numpy() == r["pane"][1])

    sig_ = {
        "T1": (lo_recent & up, hi_recent & dn),
        "T2": (lo_recent & up & stressed(prk_lo) & (rm < 0), hi_recent & dn & stressed(prk_hi) & (rm < 0)),
        "T3": (lo_recent & vw_up, hi_recent & vw_dn),
        "T4": (lo_recent & in_up, hi_recent & in_dn),
        "T5": (t5("low", lo20), t5("high", hi20)),
    }
    return {k: (once(b & live), once(s & live)) for k, (b, s) in sig_.items()}


def main():
    rules = t5_rules()
    for k in CANDS:
        for side, sgn in (("buy", 1.0), ("sell", -1.0)):
            edge.SLICES[f"{k}_{side}"] = (sgn, k)
            edge.SLICE_LABEL[f"{k}_{side}"] = f"{k} · {'▲' if sgn > 0 else '▼'} {LABEL[k]}"
    D = tu.load(ROOT, 1500 if ROOT == "data" else 2500)
    evs, base, rets, frames = [], {}, {}, {}
    for sym, (cls, d) in D.items():
        d = d[d.close.notna() & d.high.notna() & d.low.notna() & (d.close > 0)]
        lo = d[["open", "high", "low", "close"]]
        L = ledger.compute(lo, H)
        close = lo.close
        fwd = (close.shift(-1 - H) / close.shift(-1) - 1.0).to_numpy()
        sets = {k: (L[f"{k.lower()}_buy"].to_numpy(bool), L[f"{k.lower()}_sell"].to_numpy(bool)) for k in "ABC"}
        sets.update(candidates(d, L, rules))
        for k, (b, s) in sets.items():
            for m0, sgn in ((b, 1.0), (s, -1.0)):
                m = m0 & np.isfinite(fwd)
                if m.any():
                    evs.append(pd.DataFrame({"date": lo.index[m], "side": sgn, "kind": k, "fwd": fwd[m],
                                             "symbol": sym, "cls": cls}))
        base[sym] = edge.symbol_baseline(close, H)
        rets[sym] = close.pct_change()
        frames[sym] = (cls, lo, sets, L["lg_sigma"].notna().to_numpy())
    ev = pd.concat(evs, ignore_index=True)
    S = edge.measure(ev, base, pd.DataFrame(rets), universe="Indicator research universe", selected_index=None,
                     timeframe=TAG, iclass="mixed", length=20, trigger="sets A/B/C vs T1-T5", horizon=H,
                     cost_bps=eng.COST_BPS, n_symbols_universe=len(D),
                     n_bars_median=int(np.median([len(f[1]) for f in frames.values()])))
    tot = sum(len(f[1]) for f in frames.values())
    rows = []
    for key in edge.SLICES:
        k, side = key.split("_")
        lbl, _kind, _d = S.verdict(side, k)
        for era in ("discovery", "holdout", "full"):
            r = S.get(key, era)
            if r is None:
                continue
            rows.append(dict(signal=k, side="▲" if side == "buy" else "▼", era=era, events=r.n_events,
                             per_1k_bars=round(1000 * ((ev.kind == k) & (ev.side == (1 if side == "buy" else -1))).sum() / tot, 2),
                             edge=round(r.edge, 4), ci_lo=round(r.ci_lo, 4), ci_hi=round(r.ci_hi, 4), net=round(r.net, 4),
                             hit=round(r.hit, 1), sig="yes" if r.significant else ("ANTI" if r.anti else "no"),
                             verdict=lbl if era == "full" else ""))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, f"signals_vs_sets_{TAG}.csv"), index=False)

    # the developed signal: best DISCOVERY edge among the candidates, per side
    dev = {}
    for side in ("▲", "▼"):
        disc = R[(R.era == "discovery") & (R.side == side) & R.signal.isin(CANDS) & (R.events >= edge.MIN_EVENTS)]
        best = disc.sort_values("edge", ascending=False).iloc[0]
        hold = R[(R.era == "holdout") & (R.side == side)].set_index("signal")
        cand_h = hold.loc[best.signal]
        sets_h = hold.loc[[k for k in "ABC" if k in hold.index]]
        dev[side] = dict(signal=best.signal, label=LABEL[best.signal], discovery_edge=float(best.edge),
                         holdout_edge=float(cand_h.edge), holdout_ci=[float(cand_h.ci_lo), float(cand_h.ci_hi)],
                         holdout_significant=cand_h.sig == "yes",
                         best_set_holdout=dict(set=str(sets_h.edge.idxmax()), edge=float(sets_h.edge.max())),
                         beats_sets=bool(cand_h.sig == "yes" and cand_h.edge > sets_h.edge.max()))
    summary = dict(timeframe=TAG, split=str(S.split_date), part_ratio=round(S.part_ratio, 1), t5_rules=rules, developed=dev)
    json.dump(summary, open(os.path.join(OUT, f"signals_vs_sets_{TAG}.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(f"{TAG} · split {S.split_date} · participation ratio {S.part_ratio:.1f}")
    print(R.to_string(index=False))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
