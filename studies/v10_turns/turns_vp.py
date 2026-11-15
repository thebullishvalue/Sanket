"""VWAP and the volume profile at turning points - the same turns as turns.py, read on the price chart.

    python3 studies/v10_turns/turns_vp.py <Indicator/research> [data|data1h] [outdir]

The VWAP and profile are pragyam.pine's own (section 7, from Pragati's PVOI v2.1), ported from bar
data. The SESSION is a week on a daily chart and a day on an hourly chart (vpAnchorTf). On bars:
  session VWAP      sum(hlc3 * volume) / sum(volume) since the session began (time-weighted where
                    an instrument reports volume on fewer than half its bars - the Pine's auto-TPO)
  profile           each bar's volume spread evenly over its range, in rows of ATR(20) / 4 frozen at
                    the session's start; POC = the heaviest row; value area = 70% of the volume,
                    walked outward from the POC (the larger neighbour first)
  prior session     last session's POC, VAH, VAL
  naked POCs        prior sessions' POCs that price has not traded through since, at most 5, none
                    older than 250 bars
Plus relative volume (volume over its 50-bar median), the volume read every profile is built from.
The Pine reads the developing profile from intraday bars where TradingView has them; a bar-data port
cannot, so a developing profile here is the bars' own ranges (what the Pine does without them).

Same turns, same candidates and same reports as turns.py: fingerprint, real vs failed (single readings,
and every VWAP / profile reading crossed with the pane's), timing, reverse.
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import turns as tu                                    # same arguments: <research> [data|data1h] [outdir]
import ledger

ROOT, OUT, H = tu.ROOT, tu.OUT, tu.H
VA = 0.70
VPF = ["vwap", "profile (session)", "vs POC", "prior session", "naked POC below", "naked POC above", "volume"]
VPN = ["vs vwap (σ)", "vs POC (σ)", "rel volume"]


def sessions(index, root):
    """Session id per bar: ISO week on daily bars, calendar day on hourly bars."""
    idx = pd.DatetimeIndex(index)
    if root == "data":
        iso = idx.isocalendar()
        return (iso.year.to_numpy() * 100 + iso.week.to_numpy()).astype(np.int64)
    return (idx.year * 10000 + idx.month * 100 + idx.day).to_numpy().astype(np.int64)


def value_area(vol):
    """(poc index, lo index, hi index) of a profile's rows: 70% of volume, walked out from the POC."""
    p = int(np.argmax(vol)); tot = vol.sum(); lo = hi = p; acc = vol[p]
    while acc < VA * tot and (lo > 0 or hi < len(vol) - 1):
        a = vol[lo - 1] if lo > 0 else -1.0
        b = vol[hi + 1] if hi < len(vol) - 1 else -1.0
        if b >= a:
            hi += 1; acc += b
        else:
            lo -= 1; acc += a
    return p, lo, hi


def vp_readings(d, sig, root):
    o, h, l, c, v = (d[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    n = len(c)
    v = np.where(np.isfinite(v) & (v > 0), v, 0.0)
    tpo = (v > 0).mean() < 0.5                                   # the Pine's auto-TPO fallback
    w = np.ones(n) if tpo else np.maximum(v, 1e-10)
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.r_[c[0], c[:-1]]), np.abs(l - np.r_[c[0], c[:-1]])))
    atr = pd.Series(tr).ewm(alpha=1 / 20, adjust=False).mean().to_numpy()       # ta.atr: RMA
    sid = sessions(d.index, root)
    new = np.r_[True, sid[1:] != sid[:-1]]
    vwap = np.full(n, np.nan); poc = np.full(n, np.nan); vah = np.full(n, np.nan); val = np.full(n, np.nan)
    ppoc = np.full(n, np.nan); pvah = np.full(n, np.nan); pval = np.full(n, np.nan)
    nk_below = np.zeros(n, bool); nk_above = np.zeros(n, bool)
    pv = vv = 0.0; rows = {}; binw = np.nan
    last = (np.nan, np.nan, np.nan); naked = []                 # [(price, bar born)]
    for t in range(n):
        if new[t]:
            if t > 0:
                last = (poc[t - 1], vah[t - 1], val[t - 1])
                if np.isfinite(last[0]) and not any(abs(p - last[0]) <= atr[t] * 0.1 for p, _ in naked):
                    naked.append((last[0], t))
                    naked = naked[-5:]
            # the Pine floors the row at the symbol's tick; bar data has no tick, so the floor is
            # 5 bp of price - without it a zero-range opening bar makes ATR ~0 and a row ~0 wide
            pv = vv = 0.0; rows = {}; binw = max(atr[t] / 4.0, c[t] * 5e-4)
        naked = [(p, b) for p, b in naked if not (l[t] <= p <= h[t]) and t - b <= 250]
        pv += (h[t] + l[t] + c[t]) / 3.0 * w[t]; vv += w[t]
        vwap[t] = pv / vv
        k0, k1 = int(np.floor(l[t] / binw)), int(np.floor(h[t] / binw))
        if k1 - k0 > 400:                        # a bar far wider than the session's ATR: coarsen it
            k1 = k0 + 400
        share = w[t] / (k1 - k0 + 1)
        for k in range(k0, k1 + 1):
            rows[k] = rows.get(k, 0.0) + share
        keys = np.fromiter(sorted(rows), dtype=np.int64)
        vol = np.array([rows[k] for k in keys])
        p, lo_i, hi_i = value_area(vol)
        poc[t] = (keys[p] + 0.5) * binw; vah[t] = (keys[hi_i] + 1.0) * binw; val[t] = keys[lo_i] * binw
        ppoc[t], pvah[t], pval[t] = last
        nk_below[t] = any(c[t] - atr[t] <= p < c[t] for p, _ in naked)
        nk_above[t] = any(c[t] < p <= c[t] + atr[t] for p, _ in naked)
    sH = sig * np.sqrt(H)
    with np.errstate(invalid="ignore", divide="ignore"):
        dv = (np.log(c) - np.log(vwap)) / sH
        dp = (np.log(c) - np.log(poc)) / sH
        rv = v / pd.Series(v).rolling(50, min_periods=20).median().replace(0, np.nan).to_numpy()
    cat = pd.DataFrame({
        "vwap": np.where(c >= vwap, "close above", "close below"),
        "profile (session)": np.where(c > vah, "above VAH", np.where(c < val, "below VAL", "inside value")),
        "vs POC": np.where(c >= poc, "above POC", "below POC"),
        "prior session": np.where(~np.isfinite(pvah), "none", np.where(c > pvah, "above pVAH",
                                  np.where(c < pval, "below pVAL", "inside pValue"))),
        "naked POC below": np.where(nk_below, "within 1 ATR", "none near"),
        "naked POC above": np.where(nk_above, "within 1 ATR", "none near"),
        "volume": np.where(tpo | ~np.isfinite(rv), "no volume", np.where(rv >= 2, "heavy (2x+)",
                           np.where(rv >= 1.2, "above usual", np.where(rv < 0.8, "light", "usual")))),
    })
    num = pd.DataFrame({"vs vwap (σ)": dv, "vs POC (σ)": dp, "rel volume": np.where(tpo, np.nan, rv)})
    return cat, num, tpo


def harvest():
    D = tu.load(ROOT, 1500 if ROOT == "data" else 2500)
    n_inst = len(D)
    base, turns_, cands, paths, timing = [], [], [], [], []
    for sym, (cls, d) in D.items():
        d = d[d.close.notna() & d.high.notna() & d.low.notna() & (d.close > 0)]
        lo = d[["open", "high", "low", "close"]]
        L = ledger.compute(lo, H)
        c = lo.close.to_numpy(float); lc = np.log(c); n = len(c); sig = L.lg_sigma.to_numpy(float)
        pcat, pnum, live = tu.readings(L, c)
        vcat, vnum, tpo = vp_readings(d, sig, ROOT)
        grp = "crypto" if cls == "crypto" else ("equity" if cls in ("us", "nse", "index") else "macro")
        meta = pd.DataFrame({"sym": sym, "cls": cls, "grp": grp, "date": d.index})
        allc = pd.concat([meta, pcat, vcat, pnum, vnum], axis=1)
        # the all-bars frame needs only the VWAP / profile readings - kept as categories, it is a
        # tenth of the memory of object strings across a million bars
        base.append(pd.concat([meta[["sym", "date"]], vcat], axis=1)[live].astype({f: "category" for f in VPF + ["sym"]}))
        for scale, (k, hh) in tu.SCALES.items():
            thr = k * sig * np.sqrt(hh)
            piv = tu.zigzag(lc, thr)
            for t, kind in piv:
                if not live[t]:
                    continue
                r = allc.iloc[t].to_dict(); r.update(scale=scale, turn="bottom" if kind < 0 else "top")
                turns_.append(r)
                if scale != "swing":
                    continue
                for off in range(-tu.W, tu.W + 1):
                    j = t + off
                    if 0 <= j < n and live[j]:
                        paths.append((kind, off, *vnum.iloc[j][VPN].to_numpy()))
                sg = -kind; end = min(n, t + 61)
                tm = {"turn": "bottom" if kind < 0 else "top", "grp": grp}
                for name, ser in (("vwap", vnum["vs vwap (σ)"].to_numpy()), ("POC", vnum["vs POC (σ)"].to_numpy())):
                    side = np.where(ser >= 0, 1, -1) == sg
                    tm[f"{name} at turn"] = bool(side[t])
                    later = np.where(side[t:end])[0]
                    tm[f"{name} bars"] = int(later[0]) if len(later) else np.nan
                timing.append(tm)
        thr = tu.SCALES["swing"][0] * sig * np.sqrt(tu.SCALES["swing"][1])
        for side, ext in ((-1, pd.Series(lc).rolling(tu.LOOK).min().to_numpy()),
                          (1, pd.Series(lc).rolling(tu.LOOK).max().to_numpy())):
            cand = np.where((lc == ext) & live & np.isfinite(thr))[0]
            res = tu.outcome(lc, thr, cand, side)
            keep = res >= 0
            cc = allc.iloc[cand[keep]].reset_index(drop=True)
            cc = cc.astype({f: "category" for f in tu.FEATS + VPF})
            cc["extreme"] = "low" if side < 0 else "high"; cc["turned"] = res[keep]
            cands.append(cc)
    del D
    B = pd.concat(base, ignore_index=True)
    for f in VPF + ["sym"]:
        B[f] = B[f].astype(str).astype("category")
    C = pd.concat(cands, ignore_index=True)
    for f in tu.FEATS + VPF:
        C[f] = C[f].astype(str)
    return (n_inst, pd.DataFrame(turns_), C, B,
            pd.DataFrame(paths, columns=["kind", "off", *VPN]), pd.DataFrame(timing))


def fingerprint(T, B):
    rows = []
    for (scale, turn), t in T.groupby(["scale", "turn"]):
        for f in VPF:
            st, sb = t[f].value_counts(normalize=True) * 100, B[f].astype(str).value_counts(normalize=True) * 100
            for lvl in sb.index:
                rows.append(dict(scale=scale, turn=turn, reading=f, state=lvl, n=len(t), at_turn=st.get(lvl, 0.0),
                                 all_bars=sb[lvl], lift=st.get(lvl, 0.0) / sb[lvl]))
    return pd.DataFrame(rows)


def real_vs_failed(C):
    rows = []
    for ext, c in C.groupby("extreme"):
        p0 = c.turned.mean()
        for f in VPF:
            for lvl, r in c.groupby(f).turned.agg(["mean", "size"]).iterrows():
                rows.append(dict(extreme=ext, reading=f, state=lvl, n=int(r["size"]), turned_pct=100 * r["mean"],
                                 base=100 * p0, lift=r["mean"] / p0))
    return pd.DataFrame(rows)


def crossed(C, min_n):
    """Every VWAP / profile reading crossed with every pane reading - found on the first half, checked on the second."""
    out = []
    for ext, c in C.groupby("extreme"):
        c = c.copy(); split = c.date.quantile(0.5); c["half"] = np.where(c.date <= split, "first", "second")
        p = c.groupby("half").turned.mean()
        for vf in VPF:
            for pf in tu.FEATS:
                g = c.groupby([vf, pf, "half"]).turned.agg(["mean", "size"]).unstack("half")
                g.columns = [f"{a}_{b}" for a, b in g.columns]
                g = g.dropna()
                g = g[(g["size_first"] >= min_n) & (g["size_second"] >= min_n)]
                if g.empty:
                    continue
                g["lift_first"] = g.mean_first / p["first"]; g["lift_second"] = g.mean_second / p["second"]
                g = g.reset_index().rename(columns={vf: "vp_state", pf: "pane_state"})
                g["vp_reading"], g["pane_reading"], g["extreme"] = vf, pf, ext
                out.append(g)
    return pd.concat(out, ignore_index=True)


def reverse(B, T):
    rows = []
    sw = T[T.scale == "swing"]; Bs = B.reset_index(drop=True); groups = Bs.groupby("sym", observed=True).groups
    for turn in ("bottom", "top"):
        tset = {(s, dd) for s, dd in sw[sw.turn == turn][["sym", "date"]].itertuples(index=False)}
        hit = np.fromiter(((s, dd) in tset for s, dd in zip(Bs.sym.astype(str), Bs.date)), dtype=bool, count=len(Bs))
        mark = np.zeros(len(Bs), bool)
        for _s, gi in groups.items():
            gi = np.asarray(gi); h_ = hit[gi]; hh = h_.copy()
            for off in (1, 2):
                hh[:-off] |= h_[off:]; hh[off:] |= h_[:-off]
            mark[gi] = hh
        mark = pd.Series(mark); p0 = mark.mean()
        for f in VPF:
            for lvl, r in mark.groupby(Bs[f].astype(str)).agg(["mean", "size"]).iterrows():
                rows.append(dict(turn=turn, reading=f, state=lvl, bars=int(r["size"]), p_turn_near=100 * r["mean"],
                                 base=100 * p0, lift=r["mean"] / p0))
    return pd.DataFrame(rows)


def chart(P, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=110)
    for ax, f in zip(axes, VPN):
        for kind, col, lab in ((-1, "#2CA36B", "bottoms"), (1, "#DD5A5A", "tops")):
            g = P[P.kind == kind].groupby("off")[f]
            m = g.median()
            ax.plot(m.index, m.values, color=col, lw=2, label=lab)
            ax.fill_between(m.index, g.quantile(.25).values, g.quantile(.75).values, color=col, alpha=0.12)
        ax.axvline(0, color="#888", lw=1, ls="--"); ax.axhline(1 if f == "rel volume" else 0, color="#888", lw=0.8)
        ax.set_title(f); ax.set_xlabel("bars from the turn"); ax.grid(alpha=0.2)
    axes[0].legend()
    fig.suptitle(f"VWAP, POC and volume around swing turns ({ROOT}) · median and inter-quartile range")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    D, T, C, B, P, M = harvest()
    tag = "daily" if ROOT == "data" else "hourly"
    chart(P, os.path.join(OUT, f"vp_paths_{tag}.png"))
    for name, df in (("vp_fingerprint", fingerprint(T, B)), ("vp_real_vs_failed", real_vs_failed(C)),
                     ("vp_crossed", crossed(C, 300 if ROOT == "data" else 400)), ("vp_reverse", reverse(B, T)),
                     ("vp_timing", M)):
        df.to_csv(os.path.join(OUT, f"{name}_{tag}.csv"), index=False)
    C["era"] = pd.cut(C.date.dt.year, [1999, 2012, 2019, 2030], labels=["03-12", "13-19", "20-26"]).astype(str)
    sp = []
    for by in ("grp", "era"):
        for (ext, lvl), gg in C.groupby(["extreme", by]):
            p0 = gg.turned.mean()
            for f in VPF:
                for st_, r in gg.groupby(f).turned.agg(["mean", "size"]).iterrows():
                    sp.append(dict(by=by, group=lvl, extreme=ext, reading=f, state=st_, n=int(r["size"]),
                                   turned_pct=100 * r["mean"], lift=r["mean"] / p0))
    pd.DataFrame(sp).to_csv(os.path.join(OUT, f"vp_real_vs_failed_splits_{tag}.csv"), index=False)
    print(json.dumps({"instruments": D, "volume_instruments": int(B.groupby("sym", observed=True).volume.apply(lambda s: (s != "no volume").mean() > 0.5).sum()),
                      "turns": {f"{a} {b}": int(v) for (a, b), v in T.groupby(["scale", "turn"]).size().items()}}, indent=1))
