"""What the tapes and the column read when an instrument bottomed and turned up - and topped and turned down.

    python3 studies/v10_turns/turns.py <Indicator/research> [data|data1h] [outdir]

Every reading comes from ledger.py - the exact code Pragyam's pane and Sanket run - on each
instrument's own OHLC, and each is CAUSAL (a reading at bar t uses bars up to t only). The turning
points are labelled with hindsight; that is the study's premise: find the turns, then read the tape.

TURNS (a zig-zag on log closes, its reversal sized to each instrument's own risk)
    swing   a bottom is the lowest close before price rises 2.5 x sigma x sqrt(10) without first
            closing lower; a top mirrors. sigma is the risk forecast at the extreme.
    major   3 x sigma x sqrt(40).
CANDIDATES (for "real or failed")
    every 20-bar closing low. TURNED if price then rises the swing amount before closing lower;
    FAILED if it closes lower first. 20-bar highs mirror. Undecided at the end of data: dropped.

READINGS (the pane: four tapes, the column, the midline and the cone)
    inner risk tape   risk's rank in its last 500 bars, in thirds: calm / normal / stressed
    outer risk tape   rising (sigma up > 10% over 10 bars, gold) or not (grey)
    inner trend tape  T in sigma: up (> 0.5) / flat / down (< -0.5); and its sign (green / red)
    outer trend tape  the stretch ladder's lean: green (>= 0) / red; rungs outside the band
    column            the surprise: green (>= 0) / red
    midline           close above / below the midline
    cone              the close below / inside / above the landing band
    ledger            the bar's state proven + / proven - / unproven
ANSWERS
    A  fingerprint   each reading at bottoms and tops vs all bars (lift = share at turns / share on all bars)
    B  real/failed   each reading at 20-bar lows that turned vs those that failed; then every
                     combination, found on the first half of the dates and checked on the second
    C  timing        bars from the turn until each tape changes colour; when B / C fired, and how
                     much of the move was left by then
    D  reverse       P(a swing bottom within +-2 bars | reading) - is the fingerprint rare enough to use?
    E  paths         each reading averaged from 20 bars before to 20 after the turn (charts)
"""
import os, sys, warnings, json
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
RESEARCH = os.path.abspath(sys.argv[1])
OUT = os.path.abspath(sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "out"))   # before the chdir
sys.path.insert(0, RESEARCH)
import numpy as np, pandas as pd
import ledger
os.chdir(RESEARCH)
from lib import load                                          # noqa: E402

ROOT = sys.argv[2] if len(sys.argv) > 2 else "data"
os.makedirs(OUT, exist_ok=True)
H = 10
SCALES = {"swing": (2.5, 10), "major": (3.0, 40)}
LOOK = 20
W = 20
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)


def zigzag(lc, thr):
    """Pivots of log closes: [(index, -1 bottom / +1 top)], each reversal at least thr[extreme]."""
    ok = np.where(np.isfinite(thr))[0]
    if len(ok) < 2:
        return []
    s = ok[0]
    piv, d, hi, lo = [], 0, s, s
    for t in range(s + 1, len(lc)):
        if d >= 0 and lc[t] > lc[hi]:
            hi = t
        if d <= 0 and lc[t] < lc[lo]:
            lo = t
        if d >= 0 and np.isfinite(thr[hi]) and lc[hi] - lc[t] >= thr[hi]:
            piv.append((hi, 1)); d, lo = -1, t
        elif d <= 0 and np.isfinite(thr[lo]) and lc[t] - lc[lo] >= thr[lo]:
            piv.append((lo, -1)); d, hi = 1, t
    return piv[1:]                    # the first pivot is wherever the scan began


def outcome(lc, thr, cand, side):
    """For 20-bar lows (side -1) / highs (+1): 1 turned, 0 failed, -1 undecided."""
    n = len(lc); res = np.full(len(cand), -1)
    for k, t in enumerate(cand):
        x0, th = lc[t], thr[t]
        for j in range(t + 1, n):
            mv = (lc[j] - x0) * (-side)          # move in the reversal's direction
            if mv >= th:
                res[k] = 1; break
            if mv < 0:
                res[k] = 0; break
    return res


def readings(L, c):
    """The pane's readings as categories (the colour a trader sees) and as numbers."""
    prk, rm, T, lad, u = (L[k].to_numpy(float) for k in ("lg_prk", "lg_rmom", "lg_trend", "lg_ladx", "lg_u"))
    mid = L["lg_mid"].to_numpy(float); st = L["lg_state"].to_numpy(int); pv = L["lg_proven"].to_numpy(int)
    sig = L["lg_sigma"].to_numpy(float)
    live = np.isfinite(prk) & np.isfinite(rm) & np.isfinite(T) & np.isfinite(lad) & np.isfinite(u) & np.isfinite(mid) & (st >= 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        midz = (np.log(c) - np.log(mid)) / (sig * np.sqrt(H))
    cat = pd.DataFrame({
        "risk (inner)": np.where(prk < 100 / 3, "calm", np.where(prk < 200 / 3, "normal", "stressed")),
        "risk (outer)": np.where(rm > ledger.RISE, "rising", "not rising"),
        "trend (inner)": np.where(T > 0.5, "up", np.where(T < -0.5, "down", "flat")),
        "trend colour": np.where(T >= 0, "green", "red"),
        "ladder (outer)": np.where(lad >= 0, "green", "red"),
        "column": np.where(u >= 0, "green", "red"),
        "midline": np.where(midz > 0, "close above", "close below"),
        "cone": np.array(["below band", "inside band", "above band"])[np.clip(st % 3, 0, 2)],
        "ledger": np.where(pv > 0, "proven +", np.where(pv < 0, "proven -", "unproven")),
    })
    num = pd.DataFrame({"risk rank": prk, "risk mom": rm, "trend T": T, "ladder": lad, "column u": u,
                        "vs midline (σ)": midz, "rungs out up": L["lg_lad_up"].to_numpy(float),
                        "rungs out down": L["lg_lad_dn"].to_numpy(float)})
    return cat, num, live


FEATS = ["risk (inner)", "risk (outer)", "trend (inner)", "trend colour", "ladder (outer)", "column",
         "midline", "cone", "ledger"]
NUMS = ["risk rank", "risk mom", "trend T", "ladder", "column u", "vs midline (σ)"]


def harvest():
    D = load(ROOT, 1500 if ROOT == "data" else 2500)
    turns, cands, base, paths, timing = [], [], [], [], []
    for sym, (cls, d) in D.items():
        lo = d[["open", "high", "low", "close"]]
        lo = lo[lo.close.notna() & lo.high.notna() & lo.low.notna() & (lo.close > 0)]
        L = ledger.compute(lo, H)
        c = lo.close.to_numpy(float); lc = np.log(c); n = len(c)
        sig = L.lg_sigma.to_numpy(float)
        cat, num, live = readings(L, c)
        dates = lo.index
        grp = "crypto" if cls == "crypto" else ("equity" if cls in ("us", "nse", "index") else "macro")
        meta = pd.DataFrame({"sym": sym, "cls": cls, "grp": grp, "date": dates})
        b = pd.concat([meta, cat, num], axis=1)[live]
        base.append(b)
        bb, bs = (L[f"b_{s}"].to_numpy(bool) for s in ("buy", "sell"))
        cb, cs = (L[f"c_{s}"].to_numpy(bool) for s in ("buy", "sell"))
        # ---- turns at both scales ----
        for scale, (k, h) in SCALES.items():
            thr = k * sig * np.sqrt(h)
            piv = zigzag(lc, thr)
            for i, (t, kind) in enumerate(piv):
                if not live[t]:
                    continue
                nxt = piv[i + 1][0] if i + 1 < len(piv) else None
                row = {"sym": sym, "cls": cls, "grp": grp, "date": dates[t], "scale": scale,
                       "turn": "bottom" if kind < 0 else "top", "t": t,
                       "swing %": (abs(lc[nxt] - lc[t]) * 100) if nxt is not None else np.nan,
                       "bars to next": (nxt - t) if nxt is not None else np.nan}
                row.update(cat.iloc[t].to_dict()); row.update(num.iloc[t].to_dict())
                turns.append(row)
                if scale != "swing":
                    continue
                # paths, -W..+W
                for off in range(-W, W + 1):
                    j = t + off
                    if 0 <= j < n and live[j]:
                        paths.append((kind, off, *num.iloc[j][NUMS].to_numpy()))
                # timing: bars until each reading takes the new direction (green for a bottom)
                sg = -kind                       # +1 after a bottom, -1 after a top
                tm = {"turn": "bottom" if kind < 0 else "top", "grp": grp, "date": dates[t]}
                end = min(n, t + 61)
                for name, ser in (("column", num["column u"].to_numpy()), ("ladder", num["ladder"].to_numpy()),
                                  ("trend T", num["trend T"].to_numpy()), ("midline", num["vs midline (σ)"].to_numpy())):
                    side = np.sign(np.where(ser >= 0, 1, -1)) == sg
                    tm[f"{name} at turn"] = bool(side[t])
                    later = np.where(side[t:end])[0]
                    tm[f"{name} bars"] = int(later[0]) if len(later) else np.nan
                ev_b = (bb if kind < 0 else bs); ev_c = (cb if kind < 0 else cs)
                for nm, ev in (("B", ev_b), ("C", ev_c)):
                    w = np.where(ev[max(0, t - 10): end])[0]
                    first = (w[0] + max(0, t - 10) - t) if len(w) else np.nan
                    tm[f"{nm} first fire"] = first
                    if np.isfinite(first) and nxt is not None and first >= 0 and t + first < nxt:
                        tot = lc[nxt] - lc[t]
                        tm[f"{nm} move left %"] = 100 * (lc[nxt] - lc[int(t + first)]) / tot if tot != 0 else np.nan
                    else:
                        tm[f"{nm} move left %"] = np.nan
                timing.append(tm)
        # ---- 20-bar lows and highs: turned or failed ----
        thr = SCALES["swing"][0] * sig * np.sqrt(SCALES["swing"][1])
        roll_lo = pd.Series(lc).rolling(LOOK).min().to_numpy()
        roll_hi = pd.Series(lc).rolling(LOOK).max().to_numpy()
        for side, ext in ((-1, roll_lo), (1, roll_hi)):
            cand = np.where((lc == ext) & live & np.isfinite(thr))[0]
            res = outcome(lc, thr, cand, side)
            keep = res >= 0
            cc = pd.concat([meta.iloc[cand[keep]].reset_index(drop=True),
                            cat.iloc[cand[keep]].reset_index(drop=True),
                            num.iloc[cand[keep]].reset_index(drop=True)], axis=1)
            cc["extreme"] = "low" if side < 0 else "high"
            cc["turned"] = res[keep]
            cands.append(cc)
    T = pd.DataFrame(turns); C = pd.concat(cands, ignore_index=True); B = pd.concat(base, ignore_index=True)
    P = pd.DataFrame(paths, columns=["kind", "off", *NUMS]); M = pd.DataFrame(timing)
    return D, T, C, B, P, M


def share(df, f):
    return df[f].value_counts(normalize=True) * 100


def fingerprint(T, B):
    rows = []
    for scale in SCALES:
        for turn in ("bottom", "top"):
            t = T[(T.scale == scale) & (T.turn == turn)]
            for f in FEATS:
                s_t, s_b = share(t, f), share(B, f)
                for lvl in s_b.index:
                    rows.append(dict(scale=scale, turn=turn, reading=f, state=lvl, n=len(t),
                                     at_turn=s_t.get(lvl, 0.0), all_bars=s_b[lvl],
                                     lift=s_t.get(lvl, 0.0) / s_b[lvl] if s_b[lvl] else np.nan))
    return pd.DataFrame(rows)


def real_vs_failed(C):
    rows = []
    for ext in ("low", "high"):
        c = C[C.extreme == ext]
        p0 = c.turned.mean()
        for f in FEATS:
            g = c.groupby(f).turned.agg(["mean", "size"])
            for lvl, r in g.iterrows():
                rows.append(dict(extreme=ext, reading=f, state=lvl, n=int(r["size"]),
                                 share_of_candidates=100 * r["size"] / len(c), turned_pct=100 * r["mean"],
                                 base_turned_pct=100 * p0, lift=r["mean"] / p0))
    return pd.DataFrame(rows)


def combos(C, min_n):
    """Every combination of the pane's colours: found on the first half of the dates, checked on the second."""
    keys = ["risk (inner)", "risk (outer)", "trend (inner)", "ladder (outer)", "column", "midline", "cone"]
    out = []
    for ext in ("low", "high"):
        c = C[C.extreme == ext].copy()
        split = c.date.quantile(0.5)
        c["half"] = np.where(c.date <= split, "first", "second")
        p = c.groupby("half").turned.mean()
        g = c.groupby(keys + ["half"]).turned.agg(["mean", "size"]).unstack("half")
        g.columns = [f"{a}_{b}" for a, b in g.columns]
        g = g.dropna()
        g = g[(g["size_first"] >= min_n) & (g["size_second"] >= min_n)]
        g["lift_first"] = g["mean_first"] / p["first"]; g["lift_second"] = g["mean_second"] / p["second"]
        g["extreme"] = ext; g["split"] = str(split.date())
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def reverse(B, T):
    """P(a swing bottom (top) within +-2 bars | the reading on this bar), against the base rate."""
    rows = []
    sw = T[T.scale == "swing"]
    Bs = B.reset_index(drop=True)
    groups = Bs.groupby("sym").groups
    for turn in ("bottom", "top"):
        tset = {(s_, d_) for s_, d_ in sw[sw.turn == turn][["sym", "date"]].itertuples(index=False)}
        hit = np.fromiter(((s_, d_) in tset for s_, d_ in zip(Bs.sym, Bs.date)), dtype=bool, count=len(Bs))
        mark = np.zeros(len(Bs), dtype=bool)
        for _s, gi in groups.items():
            gi = np.asarray(gi)
            h = hit[gi]; hh = h.copy()
            for off in (1, 2):                    # within +-2 bars of the turn
                hh[:-off] |= h[off:]; hh[off:] |= h[:-off]
            mark[gi] = hh
        mark = pd.Series(mark, index=Bs.index)
        p0 = mark.mean()
        for f in FEATS:
            g = mark.groupby(Bs[f]).agg(["mean", "size"])
            for lvl, r in g.iterrows():
                rows.append(dict(turn=turn, reading=f, state=lvl, bars=int(r["size"]),
                                 p_turn_near=100 * r["mean"], base=100 * p0, lift=r["mean"] / p0))
    return pd.DataFrame(rows)


def chart(P, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["risk rank", "risk mom", "trend T", "ladder", "column u", "vs midline (σ)"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=110)
    for ax, f in zip(axes.ravel(), show):
        for kind, col, lab in ((-1, "#2CA36B", "bottoms"), (1, "#DD5A5A", "tops")):
            g = P[P.kind == kind].groupby("off")[f]
            m, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
            ax.plot(m.index, m.values, color=col, lw=2, label=lab)
            ax.fill_between(m.index, q1.values, q3.values, color=col, alpha=0.12)
        ax.axvline(0, color="#888", lw=1, ls="--")
        if f in ("trend T", "ladder", "column u", "vs midline (σ)", "risk mom"):
            ax.axhline(0 if f != "risk mom" else ledger.RISE, color="#888", lw=0.8)
        if f == "risk rank":
            ax.axhline(100 / 3, color="#888", lw=0.8)
        ax.set_title(f); ax.set_xlabel("bars from the turn"); ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle(f"The pane around swing turns ({ROOT}) · median and inter-quartile range")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    D, T, C, B, P, M = harvest()
    tag = "daily" if ROOT == "data" else "hourly"
    fp, rf, cb, rv = fingerprint(T, B), real_vs_failed(C), combos(C, 300 if ROOT == "data" else 400), reverse(B, T)
    chart(P, os.path.join(OUT, f"paths_{tag}.png"))
    for name, df in (("fingerprint", fp), ("real_vs_failed", rf), ("combos", cb), ("reverse", rv), ("timing", M)):
        df.to_csv(os.path.join(OUT, f"{name}_{tag}.csv"), index=False)
    # era and group splits of real vs failed (for stability)
    C["era"] = pd.cut(C.date.dt.year, [1999, 2012, 2019, 2030], labels=["03-12", "13-19", "20-26"]).astype(str)
    split = []
    for by in ("grp", "era"):
        for ext in ("low", "high"):
            c = C[C.extreme == ext]
            for lvl, gg in c.groupby(by):
                p0 = gg.turned.mean()
                for f in FEATS:
                    for st, r in gg.groupby(f).turned.agg(["mean", "size"]).iterrows():
                        split.append(dict(by=by, group=lvl, extreme=ext, reading=f, state=st, n=int(r["size"]),
                                          turned_pct=100 * r["mean"], lift=r["mean"] / p0))
    pd.DataFrame(split).to_csv(os.path.join(OUT, f"real_vs_failed_splits_{tag}.csv"), index=False)
    summary = {
        "instruments": len(D), "bars_live": int(len(B)),
        "turns": T.groupby(["scale", "turn"]).size().to_dict(),
        "swing_size_median_pct": T.groupby(["scale", "turn"])["swing %"].median().round(2).to_dict(),
        "bars_between_turns_median": T.groupby(["scale", "turn"])["bars to next"].median().to_dict(),
        "candidates": C.groupby("extreme").size().to_dict(),
        "candidates_turned_pct": (C.groupby("extreme").turned.mean() * 100).round(1).to_dict(),
    }
    json.dump({k: {str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v for k, v in summary.items()},
              open(os.path.join(OUT, f"summary_{tag}.json"), "w"), indent=1)
    print(json.dumps({k: {str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v
                      for k, v in summary.items()}, indent=1))
