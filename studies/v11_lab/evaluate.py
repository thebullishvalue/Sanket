"""Apply PROTOCOL.md's four stages to the lab's output. No judgement calls: the stages, as declared.

    python3 studies/v11_lab/evaluate.py
"""
import os
import numpy as np, pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
E = {m: pd.read_csv(os.path.join(OUT, f"edge_{m}.csv")) for m in ("daily", "hourly", "weekly")}
SC = {m: pd.read_pickle(os.path.join(OUT, f"scores_{m}.pkl")) for m in ("daily", "hourly", "weekly")}
ERAS = (("2003-12", None, "2013-01-01"), ("2013-19", "2013-01-01", "2020-01-01"), ("2020-26", "2020-01-01", None))


def get(m, sig, side, era):
    x = E[m][(E[m].signal == sig) & (E[m].side == side) & (E[m].era == era)]
    return x.iloc[0] if len(x) else None


rows = []
for sig in sorted(E["daily"].signal.unique()):
    for side in ("buy", "sell"):
        dsc = get("daily", sig, side, "discovery"); hold = get("daily", sig, side, "holdout")
        if dsc is None and hold is None:
            continue
        rep_mode = "weekly" if sig.startswith("P") else ("hourly" if sig.startswith("K") else None)
        r = dict(signal=sig, side="▲" if side == "buy" else "▼")
        r["disc edge"] = dsc.edge if dsc is not None else np.nan
        r["S1 disc"] = bool(dsc is not None and dsc.ci_lo > 0)
        r["hold edge"] = hold.edge if hold is not None else np.nan
        r["hold CI"] = f"[{hold.ci_lo:+.3f}, {hold.ci_hi:+.3f}]" if hold is not None else ""
        r["hold net"] = hold.net if hold is not None else np.nan
        r["S2 hold"] = bool(hold is not None and hold.ci_lo > 0 and hold.net > 0)
        # S3: replication (sets: replicated on BOTH second timeframes, for the record)
        reps = [rep_mode] if rep_mode else ["hourly", "weekly"]
        s3 = []
        for rm in reps:
            f = get(rm, sig, side, "full")
            s3.append(f is not None and f.ci_lo > 0)
            r[f"{rm} full edge"] = f.edge if f is not None else np.nan
        r["S3 repl"] = bool(all(s3)) if rep_mode else bool(any(s3))
        # S4: breadth - groups (holdout) and eras (full), on the per-event scores
        sc = SC["daily"]; x = sc[(sc.kind == sig) & (sc.side == (1 if side == "buy" else -1))]
        g = x[x.holdout].groupby("grp").score.agg(["mean", "size"])
        g = g[g["size"] >= 30]
        pos_groups = int((g["mean"] > 0).sum())
        era_means = []
        for name, a, b in ERAS:
            m = np.ones(len(x), bool)
            if a: m &= x.date.to_numpy() >= np.datetime64(a)
            if b: m &= x.date.to_numpy() < np.datetime64(b)
            era_means.append(x.score[m].mean() if m.sum() >= 20 else np.nan)
        r["groups +"] = f"{pos_groups}/{len(g)}"
        r["eras"] = " ".join(f"{v:+.3f}" if np.isfinite(v) else "n/a" for v in era_means)
        r["S4 breadth"] = bool(pos_groups >= 2 and all(np.isfinite(v) and v > 0 for v in era_means))
        r["PASS"] = r["S1 disc"] and r["S2 hold"] and r["S3 repl"] and r["S4 breadth"]
        r["events"] = int(get("daily", sig, side, "full").events) if get("daily", sig, side, "full") is not None else 0
        r["per 1k"] = round(float(get("daily", sig, side, "full").per_1k), 2) if get("daily", sig, side, "full") is not None else 0
        rows.append(r)
R = pd.DataFrame(rows)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30)
cols = ["signal", "side", "events", "per 1k", "disc edge", "S1 disc", "hold edge", "hold CI", "hold net", "S2 hold",
        "hourly full edge", "weekly full edge", "S3 repl", "groups +", "eras", "S4 breadth", "PASS"]
print(R[[c for c in cols if c in R.columns]].round(4).to_string(index=False))
R.to_csv(os.path.join(OUT, "evaluation.csv"), index=False)
win = R[R.PASS & ~R.signal.isin(list("ABC"))].sort_values("hold net", ascending=False)
print("\nWORKS (passes S1-S4):", "none" if win.empty else ", ".join(f"{a} {b}" for a, b in zip(win.signal, win.side)))
print("sets passing:", ", ".join(f"{a} {b}" for a, b in zip(R[R.PASS & R.signal.isin(list('ABC'))].signal,
                                                              R[R.PASS & R.signal.isin(list('ABC'))].side)) or "none")
