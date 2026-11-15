"""v13 lab - which readings separate good pool events from bad, the same way on every dataset?

    python3 studies/v13_lab/analyze.py [RD RH FD FH]

For each pool x side and each feature: quintile cut points from all datasets pooled (so every
dataset is binned the same way), the mean score per quintile per dataset, and the top-minus-bottom
spread per dataset. A feature is CONSISTENT when its spread has the same sign on every dataset.
Writes out/quintiles.csv and prints the consistent ones, strongest worst-case first.
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scorers                                              # noqa: E402
SCORE = os.environ.get("V13_SCORE", "score")              # score (drift-free) | score_xs | score_raw
OUT = os.path.join(HERE, "out")
DSS = sys.argv[1:] or ["RD", "RH", "FD", "FH"]
NUM = ["T", "ladder", "rungs_up", "rungs_dn", "u", "u_min10", "u_max10", "stretch", "vs_mid", "risk_rank",
       "risk_mom", "z", "ret5", "rel_vol", "vs_vwap", "since_prev", "c_tape", "v_tape", "trace"]


def load(ds):
    E = scorers.load(ds)
    E["score"] = E[SCORE]                                   # the scorer under test
    return E[np.isfinite(E["score"])]


def tclust(x, d):
    """Date-clustered t of a mean (events on one date are not independent)."""
    g = pd.DataFrame({"x": x, "d": d}).groupby("d").x.mean()
    return g.mean() / (g.std() / np.sqrt(len(g))) if len(g) > 10 and g.std() > 0 else np.nan


def main():
    A = pd.concat([load(d) for d in DSS], ignore_index=True)
    rows = []
    for (pool, side), P in A.groupby(["pool", "side"]):
        for f in NUM:
            if f not in P or P[f].notna().sum() < 1000:
                continue
            q = P[f].quantile([.2, .4, .6, .8]).to_numpy()
            if len(np.unique(q)) < 4:
                continue
            P2 = P[P[f].notna()].assign(bin=lambda x: np.searchsorted(q, x[f], side="right"))
            for ds, Q in P2.groupby("ds"):
                if Q[f].notna().sum() < 500:
                    continue
                m = Q.groupby("bin").score.mean()
                rows.append(dict(pool=pool, side=int(side), feature=f, ds=ds, n=len(Q),
                                 **{f"q{b}": m.get(b, np.nan) for b in range(5)},
                                 spread=m.get(4, np.nan) - m.get(0, np.nan),
                                 t_top=tclust(Q.score[Q.bin == 4], Q.date[Q.bin == 4]),
                                 t_bot=tclust(Q.score[Q.bin == 0], Q.date[Q.bin == 0]),
                                 cuts=",".join(f"{v:.3g}" for v in q)))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, f"quintiles_{SCORE}_{'_'.join(DSS)}.csv"), index=False)
    S = R.groupby(["pool", "side", "feature"]).agg(n_ds=("ds", "size"), min_spread=("spread", "min"),
                                                   max_spread=("spread", "max"), cuts=("cuts", "first"))
    S["consistent"] = (S.min_spread > 0) | (S.max_spread < 0)
    S["worst"] = np.where(S.min_spread > 0, S.min_spread, np.where(S.max_spread < 0, -S.max_spread, 0))
    S = S[S.consistent].sort_values("worst", ascending=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(S.round(3).to_string())


if __name__ == "__main__":
    main()
