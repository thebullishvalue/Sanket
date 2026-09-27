"""Build a per-instrument feature cache of Pragati v8 (the Sanket port) for the fresh audit."""
import os, sys, pickle, time, warnings
warnings.filterwarnings("ignore")
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); sys.path.insert(0, REPO)
os.environ.setdefault("PINE_AUDIT_CACHE", os.environ.get("PINE_AUDIT_CACHE", "cache"))
import numpy as np, pandas as pd
import pine_audit as pa, engine as eng, samanvaya as sv, pragati as pg, cvgrid as cg, pine_v5 as p5
OUT = os.environ.get("V9_LAB_DIR", "v9_work") + "/feat"
_D = {}
def init():
    _D["drv"] = pa.load_drivers()
def job(a):
    g, tkr, df = a
    fn = f"{OUT}/{g}__{tkr.replace('/','_').replace('=','_').replace('^','_')}.pkl"
    if os.path.exists(fn): return tkr, "cached"
    try:
        lo = eng._chart_bars(df)
        val = sv.compute_value(lo, _D["drv"], tkr, chart="D")
        P = eng.settings_for(None, None, "Daily").params
        out = pg.compute(lo, val, P, chart="D")
        ch = pg.chart_conviction(lo, P)
        cvr = np.cumsum(ch["sd_ok"].to_numpy(bool)) > P.norm + P.smooth + P.signal
        gr = cg.classify(out, ch["osc"], ch["raw_sd"], cvr, P)
        sig = p5.signals(out, ch["osc"], P)
        div = p5.divergence(lo, out, ch["osc"], cvr, P)
        f = pd.DataFrame(index=lo.index)
        for c in ("open", "high", "low", "close", "volume"): f[c] = lo[c]
        for c in ("conv", "value", "trace", "hist", "hist_z", "push", "push_tier", "quiet", "c_tape", "v_tape",
                  "c_ready", "v_ready", "stack_ok", "trace_ok", "eff_pct", "eff_abs", "abs_seen", "decl", "armed"):
            f[c] = out[c]
        for c in ("rv_z", "breadth_z", "value_z", "hedge", "model_built", "basket_warm"): f["sv_" + c] = val[c].reindex(lo.index)
        f["cv"] = ch["osc"]; f["cv_ready"] = cvr
        for c in gr.columns: f[c] = gr[c]
        f["turn_buy"] = sig.v5_buy.to_numpy(); f["turn_sell"] = sig.v5_sell.to_numpy()
        f["con_l"] = sig.v5_con_l.to_numpy(); f["con_s"] = sig.v5_con_s.to_numpy(); f["v5_decl"] = sig.v5_decl.to_numpy()
        f["r_bull"] = div.v5_r_bull.to_numpy(); f["r_bear"] = div.v5_r_bear.to_numpy()
        f["h_bull"] = div.v5_h_bull.to_numpy(); f["h_bear"] = div.v5_h_bear.to_numpy()
        f.to_pickle(fn)
        return tkr, "ok"
    except Exception as e:
        return tkr, f"ERR {type(e).__name__}: {e}"
if __name__ == "__main__":
    from multiprocessing import Pool
    groups = sys.argv[1:] or ["nse", "us", "idx", "cmd", "fx", "crypto"]
    jobs = [(g, t, d) for g in groups for t, d in pa.load_group(g).items()]
    t0 = time.time(); n = 0
    with Pool(4, initializer=init) as pool:
        for tkr, st in pool.imap_unordered(job, jobs, chunksize=2):
            n += 1
            if st != "cached" and (n % 20 == 0 or st.startswith("ERR")): print(n, len(jobs), tkr, st, f"{time.time()-t0:.0f}s", flush=True)
    print("done", n, f"{time.time()-t0:.0f}s")
