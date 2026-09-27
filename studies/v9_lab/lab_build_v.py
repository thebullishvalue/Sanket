"""Ablation builds: python lab_build_v.py <variant>. Writes feat_<variant>/ with the grid columns only."""
import os, sys, pickle, time, warnings
warnings.filterwarnings("ignore")
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); sys.path.insert(0, REPO)
CACHE = os.environ.get("PINE_AUDIT_CACHE", "cache")
os.environ.setdefault("PINE_AUDIT_CACHE", CACHE)
import numpy as np, pandas as pd
import pine_audit as pa, engine as eng, samanvaya as sv, pragati as pg, cvgrid as cg
from dataclasses import replace
V = sys.argv[1]
OUT = os.environ.get("V9_LAB_DIR", "v9_work") + f"/feat_{V}"
os.makedirs(OUT, exist_ok=True)
PARAMS = {"part_off": {"participation": "Off"}, "den_effort": {"denominator": "Effort"},
          "z1_20": {"z1": 20.0}, "z1_40": {"z1": 40.0}}
_D = {}
def init():
    if V == "leg_rv": sv.LEG_MIX = 1.0
    if V == "leg_br": sv.LEG_MIX = 0.0
    if V in ("theta_10", "theta_20"):
        sv.THETA = 1.0 if V == "theta_10" else 2.0
        sv.THETA_OSC = float(100.0 * sv.softbound(sv.THETA * sv.GAIN))
    _D["drv"] = None if V == "hedge_off" else pa.load_drivers()
def job(a):
    g, tkr, df = a
    fn = f"{OUT}/{g}__{tkr.replace('/','_').replace('=','_').replace('^','_')}.pkl"
    if os.path.exists(fn): return tkr, "cached"
    try:
        lo = eng._chart_bars(df)
        val = sv.compute_value(lo, _D["drv"], tkr, chart="D")
        P = eng.settings_for(None, None, "Daily").params
        if V in PARAMS: P = replace(P, **PARAMS[V])
        out = pg.compute(lo, val, P, chart="D")
        ch = pg.chart_conviction(lo, P)
        cvr = np.cumsum(ch["sd_ok"].to_numpy(bool)) > P.norm + P.smooth + P.signal
        gr = cg.classify(out, ch["osc"], ch["raw_sd"], cvr, P)
        f = pd.DataFrame(index=lo.index)
        for c in ("open", "close"): f[c] = lo[c]
        f["stack_ok"] = out["stack_ok"]; f["cv_ready"] = cvr; f["v_tape"] = out["v_tape"]; f["c_tape"] = out["c_tape"]
        for c in ("cvg_cell", "cvg_units", "cvg_vph", "cvg_cph", "cvg_held"): f[c] = gr[c]
        f.to_pickle(fn); return tkr, "ok"
    except Exception as e:
        return tkr, f"ERR {type(e).__name__}: {e}"
if __name__ == "__main__":
    from multiprocessing import Pool
    jobs = [(g, t, d) for g in ["nse", "us", "idx", "cmd", "fx", "crypto"] for t, d in pa.load_group(g).items()]
    t0 = time.time(); errs = 0
    with Pool(4, initializer=init) as pool:
        for tkr, st in pool.imap_unordered(job, jobs, chunksize=2):
            if st.startswith("ERR"): errs += 1; print(tkr, st)
    print(V, "done", f"{time.time()-t0:.0f}s", "errors", errs, flush=True)
