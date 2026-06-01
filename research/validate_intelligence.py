#!/usr/bin/env python3
"""
Sanket — Signal-Intelligence validation harness (headless).

Answers the one question the synthetic tests cannot: on REAL market data, does the
Layer-2 confidence model actually separate true from false signals, and does the
Set-A ranking calibrate to a positive out-of-sample edge?

It runs the *production* pipeline — the same run_timeseries_analysis harvest and the
same intelligence.py calibrators the app uses — by stubbing only Streamlit (so no
UI/server is needed). Nothing about the math is reimplemented, so the numbers it
prints are the numbers the app would produce.

Usage:
    python validate_intelligence.py --universe "India Indexes" --index "NIFTY 50" \
        --timeframe Daily --lookback-days 730 --trials 60

    python validate_intelligence.py --universe Crypto                # uses that universe's default index
    python validate_intelligence.py --universe "US Indexes" --index "DOW JONES" --ab-f7

Exit code is 0 on success, 1 if the harvest produced no usable panel.
"""
from __future__ import annotations

# Path shim: this script lives in research/ but imports the live app modules
# (sanket, priority_engine, intelligence, gate_engine) and sibling research
# scripts from the repo ROOT. Put the parent dir on sys.path so those resolve
# whether run as "python research/foo.py" or from inside research/.
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_here))   # repo root: sanket, priority_engine, ...
_sys.path.insert(0, _here)                       # research/: sibling scripts (validate_intelligence)

import argparse
import datetime
import sys
import types


# ──────────────────────────────────────────────────────────────────────────────
# Headless Streamlit stub — installed BEFORE importing sanket so module-level
# st.* calls (set_page_config, session_state init, inject_css) become no-ops.
# session_state is a real dict, so run_timeseries_analysis's writes/reads work.
# ──────────────────────────────────────────────────────────────────────────────
def _install_streamlit_stub():
    if "streamlit" in sys.modules:
        return

    class _Slot:
        """Stand-in for st.empty() / a container: every method is a no-op."""
        def __getattr__(self, _name):
            return lambda *a, **k: None
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    class _FakeST(types.ModuleType):
        def __init__(self):
            super().__init__("streamlit")
            self.session_state = {}

        # cache decorators → identity (support @st.cache_data and @st.cache_data(...))
        def cache_data(self, *a, **k):
            if len(a) == 1 and callable(a[0]) and not k:
                return a[0]
            return lambda f: f
        cache_resource = cache_data

        def empty(self, *a, **k):
            return _Slot()
        def container(self, *a, **k):
            return _Slot()
        def columns(self, spec, *a, **k):
            n = spec if isinstance(spec, int) else len(spec)
            return [_Slot() for _ in range(n)]
        def expander(self, *a, **k):
            return _Slot()
        def tabs(self, labels, *a, **k):
            return [_Slot() for _ in labels]
        def sidebar(self):
            return _Slot()

        # everything else (markdown, error, set_page_config, rerun, …) → no-op
        def __getattr__(self, _name):
            return lambda *a, **k: None

    st = _FakeST()
    comp = types.ModuleType("streamlit.components")
    v1 = types.ModuleType("streamlit.components.v1")
    v1.html = lambda *a, **k: None
    comp.v1 = v1
    # Attach as real attributes so `st.components.v1.html(...)` resolves to the stub
    # instead of falling through __getattr__ (which would hand back a bare no-op fn).
    st.components = comp
    sys.modules["streamlit"] = st
    sys.modules["streamlit.components"] = comp
    sys.modules["streamlit.components.v1"] = v1
    return st


def _fmt(x, spec="+.3f", na="—"):
    return format(x, spec) if isinstance(x, (int, float)) else na


def _set_section(title):
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


def main():
    ap = argparse.ArgumentParser(description="Sanket Signal-Intelligence validation harness")
    ap.add_argument("--universe", default="India Indexes",
                    help='e.g. "India Indexes", "US Indexes", "Crypto", "Commodities", "Currency"')
    ap.add_argument("--index", default=None, help="Index within the universe (universe default if omitted)")
    ap.add_argument("--timeframe", default="Daily", choices=["Daily", "Weekly"])
    ap.add_argument("--lookback-days", type=int, default=730)
    ap.add_argument("--trials", type=int, default=60, help="Optuna trials for the ranking calibration")
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--ab-f7", action="store_true",
                    help="Also calibrate ranking with F7 (LO) enabled and compare val IR")
    args = ap.parse_args()

    _install_streamlit_stub()

    # Import production modules AFTER the stub is in place.
    import sanket as S
    import intelligence as intel
    import priority_engine as pe

    # Resolve a default index per universe when not supplied (mirrors the sidebar).
    index = args.index
    if index is None:
        index = {
            "India Indexes":  "NIFTY 50",
            "US Indexes":     "DOW JONES",
            "Global Indexes": "Global Benchmark Indexes",
            "Commodities":    "Global Commodities",
            "Currency":       "Major FX Pairs",
            "Crypto":         "Digital Assets (Top 20)",
            "ETF Index":      "NSE ETF Universe",
            "Global Macro":   "Global Macro Bonds",
        }.get(args.universe)

    reg_len, wt_n1, wt_n2 = 20, 10, 21
    levels = (80, 40, -80, -40)
    wt2_len, wt2_type = 20, "ALMA"
    end_date = S._today_ist()
    start_date = end_date - datetime.timedelta(days=args.lookback_days)

    _set_section("SANKET — SIGNAL INTELLIGENCE VALIDATION (real data)")
    print(f"  Universe   : {args.universe}  ·  Index: {index}")
    print(f"  Timeframe  : {args.timeframe}")
    print(f"  Harvest    : {start_date} → {end_date}  (~{args.lookback_days}d)")
    print(f"  Trials     : {args.trials}   Train/Val: {int(args.train_frac*100)}/{100-int(args.train_frac*100)}")

    # ── 1. Harvest the real panel via the production path ──────────────────────
    print("\n  Harvesting (real fetch + full/regime/divergence analysis)… this hits the network.")
    S.run_timeseries_analysis(
        args.universe, index, start_date, end_date,
        reg_len, wt_n1, wt_n2, levels, args.timeframe,
        wt2_len=wt2_len, wt2_type=wt2_type,
    )
    ts_df = sys.modules["streamlit"].session_state.get("ts_results_df")
    if ts_df is None or getattr(ts_df, "empty", True):
        print("\n  ✗ Harvest produced no usable panel (empty/failed fetch). Try a broader universe "
              "or a longer lookback.")
        return 1

    n_rows = len(ts_df)
    n_dates = ts_df["Date"].nunique()
    n_syms = ts_df["Symbol"].nunique()
    fired = ts_df["SignalType"].isin(["A: Long", "A: Short", "B: Long", "B: Short", "C: Long", "C: Short"])
    print(f"  ✓ Panel: {n_rows} rows · {n_dates} dates · {n_syms} symbols · {int(fired.sum())} fired signals")

    # ── 2. Layer-2 signal-confidence calibration (the headline AUC) ────────────
    _set_section("LAYER 2 · SIGNAL CONFIDENCE  (does it separate true vs false?)")
    model = intel.calibrate_signal_confidence(ts_df, train_frac=args.train_frac)
    if not model:
        print("  ✗ Too sparse to calibrate a confidence model — falls back to the Layer-1 heuristic live.")
        print("    (Need ≥150 fired signals with resolved forward returns.)")
    else:
        auc   = model.get("val_auc")
        lift  = model.get("val_precision_lift")
        tprec = model.get("val_top_half_precision")
        base  = model.get("base_rate_val", model.get("base_rate"))
        sets  = [s for s in ("A", "B", "C") if s in model.get("sets", {})]
        verdict = ("STRONG"  if isinstance(auc, (int, float)) and auc >= 0.60 else
                   "USEFUL"  if isinstance(auc, (int, float)) and auc >= 0.55 else
                   "WEAK"    if isinstance(auc, (int, float)) and auc >= 0.52 else
                   "NO EDGE")
        print(f"  Confirm AUC (out-of-sample) : {_fmt(auc, '.3f')}   → {verdict}")
        print(f"  Precision lift (top-half)   : {_fmt(lift, '+.1%')}   "
              f"(top {_fmt(tprec, '.1%')} vs base {_fmt(base, '.1%')})")
        print(f"  Horizons / deadband         : {model.get('horizons')}  /  {_fmt(model.get('deadband'), '.5f')}")
        print(f"  Sets modeled                : {', '.join(sets) or 'pooled-only'}   ({model.get('n_train')} train signals)")
        # Per-set coefficient read (which features the model leaned on)
        names = model.get("feature_names", [])
        for s in (["_pooled"] + sets):
            m = model["sets"].get(s)
            if not m:
                continue
            top = sorted(zip(names, m["coef"]), key=lambda kv: -abs(kv[1]))[:4]
            tag = "pooled" if s == "_pooled" else f"Set {s}"
            print(f"    {tag:8s} top features: " + ", ".join(f"{n} {c:+.2f}" for n, c in top))
        print("\n  READ: AUC ≥0.55 = the Hide filter / confluence penalty are meaningful. "
              "<0.52 = treat as noise; keep the filter Off / advisory.")

    # ── 3. Layer-3 ranking calibration (IR) — baseline (F7 off) ────────────────
    _set_section("LAYER 3 · RANKING CALIBRATION  (out-of-sample IR)")
    tuner = intel.PriorityTuner(ts_df, train_frac=args.train_frac, enable_f7=False)
    if tuner._train_pre.empty or tuner._train_pre.n_groups < 10:
        print("  ✗ Not enough usable training dates for a stable IC calibration "
              f"(have {0 if tuner._train_pre.empty else tuner._train_pre.n_groups}, need ≥10).")
    else:
        _bw, train_ir = tuner.optimize(n_trials=args.trials)
        val_ir = tuner.evaluate_validation()
        imp = tuner.get_param_importance()
        top = sorted(imp.items(), key=lambda kv: -kv[1])[:5] if imp else []
        print(f"  Train IR : {_fmt(train_ir)}    Val IR : {_fmt(val_ir)}   "
              f"({'edge' if isinstance(val_ir,(int,float)) and val_ir > 0 else 'no demonstrated edge'})")
        if top:
            print("  Top factors (fANOVA): " + ", ".join(f"{k} {v:.0f}%" for k, v in top))

        # ── 4. Optional F7 A/B ──────────────────────────────────────────────────
        if args.ab_f7:
            tuner_f7 = intel.PriorityTuner(ts_df, train_frac=args.train_frac, enable_f7=True)
            _bw7, train_ir7 = tuner_f7.optimize(n_trials=args.trials)
            val_ir7 = tuner_f7.evaluate_validation()
            imp7 = tuner_f7.get_param_importance()
            f7share = sum(v for k, v in imp7.items() if "F7" in k)
            delta = (val_ir7 - val_ir) if isinstance(val_ir, (int, float)) and isinstance(val_ir7, (int, float)) else None
            print("\n  ── F7 (LO reversion) A/B ──")
            print(f"  Val IR  no-F7: {_fmt(val_ir)}   with-F7: {_fmt(val_ir7)}   Δ: {_fmt(delta)}")
            print(f"  F7 fANOVA importance: {f7share:.1f}%   "
                  f"learned β_long: {_fmt(_bw7.get('beta_F7_liq_long'), '+.1f')}")
            keep = (isinstance(delta, (int, float)) and delta > 0.0 and f7share >= 5.0)
            print(f"  VERDICT: {'F7 earns its place — consider enabling.' if keep else 'F7 does NOT beat the baseline — keep it gated (default).'}")

    _set_section("DONE")
    print("  This ran the production harvest + calibrators on real data. The Confirm AUC above")
    print("  is the number that decides whether the intelligence layer is a sharp tool or scaffolding.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
