# v9 audit lab

The scripts behind [`../pragati_v9_audit.md`](../pragati_v9_audit.md). Research only — nothing
in the app imports them.

**Data.** The 380-instrument daily cache and the macro drivers written by the v8 audit's fetch
step (`pine_audit.py`; `multi_*.pkl`, `drivers_raw20.pkl`), and for `expOI.py` the NSE F&O
bhavcopy OI extracts in `<cache>/oi/`. Point `PINE_AUDIT_CACHE` at that folder. Work files go
to `V9_LAB_DIR` (default `./v9_work`); run the experiments from inside it.

```
export PINE_AUDIT_CACHE=/path/to/cache  V9_LAB_DIR=/path/to/v9_work
python lab_build.py                  # daily feature cache (the port, every instrument)  ~35 s on 4 cores
python lab_build_w.py                # the same on weekly bars
for v in part_off den_effort leg_rv leg_br hedge_off z1_20 z1_40 theta_10 theta_20; do
    python lab_build_v.py $v; done    # ablation rebuilds
cd $V9_LAB_DIR
python -c "import lab_core as L, pickle; P = L.load(); [L.add_u(f) for g in P for f in P[g].values()]; pickle.dump(P, open('lab_P.pkl', 'wb'))"
python null1.py; python null2.py; python null3.py     # §1  the measuring stick on random walks
python exp1b.py; python exp2.py                       # §2  every reading, time-series and cross-section
python exp3.py                                        # §3  the grid cell by cell
python exp4.py; python exp5.py; python exp6.py        # §4-6  signals, capitulation, the short side
python exp7.py                                        # §5  refinements of the capitulation turn
python expW.py; python exp8.py                        # weekly; ▼ mirrors, phases, grading
python expA.py                                        # §7  ablations
python expOI.py                                       # §8  open interest
```

`lab_core.py` holds the scorers: `score_c(..., mode="tc")` (position centred on its trailing
250-bar mean, return over trailing volatility) and `mode="xs"` (return net of the group's
same-date mean), both checked to read ≈ 0 on random walks; `score` is the v8 stick, kept only
so `null1.py` can show its bias.
