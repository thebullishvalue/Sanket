"""
trace_study.py — which trace should Sanket screen on? Measured, not assumed.

pragati.pine offers three readings for its trace — the line the histogram, the push,
the TURN / RESUME signals and the grid's row timing all follow:

    Conviction × value   both ingredients in σ, blended equally   (the indicator's default)
    Conviction only      flow — who is in control
    Value only           position — where price stands against fair value

Sanket shipped the default. This module backtests the SCREENER under all three on the same
names, the same bars and the same value engine, and reports which one screens names that
work — with confidence intervals, a sealed holdout, and the three compared PAIRED on the
same dates, so "better" means better beyond noise rather than a bigger point estimate.

What is held fixed across the three
-----------------------------------
The value engine (Samanvaya) is computed ONCE per name and shared — the trace setting never
touches it. The tapes and the grid's cell never follow the trace either. Every other input
is the Pine's default. Only ``Params.mix`` changes (0.5 / 0.0 / 1.0), and every column is
produced by ``engine.add_pragati_features`` — the call the screener makes — so the study
measures exactly what the screen would have shown.

Two measurements per mode
-------------------------
A · THE EVENTS (``edge.measure``, unchanged). ▲▼ TURN and ◆ RESUME, entered on the next bar
    and held ``horizon`` bars; drift-removed and vol-normalised within era; block-bootstrap
    CI over dates; costs charged; power stated. The same six slices as the Edge Study.

B · THE SCREEN. On every date, the names are ranked by the screener's priority
    (``engine.priorities`` — the function the Action Dashboard ranks with). Measured:

      Long book    the top ``quantile`` of names by long priority — their mean score MINUS
                   the whole cross-section's mean that date (selection, not market beta)
      Short book   the same for short priority, sign-folded (positive = they fell behind)
      Long−short   long book minus short book
      IC long / IC short   Spearman rank correlation of each side's priority with the
                   forward score, per date

    A name's score is its h-bar forward return, minus its own mean and divided by its own σ
    inside the era (as the Edge Study scores). Per-date values overlap by ``horizon`` bars,
    so the CI resamples contiguous blocks of ``horizon`` dates.

Deciding
--------
The rule is fixed before the numbers are seen: a mode is better than the default only if
the HOLDOUT confidence interval of its PAIRED per-date difference in the long−short spread
excludes zero. Anything short of that is "no measurable difference", and the default stays.
Three modes, several metrics — a lone interval that barely clears zero is weak evidence,
and the report says which metrics agree.

Run it
------
    python trace_study.py --index "NIFTY 50"                      # NSE index constituents
    python trace_study.py --tickers RELIANCE.NS,TCS.NS,INFY.NS ... # any yfinance tickers
    python trace_study.py --index "NIFTY 100" --timeframe Weekly --years 15

Writes ``trace_study_<universe>_<tf>.md`` (the report) and ``.json`` (every number).
Needs network access to Yahoo Finance (prices) and, for ``--index``, NSE archives.

Author: @thebullishvalue
"""
from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import sys
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

import edge
import engine as eng
import samanvaya as sv

#: The three readings, in the Pine's order. key -> (label, value's weight in the trace).
MODES = {
    "cxv":   ("Conviction × value", 0.5),
    "conv":  ("Conviction only",    0.0),
    "value": ("Value only",         1.0),
}
DEFAULT_MODE = "cxv"

QUANTILE = 0.20          # the screen's book: top fifth of names by priority
MIN_NAMES = 5            # a date with fewer eligible names does not form books
HOLDOUT_FRAC = 0.40      # the most recent 40% of dates, sealed — as the Edge Study
MIN_ERA_BARS = 30        # a name needs this many in-era forward returns to be scored

_KEEP = ("turn_buy", "turn_sell", "resume_long", "resume_short", "PRG_Trace", "CVG_Units",
         "PRG_Armed", "PRG_Armed_Age", "PRG_Hold_Dir", "PRG_Hold_Age", "PRG_Push")

SCREEN_METRICS = {
    "ls":       "Long − short spread",
    "long":     "Long book vs cross-section",
    "short":    "Short book vs cross-section",
    "ic_long":  "IC · long priority",
    "ic_short": "IC · short priority",
}


# ════════════════════════════════════════════════════════════════════════════════════════
# PER-SYMBOL REDUCTION  (one value engine, three traces)
# ════════════════════════════════════════════════════════════════════════════════════════
def mode_settings(base: "eng.EngineSettings", mode: str) -> "eng.EngineSettings":
    return replace(base, params=replace(base.params, mix=float(MODES[mode][1])))


def reduce_symbol(df: pd.DataFrame, drivers, symbol: str, base: "eng.EngineSettings",
                  daily: pd.DataFrame | None = None) -> dict:
    """{mode: frame of the screen's inputs} for one name, plus 'fwd' and 'events'.

    The value engine runs once; each mode then runs the full Pragati stack through
    ``engine.add_pragati_features`` with that value frame.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    h = int(base.horizon)
    close = pd.to_numeric(df["Close"], errors="coerce")
    # EXEC-B, as edge.symbol_events: enter on the bar after the signal, hold h bars.
    fwd = close.shift(-1 - h) / close.shift(-1) - 1.0
    val = eng.value_frame(df, drivers, symbol, base)
    out = {"fwd": fwd, "events": {}}
    for mode in MODES:
        s = mode_settings(base, mode)
        f = eng.add_pragati_features(df, drivers, symbol, s, daily, value=val)
        keep = f[list(_KEEP)].copy()
        out[mode] = keep
        # The events, exactly as edge.symbol_events extracts them.
        parts = []
        fw = fwd.reindex(f.index).to_numpy()
        for col, side, kind in (("turn_buy", 1.0, "turn"), ("turn_sell", -1.0, "turn"),
                                ("resume_long", 1.0, "resume"), ("resume_short", -1.0, "resume")):
            m = f[col].fillna(False).astype(bool).to_numpy() & np.isfinite(fw)
            if m.any():
                parts.append(pd.DataFrame({"date": f.index[m], "side": side, "kind": kind,
                                           "fwd": fw[m], "symbol": symbol}))
        out["events"][mode] = (pd.concat(parts, ignore_index=True) if parts else None)
    return out


# ════════════════════════════════════════════════════════════════════════════════════════
# THE SCREEN  (per-date books from the banded priority)
# ════════════════════════════════════════════════════════════════════════════════════════
def _panel(frames: dict, col: str, index: pd.DatetimeIndex) -> np.ndarray:
    return pd.DataFrame({s: f[col] for s, f in frames.items()}).reindex(index).to_numpy(dtype=float)


def priority_panels(frames: dict, index: pd.DatetimeIndex, base: "eng.EngineSettings"):
    """(P_long, P_short, trace) as date × symbol arrays, via engine.priorities."""
    b = lambda c: np.nan_to_num(_panel(frames, c, index)).astype(bool)   # noqa: E731
    trace = _panel(frames, "PRG_Trace", index)
    pl, ps = eng.priorities(b("turn_buy"), b("turn_sell"), trace, np.isfinite(trace))
    return pl, ps, trace


def era_scores(fwd: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Forward returns → drift-removed, vol-normalised scores, stats taken inside ``rows``."""
    sub = fwd[rows]
    enough = np.isfinite(sub).sum(axis=0) >= MIN_ERA_BARS
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.where(enough, np.nanmean(sub, axis=0), np.nan)
        sd = np.where(enough, np.nanstd(sub, axis=0, ddof=1), np.nan)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, np.nan)
    out = np.full_like(fwd, np.nan)
    out[rows] = (sub - mu) / sd
    return out


def _rowwise_spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman correlation across columns, one value per row, on jointly finite cells."""
    ok = np.isfinite(a) & np.isfinite(b)
    ra = pd.DataFrame(np.where(ok, a, np.nan)).rank(axis=1).to_numpy()
    rb = pd.DataFrame(np.where(ok, b, np.nan)).rank(axis=1).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        ra = ra - np.nanmean(ra, axis=1, keepdims=True)
        rb = rb - np.nanmean(rb, axis=1, keepdims=True)
        num = np.nansum(ra * rb, axis=1)
        den = np.sqrt(np.nansum(ra * ra, axis=1) * np.nansum(rb * rb, axis=1))
        r = num / den
    r[ok.sum(axis=1) < MIN_NAMES] = np.nan
    return r


def screen_series(pl: np.ndarray, ps: np.ndarray, score: np.ndarray,
                  quantile: float = QUANTILE) -> dict:
    """Per-date screen metrics (NaN on dates that cannot form books)."""
    ok = np.isfinite(pl) & np.isfinite(ps) & np.isfinite(score)
    n = ok.sum(axis=1)
    usable = n >= MIN_NAMES
    s = np.where(ok, score, np.nan)
    with np.errstate(invalid="ignore"):
        xs_mean = np.nanmean(s, axis=1)

    def _book(p):
        # Percentile rank inside the date's eligible names, ties averaged, so a tie is
        # never broken by column order. A book is every name at or above the (1 − q) percentile.
        pr = pd.DataFrame(np.where(ok, p, np.nan)).rank(axis=1, pct=True).to_numpy()
        m = ok & (pr >= 1.0 - quantile)
        with np.errstate(invalid="ignore"):
            return np.nansum(np.where(m, s, 0.0), axis=1) / np.where(m.sum(axis=1) > 0, m.sum(axis=1), np.nan)

    long_b, short_b = _book(pl), _book(ps)
    out = {
        "long": long_b - xs_mean,
        "short": xs_mean - short_b,
        "ls": long_b - short_b,
        "ic_long": _rowwise_spearman(pl, s),
        "ic_short": _rowwise_spearman(ps, -s),
    }
    for k in out:
        out[k] = np.where(usable, out[k], np.nan)
    return out


def _ci(values: np.ndarray, block: int) -> dict:
    """Mean and block-bootstrap CI of a per-date series (NaN dates dropped, order kept)."""
    v = values[np.isfinite(values)]
    if v.size < 2:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_dates": int(v.size)}
    lo, hi = edge.block_bootstrap_ci(v, np.arange(v.size), v.size, block=block)
    return {"mean": float(v.mean()), "lo": lo, "hi": hi, "n_dates": int(v.size)}


# ════════════════════════════════════════════════════════════════════════════════════════
# THE STUDY
# ════════════════════════════════════════════════════════════════════════════════════════
def run(data: dict, drivers, base: "eng.EngineSettings", *, timeframe: str = "Daily",
        quantile: float = QUANTILE, log=print) -> dict:
    """Measure the three traces on ``data`` = {ticker: daily OHLCV}. Returns a result dict."""
    h = int(base.horizon)
    frames = {m: {} for m in MODES}
    fwds, baselines, ret_cols, events = {}, {}, {}, {m: [] for m in MODES}
    t0 = time.time()
    for i, (tkr, daily) in enumerate(data.items()):
        f = resample_weekly(daily) if timeframe == "Weekly" else daily
        if f is None or len(f) < base.min_bars + h + 3:
            log(f"  skip {tkr}: {0 if f is None else len(f)} bars < {base.min_bars + h + 3}")
            continue
        try:
            r = reduce_symbol(f, drivers, tkr, base, daily if timeframe == "Weekly" else None)
        except Exception as e:                       # one bad name never ends the study
            log(f"  skip {tkr}: {type(e).__name__}: {e}")
            continue
        for m in MODES:
            frames[m][tkr] = r[m]
            if r["events"][m] is not None:
                events[m].append(r["events"][m])
        fwds[tkr] = r["fwd"]
        baselines[tkr] = edge.symbol_baseline(pd.to_numeric(f["Close"], errors="coerce"), h)
        ret_cols[tkr] = pd.to_numeric(f["Close"], errors="coerce").pct_change().tail(1000)
        log(f"  [{i + 1}/{len(data)}] {tkr} · {len(f)} bars · {time.time() - t0:.0f}s")
    names = list(fwds)
    if len(names) < MIN_NAMES:
        raise RuntimeError(f"only {len(names)} usable names — too thin to screen")

    index = pd.DatetimeIndex(sorted(set().union(*[fwds[s].index for s in names])))
    fwd = pd.DataFrame(fwds).reindex(index)[names].to_numpy(dtype=float)
    ret_matrix = pd.DataFrame(ret_cols)
    # A date belongs to the study once any name's priority is defined on it.
    pan = {m: priority_panels(frames[m], index, base) for m in MODES}
    live = np.zeros(len(index), dtype=bool)
    for m in MODES:
        live |= np.isfinite(pan[m][0]).any(axis=1)
    live_idx = np.flatnonzero(live)
    cut = live_idx[int(len(live_idx) * (1.0 - HOLDOUT_FRAC))]
    eras = {"discovery": (live & (np.arange(len(index)) < cut)),
            "holdout": (live & (np.arange(len(index)) >= cut)),
            "full": live}
    split_date = str(index[cut].date())

    res = {"meta": {
        "timeframe": timeframe, "horizon": h, "cost_bps": base.cost_bps, "quantile": quantile,
        "n_names": len(names), "names": names, "start": str(index[live_idx[0]].date()),
        "end": str(index[live_idx[-1]].date()), "split_date": split_date,
        "holdout_frac": HOLDOUT_FRAC, "part_ratio": edge.participation_ratio(ret_matrix),
        "drivers": drivers is not None, "params": asdict(base.params),
        "measured_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }, "screen": {}, "paired": {}, "events": {}}

    # ── B · the screen, per era, per mode, and paired against the default ──
    per_date = {}
    for era, rows in eras.items():
        score = era_scores(fwd, rows)
        score[~rows] = np.nan
        for m in MODES:
            pl, ps, trace = pan[m]
            ser = screen_series(pl, ps, score, quantile)
            ser["ic_trace"] = _rowwise_spearman(trace, score)
            per_date[(era, m)] = ser
            res["screen"].setdefault(m, {})[era] = {k: _ci(v, h) for k, v in ser.items()}
        for m in MODES:
            if m == DEFAULT_MODE:
                continue
            res["paired"].setdefault(m, {})[era] = {
                k: _ci(per_date[(era, m)][k] - per_date[(era, DEFAULT_MODE)][k], h)
                for k in SCREEN_METRICS}

    # ── A · the events, through the Edge Study's own measurement ──
    for m in MODES:
        ev = pd.concat(events[m], ignore_index=True) if events[m] else None
        st = edge.measure(ev, baselines, ret_matrix, universe="trace-study", selected_index=None,
                          timeframe=timeframe, iclass=base.iclass, length=base.params.length,
                          trigger=f"TURN · RESUME on the {MODES[m][0]} trace", horizon=h,
                          cost_bps=base.cost_bps, n_symbols_universe=len(names),
                          n_bars_median=int(np.median([len(b) for b in baselines.values()])),
                          holdout_frac=HOLDOUT_FRAC)
        res["events"][m] = {"study": st.to_dict(),
                            "verdicts": {k: list(st.verdict(k)) for k in edge.SLICES}}
    res["decision"] = decide(res)
    return res


def decide(res: dict) -> dict:
    """Apply the pre-declared rule. Returns {winner, reason, agree}."""
    better, worse = [], []
    for m, eras in res["paired"].items():
        d = eras.get("holdout", {}).get("ls")
        if d and np.isfinite(d["lo"]) and d["lo"] > 0:
            better.append((d["mean"], m))
        if d and np.isfinite(d["hi"]) and d["hi"] < 0:
            worse.append(m)
    agree = {}
    for m, eras in res["paired"].items():
        h = eras.get("holdout", {})
        agree[m] = {k: ("better" if v["lo"] > 0 else "worse" if v["hi"] < 0 else "tie")
                    for k, v in h.items() if np.isfinite(v["lo"])}
    # Does any setting's screen WORK on its own — holdout long−short CI above zero?
    works = {m: bool(np.isfinite(v["holdout"]["ls"]["lo"]) and v["holdout"]["ls"]["lo"] > 0)
             for m, v in res["screen"].items()}
    backwards = [m for m, v in res["screen"].items()
                 if np.isfinite(v["holdout"]["ls"]["hi"]) and v["holdout"]["ls"]["hi"] < 0]
    if better:
        _, win = max(better)
        disc = res["paired"][win].get("discovery", {}).get("ls", {})
        both = bool(disc) and np.isfinite(disc.get("lo", np.nan)) and disc["lo"] > 0
        own = ("and its own screen works (holdout long−short CI above zero)" if works[win] else
               "BUT its own screen shows no edge on the holdout — it loses less than the "
               "default, it does not pick winners")
        return {"winner": win, "agree": agree, "worse": worse, "works": works,
                "backwards": backwards,
                "reason": (f"{MODES[win][0]} beats {MODES[DEFAULT_MODE][0]} on the holdout "
                           f"long−short spread, paired, CI excluding zero, {own}"
                           + ("; discovery confirms the difference" if both else
                              "; discovery does not confirm the difference"))}
    return {"winner": DEFAULT_MODE, "agree": agree, "worse": worse, "works": works,
            "backwards": backwards,
            "reason": ("no trace setting beats the default on the holdout long−short spread "
                       "beyond noise — the default stays")}


# ════════════════════════════════════════════════════════════════════════════════════════
# REPORT
# ════════════════════════════════════════════════════════════════════════════════════════
def _fmt(d: dict, pct: bool = False) -> str:
    if not d or not np.isfinite(d.get("mean", np.nan)):
        return "—"
    return f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"


def report_md(res: dict, title: str) -> str:
    m = res["meta"]
    L = [f"# Trace study · {title}", "",
         f"{m['n_names']} names · {m['timeframe']} · {m['start']} → {m['end']} · holdout from "
         f"{m['split_date']} ({int(m['holdout_frac'] * 100)}%) · hold {m['horizon']} bars · "
         f"book = top {int(m['quantile'] * 100)}% · participation ratio {m['part_ratio']:.1f} · "
         f"macro drivers {'on' if m['drivers'] else 'OFF (unhedged)'} · measured {m['measured_at']}",
         "", f"**Decision:** {res['decision']['reason']}.", ""]
    if res["decision"].get("backwards"):
        L += [f"**Warning:** the screen ranks BACKWARDS on the holdout under "
              f"{', '.join(MODES[k][0] for k in res['decision']['backwards'])} — the long book "
              "trailed the short book beyond noise.", ""]
    L += [
         "Scores are in σ of each name's own h-bar return (drift removed within era). "
         "Brackets are 95% block-bootstrap intervals.", ""]
    for era in ("holdout", "discovery"):
        L += [f"## The screen · {era}", "",
              "| Metric | " + " | ".join(MODES[k][0] for k in MODES) + " |",
              "|---|" + "---|" * len(MODES)]
        for k, lab in list(SCREEN_METRICS.items()) + [("ic_trace", "IC · the trace level")]:
            L.append(f"| {lab} | " + " | ".join(_fmt(res['screen'][md][era].get(k)) for md in MODES) + " |")
        L += ["", f"**Paired against {MODES[DEFAULT_MODE][0]}** (mode − default, same dates):", "",
              "| Metric | " + " | ".join(MODES[k][0] for k in MODES if k != DEFAULT_MODE) + " |",
              "|---|" + "---|" * (len(MODES) - 1)]
        for k, lab in SCREEN_METRICS.items():
            L.append(f"| {lab} | " + " | ".join(_fmt(res['paired'][md][era].get(k))
                                                 for md in MODES if md != DEFAULT_MODE) + " |")
        L.append("")
    L += ["## The events (Edge Study method)", "",
          "| Slice | " + " | ".join(MODES[k][0] for k in MODES) + " |",
          "|---|" + "---|" * len(MODES)]
    for sl in edge.SLICES:
        cells = []
        for md in MODES:
            st = edge.EdgeStudy.from_dict(res["events"][md]["study"])
            r = st.get(sl, "holdout")
            lbl = res["events"][md]["verdicts"][sl][0]
            cells.append(f"{lbl} · " + (f"{r.edge:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] · n {r.n_events}"
                                        if r is not None else "no events"))
        L.append(f"| {edge.SLICE_LABEL[sl]} (holdout) | " + " | ".join(cells) + " |")
    L += ["", "Rule, fixed in advance: a mode replaces the default only if the holdout interval "
          "of its paired long−short difference excludes zero.", ""]
    return "\n".join(L)


# ════════════════════════════════════════════════════════════════════════════════════════
# DATA  (the CLI's; the app has its own fetch path)
# ════════════════════════════════════════════════════════════════════════════════════════
def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """The app's weekly bars (W-MON, left-closed), complete weeks only."""
    w = df.resample("W-MON", closed="left", label="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    return w.dropna()


def index_constituents(index: str) -> list:
    import requests
    slug = {"NIFTY 50": "nifty50", "NIFTY NEXT 50": "niftynext50", "NIFTY 100": "nifty100",
            "NIFTY 200": "nifty200", "NIFTY 500": "nifty500", "NIFTY MIDCAP 100": "niftymidcap100",
            "NIFTY SMLCAP 100": "niftysmallcap100", "NIFTY BANK": "niftybank"}.get(index.upper())
    if slug is None:
        raise SystemExit(f"unknown index {index!r}; pass --tickers instead")
    hdr = {"User-Agent": "Mozilla/5.0"}   # NSE refuses some full browser strings; the short one passes
    # NSE throttles bursts with a 403 — retry each host with backoff before giving up.
    for host in [h for _ in range(3) for h in ("nsearchives.nseindia.com", "archives.nseindia.com")]:
        try:
            r = requests.get(f"https://{host}/content/indices/ind_{slug}list.csv", headers=hdr, timeout=20)
            r.raise_for_status()
            d = pd.read_csv(io.StringIO(r.text))
            col = next(c for c in d.columns if c.lower() == "symbol")
            return [f"{s}.NS" for s in d[col].astype(str).str.strip() if s]
        except Exception as e:
            print(f"  {host}: {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(5)
    raise SystemExit("could not fetch the index constituents; pass --tickers instead")


def fetch(tickers: list, years: int, chunk: int = 20) -> dict:
    import yfinance as yf
    end = _dt.date.today() + _dt.timedelta(days=1)
    start = end - _dt.timedelta(days=int(years * 365.25))
    out = {}
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        raw = yf.download(part, start=start, end=end, progress=False, auto_adjust=True,
                          group_by="ticker", threads=True)
        for t in part:
            try:
                f = raw.xs(t, level=0, axis=1) if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                continue
            f = f.dropna(subset=["Close"])
            if len(f):
                f.index = pd.to_datetime(f.index).tz_localize(None)
                out[t] = f
    return out


def fetch_drivers(years: int, chart: str):
    import yfinance as yf
    end = _dt.date.today() + _dt.timedelta(days=5)
    start = end - _dt.timedelta(days=int(years * 365.25) + 365)
    try:
        raw = yf.download(sv.DRIVER_TICKERS, start=start, end=end, progress=False,
                          auto_adjust=True, threads=True)
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        close = pd.DataFrame(close).dropna(how="all", axis=1)
        close.index = pd.to_datetime(close.index).tz_localize(None)
        return sv.prepare_drivers(close, chart) if not close.empty else None
    except Exception as e:
        print(f"  macro drivers unavailable ({type(e).__name__}: {e}) — value runs unhedged",
              file=sys.stderr)
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--index", help='NSE index, e.g. "NIFTY 50"')
    g.add_argument("--tickers", help="comma-separated yfinance tickers")
    ap.add_argument("--timeframe", default="Daily", choices=("Daily", "Weekly"))
    ap.add_argument("--years", type=int, default=15)
    ap.add_argument("--cap", type=int, default=80, help="max names (fixed-seed sample above it)")
    ap.add_argument("--quantile", type=float, default=QUANTILE)
    ap.add_argument("--out", default=".")
    a = ap.parse_args(argv)

    tickers = index_constituents(a.index) if a.index else [t.strip() for t in a.tickers.split(",") if t.strip()]
    if len(tickers) > a.cap:
        rng = np.random.default_rng(20260927)
        tickers = sorted(rng.choice(tickers, size=a.cap, replace=False).tolist())
    label = a.index or f"{len(tickers)} tickers"
    print(f"Trace study · {label} · {a.timeframe} · {a.years}y · {len(tickers)} names")
    base = eng.settings_for(None, None, a.timeframe)
    data = fetch(tickers, a.years)
    print(f"  fetched {len(data)} of {len(tickers)}")
    drivers = fetch_drivers(a.years, eng.chart_of(a.timeframe))
    res = run(data, drivers, base, timeframe=a.timeframe, quantile=a.quantile)
    md = report_md(res, f"{label} · {a.timeframe}")
    slug = "".join(ch if ch.isalnum() else "_" for ch in label.lower()).strip("_")
    stem = f"{a.out.rstrip('/')}/trace_study_{slug}_{a.timeframe.lower()}"
    open(stem + ".md", "w", encoding="utf-8").write(md)
    json.dump(res, open(stem + ".json", "w"), indent=1, default=float)
    print("\n" + md)
    print(f"\nwrote {stem}.md and {stem}.json")


if __name__ == "__main__":
    main()
