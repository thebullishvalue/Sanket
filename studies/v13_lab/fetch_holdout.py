"""Fetch the v13 confirmation universe (PROTOCOL.md part 2). Downloads only - nothing is computed on it here.

    python3 studies/v13_lab/fetch_holdout.py <Indicator/research> <daily cache> <hourly cache>

S&P MidCap 400 + S&P SmallCap 600 (Wikipedia) + Nifty 500 minus Nifty 200 (niftyindices.com), minus every
name in the research set and the v11 fresh set. Daily 2004-2026; hourly the last 730 days.
"""
import io, os, sys, glob, time
import pandas as pd, requests, yfinance as yf

RESEARCH, DCACHE, HCACHE = sys.argv[1], sys.argv[2], sys.argv[3]
HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
used = {os.path.basename(f)[:-4] for f in glob.glob(f"{RESEARCH}/data/*/*.csv")}
used |= set(pd.read_csv(os.path.join(HERE, "..", "v11_lab", "fresh_tickers.csv")).ticker)


def wiki(url):
    tabs = pd.read_html(io.StringIO(requests.get(url, headers=UA, timeout=30).text))
    t = next(t for t in tabs if any(str(c).lower() in ("symbol", "ticker symbol", "ticker") for c in t.columns))
    col = next(c for c in t.columns if str(c).lower() in ("symbol", "ticker symbol", "ticker"))
    return [str(s).replace(".", "-") for s in t[col]]


us = wiki("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies") + \
     wiki("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies")
n500 = pd.read_csv(io.StringIO(requests.get("https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
                                            headers=UA, timeout=30).text))
n200 = pd.read_csv(io.StringIO(requests.get("https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv",
                                            headers=UA, timeout=30).text))
ns = [f"{s}.NS" for s in sorted(set(n500.Symbol) - set(n200.Symbol))]
tick = [("us", t) for t in dict.fromkeys(us) if t not in used] + [("nse", t) for t in ns if t not in used]
pd.DataFrame(tick, columns=["market", "ticker"]).to_csv(os.path.join(HERE, "holdout_tickers.csv"), index=False)
print(f"{len(tick)} holdout tickers ({sum(m == 'us' for m, _ in tick)} US, {sum(m == 'nse' for m, _ in tick)} NSE)", flush=True)


def fetch(cache, kw, label):
    os.makedirs(cache, exist_ok=True)
    todo = [(m, t) for m, t in tick if not os.path.exists(os.path.join(cache, f"{m}__{t}.csv"))]
    for i in range(0, len(todo), 40):
        chunk, raw = todo[i:i + 40], None
        for attempt in range(4):
            try:
                raw = yf.download([t for _, t in chunk], progress=False, threads=True, group_by="ticker",
                                  auto_adjust=True, **kw)
                break
            except Exception as e:
                print("retry", type(e).__name__, flush=True); time.sleep(2 ** attempt * 5)
        if raw is None:
            continue
        for m, t in chunk:
            try:
                d = raw[t].dropna(how="all")
            except KeyError:
                continue
            d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
            if d.index.tz is not None:
                d.index = d.index.tz_convert(None)
            if len(d) > 1500:
                d.to_csv(os.path.join(cache, f"{m}__{t}.csv"))
        print(f"{label} {min(i + 40, len(todo))}/{len(todo)}", flush=True)
    print(label, "cached:", len(glob.glob(f"{cache}/*.csv")), flush=True)


fetch(DCACHE, dict(start="2004-01-01", end="2026-11-17"), "daily")
fetch(HCACHE, dict(period="730d", interval="1h"), "hourly")
