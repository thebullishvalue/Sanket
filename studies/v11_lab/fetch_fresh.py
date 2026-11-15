"""Fetch the confirmation universe (CONFIRM.md): S&P 500 + Nifty 200 today, minus the research names.

    python3 studies/v11_lab/fetch_fresh.py <Indicator/research> <cache dir>

Writes <cache>/<market>__<ticker>.csv (daily OHLCV, adjusted) and studies/v11_lab/fresh_tickers.csv.
"""
import io, os, sys, glob, time
import pandas as pd, requests, yfinance as yf

RESEARCH, CACHE = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(CACHE, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

seen = {os.path.basename(f)[:-4] for f in glob.glob(f"{RESEARCH}/data/*/*.csv")}
sp = pd.read_html(io.StringIO(requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                                           headers=UA, timeout=30).text))[0]
us = [s.replace(".", "-") for s in sp["Symbol"].astype(str)]
nf = pd.read_csv(io.StringIO(requests.get("https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv",
                                          headers=UA, timeout=30).text))
ns = [f"{s}.NS" for s in nf["Symbol"].astype(str)]
tick = [("us", t) for t in us if t not in seen] + [("nse", t) for t in ns if t not in seen]
pd.DataFrame(tick, columns=["market", "ticker"]).to_csv(os.path.join(HERE, "fresh_tickers.csv"), index=False)
print(f"{len(tick)} fresh tickers ({sum(m == 'us' for m, _ in tick)} US, {sum(m == 'nse' for m, _ in tick)} NSE); "
      f"{len(us) + len(ns) - len(tick)} already in the research set left out")

todo = [(m, t) for m, t in tick if not os.path.exists(os.path.join(CACHE, f"{m}__{t}.csv"))]
for i in range(0, len(todo), 40):
    chunk = todo[i:i + 40]
    for attempt in range(4):
        try:
            raw = yf.download([t for _, t in chunk], start="2004-01-01", end="2026-11-15", auto_adjust=True,
                              progress=False, threads=True, group_by="ticker")
            break
        except Exception as e:
            print("retry", type(e).__name__); time.sleep(2 ** attempt * 5)
    for m, t in chunk:
        try:
            d = raw[t].dropna(how="all")
        except KeyError:
            continue
        d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        if len(d) > 1500:
            d.to_csv(os.path.join(CACHE, f"{m}__{t}.csv"))
    print(f"{min(i + 40, len(todo))}/{len(todo)}", flush=True)
print("cached:", len(glob.glob(f"{CACHE}/*.csv")))
