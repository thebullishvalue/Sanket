"""Fresh HOURLY bars for the v11 fresh tickers (the last 730 days - all yfinance serves at 1h).

    python3 studies/v12_sets/fetch_fresh_1h.py <cache dir>
"""
import os, sys, glob, time
import pandas as pd, yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = sys.argv[1]; os.makedirs(CACHE, exist_ok=True)
tick = pd.read_csv(os.path.join(HERE, "..", "v11_lab", "fresh_tickers.csv"))
todo = [(m, t) for m, t in tick.itertuples(index=False) if not os.path.exists(os.path.join(CACHE, f"{m}__{t}.csv"))]
for i in range(0, len(todo), 40):
    chunk = todo[i:i + 40]
    raw = None
    for attempt in range(4):
        try:
            raw = yf.download([t for _, t in chunk], period="730d", interval="1h", auto_adjust=True,
                              progress=False, threads=True, group_by="ticker")
            break
        except Exception as e:
            print("retry", type(e).__name__); time.sleep(2 ** attempt * 5)
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
            d.to_csv(os.path.join(CACHE, f"{m}__{t}.csv"))
    print(f"{min(i + 40, len(todo))}/{len(todo)}", flush=True)
print("cached:", len(glob.glob(f"{CACHE}/*.csv")))
