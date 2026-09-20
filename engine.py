    # ── The two plotted events: Pine ta.crossover / ta.crossunder against ±thr ──
    buy_cond  = ((hist > thr) & (hist.shift(1) <= thr.shift(1)) & (hist > hist.shift(1))).fillna(False).to_numpy(dtype=bool) & valid
    sell_cond = ((hist < -thr) & (hist.shift(1) >= -thr.shift(1)) & (hist < hist.shift(1))).fillna(False).to_numpy(dtype=bool) & valid
    df['buy_cond']  = buy_cond
    df['sell_cond'] = sell_cond
