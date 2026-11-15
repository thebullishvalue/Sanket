"""Add the raw and cross-sectional scorers to a harvested dataset (score = Edge Study's drift-free one)."""
import os
import pandas as pd
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def load(ds):
    E = pd.read_pickle(os.path.join(OUT, f"events_{ds}.pkl"))
    xs = pd.read_pickle(os.path.join(OUT, f"xs_{ds}.pkl"))
    key = pd.MultiIndex.from_arrays([pd.DatetimeIndex(E["date"]), E["cls"].astype(str)])
    E["peer"] = xs.reindex(key).to_numpy()
    E["score_raw"] = E["side"] * E["fwd"] / E["_sigma"]
    E["score_xs"] = E["side"] * (E["fwd"] - E["peer"]) / E["_sigma"]
    E["ds"] = ds
    return E
