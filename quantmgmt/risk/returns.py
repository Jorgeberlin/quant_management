import numpy as np
import pandas as pd


def to_returns(prices: pd.Series | pd.DataFrame,method: str = "simple") -> pd.Series | pd.DataFrame:

    if method == "simple":
        return prices.pct_change()

    if method == "log":
        return np.log(prices / prices.shift(1))

    raise ValueError("method must be either 'simple' or 'log'")