import numpy as np
import pandas as pd

# cumulative_returns podría "heredar" de to_returns. pensar esto.

def to_returns(prices: pd.Series | pd.DataFrame,method: str = "simple") -> pd.Series | pd.DataFrame:

    if method == "simple":
        return prices.pct_change()

    if method == "log":
        return np.log(prices / prices.shift(1))

    raise ValueError("method must be either 'simple' or 'log'")

def cumulative_returns(prices: pd.Series | pd.DataFrame,method: str = "simple",) -> pd.Series | pd.DataFrame:

    if method == "simple":
        returns = prices.pct_change()
        return (1 + returns).prod() - 1

    if method == "log":
        returns = np.log(prices / prices.shift(1))
        return returns.iloc[1:].sum()

    raise ValueError("method must be either 'simple' or 'log'")