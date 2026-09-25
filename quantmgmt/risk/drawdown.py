import pandas as pd


def drawdown_series(
    prices: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:

    running_max = prices.cummax()

    return prices / running_max - 1

def max_drawdown(
    prices: pd.Series | pd.DataFrame,
) -> float | pd.Series:

    drawdowns = drawdown_series(prices)

    return drawdowns.min()

def time_under_water(
    prices: pd.Series,
) -> pd.Series:

    drawdowns = drawdown_series(prices)

    underwater = drawdowns < 0

    groups = (~underwater).cumsum()

    return underwater.groupby(groups).cumsum()