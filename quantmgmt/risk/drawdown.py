import pandas as pd
import numpy as np


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

import pandas as pd


import pandas as pd


def max_time_under_water(
    prices: pd.Series | pd.DataFrame,
) -> int | pd.Series:

    def _max_time(series: pd.Series) -> int:
        cumulative = series / series.iloc[0]
        running_max = cumulative.cummax()

        underwater = cumulative < running_max
        groups = (~underwater).cumsum()

        return int(underwater.groupby(groups).cumsum().max())

    if isinstance(prices, pd.Series):
        return _max_time(prices)

    elif isinstance(prices, pd.DataFrame):
        return prices.apply(_max_time)

    else:
        raise TypeError("prices must be a pandas Series or DataFrame")

def _recovery_time_1d(prices: pd.Series) -> float:
    prices = prices.dropna()
    drawdowns = drawdown_series(prices)

    if drawdowns.min() == 0:
        return 0.0

    trough = drawdowns.idxmin()
    peak_value = prices.loc[:trough].max()
    after_trough = prices.loc[trough:]
    recovered = after_trough[after_trough >= peak_value]

    if recovered.empty:
        return np.nan

    return float(prices.index.get_loc(recovered.index[0]) - prices.index.get_loc(trough))


def recovery_time(
    prices: pd.Series | pd.DataFrame,
) -> float | pd.Series:
    """Periodos desde el valle del máximo drawdown hasta recuperar el pico previo.

    * Unidades: periodos (días hábiles con datos diarios), igual que
      ``time_under_water``.
    * NaN si al final de la muestra aún no se ha recuperado el pico.
    * 0 si no ha habido ningún drawdown.
    """
    if isinstance(prices, pd.DataFrame):
        return prices.apply(_recovery_time_1d)

    return _recovery_time_1d(prices)