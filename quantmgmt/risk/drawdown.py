import numpy as np
import pandas as pd


def drawdown_series(
    prices: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """Caída respecto al máximo previo en cada fecha: ``P_t / max(P_s, s<=t) - 1``.

    Siempre <= 0. Los NaN iniciales (activo que empieza más tarde) se mantienen.
    """
    running_max = prices.cummax()

    return prices / running_max - 1


def max_drawdown(
    prices: pd.Series | pd.DataFrame,
) -> float | pd.Series:
    """Peor caída de pico a valle de toda la muestra (número negativo)."""
    drawdowns = drawdown_series(prices)

    return drawdowns.min()


def _time_under_water_1d(prices: pd.Series) -> pd.Series:
    drawdowns = drawdown_series(prices.dropna())

    underwater = drawdowns < 0
    groups = (~underwater).cumsum()

    return underwater.groupby(groups).cumsum()


def time_under_water(
    prices: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """Periodos consecutivos por debajo del máximo previo, en cada fecha.

    Vuelve a 0 al marcar un nuevo máximo. Unidades: periodos (días hábiles
    con datos diarios). Con DataFrame se calcula columna a columna, ignorando
    los NaN de cada activo.
    """
    if isinstance(prices, pd.DataFrame):
        return prices.apply(_time_under_water_1d)

    return _time_under_water_1d(prices)


def max_time_under_water(
    prices: pd.Series | pd.DataFrame,
) -> int | pd.Series:
    """Racha más larga (en periodos) sin recuperar el máximo previo."""
    if isinstance(prices, pd.Series):
        return int(time_under_water(prices).max())

    if isinstance(prices, pd.DataFrame):
        return time_under_water(prices).max().astype(int)

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