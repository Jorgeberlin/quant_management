import numpy as np
import pandas as pd

from .returns import to_returns
from .utils import TRADING_DAYS, validate_periods_per_year


def annualized_volatility(
    prices: pd.Series | pd.DataFrame,
    periods_per_year: int = TRADING_DAYS,
    method: str = "simple",
) -> pd.Series | pd.DataFrame:

    validate_periods_per_year(periods_per_year)

    rets = to_returns(prices=prices, method=method)

    return rets.std(ddof=1) * np.sqrt(periods_per_year)

def downside_deviation(
    prices: pd.Series | pd.DataFrame,
    periods_per_year: int = TRADING_DAYS,
    mar: float = 0,
    method: str = "simple",
) -> pd.Series | pd.DataFrame:
    """
    El MAR es el minimum accepted rate de la fórmula.
    """

    validate_periods_per_year(periods_per_year)

    rets = to_returns(prices=prices, method=method)

    downside_returns = (rets - mar).clip(upper=0)

    return np.sqrt((downside_returns ** 2).mean()) * np.sqrt(periods_per_year)

def rolling_volatility(
    prices: pd.Series | pd.DataFrame,
    window: int = 21,
    periods_per_year: int = TRADING_DAYS,
    method: str = "simple",
) -> pd.Series | pd.DataFrame:

    validate_periods_per_year(periods_per_year)

    if window <= 0:
        raise ValueError("window must be greater than 0")

    rets = to_returns(prices=prices, method=method)

    return rets.rolling(window=window).std(ddof=1) * np.sqrt(periods_per_year)