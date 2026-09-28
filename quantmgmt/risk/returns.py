import numpy as np
import pandas as pd
from typing import Literal
# cumulative_returns podría "heredar" de to_returns. pensar esto. se podría hacer una
# clase para todos los metodos y encapsular todo mejor. de mopmento lo dejo naive.

def to_returns(prices: pd.Series | pd.DataFrame,method: str = "simple") -> pd.Series | pd.DataFrame:

    if method == "simple":
        return prices.pct_change()

    if method == "log":
        return np.log(prices / prices.shift(1))

    raise ValueError("method must be either 'simple' or 'log'")

def cumulative_returns(prices: pd.Series | pd.DataFrame,method: str = "simple") -> pd.Series | pd.DataFrame:

    if method == "simple":
        returns = prices.pct_change()
        return (1 + returns).prod() - 1

    if method == "log":
        returns = np.log(prices / prices.shift(1))
        return returns.iloc[1:].sum()

    raise ValueError("method must be either 'simple' or 'log'")

def cagr(
    prices: pd.Series | pd.DataFrame,
    method: str = "simple",
) -> float | pd.Series:
    """Tasa de crecimiento anual compuesta.

    ``CAGR = prod(1 + r) ** (252 / N) - 1``, con N el número de retornos
    válidos de cada columna. Así un activo que empieza a cotizar más tarde
    (NaN iniciales) se anualiza con sus propios años.
    """
    if method != "simple":
        raise ValueError("method must be 'simple'")

    returns = to_returns(prices)

    n = returns.count()
    growth = (1 + returns).prod()

    return growth ** (252 / n) - 1


    
def annualized_return( prices: pd.Series | pd.DataFrame, 
                      period: Literal["daily", "monthly"] = "daily", 
                      method: str = "simple",) -> float | pd.Series:
    
    if period == "daily":
        periods_per_year = 252
    elif period == "monthly":
        periods_per_year = 12
    else:
        raise ValueError("period must be either 'daily' or 'monthly'")

    returns = to_returns(prices, method=method)

    return returns.mean() * periods_per_year