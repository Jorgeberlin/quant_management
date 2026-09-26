import numpy as np
import pandas as pd
from typing import Literal

from .returns import to_returns, cagr
from .volatility import downside_deviation
from .drawdown import max_drawdown

# asumiendo que la rf es constante, esto podmeos cambiarlo en el futuro!
def sharpe_ratio(   
    prices: pd.Series | pd.DataFrame,
    period: Literal["daily", "monthly"] = "daily",
    risk_free_rate: float = 0,
    method: str = "simple",
) -> float | pd.Series:

    if period == "daily":
        periods_per_year = 252
    elif period == "monthly":
        periods_per_year = 12
    else:
        raise ValueError("period must be either 'daily' or 'monthly'")

    returns = to_returns(prices, method=method)

    risk_free_period = risk_free_rate / periods_per_year

    excess_returns = returns - risk_free_period

    return (
        excess_returns.mean()
        / returns.std(ddof=1)
        * np.sqrt(periods_per_year)
    )

def sortino_ratio(
    prices: pd.Series | pd.DataFrame,
    period: Literal["daily", "monthly"] = "daily",
    risk_free_rate: float = 0,
    method: str = "simple",
) -> float | pd.Series:

    if period == "daily":
        periods_per_year = 252
    elif period == "monthly":
        periods_per_year = 12
    else:
        raise ValueError("period must be either 'daily' or 'monthly'")

    returns = to_returns(prices, method=method)

    annualized_return = returns.mean() * periods_per_year
    excess_return = annualized_return - risk_free_rate

    downside = downside_deviation(
        prices,
        period=period,
        mar=risk_free_rate / periods_per_year,
        method=method,
    )

    return excess_return / downside

def calmar_ratio(
    prices: pd.Series | pd.DataFrame,
) -> float | pd.Series:

    annualized_return = cagr(prices)
    drawdown = max_drawdown(prices)

    return annualized_return / abs(drawdown)

def tracking_error(
    prices: pd.Series | pd.DataFrame,
    benchmark_prices: pd.Series,
    period: Literal["daily", "monthly"] = "daily",
    method: str = "simple",
) -> float | pd.Series:
    """Tracking error anualizado: volatilidad de los retornos activos
    (cartera - benchmark).

    Mide cuánto se separa la cartera de su índice de referencia. Si
    ``prices`` es un DataFrame, se calcula para cada columna contra el
    mismo benchmark. Las fechas se alinean por índice.
    """
    if period == "daily":
        periods_per_year = 252
    elif period == "monthly":
        periods_per_year = 12
    else:
        raise ValueError("period must be either 'daily' or 'monthly'")

    returns = to_returns(prices, method=method)
    benchmark_returns = to_returns(benchmark_prices, method=method)

    active_returns = returns.sub(benchmark_returns, axis=0)

    return active_returns.std(ddof=1) * np.sqrt(periods_per_year)