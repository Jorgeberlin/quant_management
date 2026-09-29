import numpy as np
import pandas as pd

from .returns import to_returns, cagr
from .volatility import downside_deviation
from .drawdown import max_drawdown
from .utils import TRADING_DAYS, validate_periods_per_year

# asumiendo que la rf es constante, esto podmeos cambiarlo en el futuro!
def sharpe_ratio(
    prices: pd.Series | pd.DataFrame,
    periods_per_year: int = TRADING_DAYS,
    risk_free_rate: float = 0,
    method: str = "simple",
) -> float | pd.Series:

    validate_periods_per_year(periods_per_year)

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
    periods_per_year: int = TRADING_DAYS,
    risk_free_rate: float = 0,
    method: str = "simple",
) -> float | pd.Series:

    validate_periods_per_year(periods_per_year)

    returns = to_returns(prices, method=method)

    annualized_return = returns.mean() * periods_per_year
    excess_return = annualized_return - risk_free_rate

    downside = downside_deviation(
        prices,
        periods_per_year=periods_per_year,
        mar=risk_free_rate / periods_per_year,
        method=method,
    )

    return excess_return / downside

def calmar_ratio(
    prices: pd.Series | pd.DataFrame,
    periods_per_year: int = TRADING_DAYS,
) -> float | pd.Series:
    """CAGR / |máximo drawdown|. NaN si no ha habido drawdown."""
    annualized_return = cagr(prices, periods_per_year=periods_per_year)
    drawdown = abs(max_drawdown(prices))

    if isinstance(drawdown, pd.Series):
        return annualized_return / drawdown.replace(0, np.nan)

    return annualized_return / drawdown if drawdown > 0 else np.nan

def tracking_error(
    prices: pd.Series | pd.DataFrame,
    benchmark_prices: pd.Series,
    periods_per_year: int = TRADING_DAYS,
    method: str = "simple",
) -> float | pd.Series:
    """Tracking error anualizado: volatilidad de los retornos activos
    (cartera - benchmark).

    Mide cuánto se separa la cartera de su índice de referencia. Si
    ``prices`` es un DataFrame, se calcula para cada columna contra el
    mismo benchmark. Las fechas se alinean por índice.
    """
    validate_periods_per_year(periods_per_year)

    returns = to_returns(prices, method=method)
    benchmark_returns = to_returns(benchmark_prices, method=method)

    active_returns = returns.sub(benchmark_returns, axis=0)

    return active_returns.std(ddof=1) * np.sqrt(periods_per_year)