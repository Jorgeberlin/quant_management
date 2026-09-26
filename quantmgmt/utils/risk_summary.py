import pandas as pd

from quantmgmt.risk.returns import annualized_return
from quantmgmt.risk.volatility import annualized_volatility
from quantmgmt.risk.ratios import (
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
)
from quantmgmt.risk.drawdown import (
    max_drawdown,
    max_time_under_water,
)


def generate_risk_summary(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate a summary of the main risk and performance metrics.

    Parameters
    ----------
    prices : pd.DataFrame
        Asset prices. Each column represents an asset.

    Returns
    -------
    pd.DataFrame
        Risk and performance metrics for each asset.
    """

    summary = pd.DataFrame({
        "Annualized Return": annualized_return(prices),
        "Annualized Volatility": annualized_volatility(prices),
        "Sharpe Ratio": sharpe_ratio(prices),
        "Sortino Ratio": sortino_ratio(prices),
        "Calmar Ratio": calmar_ratio(prices),
        "Maximum Drawdown": max_drawdown(prices),
        "Max Time Under Water": max_time_under_water(prices),
    })

    return summary