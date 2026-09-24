import numpy as np
import pandas as pd
from typing import Literal

from .returns import to_returns

#aquí podemos hacer que el period venga de una estructura superior para que las metricas sean
#todas homogeneas si se quiere.

def annualized_volatility(
    prices: pd.Series | pd.DataFrame,
    period: Literal["daily", "monthly"],
    method: str = "simple",
) -> pd.Series | pd.DataFrame:

    if period == "daily":
        periods_per_year = 252
    elif period == "monthly":
        periods_per_year = 12
    else:
        raise ValueError("period must be either 'daily' or 'monthly'")

    rets = to_returns(prices=prices, method=method)

    return rets.std(ddof=1) * np.sqrt(periods_per_year)