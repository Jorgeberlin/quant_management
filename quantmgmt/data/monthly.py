"""Utilidades para trabajar con datos mensuales."""

import pandas as pd


def monthly_returns(
    prices: pd.DataFrame,
    today: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Retornos simples mensuales con índice ``PeriodIndex`` (``2010-01``, ...).

    * yfinance etiqueta cada barra mensual con el día 1 del mes; aquí se
      pasa a periodo para alinear con los factores de French.
    * Si la última barra corresponde al mes en curso, está incompleta (precio
      de mitad de mes) y se elimina.
    """
    today = pd.Timestamp.today() if today is None else pd.Timestamp(today)

    returns = prices.pct_change(fill_method=None).iloc[1:]
    returns.index = pd.DatetimeIndex(returns.index).to_period("M")
    returns.index.name = "month"

    if returns.index[-1] >= today.to_period("M"):
        returns = returns.iloc[:-1]

    return returns


def excess_returns(returns: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    """Retornos en exceso sobre el tipo libre de riesgo, solo en los meses comunes."""
    common = returns.index.intersection(rf.index)
    return returns.loc[common].sub(rf.loc[common], axis=0)