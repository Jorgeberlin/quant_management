"""Métricas de cola: VaR, CVaR, asimetría y curtosis.


* Reciben precios (como el resto del módulo) y los convierten a retornos.
* VaR y CVaR se expresan como **pérdida positiva**: VaR = 0.02 significa
  "perder más de un 2 % solo ocurre un (1 - confidence) de las veces".
* El horizonte es el de los datos: con precios diarios, VaR diario.
  No se anualizan (escalar el VaR con sqrt(t) solo vale bajo normalidad).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from .returns import to_returns


def _check_confidence(confidence: float) -> None:
    if not 0 < confidence < 1:
        raise ValueError("confidence debe estar entre 0 y 1")


def var_historical(
    prices: pd.Series | pd.DataFrame,
    confidence: float = 0.95,
    method: str = "simple",
) -> float | pd.Series:
    """VaR histórico: menos el percentil (1 - confidence) de los retornos.

    No asume ninguna distribución; usa las pérdidas observadas.
    """
    _check_confidence(confidence)
    returns = to_returns(prices, method=method)

    return -returns.quantile(1 - confidence)


def cvar_historical(
    prices: pd.Series | pd.DataFrame,
    confidence: float = 0.95,
    method: str = "simple",
) -> float | pd.Series:
    """CVaR (Expected Shortfall) histórico: pérdida media en los días
    iguales o peores que el VaR.
    """
    _check_confidence(confidence)
    returns = to_returns(prices, method=method)

    threshold = returns.quantile(1 - confidence)
    tail = returns[returns <= threshold]

    return -tail.mean()


def skewness(
    prices: pd.Series | pd.DataFrame,
    method: str = "simple",
) -> float | pd.Series:
    """Asimetría de los retornos (corregida por sesgo muestral).

    Negativa: las pérdidas grandes son más frecuentes que las ganancias grandes.
    """
    returns = to_returns(prices, method=method)

    return returns.skew()


def kurtosis(
    prices: pd.Series | pd.DataFrame,
    method: str = "simple",
) -> float | pd.Series:
    """**Exceso** de curtosis de los retornos (normal = 0, corregida por sesgo).

    Positiva: colas más gordas que la normal, eventos extremos más probables.
    """
    returns = to_returns(prices, method=method)

    return returns.kurt()


def var_parametric(
    prices: pd.Series | pd.DataFrame,
    confidence: float = 0.95,
    distribution: str = "gaussian",
    method: str = "simple",
) -> float | pd.Series:
    """VaR paramétrico: ``-(mu + z * sigma)``.

    Parámetros
    ----------
    distribution :
        ``"gaussian"`` -> z es el cuantil de la normal.
        ``"cornish_fisher"`` -> z se corrige con la asimetría (S) y el exceso
        de curtosis (K) observados:
        ``z_cf = z + (z²-1)S/6 + (z³-3z)K/24 - (2z³-5z)S²/36``
    """
    _check_confidence(confidence)
    returns = to_returns(prices, method=method)

    mu = returns.mean()
    sigma = returns.std(ddof=1)
    z = norm.ppf(1 - confidence)

    if distribution == "gaussian":
        z_adj = z
    elif distribution == "cornish_fisher":
        s = returns.skew()
        k = returns.kurt()
        z_adj = (
            z
            + (z**2 - 1) * s / 6
            + (z**3 - 3 * z) * k / 24
            - (2 * z**3 - 5 * z) * s**2 / 36
        )
    else:
        raise ValueError("distribution debe ser 'gaussian' o 'cornish_fisher'")

    return -(mu + z_adj * sigma)


def cvar_parametric(
    prices: pd.Series | pd.DataFrame,
    confidence: float = 0.95,
    method: str = "simple",
) -> float | pd.Series:
    """CVaR gaussiano: ``-(mu - sigma * phi(z) / (1 - confidence))``,
    con phi la densidad de la normal estándar.
    """
    _check_confidence(confidence)
    returns = to_returns(prices, method=method)

    mu = returns.mean()
    sigma = returns.std(ddof=1)
    z = norm.ppf(1 - confidence)

    return -(mu - sigma * norm.pdf(z) / (1 - confidence))