"""Métrica propia: volatilidad ajustada por cola (Tail-Adjusted Volatility, TAV).

Idea
----
La volatilidad resume el riesgo con un solo número, pero supone
implícitamente que la cola izquierda es la de una normal. Dos activos con
la misma volatilidad pueden tener pérdidas extremas muy distintas.

La TAV parte de la volatilidad y la **penaliza en la medida en que la cola
izquierda real es peor que la que predice esa volatilidad**:

    tau = ES_histórico(r - mu) / ES_gaussiano(sigma)        (tail excess ratio)
    TAV = sigma_anual * max(1, tau)

* ``tau`` compara la pérdida media en el peor (1 - confidence) de los
  días observada con la que tendría una normal de la misma volatilidad.
  Se calcula sobre retornos centrados, así mide **solo la forma de la
  cola**, no el nivel de rentabilidad.
* Si la cola es gaussiana (tau ~ 1), TAV = volatilidad.
  Si es más gorda (tau > 1), TAV > volatilidad.
* El suelo en 1 evita premiar colas más finas que la normal, que con
  muestras finitas suelen ser ruido de estimación.

Propiedades
-----------
* Está en **unidades de volatilidad anual**: se lee igual que sigma y
  sustituye a sigma en cualquier sitio donde se use (ratio tipo Sharpe,
  optimización media-riesgo en P2).
* Es **homogénea**: si se escalan los retornos por k, TAV se escala por k
  y tau no cambia. Por eso sirve para comparar carteras de distinto
  apalancamiento o NAVs de distinta escala.
* No depende de las demás carteras de la comparación.

Nivel de confianza (0.975)
--------------------------
Con colas gordas moderadas, al 95 % la cola todavía se parece a la normal
y tau apenas se mueve; al 99 % se mueve más, pero con 10 años diarios solo
hay ~25 observaciones en la cola y la estimación es ruidosa. Simulando 10
años de retornos t-Student(4) frente a normales, el 97,5 % es el nivel con
mejor relación señal/ruido (desviación de tau respecto a 1 entre su error
de estimación: ~2,8 al 95 %, ~3,8 al 97,5 %, ~3,6 al 99 %). Coincide con el
nivel de Expected Shortfall de Basilea (FRTB).

Limitaciones
------------
* Hereda el ruido del CVaR histórico: con 10 años diarios, tau de un
  activo normal oscila en torno a 1 +/- 0.02.
* Solo mira la cola izquierda de los retornos de un periodo; no captura
  pérdidas lentas y prolongadas (para eso están drawdown y time under water).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from .returns import annualized_return, to_returns
from .utils import TRADING_DAYS, validate_periods_per_year
from .volatility import annualized_volatility


def tail_excess_ratio(
    prices: pd.Series | pd.DataFrame,
    confidence: float = 0.975,
    method: str = "simple",
) -> float | pd.Series:
    """Cociente entre el CVaR histórico y el gaussiano, sobre retornos centrados.

    ~1: cola como la normal. >1: cola más gorda de lo que dice la volatilidad.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence debe estar entre 0 y 1")

    returns = to_returns(prices, method=method)
    centered = returns - returns.mean()
    sigma = returns.std(ddof=1)

    threshold = centered.quantile(1 - confidence)
    es_historical = -centered[centered <= threshold].mean()

    z = norm.ppf(1 - confidence)
    es_gaussian = sigma * norm.pdf(z) / (1 - confidence)

    return es_historical / es_gaussian


def tail_adjusted_volatility(
    prices: pd.Series | pd.DataFrame,
    periods_per_year: int = TRADING_DAYS,
    confidence: float = 0.975,
    method: str = "simple",
) -> float | pd.Series:
    """Volatilidad anual penalizada por exceso de cola: ``sigma * max(1, tau)``."""
    validate_periods_per_year(periods_per_year)

    sigma = annualized_volatility(prices, periods_per_year=periods_per_year, method=method)
    tau = tail_excess_ratio(prices, confidence=confidence, method=method)

    return sigma * np.maximum(tau, 1.0)


def tail_adjusted_sharpe(
    prices: pd.Series | pd.DataFrame,
    periods_per_year: int = TRADING_DAYS,
    risk_free_rate: float = 0.0,
    confidence: float = 0.975,
    method: str = "simple",
) -> float | pd.Series:
    """Sharpe con la TAV en el denominador: ``(mu_anual - rf) / TAV``.

    Siempre <= Sharpe (en valor absoluto), con igualdad si la cola es gaussiana.
    La diferencia entre ambos es el "riesgo de cola escondido".
    """
    excess = annualized_return(prices, periods_per_year=periods_per_year, method=method) - risk_free_rate
    tav = tail_adjusted_volatility(
        prices, periods_per_year=periods_per_year, confidence=confidence, method=method
    )

    return excess / tav