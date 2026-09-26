"""Utilidades comunes del módulo de riesgo.

Convenciones de ``quantmgmt.risk``
----------------------------------
* Todas las métricas reciben **retornos simples** (``pd.Series`` o
  ``pd.DataFrame``) indexados por fecha. Se obtienen a partir de precios
  con ``to_returns``.
* ``periods_per_year`` indica la frecuencia de los datos: 252 diarios,
  52 semanales, 12 mensuales.
* Los NaN se ignoran (``skipna`` de pandas), así que el NaN inicial que
  genera ``to_returns`` y los activos con histórico más corto se tratan
  sin limpieza previa.
* Las métricas escalares devuelven un ``float`` si reciben una Series y
  una ``pd.Series`` (un valor por columna) si reciben un DataFrame.
"""

import pandas as pd

TRADING_DAYS = 252  # días hábiles en un año
WEEKS = 52
MONTHS = 12


def validate_periods_per_year(periods_per_year: int) -> None:
    """Lanza ``ValueError`` si ``periods_per_year`` no es un número positivo."""
    if not isinstance(periods_per_year, (int, float)) or periods_per_year <= 0:
        raise ValueError("periods_per_year debe ser un número positivo")


def to_wealth(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Curva de capital partiendo de 1: ``W_t = prod_{s<=t} (1 + r_s)``.

    Los retornos NaN se tratan como 0 (el capital no varía ese periodo).
    """
    return (1 + returns.fillna(0)).cumprod()