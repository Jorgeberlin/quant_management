"""Utilidades comunes del módulo de riesgo.

Convenciones de ``quantmgmt.risk``
----------------------------------
* Las métricas reciben **precios** (``pd.Series`` o ``pd.DataFrame``)
  indexados por fecha y los convierten a retornos con ``to_returns``.
* ``periods_per_year`` indica la frecuencia de los datos: 252 diarios,
  52 semanales, 12 mensuales. Por defecto, 252.
* Los NaN se ignoran, así que los activos con histórico más corto se
  tratan sin limpieza previa.
* Las métricas escalares devuelven un ``float`` si reciben una Series y
  una ``pd.Series`` (un valor por columna) si reciben un DataFrame.
"""

TRADING_DAYS = 252  # días hábiles en un año
WEEKS = 52
MONTHS = 12


def validate_periods_per_year(periods_per_year: int) -> None:
    """Lanza ``ValueError`` si ``periods_per_year`` no es un número positivo."""
    if not isinstance(periods_per_year, (int, float)) or periods_per_year <= 0:
        raise ValueError("periods_per_year debe ser un número positivo")