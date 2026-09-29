import pandas as pd

from quantmgmt.risk import (
    annualized_return,
    annualized_volatility,
    cagr,
    calmar_ratio,
    cvar_historical,
    cvar_parametric,
    downside_deviation,
    kurtosis,
    max_drawdown,
    max_time_under_water,
    recovery_time,
    sharpe_ratio,
    skewness,
    sortino_ratio,
    tail_adjusted_sharpe,
    tail_adjusted_volatility,
    tail_excess_ratio,
    tracking_error,
    var_historical,
    var_parametric,
)
from quantmgmt.risk.utils import TRADING_DAYS


def generate_risk_summary(
    prices: pd.DataFrame,
    periods_per_year: int = TRADING_DAYS,
    risk_free_rate: float = 0.0,
    confidence: float = 0.95,
    benchmark_prices: pd.Series | None = None,
) -> pd.DataFrame:
    """Tabla con todas las métricas de riesgo y rentabilidad, una fila por activo.

    Parámetros
    ----------
    prices : pd.DataFrame
        Precios, una columna por activo, cartera o NAV.
    periods_per_year : int
        Frecuencia de los datos (252 diarios, 52 semanales, 12 mensuales).
    risk_free_rate : float
        Tipo libre de riesgo **anual** para Sharpe y Sortino.
    confidence : float
        Nivel de confianza de VaR y CVaR.
    benchmark_prices : pd.Series, opcional
        Si se pasa, se añade el tracking error frente a él.

    Notas
    -----
    VaR y CVaR están en el horizonte de los datos (diario con precios
    diarios) y como pérdida positiva. Los tiempos están en periodos.
    """
    pct = f"{confidence:.0%}"

    summary = pd.DataFrame({
        # Rentabilidad
        "CAGR": cagr(prices, periods_per_year=periods_per_year),
        "Annualized Return": annualized_return(prices, periods_per_year=periods_per_year),
        # Dispersión
        "Annualized Volatility": annualized_volatility(prices, periods_per_year=periods_per_year),
        "Downside Deviation": downside_deviation(prices, periods_per_year=periods_per_year),
        # Ratios
        "Sharpe Ratio": sharpe_ratio(prices, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate),
        "Sortino Ratio": sortino_ratio(prices, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate),
        "Calmar Ratio": calmar_ratio(prices, periods_per_year=periods_per_year),
        # Drawdown
        "Maximum Drawdown": max_drawdown(prices),
        "Max Time Under Water": max_time_under_water(prices),
        "Recovery Time": recovery_time(prices),
        # Cola
        f"VaR {pct} Historical": var_historical(prices, confidence=confidence),
        f"VaR {pct} Gaussian": var_parametric(prices, confidence=confidence, distribution="gaussian"),
        f"VaR {pct} Cornish-Fisher": var_parametric(prices, confidence=confidence, distribution="cornish_fisher"),
        f"CVaR {pct} Historical": cvar_historical(prices, confidence=confidence),
        f"CVaR {pct} Gaussian": cvar_parametric(prices, confidence=confidence),
        "Skewness": skewness(prices),
        "Excess Kurtosis": kurtosis(prices),
                # Métrica propia (nivel fijo 97,5 %, ver quantmgmt/risk/custom.py)
        "Tail Excess Ratio": tail_excess_ratio(prices),
        "Tail-Adjusted Volatility": tail_adjusted_volatility(prices, periods_per_year=periods_per_year),
        "Tail-Adjusted Sharpe": tail_adjusted_sharpe(prices, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate),
    })

    if benchmark_prices is not None:
        summary["Tracking Error"] = tracking_error(prices, benchmark_prices, periods_per_year=periods_per_year)

    return summary


PERCENT_COLUMNS = (
    "CAGR", "Annualized Return", "Annualized Volatility", "Downside Deviation",
    "Maximum Drawdown", "VaR", "CVaR", "Tracking Error", "Tail-Adjusted Volatility",
)


def format_risk_summary(summary: pd.DataFrame, transpose: bool = False):
    """Estilo para mostrar la tabla en el notebook: porcentajes y 2 decimales.

    Con ``transpose=True`` muestra las métricas en filas y los activos en
    columnas (más legible con muchas métricas). Devuelve un ``Styler``; los
    datos originales no se modifican.
    """
    def _fmt(metric: str) -> str:
        if metric in ("Max Time Under Water", "Recovery Time"):
            return "{:.0f}"
        return "{:.2%}" if metric.startswith(PERCENT_COLUMNS) else "{:.2f}"

    if not transpose:
        formats = {col: _fmt(col) for col in summary.columns}
        return summary.style.format(formats, na_rep="no recuperado")

    styler = summary.T.style
    for metric in summary.columns:
        styler = styler.format(
            _fmt(metric), subset=pd.IndexSlice[[metric], :], na_rep="no recuperado"
        )
    return styler