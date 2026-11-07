"""Modelos multifactoriales: CAPM, Fama-French 3, Carhart 4 y Fama-French 5.

Convenciones
------------
* Los retornos de los activos deben estar **en exceso** sobre el tipo libre
  de riesgo (``quantmgmt.data.excess_returns``), igual que los factores.
* Datos mensuales: la alfa se anualiza multiplicando por 12.
* Los t-estadísticos usan errores estándar de Newey-West (HAC), robustos a
  heterocedasticidad y autocorrelación, habituales en retornos mensuales.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

MODELS = {
    "CAPM": ("ff3", ["Mkt-RF"]),
    "FF3": ("ff3", ["Mkt-RF", "SMB", "HML"]),
    "Carhart": ("ff3", ["Mkt-RF", "SMB", "HML", "Mom"]),
    "FF5": ("ff5", ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]),
}


def factor_matrix(factors: dict[str, pd.DataFrame], model: str) -> pd.DataFrame:
    """Factores del modelo a partir de ``quantmgmt.data.get_factors()``.

    CAPM, FF3 y Carhart usan el fichero FF3 (más momentum en Carhart); FF5
    usa el suyo, porque su SMB se construye de otra forma.
    """
    if model not in MODELS:
        raise ValueError(f"model debe ser uno de {list(MODELS)}")

    dataset, columns = MODELS[model]
    base = factors[dataset]
    if "Mom" in columns:
        base = base.join(factors["mom"], how="inner")

    return base[columns]


def _newey_west_lags(n_obs: int) -> int:
    """Regla habitual de retardos: floor(4 (T/100)^(2/9))."""
    return int(np.floor(4 * (n_obs / 100) ** (2 / 9)))


def _regress_one(y: pd.Series, X: pd.DataFrame, periods_per_year: int) -> pd.Series:
    data = pd.concat([y, X], axis=1, join="inner").dropna()
    y, X = data.iloc[:, 0], sm.add_constant(data.iloc[:, 1:])

    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": _newey_west_lags(len(y))})

    out = {
        "alpha (ann.)": fit.params["const"] * periods_per_year,
        "t(alpha)": fit.tvalues["const"],
    }
    for factor in X.columns.drop("const"):
        out[f"beta {factor}"] = fit.params[factor]
        out[f"t {factor}"] = fit.tvalues[factor]
    out["adj R2"] = fit.rsquared_adj
    out["n_obs"] = int(fit.nobs)

    return pd.Series(out)


def run_factor_regression(
    returns: pd.Series | pd.DataFrame,
    factors: dict[str, pd.DataFrame],
    model: str,
    periods_per_year: int = 12,
) -> pd.Series | pd.DataFrame:
    """Regresión ``r_i - rf = alpha + sum(beta_k * F_k) + e``.

    Devuelve alfa anualizada, betas, t-estadísticos (Newey-West), R²
    ajustado y número de observaciones. Con un DataFrame, una fila por
    activo. Solo se usan los meses comunes a retornos y factores.
    """
    X = factor_matrix(factors, model)

    if isinstance(returns, pd.Series):
        return _regress_one(returns, X, periods_per_year)

    return pd.DataFrame({
        col: _regress_one(returns[col], X, periods_per_year) for col in returns.columns
    }).T


def compare_models(
    returns: pd.DataFrame,
    factors: dict[str, pd.DataFrame],
    models: tuple[str, ...] = ("CAPM", "FF3", "Carhart", "FF5"),
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """Tabla comparativa de alfa anualizada, t(alfa) y R² ajustado por modelo.

    Columnas en dos niveles: (modelo, métrica). Una fila por activo.
    """
    tables = {
        model: run_factor_regression(returns, factors, model, periods_per_year)[
            ["alpha (ann.)", "t(alpha)", "adj R2"]
        ]
        for model in models
    }
    return pd.concat(tables, axis=1)


def rolling_betas(
    returns: pd.Series,
    factors: dict[str, pd.DataFrame],
    model: str,
    window: int = 36,
) -> pd.DataFrame:
    """Betas de un activo estimadas en una ventana móvil de ``window`` meses.

    Los primeros ``window - 1`` meses quedan en NaN.
    """
    X = factor_matrix(factors, model)
    data = pd.concat([returns, X], axis=1, join="inner").dropna()

    fit = RollingOLS(data.iloc[:, 0], sm.add_constant(data.iloc[:, 1:]), window=window).fit()

    return fit.params.drop(columns="const")