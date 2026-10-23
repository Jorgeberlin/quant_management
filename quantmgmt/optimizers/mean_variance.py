"""Optimización media-varianza (Markowitz) con restricciones realistas.

Convenciones
------------
* ``mu`` y ``cov`` están **anualizados** (salen de ``estimate_inputs``).
* Los pesos se devuelven como ``pd.Series`` indexada por activo y suman 1.
* Todas las carteras se resuelven con ``scipy.optimize.minimize`` (SLSQP).
* Las restricciones se pasan con un objeto ``Constraints``; sin él, se usa
  long-only sin límite de peso.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# --------------------------------------------------------------------- inputs


def estimate_inputs(
    returns: pd.DataFrame,
    periods_per_year: int = 12,
) -> tuple[pd.Series, pd.DataFrame]:
    """Rentabilidad esperada y covarianza muestrales, anualizadas.

    ``mu = media * periods_per_year`` y ``cov = cov_muestral * periods_per_year``.

    Exige datos completos: con NaN, pandas calcula la covarianza par a par
    (cada par con sus propios meses) y la matriz resultante puede no ser
    semidefinida positiva, lo que rompe la optimización. Decide antes qué
    hacer con los activos de histórico corto (recortar la ventana o
    excluirlos).
    """
    if returns.isna().any().any():
        incomplete = returns.columns[returns.isna().any()].tolist()
        raise ValueError(
            f"Hay NaN en {incomplete}. Recorta la ventana o excluye esos activos."
        )

    mu = returns.mean() * periods_per_year
    cov = returns.cov() * periods_per_year

    return mu, cov


# ---------------------------------------------------------------- restricciones


@dataclass
class Constraints:
    """Restricciones de la cartera (los pesos siempre suman 1).

    long_only : bool
        Prohíbe posiciones cortas (peso >= 0).
    max_weight : float, opcional
        Peso máximo por activo (p. ej. 0.20).
    """

    long_only: bool = True
    max_weight: float | None = None

    def bounds(self, n: int) -> list[tuple[float | None, float | None]]:
        lower = 0.0 if self.long_only else None
        upper = self.max_weight
        if self.max_weight is not None and self.max_weight * n < 1:
            raise ValueError(
                f"Infactible: {n} activos x {self.max_weight:.0%} no llegan al 100 %."
            )
        return [(lower, upper)] * n

    def scipy_constraints(self) -> list[dict]:
        return [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]


# ------------------------------------------------------------- utilidades


def portfolio_return(weights, mu) -> float:
    return float(np.asarray(weights) @ np.asarray(mu))


def portfolio_volatility(weights, cov) -> float:
    w = np.asarray(weights)
    return float(np.sqrt(w @ np.asarray(cov) @ w))


def portfolio_sharpe(weights, mu, cov, risk_free_rate: float = 0.0) -> float:
    return (portfolio_return(weights, mu) - risk_free_rate) / portfolio_volatility(weights, cov)


def _solve(objective, n, constraints: Constraints, extra=(), x0=None) -> np.ndarray:
    """Ejecuta SLSQP y comprueba que ha convergido."""
    x0 = np.full(n, 1.0 / n) if x0 is None else x0
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=constraints.bounds(n),
        constraints=constraints.scipy_constraints() + list(extra),
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"La optimización no ha convergido: {result.message}")
    return result.x


# -------------------------------------------------------------- optimizadores


def min_variance(
    cov: pd.DataFrame,
    constraints: Constraints | None = None,
) -> pd.Series:
    """Cartera de mínima varianza: ``min w' Σ w``."""
    constraints = constraints or Constraints()
    sigma = cov.to_numpy()

    w = _solve(lambda w: w @ sigma @ w, len(cov), constraints)

    return pd.Series(w, index=cov.index, name="min_variance")


def max_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = 0.0,
    constraints: Constraints | None = None,
) -> pd.Series:
    """Cartera de máximo Sharpe (tangente): ``max (w'μ - rf) / sqrt(w' Σ w)``.

    Si ``mu`` ya está en exceso sobre el tipo libre de riesgo, deja
    ``risk_free_rate = 0``.
    """
    constraints = constraints or Constraints()
    m, sigma = mu.to_numpy(), cov.to_numpy()

    def neg_sharpe(w):
        return -(w @ m - risk_free_rate) / np.sqrt(w @ sigma @ w)

    w = _solve(neg_sharpe, len(mu), constraints)

    return pd.Series(w, index=mu.index, name="max_sharpe")