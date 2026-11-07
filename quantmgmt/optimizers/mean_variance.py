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
    prev_weights : pd.Series, opcional
        Cartera actual, para limitar la rotación respecto a ella.
    max_turnover : float, opcional
        Rotación máxima ``sum |w - w_prev|`` (p. ej. 0.30 = mover como mucho
        un 30 % de la cartera sumando compras y ventas). Requiere ``prev_weights``.
    """

    long_only: bool = True
    max_weight: float | None = None
    prev_weights: pd.Series | None = None
    max_turnover: float | None = None

    def __post_init__(self):
        if (self.max_turnover is None) != (self.prev_weights is None):
            raise ValueError("max_turnover y prev_weights van juntos: indica los dos o ninguno.")

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


def effective_n(weights) -> float:
    """Número efectivo de activos: ``1 / sum(w²)``.

    Mide la concentración: N si la cartera es equiponderada entre N activos,
    1 si todo está en un solo activo.
    """
    w = np.asarray(weights)
    return float(1.0 / np.sum(w**2))


def _solve(objective, index, constraints: Constraints, extra=(), x0=None) -> np.ndarray:
    """Ejecuta SLSQP y comprueba que ha convergido.

    ``objective`` y las restricciones de ``extra`` reciben solo los pesos w.

    Rotación: ``sum |w - w_prev| <= L`` no es derivable (valor absoluto) y
    SLSQP la maneja mal. Se reformula con variables auxiliares u >= 0:

        u_i >= w_i - w_prev_i,   u_i >= -(w_i - w_prev_i),   sum(u) <= L

    En el óptimo u_i = |w_i - w_prev_i| y todas las restricciones son lineales.
    """
    n = len(index)
    options = {"ftol": 1e-12, "maxiter": 1000}
    cons = constraints.scipy_constraints() + list(extra)
    bounds = constraints.bounds(n)

    if constraints.max_turnover is None:
        x0 = np.full(n, 1.0 / n) if x0 is None else x0
        result = minimize(objective, x0, method="SLSQP", bounds=bounds,
                          constraints=cons, options=options)
    else:
        w_prev = constraints.prev_weights.reindex(index).fillna(0.0).to_numpy()
        limit = constraints.max_turnover

        def lift(f):
            return lambda x: f(x[:n])

        cons = [{"type": c["type"], "fun": lift(c["fun"])} for c in cons] + [
            {"type": "ineq", "fun": lambda x: x[n:] - (x[:n] - w_prev)},
            {"type": "ineq", "fun": lambda x: x[n:] + (x[:n] - w_prev)},
            {"type": "ineq", "fun": lambda x: limit - np.sum(x[n:])},
        ]
        # Se parte de la cartera actual: rotación 0, siempre factible
        x0 = np.concatenate([w_prev, np.zeros(n)])
        result = minimize(lift(objective), x0, method="SLSQP",
                          bounds=bounds + [(0.0, None)] * n,
                          constraints=cons, options=options)

    if not result.success:
        raise RuntimeError(f"La optimización no ha convergido: {result.message}")

    return result.x[:n]


# -------------------------------------------------------------- optimizadores


def min_variance(
    cov: pd.DataFrame,
    constraints: Constraints | None = None,
) -> pd.Series:
    """Cartera de mínima varianza: ``min w' Σ w``."""
    constraints = constraints or Constraints()
    sigma = cov.to_numpy()

    w = _solve(lambda w: w @ sigma @ w, cov.index, constraints)

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

    w = _solve(neg_sharpe, mu.index, constraints)

    return pd.Series(w, index=mu.index, name="max_sharpe")


def target_volatility(
    mu: pd.Series,
    cov: pd.DataFrame,
    target_vol: float,
    constraints: Constraints | None = None,
) -> pd.Series:
    """Cartera de máxima rentabilidad con volatilidad <= ``target_vol`` (anual)."""
    constraints = constraints or Constraints()
    m, sigma = mu.to_numpy(), cov.to_numpy()

    min_vol = portfolio_volatility(min_variance(cov, constraints), cov)
    if target_vol < min_vol - 1e-9:
        raise ValueError(
            f"Infactible: la volatilidad mínima alcanzable es {min_vol:.2%}."
        )

    vol_limit = {"type": "ineq", "fun": lambda w: target_vol**2 - w @ sigma @ w}
    w = _solve(lambda w: -(w @ m), mu.index, constraints, extra=[vol_limit])

    return pd.Series(w, index=mu.index, name=f"target_vol_{target_vol:.0%}")


def efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_points: int = 50,
    constraints: Constraints | None = None,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Frontera eficiente: cartera de mínima varianza para cada rentabilidad objetivo.

    Las rentabilidades objetivo van de la cartera de mínima varianza a la
    máxima alcanzable con las restricciones (con cortos permitidos, la
    rentabilidad no tiene techo y se usa la del mejor activo).

    Devuelve una fila por punto con ``return``, ``volatility``, ``sharpe``,
    ``effective_n`` (concentración) y el peso de cada activo.
    """
    constraints = constraints or Constraints()
    m, sigma = mu.to_numpy(), cov.to_numpy()

    w_min = min_variance(cov, constraints).to_numpy()
    r_min = portfolio_return(w_min, m)

    if constraints.long_only:
        r_max = portfolio_return(_solve(lambda w: -(w @ m), mu.index, constraints), m)
    else:
        r_max = float(m.max())

    rows = []
    x0 = w_min
    for target in np.linspace(r_min, r_max, n_points):
        on_target = {"type": "eq", "fun": lambda w, t=target: w @ m - t}
        w = _solve(lambda w: w @ sigma @ w, mu.index, constraints,
                   extra=[on_target], x0=x0)
        x0 = w  # arrancar desde el punto anterior acelera y estabiliza
        rows.append({
            "return": portfolio_return(w, m),
            "volatility": portfolio_volatility(w, sigma),
            "sharpe": portfolio_sharpe(w, m, sigma, risk_free_rate),
            "effective_n": effective_n(w),
            **dict(zip(mu.index, w)),
        })

    return pd.DataFrame(rows)