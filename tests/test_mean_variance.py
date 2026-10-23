import numpy as np
import pandas as pd
import pytest

from quantmgmt.optimizers import (
    Constraints,
    effective_n,
    efficient_frontier,
    estimate_inputs,
    max_sharpe,
    min_variance,
    portfolio_sharpe,
    portfolio_volatility,
    target_volatility,
)

ASSETS = ["A", "B", "C"]


@pytest.fixture
def mu():
    return pd.Series([0.08, 0.10, 0.12], index=ASSETS)


@pytest.fixture
def cov():
    vol = np.array([0.10, 0.15, 0.20])
    corr = np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]])
    return pd.DataFrame(np.outer(vol, vol) * corr, index=ASSETS, columns=ASSETS)


UNCONSTRAINED = Constraints(long_only=False)


# --------------------------------------------------------------------- inputs


def test_estimate_inputs_annualizes():
    r = pd.DataFrame({"A": [0.01, 0.03, 0.02], "B": [0.00, 0.02, -0.02]})

    mu, cov = estimate_inputs(r, periods_per_year=12)

    assert mu.tolist() == pytest.approx([0.02 * 12, 0.0])
    assert cov.loc["A", "A"] == pytest.approx(r["A"].var() * 12)


def test_estimate_inputs_rejects_nan():
    r = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [np.nan, 0.01, 0.02]})

    with pytest.raises(ValueError, match="B"):
        estimate_inputs(r)


# --------------------------------------------------------------- min_variance


def test_min_variance_two_uncorrelated_assets():
    # Solución cerrada: w_A = σ_B² / (σ_A² + σ_B²)
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.01]], index=["A", "B"], columns=["A", "B"])

    w = min_variance(cov)

    assert w["A"] == pytest.approx(0.01 / 0.05, abs=1e-6)
    assert w.sum() == pytest.approx(1.0)


def test_min_variance_unconstrained_matches_closed_form(cov):
    # w = Σ⁻¹1 / 1'Σ⁻¹1
    inv = np.linalg.inv(cov.to_numpy())
    expected = inv @ np.ones(3) / (np.ones(3) @ inv @ np.ones(3))

    w = min_variance(cov, UNCONSTRAINED)

    assert w.to_numpy() == pytest.approx(expected, abs=1e-6)


# ----------------------------------------------------------------- max_sharpe


def test_max_sharpe_unconstrained_matches_closed_form(mu, cov):
    # Cartera tangente: w ∝ Σ⁻¹ (μ - rf)
    rf = 0.02
    raw = np.linalg.solve(cov.to_numpy(), mu.to_numpy() - rf)
    expected = raw / raw.sum()

    w = max_sharpe(mu, cov, risk_free_rate=rf, constraints=UNCONSTRAINED)

    assert w.to_numpy() == pytest.approx(expected, abs=1e-5)


def test_max_sharpe_beats_equal_weight(mu, cov):
    w = max_sharpe(mu, cov, risk_free_rate=0.02)
    ew = np.full(3, 1 / 3)

    assert portfolio_sharpe(w, mu, cov, 0.02) >= portfolio_sharpe(ew, mu, cov, 0.02)


# -------------------------------------------------------------- restricciones


def test_long_only_and_max_weight_respected(mu, cov):
    constraints = Constraints(long_only=True, max_weight=0.4)

    for w in (min_variance(cov, constraints), max_sharpe(mu, cov, 0.02, constraints)):
        assert w.sum() == pytest.approx(1.0)
        assert (w >= -1e-9).all()
        assert (w <= 0.4 + 1e-9).all()


def test_infeasible_max_weight_raises(cov):
    with pytest.raises(ValueError, match="Infactible"):
        min_variance(cov, Constraints(max_weight=0.3))  # 3 x 30 % = 90 %

        # ------------------------------------------------------------------- rotación


def test_turnover_zero_keeps_previous_portfolio(mu, cov):
    prev = pd.Series([0.5, 0.3, 0.2], index=ASSETS)
    constraints = Constraints(prev_weights=prev, max_turnover=0.0)

    w = max_sharpe(mu, cov, 0.02, constraints)

    assert w.to_numpy() == pytest.approx(prev.to_numpy(), abs=1e-6)


def test_turnover_limit_respected(mu, cov):
    prev = pd.Series([1 / 3] * 3, index=ASSETS)
    constraints = Constraints(prev_weights=prev, max_turnover=0.10)

    w = max_sharpe(mu, cov, 0.02, constraints)

    assert (w - prev).abs().sum() <= 0.10 + 1e-6
    assert portfolio_sharpe(w, mu, cov, 0.02) >= portfolio_sharpe(prev, mu, cov, 0.02)


def test_large_turnover_equals_no_turnover_limit(mu, cov):
    prev = pd.Series([1 / 3] * 3, index=ASSETS)
    free = max_sharpe(mu, cov, 0.02)
    loose = max_sharpe(mu, cov, 0.02, Constraints(prev_weights=prev, max_turnover=2.0))

    assert loose.to_numpy() == pytest.approx(free.to_numpy(), abs=1e-5)


def test_turnover_needs_prev_weights():
    with pytest.raises(ValueError):
        Constraints(max_turnover=0.2)


# --------------------------------------------------------------- frontera


def test_frontier_matches_analytic_unconstrained(mu, cov):
    # Frontera sin restricciones: σ²(r) = (A r² - 2 B r + C) / D
    inv = np.linalg.inv(cov.to_numpy())
    one, m = np.ones(3), mu.to_numpy()
    A, B, C = one @ inv @ one, one @ inv @ m, m @ inv @ m
    D = A * C - B**2

    frontier = efficient_frontier(mu, cov, n_points=10, constraints=UNCONSTRAINED)
    r = frontier["return"].to_numpy()
    expected_vol = np.sqrt((A * r**2 - 2 * B * r + C) / D)

    assert frontier["volatility"].to_numpy() == pytest.approx(expected_vol, abs=1e-6)


def test_frontier_volatility_increases_with_return(mu, cov):
    frontier = efficient_frontier(mu, cov, n_points=15)

    assert frontier["volatility"].is_monotonic_increasing
    assert frontier[ASSETS].sum(axis=1).to_numpy() == pytest.approx(1.0)


def test_weight_cap_reduces_concentration(mu, cov):
    free = max_sharpe(mu, cov, 0.02)
    capped = max_sharpe(mu, cov, 0.02, Constraints(max_weight=0.4))

    assert effective_n(capped) >= effective_n(free)


def test_effective_n_known_values():
    assert effective_n([0.25] * 4) == pytest.approx(4.0)
    assert effective_n([1.0, 0.0, 0.0]) == pytest.approx(1.0)


# ------------------------------------------------------- target_volatility


def test_target_volatility_hits_target(mu, cov):
    w = target_volatility(mu, cov, target_vol=0.12)

    assert portfolio_volatility(w, cov) == pytest.approx(0.12, abs=1e-5)
    assert w.sum() == pytest.approx(1.0)


def test_target_volatility_infeasible(mu, cov):
    with pytest.raises(ValueError, match="Infactible"):
        target_volatility(mu, cov, target_vol=0.01)