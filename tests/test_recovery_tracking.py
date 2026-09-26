import numpy as np
import pandas as pd
import pytest

from quantmgmt.risk import recovery_time, tracking_error


def prices_from_returns(returns, index=None) -> pd.Series:
    """Precios que empiezan en 100 y generan exactamente esos retornos."""
    returns = np.asarray(returns, dtype=float)
    return pd.Series(100 * np.cumprod(np.concatenate([[1.0], 1 + returns])), index=index)


# ------------------------------------------------------------ recovery_time


def test_recovery_time_known_path():
    # Pico 120 (t=1), valle 90 (t=2), vuelve a 120 en t=4 -> 2 periodos
    prices = pd.Series([100.0, 120.0, 90.0, 100.0, 120.0, 130.0])

    assert recovery_time(prices) == 2


def test_recovery_time_with_dates():
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    prices = pd.Series([100.0, 120.0, 90.0, 100.0, 120.0, 130.0], index=dates)

    assert recovery_time(prices) == 2


def test_recovery_time_not_recovered_is_nan():
    prices = pd.Series([100.0, 120.0, 90.0, 100.0])

    assert np.isnan(recovery_time(prices))


def test_recovery_time_no_drawdown_is_zero():
    prices = pd.Series([100.0, 101.0, 102.0])

    assert recovery_time(prices) == 0


def test_recovery_time_dataframe():
    df = pd.DataFrame({
        "A": [100.0, 120.0, 90.0, 100.0, 120.0],
        "B": [100.0, 120.0, 90.0, 100.0, 110.0],
    })

    result = recovery_time(df)

    assert result["A"] == 2
    assert np.isnan(result["B"])


# ----------------------------------------------------------- tracking_error


def test_tracking_error_identical_is_zero(prices):
    assert tracking_error(prices, prices, period="daily") == pytest.approx(0.0)


def test_tracking_error_constant_outperformance_is_zero():
    # Ganar siempre un 0,1 % más que el índice no es "separarse" de él:
    # el retorno activo es constante -> volatilidad 0
    bench_returns = np.random.default_rng(1).normal(0, 0.01, 100)
    bench = prices_from_returns(bench_returns)
    portfolio = prices_from_returns(bench_returns + 0.001)

    assert tracking_error(portfolio, bench, period="daily") == pytest.approx(0.0, abs=1e-12)


def test_tracking_error_known_value():
    # Retorno activo +1% / -1% alterno -> TE = std * sqrt(252)
    bench = prices_from_returns([0.0] * 100)
    active = [0.01, -0.01] * 50
    portfolio = prices_from_returns(active)

    expected = np.std(active, ddof=1) * np.sqrt(252)

    assert tracking_error(portfolio, bench, period="daily") == pytest.approx(expected)


def test_tracking_error_dataframe(prices):
    df = pd.DataFrame({"A": prices, "B": prices * 1.05})

    result = tracking_error(df, prices, period="daily")

    assert isinstance(result, pd.Series)
    assert result.tolist() == pytest.approx([0.0, 0.0])