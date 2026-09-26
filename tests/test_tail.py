import numpy as np
import pandas as pd
import pytest

from quantmgmt.risk import (
    cvar_historical,
    cvar_parametric,
    kurtosis,
    skewness,
    var_historical,
    var_parametric,
)


def prices_from_returns(returns) -> pd.Series:
    """Precios que empiezan en 100 y generan exactamente esos retornos."""
    returns = np.asarray(returns, dtype=float)
    return pd.Series(100 * np.cumprod(np.concatenate([[1.0], 1 + returns])))


def test_var_and_cvar_historical_known_values():
    # Percentil 25 de [-3%, -1%, 1%, 3%, 5%] = -1% -> VaR 1 %
    # Días <= -1%: -3% y -1% -> media -2% -> CVaR 2 %
    prices = prices_from_returns([-0.03, -0.01, 0.01, 0.03, 0.05])

    assert var_historical(prices, confidence=0.75) == pytest.approx(0.01)
    assert cvar_historical(prices, confidence=0.75) == pytest.approx(0.02)


def test_var_gaussian_known_value():
    # +1% / -1% alternos: media 0 -> VaR95 = 1.6449 * sigma
    returns = [0.01, -0.01] * 50
    prices = prices_from_returns(returns)
    sigma = np.std(returns, ddof=1)

    assert var_parametric(prices) == pytest.approx(1.6448536 * sigma, rel=1e-4)


def test_cvar_gaussian_known_value():
    # Con media 0: CVaR95 = sigma * phi(1.645) / 0.05 = 2.0627 * sigma
    returns = [0.01, -0.01] * 50
    prices = prices_from_returns(returns)
    sigma = np.std(returns, ddof=1)

    assert cvar_parametric(prices) == pytest.approx(2.0627128 * sigma, rel=1e-4)


def test_cvar_at_least_var(prices):
    assert cvar_historical(prices) >= var_historical(prices)
    assert cvar_parametric(prices) >= var_parametric(prices)


def test_cornish_fisher_close_to_gaussian_for_normal_returns():
    rng = np.random.default_rng(0)
    prices = prices_from_returns(rng.normal(0, 0.01, 20_000))

    gauss = var_parametric(prices, distribution="gaussian")
    cf = var_parametric(prices, distribution="cornish_fisher")

    assert cf == pytest.approx(gauss, rel=0.02)


def test_fat_tails_detected():
    # t-Student con 3 g.l.: colas gordas -> exceso de curtosis > 0 y
    # VaR99 de Cornish-Fisher por encima del gaussiano
    rng = np.random.default_rng(0)
    prices = prices_from_returns(0.005 * rng.standard_t(3, 20_000))

    assert kurtosis(prices) > 1
    assert var_parametric(prices, 0.99, "cornish_fisher") > var_parametric(prices, 0.99)


def test_skewness_symmetric_is_zero():
    prices = prices_from_returns([0.01, -0.01] * 50)

    assert skewness(prices) == pytest.approx(0.0, abs=1e-10)


def test_invalid_arguments(prices):
    with pytest.raises(ValueError):
        var_historical(prices, confidence=1.5)
    with pytest.raises(ValueError):
        var_parametric(prices, distribution="student")


def test_tail_metrics_dataframe(prices):
    df = pd.DataFrame({"A": prices, "B": prices * 1.05})

    for metric in (var_historical, cvar_historical, var_parametric,
                   cvar_parametric, skewness, kurtosis):
        result = metric(df)
        assert isinstance(result, pd.Series)
        assert list(result.index) == ["A", "B"]