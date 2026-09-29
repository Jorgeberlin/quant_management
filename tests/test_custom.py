import numpy as np
import pandas as pd
import pytest

from quantmgmt.risk import (
    annualized_volatility,
    sharpe_ratio,
    tail_adjusted_sharpe,
    tail_adjusted_volatility,
    tail_excess_ratio,
)


def prices_from_returns(returns) -> pd.Series:
    returns = np.asarray(returns, dtype=float)
    return pd.Series(100 * np.cumprod(np.concatenate([[1.0], 1 + returns])))


@pytest.fixture
def normal_prices():
    return prices_from_returns(np.random.default_rng(0).normal(0.0003, 0.01, 50_000))


@pytest.fixture
def fat_tail_prices():
    # t-Student 3 g.l. escalada: misma escala aproximada, cola mucho más gorda
    return prices_from_returns(0.006 * np.random.default_rng(0).standard_t(3, 50_000))


def test_gaussian_returns_ratio_close_to_one(normal_prices):
    assert tail_excess_ratio(normal_prices) == pytest.approx(1.0, abs=0.03)


def test_gaussian_returns_tav_equals_volatility(normal_prices):
    tav = tail_adjusted_volatility(normal_prices)
    vol = annualized_volatility(normal_prices)

    assert tav == pytest.approx(vol, rel=0.03)


def test_fat_tails_penalized(fat_tail_prices):
    assert tail_excess_ratio(fat_tail_prices) > 1.15
    assert tail_adjusted_volatility(fat_tail_prices) > annualized_volatility(fat_tail_prices)


def test_tav_never_below_volatility(normal_prices, fat_tail_prices):
    for prices in (normal_prices, fat_tail_prices):
        assert tail_adjusted_volatility(prices) >= annualized_volatility(prices)


def test_homogeneity():
    # Escalar los retornos por k escala la TAV por k y deja tau igual
    r = 0.006 * np.random.default_rng(1).standard_t(4, 20_000)
    base, levered = prices_from_returns(r), prices_from_returns(2 * r)

    assert tail_excess_ratio(levered) == pytest.approx(tail_excess_ratio(base), rel=1e-9)
    assert tail_adjusted_volatility(levered) == pytest.approx(
        2 * tail_adjusted_volatility(base), rel=1e-9
    )


def test_tail_adjusted_sharpe_below_sharpe(fat_tail_prices):
    # Mismo activo con deriva positiva para que el Sharpe sea > 0
    returns = fat_tail_prices.pct_change().dropna() + 0.0005
    shifted = prices_from_returns(returns)

    assert tail_adjusted_sharpe(shifted) < sharpe_ratio(shifted)


def test_dataframe(normal_prices, fat_tail_prices):
    df = pd.DataFrame({"normal": normal_prices, "fat": fat_tail_prices})

    tav = tail_adjusted_volatility(df)

    assert isinstance(tav, pd.Series)
    assert tav["fat"] / annualized_volatility(df["fat"]) > tav["normal"] / annualized_volatility(df["normal"])