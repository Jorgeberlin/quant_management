import numpy as np
import pandas as pd
import pytest


from quantmgmt.risk import to_returns, cumulative_returns, cagr, annualized_volatility, downside_deviation

def test_to_returns_simple():
    prices = pd.Series([100, 110, 99])

    result = to_returns(prices, method="simple")

    expected = pd.Series([np.nan, 0.10, -0.10])

    pd.testing.assert_series_equal(result, expected)


def test_to_returns_log():
    prices = pd.Series([100, 110, 99])

    result = to_returns(prices, method="log")

    expected = pd.Series([
        np.nan,
        np.log(1.10),
        np.log(0.90),
    ])

    pd.testing.assert_series_equal(result, expected)


def test_to_returns_dataframe():
    prices = pd.DataFrame({
        "Asset_A": [100, 110, 99],
        "Asset_B": [200, 220, 242],
    })

    result = to_returns(prices)

    expected = pd.DataFrame({
        "Asset_A": [np.nan, 0.10, -0.10],
        "Asset_B": [np.nan, 0.10, 0.10],
    })

    pd.testing.assert_frame_equal(result, expected)


def test_to_returns_invalid_method():
    prices = pd.Series([100, 110, 99])

    with pytest.raises(ValueError):
        to_returns(prices, method="invalid")


def test_cumulative_returns_simple():
    prices = pd.Series([100, 110, 99])

    result = cumulative_returns(prices, method="simple")

    expected = 99 / 100 - 1

    assert result == pytest.approx(expected)

def test_cumulative_returns_log():
    prices = pd.Series([100, 110, 99])

    result = cumulative_returns(prices, method="log")

    expected = np.log(99 / 100)

    assert result == pytest.approx(expected)

def test_cagr():
    prices = pd.Series([100, 110, 99])
    result = cagr(prices)

    assert result < 0

def test_annualized_volatility():
    prices = pd.Series([100, 110, 105, 115, 112])

    result = annualized_volatility(
        prices,
        period="daily"
    )

    returns = prices.pct_change()
    expected = returns.std(ddof=1) * np.sqrt(252)

    assert result == pytest.approx(expected)


def test_annualized_volatility_dataframe():
    prices = pd.DataFrame({
        "Asset_A": [100, 110, 105, 115, 112],
        "Asset_B": [200, 210, 220, 215, 225],
    })

    result = annualized_volatility(
        prices,
        period="daily"
    )

    expected = prices.pct_change().std(ddof=1) * np.sqrt(252)

    pd.testing.assert_series_equal(result, expected)

def test_annualized_volatility_periods_per_year():
    prices = pd.Series([100, 110, 105, 115, 112])

    daily_vol = annualized_volatility(
        prices,
        period="daily"
    )

    monthly_vol = annualized_volatility(
        prices,
        period="monthly"
    )

    assert monthly_vol == pytest.approx(
        daily_vol * np.sqrt(12 / 252)
    )

def test_downside_deviation():
    prices = pd.Series([100, 110, 99, 108, 102])

    result = downside_deviation(
        prices,
        period="daily"
    )

    returns = prices.pct_change()
    downside_returns = returns.clip(upper=0)

    expected = np.sqrt(
        (downside_returns ** 2).mean()
    ) * np.sqrt(252)

    assert result == pytest.approx(expected)