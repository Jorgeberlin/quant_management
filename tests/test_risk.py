import numpy as np
import pandas as pd
import pytest
from quantmgmt.risk import to_returns, cumulative_returns, cagr, annualized_return,  annualized_volatility, downside_deviation, rolling_volatility, drawdown_series, max_drawdown, time_under_water, max_time_under_water

# TODO: GENERAR SERIES PARA TESTEAR Y PONERLAS EN CONFTEST, DE MOMENTO SE HACE CON SERIES DUMMY 
# PERO MEJOR HACERLO TODO HOMOGENEO DESDE CONFTEST.
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
        periods_per_year=252
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
        periods_per_year=252
    )

    expected = prices.pct_change().std(ddof=1) * np.sqrt(252)

    pd.testing.assert_series_equal(result, expected)

def test_annualized_volatility_periods_per_year():
    prices = pd.Series([100, 110, 105, 115, 112])

    daily_vol = annualized_volatility(
        prices,
        periods_per_year=252
    )

    monthly_vol = annualized_volatility(
        prices,
        periods_per_year=12
    )

    assert monthly_vol == pytest.approx(
        daily_vol * np.sqrt(12 / 252)
    )

def test_downside_deviation():
    prices = pd.Series([100, 110, 99, 108, 102])

    result = downside_deviation(
        prices,
        periods_per_year=252
    )

    returns = prices.pct_change()
    downside_returns = returns.clip(upper=0)

    expected = np.sqrt(
        (downside_returns ** 2).mean()
    ) * np.sqrt(252)

    assert result == pytest.approx(expected)


def test_rolling_volatility(prices):
    result = rolling_volatility(
        prices,
        window=21,
        periods_per_year=252
    )

    returns = prices.pct_change()

    expected = (
        returns
        .rolling(window=21)
        .std(ddof=1)
        * np.sqrt(252)
    )

    pd.testing.assert_series_equal(result, expected)


def test_drawdown_series(prices):
    result = drawdown_series(prices)

    running_max = prices.cummax()
    expected = prices / running_max - 1

    pd.testing.assert_series_equal(result, expected)


def test_drawdown_series_dataframe(prices):
    prices_df = pd.DataFrame({
        "Asset_A": prices,
        "Asset_B": prices * 1.05,
    })

    result = drawdown_series(prices_df)

    running_max = prices_df.cummax()
    expected = prices_df / running_max - 1

    pd.testing.assert_frame_equal(result, expected)


def test_max_drawdown(prices):
    result = max_drawdown(prices)

    drawdowns = prices / prices.cummax() - 1
    expected = drawdowns.min()

    assert result == pytest.approx(expected)


def test_max_drawdown_dataframe(prices):
    prices_df = pd.DataFrame({
        "Asset_A": prices,
        "Asset_B": prices * 1.05,
    })

    result = max_drawdown(prices_df)

    drawdowns = prices_df / prices_df.cummax() - 1
    expected = drawdowns.min()

    pd.testing.assert_series_equal(result, expected)


def test_public_api():
    import quantmgmt.risk as risk

    for name in risk.__all__:
        assert hasattr(risk, name), f"{name} está en __all__ pero no existe"


def test_cagr_doubling_one_year_with_dates():
    # 253 precios = 252 retornos = 1 año; de 100 a 200 -> CAGR 100 %
    dates = pd.bdate_range("2020-01-01", periods=253)
    prices = pd.Series(np.linspace(100, 200, 253), index=dates)

    assert cagr(prices) == pytest.approx(1.0)


def test_cagr_asset_starting_later():
    # B empieza un periodo después: su CAGR se calcula con sus propios datos
    a = np.linspace(100, 200, 253)
    df = pd.DataFrame({"A": a, "B": np.r_[np.nan, a[:-1]]})

    result = cagr(df)

    assert result["A"] == pytest.approx(1.0)
    assert not np.isnan(result["B"])


def test_time_under_water_known_path():
    prices = pd.Series([100.0, 90.0, 95.0, 100.0, 80.0, 120.0])

    assert time_under_water(prices).tolist() == [0, 1, 2, 0, 1, 0]


def test_time_under_water_dataframe():
    prices = [100.0, 90.0, 95.0, 100.0, 80.0, 120.0]
    df = pd.DataFrame({"A": prices, "B": prices})

    result = time_under_water(df)

    assert isinstance(result, pd.DataFrame)
    assert result["B"].tolist() == [0, 1, 2, 0, 1, 0]


def test_max_time_under_water_ignores_leading_nan():
    # B tiene una racha de 2 periodos bajo el agua aunque empiece con NaN
    df = pd.DataFrame({
        "A": [100.0, 90.0, 95.0, 100.0, 80.0, 120.0],
        "B": [np.nan, 100.0, 90.0, 95.0, 100.0, 80.0],
    })

    result = max_time_under_water(df)

    assert result["A"] == 2
    assert result["B"] == 2