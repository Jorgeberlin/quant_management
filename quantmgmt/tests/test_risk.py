import numpy as np
import pandas as pd
import pytest

from quantmgmt.risk import to_returns


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