import numpy as np
import pandas as pd
import pytest

from quantmgmt.data import excess_returns, get_french_dataset, monthly_returns


# ------------------------------------------------------------ monthly_returns


def make_monthly_prices(n=5, start="2024-01-01"):
    dates = pd.date_range(start, periods=n, freq="MS")  # día 1, como yfinance
    return pd.DataFrame({"A": 100 * 1.01 ** np.arange(n)}, index=dates)


def test_monthly_returns_period_index():
    r = monthly_returns(make_monthly_prices(), today="2030-01-15")

    assert isinstance(r.index, pd.PeriodIndex)
    assert str(r.index[0]) == "2024-02"
    assert r["A"].tolist() == pytest.approx([0.01] * 4)


def test_monthly_returns_drops_incomplete_month():
    # Última barra = mayo 2024; si hoy es 15 de mayo, ese mes está a medias
    prices = make_monthly_prices()

    complete = monthly_returns(prices, today="2024-06-02")
    partial = monthly_returns(prices, today="2024-05-15")

    assert str(complete.index[-1]) == "2024-05"
    assert str(partial.index[-1]) == "2024-04"


def test_excess_returns_aligns_months():
    r = monthly_returns(make_monthly_prices(), today="2030-01-01")
    rf = pd.Series([0.002, 0.002], index=pd.period_range("2024-03", periods=2, freq="M"))

    ex = excess_returns(r, rf)

    assert [str(m) for m in ex.index] == ["2024-03", "2024-04"]
    assert ex["A"].tolist() == pytest.approx([0.008, 0.008])


# ----------------------------------------------------------------- French


def fake_datareader(name, source, start=None):
    """Imita pandas_datareader: dict con [0] mensual, en %, columnas con espacios."""
    index = pd.period_range("2010-01", periods=3, freq="M")
    df = pd.DataFrame({"Mkt-RF": [1.0, -2.0, 3.0], "Mom   ": [0.5, 0.5, 0.5],
                       "RF": [0.01, 0.01, 0.01]}, index=index)
    return {0: df, "DESCR": "fake"}


def test_french_units_columns_and_cache(tmp_path, monkeypatch):
    calls = []

    def spy(*args, **kwargs):
        calls.append(args)
        return fake_datareader(*args, **kwargs)

    monkeypatch.setattr("quantmgmt.data.french.web.DataReader", spy)

    first = get_french_dataset("ff3", cache_dir=tmp_path)
    second = get_french_dataset("ff3", cache_dir=tmp_path)

    assert len(calls) == 1                          # la segunda sale de caché
    assert "Mom" in first.columns                   # espacios eliminados
    assert first["Mkt-RF"].tolist() == pytest.approx([0.01, -0.02, 0.03])  # % -> tanto por uno
    assert isinstance(second.index, pd.PeriodIndex)
    pd.testing.assert_frame_equal(first, second)


def test_french_invalid_name():
    with pytest.raises(ValueError):
        get_french_dataset("ff7")