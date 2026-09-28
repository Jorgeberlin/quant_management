import numpy as np
import pandas as pd

from quantmgmt.data import YahooFinanceData


def fake_yf_download(tickers, **kwargs):
    """Imita la salida de yf.download: columnas MultiIndex (campo, ticker)."""
    dates = pd.bdate_range("2020-01-01", periods=5)
    fields = ["Adj Close", "Close"]
    columns = pd.MultiIndex.from_product([fields, tickers], names=["Price", "Ticker"])
    values = np.arange(len(dates) * len(columns), dtype=float).reshape(len(dates), -1) + 1
    data = pd.DataFrame(values, index=dates, columns=columns)
    data.iloc[0, 0] = np.nan  # primer ticker empieza un día más tarde
    return data


def test_get_prices_uses_adj_close_and_cache(tmp_path, monkeypatch):
    calls = []

    def spy(tickers, **kwargs):
        calls.append(tickers)
        return fake_yf_download(tickers, **kwargs)

    monkeypatch.setattr("quantmgmt.data.yfinance_loader.yf.download", spy)
    loader = YahooFinanceData(output_dir=tmp_path)

    first = loader.get_prices(["MSFT", "AAPL"], "2020-01-01", "2020-01-10")
    second = loader.get_prices(["AAPL", "MSFT"], "2020-01-01", "2020-01-10")

    # Una sola llamada a Yahoo: la segunda sale de la caché
    assert len(calls) == 1
    assert list(first.columns) == ["AAPL", "MSFT"]
    # Toma "Adj Close", no "Close"
    expected = fake_yf_download(["AAPL", "MSFT"])["Adj Close"]
    assert first["MSFT"].tolist() == expected["MSFT"].tolist()
    # La caché devuelve lo mismo (incluido el NaN inicial)
    pd.testing.assert_frame_equal(first, second, check_freq=False)