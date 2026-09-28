from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

# Raíz del repo: quantmgmt/data/yfinance_loader.py -> sube dos niveles
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache"


class YahooFinanceData:
    """
    Descarga de datos históricos de Yahoo Finance con caché local en CSV.
    """

    def __init__(self, output_dir: str | Path = DEFAULT_CACHE_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(
        self,
        ticker: str | list[str],
        start: str | datetime,
        end: str | datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Download historical market data for one or several tickers.

        Parameters
        ----------
        ticker : str or list of str
            Yahoo Finance ticker(s), e.g. "AAPL" or ["AAPL", "MSFT"].
        start : str or datetime
            Start date.
        end : str or datetime
            End date.
        interval : str
            Data frequency. Examples: "1d", "1h", "1wk".

        Returns
        -------
        pd.DataFrame
            Historical market data (sin ajustar, incluye "Adj Close").
        """

        data = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )

        if data.empty:
            raise ValueError(
                f"No data found for ticker '{ticker}' "
                f"between {start} and {end}."
            )

        return data

    def get_prices(
        self,
        tickers: list[str],
        start: str,
        end: str,
        interval: str = "1d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Precios ajustados por splits **y dividendos** (``Adj Close``).

        Una columna por ticker, índice de fechas. Es la serie correcta para
        medir rentabilidad total. Los NaN de activos que empiezan más tarde
        se mantienen; las filas sin ningún dato se eliminan.

        Si ``use_cache`` y ya existe el CSV para esa combinación de tickers,
        fechas e intervalo, se lee de disco sin llamar a Yahoo.
        """
        tickers = sorted(tickers)
        cache_file = self.output_dir / (
            f"{'_'.join(tickers)}_{start}_{end}_{interval}.csv"
        )

        if use_cache and cache_file.exists():
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)

        data = self.download(tickers, start=start, end=end, interval=interval)

        prices = data["Adj Close"]
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(tickers[0])

        prices = prices[tickers].dropna(how="all")
        prices.columns.name = None

        prices.to_csv(cache_file)

        return prices

    def save(
        self,
        data: pd.DataFrame,
        ticker: str,
    ) -> Path:
        """
        Save downloaded data as CSV.
        """

        file_path = self.output_dir / f"{ticker}.csv"
        data.to_csv(file_path)

        return file_path

    def download_and_save(
        self,
        ticker: str,
        start: str | datetime,
        end: str | datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Download historical data and save it as CSV.
        """

        data = self.download(
            ticker=ticker,
            start=start,
            end=end,
            interval=interval,
        )

        self.save(data, ticker)

        return data