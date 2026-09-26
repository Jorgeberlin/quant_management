# data/yahoo_finance.py

from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf


class YahooFinanceData:
    """
    Downloader for historical market data from Yahoo Finance.
    """

    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(
        self,
        ticker: str,
        start: str | datetime,
        end: str | datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Download historical market data for a ticker.

        Parameters
        ----------
        ticker : str
            Yahoo Finance ticker, e.g. "AAPL".
        start : str or datetime
            Start date.
        end : str or datetime
            End date.
        interval : str
            Data frequency. Examples: "1d", "1h", "1wk".

        Returns
        -------
        pd.DataFrame
            Historical market data.
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