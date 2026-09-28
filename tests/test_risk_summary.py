import numpy as np
import pandas as pd

from quantmgmt.utils import format_risk_summary, generate_risk_summary


def make_prices() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0005, 0.01, (500, 3))
    dates = pd.bdate_range("2020-01-01", periods=500)
    return pd.DataFrame(100 * np.cumprod(1 + returns, axis=0),
                        index=dates, columns=["A", "B", "C"])


def test_summary_one_row_per_asset_no_nan():
    prices = make_prices()

    summary = generate_risk_summary(prices)

    assert list(summary.index) == ["A", "B", "C"]
    # recovery_time puede ser NaN legítimamente (pico no recuperado)
    assert not summary.drop(columns="Recovery Time").isna().any().any()


def test_summary_benchmark_adds_tracking_error():
    prices = make_prices()

    summary = generate_risk_summary(prices, benchmark_prices=prices["A"])

    assert summary.loc["A", "Tracking Error"] == 0


def test_summary_risk_free_lowers_sharpe():
    prices = make_prices()

    base = generate_risk_summary(prices)
    with_rf = generate_risk_summary(prices, risk_free_rate=0.03)

    assert (with_rf["Sharpe Ratio"] < base["Sharpe Ratio"]).all()


def test_format_does_not_fail():
    format_risk_summary(generate_risk_summary(make_prices())).to_html()