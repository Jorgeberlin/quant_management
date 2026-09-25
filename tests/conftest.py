import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def prices():
    np.random.seed(42)

    dates = pd.date_range(
        start="2025-01-01",
        periods=100,
        freq="B"
    )

    returns = np.random.normal(
        loc=0.0005,
        scale=0.015,
        size=100
    )

    return pd.Series(
        100 * np.cumprod(1 + returns),
        index=dates,
        name="Asset"
    )