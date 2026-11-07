import numpy as np
import pandas as pd
import pytest

from quantmgmt.analytics import (
    compare_models,
    factor_matrix,
    rolling_betas,
    run_factor_regression,
)

N = 240  # 20 años mensuales


@pytest.fixture
def factors():
    """Factores sintéticos con la estructura de get_factors()."""
    rng = np.random.default_rng(0)
    idx = pd.period_range("2005-01", periods=N, freq="M")
    ff5 = pd.DataFrame(rng.normal(0.005, 0.03, (N, 5)), index=idx,
                       columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA"])
    ff5["RF"] = 0.002
    ff3 = ff5[["Mkt-RF", "SMB", "HML", "RF"]].copy()
    mom = pd.DataFrame({"Mom": rng.normal(0.004, 0.04, N)}, index=idx)
    return {"ff3": ff3, "ff5": ff5, "mom": mom}


def make_asset(factors, alpha, betas, noise=0.002, seed=1):
    """Retorno en exceso con alfa (mensual) y betas conocidas.

    Ruido pequeño (0,2 % mensual) para que el error de estimación de la alfa
    (~ruido / sqrt(N)) no tape lo que se quiere comprobar.
    """
    rng = np.random.default_rng(seed)
    X = factors["ff3"].join(factors["mom"])
    y = alpha + sum(b * X[f] for f, b in betas.items())
    return y + rng.normal(0, noise, len(y))


# --------------------------------------------------------- factor_matrix


def test_factor_matrix_columns(factors):
    assert list(factor_matrix(factors, "CAPM").columns) == ["Mkt-RF"]
    assert list(factor_matrix(factors, "Carhart").columns) == ["Mkt-RF", "SMB", "HML", "Mom"]
    assert list(factor_matrix(factors, "FF5").columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def test_unknown_model(factors):
    with pytest.raises(ValueError):
        factor_matrix(factors, "APT")


# --------------------------------------------------- run_factor_regression


def test_recovers_known_alpha_and_betas(factors):
    # alfa 0,25 % mensual = 3 % anual
    betas = {"Mkt-RF": 1.2, "SMB": 0.5, "HML": -0.3, "Mom": 0.2}
    y = make_asset(factors, alpha=0.0025, betas=betas)

    res = run_factor_regression(y, factors, "Carhart")

    assert res["alpha (ann.)"] == pytest.approx(0.03, abs=0.005)
    for f, b in betas.items():
        assert res[f"beta {f}"] == pytest.approx(b, abs=0.05)
    assert res["t(alpha)"] > 2          # alfa real -> significativa
    assert res["adj R2"] > 0.9
    assert res["n_obs"] == N


def test_no_alpha_not_significant(factors):
    y = make_asset(factors, alpha=0.0, betas={"Mkt-RF": 1.0})

    res = run_factor_regression(y, factors, "CAPM")

    assert abs(res["t(alpha)"]) < 2


def test_omitted_factor_shows_up_as_alpha(factors):
    # El activo tiene exposición a momentum (que rinde +0,4 %/mes de media).
    # CAPM no lo ve y lo atribuye a alfa; Carhart lo explica.
    y = make_asset(factors, alpha=0.0, betas={"Mkt-RF": 1.0, "Mom": 0.8})

    capm = run_factor_regression(y, factors, "CAPM")
    carhart = run_factor_regression(y, factors, "Carhart")

    assert capm["alpha (ann.)"] > 0.02
    assert abs(carhart["alpha (ann.)"]) < 0.01
    assert carhart["adj R2"] > capm["adj R2"]


def test_dataframe_one_row_per_asset_and_alignment(factors):
    a = make_asset(factors, 0.0, {"Mkt-RF": 1.0}, seed=1)
    b = make_asset(factors, 0.0, {"Mkt-RF": 0.5}, seed=2).iloc[-100:]  # histórico corto
    returns = pd.DataFrame({"A": a, "B": b})

    res = run_factor_regression(returns, factors, "CAPM")

    assert list(res.index) == ["A", "B"]
    assert res.loc["A", "n_obs"] == N
    assert res.loc["B", "n_obs"] == 100


# --------------------------------------------------------- compare_models


def test_compare_models_structure(factors):
    returns = pd.DataFrame({"A": make_asset(factors, 0.001, {"Mkt-RF": 1.0})})

    table = compare_models(returns, factors)

    assert table.columns.get_level_values(0).unique().tolist() == ["CAPM", "FF3", "Carhart", "FF5"]
    assert ("FF5", "adj R2") in table.columns


# ---------------------------------------------------------- rolling_betas


def test_rolling_betas_constant_beta(factors):
    y = make_asset(factors, 0.0, {"Mkt-RF": 0.7}, noise=0.001)

    betas = rolling_betas(y, factors, "CAPM", window=36)

    assert betas["Mkt-RF"].iloc[:35].isna().all()
    assert betas["Mkt-RF"].dropna().to_numpy() == pytest.approx(0.7, abs=0.02)