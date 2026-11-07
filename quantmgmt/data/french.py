"""Factores de Fama-French y momentum (Ken French Data Library).

Los datos originales vienen en **porcentaje** y ya en **exceso sobre el tipo
libre de riesgo** (salvo la columna RF, que es el propio tipo). Aquí se
devuelven en tanto por uno (0.01 = 1 %), con índice mensual ``PeriodIndex``
para alinearlos sin ambigüedad con los retornos de los activos.
"""

from pathlib import Path

import pandas as pd
import pandas_datareader.data as web

from .yfinance_loader import DEFAULT_CACHE_DIR

DATASETS = {
    "ff3": "F-F_Research_Data_Factors",
    "ff5": "F-F_Research_Data_5_Factors_2x3",
    "mom": "F-F_Momentum_Factor",
}


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Columnas sin espacios, de % a tanto por uno e índice mensual."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df.index = pd.PeriodIndex(df.index, freq="M")
    df.index.name = "month"
    return df / 100


def get_french_dataset(
    name: str,
    start: str = "2010-01",
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Un dataset mensual de French (``"ff3"``, ``"ff5"`` o ``"mom"``), con caché en CSV."""
    if name not in DATASETS:
        raise ValueError(f"name debe ser uno de {list(DATASETS)}")

    cache_file = Path(cache_dir) / f"french_{name}_{start}.csv"

    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0)
        df.index = pd.PeriodIndex(df.index, freq="M")
        df.index.name = "month"
        return df

    raw = web.DataReader(DATASETS[name], "famafrench", start=start)[0]  # [0] = mensual
    df = _clean(raw)

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file)

    return df


def get_factors(start: str = "2010-01", use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Los tres datasets que usan los modelos (CAPM, FF3, Carhart y FF5).

    Se devuelven por separado porque **SMB no es el mismo en FF3 y en FF5**:
    French lo construye de forma distinta en cada fichero. CAPM, FF3 y
    Carhart usan el de FF3; FF5 usa el suyo.
    """
    return {name: get_french_dataset(name, start=start, use_cache=use_cache) for name in DATASETS}