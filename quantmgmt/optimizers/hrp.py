import numpy as np
import pandas as pd

def distance_correlation(
    matriz_de_correlaciones: np.ndarray | pd.DataFrame) -> np.ndarray:
    matriz_de_correlaciones = np.asarray(matriz_de_correlaciones)
    d_correlation = np.sqrt(0.5 * (1 - matriz_de_correlaciones))

    return d_correlation

    
    