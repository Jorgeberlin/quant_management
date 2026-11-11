import numpy as np
import pandas as pd

def distance_correlation(
    matriz_de_correlaciones: np.ndarray | pd.DataFrame) -> np.ndarray:
    matriz_de_correlaciones = np.asarray(matriz_de_correlaciones)
    d_correlation = np.sqrt(0.5 * (1 - matriz_de_correlaciones))

    return d_correlation

    
def get_cluster_variance(cov_cluster: np.ndarray) -> float:

    variances = np.diag(cov_cluster)
    inv_variances = 1 / variances
    weights = inv_variances / inv_variances.sum()

    return weights @ cov_cluster @ weights


def get_hrp_weights(cov: pd.DataFrame, order: list) -> pd.Series:

    cov = cov.iloc[order, order]
    weights = pd.Series(1.0, index=cov.index)
    clusters = [list(cov.index)]

    #  Método de la bisección aplicado 
    while clusters:

        new_clusters = []

        for cluster in clusters:

            if len(cluster) <= 1:
                continue

            mid = len(cluster) // 2
            cluster_1 = cluster[:mid]
            cluster_2 = cluster[mid:]

            # Extract covariance matrices
            cov_1 = cov.loc[cluster_1, cluster_1].to_numpy()
            cov_2 = cov.loc[cluster_2, cluster_2].to_numpy()

            # Las varianzas de cada cluster
            var_1 = get_cluster_variance(cov_1)
            var_2 = get_cluster_variance(cov_2)

            # Alocacion
            alpha_1 = 1 - var_1 / (var_1 + var_2)
            alpha_2 = 1 - alpha_1

            # Pfolio weights
            weights.loc[cluster_1] *= alpha_1
            weights.loc[cluster_2] *= alpha_2

            # Añadir los "subclusters" tras cad aiteración
            new_clusters.extend([cluster_1, cluster_2])

        clusters = new_clusters

    return weights / weights.sum()


def risk_contribution(weights, cov):
    portfolio_variance = weights.T @ cov @ weights
    portfolio_volatility = np.sqrt(portfolio_variance)
    vector_rc = []
    for activo in range(len(weights)):
        mrc_activo = (cov@weights / portfolio_volatility)[activo]
        rc_activo = (weights[activo] * mrc_activo) / portfolio_volatility
        vector_rc.append(rc_activo)

    return np.array(vector_rc)

def get_risk_parity_naive_weights(cov: pd.DataFrame) -> pd.Series:
    volatilities = np.sqrt(np.diag(cov))
    inv_vol = 1 / volatilities
    weights = inv_vol / inv_vol.sum()

    return pd.Series(weights, index=cov.index)