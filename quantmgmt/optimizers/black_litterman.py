
import numpy as np
import pandas as pd


def black_litterman(
    pi: np.ndarray,
    sigma: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    tau: float,
    confidence: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:

    """
    Calcula las rentabilidades esperadas posteriores de Black-Litterman.

    Si se proporciona confidence, utiliza una calibración de confianza
    de tipo Idzorek simplificada.

    Si no se proporciona, utiliza el método proporcional de He-Litterman.
    """

    pi = np.asarray(pi).reshape(-1)
    Q = np.asarray(Q).reshape(-1)

    n = len(pi)
    k = len(Q)

    if sigma.shape != (n, n):
        raise ValueError("Dimensiones incorrectas de sigma.")

    if P.shape != (k, n):
        raise ValueError("Dimensiones incorrectas de P.")

    if tau <= 0:
        raise ValueError("tau debe ser positivo.")

    # Incertidumbre previa del mercado
    cov_previa = tau * sigma

    # Construcción de Omega
    var_views = np.diag(P @ cov_previa @ P.T)

    if confidence is None:
        omega = np.diag(var_views)
    else:
        confidence = np.asarray(confidence).reshape(-1)

        if len(confidence) != k or np.any((confidence <= 0) | (confidence >= 1)):
            raise ValueError("Las confianzas deben estar entre 0 y 1.")

        omega = np.diag(((1 - confidence) / confidence) * var_views)

    # Matrices de precisión
    precision_previa = np.linalg.solve(cov_previa, np.eye(n))
    precision_views = np.linalg.solve(omega, np.eye(k))

    # Incertidumbre posterior
    precision_posterior = precision_previa + P.T @ precision_views @ P
    cov_posterior = np.linalg.solve(precision_posterior, np.eye(n))

    # Rentabilidades esperadas posteriores
    mu_bl = cov_posterior @ (precision_previa @ pi + P.T @ precision_views @ Q)

    return mu_bl, cov_posterior

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