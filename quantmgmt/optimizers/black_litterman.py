
import numpy as np


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
