"""Created on Jul 18 00:35:26 2024"""

from typing import List, Tuple

import numpy as np


def generate_multi_gaussian_data(x: np.ndarray, params: List[Tuple[float, float, float]],
                                 noise_level: float = 0.0) -> np.ndarray:
    """
    Generate multi-Gaussian data with optional noise.

    Parameters
    ----------
    x : np.ndarray
        X values.
    params : List[Tuple[float, float, float]]
        List of tuples containing the parameters for each Gaussian (amplitude, mean, standard deviation).
    noise_level : float, optional
        Standard deviation of the Gaussian noise to be added to the data, by default 0.0.

    Returns
    -------
    np.ndarray
        Y values of the multi-Gaussian data with added noise.
    """
    y = np.zeros_like(x)
    for A, mu, sigma in params:
        y += A * np.exp(-(x - mu)**2 / (2 * sigma**2))
    if noise_level > 0:
        y += noise_level * np.random.normal(size=x.size)
    return y
