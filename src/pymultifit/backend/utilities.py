"""Created on Jul 18 00:35:26 2024"""

from typing import List, Tuple

import numpy as np
from scipy.stats import skewnorm


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
    for amp, mu, sigma in params:
        y += amp * np.exp(-(x - mu)**2 / (2 * sigma**2))
    if noise_level > 0:
        y += noise_level * np.random.normal(size=x.size)
    return y


def generate_multi_skewed_normal_data(x: np.ndarray, params: List[Tuple[float, float, float, float]],
                                      noise_level: float = 0.0) -> np.ndarray:
    """
    Generate multi-Skewed Normal data with optional noise.

    Parameters
    ----------
    x : np.ndarray
        X values.
    params : List[Tuple[float, float, float, float]]
        List of tuples containing the parameters for each Skewed Normal (amplitude, shape, location, scale).
    noise_level : float, optional
        Standard deviation of the Gaussian noise to be added to the data, by default 0.0.

    Returns
    -------
    np.ndarray
        Y values of the multi-Skewed Normal data with added noise.
    """
    y = np.zeros_like(x)
    for amp, shape, loc, scale in params:
        y += amp * skewnorm.pdf(x, shape, loc=loc, scale=scale)
    if noise_level > 0:
        y += noise_level * np.random.normal(size=x.size)
    return y


def generate_multi_log_normal_data(x: np.ndarray, params: List[Tuple[float, float, float]],
                                   noise_level: float = 0.0, exact_mean=False) -> np.ndarray:
    """
    Generate multi-log_normal data with optional noise.

    Parameters
    ----------
    x : np.ndarray
        X values.
    params : List[Tuple[float, float, float]]
        List of tuples containing the parameters for each Gaussian (amplitude, mean, standard deviation).
    noise_level : float, optional
        Standard deviation of the Gaussian noise to be added to the data, by default 0.0.
    exact_mean: bool, optional
        Whether to use the exact mean provided or the log-normal mean.

    Returns
    -------
    np.ndarray
        Y values of the multi-Gaussian data with added noise.
    """
    y = np.zeros_like(x)
    for amp, mu, sigma in params:
        if exact_mean:
            mu = np.log(mu) - (sigma**2 / 2)
        y += amp * np.exp(- (np.log(x) - mu)**2 / (2 * sigma**2))
    if noise_level > 0:
        y += noise_level * np.random.normal(size=x.size)
    return y
