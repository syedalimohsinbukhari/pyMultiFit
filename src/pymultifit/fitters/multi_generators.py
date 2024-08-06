"""Created on Jul 18 00:35:26 2024"""

from typing import List, Tuple

import numpy as np
from scipy.stats import skewnorm

from ..distributions import GaussianDistribution as GDist, LogNormalDistribution


# TODO:
#   Reimplement these generators using the newly formed distributions.

def generate_multi_gaussian_data(x: np.ndarray, params: List[Tuple[float, float, float]],
                                 noise_level: float = 0.0, normalized: bool = False) -> np.ndarray:
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
    normalized: bool
        If True, the function produces normalized data (Integration[PDF] < 1). Defaults to False.

    Returns
    -------
    np.ndarray
        Y values of the multi-Gaussian data with added noise.
    """
    y = np.zeros_like(x)
    for amp, mean, std in params:
        pdf_ = GDist(mean, std, normalized).pdf(x)
        y += pdf_ if normalized else amp * pdf_
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
    for amp, shape, mean, std in params:
        y += amp * skewnorm.pdf(x, shape, loc=mean, scale=std)
    if noise_level > 0:
        y += noise_level * np.random.normal(size=x.size)
    return y


def generate_multi_log_normal_data(x: np.ndarray, params: List[Tuple[float, float, float]],
                                   noise_level: float = 0.0, normalized=False) -> np.ndarray:
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
    normalized: bool, optional
        Whether to get a normalized version of the distribution. Defaults to False.

    Returns
    -------
    np.ndarray
        Y values of the multi-Gaussian data with added noise.
    """
    y = np.zeros_like(x)
    for amp, mean, std in params:
        pdf_ = LogNormalDistribution(mean, std).pdf(x)
        y += pdf_ if normalized else amp * pdf_
    if noise_level > 0:
        y += noise_level * np.random.normal(size=x.size)
    return y
