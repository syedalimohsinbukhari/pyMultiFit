"""Created on Jul 18 00:35:26 2024"""

from typing import List, Tuple

import numpy as np

from ..distributions import (GaussianDistribution as GDist, LaplaceDistribution as lapD, line,
                             LogNormalDistribution as lNorm, SkewedNormalDistribution as skNorm)
from .. import GAUSSIAN, LAPLACE, LINE, LOG_NORMAL

gdA = GDist.with_amplitude
lnA = lNorm.with_amplitude
lpA = lapD.with_amplitude


def generate_multi_gaussian_data(x: np.ndarray, params: List[Tuple[float, float, float]],
                                 noise_level: float = 0.0, normalize: bool = False) -> np.ndarray:
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
    normalize: bool
        If True, the function produces normalized data (Integration[PDF] < 1). Defaults to False.

    Returns
    -------
    np.ndarray
        Y values of the multi-Gaussian data with added noise.
    """
    y = np.zeros_like(x)
    for amp, mean, std in params:
        if normalize:
            y += GDist(mean=mean, standard_deviation=std).pdf(x)
        else:
            y += gdA(amplitude=amp, mean=mean, standard_deviation=std).pdf(x)
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
    for amp, shape, location, scale in params:
        y += skNorm(shape=shape, location=location, scale=scale).pdf(x)
    if noise_level > 0:
        y += noise_level * np.random.normal(size=x.size)
    return y


def generate_multi_log_normal_data(x: np.ndarray, params: List[Tuple[float, float, float]],
                                   noise_level: float = 0.0, normalize: bool = False) -> np.ndarray:
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
    normalize: bool, optional
        Whether to get a normalized version of the distribution. Defaults to False.

    Returns
    -------
    np.ndarray
        Y values of the multi-Gaussian data with added noise.
    """
    y = np.zeros_like(x)
    for amp, mean, std in params:
        if normalize:
            y += lNorm(mean=mean, standard_deviation=std).pdf(x)
        else:
            y += lnA(amplitude=amp, mean=mean, standard_deviation=std).pdf(x)
    if noise_level > 0:
        y += noise_level * np.random.normal(size=x.size)
    return y


def generate_multi_laplace_data(x: np.ndarray, params: List[Tuple[float, float, float]], noise_level: float = 0.0,
                                normalize: bool = False) -> np.ndarray:
    """
    Generate multi-Laplace data with optional noise.

    Parameters
    ----------
    x : np.ndarray
        X values.
    params : List[Tuple[float, float, float]]
        List of tuples containing the parameters for each Laplace (amplitude, mean, diversity).
    noise_level : float
        Standard deviation of the Gaussian noise to be added to the data, by default 0.0.
    normalize: bool
        Whether to get a normalized version of the distribution. Defaults to False.

    Returns
    -------
    np.ndarray
        Y values of the multi-Laplace data with added noise.
    """
    y = np.zeros_like(x)
    for amp, mu, b in params:
        if normalize:
            y += lapD(mean=mu, diversity=b).pdf(x)
        else:
            y += lpA(amplitude=amp, mean=mu, diversity=b).pdf(x)
    if noise_level > 0:
        y += noise_level * np.random.normal(size=x.size)
    return y


def generate_mixed_model_data(x: np.ndarray, params: List[Tuple[float, ...]], model_list, noise_level=0.0):
    y = np.zeros_like(x)
    par_index = 0
    for model in model_list:
        if model == GAUSSIAN:
            y += gdA(*params[par_index]).pdf(x)
        elif model == LOG_NORMAL:
            y += lnA(*params[par_index]).pdf(x)
        elif model == LAPLACE:
            y += lpA(*params[par_index]).pdf(x)
        elif model == LINE:
            y += line(*params[par_index])

        par_index += 1

    if noise_level > 0:
        y += noise_level * np.random.normal(size=x.size)

    return y
