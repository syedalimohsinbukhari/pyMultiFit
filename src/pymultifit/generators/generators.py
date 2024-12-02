"""Created on Jul 18 00:35:26 2024"""

from typing import List, Tuple, Union

import numpy as np

from .. import distributions as dist, GAUSSIAN, LAPLACE, LINE, LOG_NORMAL, POWERLAW, SKEW_NORMAL

listOfTuples = List[Tuple[float, ...]]
listOfTuplesOrArray = Union[listOfTuples, np.ndarray]


def generate_multi_exponential_data(x: np.ndarray, params: listOfTuplesOrArray,
                                    noise_level: float = 0.0, normalize: bool = False) -> np.ndarray:
    """
    Generate multi-Exponential data with optional noise.

    Parameters
    ----------
    x : np.ndarray
        X values.
    params : List[Tuple[float, ...]]
        List of tuples containing the parameters for each Exponential (amplitude, scale).
    noise_level : float, optional
        Standard deviation of the Exponential noise to be added to the data, by default 0.0.
    normalize: bool
        If True, the function produces normalized data (Integration[PDF] < 1). Defaults to False.

    Returns
    -------
    np.ndarray
        Y values of the multi-Exponential data with added noise.
    """
    y = np.zeros_like(x, dtype=float)
    for pars in params:
        y += dist.ExponentialDistribution(*pars, normalize=normalize).pdf(x)
    if noise_level > 0:
        y += noise_level * np.random.normal(size=x.size)
    return y


def generate_multi_gaussian_data(x: np.ndarray, params: listOfTuplesOrArray,
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
    y = np.zeros_like(x, dtype=float)
    for pars in params:
        y += dist.GaussianDistribution(*pars, normalize=normalize).pdf(x)
    if noise_level > 0:
        y += noise_level * np.random.normal(size=x.size)
    return y


def generate_multi_skewed_normal_data(x: np.ndarray, params: listOfTuplesOrArray,
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
    y = np.zeros_like(x, dtype=float)
    for amp, shape, location, scale in params:
        y += dist.SkewedNormalDistribution(shape=shape, location=location, scale=scale).pdf(x)
    if noise_level > 0:
        y += noise_level * np.random.normal(size=x.size)
    return y


def generate_multi_log_normal_data(x: np.ndarray, params: listOfTuplesOrArray,
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
    y = np.zeros_like(x, dtype=float)
    for pars in params:
        y += dist.LogNormalDistribution(*pars, normalize=normalize).pdf(x)
    if noise_level > 0:
        y += noise_level * np.random.normal(size=x.size)
    return y


def generate_multi_laplace_data(x: np.ndarray, params: listOfTuplesOrArray,
                                noise_level: float = 0.0, normalize: bool = False) -> np.ndarray:
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
    y = np.zeros_like(x, dtype=float)
    for pars in params:
        y += dist.LaplaceDistribution(*pars, normalize=normalize).pdf(x)
    if noise_level > 0:
        y += noise_level * np.random.normal(size=x.size)
    return y


def generate_multi_powerlaw_data(x: np.ndarray, params: listOfTuplesOrArray,
                                 noise_level: float = 0.0, normalize: bool = False):
    y = np.zeros_like(x, dtype=float)
    for pars in params:
        y += dist.PowerLawDistribution(*pars, normalize=normalize).pdf(x)
    if noise_level > 0:
        y += noise_level * np.random.normal(size=x.size)
    return y


def generate_multi_norris2005_data(x: np.ndarray, params: listOfTuplesOrArray,
                                   noise_level: float = 0.0, normalize: bool = False):
    y = np.zeros_like(x, dtype=float)
    for pars in params:
        y += dist.Norris2005Distribution(*pars, normalize=normalize).pdf(x)
    if noise_level > 0:
        y += noise_level * np.random.normal(size=x.size)
    return y


def generate_multi_norris2011_data(x: np.ndarray, params: listOfTuplesOrArray,
                                   noise_level: float = 0.0, normalize: bool = False):
    y = np.zeros_like(x, dtype=float)
    for pars in params:
        y += dist.Norris2011Distribution(*pars, normalize=normalize).pdf(x)
    if noise_level > 0:
        y += noise_level * np.random.normal(size=x.size)
    return y


def generate_mixed_model_data(x: np.ndarray, params: listOfTuplesOrArray, model_list,
                              noise_level=0.0, normalize: bool = False):
    y = np.zeros_like(x, dtype=float)
    par_index = 0
    for model in model_list:
        if model == GAUSSIAN:
            y += dist.GaussianDistribution(*params[par_index], normalize=normalize).pdf(x)
        elif model == LOG_NORMAL:
            y += dist.LogNormalDistribution(*params[par_index], normalize=normalize).pdf(x)
        elif model == LAPLACE:
            y += dist.LaplaceDistribution(*params[par_index], normalize=normalize).pdf(x)
        elif model == SKEW_NORMAL:
            y += dist.SkewedNormalDistribution(*params[par_index]).pdf(x)
        elif model == POWERLAW:
            y += dist.PowerLawDistribution(*params[par_index], normalize=normalize).pdf(x)
        elif model == LINE:
            y += dist.line(x, *params[par_index])

        par_index += 1

    if noise_level > 0:
        y += noise_level * np.random.normal(size=x.size)

    return y
