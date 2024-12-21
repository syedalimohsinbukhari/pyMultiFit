"""Created on Jul 18 00:35:26 2024"""

from typing import Callable, List, Tuple, Union

import numpy as np
from custom_inherit import doc_inherit

from .. import distributions as dist, GAUSSIAN, LAPLACE, LINE, LOG_NORMAL, POWERLAW, SKEW_NORMAL

listOfTuples = List[Tuple[float, ...]]
listOfTuplesOrArray = Union[listOfTuples, np.ndarray]

doc_style = 'numpy_napoleon_with_merge'


def multi_chi_squared(x: np.ndarray, params: listOfTuplesOrArray,
                      noise_level: float = 0.0, normalize: bool = False) -> np.ndarray:
    """Generate multi-Chi-Squared data with optional noise.

    Parameters
    ----------
    x : np.ndarray
        X values.
    params : listOfTuplesOrArray
        List of tuples containing the parameters for each Chi-Squared distribution.
    noise_level : float, optional
        Standard deviation of the noise to be added to the data, by default 0.0.
    normalize : bool, optional
        If True, the function produces normalized data (Integration[PDF] < 1). Defaults to False.

    Returns
    -------
    np.ndarray
        Y values of the multi-Chi-Squared data with added noise.
    """
    return multi_base(x=x, distribution_func=dist.ChiSquareDistribution, params=params, noise_level=noise_level, normalize=normalize)


@doc_inherit(parent=multi_chi_squared, style=doc_style)
def multi_base(x: np.ndarray, distribution_func: Callable, params: listOfTuplesOrArray,
               noise_level: float = 0.0, normalize: bool = False) -> np.ndarray:
    """Generate data based on a combination of distributions with optional noise.

    Returns
    -------
    np.ndarray
        Y values of the generated data with added noise.
    """
    y = np.zeros_like(x, dtype=float)

    for param_set in params:
        y += distribution_func(*param_set, normalize=normalize).pdf(x)

    if noise_level > 0:
        y += noise_level * np.random.normal(size=x.size)

    return y


@doc_inherit(parent=multi_chi_squared, style=doc_style)
def multi_exponential(x: np.ndarray, params: listOfTuplesOrArray,
                      noise_level: float = 0.0, normalize: bool = False) -> np.ndarray:
    """Generate multi-Exponential data with optional noise.

    Returns
    -------
    np.ndarray
        Y values of the multi-Exponential data with added noise.
    """
    return multi_base(x=x, distribution_func=dist.ExponentialDistribution, params=params, noise_level=noise_level, normalize=normalize)


@doc_inherit(parent=multi_chi_squared, style=doc_style)
def multi_folded_normal(x: np.ndarray, params: listOfTuplesOrArray,
                        noise_level: float = 0.0, normalize: bool = False) -> np.ndarray:
    """Generate multi-Folded Normal data with optional noise.

    Returns
    -------
    np.ndarray
        Y values of the multi-Folded Normal data with added noise.
    """
    return multi_base(x=x, distribution_func=dist.FoldedNormalDistribution, params=params, noise_level=noise_level, normalize=normalize)


@doc_inherit(parent=multi_chi_squared, style=doc_style)
def multi_gaussian(x: np.ndarray, params: listOfTuplesOrArray,
                   noise_level: float = 0.0, normalize: bool = False) -> np.ndarray:
    """Generate multi-Gaussian data with optional noise.

    Returns
    -------
    np.ndarray
        Y values of the multi-Gaussian data with added noise.
    """
    return multi_base(x=x, distribution_func=dist.GaussianDistribution, params=params, noise_level=noise_level, normalize=normalize)


@doc_inherit(parent=multi_chi_squared, style=doc_style)
def multi_half_normal(x: np.ndarray, params: listOfTuplesOrArray,
                      noise_level: float = 0.0, normalize: bool = False) -> np.ndarray:
    """Generate multi-Half-Normal data with optional noise.

    Returns
    -------
    np.ndarray
        Y values of the multi-Half-Normal data with added noise.
    """
    return multi_base(x=x, distribution_func=dist.HalfNormalDistribution, params=params, noise_level=noise_level, normalize=normalize)


@doc_inherit(parent=multi_chi_squared, style=doc_style)
def multi_laplace(x: np.ndarray, params: listOfTuplesOrArray,
                  noise_level: float = 0.0, normalize: bool = False) -> np.ndarray:
    """Generate multi-Laplace data with optional noise.

    Returns
    -------
    np.ndarray
        Y values of the multi-Laplace data with added noise.
    """
    return multi_base(x=x, distribution_func=dist.LaplaceDistribution, params=params, noise_level=noise_level, normalize=normalize)


@doc_inherit(parent=multi_chi_squared, style=doc_style)
def multi_log_normal(x: np.ndarray, params: listOfTuplesOrArray,
                     noise_level: float = 0.0, normalize: bool = False) -> np.ndarray:
    """Generate multi-Log-Normal data with optional noise.

    Returns
    -------
    np.ndarray
        Y values of the multi-Log-Normal data with added noise.
    """
    return multi_base(x=x, distribution_func=dist.LogNormalDistribution, params=params, noise_level=noise_level, normalize=normalize)


# def multi_norris2005(x: np.ndarray, params: listOfTuplesOrArray,
#                      noise_level: float = 0.0, normalize: bool = False):
#     y = np.zeros_like(x, dtype=float)
#     for pars in params:
#         y += dist.Norris2005Distribution(*pars, normalize=normalize).pdf(x)
#     if noise_level > 0:
#         y += noise_level * np.random.normal(size=x.size)
#     return y
#
#
# def multi_norris2011(x: np.ndarray, params: listOfTuplesOrArray,
#                      noise_level: float = 0.0, normalize: bool = False):
#     y = np.zeros_like(x, dtype=float)
#     for pars in params:
#         y += dist.Norris2011Distribution(*pars, normalize=normalize).pdf(x)
#     if noise_level > 0:
#         y += noise_level * np.random.normal(size=x.size)
#     return y
#
#
# def multi_powerlaw(x: np.ndarray, params: listOfTuplesOrArray,
#                    noise_level: float = 0.0, normalize: bool = False):
#     y = np.zeros_like(x, dtype=float)
#     for pars in params:
#         y += dist.PowerLawDistribution(*pars, normalize=normalize).pdf(x)
#     if noise_level > 0:
#         y += noise_level * np.random.normal(size=x.size)
#     return y


def multi_skewed_normal(x: np.ndarray, params: listOfTuplesOrArray,
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


def multiple_models(x: np.ndarray, params: listOfTuplesOrArray, model_list,
                    noise_level=0.0, normalize: bool = False) -> np.ndarray:
    """Generate data based on a combination of different models with optional noise.

    Parameters
    ----------
    x : np.ndarray
        X values.
    params : listOfTuplesOrArray
        List of tuples containing the parameters for each model.
    model_list : list
        A list of model names corresponding to the models to be used.
    noise_level : float, optional
        Standard deviation of the noise to be added to the data, by default 0.0.
    normalize : bool, optional
        If True, the function produces normalized data (Integration[PDF] < 1). Defaults to False.

    Returns
    -------
    np.ndarray
        Y values of the generated data with added noise.
    """
    y = np.zeros_like(x, dtype=float)
    model_mapping = {GAUSSIAN: dist.GaussianDistribution,
                     LOG_NORMAL: dist.LogNormalDistribution,
                     LAPLACE: dist.LaplaceDistribution,
                     SKEW_NORMAL: dist.SkewedNormalDistribution,
                     POWERLAW: dist.PowerLawDistribution,
                     LINE: dist.line}

    for par_index, model in enumerate(model_list):
        if model in model_mapping:
            if model == LINE:
                y += model_mapping[model](x, *params[par_index])
            else:
                y += model_mapping[model](*params[par_index], normalize=normalize).pdf(x)

    if noise_level > 0:
        y += noise_level * np.random.normal(size=x.size)

    return y
