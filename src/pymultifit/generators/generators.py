"""Created on Jul 18 00:35:26 2024"""
from inspect import isfunction
from typing import Type, Callable, Optional, Dict

import numpy as np
from custom_inherit import doc_inherit

from .. import (ARC_SINE,
                BETA,
                CHI_SQUARE,
                distributions as dist,
                doc_style,
                EXPONENTIAL,
                FOLDED_NORMAL,
                GAMMA_SR,
                GAMMA_SS,
                GAUSSIAN,
                HALF_NORMAL,
                LAPLACE,
                LINE,
                listOfTuplesOrArray,
                LOG_NORMAL,
                SKEW_NORMAL)
from ..distributions.backend import BaseDistribution

model_map = {ARC_SINE: dist.ArcSineDistribution,
             BETA: dist.BetaDistribution,
             CHI_SQUARE: dist.ChiSquareDistribution,
             EXPONENTIAL: dist.ExponentialDistribution,
             FOLDED_NORMAL: dist.FoldedNormalDistribution,
             GAMMA_SS: dist.GammaDistributionSS,
             GAMMA_SR: dist.GammaDistributionSR,
             GAUSSIAN: dist.GaussianDistribution,
             HALF_NORMAL: dist.HalfNormalDistribution,
             LAPLACE: dist.LaplaceDistribution,
             LOG_NORMAL: dist.LogNormalDistribution,
             SKEW_NORMAL: dist.SkewNormalDistribution,
             LINE: dist.Line}


def multi_base(x: np.ndarray, distribution_func: Type[BaseDistribution], params: listOfTuplesOrArray,
               noise_level: float = 0.0, normalize: bool = False) -> np.ndarray:
    """Generate data based on a combination of distributions with optional noise.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.
    distribution_func: Type[BaseDistribution]
        The distribution function to be used to generate data.
    params : listOfTuplesOrArray
        List of tuples containing the parameters for the required distribution.
    noise_level : float, optional
        Standard deviation of the noise to be added to the data.
        Defaults to 0.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.ndarray
        Array of the same shape as :math:`x`, containing the evaluated values.
    """
    y = np.zeros_like(x, dtype=float)

    for param_set in params:
        if isinstance(param_set, float):
            param_set = [param_set]
        y += distribution_func(*param_set, normalize=normalize).pdf(x)

    if noise_level > 0:
        y += noise_level * np.random.normal(size=x.size)

    return y


def multi_chi_squared(x: np.ndarray, params: listOfTuplesOrArray,
                      noise_level: float = 0.0, normalize: bool = False) -> np.ndarray:
    r"""
    Generate multi-:class:`~pymultifit.distributions.chiSquare_d.ChiSquareDistribution` data with optional noise.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.
    params : listOfTuplesOrArray
        List of tuples containing the parameters for the required distribution.
    noise_level : float, optional
        Standard deviation of the noise to be added to the data.
        Defaults to 0.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.ndarray
        Array of the same shape as :math:`x`, containing the evaluated values.
    """
    return multi_base(x=x,
                      distribution_func=dist.ChiSquareDistribution, params=params, noise_level=noise_level,
                      normalize=normalize)


@doc_inherit(parent=multi_chi_squared, style=doc_style)
def multi_gamma_sr(x: np.ndarray, params: listOfTuplesOrArray,
                   noise_level: float = 0.0, normalize: bool = False) -> np.ndarray:
    r"""Generate multi-:class:`~pymultifit.distributions.gamma_d.GammaDistributionSR` data with optional noise."""
    return multi_base(x=x,
                      distribution_func=dist.GammaDistributionSR, params=params, noise_level=noise_level,
                      normalize=normalize)


@doc_inherit(parent=multi_chi_squared, style=doc_style)
def multi_gamma_ss(x: np.ndarray, params: listOfTuplesOrArray,
                   noise_level: float = 0.0, normalize: bool = False) -> np.ndarray:
    r"""Generate multi-:class:`~pymultifit.distributions.gamma_d.GammaDistributionSS` data with optional noise."""
    return multi_base(x=x,
                      distribution_func=dist.GammaDistributionSS, params=params, noise_level=noise_level,
                      normalize=normalize)


@doc_inherit(parent=multi_chi_squared, style=doc_style)
def multi_exponential(x: np.ndarray, params: listOfTuplesOrArray,
                      noise_level: float = 0.0, normalize: bool = False) -> np.ndarray:
    r"""Generate multi-:class:`~pymultifit.distributions.exponential_d.ExponentialDistribution` data with optional
    noise."""
    return multi_base(x=x,
                      distribution_func=dist.ExponentialDistribution, params=params, noise_level=noise_level,
                      normalize=normalize)


@doc_inherit(parent=multi_chi_squared, style=doc_style)
def multi_folded_normal(x: np.ndarray, params: listOfTuplesOrArray,
                        noise_level: float = 0.0, normalize: bool = False) -> np.ndarray:
    r"""Generate multi-:class:`~pymultifit.distributions.foldedNormal_d.FoldedNormalDistribution` data with optional
    noise."""
    return multi_base(x=x,
                      distribution_func=dist.FoldedNormalDistribution, params=params, noise_level=noise_level,
                      normalize=normalize)


@doc_inherit(parent=multi_chi_squared, style=doc_style)
def multi_gaussian(x: np.ndarray, params: listOfTuplesOrArray,
                   noise_level: float = 0.0, normalize: bool = False) -> np.ndarray:
    r"""Generate multi-:class:`~pymultifit.distributions.gaussian_d.GaussianDistribution` data with optional noise."""
    return multi_base(x=x,
                      distribution_func=dist.GaussianDistribution, params=params, noise_level=noise_level,
                      normalize=normalize)


@doc_inherit(parent=multi_chi_squared, style=doc_style)
def multi_half_normal(x: np.ndarray, params: listOfTuplesOrArray,
                      noise_level: float = 0.0, normalize: bool = False) -> np.ndarray:
    r"""Generate multi-:class:`~pymultifit.distributions.halfNormal_d.HalfNormalDistribution` data with optional
    noise."""
    return multi_base(x=x,
                      distribution_func=dist.HalfNormalDistribution, params=params, noise_level=noise_level,
                      normalize=normalize)


@doc_inherit(parent=multi_chi_squared, style=doc_style)
def multi_laplace(x: np.ndarray, params: listOfTuplesOrArray,
                  noise_level: float = 0.0, normalize: bool = False) -> np.ndarray:
    r"""Generate multi-:class:`~pymultifit.distributions.laplace_d.LaplaceDistribution` data with optional noise."""
    return multi_base(x=x,
                      distribution_func=dist.LaplaceDistribution, params=params, noise_level=noise_level,
                      normalize=normalize)


@doc_inherit(parent=multi_chi_squared, style=doc_style)
def multi_log_normal(x: np.ndarray, params: listOfTuplesOrArray,
                     noise_level: float = 0.0, normalize: bool = False) -> np.ndarray:
    r"""Generate multi-:class:`~pymultifit.distributions.logNormal_d.LogNormalDistribution` data with optional noise."""
    return multi_base(x=x,
                      distribution_func=dist.LogNormalDistribution, params=params, noise_level=noise_level,
                      normalize=normalize)


@doc_inherit(parent=multi_chi_squared, style=doc_style)
def multi_skew_normal(x: np.ndarray, params: listOfTuplesOrArray,
                      noise_level: float = 0.0, normalize: bool = False) -> np.ndarray:
    r"""Generate multi-:class:`~pymultifit.distributions.skewNormal_d.SkewNormalDistribution` data with optional
    noise."""
    return multi_base(x=x,
                      distribution_func=dist.SkewNormalDistribution, params=params, noise_level=noise_level,
                      normalize=normalize)


def multiple_models(x: np.ndarray, params: listOfTuplesOrArray, model_list,
                    noise_level=0.0, normalize: bool = False,
                    mapping_dict: Optional[Dict[str, Callable]] = None) -> np.ndarray:
    """
    Generate data based on a combination of different models with optional noise.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.
    params : listOfTuplesOrArray
        List of tuples containing the parameters for each model.
    model_list : list
        A list of model names corresponding to the models to be used.
    noise_level : float, optional
        Standard deviation of the noise to be added to the data, by default 0.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.
    mapping_dict: dict, optional
        A dictionary mapping between distribution names and their corresponding classes.

    Returns
    -------
    np.ndarray
        Array of the same shape as :math:`x`, containing the evaluated values.
    """

    y = np.zeros_like(a=x, dtype=float)

    model_mapping = model_map if mapping_dict is None else mapping_dict

    for par_index, model in enumerate(model_list):
        if model in model_mapping:
            _instance = model_mapping[model]
            # check if it is a function
            if isfunction(_instance):
                y += _instance(x, *params[par_index])
            # check if it is a subclass of BaseDistribution
            elif issubclass(_instance, BaseDistribution):
                y += _instance(*params[par_index], normalize=normalize).pdf(x)

    if noise_level > 0:
        y += noise_level * np.random.normal(size=x.size)

    return y
