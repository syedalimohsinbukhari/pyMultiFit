"""Created on Jul 18 00:35:26 2024"""

from inspect import isfunction
from typing import Callable, Optional, Dict

import numpy as np
from custom_inherit import doc_inherit  # type: ignore

from .. import (
    ARC_SINE,
    BETA,
    CHI_SQUARE,
    distributions as dist,
    doc_style,
    EXPONENTIAL,
    FOLDED_NORMAL,
    GAMMA,
    GAUSSIAN,
    HALF_NORMAL,
    LAPLACE,
    LINE,
    LOG_NORMAL,
    SKEW_NORMAL,
    Params_,
    OneDArray,
)

model_map = {
    ARC_SINE: dist.ArcSineDistribution,
    BETA: dist.BetaDistribution,
    CHI_SQUARE: dist.ChiSquareDistribution,
    EXPONENTIAL: dist.ExponentialDistribution,
    FOLDED_NORMAL: dist.FoldedNormalDistribution,
    GAMMA: dist.GammaDistribution,
    GAUSSIAN: dist.GaussianDistribution,
    HALF_NORMAL: dist.HalfNormalDistribution,
    LAPLACE: dist.LaplaceDistribution,
    LOG_NORMAL: dist.LogNormalDistribution,
    SKEW_NORMAL: dist.SkewNormalDistribution,
    LINE: dist.LineFunction,
}


def multi_base(
    x: OneDArray, distribution_func: Callable, params: Params_, noise_level: float = 0.0, normalize: bool = False
) -> OneDArray:
    """Generate data based on a combination of distributions with optional noise.

    Parameters
    ----------
    x : Union[List[int | float], np.ndarray]
        Input array of values.
    distribution_func : Callable
        The distribution function to be used to generate data.
    params : Union[List[Tuple[int | float, ...]], np.ndarray]
        List of tuples containing the parameters for the required distribution.
    noise_level : float, optional
        Standard deviation of the noise to be added to the data.
        Defaults to 0.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.array
        Array of the same shape as :math:`x`, containing the evaluated values.
    """
    y = np.zeros_like(x, dtype=float)

    for param_set in params:
        if isinstance(param_set, float):
            param_set = [param_set]
        y += distribution_func(*param_set, normalize=normalize).pdf(x)

    noise = noise_level * np.random.normal(size=y.size)
    y += noise.astype(float)

    return y


def multi_chi_squared(x: OneDArray, params: Params_, noise_level: float = 0.0, normalize: bool = False) -> OneDArray:
    r"""
    Generate multi-:class:`~pymultifit.distributions.chiSquare_d.ChiSquareDistribution` data with optional noise.

    Parameters
    ----------
    x : Union[List[int | float], np.ndarray]
        Input array of values.
    params : Union[List[Tuple[int | float, ...]], np.ndarray]
        List of tuples or numpy array containing the parameters for the required distribution.
    noise_level : float, optional
        Standard deviation of the noise to be added to the data.
        Defaults to 0.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.array
        Array of the same shape as :math:`x`, containing the evaluated values.
    """
    return multi_base(
        x, distribution_func=model_map[CHI_SQUARE], params=params, noise_level=noise_level, normalize=normalize
    )


@doc_inherit(parent=multi_chi_squared, style=doc_style)
def multi_gamma(x: OneDArray, params: Params_, noise_level: float = 0.0, normalize: bool = False) -> OneDArray:
    r"""Generate multi-:class:`~pymultifit.distributions.gamma_d.GammaDistribution` data with optional noise."""
    return multi_base(
        x, distribution_func=model_map[GAMMA], params=params, noise_level=noise_level, normalize=normalize
    )


@doc_inherit(parent=multi_chi_squared, style=doc_style)
def multi_exponential(x: OneDArray, params: Params_, noise_level: float = 0.0, normalize: bool = False) -> OneDArray:
    r"""Generate multi-:class:`~pymultifit.distributions.exponential_d.ExponentialDistribution` data with optional
    noise."""
    return multi_base(
        x, distribution_func=model_map[EXPONENTIAL], params=params, noise_level=noise_level, normalize=normalize
    )


@doc_inherit(parent=multi_chi_squared, style=doc_style)
def multi_folded_normal(x: OneDArray, params: Params_, noise_level: float = 0.0, normalize: bool = False) -> OneDArray:
    r"""Generate multi-:class:`~pymultifit.distributions.foldedNormal_d.FoldedNormalDistribution` data with optional
    noise."""
    return multi_base(
        x, distribution_func=model_map[FOLDED_NORMAL], params=params, noise_level=noise_level, normalize=normalize
    )


@doc_inherit(parent=multi_chi_squared, style=doc_style)
def multi_gaussian(x: OneDArray, params: Params_, noise_level: float = 0.0, normalize: bool = False) -> OneDArray:
    r"""Generate multi-:class:`~pymultifit.distributions.gaussian_d.GaussianDistribution` data with optional noise."""
    return multi_base(
        x, distribution_func=model_map[GAUSSIAN], params=params, noise_level=noise_level, normalize=normalize
    )


@doc_inherit(parent=multi_chi_squared, style=doc_style)
def multi_half_normal(x: OneDArray, params: Params_, noise_level: float = 0.0, normalize: bool = False) -> OneDArray:
    r"""Generate multi-:class:`~pymultifit.distributions.halfNormal_d.HalfNormalDistribution` data with optional
    noise."""
    return multi_base(
        x, distribution_func=model_map[HALF_NORMAL], params=params, noise_level=noise_level, normalize=normalize
    )


@doc_inherit(parent=multi_chi_squared, style=doc_style)
def multi_laplace(x: OneDArray, params: Params_, noise_level: float = 0.0, normalize: bool = False) -> OneDArray:
    r"""Generate multi-:class:`~pymultifit.distributions.laplace_d.LaplaceDistribution` data with optional noise."""
    return multi_base(
        x, distribution_func=model_map[LAPLACE], params=params, noise_level=noise_level, normalize=normalize
    )


@doc_inherit(parent=multi_chi_squared, style=doc_style)
def multi_log_normal(x: OneDArray, params: Params_, noise_level: float = 0.0, normalize: bool = False) -> OneDArray:
    r"""Generate multi-:class:`~pymultifit.distributions.logNormal_d.LogNormalDistribution` data with optional noise."""
    return multi_base(
        x, distribution_func=model_map[LOG_NORMAL], params=params, noise_level=noise_level, normalize=normalize
    )


@doc_inherit(parent=multi_chi_squared, style=doc_style)
def multi_skew_normal(x: OneDArray, params: Params_, noise_level: float = 0.0, normalize: bool = False) -> OneDArray:
    r"""Generate multi-:class:`~pymultifit.distributions.skewNormal_d.SkewNormalDistribution` data with optional
    noise."""
    return multi_base(
        x, distribution_func=model_map[SKEW_NORMAL], params=params, noise_level=noise_level, normalize=normalize
    )


def multiple_models(
    x: OneDArray,
    params: Params_,
    model_list: list[str],
    noise_level: float = 0.0,
    normalize: bool = False,
    mapping_dict: Optional[Dict[str, Callable]] = None,
) -> OneDArray:
    """
    Generate data based on a combination of different models with optional noise.

    Parameters
    ----------
    x : Union[List[int | float], np.ndarray]
        Input array of values.
    params : Union[List[Tuple[int | float, ...]], np.ndarray]
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
    np.array
        Array of the same shape as :math:`x`, containing the evaluated values.
    """
    y = np.zeros_like(x, dtype=float)

    model_mapping = model_map if mapping_dict is None else mapping_dict

    for par_index, model in enumerate(model_list):
        if model in model_mapping:
            _instance = model_mapping[model]
            if isfunction(_instance):
                y += _instance(x, *params[par_index])
            else:
                y += _instance(*params[par_index], normalize=normalize).pdf(np.asarray(x))

    noise = noise_level * np.random.normal(size=y.size)
    y += noise.astype(float)

    return y
