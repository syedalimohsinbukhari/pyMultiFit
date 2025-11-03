"""Created on Aug 03 17:13:21 2024"""

__all__ = [
    "_beta_expr",
    "_pdf_scaling",
    "preprocess_input",
    "arc_sine_pdf_",
    "arc_sine_cdf_",
    "arc_sine_log_pdf_",
    "arc_sine_log_cdf_",
    "beta_pdf_",
    "beta_cdf_",
    "beta_log_pdf_",
    "beta_log_cdf_",
    "beta_prime_pdf_",
    "beta_prime_cdf_",
    "beta_prime_log_pdf_",
    "beta_prime_log_cdf_",
    "chi_square_pdf_",
    "chi_square_cdf_",
    "chi_square_log_pdf_",
    "chi_square_log_cdf_",
    "exponential_pdf_",
    "exponential_cdf_",
    "exponential_log_pdf_",
    "exponential_log_cdf_",
    "folded_normal_pdf_",
    "folded_normal_cdf_",
    "folded_normal_log_pdf_",
    "folded_normal_log_cdf_",
    "gamma_pdf_",
    "gamma_log_pdf_",
    "gamma_cdf_",
    "gamma_log_cdf_",
    "sym_gen_normal_pdf_",
    "sym_gen_normal_cdf_",
    "sym_gen_normal_log_pdf_",
    "sym_gen_normal_log_cdf_",
    "gaussian_pdf_",
    "gaussian_cdf_",
    "gaussian_log_pdf_",
    "gaussian_log_cdf_",
    "half_normal_pdf_",
    "half_normal_cdf_",
    "half_normal_log_pdf_",
    "half_normal_log_cdf_",
    "johnsonSU_pdf",
    "johnsonSU_log_pdf_",
    "johnsonSU_cdf_",
    "johnsonSU_log_cdf_",
    "laplace_pdf_",
    "laplace_cdf_",
    "laplace_log_pdf_",
    "laplace_log_cdf_",
    "log_normal_pdf_",
    "log_normal_cdf_",
    "log_normal_log_pdf_",
    "log_normal_log_cdf_",
    "scaled_inv_chi_square_pdf_",
    "scaled_inv_chi_square_log_pdf_",
    "scaled_inv_chi_square_cdf_",
    "scaled_inv_chi_square_log_cdf_",
    "skew_normal_pdf_",
    "skew_normal_cdf_",
    "uniform_pdf_",
    "uniform_cdf_",
    "uniform_log_pdf_",
    "uniform_log_cdf_",
    "line",
    "quadratic",
    "cubic",
]

import functools
from typing import Union, Callable

import numpy as np
import scipy.special as ssp
from custom_inherit import doc_inherit  # type: ignore

from .. import (
    doc_style,
    LOG,
    INF,
    OneDArray,
    SQRT_TWO,
    LOG_TWO,
    PI,
    SQRT_TWO_PI,
    LOG_SQRT_TWO_PI,
    TWO_BY_PI,
    SQRT_TWO_BY_PI,
    LOG_SQRT_TWO_BY_PI,
)


def suppress_numpy_warnings():
    """
    A decorator that suppresses NumPy warnings using np.errstate.

    Parameters (all optional):
        divide, over, under, invalid: Can be 'ignore', 'warn', 'raise', 'call', 'print', or 'log'
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with np.errstate(all="ignore"):
                return func(*args, **kwargs)

        return wrapper

    return decorator


@suppress_numpy_warnings()
def arc_sine_pdf_(
    x: OneDArray, amplitude: float = 1.0, loc: float = 0.0, scale: float = 1.0, normalize: bool = False
) -> OneDArray:
    r"""
    Compute PDF of :class:`~pymultifit.distributions.arcSine_d.ArcSineDistribution`.

    Parameters
    ----------
    x : np.ndarray
        Input array of values where PDF is evaluated.
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    loc : float, optional
        The location parameter specifying the lower bound of the distribution.
        Defaults to 0.0.
    scale : float, optional
        The scale parameter, specifying the width of the distribution.
        Defaults to 1.0.
    normalize : bool, optional
        If True, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.ndarray
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The ArcSine PDF is defined as:

    .. math:: f(y) = \frac{1}{\pi \sqrt{y(1-y)}}

    where, :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \frac{x - \text{loc}}{\text{scale}}

    The final PDF is expressed as :math:`f(y)/\text{scale}`.
    """
    y = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    if scale < 0:
        return np.full(y.shape, np.nan)

    c1 = (y > 0) & (y < 1)
    c2 = y == 0
    c3 = y == 1

    z = y * (1 - y)

    pdf_ = np.select(condlist=[c1, c2, c3], choicelist=[1 / PI / np.sqrt(z), INF, INF], default=0.0)
    pdf_ /= scale

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_


@suppress_numpy_warnings()
@doc_inherit(parent=arc_sine_pdf_, style=doc_style)
def arc_sine_log_pdf_(
    x: OneDArray, amplitude: float = 1.0, loc: float = 0.0, scale: float = 1.0, normalize: bool = False
) -> OneDArray:
    r"""
    Compute logPDF of :class:`~pymultifit.distributions.arcSine_d.ArcSineDistribution`.

    Notes
    -----
    The ArcSine logPDF is defined as:

    .. math:: \ell(y) = -\ln(\pi)  - 0.5\ln(y-y^2)

    where, :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \frac{x - \text{loc}}{\text{scale}}

    The final logPDF is expressed as :math:`\ell(y) - \ln(\text{scale})`.
    """
    y = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    c1 = (y > 0) & (y < 1)
    c2 = y == 0
    c3 = y == 1

    z = y * (1 - y)

    log_pdf_ = np.select(condlist=[c1, c2, c3], choicelist=[LOG(1 / PI / np.sqrt(z)), INF, INF], default=-INF)
    log_pdf_ -= LOG(scale)

    if not normalize:
        log_pdf_ = _log_pdf_scaling(log_pdf_=log_pdf_, amplitude=amplitude)

    return log_pdf_


@suppress_numpy_warnings()
@doc_inherit(parent=arc_sine_pdf_, style=doc_style)
def arc_sine_cdf_(
    x: OneDArray, amplitude: float = 1.0, loc: float = 0.0, scale: float = 1.0, normalize: bool = False
) -> OneDArray:
    y = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    if scale < 0:
        return np.full(y.shape, np.nan)

    c1 = (y > 0) & (y < 1)
    c2 = y < 1

    return np.select(condlist=[c1, c2], choicelist=[TWO_BY_PI * np.arcsin(np.sqrt(y)), 0.0], default=1.0)


@suppress_numpy_warnings()
@doc_inherit(parent=arc_sine_cdf_, style=doc_style)
def arc_sine_log_cdf_(
    x: OneDArray, amplitude: float = 1.0, loc: float = 0.0, scale: float = 1.0, normalize: bool = False
) -> OneDArray:
    r"""
    Compute log CDF of :class:`~pymultifit.distributions.arcSine_d.ArcSineDistribution`.

    Notes
    -----
    The ArcSine log CDF is defined as:

    .. math:: \mathcal{L}(y) = \ln\left(\frac{2}{\pi}\right) + \ln\arcsin(\sqrt{y})

    where :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \dfrac{x - \text{loc}}{\text{scale}}

    The final logCDF is expressed as :math:`\mathcal{L}(y)`.
    """
    y = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    c1 = (y > 0) & (y < 1)
    c2 = y < 1

    return np.select(condlist=[c1, c2], choicelist=[LOG(TWO_BY_PI * np.arcsin(np.sqrt(y))), -INF], default=0.0)


@suppress_numpy_warnings()
def beta_pdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    alpha: float = 1.0,
    beta_: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> OneDArray:
    r"""
    Compute PDF of :class:`~pymultifit.distributions.beta_d.BetaDistribution`.

    Parameters
    ----------
    beta_
    x : np.ndarray
        Input array of values where PDF is evaluated.
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    alpha : float, optional
        The :math:`\alpha` parameter.
        Default is 1.0.
    beta_ : float, optional
        The :math:`\beta` parameter.
        Default is 1.0.
    loc : float, optional
        The location parameter, for shifting.
        Default is 0.0.
    scale : float, optional
        The scale parameter, for scaling.
        Default is 1.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.ndarray
        Array of the same shape as `x`, containing the evaluated values.

    Notes
    -----
    The Beta PDF is defined as:

    .. math:: f(y; \alpha, \beta) = \frac{y^{\alpha - 1} (1 - y)^{\beta - 1}}{B(\alpha, \beta)}

    where :math:`B(\alpha, \beta)` is the Beta function (see, :obj:`ssp.beta`), and :math:`y` is the
    transformed value of :math:`x` such that:

    .. math:: y = \frac{x - \text{loc}}{\text{scale}}

    The final PDF is expressed as :math:`f(y)/\text{scale}`.
    """
    y = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    conditions, main = _beta_expr(y=y, a=alpha, b=beta_, un_log=True)

    pdf_ = np.select(condlist=conditions, choicelist=[1, np.nan, main], default=0)
    pdf_ /= scale

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_


@suppress_numpy_warnings()
@doc_inherit(parent=beta_pdf_, style=doc_style)
def beta_log_pdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    alpha: float = 1.0,
    beta_: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> OneDArray:
    r"""Compute logPDF for :class:`~pymultifit.distributions.beta_d.BetaDistribution`.

    Notes
    -----
    The Beta logPDFis defined as

    .. math:: \ell(y) = (\alpha - 1)\ln(y) + (\beta - 1)\ln(1 - y) - \ln(\text{Beta}(\alpha, \beta))

    where :math:`B(\alpha, \beta)` is the :obj:`~ssp.beta` function, and :math:`y` is the
    transformed value of :math:`x` such that:

    .. math:: y = \frac{x - \text{loc}}{\text{scale}}

    The final logPDF is expressed as :math:`\ell(y) - \ln(\text{scale})`.
    """
    y = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    conditions, main = _beta_expr(y=y, a=alpha, b=beta_)

    log_pdf_ = np.select(condlist=conditions, choicelist=[0.0, np.nan, main], default=-INF)
    log_pdf_ -= LOG(scale)

    if not normalize:
        log_pdf_ = _log_pdf_scaling(log_pdf_=log_pdf_, amplitude=amplitude)

    return log_pdf_


@suppress_numpy_warnings()
@doc_inherit(parent=beta_pdf_, style=doc_style)
def beta_cdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    alpha: float = 1.0,
    beta_: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> OneDArray:
    r"""
    Compute CDF for :class:`~pymultifit.distributions.beta_d.BetaDistribution`.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.
    amplitude : float, optional
        For API consistency only.
    normalize : bool, optional
        For API consistency only.

    Notes
    -----
    The Beta CDF is defined as:

    .. math:: F(y) = I_y(\alpha, \beta)

    where :math:`I_y(\alpha, \beta)` is the :obj:`~ssp.betainc` function, and :math:`y` is the transformed
    value of :math:`x`, defined as:

    .. math:: y = \frac{x - \text{loc}}{\text{scale}}

    The final CDF is expressed as :math:`F(y)`.
    """
    y = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    return np.select(condlist=[y > 1, y < 0], choicelist=[1, 0], default=ssp.betainc(alpha, beta_, y))


@suppress_numpy_warnings()
@doc_inherit(parent=beta_cdf_, style=doc_style)
def beta_log_cdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    alpha: float = 1.0,
    beta_: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> OneDArray:
    r"""
    Compute logCDF for :class:`~pymultifit.distributions.beta_d.BetaDistribution`.

    Notes
    -----
    The Beta logCDF is defined as:

    .. math:: \mathcal{L}(y) = \ln I_y(\alpha, \beta)

    where :math:`I_y(\alpha, \beta)` is the :obj:`~ssp.betainc` function, and :math:`y` is the transformed
    value of :math:`x`, defined as:

    .. math:: y = \frac{x - \text{loc}}{\text{scale}}

    The final logCDF is expressed as :math:`\mathcal{L}(y)`.
    """
    y = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    return np.select(condlist=[y > 1, y < 0], choicelist=[0, -INF], default=LOG(ssp.betainc(alpha, beta_, y)))


@suppress_numpy_warnings()
def beta_prime_pdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    alpha: float = 1.0,
    beta_: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> OneDArray:
    r"""
    Compute PDF of :class:`~pymultifit.distributions.beta_d.BetaDistribution`.

    Parameters
    ----------
    beta_
    x : np.ndarray
        Input array of values where PDF is evaluated.
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    alpha : float, optional
        The :math:`\alpha` parameter.
        Default is 1.0.
    beta_ : float, optional
        The :math:`\beta` parameter.
        Default is 1.0.
    loc : float, optional
        The location parameter, for shifting.
        Default is 0.0.
    scale : float, optional
        The scale parameter, for scaling.
        Default is 1.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.ndarray
        Array of the same shape as `x`, containing the evaluated values.

    Notes
    -----
    The Beta-prime PDF is defined as:

    .. math:: f(y; \alpha, \beta) = \frac{y^{\alpha - 1} (1 + y)^{-(\alpha + \beta)}}{B(\alpha, \beta)}

    where :math:`B(\alpha, \beta)` is the Beta function (see, :obj:`ssp.beta`), and :math:`y` is the
    transformed value of :math:`x` such that:

    .. math:: y = \frac{x - \text{loc}}{\text{scale}}

    The final PDF is expressed as :math:`f(y)/\text{scale}`.
    """
    y = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    log_expr = ssp.xlogy(alpha - 1.0, y) - ssp.xlog1py(alpha + beta_, y) - ssp.betaln(alpha, beta_)
    pdf_ = np.select(condlist=[y > 0, (y == 0) & (alpha <= 1)], choicelist=[np.exp(log_expr), np.nan], default=0.0)
    pdf_ /= scale

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_


@suppress_numpy_warnings()
@doc_inherit(parent=beta_prime_pdf_, style=doc_style)
def beta_prime_log_pdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    alpha: float = 1.0,
    beta_: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> OneDArray:
    r"""Compute logPDF for :class:`~pymultifit.distributions.beta_d.BetaDistribution`.

    Notes
    -----
    The Beta-prime logPDF is defined as

    .. math:: \ell(y) = (\alpha - 1)\ln(y) - (\alpha + \beta)\ln(1 - y) - \ln(\text{Beta}(\alpha, \beta))

    where :math:`B(\alpha, \beta)` is the :obj:`~ssp.beta` function, and :math:`y` is the
    transformed value of :math:`x` such that:

    .. math:: y = \frac{x - \text{loc}}{\text{scale}}

    The final logPDF is expressed as :math:`\ell(y) - \ln(\text{scale})`.
    """
    y = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    expr = ssp.xlogy(alpha - 1.0, y) - ssp.xlog1py(alpha + beta_, y) - ssp.betaln(alpha, beta_)
    log_pdf_ = np.select(condlist=[y > 0, (y == 0) & (alpha <= 1)], choicelist=[expr, np.nan], default=-INF)
    log_pdf_ -= LOG(scale)

    if not normalize:
        log_pdf_ = _log_pdf_scaling(pdf_=log_pdf_, amplitude=amplitude)

    return log_pdf_


@suppress_numpy_warnings()
@doc_inherit(parent=beta_prime_pdf_, style=doc_style)
def beta_prime_cdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    alpha: float = 1.0,
    beta_: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> OneDArray:
    r"""
    Compute CDF for :class:`~pymultifit.distributions.beta_d.BetaDistribution`.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.
    amplitude : float, optional
        For API consistency only.
    normalize : bool, optional
        For API consistency only.

    Notes
    -----
    The Beta-prime CDF is defined as:

    .. math:: F(y) = I_{\dfrac{y}{1+y}}(\alpha, \beta)

    where :math:`I_y(\alpha, \beta)` is the :obj:`~ssp.betainc` function, and :math:`y` is the transformed
    value of :math:`x`, defined as:

    .. math:: y = \frac{x - \text{loc}}{\text{scale}}

    The final CDF is expressed as :math:`F(y)`.
    """
    y = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    z = y / (1 + y)
    return np.where(y > 0, ssp.betainc(alpha, beta_, z), 0)


def beta_prime_log_cdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    alpha: float = 1.0,
    beta_: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> OneDArray:
    r"""
    Compute logCDF for :class:`~pymultifit.distributions.beta_d.BetaDistribution`.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.
    amplitude : float, optional
        For API consistency only.
    normalize : bool, optional
        For API consistency only.

    Notes
    -----
    The Beta-prime logCDF is defined as:

    .. math:: F(y) = \ln\left[I_{\dfrac{y}{1+y}}(\alpha, \beta)\right]a

    where :math:`I_y(\alpha, \beta)` is the :obj:`~ssp.betainc` function, and :math:`y` is the transformed
    value of :math:`x`, defined as:

    .. math:: y = \frac{x - \text{loc}}{\text{scale}}

    The final logCDF is expressed as :math:`\mathcal{L}(y)`.
    """
    y = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    z = y / (1 + y)
    return np.where(y > 0, LOG(ssp.betainc(alpha, beta_, z)), -INF)


@suppress_numpy_warnings()
def chi_square_pdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    degree_of_freedom: Union[int, float] = 1,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> OneDArray:
    r"""
    Compute PDF for :mod:`~pymultifit.distributions.chiSquare_d.ChiSquareDistribution`.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    degree_of_freedom : int, optional
        The degrees of freedom parameter.
        Defaults to 1.
    loc : float, optional
        The location parameter, for shifting.
        Defaults to 0.0.
    scale: float, optional
        The scale parameter, for scaling.
        Defaults to 1.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.ndarray
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The ChiSquare PDF is defined as:

    .. math:: f(y\ |\ k) = \dfrac{y^{(k/2) - 1} e^{-y/2}}{2^{k/2} \Gamma(k/2)}

    where :math:`\Gamma(\cdot)` is the :obj:`~ssp.gamma` function, and :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \dfrac{x - \text{loc}}{\text{scale}}

    The final PDF is expressed as :math:`f(y)/\text{scale}`.
    """
    y = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    df_half = degree_of_freedom / 2.0

    pdf_ = np.where(y > 0, np.exp(_chi2(y=y, df_half=df_half)), 0)
    pdf_ /= scale

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_


def _chi2(y, df_half):
    return ssp.xlogy(df_half - 1, y) - (y / 2) - ssp.gammaln(df_half) - (LOG_TWO * df_half)


@suppress_numpy_warnings()
@doc_inherit(parent=chi_square_pdf_, style=doc_style)
def chi_square_log_pdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    degree_of_freedom: Union[int, float] = 1,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> OneDArray:
    r"""
    Compute log PDF for :mod:`~pymultifit.distributions.chiSquare_d.ChiSquareDistribution`.

    Notes
    -----
    The ChiSquare log PDF is defined as:

    .. math:: \ell(y\ |\ k) = \left(\dfrac{k}{2} - 1\right)\ln(y) - \dfrac{y}{2} - \dfrac{k}{2}\ln(2) - \ln\Gamma\left(\dfrac{k}{2}\right)

    where :math:`\ln\Gamma(\cdot)` is the :obj:`~ssp.gammaln` function,
    and :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \dfrac{x - \text{loc}}{\text{scale}}

    The final PDF is expressed as :math:`\ell(y) - \ln(\text{scale})`.

    """
    y = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    df_half = degree_of_freedom / 2.0

    log_pdf_ = np.where(y > 0, _chi2(y=y, df_half=df_half), -INF)
    log_pdf_ -= LOG(scale)

    if not normalize:
        log_pdf_ = _log_pdf_scaling(log_pdf_=log_pdf_, amplitude=amplitude)

    return log_pdf_


@suppress_numpy_warnings()
@doc_inherit(parent=chi_square_pdf_, style=doc_style)
def chi_square_cdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    degree_of_freedom: Union[int, float] = 1,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> OneDArray:
    r"""
    Compute CDF for :mod:`~pymultifit.distributions.chiSquare_d.ChiSquareDistribution`.

    Parameters
    ----------
    amplitude: float, optional
        For API consistency only.
    normalize: bool, optional
        For API consistency only.
    Notes
    -----
    The ChiSquare CDF is defined as:

    .. math:: F(y) = \gamma\left(\dfrac{\nu}{2}, \dfrac{y}{2}\right)

    where, :math:`\gamma\left(\cdot, \cdot\right)` is the :obj:`~ssp.gammainc` lower regularized incomplete
    gamma function, :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \dfrac{x - \text{loc}}{\text{scale}}

    The final CDF is expressed as :math:`F(y)`.
    """
    y = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    return np.where(y > 0, ssp.gammainc(degree_of_freedom / 2, y / 2), 0)


@suppress_numpy_warnings()
@doc_inherit(parent=chi_square_cdf_, style=doc_style)
def chi_square_log_cdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    degree_of_freedom: Union[int, float] = 1,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> OneDArray:
    r"""
    Compute log CDF for :mod:`~pymultifit.distributions.chiSquare_d.ChiSquareDistribution`.

    Notes
    -----
    The ChiSquare logCDF is defined as:

    .. math:: \mathcal{L}(y) = \ln\gamma\left(\dfrac{\nu}{2}, \dfrac{y}{2}\right)

    where, :math:`\gamma\left(\cdot, \cdot\right)` is the :obj:`~ssp.gammainc` lower regularized incomplete
    gamma function, :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \dfrac{x - \text{loc}}{\text{scale}}

    The final log CDF is expressed as :math:`\mathcal{L}(y)`.
    """
    y = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    return np.where(y > 0, LOG(ssp.gammainc(degree_of_freedom / 2, y / 2)), -INF)


@suppress_numpy_warnings()
def cubic(x: OneDArray, a: float = 1.0, b: float = 1.0, c: float = 1.0, d: float = 1.0) -> OneDArray:
    r"""
    Computes the y-values of a cubic function given x-values.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.
    a : float
        The coefficient of the cubic term (x^3).
    b : float
        The coefficient of the quadratic term (x^2).
    c : float
        The coefficient of the linear term (x).
    d : float
        The constant term (y-intercept).

    Returns
    -------
    np.ndarray
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The cubic function is defined as:

    .. math:: y = ax^3 + bx^2 + cx + d

    where, :math:`a`, math:`b`, :math:`c`, and :math:`d` are the cubic coefficients.
    """
    return a * x**3 + b * x**2 + c * x + d


@suppress_numpy_warnings()
def exponential_pdf_(
    x: OneDArray, amplitude: float = 1.0, lambda_: float = 1.0, loc: float = 0.0, normalize: bool = False
) -> OneDArray:
    r"""
    Compute PDF for :class:`~pymultifit.distributions.exponential_d.ExponentialDistribution`.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    lambda_ : float, optional
        The scale parameter, :math:`\lambda`.
        Defaults to 1.0.
    loc : float, optional
        The location parameter, for shifting.
        Defaults to 0.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.ndarray
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The Exponential PDF is defined as:

    .. math::
        f(y, \lambda) =
        \begin{cases}
        \exp(-y) &;& y \geq 0, \\
        0 &;& y < 0.
        \end{cases}

    where, :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \dfrac{x - \text{loc}}{\theta}

    and :math:`\theta = \dfrac{1}{\lambda}`. The final PDF is expressed as :math:`f(y)/\theta`.
    """
    rate = 1 / lambda_
    y = preprocess_input(x=x, loc=loc, scale=rate)

    if y.size == 0:
        return y

    pdf_ = np.where(y >= 0, np.exp(-y), 0)
    pdf_ /= rate

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_


@suppress_numpy_warnings()
@doc_inherit(parent=exponential_pdf_, style=doc_style)
def exponential_log_pdf_(
    x: OneDArray, amplitude: float = 1.0, lambda_: float = 1.0, loc: float = 0.0, normalize: bool = False
) -> OneDArray:
    r"""
    Compute log PDF for :class:`~pymultifit.distributions.exponential_d.ExponentialDistribution`.

    Parameters
    ----------
    amplitude : float, optional
        For API consistency only.
    normalize : bool, optional
        For API consistency only.

    Notes
    -----
    The Exponential log PDF is defined as:

    .. math::
        \ell(y, \lambda) =
        \begin{cases}
        - y &;& y \geq 0, \\
        -\infty &;& y < 0.
        \end{cases}

    where, :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \dfrac{x - \text{loc}}{\theta}

    and :math:`\theta = \dfrac{1}{\lambda}`. The final log PDF is expressed as :math:`\ell(y) - \ln(\theta)`.
    """
    rate = 1 / lambda_
    y = preprocess_input(x=x, loc=loc, scale=rate)

    if y.size == 0:
        return y

    log_pdf_ = np.where(y >= 0, -y, -INF)
    log_pdf_ -= LOG(rate)

    if not normalize:
        log_pdf_ = _log_pdf_scaling(log_pdf_=log_pdf_, amplitude=amplitude)

    return log_pdf_


@suppress_numpy_warnings()
@doc_inherit(parent=exponential_pdf_, style=doc_style)
def exponential_cdf_(
    x: OneDArray, amplitude: float = 1.0, lambda_: float = 1.0, loc: float = 0.0, normalize: bool = False
) -> OneDArray:
    r"""
    Compute CDF of :class:`~pymultifit.distributions.exponential_d.ExponentialDistribution`.

    .. note::
        This function uses :obj:`ssp.gammainc` to calculate the CDF with
        :math:`a = 1` and :math:`x = \dfrac{x - \text{loc}}{\theta}`, where :math:`\theta = \dfrac{1}{\lambda}`.

    Parameters
    ----------
    amplitude : float, optional
        For API consistency only.
    normalize : bool, optional
        For API consistency only.

    Notes
    -----
    The Exponential CDF is defined as:

    .. math:: F(y) = 1 - \exp\left[-y\right].

    where, :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \dfrac{x - \text{loc}}{\theta}

    and :math:`\theta = \dfrac{1}{\lambda}`. The final CDF is expressed as :math:`F(y)`.
    """
    y = preprocess_input(x=x, loc=loc, scale=1 / lambda_)

    if y.size == 0:
        return y

    return np.where(y >= 0, -ssp.expm1(-y), 0)


@suppress_numpy_warnings()
@doc_inherit(parent=exponential_cdf_, style=doc_style)
def exponential_log_cdf_(
    x: OneDArray, amplitude: float = 1.0, lambda_: float = 1.0, loc: float = 0.0, normalize: bool = False
) -> OneDArray:
    r"""
    Compute log CDF of :class:`~pymultifit.distributions.exponential_d.ExponentialDistribution`.

    .. note::
        This function uses log transformation of :obj:`ssp.gammainc` to calculate the log CDF with
        :math:`a = 1` and :math:`x = \dfrac{x - \text{loc}}{\theta}`, where :math:`\theta = \dfrac{1}{\lambda}`.

    Notes
    -----
    The Exponential log CDF is defined as:

    .. math:: \mathcal{L}(y) = \ln\left(1 -\exp(y)\right).

    where :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \dfrac{x - \text{loc}}{\theta}.

    and :math:`\theta = \dfrac{1}{\lambda}`. The final log CDF is expressed as :math:`\mathcal{L}(y)`.
    """
    y = preprocess_input(x=x, loc=loc, scale=1 / lambda_)

    if y.size == 0:
        return y

    return np.where(y >= 0, LOG(-ssp.expm1(-y)), -INF)


@suppress_numpy_warnings()
def folded_normal_pdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    mean: float = 0.0,
    sigma: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> OneDArray:
    r"""
    Compute PDF for :class:`~pymultifit.distributions.foldedNormal_d.FoldedNormalDistribution`.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    mean : float, optional
        The mean parameter, :math:`\mu`.
        Defaults to 0.0.
    sigma : float, optional
        The standard deviation parameter, :math:`\sigma`.
        Defaults to 1.0.
    loc : float, optional
        The location parameter, for shifting.
        Defaults to 0.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.ndarray
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The FoldedNormal PDF is defined as:

    .. math:: f(y\ |\ \mu, \sigma) = \phi(y\ |\ \mu, 1) + \phi(y\ |\ -\mu, 1),

    where :math:`\phi` is the PDF of :class:`~pymultifit.distributions.gaussian_d.GaussianDistribution`,
    and :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \dfrac{x - \text{loc}}{\text{scale}}

    The final PDF is expressed as :math:`f(y)/\text{scale}`.
    """
    y = preprocess_input(x=x, loc=loc, scale=sigma)

    if y.size == 0:
        return y

    pdf_ = np.where(
        y >= 0, gaussian_pdf_(y, mean=mean, normalize=True) + gaussian_pdf_(y, mean=-mean, normalize=True), 0
    )
    pdf_ /= sigma

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_


@suppress_numpy_warnings()
@doc_inherit(parent=folded_normal_pdf_, style=doc_style)
def folded_normal_log_pdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    mean: float = 0.0,
    sigma: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
):
    r"""
    Compute log PDF for :class:`~pymultifit.distributions.foldedNormal_d.FoldedNormalDistribution`.

    Notes
    -----
    The FoldedNormal PDF is defined as:

    .. math:: \ell(y\ |\ \mu, \sigma) = \ln\left(\phi(y\ |\ \mu, 1) + \phi(y\ |\ -\mu, 1)\right),

    where :math:`\phi` is the PDF of :class:`~pymultifit.distributions.gaussian_d.GaussianDistribution`,
    and :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \dfrac{x - \text{loc}}{\sigma}

    The final log PDF is expressed as :math:`\ell(y) - \ln(\text{scale})`.
    """
    y = preprocess_input(x=x, loc=loc, scale=sigma)

    if y.size == 0:
        return y

    log_pdf_ = np.where(
        y >= 0, LOG(gaussian_pdf_(y, mean=mean, normalize=True) + gaussian_pdf_(y, mean=-mean, normalize=True)), -INF
    )
    log_pdf_ -= LOG(sigma)

    if not normalize:
        log_pdf_ = _log_pdf_scaling(log_pdf_=log_pdf_, amplitude=amplitude)

    return log_pdf_


@suppress_numpy_warnings()
@doc_inherit(parent=folded_normal_pdf_, style=doc_style)
def folded_normal_cdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    mean: float = 0.0,
    sigma: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> OneDArray:
    r"""
    Compute CDF for :class:`~pymultifit.distributions.foldedNormal_d.FoldedNormalDistribution`.

    Parameters
    ----------
    amplitude : float, optional
        For API consistency only.
    normalize : bool, optional
        For API consistency only.

    Notes
    -----
    The FoldedNormal CDF is defined as:

    .. math:: F(y) = \Phi(y\ | \mu, 1) + \Phi(y\ | -\mu, 1) - 1

    where :math:`\Phi` is the CDF of :class:`~pymultifit.distributions.gaussian_d.GaussianDistribution`,
    and :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \dfrac{x - \text{loc}}{\sigma}.

    The final CDF is expressed as :math:`F(y)`.
    """
    y = preprocess_input(x=x, loc=loc, scale=sigma)

    if y.size == 0:
        return y

    q = (y + mean) / SQRT_TWO
    r = (y - mean) / SQRT_TWO

    return np.where(y >= 0, _folded_cdf(q=q, r=r), 0)


@suppress_numpy_warnings()
@doc_inherit(parent=folded_normal_cdf_, style=doc_style)
def folded_normal_log_cdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    mean: float = 0.0,
    sigma: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> OneDArray:
    r"""
    Compute log CDF for :class:`~pymultifit.distributions.foldedNormal_d.FoldedNormalDistribution`.

    Notes
    -----
    The FoldedNormal log CDF is defined as:

    .. math:: \mathcal{L}(y) = -\ln(2) + \ln\left[\text{erf}\left(\dfrac{q}{\sqrt{2}}\right) +
              \text{erf}\left(\dfrac{r}{\sqrt{2}}\right)\right]

    where :math:`q = y + \mu`, :math:`r = y - \mu`, :math:`\text{erf}` is :obj:`~ssp.erf` function and
    :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \dfrac{x - \text{loc}}{\sigma}.

    The final logCDF is expressed as :math:`\mathcal{L}(y)`.
    """
    y = preprocess_input(x=x, loc=loc, scale=sigma)

    if y.size == 0:
        return y

    q = (y + mean) / SQRT_TWO
    r = (y - mean) / SQRT_TWO

    return np.where(y >= 0, LOG(_folded_cdf(q=q, r=r)), -INF)


@suppress_numpy_warnings()
def _folded(x: OneDArray, mean: float, loc: float, scale: float, g_func: Callable):
    r"""
    Precompute the gaussian part of :class:`~pymultifit.distributions.foldedNormal_d.FoldedNormalDistribution`.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.
    mean : float, optional
        The mean parameter, :math:`\mu`.
        Defaults to 0.0.
    scale : float, optional
        The standard deviation parameter, :math:`\sigma`.
        Defaults to 1.0.
    loc : float, optional
        The location parameter, for shifting.
        Defaults to 0.0.
    g_func : Callable
        The gaussian function, either PDF or CDF.

    Returns
    -------
    np.ndarray
        The additive gaussian part of the folded normal distribution.
    """
    y = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    if scale <= 0 or mean < 0:
        return np.full(shape=x.size, fill_value=np.nan)

    g1 = g_func(x=y, mean=mean, normalize=True)
    g2 = g_func(x=y, mean=-mean, normalize=True)

    return y >= 0, np.where(y >= 0, g1 + g2, 0)


@suppress_numpy_warnings()
def gamma_pdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    alpha: float = 1.0,
    theta: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> OneDArray:
    r"""
    Compute PDF for :class:`~pymultifit.distributions.gamma_d.GammaDistribution`

    Parameters
    ----------
    x : np.ndarray
        Input array of values.
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    alpha : float, optional
        The shape parameter, :math:`\alpha`.
        Defaults to 1.0.
    theta : float, optional
        The scale parameter, :math:`\theta`.
        Defaults to 1.0.
    loc : float, optional
        The location parameter, for shifting.
        Defaults to 0.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.ndarray
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    .. important::

        The Gamma SS PDF is calculated via exponentiation of :func:`gamma_sr_log_pdf_` by setting
        :math:`\lambda = \dfrac{1}{\theta}`.
    """
    y = preprocess_input(x=x, loc=loc, scale=theta)

    if y.size == 0:
        return y

    pdf_ = _gamma(x=y, a=alpha, un_log=True)
    pdf_ /= theta

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_


@suppress_numpy_warnings()
@doc_inherit(parent=gamma_pdf_, style=doc_style)
def gamma_log_pdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    alpha: float = 1.0,
    theta: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> OneDArray:
    r"""
    Compute log PDF for :class:`~pymultifit.distributions.gamma_d.GammaDistribution`

    Notes
    -----
    .. important::

        The Gamma SS log PDF is calculated via :func:`gamma_sr_log_pdf_` by setting :math:`\lambda = \dfrac{1}{\theta}`.
    """
    y = preprocess_input(x=x, loc=loc, scale=theta)

    if y.size == 0:
        return y

    log_pdf_ = _gamma(x=y, a=alpha)
    log_pdf_ -= LOG(theta)

    if not normalize:
        log_pdf_ = _log_pdf_scaling(log_pdf_=log_pdf_, amplitude=amplitude)

    return log_pdf_


def _gamma(x, a, un_log=False):
    value = np.where(x >= 0, ssp.xlogy(a - 1.0, x) - x - ssp.gammaln(a), -INF)
    return np.exp(value) if un_log else value


@suppress_numpy_warnings()
@doc_inherit(parent=gamma_pdf_, style=doc_style)
def gamma_cdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    alpha: float = 1.0,
    theta: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> OneDArray:
    r"""
    Compute CDF for :class:`~pymultifit.distributions.gamma_d.GammaDistribution`.

    Parameters
    ----------
    amplitude: float, optional
        For API consistency only.
    normalize: bool, optional
        For API consistency only.

    Notes
    -----
    .. important::

        The Gamma SS CDF is calculated via :func:`gamma_sr_cdf_` by setting :math:`\lambda = \dfrac{1}{\theta}`.
    """
    y = preprocess_input(x=x, loc=loc, scale=theta)

    if y.size == 0:
        return y

    return np.where(y > 0, ssp.gammainc(alpha, y), 0)


@suppress_numpy_warnings()
@doc_inherit(parent=gamma_cdf_, style=doc_style)
def gamma_log_cdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    alpha: float = 1.0,
    theta: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> OneDArray:
    r"""
    Compute log CDF for :class:`~pymultifit.distributions.gamma_d.GammaDistribution`.

    Notes
    -----
    .. important::

        The Gamma SS log CDF is calculated via logarithm of :func:`gamma_sr_cdf_` by setting
        :math:`\lambda = \dfrac{1}{\theta}`.
    """
    y = preprocess_input(x=x, loc=loc, scale=theta)

    if y.size == 0:
        return y

    return LOG(np.where(y > 0, ssp.gammainc(alpha, y), 0))


@suppress_numpy_warnings()
def gaussian_pdf_(x: OneDArray, amplitude=1.0, mean=0.0, std=1.0, normalize=False) -> OneDArray:
    r"""
    Compute PDF for :class:`~pymultifit.distributions.gaussian_d.GaussianDistribution`

    Parameters
    ----------
    x : np.ndarray
        Input array of values.
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    mean : float, optional
        The mean parameter, :math:`\mu`.
        Defaults to 0.0.
    std : float, optional
        The standard deviation parameter, :math:`\sigma`.
        Defaults to 1.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.ndarray
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The Gaussian PDF is defined as:

    .. math::
        f(x; \mu, \sigma) = \phi\left(\dfrac{x-\mu}{\sigma}\right) =
        \dfrac{1}{\sqrt{2\pi\sigma}}\exp\left[-\dfrac{1}{2}\left(\dfrac{x-\mu}{\sigma}\right)^2\right]

    The final PDF is expressed as :math:`f(x)`.
    """
    y = preprocess_input(x=x, loc=mean, scale=std)

    if y.size == 0:
        return y

    pdf_ = np.exp(-0.5 * y**2) / SQRT_TWO_PI / std

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_


@suppress_numpy_warnings()
@doc_inherit(parent=gaussian_pdf_, style=doc_style)
def gaussian_log_pdf_(
    x: OneDArray, amplitude: float = 1.0, mean: float = 0.0, std: float = 1.0, normalize: bool = False
) -> OneDArray:
    r"""
    Compute log PDF for :class:`~pymultifit.distributions.gaussian_d.GaussianDistribution`

    Notes
    -----
    The Gaussian log PDF is defined as:

    .. math::
        \ell(x; \mu, \sigma) = -\dfrac{1}{2}\ln(2\pi) - \ln\sigma - \dfrac{1}{2}\left(\dfrac{x-\mu}{\sigma}\right)^2

    The final log PDF is expressed as :math:`\ell(x)`.
    """
    y = preprocess_input(x=x, loc=mean, scale=std)

    if y.size == 0:
        return y

    log_pdf_ = -(y**2) / 2.0 - LOG_SQRT_TWO_PI - LOG(std)

    if not normalize:
        log_pdf_ = _log_pdf_scaling(log_pdf_=log_pdf_, amplitude=amplitude)

    return log_pdf_


@suppress_numpy_warnings()
@doc_inherit(parent=gaussian_pdf_, style=doc_style)
def gaussian_cdf_(
    x: OneDArray, amplitude: float = 1.0, mean: float = 0.0, std: float = 1.0, normalize: bool = False
) -> OneDArray:
    r"""
    Compute CDF for :class:`~pymultifit.distributions.gaussian_d.GaussianDistribution`

    .. important::

        The calculation of gaussian CDF is done using :obj:`ssp.ndtr` function.

    Parameters
    ----------
    amplitude: float, optional
        For API consistency only.
    normalize: float, optional
        For API consistency only.

    Notes
    -----
    The Gaussian CDF is defined as:

    .. math::
        F(x) = \Phi\left(\dfrac{x-\mu}{\sigma}\right) =
        \dfrac{1}{2} \left[1 + \text{erf}\left(\dfrac{x - \mu}{\sigma\sqrt{2}}\right)\right]

    The final CDF is expressed as :math:`F(x)`.
    """
    return ssp.ndtr((x - mean) / std)


@suppress_numpy_warnings()
@doc_inherit(parent=gaussian_cdf_, style=doc_style)
def gaussian_log_cdf_(
    x: OneDArray, amplitude: float = 1.0, mean: float = 0.0, std: float = 1.0, normalize: bool = False
) -> OneDArray:
    r"""
    Compute log CDF for :class:`~pymultifit.distributions.gaussian_d.GaussianDistribution`

    .. important::

        The calculation of gaussian log CDF is done using :obj:`ssp.log_ndtr` function.

    Notes
    -----
    The Gaussian log CDF is defined as:

    .. math::
        \mathcal{L}(x) = \ln\Phi\left(\dfrac{x-\mu}{\sigma}\right)

    The final log CDF is expressed as :math:`\mathcal{L}(x)`.
    """
    return ssp.log_ndtr((x - mean) / std)


@suppress_numpy_warnings()
def half_normal_pdf_(
    x: OneDArray, amplitude: float = 1.0, sigma: float = 1.0, loc: float = 0.0, normalize: bool = False
) -> OneDArray:
    r"""
    Compute PDF for the :class:`~pymultifit.distributions.halfNormal_d.HalfNormalDistribution`.

    .. note::
        The :class:`~pymultifit.distributions.halfNormal_d.HalfNormalDistribution` is a special case of the
        :class:`~pymultifit.distributions.foldedNormal_d.FoldedNormalDistribution` with :math:`\mu = 0`.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    sigma : float, optional
        The standard deviation :math:`\sigma`.
        Defaults to 1.0.
    loc : float, optional
        The location parameter, for shifting.
        Defaults to 0.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.ndarray
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The HalfNormal PDF is defined as:

    .. math::
        f(y\ |\ \sigma) = \sqrt{\dfrac{2}{\pi}}\exp\left(-\dfrac{y^2}{2}\right)

    where :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \dfrac{x - \text{loc}}{\text{scale}}.

    The final PDF is expressed as :math:`f(y)/\text{scale}`.
    """
    y = preprocess_input(x=x, loc=loc, scale=sigma)

    if y.size == 0:
        return y

    pdf_ = np.where(y >= 0, SQRT_TWO_BY_PI * np.exp(-0.5 * y**2), 0)
    pdf_ /= sigma

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_


@suppress_numpy_warnings()
@doc_inherit(parent=half_normal_pdf_, style=doc_style)
def half_normal_log_pdf_(
    x: OneDArray, amplitude: float = 1.0, sigma: float = 1.0, loc: float = 0.0, normalize: bool = False
) -> OneDArray:
    r"""
    Compute log PDF for the :class:`~pymultifit.distributions.halfNormal_d.HalfNormalDistribution`.

    Notes
    -----
    The HalfNormal log PDF is defined as:

    .. math:: \ell(y\ |\ \sigma) = \dfrac{1}{2}\ln\left(\dfrac{2}{\pi}\right) - \dfrac{y^2}{2}

    where :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \dfrac{x - \text{loc}}{\text{scale}}.

    The final log PDF is expressed as :math:`\ell(y) - \ln\left(\text{scale}\right)`.
    """
    y = preprocess_input(x=x, loc=loc, scale=sigma)

    if y.size == 0:
        return y

    log_pdf_ = np.where(y >= 0, LOG_SQRT_TWO_BY_PI - 0.5 * y**2, -INF)
    log_pdf_ -= LOG(sigma)

    if not normalize:
        log_pdf_ = _log_pdf_scaling(log_pdf_=log_pdf_, amplitude=amplitude)

    return log_pdf_


@suppress_numpy_warnings()
@doc_inherit(parent=half_normal_pdf_, style=doc_style)
def half_normal_cdf_(
    x: OneDArray, amplitude: float = 1.0, sigma: float = 1.0, loc: float = 0.0, normalize: bool = False
) -> OneDArray:
    r"""
    Compute the CDF for :class:`~pymultifit.distributions.halfNormal_d.HalfNormalDistribution`.

    Parameters
    ----------
    amplitude: float, optional
        For API consistency only.
    normalize: float, optional
        For API consistency only.

    Notes
    -----
    The HalfNormal CDF is defined as:

    .. math:: F(y) = \text{erf}\left(\frac{y}{\sqrt{2}}\right)

    where :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \dfrac{x - \text{loc}}{\text{scale}}.

    The final CDF is expressed as :math:`F(y)`.
    """
    y = preprocess_input(x=x, loc=loc, scale=sigma)

    if y.size == 0:
        return y

    return np.where(y >= 0, ssp.erf(y / SQRT_TWO), 0)


@suppress_numpy_warnings()
@doc_inherit(parent=half_normal_cdf_, style=doc_style)
def half_normal_log_cdf_(
    x: OneDArray, amplitude: float = 1.0, sigma: float = 1.0, loc: float = 0.0, normalize: bool = False
) -> OneDArray:
    r"""
    Compute the log CDF for :class:`~pymultifit.distributions.halfNormal_d.HalfNormalDistribution`.

    Notes
    -----
    The HalfNormal log CDF is defined as:

    .. math:: \mathcal{L}(y) = \ln\text{erf}\left(\frac{y}{\sqrt{2}}\right)

    where :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \dfrac{x - \text{loc}}{\text{scale}}.

    The final log CDF is expressed as :math:`\mathcal{L}(y)`.
    """
    y = preprocess_input(x=x, loc=loc, scale=sigma)

    if y.size == 0:
        return y

    return np.where(y >= 0, LOG(ssp.erf(y / SQRT_TWO)), -INF)


def johnsonSU_pdf(
    x,
    amplitude: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
    xi: float = 0.0,
    lambda_: float = 1.0,
    normalize: bool = False,
):
    y = preprocess_input(x=x, loc=xi, scale=lambda_)

    if y.size == 0:
        return y

    f1 = delta / SQRT_TWO_PI
    f2 = np.sqrt(1 + y**2)
    f3 = np.exp(-0.5 * (gamma + delta * np.arcsinh(y)) ** 2)

    pdf_ = f1 / f2 * f3
    pdf_ /= lambda_

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_


def johnsonSU_log_pdf_(
    x,
    amplitude: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
    xi: float = 0.0,
    lambda_: float = 1.0,
    normalize: bool = False,
):
    y = preprocess_input(x=x, loc=xi, scale=lambda_)

    if y.size == 0:
        return y

    f1 = LOG(delta) - LOG_SQRT_TWO_PI
    f2 = -0.5 * np.log1p(y**2)
    f3 = -0.5 * (gamma + delta * np.arcsinh(y)) ** 2

    log_pdf_ = f1 + f2 + f3
    log_pdf_ -= LOG(lambda_)

    if not normalize:
        log_pdf_ = _log_pdf_scaling(log_pdf_=log_pdf_, amplitude=amplitude)

    return log_pdf_


def johnsonSU_cdf_(
    x,
    amplitude: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
    xi: float = 0.0,
    lambda_: float = 1.0,
    normalize: bool = False,
):
    y = preprocess_input(x=x, loc=xi, scale=lambda_)

    if y.size == 0:
        return y

    y_new = gamma + delta * np.arcsinh(y)
    return gaussian_cdf_(x=y_new, normalize=True)


def johnsonSU_log_cdf_(
    x,
    amplitude: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
    xi: float = 0.0,
    lambda_: float = 1.0,
    normalize: bool = False,
):
    y = preprocess_input(x=x, loc=xi, scale=lambda_)

    if y.size == 0:
        return y

    y_new = gamma + delta * np.arcsinh(y)
    return gaussian_log_cdf_(x=y_new, normalize=True)


@suppress_numpy_warnings()
def laplace_pdf_(
    x: OneDArray, amplitude: float = 1.0, mean: float = 0.0, diversity: float = 1.0, normalize: bool = False
) -> OneDArray:
    r"""
    Compute PDF for the :class:`~pymultifit.distributions.laplace_d.LaplaceDistribution`.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    mean : float, optional
        The mean of laplace distribution.
        Defaults to 0.0.
    diversity : float, optional
        The diversity parameter for laplace distribution.
        Defaults to 1.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.ndarray
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The Laplace PDF is defined as:

    .. math:: f(y\ |\ \mu, b) = \dfrac{1}{2b}\exp\left(-\dfrac{|y|}{b}\right)

    where :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = x - \mu.

    The final PDF is expressed as :math:`f(y)`.
    """
    y = preprocess_input(x=x, loc=mean, scale=diversity)

    if y.size == 0:
        return y

    pdf_ = (1 / 2) * np.exp(-np.abs(y))
    pdf_ /= diversity

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_


@suppress_numpy_warnings()
@doc_inherit(parent=laplace_pdf_, style=doc_style)
def laplace_log_pdf_(
    x: OneDArray, amplitude: float = 1.0, mean: float = 0.0, diversity: float = 1.0, normalize: bool = False
) -> OneDArray:
    r"""
    Compute log PDF for the :class:`~pymultifit.distributions.laplace_d.LaplaceDistribution`.

    Notes
    -----
    The Laplace log PDF is defined as:

    .. math:: \ell(y\ |\ \mu, b) = -\ln(2b) - \dfrac{|y|}{b}

    where :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = x - \mu

    The final log PDF is expressed as :math:`\ell(y)`.
    """
    y = preprocess_input(x=x, loc=mean, scale=diversity)

    if y.size == 0:
        return y

    log_pdf_ = LOG(0.5 * np.exp(-np.abs(y)))
    log_pdf_ -= LOG(diversity)

    if not normalize:
        log_pdf_ = _log_pdf_scaling(log_pdf_=log_pdf_, amplitude=amplitude)

    return log_pdf_


@suppress_numpy_warnings()
@doc_inherit(parent=laplace_pdf_, style=doc_style)
def laplace_cdf_(
    x: OneDArray, amplitude: float = 1.0, mean: float = 0.0, diversity: float = 1.0, normalize: bool = False
) -> OneDArray:
    r"""
    Compute CDF for :class:`~pymultifit.distributions.laplace_d.LaplaceDistribution`.

    Parameters
    ----------
    amplitude: float, optional
        For API consistency only.
    normalize: bool, optional
        For API consistency only.

    Returns
    -------
    np.ndarray
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The Laplace CDF is defined as:

    .. math:: F(x) =
        \begin{cases}
        \dfrac{1}{2}\exp\left(\dfrac{x-\mu}{b}\right) &,&x\leq\mu\\
        1 - \dfrac{1}{2}\exp\left(-\dfrac{x-\mu}{b}\right) &,&x\geq\mu
        \end{cases}

    The final CDF is expressed as :math:`F(x)`.
    """
    y = preprocess_input(x=x, loc=mean, scale=diversity)

    if y.size == 0:
        return y

    return np.where(y > 0, 1.0 - 0.5 * np.exp(-y), 0.5 * np.exp(y))


@suppress_numpy_warnings()
@doc_inherit(parent=laplace_cdf_, style=doc_style)
def laplace_log_cdf_(
    x: OneDArray, amplitude: float = 1.0, mean: float = 0.0, diversity: float = 1.0, normalize: bool = False
) -> OneDArray:
    r"""
    Compute log CDF for :class:`~pymultifit.distributions.laplace_d.LaplaceDistribution`.

    Notes
    -----
    The Laplace log CDF is defined as:

    .. math:: \mathcal{L}(x) =
        \begin{cases}
        -\ln(2) + \dfrac{x-\mu}{b} &,&x\leq\mu\\
        \ln\left[1 - \dfrac{1}{2}\exp\left(-\dfrac{x-\mu}{b}\right)\right] &,&x\geq\mu
        \end{cases}
    """
    y = preprocess_input(x=x, loc=mean, scale=diversity)

    if y.size == 0:
        return y

    return np.where(y > 0, np.log1p(-0.5 * np.exp(-y)), -LOG_TWO + y)


@suppress_numpy_warnings()
def line(x: OneDArray, slope: float = 1.0, intercept: float = 0.0) -> OneDArray:
    r"""
    Computes the y-values of a line given x-values, slope, and intercept.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.
    slope : float
        The slope of the line.
    intercept : float
        The y-intercept of the line.

    Returns
    -------
    np.ndarray
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The line/linear function is defined as:

    .. math:: y = mx + c

    where :math:`m` is the slope and :math:`c` is the intercept of the function.
    """
    return slope * x + intercept


@suppress_numpy_warnings()
def log_normal_pdf_(
    x: OneDArray, amplitude: float = 1.0, mean: float = 1.0, std: float = 1.0, loc: float = 0.0, normalize: bool = False
) -> OneDArray:
    r"""
    Compute PDF for :class:`~pymultifit.distributions.logNormal_d.LogNormalDistribution`.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.
    amplitude : float, optional
        The amplitude of the PDF.
        Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    mean : float, optional
        The mean parameter, :math:`\mu`.
        Defaults to 0.0.
    std : float, optional
        The standard deviation parameter, :math:`\sigma`.
        Defaults to 1.0.
    loc : float, optional
        The location parameter, for shifting.
        Defaults to 0.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.ndarray
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The LogNormal PDF is defined as:

    .. math::
        f(y\ |\ \mu, \sigma) = \dfrac{1}{\sigma y\sqrt{2\pi}}\exp\left(-\dfrac{(\ln y - \mu)^2}{2\sigma^2}\right)

    where, :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = x - \text{loc}

    The final PDF is expressed as :math:`f(y)`.
    """
    y = preprocess_input(x=x, loc=loc)

    if y.size == 0:
        return y

    q = (LOG(y) - mean) / std

    pdf_ = np.where(y > 0, 1 / y / np.exp(q**2 / 2) / SQRT_TWO_PI, 0)
    pdf_ /= std

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_


@suppress_numpy_warnings()
@doc_inherit(parent=log_normal_pdf_, style=doc_style)
def log_normal_log_pdf_(
    x: OneDArray, amplitude: float = 1.0, mean: float = 1.0, std: float = 1.0, loc: float = 0.0, normalize: bool = False
) -> OneDArray:
    r"""
    Compute log PDF for :class:`~pymultifit.distributions.logNormal_d.LogNormalDistribution`.

    Notes
    -----
    The LogNormal log PDF is defined as:

    .. math::
        f(y\ |\ \mu, \sigma) = -\ln(\sigma) -\ln(y) - 0.5\ln(2\pi) -\dfrac{1}{2}\dfrac{(\ln y - \mu)^2}{\sigma^2}

    where, :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = x - \text{loc}

    The final PDF is expressed as :math:`f(y)`.
    """
    y = preprocess_input(x=x, loc=loc)

    if y.size == 0:
        return y

    q = (LOG(y) - mean) / std

    log_pdf_ = np.where(y > 0, -LOG(y) - (q**2 / 2.0) - LOG_SQRT_TWO_PI, -INF)
    log_pdf_ -= LOG(std)

    if not normalize:
        log_pdf_ = _log_pdf_scaling(log_pdf_=log_pdf_, amplitude=amplitude)

    return log_pdf_


@suppress_numpy_warnings()
@doc_inherit(parent=log_normal_pdf_, style=doc_style)
def log_normal_cdf_(
    x: OneDArray, amplitude: float = 1.0, mean: float = 1.0, std=1.0, loc: float = 0.0, normalize: bool = False
) -> OneDArray:
    r"""
    Compute CDF of :class:`~pymultifit.distributions.logNormal_d.LogNormalDistribution`.

    Parameters
    ----------
    amplitude: float, optional
        For API consistency only.
    normalize: bool, optional
        For API consistency only

    Returns
    -------
    np.ndarray
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    .. important::
        The LogNormal CDF is defined as:

        .. math::
            F(x) = \Phi\left(\dfrac{\ln x - \mu}{\sigma}\right)

        which can be calculated via :obj:`ssp.ndtr` function with ``ndtr(y)``, where :math:`y` is the
        transformed value of :math:`x`, defined as:

        .. math:: y = \dfrac{\ln(x - \text{loc}) - \mu}{\sigma}.
    """
    y = preprocess_input(x=x, loc=loc, scale=np.exp(mean))

    return np.where(y > 0, ssp.ndtr(LOG(y) / std), 0)


@suppress_numpy_warnings()
@doc_inherit(parent=log_normal_cdf_, style=doc_style)
def log_normal_log_cdf_(
    x: OneDArray, amplitude: float = 1.0, mean: float = 1.0, std: float = 1.0, loc: float = 0.0, normalize: bool = False
) -> OneDArray:
    r"""
    Compute log CDF of :class:`~pymultifit.distributions.logNormal_d.LogNormalDistribution`.

    Notes
    -----
    .. important::
        The LogNormal log CDF is defined as:

        .. math::
            F(x) = \ln\left[\Phi\left(\dfrac{\ln x - \mu}{\sigma}\right)\right]

        which can be calculated via :obj:`ssp.log_ndtr` function function with ``log_ndtr(y)``, where :math:`y`
        is the transformed value of :math:`x`, defined as:

        .. math:: y = \dfrac{\ln(x - \text{loc}) - \mu}{\sigma}.
    """
    y = preprocess_input(x=x, loc=loc, scale=np.exp(mean))

    return np.where(y > 0, ssp.log_ndtr(LOG(y) / std), -INF)


@suppress_numpy_warnings()
def uniform_pdf_(
    x: OneDArray, amplitude: float = 1.0, low: float = 0.0, high: float = 1.0, normalize: bool = False
) -> OneDArray:
    r"""
    Compute PDF of :class:`~pymultifit.distributions.uniform_d.UniformDistribution`.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.
    amplitude : float, optional
        The amplitude of the PDF.
        Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    low : float, optional
        The lower bound, :math:`a`.
        Defaults to 0.0.
    high : float, optional
        The upper bound, :math:`b`.
        Defaults to 1.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.ndarray
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The Uniform PDF is defined as:

    .. math:: f(x\ |\ a, b) = \dfrac{1}{\beta - a}

    Where :math:`\beta = a + b` consistent with ``loc`` and ``scale`` factors and the final PDF is expressed as,
    :math:`f(x)`.
    """
    x = np.asarray(a=x, dtype=float)

    if x.size == 0:
        return x

    high_ = high + low

    if high_ == low:
        return np.full(shape=x.size, fill_value=np.nan)

    pdf_ = np.where((x >= low) & (x <= high_), 1 / high, 0)

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_


@suppress_numpy_warnings()
def uniform_log_pdf_(
    x: OneDArray, amplitude: float = 1.0, low: float = 0.0, high: float = 1.0, normalize: bool = False
) -> OneDArray:
    r"""
    Compute log PDF of :class:`~pymultifit.distributions.uniform_d.UniformDistribution`.

    Notes
    -----
    The Uniform log PDF is defined as:

    .. math:: \ell(x\ |\ a, b) = -\ln(\beta - a)

    where :math:`\beta = a + b` is consistent with ``loc`` and ``scale`` factors, and the final logPDF is expressed as,
    :math:`\ell(x)`.
    """
    x = np.asarray(a=x, dtype=float)

    if x.size == 0:
        return x

    high_ = high + low

    log_pdf_ = np.where((x >= low) & (x <= high_), -LOG(high), -INF)

    if not normalize:
        log_pdf_ = _log_pdf_scaling(log_pdf_=log_pdf_, amplitude=amplitude)

    return log_pdf_


@suppress_numpy_warnings()
@doc_inherit(parent=uniform_pdf_, style=doc_style)
def uniform_cdf_(
    x: OneDArray, amplitude: float = 1.0, low: float = 0.0, high: float = 1.0, normalize: bool = False
) -> OneDArray:
    r"""
    Compute CDF of :class:`~pymultifit.distributions.uniform_d.UniformDistribution`.

    Parameters
    ----------
    amplitude: float, optional
        For API consistency only.
    normalize: bool, optional
        For API consistency only.

    Notes
    -----
    The Uniform CDF is defined as:

    .. math:: F(x) = \begin{cases}
                        0 &,& x < a\\
                        \dfrac{x-a}{b-a} &,& x \in [a, b]\\
                        1 &,& x > b
                        \end{cases}
    """
    y = preprocess_input(x=x, loc=low, scale=high)

    if y.size == 0:
        return y

    high_ = high + low

    if low == high_ == 0:
        return np.full(shape=y.size, fill_value=np.nan)

    return np.select(condlist=[y < 0, y > 1], choicelist=[0, 1], default=y)


@suppress_numpy_warnings()
@doc_inherit(parent=uniform_cdf_, style=doc_style)
def uniform_log_cdf_(
    x: OneDArray, amplitude: float = 1.0, low: float = 0.0, high: float = 1.0, normalize: bool = False
) -> OneDArray:
    r"""
    Compute log CDF of :class:`~pymultifit.distributions.uniform_d.UniformDistribution`.

    Notes
    -----
    The Uniform log CDF is defined as:

    .. math:: \mathcal{L}(x) = \begin{cases}
                        -\infty &,& x < a\\
                        \ln\left(\dfrac{x-a}{\beta-a}\right) &,& x \in [a, b]\\
                        0 &,& x > \beta
                        \end{cases}

    The final logCDF is expressed as, :math:`\mathcal{L}(x)`.
    """
    y = preprocess_input(x=x, loc=low, scale=high)

    if y.size == 0:
        return y

    high_ = high + low

    if low == high_ == 0:
        return np.full(shape=y.shape, fill_value=-INF)

    return np.select(condlist=[y < 0, y > 1], choicelist=[-INF, 0], default=LOG(y))


@suppress_numpy_warnings()
def scaled_inv_chi_square_pdf_(
    x: OneDArray, amplitude: float = 1.0, df: float = 1.0, scale: float = 1.0, loc: float = 0.0, normalize: bool = False
):
    r"""
    Compute PDF of :class:`~pymultifit.distributions.scaledInvChiSquare_d.ScaledInverseChiSquareDistribution`.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.
    amplitude : float, optional
        The amplitude of the PDF.
        Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    df : float, optional
        The degree of freedom.
        Defaults to 1.0.
    scale: float, optional
        The scale parameter, for scaling.
        Defaults to 1.0,
    loc : float, optional
        The location parameter, for shifting.
        Defaults to 0.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.ndarray
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The Scaled Inverse ChiSquare PDF is defined as:

    .. math:: f(y\ | \nu,\phi) = \dfrac{\tau^2\nu_2}{\Gamma(\nu_2)}\dfrac{1}{y^{1+\nu_2}}\exp\left[-\dfrac{\nu\tau^2}{2y}\right]

    where :math:`\nu_2 = \dfrac{\nu}{2}`, :math:`\tau^2 = \dfrac{\phi}{\nu}` and :math:`y` is the transformed
    value of :math:`x`, defined as:

    .. math:: y = x - \text{loc}

    The final PDF is expressed as :math:`f(y)`.
    """
    y = preprocess_input(x=x, loc=loc)

    if y.size == 0:
        return y

    tau2 = scale / df
    df_half = df / 2

    f1 = np.power(tau2 * df_half, df_half) * ssp.rgamma(df_half)
    f2 = np.exp(-(df * tau2) / (2 * y)) / np.power(y, 1 + df_half)

    pdf_ = np.where(y > 0, f1 * f2, 0)

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_


@suppress_numpy_warnings()
@doc_inherit(parent=scaled_inv_chi_square_pdf_, style=doc_style)
def scaled_inv_chi_square_log_pdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    df: Union[int, float] = 1.0,
    scale: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
):
    r"""
    Compute logPDF of :class:`~pymultifit.distributions.scaledInvChiSquare_d.ScaledInverseChiSquareDistribution`.

    Notes
    -----
    The Scaled Inverse ChiSquare PDF is defined as:

    .. math:: \ell(y) = \ln(\tau^2\nu_2) - \ln\Gamma(\nu_2) - (1+\nu_2)\ln(\nu) - \dfrac{\nu\tau^2}{2y}

    where :math:`\ln` is the natural logarithm, :math:`\ln\Gamma(\cdot)` is the :obj:`~ssp.gammaln` function,
    :math:`\nu_2 = \dfrac{\nu}{2}`, :math:`\tau^2 = \dfrac{\phi}{\nu}` and :math:`y` is the transformed value of
    :math:`x`, defined as:

    .. math:: y = x - \text{loc}

    The final PDF is expressed as :math:`\ell(y)`.
    """
    y = preprocess_input(x=x, loc=loc)

    if y.size == 0:
        return y

    tau2 = scale / df
    df_half = df / 2

    f1 = ssp.xlogy(df_half, tau2 * df_half) - ssp.gammaln(df_half)
    f2 = -(tau2 * df) / (2 * y) - ssp.xlogy(1 + df_half, y)

    log_pdf_ = np.where(y > 0, f1 + f2, -INF)

    if not normalize:
        log_pdf_ = _log_pdf_scaling(log_pdf_=log_pdf_, amplitude=amplitude)

    return log_pdf_


@suppress_numpy_warnings()
@doc_inherit(parent=scaled_inv_chi_square_pdf_, style=doc_style)
def scaled_inv_chi_square_cdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    df: Union[int, float] = 1.0,
    scale: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
):
    r"""
    Compute CDF of :class:`~pymultifit.distributions.scaledInvChiSquare_d.ScaledInverseChiSquareDistribution`.

    Parameters
    ----------
    amplitude : float, optional
        For API consistency only.
    normalize : bool, optional
        For API consistency only.

    Notes
    -----
    The Scaled Inverse ChiSquare CDF is defined as:

    .. math:: F(y) = \Gamma\left(\nu_2, \dfrac{\tau^2\nu_2}{y}\right)

    where :math:`\nu_2 = \dfrac{\nu}{2}`, :math:`\tau^2 = \dfrac{\phi}{\nu}`, :math:`\Gamma(a, b)` is the regularized
    upper gamma function, see :obj:`ssp.gammaincc`,and :math:`y` is the transformed value of :math:`x`,
    defined as:

    .. math:: y = x - \text{loc}

    The final CDF is expressed as :math:`F(y)`.
    """
    y = preprocess_input(x=x, loc=loc)

    if y.size == 0:
        return y

    tau2 = scale / df
    df_half = df / 2

    return np.where(y > 0, ssp.gammaincc(df_half, (tau2 * df_half) / y), 0)


@suppress_numpy_warnings()
@doc_inherit(parent=scaled_inv_chi_square_pdf_, style=doc_style)
def scaled_inv_chi_square_log_cdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    df: Union[int, float] = 1.0,
    scale: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> OneDArray:
    r"""
    Compute log CDF of :class:`~pymultifit.distributions.scaledInvChiSquare_d.ScaledInverseChiSquareDistribution`.

    Notes
    -----
    The Scaled Inverse ChiSquare log CDF is defined as:

    .. math:: \mathcal{L}(y) = \ln\left[\Gamma\left(\nu_2, \dfrac{\tau^2\nu_2}{y}\right)\right]

    where :math:`\nu_2 = \dfrac{\nu}{2}`, :math:`\tau^2 = \dfrac{\phi}{\nu}`, :math:`\Gamma(a, b)` is the regularized
    upper gamma function, see :obj:`ssp.gammaincc`,and :math:`y` is the transformed value of :math:`x`,
    defined as:

    .. math:: y = x - \text{loc}

    The final log CDF is expressed as :math:`\mathcal{L}(y)`.
    """
    y = preprocess_input(x=x, loc=loc)

    if y.size == 0:
        return y

    tau2 = scale / df
    df_half = df / 2

    return np.where(y > 0, LOG(ssp.gammaincc(df_half, (tau2 * df_half) / y)), -INF)


@suppress_numpy_warnings()
def skew_normal_pdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    shape: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> OneDArray:
    r"""
    Compute PDF of :class:`~pymultifit.distributions.skewNormal_d.SkewNormalDistribution`.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.
    amplitude : float, optional
        The amplitude of the PDF.
        Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    shape : float, optional
        The shape parameter, :math:`\alpha`.
        Defaults to 0.0.
    loc : float, optional
        The location parameter, :math:`\xi`.
        Defaults to 0.0.
    scale: float, optional
        The scale parameter, :math:`\omega`
        Defaults to 1.0,
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.ndarray
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The SkewNormal PDF is defined as:

    .. math:: f(y\ |\ \alpha, \xi, \omega) =
             2\phi(y)\Phi(\alpha y)

    where, :math:`\phi(y)` and :math:`\Phi(\alpha y)` are the
    :class:`~pymultifit.distributions.gaussian_d.GaussianDistribution` PDF and CDF defined at :math:`y` and
    :math:`\alpha y` respectively. Additionally, :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \dfrac{x - \xi}{\omega}

    The final PDF is expressed as :math:`f(y)/\omega`.
    """
    y = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    pdf_ = 2 * gaussian_pdf_(x=y, normalize=True) * gaussian_cdf_(x=shape * y, normalize=True)
    pdf_ /= scale

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_


@suppress_numpy_warnings()
@doc_inherit(parent=skew_normal_pdf_, style=doc_style)
def skew_normal_log_pdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    shape: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> OneDArray:
    r"""
    Compute log PDF of :class:`~pymultifit.distributions.skewNormal_d.SkewNormalDistribution`.

    Notes
    -----
    The SkewNormal log PDF is defined as:

    .. math:: \ell(y\ |\ \alpha, \xi, \omega) = \ln(2) + \ln\phi(y) + \ln\Phi(\alpha y)

    where, :math:`\phi(y)` and :math:`\Phi(\alpha y)` are the
    :class:`~pymultifit.distributions.gaussian_d.GaussianDistribution` PDF and CDF defined at :math:`y` and
    :math:`\alpha y` respectively. Additionally, :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \dfrac{x - \xi}{\omega}

    The final log PDF is expressed as :math:`\ell(y)/\omega`.
    """
    y = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    log_pdf_ = LOG_TWO + gaussian_log_pdf_(x=y, normalize=True) + gaussian_log_cdf_(x=shape * y, normalize=True)
    log_pdf_ -= LOG(scale)

    if not normalize:
        log_pdf_ = _log_pdf_scaling(log_pdf_=log_pdf_, amplitude=amplitude)

    return log_pdf_


@suppress_numpy_warnings()
@doc_inherit(parent=skew_normal_pdf_, style=doc_style)
def skew_normal_cdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    shape: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
):
    r"""
    Compute CDF of :class:`~pymultifit.distributions.skewNormal_d.SkewNormalDistribution`.

    Parameters
    ----------
    amplitude: float, optional
        For API consistency only.
    normalize: float, optional
        For API consistency only.

    Notes
    ------
    The SkewNormal CDF is defined as:

    .. math:: F(y) = \Phi(y) - 2T(y, \alpha)

    where, :math:`T` is the Owen's T function, see :obj:`ssp.owens_t`, and
    :math:`\Phi(\cdot)` is the :class:`~pymultifit.distributions.gaussian_d.GaussianDistribution` CDF function, and
    :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \frac{x - \text{loc}}{\text{scale}}

    The final CDF is expressed as :math:`F(y)`.
    """
    y = preprocess_input(x=x, loc=loc, scale=scale)

    return gaussian_cdf_(x=y, normalize=True) - 2 * ssp.owens_t(y, shape)


@suppress_numpy_warnings()
def sym_gen_normal_pdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    shape: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> OneDArray:
    r"""
    Compute PDF of :class:`~pymultifit.distributions.generalized.genNorm_d.SymmetricGeneralizedNormalDistribution`.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.
    amplitude : float, optional
        The amplitude of the PDF.
        Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    shape : float, optional
        The shape parameter, :math:`\beta`.
        Defaults to 1.0.
    loc : float, optional
        The location parameter, :math:`\mu`.
        Defaults to 0.0.
    scale: float, optional
        The scale parameter, :math:`\alpha`
        Defaults to 1.0,
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.ndarray
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The SymmetricGeneralizedNormalDistribution PDF is defined as:

    .. math:: f(y\ |\ \beta, \mu, \alpha) = \dfrac{\beta}{2\Gamma(1/\beta)}\exp\left(-|y|^\beta\right)

    where, :math:`\Gamma` is the :obj:`ssp.gamma` function, and :math:`y` is the transformed value of
    :math:`x`, defined as:

    .. math:: y = \frac{x - \mu}{\alpha}

    The final PDF is expressed as :math:`f(y)/\alpha`.
    """
    y = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    _, _, beta = loc, scale, shape

    log_pdf_ = beta / 2 / ssp.gamma(1 / beta) * np.exp(-np.power(np.abs(y), beta))
    log_pdf_ /= scale

    if not normalize:
        log_pdf_ = _log_pdf_scaling(log_pdf_=log_pdf_, amplitude=amplitude)

    return log_pdf_


@suppress_numpy_warnings()
@doc_inherit(parent=sym_gen_normal_pdf_, style=doc_style)
def sym_gen_normal_log_pdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    shape: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> OneDArray:
    r"""
    Compute log PDF of :class:`~pymultifit.distributions.generalized.genNorm_d.SymmetricGeneralizedNormalDistribution`.

    Notes
    -----
    The SymmetricGeneralizedNormalDistribution log PDF is defined as:

    .. math:: \ell(y\ |\ \beta, \mu, \alpha) = \ln(\beta) - \ln(2) - \ln\Gamma\left(\dfrac{1}{\beta}\right) - |y|^\beta

    where, :math:`\Gamma` is the :obj:`ssp.gamma` function, and :math:`y` is the transformed value of
    :math:`x`, defined as:

    .. math:: y = \frac{x - \mu}{\alpha}

    The final log PDF is expressed as :math:`\ell(y)/\alpha`.
    """
    y = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    _, _, beta = loc, scale, shape

    log_pdf_ = LOG(beta) - LOG_TWO - ssp.gammaln(1 / beta) - np.power(np.abs(y), beta)
    log_pdf_ -= LOG(scale)

    if not normalize:
        log_pdf_ = _log_pdf_scaling(log_pdf_=log_pdf_, amplitude=amplitude)

    return log_pdf_


@suppress_numpy_warnings()
@doc_inherit(parent=sym_gen_normal_pdf_, style=doc_style)
def sym_gen_normal_cdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    shape: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> OneDArray:
    r"""
    Compute CDF of :class:`~pymultifit.distributions.generalized.genNorm_d.SymmetricGeneralizedNormalDistribution`.

    Parameters
    ----------
    amplitude: float, optional
        For API consistency only.
    normalize: bool, optional
        For API consistency only.

    Notes
    -----
    The SymmetricGeneralizedNormalDistribution CDF is defined as:

    .. math:: F(y) = \dfrac{1}{2} + \dfrac{\text{sign}(y)}{2}\gamma\left(\dfrac{1}{\beta},|y|^\beta\,\right)

    where :math:`\gamma(\cdot,\cdot)` is the regularized lower incomplete gamma function, see
    :obj:`~ssp.gammainc`, and :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \frac{x - \text{loc}}{\text{scale}}

    The final CDF is expressed as :math:`F(y)`.
    """
    y = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    _, _, beta = loc, scale, shape

    return 0.5 + np.sign(y) * 0.5 * ssp.gammainc(1 / beta, np.power(np.abs(y), beta))


@suppress_numpy_warnings()
@doc_inherit(parent=sym_gen_normal_cdf_, style=doc_style)
def sym_gen_normal_log_cdf_(
    x: OneDArray,
    amplitude: float = 1.0,
    shape: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> OneDArray:
    r"""
    Compute log CDF of :class:`~pymultifit.distributions.generalized.genNorm_d.SymmetricGeneralizedNormalDistribution`.

    Notes
    -----
    The SymmetricGeneralizedNormalDistribution log CDF is defined as:

    .. math:: \mathcal{L}(y) =
     \ln\left[\dfrac{1}{2} + \dfrac{\text{sign}(y)}{2}\gamma\left(\dfrac{1}{\beta},|y|^\beta\,\right)\right]

    where :math:`\gamma(\cdot,\cdot)` is the lower incomplete gamma function, see :obj:`~ssp.gammainc`, and
    :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \frac{x - \text{loc}}{\text{scale}}

    The final PDF is expressed as :math:`f(y)/\text{scale}`.

    """
    cdf_ = sym_gen_normal_cdf_(x=x, amplitude=amplitude, shape=shape, loc=loc, scale=scale, normalize=normalize)

    return LOG(cdf_)


@suppress_numpy_warnings()
def quadratic(x: OneDArray, a: float = 1.0, b: float = 1.0, c: float = 1.0) -> OneDArray:
    r"""
    Computes the y-values of a quadratic function given x-values.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.
    a : float
        The coefficient of the quadratic term (x^2).
    b : float
        The coefficient of the linear term (x).
    c : float
        The constant term (y-intercept).

    Returns
    -------
    np.ndarray
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The quadratic function is defined as:

    .. math:: y = ax^2 + bx + c

    where, :math:`a`, :math:`b`, and :math:`c` are the quadratic coefficients.
    """
    return a * x**2 + b * x + c


@suppress_numpy_warnings()
def _beta_expr(y: OneDArray, a: float, b: float, un_log: bool = False):
    in_range = (y > 0) & (y < 1)

    undefined_0 = (y == 0) & (a <= 1)
    undefined_1 = (y == 1) & (b <= 1)
    special_case = (y == 1) & (a == 1) & (b == 1)

    expr = ssp.xlog1py(b - 1.0, -y) + ssp.xlogy(a - 1, y) - ssp.betaln(a, b)
    expr2 = np.power(y, a - 1) * np.power(1.0 - y, b - 1.0) / ssp.beta(a, b)

    return [special_case, undefined_0 | undefined_1, in_range], expr2 if un_log else expr


@suppress_numpy_warnings()
def _folded_cdf(q: float, r: float) -> float:
    _f = 0.5 * (ssp.erf(r) + ssp.erf(q))
    return _f.astype(float)


@suppress_numpy_warnings()
def _pdf_scaling(pdf_: OneDArray, amplitude: float) -> OneDArray:
    """Scales a given PDF by a specified amplitude, normalizing it relative to its maximum value.

    Parameters
    ----------
    pdf_ : np.ndarray
        The input probability density function values (not necessarily normalized).
    amplitude : float
        The amplitude factor to scale the normalized PDF.

    Returns
    -------
    NdArray
        The scaled PDF array.
    """
    with np.errstate(all="ignore"):
        return amplitude * (pdf_ / np.max(pdf_))


@suppress_numpy_warnings()
def _log_pdf_scaling(log_pdf_: OneDArray, amplitude: float) -> OneDArray:
    with np.errstate(all="ignore"):
        return LOG(amplitude) + (log_pdf_ - np.max(log_pdf_))


@suppress_numpy_warnings()
def preprocess_input(x: OneDArray, loc: float = 0.0, scale: float = 1.0) -> OneDArray:
    """
    Preprocess the input array, checking for scalar input, handling empty arrays, and loc-scale normalizaing the data.

    Parameters
    ----------
    x : np.ndarray
        Input data.
    loc : float, optional
        The location parameter, for shifting, defaults to 0.0.
    scale: float, optional
        The scale parameter, for scaling, defaults to 1.0,

    Returns
    -------
    np.ndarray
        loc-scale shifted numpy array.
    """
    x = np.asarray(a=x, dtype=float)

    if x.size == 0:
        return np.array([])

    return (x - loc) / scale
