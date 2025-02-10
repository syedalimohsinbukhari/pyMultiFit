"""Created on Aug 03 17:13:21 2024"""

__all__ = ['_beta_masking', '_pdf_scaling', '_remove_nans',
           'arc_sine_pdf_', 'arc_sine_cdf_', 'arc_sine_log_pdf_', 'arc_sine_log_cdf_',
           'beta_pdf_', 'beta_cdf_', 'beta_log_pdf_', 'beta_log_cdf_',
           'chi_square_pdf_', 'chi_square_cdf_', 'chi_square_log_pdf_', 'chi_square_log_cdf_',
           'exponential_pdf_', 'exponential_cdf_', 'exponential_log_pdf_', 'exponential_log_cdf_',
           'folded_normal_pdf_', 'folded_normal_cdf_', 'folded_normal_log_pdf_', 'folded_normal_log_cdf_',
           'gamma_sr_pdf_', 'gamma_sr_cdf_', 'gamma_sr_log_pdf_', 'gamma_sr_log_cdf_',
           'gamma_ss_pdf_', 'gamma_ss_log_pdf_', 'gamma_ss_cdf_', 'gamma_ss_log_cdf_',
           'sym_gen_normal_pdf_', 'sym_gen_normal_cdf_',
           'gaussian_pdf_', 'gaussian_cdf_', 'gaussian_log_pdf_', 'gaussian_log_cdf_',
           'half_normal_pdf_', 'half_normal_cdf_', 'half_normal_log_pdf_', 'half_normal_log_cdf_',
           'laplace_pdf_', 'laplace_cdf_', 'laplace_log_pdf_', 'laplace_log_cdf_',
           'log_normal_pdf_', 'log_normal_cdf_', 'log_normal_log_pdf_', 'log_normal_log_cdf_',
           'scaled_inv_chi_square_pdf_', 'scaled_inv_chi_square_log_pdf_',
           'scaled_inv_chi_square_cdf_', 'scaled_inv_chi_square_log_cdf_',
           'skew_normal_pdf_', 'skew_normal_cdf_',
           'uniform_pdf_', 'uniform_cdf_', 'uniform_log_pdf_', 'uniform_log_cdf_']

from typing import Union, Callable

import numpy as np
from custom_inherit import doc_inherit
from scipy.special import betainc, erf, gamma, gammainc, gammaln, owens_t, gammaincc, log_ndtr, ndtr

from .. import fArray, doc_style


def arc_sine_pdf_(x: fArray,
                  amplitude: float = 1.0,
                  loc: float = 0.0, scale: float = 1.0, normalize: bool = False) -> fArray:
    r"""
    Compute PDF of :class:`~pymultifit.distributions.arcSine_d.ArcSineDistribution`.

    Parameters
    ----------
    x : fArray
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
    y, scalar_input = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    mask_ = np.logical_and(y > 0, y < 1)

    pdf_ = np.zeros_like(a=y, dtype=float)
    pdf_[mask_] = 1 / (np.pi * np.sqrt(y[mask_] * (1 - y[mask_])))
    pdf_ = _remove_nans(pdf_) / scale

    pdf_[y == 0] = np.inf
    pdf_[y == 1] = np.inf

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_.item() if scalar_input else pdf_


@doc_inherit(parent=arc_sine_pdf_, style=doc_style)
def arc_sine_log_pdf_(x: fArray,
                      amplitude: float = 1.0,
                      loc: float = 0.0, scale: float = 1.0, normalize: bool = False):
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
    y, scalar_input = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    mask_ = np.logical_and(y > 0, y < 1)

    log_pdf_ = np.full(shape=y.shape, fill_value=-np.inf)
    log_pdf_[mask_] = -np.log(np.pi) - 0.5 * np.log(y[mask_] - y[mask_]**2)
    log_pdf_ -= np.log(scale)

    log_pdf_[y == 0] = np.inf
    log_pdf_[y == 1] = np.inf

    if not normalize:
        log_pdf_ = _log_pdf_scaling(log_pdf_=log_pdf_, amplitude=amplitude)

    return log_pdf_.item() if scalar_input else log_pdf_


@doc_inherit(parent=arc_sine_pdf_, style=doc_style)
def arc_sine_cdf_(x: fArray,
                  amplitude: float = 1.0,
                  loc: float = 0.0, scale: float = 1.0, normalize: bool = False) -> fArray:
    r"""
    Compute CDF of :class:`~pymultifit.distributions.arcSine_d.ArcSineDistribution`.

    Parameters
    ----------
    amplitude: float, optional
        For API consistency only.
    normalize: bool, optional
        For API consistency only.

    Notes
    -----
    The ArcSine CDF is defined as:

    .. math:: F(y) = \left(\frac{2}{\pi}\right)\arcsin(\sqrt{y})

    where :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \dfrac{x - \text{loc}}{\text{scale}}.
    """
    y, scalar_input = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    mask_ = np.logical_and(y > 0, y < 1)

    cdf_ = np.where(y < 1, 0, 1).astype(float)
    cdf_[mask_] = (2 / np.pi) * np.arcsin(np.sqrt(y[mask_]))

    return cdf_.item() if scalar_input else cdf_


@doc_inherit(parent=arc_sine_cdf_, style=doc_style)
def arc_sine_log_cdf_(x: fArray,
                      amplitude: float = 1.0,
                      loc: float = 0.0, scale: float = 1.0, normalize: bool = False) -> fArray:
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
    y, scalar_input = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    mask_ = np.logical_and(y > 0, y < 1)

    log_cdf_ = np.where(y < 1, -np.inf, 0)
    log_cdf_[mask_] = np.log(2 / np.pi) + np.log(np.arcsin(np.sqrt(y[mask_])))

    return log_cdf_.item() if scalar_input else log_cdf_


def beta_pdf_(x: fArray,
              amplitude: float = 1.0, alpha: float = 1.0, beta: float = 1.0,
              loc: float = 0.0, scale: float = 1.0, normalize: bool = False) -> fArray:
    r"""
    Compute PDF of :class:`~pymultifit.distributions.beta_d.BetaDistribution`.

    Parameters
    ----------
    x : fArray
        Input array of values where PDF is evaluated.
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    alpha : float, optional
        The :math:`\alpha` parameter.
        Default is 1.0.
    beta : float, optional
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

    where :math:`B(\alpha, \beta)` is the Beta function (see, :obj:`scipy.special.beta`), and :math:`y` is the
    transformed value of :math:`x` such that:

    .. math:: y = \frac{x - \text{loc}}{\text{scale}}

    The final PDF is expressed as :math:`f(y)/\text{scale}`.
    """
    # evaluating log_pdf_ for Beta distribution is safer than evaluating direct pdf_ due to less over/under flow issues
    log_pdf_ = beta_log_pdf_(x,
                             amplitude=amplitude, alpha=alpha, beta=beta,
                             loc=loc, scale=scale, normalize=normalize)

    return _remove_nans(x=np.exp(log_pdf_), nan_value=-np.inf)


@doc_inherit(parent=beta_pdf_, style=doc_style)
def beta_log_pdf_(x: fArray,
                  amplitude: float = 1.0, alpha: float = 1.0, beta: float = 1.0,
                  loc: float = 0.0, scale: float = 1.0, normalize: bool = False) -> fArray:
    r"""Compute logPDF for :class:`~pymultifit.distributions.beta_d.BetaDistribution`.

    Notes
    -----
    The Beta logPDFis defined as

    .. math:: \ell(y) = (\alpha - 1)\ln(y) + (\beta - 1)\ln(1 - y) - \ln(\text{Beta}(\alpha, \beta))

    where :math:`B(\alpha, \beta)` is the :obj:`~scipy.special.beta` function, and :math:`y` is the
    transformed value of :math:`x` such that:

    .. math:: y = \frac{x - \text{loc}}{\text{scale}}

    The final logPDF is expressed as :math:`\ell(y) - \ln(\text{scale})`.
    """
    y, scalar_input = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    log_pdf_ = np.full_like(a=y, fill_value=-np.inf, dtype=float)
    mask_ = ~_beta_masking(y=y, alpha=alpha, beta=beta)

    normalization_factor = gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)

    log_pdf_[mask_] = (alpha - 1) * np.log(y[mask_]) + (beta - 1) * np.log(1 - y[mask_])
    log_pdf_[mask_] = log_pdf_[mask_] - normalization_factor
    log_pdf_ -= np.log(scale)

    if not normalize:
        log_pdf_ = _log_pdf_scaling(log_pdf_=log_pdf_, amplitude=amplitude)

    if alpha <= 1:
        log_pdf_[y == 0] = np.nan
    if beta <= 1:
        log_pdf_[y == 1] = np.nan
    if alpha == 1 and beta == 1:
        log_pdf_[y == 1] = 0

    return log_pdf_.item() if scalar_input else log_pdf_


@doc_inherit(parent=beta_pdf_, style=doc_style)
def beta_cdf_(x: fArray,
              amplitude: float = 1.0, alpha: float = 1.0, beta: float = 1.0,
              loc: float = 0.0, scale: float = 1.0, normalize: bool = False) -> fArray:
    r"""
    Compute CDF for :class:`~pymultifit.distributions.beta_d.BetaDistribution`.

    Parameters
    ----------
    x : fArray
        Input array of values.
    amplitude : float, optional
        For API consistency only.
    normalize : bool, optional
        For API consistency only.

    Notes
    -----
    The Beta CDF is defined as:

    .. math:: F(y) = I_y(\alpha, \beta)

    where :math:`I_y(\alpha, \beta)` is the :obj:`~scipy.special.betainc` function, and :math:`y` is the transformed
    value of :math:`x`, defined as:

    .. math:: y = \frac{x - \text{loc}}{\text{scale}}

    The final CDF is expressed as :math:`F(y)`.
    """
    y, scalar_input = preprocess_input(x=x, loc=loc, scale=scale)
    if y.size == 0:
        return y

    if scale < 0:
        return np.full(shape=x.shape, fill_value=np.nan)

    mask_ = np.logical_and(y > 0, y < 1)

    cdf_ = np.where(y < 1, 0, 1).astype(float)
    cdf_[mask_] = betainc(alpha, beta, y[mask_])

    return cdf_.item() if scalar_input else cdf_


@doc_inherit(parent=beta_cdf_, style=doc_style)
def beta_log_cdf_(x: fArray,
                  amplitude: float = 1.0, alpha: float = 1.0, beta: float = 1.0,
                  loc: float = 0.0, scale: float = 1.0, normalize: bool = False) -> fArray:
    r"""
    Compute logCDF for :class:`~pymultifit.distributions.beta_d.BetaDistribution`.

    Notes
    -----
    The Beta logCDF is defined as:

    .. math:: \mathcal{L}(y) = \ln I_y(\alpha, \beta)

    where :math:`I_y(\alpha, \beta)` is the :obj:`~scipy.special.betainc` function, and :math:`y` is the transformed
    value of :math:`x`, defined as:

    .. math:: y = \frac{x - \text{loc}}{\text{scale}}

    The final logCDF is expressed as :math:`\mathcal{L}(y)`.
    """
    # calculating cdf requires a special function that doesn't have corresponding log-function in scipy,
    # so it's cheaper to go from cdf_ -> log_cdf_ rather than recalculating log_cdf_ using same method.
    cdf_ = beta_cdf_(x,
                     amplitude=amplitude, alpha=alpha, beta=beta,
                     loc=loc, scale=scale, normalize=normalize)

    return np.log(cdf_)


def chi_square_pdf_(x: fArray,
                    amplitude: float = 1.0, degree_of_freedom: Union[int, float] = 1,
                    loc: float = 0.0, scale: float = 1.0, normalize: bool = False) -> fArray:
    r"""
    Compute PDF for :mod:`~pymultifit.distributions.chiSquare_d.ChiSquareDistribution`.

    Parameters
    ----------
    x : fArray
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

    where :math:`\Gamma(\cdot)` is the :obj:`~scipy.special.gamma` function, and :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \dfrac{x - \text{loc}}{\text{scale}}

    The final PDF is expressed as :math:`f(y)/\text{scale}`.
    """
    # evaluating log_pdf_ for chi2 distribution is safer than evaluating direct pdf_ due to less over/under flow issues
    log_pdf_ = chi_square_log_pdf_(x,
                                   amplitude=amplitude, degree_of_freedom=degree_of_freedom,
                                   loc=loc, scale=scale, normalize=normalize)
    return np.exp(log_pdf_)


@doc_inherit(parent=chi_square_pdf_, style=doc_style)
def chi_square_log_pdf_(x: fArray,
                        amplitude: float = 1.0, degree_of_freedom: Union[int, float] = 1,
                        loc: float = 0.0, scale: float = 1.0, normalize: bool = False) -> fArray:
    r"""
    Compute log PDF for :mod:`~pymultifit.distributions.chiSquare_d.ChiSquareDistribution`.

    Notes
    -----
    The ChiSquare log PDF is defined as:

    .. math:: \ell(y\ |\ k) = \left(\dfrac{k}{2} - 1\right)\ln(y) - \dfrac{y}{2} - \dfrac{k}{2}\ln(2) - \ln\Gamma\left(\dfrac{k}{2}\right)

    where :math:`\ln\Gamma(\cdot)` is the :obj:`~scipy.special.gammaln` function,
    and :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \dfrac{x - \text{loc}}{\text{scale}}

    The final PDF is expressed as :math:`\ell(y) - \ln(\text{scale})`.

    """
    y, scalar_input = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    mask_ = y > 0

    log_pdf_ = np.full(shape=y.shape, fill_value=-np.inf)
    df_half = degree_of_freedom / 2
    log_pdf_[mask_] = (df_half - 1) * np.log(y[mask_]) - (y[mask_] / 2) - (df_half * np.log(2)) - gammaln(df_half)
    log_pdf_ -= np.log(scale)

    if not normalize:
        log_pdf_ = _log_pdf_scaling(log_pdf_=log_pdf_, amplitude=amplitude)

    return log_pdf_.item() if scalar_input else log_pdf_


@doc_inherit(parent=chi_square_pdf_, style=doc_style)
def chi_square_cdf_(x: fArray,
                    amplitude: float = 1.0, degree_of_freedom: Union[int, float] = 1,
                    loc: float = 0.0, scale: float = 1.0, normalize: bool = False) -> fArray:
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

    where, :math:`\gamma\left(\cdot, \cdot\right)` is the :obj:`~scipy.special.gammainc` lower regularized incomplete
    gamma function, :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \dfrac{x - \text{loc}}{\text{scale}}

    The final CDF is expressed as :math:`F(y)`.
    """
    y, scalar_input = preprocess_input(x=x, loc=loc, scale=scale)

    if y.size == 0:
        return y

    mask_ = y >= 0

    cdf_ = np.zeros(shape=y.shape, dtype=float)
    cdf_[mask_] = gammainc(degree_of_freedom / 2, y[mask_] / 2)

    return cdf_.item() if scalar_input else cdf_


@doc_inherit(parent=chi_square_cdf_, style=doc_style)
def chi_square_log_cdf_(x: fArray,
                        amplitude: float = 1.0, degree_of_freedom: Union[int, float] = 1,
                        loc: float = 0.0, scale: float = 1.0, normalize: bool = False) -> fArray:
    r"""
    Compute log CDF for :mod:`~pymultifit.distributions.chiSquare_d.ChiSquareDistribution`.

    Notes
    -----
    The ChiSquare logCDF is defined as:

    .. math:: \mathcal{L}(y) = \ln\gamma\left(\dfrac{\nu}{2}, \dfrac{y}{2}\right)

    where, :math:`\gamma\left(\cdot, \cdot\right)` is the :obj:`~scipy.special.gammainc` lower regularized incomplete
    gamma function, :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \dfrac{x - \text{loc}}{\text{scale}}

    The final log CDF is expressed as :math:`\mathcal{L}(y)`.
    """
    cdf_ = chi_square_cdf_(x,
                           amplitude=amplitude, degree_of_freedom=degree_of_freedom,
                           loc=loc, scale=scale, normalize=normalize)

    return np.log(cdf_)


def exponential_pdf_(x: fArray,
                     amplitude: float = 1.0, lambda_: float = 1.0, loc: float = 0.0,
                     normalize: bool = False) -> fArray:
    r"""
    Compute PDF for :class:`~pymultifit.distributions.exponential_d.ExponentialDistribution`.

    Parameters
    ----------
    x : fArray
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
        \lambda \exp\left[-\lambda y\right] &;& y \geq 0, \\
        0 &;& y < 0.
        \end{cases}

    where, :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = x - \text{loc}

    The final PDF is expressed as :math:`f(y)`.
    """
    y, scalar_input = preprocess_input(x=x, loc=loc)

    if y.size == 0:
        return y

    mask_ = y >= 0

    pdf_ = np.zeros(shape=y.shape, dtype=float)
    pdf_[mask_] = lambda_ * np.exp(-lambda_ * y[mask_])

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_.item() if scalar_input else pdf_


@doc_inherit(parent=exponential_pdf_, style=doc_style)
def exponential_log_pdf_(x: fArray,
                         amplitude: float = 1.0, lambda_: float = 1.0, loc: float = 0.0,
                         normalize: bool = False) -> fArray:
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
        \ln\lambda -\lambda y &;& y \geq 0, \\
        -inf &;& y < 0.
        \end{cases}

    where, :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = x - \text{loc}

    The final log PDF is expressed as :math:`\ell(y)`.
    """
    y, scalar_input = preprocess_input(x=x, loc=loc)

    if y.size == 0:
        return y

    mask_ = y >= 0

    log_pdf_ = np.full(shape=y.shape, fill_value=-np.inf)
    log_pdf_[mask_] = np.log(lambda_) - (lambda_ * y[mask_])

    if not normalize:
        log_pdf_ = _log_pdf_scaling(log_pdf_=log_pdf_, amplitude=amplitude)

    return log_pdf_.item() if scalar_input else log_pdf_


@doc_inherit(parent=exponential_pdf_, style=doc_style)
def exponential_cdf_(x: fArray,
                     amplitude: float = 1., lambda_: float = 1., loc: float = 0.0,
                     normalize: bool = False) -> fArray:
    r"""
    Compute CDF of :class:`~pymultifit.distributions.exponential_d.ExponentialDistribution`.

    .. note::
        This function uses :func:`~pymultifit.distributions.utilities_d.gamma_sr_cdf_` to calculate the CDF with
        :math:`\alpha = 1` and :math:`\lambda_\text{gammaSR} = \lambda_\text{expon}`.

    Parameters
    ----------
    amplitude : float, optional
        For API consistency only.
    normalize : bool, optional
        For API consistency only.

    Notes
    -----
    The Exponential CDF is defined as:

    .. math:: F(x) = 1 - \exp\left[-\lambda x\right].
    """
    y, scalar_input = preprocess_input(x=x, loc=loc)

    if y.size == 0:
        return y

    mask_ = y > 0

    cdf_ = np.zeros_like(a=y, dtype=float)
    cdf_[mask_] = 1 - np.exp(-lambda_ * y[mask_])

    return cdf_.item() if scalar_input else cdf_


@doc_inherit(parent=exponential_cdf_, style=doc_style)
def exponential_log_cdf_(x: fArray,
                         amplitude: float = 1., lambda_: float = 1.0, loc: float = 0.0,
                         normalize: bool = False) -> fArray:
    r"""
    Compute log CDF of :class:`~pymultifit.distributions.exponential_d.ExponentialDistribution`.

    Notes
    -----
    The Exponential log CDF is defined as:

    .. math:: \mathcal{L}(y) = \ln\left(1 -\exp(\lambda y)\right).

    where :math:`\ln(1-\theta)` is calculated using :obj:`~numpy.log1p` function, and :math:`y` is the transformed value
    of :math:`x`, defined as:

    .. math:: y = x - \text{loc}.

    The final log CDF is expressed as :math:`\mathcal{L}(y)`.
    """
    y, scalar_input = preprocess_input(x=x, loc=loc)

    mask_ = y >= 0

    log_cdf_ = np.full(shape=y.shape, fill_value=-np.inf)
    log_cdf_[mask_] = np.log1p(-np.exp(-lambda_ * y[mask_]))

    return log_cdf_.item() if scalar_input else log_cdf_


def folded_normal_pdf_(x: fArray,
                       amplitude: float = 1., mean: float = 0.0, sigma: float = 1.0,
                       loc: float = 0.0, normalize: bool = False) -> fArray:
    r"""
    Compute PDF for :class:`~pymultifit.distributions.foldedNormal_d.FoldedNormalDistribution`.

    Parameters
    ----------
    x : fArray
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

    .. math:: y = \dfrac{x - \text{loc}}{\sigma}

    The final PDF is expressed as :math:`f(y)/\sigma`.
    """
    x, scalar_input = preprocess_input(x=x)

    if x.size == 0:
        return x

    _, pdf_ = _folded(x=x, mean=mean, sigma=sigma, loc=loc, g_func=gaussian_pdf_)
    pdf_ = _remove_nans(pdf_) / sigma

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_.item() if scalar_input else pdf_


@doc_inherit(parent=folded_normal_pdf_, style=doc_style)
def folded_normal_log_pdf_(x: fArray,
                           amplitude: float = 1.0, mean: float = 0.0, sigma: float = 1.0,
                           loc: float = 0.0, normalize: bool = False):
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
    x, scalar_input = preprocess_input(x=x)

    if x.size == 0:
        return x

    _, pdf_ = _folded(x=x, mean=mean, sigma=sigma, loc=loc, g_func=gaussian_pdf_)
    log_pdf_ = np.log(pdf_) - np.log(sigma)

    if not normalize:
        log_pdf_ = _log_pdf_scaling(log_pdf_=log_pdf_, amplitude=amplitude)

    return log_pdf_.item() if scalar_input else log_pdf_


@doc_inherit(parent=folded_normal_pdf_, style=doc_style)
def folded_normal_cdf_(x: fArray,
                       amplitude: float = 1., mean: float = 0.0, sigma: float = 1.0,
                       loc: float = 0.0, normalize: bool = False) -> fArray:
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
    x, scalar_input = preprocess_input(x=x)

    if x.size == 0:
        return x

    mask_, cdf_ = _folded(x=x, mean=mean, sigma=sigma, loc=loc, g_func=gaussian_cdf_)
    cdf_[mask_] -= 1

    return cdf_.item() if scalar_input else cdf_


@doc_inherit(parent=folded_normal_cdf_, style=doc_style)
def folded_normal_log_cdf_(x: fArray,
                           amplitude: float = 1.0, mean: float = 0.0, sigma: float = 1.0,
                           loc: float = 0.0, normalize: bool = False) -> fArray:
    r"""
    Compute log CDF for :class:`~pymultifit.distributions.foldedNormal_d.FoldedNormalDistribution`.

    Notes
    -----
    The FoldedNormal log CDF is defined as:

    .. math:: \mathcal{L}(y) = -\ln(2) + \ln\left[\text{erf}\left(\dfrac{q}{\sqrt{2}}\right) +
              \text{erf}\left(\dfrac{r}{\sqrt{2}}\right)\right]

    where :math:`q = y + \mu`, :math:`r = y - \mu`, :math:`\text{erf}` is :obj:`~scipy.special.erf` function and
    :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \dfrac{x - \text{loc}}{\sigma}.

    The final logCDF is expressed as :math:`\mathcal{L}(y)`.
    """
    y, scalar_input = preprocess_input(x=x, loc=loc, scale=sigma)

    if y.size == 0:
        return y

    mask_ = y >= 0

    log_cdf_ = np.full(shape=y.shape, fill_value=-np.inf)

    if np.any(mask_):
        y_valid = y[mask_]
        q = y_valid + mean
        r = y_valid - mean
        log_cdf_[mask_] = -np.log(2) + np.log(erf(q / np.sqrt(2)) + erf(r / np.sqrt(2)))

    return log_cdf_.item() if scalar_input else log_cdf_


def _folded(x: fArray, mean: float, sigma: float, loc: float, g_func: Callable):
    r"""
    Precompute the gaussian part of :class:`~pymultifit.distributions.foldedNormal_d.FoldedNormalDistribution`.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.
    mean : float, optional
        The mean parameter, :math:`\mu`.
        Defaults to 0.0.
    sigma : float, optional
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
    if sigma <= 0 or mean < 0:
        return np.full(shape=x.size, fill_value=np.nan)

    y = (x - loc) / sigma
    distribution_ = np.zeros_like(a=y, dtype=float)

    mask = y >= 0
    g1 = g_func(x=y[mask], mean=mean, normalize=True)
    g2 = g_func(x=y[mask], mean=-mean, normalize=True)
    distribution_[mask] = g1 + g2

    return mask, distribution_


def gamma_sr_pdf_(x: fArray,
                  amplitude: float = 1.0, alpha: float = 1.0, lambda_: float = 1.0,
                  loc: float = 0.0, normalize: bool = False) -> fArray:
    r"""
    Compute PDF for :class:`~pymultifit.distributions.gamma_d.GammaDistributionSR`.

    Parameters
    ----------
    x : fArray
        Input array of values.
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    alpha : float, optional
        The shape parameter, :math:`\alpha`.
        Defaults to 1.0.
    lambda_ : float, optional
        The rate parameter, :math:`\lambda`.
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
    The Gamma SR PDF is defined as:

    .. math::
        f(y; \alpha, \lambda) =
        \begin{cases}
        \dfrac{\lambda^\alpha}{\Gamma(\alpha)} y^{\alpha - 1} \exp\left[-\lambda y\right] &,& y > 0, \\
         & \\
        0 &,& y < 0.
        \end{cases}

    where :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = x - \text{loc}

    The final PDF is expressed as :math:`f(y)`.
    """
    # evaluating log_pdf_ for Gamma distribution is safer than evaluating direct pdf_ due to less over/under flow issues
    log_pdf_ = gamma_sr_log_pdf_(x,
                                 amplitude=amplitude, alpha=alpha, lambda_=lambda_,
                                 loc=loc, normalize=normalize)
    return np.exp(log_pdf_)


@doc_inherit(parent=gamma_sr_pdf_, style=doc_style)
def gamma_sr_log_pdf_(x: fArray,
                      amplitude: float = 1.0, alpha: float = 1.0, lambda_: float = 1.0,
                      loc: float = 0.0, normalize: bool = False) -> fArray:
    r"""
    Compute log PDF for :class:`~pymultifit.distributions.gamma_d.GammaDistributionSR`.

    Notes
    -----
    The Gamma SR log PDF is defined as:

    .. math::
        \ell(y; \alpha, \lambda) =
        \begin{cases}
        \alpha\ln\lambda + (\alpha - 1)\ln(y) - \lambda y - \ln\Gamma(\alpha) &,& y > 0, \\
         & \\
        -\infty &,& y < 0.
        \end{cases}

    where :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = x - \text{loc}

    The final PDF is expressed as :math:`\ell(y)`.
    """
    y, scalar_input = preprocess_input(x=x, loc=loc)

    if y.size == 0:
        return y

    mask_ = y >= 0

    log_pdf_ = np.full(shape=y.shape, fill_value=-np.inf)
    log_pdf_[mask_] = alpha * np.log(lambda_) + (alpha - 1) * np.log(y[mask_]) - (lambda_ * y[mask_]) - gammaln(alpha)

    if not normalize:
        log_pdf_ = _log_pdf_scaling(log_pdf_=log_pdf_, amplitude=amplitude)

    return log_pdf_.item() if scalar_input else log_pdf_


@doc_inherit(parent=gamma_sr_pdf_, style=doc_style)
def gamma_sr_cdf_(x: fArray,
                  amplitude: float = 1.0, alpha: float = 1.0, lambda_: float = 1.0,
                  loc: float = 0.0, normalize: bool = False) -> fArray:
    r"""
    Compute CDF for :class:`~pymultifit.distributions.gamma_d.GammaDistributionSR`.

    Parameters
    ----------
    amplitude: float, optional
        For API consistency only.
    normalize: float, optional
        For API consistency only.

    Notes
    -----
    The Gamma SR CDF is defined as:

    .. math:: F(y) = \dfrac{1}{\Gamma(\alpha)}\gamma(\alpha, \lambda y)

    where, :math:`\dfrac{\gamma(a, b)}{\Gamma(a)}` is the regularized lower incomplete gamma function,
    see :obj:`~scipy.special.gammainc`, and :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = x - \text{loc}

    The final CDF is expressed as :math:`F(y)`.
    """
    y, scalar_input = preprocess_input(x=x, loc=loc)

    if y.size == 0:
        return y

    cdf_ = gammainc(alpha, lambda_ * np.maximum(y, 0))

    return cdf_.item() if scalar_input else cdf_


@doc_inherit(parent=gamma_sr_cdf_, style=doc_style)
def gamma_sr_log_cdf_(x: fArray,
                      amplitude: float = 1.0, alpha: float = 1.0, lambda_: float = 1.0,
                      loc: float = 0.0, normalize: bool = False) -> fArray:
    r"""
    Compute log CDF for :class:`~pymultifit.distributions.gamma_d.GammaDistributionSR`.

    Notes
    -----
    The Gamma SR log CDF is defined as:

    .. math:: \mathcal{L}(y) = -\ln\Gamma(\alpha) + \ln\gamma(\alpha, \lambda y)

    where, :math:`-\ln\Gamma(a) + \ln\gamma(a, b)` is the logarithm of regularized lower incomplete gamma function,
    see :obj:`~scipy.special.gammainc`, and :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = x - \text{loc}

    The final log CDF is expressed as :math:`\mathcal{L}(y)`.
    """
    cdf_ = gamma_sr_cdf_(x,
                         amplitude=amplitude, alpha=alpha, lambda_=lambda_,
                         loc=loc, normalize=normalize)

    return np.log(cdf_)


def gamma_ss_pdf_(x: fArray,
                  amplitude: float = 1.0, alpha: float = 1.0, theta: float = 1.0,
                  loc: float = 0.0, normalize: bool = False) -> fArray:
    r"""
    Compute PDF for :class:`~pymultifit.distributions.gamma_d.GammaDistributionSS`

    Parameters
    ----------
    x : fArray
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
    log_pdf_ = gamma_sr_log_pdf_(x,
                                 amplitude=amplitude, alpha=alpha, lambda_=1 / theta,
                                 loc=loc, normalize=normalize)
    return np.exp(log_pdf_)


@doc_inherit(parent=gamma_ss_pdf_, style=doc_style)
def gamma_ss_log_pdf_(x: fArray,
                      amplitude: float = 1.0, alpha: float = 1.0, theta: float = 1.0,
                      loc: float = 0.0, normalize: bool = False) -> fArray:
    r"""
    Compute log PDF for :class:`~pymultifit.distributions.gamma_d.GammaDistributionSS`

    Notes
    -----
    .. important::

        The Gamma SS log PDF is calculated via :func:`gamma_sr_log_pdf_` by setting :math:`\lambda = \dfrac{1}{\theta}`.
    """
    return gamma_sr_log_pdf_(x,
                             amplitude=amplitude, alpha=alpha, lambda_=1 / theta,
                             loc=loc, normalize=normalize)


@doc_inherit(parent=gamma_ss_pdf_, style=doc_style)
def gamma_ss_cdf_(x: fArray,
                  amplitude: float = 1.0, alpha: float = 1.0, theta: float = 1.0,
                  loc: float = 0.0, normalize: bool = False) -> fArray:
    r"""
    Compute CDF for :class:`~pymultifit.distributions.gamma_d.GammaDistributionSS`.

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
    return gamma_sr_cdf_(x,
                         amplitude=amplitude, alpha=alpha, lambda_=1 / theta,
                         loc=loc, normalize=normalize)


@doc_inherit(parent=gamma_ss_cdf_, style=doc_style)
def gamma_ss_log_cdf_(x: fArray,
                      amplitude: float = 1.0, alpha: float = 1.0, theta: float = 1.0,
                      loc: float = 0.0, normalize: bool = False) -> fArray:
    r"""
    Compute log CDF for :class:`~pymultifit.distributions.gamma_d.GammaDistributionSS`.

    Notes
    -----
    .. important::

        The Gamma SS log CDF is calculated via logarithm of :func:`gamma_sr_cdf_` by setting
        :math:`\lambda = \dfrac{1}{\theta}`.
    """
    cdf_ = gamma_sr_cdf_(x,
                         amplitude=amplitude, alpha=alpha, lambda_=1 / theta,
                         loc=loc, normalize=normalize)

    return np.log(cdf_)


def gaussian_pdf_(x: fArray,
                  amplitude: float = 1.0, mean: float = 0.0, std: float = 1.0,
                  normalize: bool = False) -> fArray:
    r"""
    Compute PDF for :class:`~pymultifit.distributions.gaussian_d.GaussianDistribution`

    Parameters
    ----------
    x : fArray
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
    x, scalar_input = preprocess_input(x=x)

    if x.size == 0:
        return x

    exp_factor = (x - mean) / std
    pdf_ = np.exp(-0.5 * exp_factor**2) / (np.sqrt(2 * np.pi * std**2))

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_.item() if scalar_input else pdf_


@doc_inherit(parent=gaussian_pdf_, style=doc_style)
def gaussian_log_pdf_(x: fArray,
                      amplitude: float = 1.0, mean: float = 0.0, std: float = 1.0,
                      normalize: bool = False) -> fArray:
    r"""
    Compute log PDF for :class:`~pymultifit.distributions.gaussian_d.GaussianDistribution`

    Notes
    -----
    The Gaussian log PDF is defined as:

    .. math::
        \ell(x; \mu, \sigma) = -\dfrac{1}{2}\ln(2\pi) - \ln\sigma - \dfrac{1}{2}\left(\dfrac{x-\mu}{\sigma}\right)^2

    The final log PDF is expressed as :math:`\ell(x)`.
    """
    x, scalar_input = preprocess_input(x=x)

    if x.size == 0:
        return x

    exp_factor = (x - mean) / std
    log_pdf_ = -0.5 * np.log(2 * np.pi) - np.log(std) - 0.5 * exp_factor**2

    if not normalize:
        log_pdf_ = _log_pdf_scaling(log_pdf_=log_pdf_, amplitude=amplitude)

    return log_pdf_.item() if scalar_input else log_pdf_


@doc_inherit(parent=gaussian_pdf_, style=doc_style)
def gaussian_cdf_(x: fArray,
                  amplitude: float = 1.0, mean: float = 0.0, std: float = 1.0,
                  normalize: bool = False) -> fArray:
    r"""
    Compute CDF for :class:`~pymultifit.distributions.gaussian_d.GaussianDistribution`

    .. important::

        The calculation of gaussian CDF is done using :obj:`scipy.special.ndtr` function.

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
    return ndtr((x - mean) / std)


@doc_inherit(parent=gaussian_cdf_, style=doc_style)
def gaussian_log_cdf_(x: fArray, amplitude: float = 1.0, mean: float = 0.0, std: float = 1.0,
                      normalize: bool = False) -> fArray:
    r"""
    Compute log CDF for :class:`~pymultifit.distributions.gaussian_d.GaussianDistribution`

    .. important::

        The calculation of gaussian log CDF is done using :obj:`scipy.special.log_ndtr` function.

    Notes
    -----
    The Gaussian log CDF is defined as:

    .. math::
        \mathcal{L}(x) = \ln\Phi\left(\dfrac{x-\mu}{\sigma}\right)

    The final log CDF is expressed as :math:`\mathcal{L}(x)`.
    """
    return log_ndtr((x - mean) / std)


def half_normal_pdf_(x: fArray,
                     amplitude: float = 1.0, sigma: float = 1.0,
                     loc: float = 0.0, normalize: bool = False) -> fArray:
    r"""
    Compute PDF for the :class:`~pymultifit.distributions.halfNormal_d.HalfNormalDistribution`.

    .. note::
        The :class:`~pymultifit.distributions.halfNormal_d.HalfNormalDistribution` is a special case of the
        :class:`~pymultifit.distributions.foldedNormal_d.FoldedNormalDistribution` with :math:`\mu = 0`.

    Parameters
    ----------
    x : fArray
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
    x = np.asarray(a=x, dtype=float)
    scalar_input = np.isscalar(x)

    if x.size == 0:
        return np.array([])

    y = (x - loc) / sigma
    pdf_ = np.zeros_like(a=y, dtype=float)
    mask_ = y >= 0

    f1 = np.sqrt(2 / np.pi)
    pdf_[mask_] = f1 * np.exp(-0.5 * y[mask_]**2)

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    pdf_ = _remove_nans(pdf_) / sigma

    return pdf_.item() if scalar_input else pdf_


@doc_inherit(parent=half_normal_pdf_, style=doc_style)
def half_normal_log_pdf_(x: fArray,
                         amplitude: float = 1.0, sigma: float = 1.0,
                         loc: float = 0.0, normalize: bool = False) -> fArray:
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
    x = np.asarray(a=x, dtype=float)
    scalar_input = np.isscalar(x)

    if x.size == 0:
        return np.array([])

    y = (x - loc) / sigma
    log_pdf_ = np.full(shape=y.shape, fill_value=-np.inf)
    mask_ = y >= 0
    log_pdf_[mask_] = 0.5 * np.log(2 / np.pi) - 0.5 * y[mask_]**2

    if not normalize:
        log_pdf_ = _log_pdf_scaling(log_pdf_=log_pdf_, amplitude=amplitude)

    log_pdf_ -= np.log(sigma)

    return log_pdf_.item() if scalar_input else log_pdf_


@doc_inherit(parent=half_normal_pdf_, style=doc_style)
def half_normal_cdf_(x: fArray,
                     amplitude: float = 1.0, sigma: float = 1.0,
                     loc: float = 0.0, normalize: bool = False) -> fArray:
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
    x = np.asarray(a=x, dtype=float)
    scalar_input = np.isscalar(x)

    if x.size == 0:
        return np.array([])

    y = (x - loc) / sigma
    mask_ = y >= 0
    cdf_ = np.zeros_like(a=y, dtype=float)
    cdf_[mask_] = erf(y[mask_] / np.sqrt(2))

    return cdf_.item() if scalar_input else cdf_


@doc_inherit(parent=half_normal_cdf_, style=doc_style)
def half_normal_log_cdf_(x: fArray,
                         amplitude: float = 1.0, sigma: float = 1.0,
                         loc: float = 0.0, normalize: bool = False) -> fArray:
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
    cdf_ = half_normal_cdf_(x,
                            amplitude=amplitude, sigma=sigma,
                            loc=loc, normalize=normalize)

    return np.log(cdf_)


def laplace_pdf_(x: fArray,
                 amplitude: float = 1.0, mean: float = 0.0, diversity: float = 1.0,
                 normalize: bool = False) -> fArray:
    x = np.asarray(a=x, dtype=float)
    scalar_input = np.isscalar(x)

    if x.size == 0:
        return np.array([])

    pdf_ = abs(x - mean) / diversity
    pdf_ = np.exp(-pdf_) / (2 * diversity)

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_.item() if scalar_input else pdf_


def laplace_log_pdf_(x: fArray,
                     amplitude: float = 1.0, mean: float = 0.0, diversity: float = 1.0,
                     normalize: bool = False) -> fArray:
    pdf_ = laplace_pdf_(x,
                        amplitude=amplitude, mean=mean, diversity=diversity, normalize=normalize)

    return np.log(pdf_)


@doc_inherit(parent=laplace_pdf_, style=doc_style)
def laplace_cdf_(x: fArray,
                 amplitude: float = 1.0, mean: float = 0.0, diversity: float = 1.0,
                 normalize: bool = False) -> fArray:
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
    """
    x = np.asarray(a=x, dtype=float)
    scalar_input = np.isscalar(x)

    if x.size == 0:
        return np.array([])

    def _cdf1(x_):
        return 0.5 * np.exp((x_ - mean) / diversity)

    def _cdf2(x_):
        return 1 - 0.5 * np.exp(-(x_ - mean) / diversity)

    cdf_ = np.zeros_like(a=x, dtype=float)

    mask_leq = x <= mean
    cdf_[mask_leq] += _cdf1(x[mask_leq])
    cdf_[~mask_leq] += _cdf2(x[~mask_leq])

    return cdf_.item() if scalar_input else cdf_


def laplace_log_cdf_(x: fArray,
                     amplitude: float = 1.0, mean: float = 0.0, diversity: float = 1.0,
                     normalize: bool = False) -> fArray:
    cdf_ = laplace_cdf_(x,
                        amplitude=amplitude, mean=mean, diversity=diversity, normalize=normalize)

    return np.log(cdf_)


def log_normal_pdf_(x: fArray,
                    amplitude: float = 1., mean: float = 0., std: float = 1.,
                    loc: float = 0., normalize: bool = False) -> fArray:
    r"""
    Compute PDF for :class:`~pymultifit.distributions.logNormal_d.LogNormalDistribution`.

    Parameters
    ----------
    x : fArray
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

    .. math::
        y = x - \text{loc}

    The final PDF is expressed as :math:`f(y)`.
    """
    x = np.asarray(a=x, dtype=float)
    scalar_input = np.isscalar(x)

    if x.size == 0:
        return np.array([])

    y = x - loc
    pdf_ = np.zeros_like(a=y, dtype=float)
    mask_ = y > 0

    if np.any(mask_):
        y_valid = y[mask_]
        f1 = (np.log(y_valid) - mean)**2 / (2 * std**2)
        f1 = np.exp(-f1)
        pdf_[mask_] = f1 / (std * y_valid * np.sqrt(2 * np.pi))

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    pdf_ = _remove_nans(pdf_)

    return pdf_.item() if scalar_input else pdf_


def log_normal_log_pdf_(x: fArray,
                        amplitude: float = 1.0, mean: float = 0.0, std: float = 1.0,
                        loc: float = 0.0, normalize: bool = False) -> fArray:
    x = np.asarray(a=x, dtype=float)
    scalar_input = np.isscalar(x)

    if x.size == 0:
        return np.array([])

    y = x - loc
    log_pdf_ = np.full(shape=y.shape, fill_value=-np.inf)
    mask_ = y > 0

    if np.any(mask_):
        y_valid = y[mask_]
        f1 = (np.log(y_valid) - mean)**2 / (2 * std**2)
        log_pdf_[mask_] = -f1
        log_pdf_[mask_] -= np.log(std * y_valid * np.sqrt(2 * np.pi))

    if not normalize:
        log_pdf_ = _log_pdf_scaling(log_pdf_=log_pdf_, amplitude=amplitude)

    log_pdf_ = _remove_nans(log_pdf_)

    return log_pdf_.item() if scalar_input else log_pdf_


@doc_inherit(parent=log_normal_pdf_, style=doc_style)
def log_normal_cdf_(x: fArray,
                    amplitude: float = 1.0, mean: float = 0.0, std=1.0,
                    loc: float = 0.0, normalize: bool = False) -> fArray:
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
    The LogNormal CDF is defined as:

    .. math::
        F(x) = \Phi\left(\dfrac{\ln x - \mu}{\sigma}\right)
    """
    return _remove_nans(gaussian_cdf_(x=np.log(x - loc),
                                      amplitude=amplitude, mean=mean, std=std, normalize=normalize))


def log_normal_log_cdf_(x: fArray,
                        amplitude: float = 1.0, mean: float = 0.0, std: float = 1.0,
                        loc: float = 0.0, normalize: bool = False) -> fArray:
    x = np.asarray(a=x, dtype=float)
    scalar_input = np.isscalar(x)

    if x.size == 0:
        return np.array([])

    y = x - loc
    log_cdf_ = gaussian_log_cdf_(np.log(y),
                                 amplitude=amplitude, mean=mean, std=std, normalize=normalize)
    log_cdf_ = _remove_nans(log_cdf_, nan_value=-np.inf)

    return log_cdf_.item() if scalar_input else log_cdf_


def uniform_pdf_(x: fArray,
                 amplitude: float = 1.0, low: float = 0.0, high: float = 1.0,
                 normalize: bool = False) -> fArray:
    r"""
    Compute PDF of :class:`~pymultifit.distributions.uniform_d.UniformDistribution`.

    Parameters
    ----------
    x : fArray
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

    .. math:: f(x\ |\ a, b) = \dfrac{1}{b-a}
    """
    x = np.asarray(a=x, dtype=float)
    scalar_input = np.isscalar(x)

    if x.size == 0:
        return np.array([])

    high_ = high + low
    pdf_ = np.zeros_like(a=x, dtype=float)

    if high_ == low:
        return np.full(shape=x.size, fill_value=np.nan)

    mask_ = np.logical_and(x >= low, x <= high_)
    pdf_[mask_] = 1 / (high_ - low)

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    pdf_ = _remove_nans(pdf_)

    return pdf_.item() if scalar_input else pdf_


def uniform_log_pdf_(x: fArray,
                     amplitude: float = 1.0, low: float = 0.0, high: float = 1.0,
                     normalize: bool = False) -> fArray:
    x = np.asarray(a=x, dtype=float)
    scalar_input = np.isscalar(x)

    if x.size == 0:
        return np.array([])

    high_ = high + low
    log_pdf_ = np.full(shape=x.shape, fill_value=-np.inf)

    # if high_ == low:
    #     return log_pdf_

    mask_ = np.logical_and(x >= low, x <= high_)
    log_pdf_[mask_] = -np.log(high_ - low)

    if not normalize:
        log_pdf_ = _log_pdf_scaling(log_pdf_=log_pdf_, amplitude=amplitude)

    return log_pdf_.item() if scalar_input else log_pdf_


@doc_inherit(parent=uniform_pdf_, style=doc_style)
def uniform_cdf_(x: fArray,
                 amplitude: float = 1.0, low: float = 0.0, high: float = 1.0,
                 normalize: bool = False) -> fArray:
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
                        0 &, x < a\\
                        \dfrac{x-a}{b-a} &, x \in [a, b]\\
                        1 &, x > b
                        \end{cases}
    """
    x = np.asarray(a=x, dtype=float)
    scalar_input = np.isscalar(x)

    if x.size == 0:
        return np.array([])

    high = high + low
    cdf_ = np.zeros_like(a=x, dtype=float)

    if low == high == 0:
        return np.full(shape=x.size, fill_value=np.nan)

    mask_ = (x >= low) & (x <= high)
    cdf_[mask_] = (x[mask_] - low) / (high - low)
    cdf_[x > high] = 1

    return cdf_.item() if scalar_input else cdf_


def uniform_log_cdf_(x: fArray,
                     amplitude: float = 1.0, low: float = 0.0, high: float = 1.0,
                     normalize: bool = False) -> fArray:
    x = np.asarray(a=x, dtype=float)
    scalar_input = np.isscalar(x)

    if x.size == 0:
        return np.array([])

    high = high + low
    log_cdf_ = np.full(shape=x.shape, fill_value=-np.inf)

    if low == high == 0:
        return log_cdf_

    mask_ = np.logical_and(x >= low, x <= high)
    log_cdf_[mask_] = np.log(x[mask_] - low) - np.log(high - low)
    log_cdf_[x > high] = 0

    return log_cdf_.item() if scalar_input else log_cdf_


def scaled_inv_chi_square_pdf_(x, amplitude: float = 1.0, df: float = 1.0, scale: float = 1.0,
                               loc: float = 0.0, normalize: bool = False):
    r"""
    Compute PDF of :class:`~pymultifit.distributions.scaledInvChiSquare_d.ScaledInverseChiSquareDistribution`.

    Parameters
    ----------
    x : fArray
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
    x = np.asarray(a=x, dtype=float)
    scalar_input = np.isscalar(x)

    if x.size == 0:
        return np.array([])

    tau2 = scale / df
    df_half = df / 2

    y = x - loc

    pdf_ = np.zeros_like(a=y, dtype=float)
    mask = y > 0

    frac1 = np.power(tau2 * df_half, df_half) / gamma(df_half)
    if np.any(mask):
        y_valid = y[mask]
        frac2 = np.exp(-(df * tau2) / (2 * y_valid)) / np.power(y_valid, 1 + df_half)

        pdf_[mask] = frac1 * frac2

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_


@doc_inherit(parent=scaled_inv_chi_square_pdf_, style=doc_style)
def scaled_inv_chi_square_log_pdf_(x: fArray, amplitude: float = 1.0, df: float = 1.0, scale: float = 1.0,
                                   loc: float = 0.0, normalize: bool = False):
    r"""
    Compute logPDF of :class:`~pymultifit.distributions.scaledInvChiSquare_d.ScaledInverseChiSquareDistribution`.

    Notes
    -----
    The Scaled Inverse ChiSquare PDF is defined as:

    .. math:: \ell(y) = \ln(\tau^2\nu_2) - \ln\Gamma(\nu_2) - (1+\nu_2)\ln(\nu) - \dfrac{\nu\tau^2}{2y}

    where :math:`\ln` is the natural logarithm, :math:`\ln\Gamma(\cdot)` is the :obj:`~scipy.special.gammaln` function,
    :math:`\nu_2 = \dfrac{\nu}{2}`, :math:`\tau^2 = \dfrac{\phi}{\nu}` and :math:`y` is the transformed value of
    :math:`x`, defined as:

    .. math:: y = x - \text{loc}

    The final PDF is expressed as :math:`\ell(y)`.
    """
    x = np.asarray(a=x, dtype=float)
    scalar_input = np.isscalar(x)

    if x.size == 0:
        return np.array([])

    tau2 = scale / df
    df_half = df / 2

    y = x - loc

    log_pdf_ = np.full(shape=y.shape, fill_value=-np.inf)
    mask_ = y > 0

    frac1 = df_half * np.log(tau2 * df_half) - gammaln(df_half)
    if np.any(mask_):
        y_valid = y[mask_]
        frac2 = -(tau2 * df) / (2 * y_valid) - (1 + df_half) * np.log(y_valid)

        log_pdf_[mask_] = frac1 + frac2

    if not normalize:
        log_pdf_ = _log_pdf_scaling(log_pdf_=log_pdf_, amplitude=amplitude)

    return log_pdf_


def scaled_inv_chi_square_cdf_(x, amplitude, df, scale, loc: float = 0.0,
                               normalize: bool = False):
    x = np.asarray(a=x, dtype=float)
    scalar_input = np.isscalar(x)

    if x.size == 0:
        return np.array([])

    tau2 = scale / df
    df_half = df / 2

    y = x - loc
    cdf_ = np.zeros_like(a=y, dtype=float)
    mask_ = y > 0
    cdf_[mask_] = gammaincc(df_half, (tau2 * df_half) / y[mask_])

    return cdf_


def scaled_inv_chi_square_log_cdf_(x, amplitude, df, scale, loc, normalize=False):
    x = np.asarray(a=x, dtype=float)
    scalar_input = np.isscalar(x)

    if x.size == 0:
        return np.array([])

    tau2 = scale / df
    df_half = df / 2

    y = x - loc
    log_cdf_ = np.full(shape=y.shape, fill_value=-np.inf)
    mask_ = y > 0
    log_cdf_[mask_] = np.log(gammaincc(df_half, (tau2 * df_half) / y[mask_]))

    return log_cdf_


def skew_normal_pdf_(x: fArray,
                     amplitude: float = 1.0, shape: float = 1.0, loc: float = 0.0, scale: float = 1.0,
                     normalize: bool = False) -> fArray:
    r"""
    Compute PDF of :class:`~pymultifit.distributions.skewNormal_d.SkewNormalDistribution`.

    Parameters
    ----------
    x : fArray
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
    x = np.asarray(a=x, dtype=float)
    scalar_input = np.isscalar(x)

    if x.size == 0:
        return np.array([])

    y = (x - loc) / scale
    g_pdf_ = gaussian_pdf_(x=y, normalize=True)
    g_cdf_ = gaussian_cdf_(x=shape * y, normalize=True)

    pdf_ = (2 / scale) * g_pdf_ * g_cdf_

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    pdf_ = _remove_nans(pdf_)

    return pdf_.item() if scalar_input else pdf_


def skew_normal_log_pdf_(x: fArray,
                         amplitude: float = 1.0, shape: float = 1.0, loc: float = 0.0, scale: float = 1.0,
                         normalize: bool = False) -> fArray:
    x = np.asarray(a=x, dtype=float)
    scalar_input = np.isscalar(x)

    if x.size == 0:
        return np.array([])

    y = (x - loc) / scale
    g_l_pdf_ = gaussian_log_pdf_(x=y, normalize=True)
    g_l_cdf_ = gaussian_log_cdf_(x=shape * y, normalize=True)

    log_pdf_ = np.log(2 / scale) + g_l_pdf_ + g_l_cdf_

    if not normalize:
        log_pdf_ = _log_pdf_scaling(log_pdf_=log_pdf_, amplitude=amplitude)

    return log_pdf_.item() if scalar_input else log_pdf_


@doc_inherit(parent=skew_normal_pdf_, style=doc_style)
def skew_normal_cdf_(x: fArray,
                     amplitude: float = 1.0, shape: float = 1.0, loc: float = 0.0, scale: float = 1.0,
                     normalize: bool = False):
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

    .. math:: F(x) = \Phi\left(\dfrac{x - \xi}{\omega}\right) - 2T\left(\dfrac{x - \xi}{\omega}, \alpha\right)

    where, :math:`T` is the Owen's T function, see :obj:`scipy.special.owens_t`, and
    :math:`\Phi(\cdot)` is the :class:`~pymultifit.distributions.gaussian_d.GaussianDistribution` CDF function.
    """
    x = np.asarray(a=x, dtype=float)
    scalar_input = np.isscalar(x)

    if x.size == 0:
        return np.array([])

    y = (x - loc) / scale
    cdf_ = gaussian_cdf_(x=y, normalize=True) - 2 * owens_t(y, shape)

    return cdf_.item() if scalar_input else cdf_


def sym_gen_normal_pdf_(x: fArray,
                        amplitude: float = 1.0, shape: float = 1.0, loc: float = 0.0, scale: float = 1.0,
                        normalize: bool = False) -> fArray:
    r"""
    Compute PDF of :class:`~pymultifit.distributions.generalized.genNorm_d.SymmetricGeneralizedNormalDistribution`.

    Parameters
    ----------
    x : fArray
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

    .. math:: f(y\ |\ \beta, \mu, \alpha) =
        \dfrac{\beta}{2\alpha\Gamma(1/\beta)}\exp\left[-\dfrac{|x-\mu|^\beta}{\alpha^\beta}\right]

    The final PDF is expressed as :math:`f(y)`.
    """
    x = np.asarray(a=x, dtype=float)
    scalar_input = np.isscalar(x)

    if x.size == 0:
        return np.array([])

    mu, alpha, beta = loc, scale, shape

    log1_ = np.log(beta) - np.log(2 * alpha * gamma(1 / beta))

    log2_ = np.log(abs(x - mu) / alpha)
    log2_ = np.exp(beta * log2_)

    pdf_ = np.exp(log1_ - log2_)

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_.item() if scalar_input else pdf_


@doc_inherit(parent=sym_gen_normal_pdf_, style=doc_style)
def sym_gen_normal_cdf_(x: fArray,
                        amplitude: float = 1.0, shape: float = 1.0, loc: float = 0.0, scale: float = 1.0,
                        normalize: bool = False) -> fArray:
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

    .. math:: F(x) =
     \dfrac{1}{2} + \dfrac{\text{sign}(x-\mu)}{2}\gamma\left(\dfrac{1}{\beta},\left|\dfrac{x - mu}{\alpha}\right|^\beta\,\right)

    where :math:`\gamma(\cdot,\cdot)` is the lower incomplete gamma function, see :obj:`~scipy.special.gammainc`.
    """
    x = np.asarray(a=x, dtype=float)
    scalar_input = np.isscalar(x)

    if x.size == 0:
        return np.array([])

    mu, alpha, beta = loc, scale, shape

    f1 = (x - mu) / alpha
    cdf_ = 0.5 + (np.sign(x - mu) * 0.5 * gammainc(1 / beta, abs(f1)**beta))

    return cdf_.item() if scalar_input else cdf_


def _beta_masking(y: fArray, alpha: float, beta: float) -> fArray:
    """
    Creates a mask for beta distributions to identify out-of-range or undefined values.

    Parameters
    ----------
    y : fArray
        Array of values to check, typically in the range [0, 1].
    alpha : float
        Alpha parameter of the beta distribution. Determines the shape of the distribution.
    beta : float
        Beta parameter of the beta distribution. Determines the shape of the distribution.

    Returns
    -------
    np.ndarray
        A boolean mask array where `True` indicates out-of-range or undefined values.
    """
    out_of_range_mask = np.logical_or(y < 0, y > 1)
    undefined_mask = np.zeros_like(a=y, dtype=bool)
    if alpha <= 1:
        undefined_mask = np.logical_or(undefined_mask, y == 0)
    if beta <= 1:
        undefined_mask = np.logical_or(undefined_mask, y == 1)
    mask_ = np.logical_or(out_of_range_mask, undefined_mask)
    return mask_


def _pdf_scaling(pdf_: fArray, amplitude: float) -> fArray:
    """
    Scales a probability density function (PDF) by a given amplitude.

    Parameters
    ----------
    pdf_ : fArray
        The input PDF array to be scaled.
    amplitude : float
        The amplitude to scale the PDF.

    Returns
    -------
    np.ndarray
        The scaled PDF array.
    """
    return amplitude * (pdf_ / np.max(pdf_))


def _log_pdf_scaling(log_pdf_: fArray, amplitude: float) -> fArray:
    scaling = log_pdf_ - np.max(log_pdf_)
    return np.log(amplitude) + scaling


def _remove_nans(x: fArray, nan_value=None) -> fArray:
    """
    Replaces NaN, positive infinity, and negative infinity values in an array.

    Parameters
    ----------
    x : fArray
        Input array that may contain NaN, positive infinity, or negative infinity values.

    Returns
    -------
    np.ndarray
        Array with NaN replaced by 0, positive infinity replaced by `np.inf`, and negative
    infinity replaced by `-np.inf`.
    """
    nan_value = 0 if nan_value is None else nan_value
    return np.nan_to_num(x=np.asarray(x), copy=False, nan=nan_value, posinf=np.inf, neginf=-np.inf)


def preprocess_input(x, loc=0.0, scale=1.0):
    """
    Preprocess the input array by converting to float, checking for scalar input, handling empty arrays, and normalizing the data.

    Parameters
    ----------
    x : np.ndarray
        Input data.
    loc : float, optional
        The location parameter, for shifting.
        Defaults to 0.0.
    scale: float, optional
        The scale parameter, for scaling.
        Defaults to 1.0,

    Returns
    -------
    tuple:
        (processed array, scalar_input_flag)
    """
    x = np.asarray(a=x, dtype=float)
    scalar_input = np.isscalar(x)

    if x.size == 0:
        return np.array([]), scalar_input

    y = (x - loc) / scale

    return y, scalar_input
