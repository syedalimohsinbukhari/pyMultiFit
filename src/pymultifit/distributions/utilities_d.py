"""Created on Aug 03 17:13:21 2024"""

__all__ = ['_beta_masking', '_pdf_scaling', '_remove_nans',
           'arc_sine_pdf_',
           'beta_pdf_', 'beta_cdf_',
           'chi_square_pdf_', 'chi_square_cdf_',
           'exponential_pdf_', 'exponential_cdf_',
           'folded_normal_pdf_', 'folded_normal_cdf_',
           'gamma_sr_pdf_', 'gamma_sr_cdf_',
           'gamma_ss_pdf_',
           'gaussian_pdf_', 'gaussian_cdf_',
           'half_normal_pdf_', 'half_normal_cdf_',
           'laplace_pdf_', 'laplace_cdf_',
           'log_normal_pdf_', 'log_normal_cdf_',
           'skew_normal_pdf_', 'skew_normal_cdf_',
           'uniform_pdf_', 'uniform_cdf_']

from typing import Union

import numpy as np
from custom_inherit import doc_inherit
from scipy.special import betainc, erf, gamma, gammainc, gammaln, owens_t

from .. import doc_style


def arc_sine_pdf_(x: np.ndarray,
                  amplitude: float = 1.0,
                  loc: float = 0.0, scale: float = 1.0, normalize: bool = False) -> np.ndarray:
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
        Array of the same shape as :math:`x`, containing the evaluated PDF values.

    Notes
    -----
    The ArcSine PDF is defined as:

    .. math:: f(y) = \frac{1}{\pi \sqrt{y(1-y)}}

    where, :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \frac{x - \text{loc}}{\text{scale}}

    The final PDF is expressed as :math:`f(y)/\text{scale}`.
    """
    return beta_pdf_(x=x, amplitude=amplitude, alpha=0.5, beta=0.5, loc=loc, scale=scale, normalize=normalize)


def beta_pdf_(x: np.ndarray,
              amplitude: float = 1.0, alpha: float = 1.0, beta: float = 1.0,
              loc: float = 0.0, scale: float = 1.0, normalize: bool = False) -> np.ndarray:
    r"""
    Compute PDF of :class:`~pymultifit.distributions.beta_d.BetaDistribution`.

    Parameters
    ----------
    x : np.ndarray
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
        Array of the same shape as `x`, containing the evaluated PDF values.

    Notes
    -----
    The Beta PDF is defined as:

    .. math:: f(y; \alpha, \beta) = \frac{y^{\alpha - 1} (1 - y)^{\beta - 1}}{B(\alpha, \beta)}

    where :math:`B(\alpha, \beta)` is the Beta function (see, :obj:`scipy.special.beta`), and :math:`y` is the
    transformed value of :math:`x` such that:

    .. math:: y = \frac{x - \text{loc}}{\text{scale}}

    The final PDF is expressed as :math:`f(y)/\text{scale}`.
    """
    if x.size == 0:
        return np.array([])

    y = (x - loc) / scale
    pdf_ = np.zeros_like(a=y, dtype=float)

    numerator = y**(alpha - 1) * (1 - y)**(beta - 1)
    normalization_factor = gamma(alpha) * gamma(beta) / gamma(alpha + beta)

    mask_ = _beta_masking(y=y, alpha=alpha, beta=beta)
    pdf_[~mask_] = numerator[~mask_] / normalization_factor

    if alpha <= 1:
        pdf_[y == 0] = np.inf
    if beta <= 1:
        pdf_[y == 1] = np.inf
    if alpha == 1 and beta == 1:
        pdf_[y == 1] = 1

    # handle the cases where nans can occur with nan_to_num
    # np.inf and -np.inf to not affect the infinite values
    pdf_ = _remove_nans(pdf_ / scale)

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_


@doc_inherit(parent=beta_pdf_, style=doc_style)
def beta_logpdf_(x: np.ndarray,
                 amplitude: float = 1.0, alpha: float = 1.0, beta: float = 1.0,
                 loc: float = 0.0, scale: float = 1.0, normalize: bool = False) -> np.ndarray:
    r"""Compute logPDF for :class:`~pymultifit.distributions.beta_d.BetaDistribution`."""
    if x.size == 0:
        return np.array([])

    y = (x - loc) / scale
    mask_ = _beta_masking(y=y, alpha=alpha, beta=beta)

    logpdf_ = np.full_like(a=y, fill_value=-np.inf, dtype=float)
    log_numerator = (alpha - 1) * np.log(y) + (beta - 1) * np.log(1 - y)

    if normalize:
        log_normalization = gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)
        amplitude = 1.0
    else:
        log_normalization = 0.0

    logpdf_[~mask_] = np.log(amplitude) + log_numerator[~mask_] - log_normalization

    if alpha <= 1:
        logpdf_[y == 0] = np.inf
    if beta <= 1:
        logpdf_[y == 1] = np.inf

    return logpdf_ - np.log(scale)


@doc_inherit(parent=beta_pdf_, style=doc_style)
def beta_cdf_(x: np.ndarray,
              amplitude: float = 1.0, alpha: float = 1.0, beta: float = 1.0,
              loc: float = 0.0, scale: float = 1.0, normalize: bool = False) -> np.ndarray:
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

    .. math:: I_x(\alpha, \beta)

    where :math:`I_x(\alpha, \beta)` is the regularized incomplete Beta function, see :obj:`~scipy.special.betainc`.
    """
    if x.size == 0:
        return np.array([])

    y = (x - loc) / scale
    cdf_ = np.zeros_like(a=y, dtype=float)

    mask_ = np.logical_and(y > 0, y < 1)
    cdf_[mask_] = betainc(alpha, beta, y[mask_])
    cdf_[y >= 1] = 1

    return cdf_


def chi_square_pdf_(x: np.ndarray,
                    amplitude: float = 1.0, degree_of_freedom: Union[int, float] = 1,
                    loc: float = 0.0, scale: float = 1.0, normalize: bool = False) -> np.ndarray:
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

    where :math:`\Gamma(k)` is the :obj:`~scipy.special.gamma` function,
    and :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = x - \text{loc}

    The final PDF is expressed as :math:`f(y)`.
    """
    y = (x - loc) / scale
    pdf_ = np.zeros_like(a=x, dtype=float)
    mask_ = y > 0
    pdf_[mask_] = gamma_sr_pdf_(x=y[mask_], amplitude=amplitude, alpha=degree_of_freedom / 2, lambda_=0.5, loc=0,
                                normalize=normalize)

    return pdf_ / scale


@doc_inherit(parent=chi_square_pdf_, style=doc_style)
def chi_square_cdf_(x: np.ndarray,
                    amplitude: float = 1.0, degree_of_freedom: Union[int, float] = 1,
                    loc: float = 0.0, scale: float = 1.0, normalize: bool = False) -> np.ndarray:
    """
    Compute PDF for :mod:`~pymultifit.distributions.chiSquare_d.ChiSquareDistribution`.

    Parameters
    ----------
    amplitude: float, optional
        For API consistency only.
    normalize: bool, optional
        For API consistency only.

    Notes
    -----
    The ChiSquare CDF is defined as:


    """
    y = (x - loc) / scale
    cdf_ = np.zeros_like(a=y, dtype=float)
    mask_ = y >= 0
    cdf_[mask_] = gammainc(degree_of_freedom / 2, y[mask_] / 2)
    return cdf_


def exponential_pdf_(x: np.ndarray,
                     amplitude: float = 1., lambda_: float = 1., loc: float = 0.0,
                     normalize: bool = False) -> np.ndarray:
    r"""
    Compute PDF for :class:`~pymultifit.distributions.exponential_d.ExponentialDistribution`.

    .. note::
        This function uses :func:`~pymultifit.distributions.utilities_d.gamma_sr_pdf_` to calculate the PDF with
        :math:`\alpha = 1` and :math:`\lambda_\text{gammaSR} = \lambda_\text{expon}`.

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
        \lambda \exp\left[-\lambda y\right] &; y \geq 0, \\
        0 &; y < 0.
        \end{cases}

    where, :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = x - \text{loc}

    The final PDF is expressed as :math:`f(y)`.
    """
    return gamma_sr_pdf_(x=x, amplitude=amplitude, alpha=1., lambda_=lambda_, loc=loc, normalize=normalize)


@doc_inherit(parent=exponential_pdf_, style=doc_style)
def exponential_cdf_(x: np.ndarray,
                     amplitude: float = 1., scale: float = 1., loc: float = 0.0,
                     normalize: bool = False) -> np.ndarray:
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
    y = x - loc
    pdf_ = np.zeros_like(a=y, dtype=float)
    mask_ = y > 0
    pdf_[mask_] = 1 - np.exp(-scale * y[mask_])
    return pdf_


def folded_normal_pdf_(x: np.ndarray,
                       amplitude: float = 1., mean: float = 0.0, sigma: float = 1.0,
                       loc: float = 0.0, normalize: bool = False) -> np.ndarray:
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

    .. math:: y = \dfrac{x - \text{loc}}{\sigma}

    The final PDF is expressed as :math:`f(y)/\sigma`.
    """
    if x.size == 0:
        return np.array([])

    _, pdf_ = _folded(x=x, mean=mean, sigma=sigma, loc=loc, g_func=gaussian_pdf_)

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_ / sigma


def _folded(x, mean, sigma, loc, g_func):
    if sigma <= 0 or mean < 0:
        return np.full(shape=x.size, fill_value=np.nan)

    y = (x - loc) / sigma
    temp_ = np.zeros_like(a=y, dtype=float)

    mask = y >= 0
    g1 = g_func(x=y[mask], mean=mean, normalize=True)
    g2 = g_func(x=y[mask], mean=-mean, normalize=True)
    temp_[mask] = g1 + g2

    return mask, temp_


@doc_inherit(parent=folded_normal_pdf_, style=doc_style)
def folded_normal_cdf_(x: np.ndarray,
                       amplitude: float = 1., mean: float = 0.0, sigma: float = 1.0,
                       loc: float = 0.0, normalize: bool = False) -> np.ndarray:
    r"""
    Compute CDF for :class:`~pymultifit.distributions.foldedNormal_d.FoldedNormalDistribution`.

    Parameters
    ----------
    amplitude: float, optional
        For API consistency only.
    normalize: bool, optional
        For API consistency only.

    Notes
    -----
    The FoldedNormal CDF is defined as:

    .. math::
        F(y) = \Phi(y\ | \mu, 1) + \Phi(y\ | -\mu, 1) - 1

    where :math:`\Phi` is the CDF of :class:`~pymultifit.distributions.gaussian_d.GaussianDistribution`,
    and :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = \dfrac{x - \text{loc}}{\sigma}
    """
    if x.size == 0:
        return np.array([])

    mask_, cdf_ = _folded(x=x, mean=mean, sigma=sigma, loc=loc, g_func=gaussian_cdf_)
    cdf_[mask_] -= 1
    return cdf_


def gamma_sr_pdf_(x: np.ndarray,
                  amplitude: float = 1.0, alpha: float = 1.0, lambda_: float = 1.0,
                  loc: float = 0.0, normalize: bool = False) -> np.ndarray:
    r"""
    Compute PDF for :class:`~pymultifit.distributions.gamma_d.GammaDistributionSR` with :math:`\alpha` (shape)
    and :math:`\lambda` (rate) parameters.

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
        Array of the same shape as :math:`x`, containing the evaluated PDF values.

    Notes
    -----
    The Gamma SR PDF is defined as:

    .. math::
        f(y; \alpha, \lambda) =
        \begin{cases}
        \dfrac{\lambda^\alpha}{\Gamma(\alpha)} y^{\alpha - 1} \exp\left[-\lambda y\right], & y > \text{loc}, \\
        0, & y \leq \text{loc}.
        \end{cases}

    where :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = x - \text{loc}

    The final PDF is expressed as :math:`f(y)`.
    """
    if x.size == 0:
        return np.array([])

    y = x - loc
    numerator = y**(alpha - 1) * np.exp(-lambda_ * y)
    normalization_factor = gamma(alpha) / lambda_**alpha

    pdf_ = numerator / normalization_factor
    pdf_[x < loc] = 0

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_


@doc_inherit(parent=gamma_sr_pdf_, style=doc_style)
def gamma_sr_cdf_(x: np.ndarray,
                  amplitude: float = 1.0, alpha: float = 1.0, lambda_: float = 1.0,
                  loc: float = 0.0, normalize: bool = False) -> np.ndarray:
    r"""
    Compute CDF for :class:`~pymultifit.distributions.gamma_d.GammaDistributionSR` with :math:`\alpha`
    and :math:`\lambda` parameters.

    Parameters
    ----------
    amplitude: float, optional
        For API consistency only.
    normalize: float, optional
        For API consistency only.

    Notes
    -----
    The Gamma CDF is defined as:

    .. math::
        F(x) = \dfrac{1}{\Gamma(\alpha)}\gamma(\alpha, \lambda x)

    where, :math:`\gamma(\alpha, \lambda x)` is the lower incomplete gamma function, see :obj:`~scipy.special.gammainc`.
    """
    if x.size == 0:
        return np.array([])

    y = x - loc
    y = np.maximum(y, 0)
    return gammainc(alpha, lambda_ * y)


def gamma_ss_pdf_(x: np.ndarray,
                  amplitude: float = 1.0, alpha: float = 1.0, theta: float = 1.0,
                  normalize: bool = False) -> np.ndarray:
    r"""
    Compute PDF for :class:`~pymultifit.distributions.gamma_d.GammaDistributionSS` with :math:`\alpha` (shape)
    and :math:`\theta` (scale) parameters.

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
        The scale parameter, :math:`\lambda`.
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
    The Gamma SS PDF is defined as:

    .. math::
        f(y\ |\ \alpha, \theta) = \dfrac{1}{\Gamma(\alpha)\theta^\alpha}y^{\alpha-1}\exp\left[-\dfrac{y}{\theta}\right]

    where :math:`y` is the transformed value of :math:`x`, defined as:

    .. math:: y = x - \text{loc}

    The final PDF is expressed as :math:`f(y)`.
    """
    return gamma_sr_pdf_(x=x, amplitude=amplitude, alpha=alpha, lambda_=1 / theta, normalize=normalize)


@doc_inherit(parent=gamma_ss_pdf_, style=doc_style)
def gamma_ss_cdf_(x: np.ndarray,
                  amplitude: float = 1., alpha: float = 1.0, theta: float = 1.0,
                  normalize: bool = False) -> np.ndarray:
    r"""
    Compute CDF for :class:`~pymultifit.distributions.gamma_d.GammaDistributionSS` with :math:`\alpha` (shape)
    and :math:`\theta` (scale) parameters.

    Parameters
    ----------
    amplitude: float, optional
        For API consistency only.
    normalize: bool, optional
        For API consistency only.

    Notes
    -----
    The Gamma CDF is defined as:

    .. math:: F(y) = \dfrac{1}{\Gamma(\alpha)}\gamma\left(\alpha, \dfrac{y}{\theta}\right)

    where, :math:`\gamma(\alpha, \lambda y)` is the lower incomplete gamma function, see :obj:`~scipy.special.gammainc`.
    """
    return gamma_sr_cdf_(x=x, amplitude=amplitude, alpha=alpha, lambda_=1 / theta, normalize=normalize)


def gaussian_pdf_(x: np.ndarray,
                  amplitude: float = 1.0, mean: float = 0.0, std: float = 1.0,
                  normalize: bool = False) -> np.ndarray:
    r"""
    Compute PDF for the :mod:`~pymultifit.distributions.gaussian_d.GaussianDistribution`

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
    if x.size == 0:
        return np.array([])

    exponent_factor = (x - mean)**2 / (2 * std**2)
    exponent_factor = np.exp(-exponent_factor)
    normalization_factor = std * np.sqrt(2 * np.pi)

    pdf_ = exponent_factor / normalization_factor

    if not normalize:
        pdf_ = pdf_ / np.max(pdf_)
        pdf_ *= amplitude

    return pdf_


@doc_inherit(parent=gaussian_pdf_, style=doc_style)
def gaussian_cdf_(x: np.ndarray,
                  amplitude: float = 1.0, mean: float = 0.0, std: float = 1.0,
                  normalize: bool = False) -> np.ndarray:
    r"""
    Compute CDF for the :mod:`~pymultifit.distributions.gaussian_d.GaussianDistribution`

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
    """
    if x.size == 0:
        return np.array([])

    num_ = x - mean
    den_ = std * np.sqrt(2)
    return 0.5 * (1 + erf(num_ / den_))


def half_normal_pdf_(x: np.ndarray,
                     amplitude: float = 1.0, sigma: float = 1.0,
                     loc: float = 0.0, normalize: bool = False) -> np.ndarray:
    r"""
    Compute PDF for the :class:`~pymultifit.distributions.halfNormal_d.HalfNormalDistribution`.

    .. note::
        The :class:`~pymultifit.distributions.halfNormal_d.HalfNormalDistribution`. is a special case of the
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
        f(x\ |\ \sigma) = \sqrt{\dfrac{2}{\pi\sigma^2}}\exp\left[-\dfrac{x^2}{2\sigma^2}\right]

    where :math:`x >= 0`.
    """
    return folded_normal_pdf_(x=x, amplitude=amplitude, mean=0, sigma=sigma, loc=loc, normalize=normalize)


@doc_inherit(parent=half_normal_pdf_, style=doc_style)
def half_normal_cdf_(x: np.ndarray,
                     amplitude: float = 1.0, scale: float = 1.0,
                     loc: float = 0.0, normalize: bool = False) -> np.ndarray:
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

    .. math:: F(x) = \text{erf}\left( \frac{x}{\sqrt{2\sigma^2}}\right)
    """
    return folded_normal_cdf_(x=x, amplitude=amplitude, normalize=normalize)


def laplace_pdf_(x: np.ndarray,
                 amplitude: float = 1., mean: float = 0., diversity: float = 1.,
                 normalize: bool = False) -> np.ndarray:
    r"""
    Compute PDF for :class:`~pymultifit.distributions.laplace_d.LaplaceDistribution`.

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
    diversity : float, optional
        The diversity parameter, :math:`b`.
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

    .. math:: f(x\ |\ \mu, b) = \dfrac{1}{2b}\exp\left(-\dfrac{|x - \mu|}{b}\right)

    The final PDF is expressed as :math:`f(x)`.
    """
    if x.size == 0:
        return np.array([])

    exponent_factor = abs(x - mean) / diversity
    exponent_factor = np.exp(-exponent_factor)
    normalization_factor = 2 * diversity

    pdf_ = exponent_factor / normalization_factor

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_


@doc_inherit(parent=laplace_pdf_, style=doc_style)
def laplace_cdf_(x: np.ndarray,
                 amplitude: float = 1.0, mean: float = 0.0, diversity: float = 1.0,
                 normalize: bool = False) -> np.ndarray:
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
    if x.size == 0:
        return np.array([])

    def _cdf1(x_):
        return 0.5 * np.exp((x_ - mean) / diversity)

    def _cdf2(x_):
        return 1 - 0.5 * np.exp(-(x_ - mean) / diversity)

    # to ensure equality with scipy, had to break down the output with empty array so that sorting is not needed.
    result = np.zeros_like(a=x, dtype=np.float64)

    mask_leq = x <= mean
    result[mask_leq] += _cdf1(x[mask_leq])
    result[~mask_leq] += _cdf2(x[~mask_leq])

    return result


def log_normal_pdf_(x: np.ndarray,
                    amplitude: float = 1., mean: float = 0., std: float = 1.,
                    loc: float = 0., normalize: bool = False) -> np.ndarray:
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

    .. math::
        y = x - \text{loc}

    The final PDF is expressed as :math:`f(y)`.
    """
    if x.size == 0:
        return np.array([])

    y = x - loc

    exponent_factor = (np.log(y) - mean)**2 / (2 * std**2)
    exponent_factor = np.exp(-exponent_factor)
    normalization_factor = std * y * np.sqrt(2 * np.pi)

    pdf_ = exponent_factor / normalization_factor

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return _remove_nans(pdf_)


@doc_inherit(parent=log_normal_pdf_, style=doc_style)
def log_normal_cdf_(x: np.ndarray,
                    amplitude: float = 1.0, mean: float = 0.0, std=1.0,
                    loc: float = 0.0, normalize: bool = False) -> np.ndarray:
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
    return _remove_nans(gaussian_cdf_(x=np.log(x - loc), mean=mean, std=std))


def uniform_pdf_(x: np.ndarray,
                 amplitude: float = 1.0, low: float = 0.0, high: float = 1.0,
                 normalize: bool = False) -> np.ndarray:
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

    .. math:: f(x\ |\ a, b) = \dfrac{1}{b-a}
    """
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

    return _remove_nans(pdf_)


@doc_inherit(parent=uniform_pdf_, style=doc_style)
def uniform_cdf_(x: np.ndarray,
                 amplitude: float = 1.0, low: float = 0.0, high: float = 1.0,
                 normalize: bool = False) -> np.ndarray:
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
    if x.size == 0:
        return np.array([])

    high = high + low

    if low == high == 0:
        return np.full(shape=x.size, fill_value=np.nan)

    cdf_values = np.zeros_like(a=x, dtype=float)
    within_bounds = (x >= low) & (x <= high)
    cdf_values[within_bounds] = (x[within_bounds] - low) / (high - low)
    cdf_values[x > high] = 1

    return cdf_values


def skew_normal_pdf_(x: np.ndarray,
                     amplitude: float = 1.0, shape: float = 0.0, loc: float = 0.0, scale: float = 1.0,
                     normalize: bool = False) -> np.ndarray:
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
    if x.size == 0:
        return np.array([])

    y = (x - loc) / scale
    g_pdf_ = gaussian_pdf_(x=y, normalize=True)
    g_cdf_ = gaussian_cdf_(x=shape * y, normalize=True)

    pdf_ = (2 / scale) * g_pdf_ * g_cdf_

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return _remove_nans(pdf_)


@doc_inherit(parent=skew_normal_pdf_, style=doc_style)
def skew_normal_cdf_(x: np.ndarray,
                     amplitude: float = 1.0, shape: float = 0.0, loc: float = 0.0, scale: float = 1.0,
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
    if x.size == 0:
        return np.array([])

    y = (x - loc) / scale
    return gaussian_cdf_(x=y, normalize=True) - 2 * owens_t(y, shape)


def _beta_masking(y: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Creates a mask for beta distributions to identify out-of-range or undefined values.

    Parameters
    ----------
    y : np.ndarray
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


def _pdf_scaling(pdf_: np.ndarray, amplitude: float) -> np.ndarray:
    """
    Scales a probability density function (PDF) by a given amplitude.

    Parameters
    ----------
    pdf_ : np.ndarray
        The input PDF array to be scaled.
    amplitude : float
        The amplitude to scale the PDF.

    Returns
    -------
    np.ndarray
        The scaled PDF array.
    """
    return amplitude * (pdf_ / np.max(pdf_))


def _remove_nans(x: np.ndarray) -> np.ndarray:
    """
    Replaces NaN, positive infinity, and negative infinity values in an array.

    Parameters
    ----------
    x : np.ndarray
        Input array that may contain NaN, positive infinity, or negative infinity values.

    Returns
    -------
    np.ndarray
        Array with NaN replaced by 0, positive infinity replaced by `np.inf`, and negative
    infinity replaced by `-np.inf`.
    """
    return np.nan_to_num(x=x, copy=False, nan=0, posinf=np.inf, neginf=-np.inf)
