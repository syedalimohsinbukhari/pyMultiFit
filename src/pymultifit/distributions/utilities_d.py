"""Created on Aug 03 17:13:21 2024"""

__all__ = ['_sanity_check',
           'arc_sine_pdf_',
           'beta_pdf_', 'beta_cdf_', 'beta_logpdf_',
           'chi_square_pdf_',
           'exponential_pdf_', 'exponential_cdf_',
           'folded_normal_pdf_', 'folded_normal_cdf_',
           'gamma_sr_pdf_', 'gamma_sr_cdf_',
           'gamma_ss_pdf_',
           'gaussian_pdf_', 'gaussian_cdf_',
           'half_normal_pdf_', 'half_normal_cdf_',
           'integral_check',
           'laplace_pdf_', 'laplace_cdf_',
           'log_normal_pdf_', 'log_normal_cdf_',
           'norris2005',
           'norris2011',
           'power_law_',
           'skew_normal_pdf_', 'skew_normal_cdf_',
           'uniform_pdf_', 'uniform_cdf_']

import numpy as np
from custom_inherit import doc_inherit
from scipy.special import betainc, erf, gamma, gammainc, gammaln, owens_t

doc_style = 'numpy_napoleon_with_merge'


def integral_check(pdf_function, x_range: tuple) -> float:
    """
    Compute the integral of a given PDF function over a specified range.

    Parameters
    ----------
    pdf_function : function
        The PDF function to integrate.
    x_range : tuple
        The range (a, b) over which to integrate the PDF.

    Returns
    -------
    float
        The integral result of the PDF function over the specified range.
    """
    from scipy.integrate import quad
    integral = quad(lambda x: pdf_function(x), x_range[0], x_range[1])[0]
    return integral


def arc_sine_pdf_(x: np.array,
                  amplitude: float = 1.0,
                  loc: float = 0.0, scale: float = 1.0, normalize: bool = False) -> np.array:
    r"""
    Compute PDF of :class:`pymultifit.distributions.arcSine_d.ArcSineDistribution`.

    Parameters
    ----------
    x : np.array
        Input array of values.
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
    np.array
        Array of the same shape as :math:`x`, containing the evaluated PDF values.

    Notes
    -----
    The ArcSine PDF is defined as:

    .. math:: f(y) = \frac{1}{\pi \sqrt{y(1-y)}}

    where, :math:`y` is a transformed value of :math:`x`, defined as:

    .. math:: y = \frac{x - \text{loc}}{\text{scale}}

    The final PDF is expressed as :math:`f(y)/\text{scale}`
    """
    return beta_pdf_(x=x, amplitude=amplitude, alpha=0.5, beta=0.5, loc=loc, scale=scale, normalize=normalize)


def beta_pdf_(x: np.array,
              amplitude: float = 1.0, alpha: float = 1.0, beta: float = 1.0,
              loc: float = 0.0, scale: float = 1.0, normalize: bool = False) -> np.array:
    r"""
    Compute PDF of :class:`~pymultifit.distributions.beta_d.BetaDistribution`.

    Parameters
    ----------
    x : np.array
        Input array of values.
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
        The location parameter :math:`-` for shifting.
        Default is 0.0.
    scale : float, optional
        The scale parameter - scaling.
        Default is 1.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.array
        Array of the same shape as `x`, containing the evaluated PDF values.

    Notes
    -----
    The Beta PDF is defined as:

    .. math:: f(y; \alpha, \beta) = \frac{y^{\alpha - 1} (1 - y)^{\beta - 1}}{B(\alpha, \beta)}

    where :math:`B(\alpha, \beta)` is the Beta function (see, :obj:`scipy.special.beta`), and :math:`y` is a transformed value of :math:`x` such that:

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

    # handle the cases where nans can occur with nan_to_num
    # np.inf and -np.inf to not affect the infinite values
    pdf_ = _remove_nans(pdf_ / scale)

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_


def _remove_nans(x: np.array) -> np.array:
    return np.nan_to_num(x=x, copy=False, nan=0, posinf=np.inf, neginf=-np.inf)


@doc_inherit(parent=beta_pdf_, style=doc_style)
def beta_logpdf_(x: np.array,
                 amplitude: float = 1.0, alpha: float = 1.0, beta: float = 1.0,
                 loc: float = 0.0, scale: float = 1.0, normalize: bool = False) -> np.array:
    r"""
    Compute logPDF of the Beta distribution.

    Parameters
    ----------
    x : np.array
        Input array of values where the logPDF is evaluated.


    Returns
    -------
    np.array
        Array of the same shape as :math:`x`, containing the evaluated log-PDF values.
    """
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


def _beta_masking(y, alpha, beta):
    out_of_range_mask = np.logical_or(y < 0, y > 1)
    undefined_mask = np.zeros_like(a=y, dtype=bool)
    if alpha <= 1:
        undefined_mask = np.logical_or(undefined_mask, y == 0)
    if beta <= 1:
        undefined_mask = np.logical_or(undefined_mask, y == 1)
    mask_ = np.logical_or(out_of_range_mask, undefined_mask)
    return mask_


@doc_inherit(parent=beta_pdf_, style=doc_style)
def beta_cdf_(x: np.array,
              amplitude: float = 1.0, alpha: float = 1.0, beta: float = 1.0,
              loc: float = 0.0, scale: float = 1.0, normalize: bool = False) -> np.array:
    r"""
    Compute CDF for :class:`~pymultifit.distributions.beta_d.BetaDistribution`.

    Parameters
    ----------
    x : np.array
        The input array for which to compute the Beta PDF.
    amplitude : float, optional
        For API consistency only.
    normalize : bool, optional
        For API consistency only.

    Returns
    -------
    np.array
        Array of the same shape as :math:`x`, containing the evaluated CDF values.

    Notes
    -----
    The Beta CDF is defined as:

    .. math:: I_x(\alpha, \beta)

    where :math:`I_x(\alpha, \beta)` is the regularized incomplete Beta function, see :obj:`~scipy.special.betainc`.
    """
    y = (x - loc) / scale
    cdf_ = np.zeros_like(a=y, dtype=float)

    mask_ = np.logical_and(y > 0, y < 1)
    cdf_[mask_] = betainc(alpha, beta, y[mask_])
    cdf_[y >= 1] = 1

    return cdf_


def chi_square_pdf_(x: np.array,
                    amplitude: float = 1.0, degree_of_freedom: int = 1,
                    loc: float = 0.0, normalize: bool = False) -> np.array:
    r"""
    Compute PDF for :mod:`~pymultifit.distributions.chiSquare_d.ChiSquareDistribution`.

    Parameters
    ----------
    x : np.array
        Input array of values.
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    degree_of_freedom : int, optional
        The degrees of freedom parameter. Defaults to 1.
    loc : float, optional
        The location parameter, :math:`-` for shifting.
        Defaults to 0.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.array
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The ChiSquare PDF is defined as:

    .. math:: f(y\ |\ k) = \dfrac{y^{(k/2) - 1} e^{-y/2}}{2^{k/2} \Gamma(k/2)}

    where :math:`\Gamma(k)` is the :obj:`~scipy.special.gamma` function, and :math:`y` is a transformed value of :math:`x`, defined as:

    .. math:: y = x - \text{loc}

    The final PDF is expressed as :math:`f(y)`.
    """
    return gamma_sr_pdf_(x=x, amplitude=amplitude, alpha=degree_of_freedom / 2., lambda_=0.5, loc=loc, normalize=normalize)


def exponential_pdf_(x: np.array,
                     amplitude: float = 1., scale: float = 1., loc: float = 0.0,
                     normalize: bool = False) -> np.array:
    r"""
    Compute PDF of :class:`~pymultifit.distributions.exponential_d.ExponentialDistribution`.

    .. note::
        This function uses :func:`~pymultifit.distributions.utilities_d.gamma_sr_pdf_` to calculate the PDF with
        :math:`\alpha = 1` and :math:`\lambda_\text{gammaSR} = \lambda_\text{expon}`.

    Parameters
    ----------
    x : np.array
        Input array of values, supports :math:`x \geq 0`.
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    scale : float, optional
        The scale parameter, :math:`\lambda`. Defaults to 1.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.array
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The Exponential PDF is given by:

    .. math::
        f(y, \lambda) =
        \begin{cases}
        \lambda \exp\left[-\lambda y\right] &; y \geq 0, \\
        0 &; y < 0.
        \end{cases}

    where, :math:`y` is a transformed value of :math:`x`, defined as:

    .. math:: y = x - \text{loc}

    The final PDF is expressed as :math:`f(y)`.
    """
    return gamma_sr_pdf_(x, amplitude=amplitude, alpha=1., lambda_=scale, loc=loc, normalize=normalize)


@doc_inherit(parent=exponential_pdf_, style=doc_style)
def exponential_cdf_(x: np.array,
                     amplitude: float = 1., scale: float = 1., loc: float = 0.0,
                     normalize: bool = False) -> np.array:
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


def _sanity_check(x: np.array):
    if x.size == 0:
        return np.array([])
    return x


def folded_normal_pdf_(x: np.array,
                       amplitude: float = 1., mu: float = 0.0, variance: float = 1.0,
                       normalize: bool = False) -> np.array:
    r"""
    Compute PDF for :class:`~pymultifit.distributions.foldedNormal_d.FoldedNormalDistribution`.

    Parameters
    ----------
    x : np.array
        Input array of values where PDF is evaluated.
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    mu : float, optional
        The mean parameter, :math:`\mu`. Defaults to 0.0.
    variance : float, optional
        The variance parameter, :math:`\sigma^2`. Defaults to 1.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.array
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The PDF of :class:`~pymultifit.distributions.foldedNormal_d.FoldedNormalDistribution` is given by:

    .. math:: f(x; \mu, \sigma^2) = \phi(x; \mu, \sigma) + \phi(x; -\mu, \sigma),

    where :math:`\phi` is the PDF of :class:`~pymultifit.distributions.gaussian_d.GaussianDistribution`.
    """
    sigma = np.sqrt(variance)
    pdf_ = np.zeros_like(a=x, dtype=float)

    mask = x >= 0
    g1 = gaussian_pdf_(x=x[mask], mean=mu, std=sigma, normalize=True)
    g2 = gaussian_pdf_(x=x[mask], mean=-mu, std=sigma, normalize=True)
    pdf_[mask] = g1 + g2

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_


@doc_inherit(parent=folded_normal_pdf_, style=doc_style)
def folded_normal_cdf_(x: np.array,
                       amplitude: float = 1., mu: float = 0.0, variance: float = 1.0,
                       normalize: bool = False) -> np.array:
    r"""
    Compute CDF for :class:`~pymultifit.distributions.foldedNormal_d.FoldedNormalDistribution`.

    Parameters
    ----------
    amplitude: float, optional
        For API consistency only.
    normalize: bool, optional
        For API consistency only.
    """
    y = np.zeros_like(a=x, dtype=float)
    mask = x >= 0
    frac1, frac2 = (x[mask] - mu) / np.sqrt(2 * variance), (x[mask] + mu) / np.sqrt(2 * variance)
    y[mask] += 0.5 * (erf(frac1) + erf(frac2))
    return y


def gamma_sr_pdf_(x: np.array,
                  amplitude: float = 1.0, alpha: float = 1.0, lambda_: float = 1.0,
                  loc: float = 0.0, normalize: bool = False) -> np.array:
    r"""
    Compute PDF for :class:`~pymultifit.distributions.gamma_d.GammaDistributionSR` with :math:`\alpha` (shape) and :math:`\lambda` (rate) parameters.

    Parameters
    ----------
    x : np.array
        Input array of values.
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    alpha : float, optional
        The shape parameter, :math:`\alpha`. Defaults to 1.0.
    lambda_ : float, optional
        The rate parameter, :math:`\lambda`. Defaults to 1.0.
    loc : float, optional
        The location parameter :math:`-` for shifting. Defaults to 0.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.array
        Array of the same shape as :math:`x`, containing the evaluated PDF values.

    Notes
    -----
    The Gamma PDF is given by:

    .. math::
        f(y; \alpha, \lambda) =
        \begin{cases}
        \dfrac{\lambda^\alpha}{\Gamma(\alpha)} y^{\alpha - 1} \exp\left[-\lambda y\right], & y > \text{loc}, \\
        0, & y \leq \text{loc}.
        \end{cases}

    where :math:`y` is a transformed value of :math:`x`, defined as:

    .. math:: y = x - \text{loc}

    The final PDF is expressed as :math:`f(y)`.
    """
    y = x - loc
    numerator = y**(alpha - 1) * np.exp(-lambda_ * y)
    normalization_factor = gamma(alpha) / lambda_**alpha

    pdf_ = numerator / normalization_factor
    pdf_[x < loc] = 0

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_


@doc_inherit(parent=gamma_sr_pdf_, style=doc_style)
def gamma_sr_cdf_(x: np.array,
                  amplitude: float = 1.0, alpha: float = 1.0, lambda_: float = 1.0,
                  normalize: bool = False) -> np.array:
    r"""
    Compute CDF for :class:`~pymultifit.distributions.gamma_d.GammaDistributionSR` with :math:`\alpha` and :math:`\lambda` parameters.

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
        F(y; \alpha, \lambda) = \dfrac{1}{\Gamma(\alpha)}\gamma(\alpha, \lambda y)

    where, :math:`\gamma(\alpha, \lambda y)` is the lower incomplete gamma function, see :obj:`~scipy.special.gammainc`.
    """
    return gammainc(alpha, lambda_ * x)


def gamma_ss_pdf_(x: np.array,
                  amplitude: float = 1.0, alpha: float = 1.0, theta: float = 1.0,
                  normalize: bool = False) -> np.array:
    r"""
    Compute PDF for :class:`~pymultifit.distributions.gamma_d.GammaDistributionSS` with :math:`\alpha` (shape) and :math:`\theta` (scale) parameters.

    Parameters
    ----------
    x : np.array
        Input array of values.
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    alpha : float, optional
        The shape parameter, :math:`\alpha`. Defaults to 1.0.
    theta : float, optional
        The scale parameter, :math:`\lambda`. Defaults to 1.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.array
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The Gamma PDF parameterized by shape and scale is defined as:

    .. math::
        f(x; \alpha, \theta) = \frac{x^{\alpha-1} \exp^{-x / \theta}}{\theta^k \Gamma(k)}
    """
    return gamma_sr_pdf_(x=x, amplitude=amplitude, alpha=alpha, lambda_=1 / theta, normalize=normalize)


@doc_inherit(parent=gamma_ss_pdf_, style=doc_style)
def gamma_ss_cdf_(x: np.array,
                  amplitude: float = 1., alpha: float = 1.0, theta: float = 1.0,
                  normalize: bool = False) -> np.array:
    r"""
    Compute CDF for :class:`~pymultifit.distributions.gamma_d.GammaDistributionSS` with :math:`\alpha` (shape) and :math:`\theta` (scale) parameters.

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


def gaussian_pdf_(x: np.array,
                  amplitude: float = 1.0, mean: float = 0.0, std: float = 1.0,
                  normalize: bool = False) -> np.array:
    r"""
    Compute PDF for the :mod:`~pymultifit.distributions.gaussian_d.GaussianDistribution`

    Parameters
    ----------
    x : np.array
        Input array of values.
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    mean : float, optional
        The mean parameter, :math:`\mu`. Defaults to 0.0.
    std : float, optional
        The standard deviation parameter, :math:`\sigma`. Defaults to 1.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.array
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The Gaussian PDF is defined as:

    .. math::
        f(x; \mu, \sigma) = \phi\left(\dfrac{x-\mu}{\sigma}\right) =
        \dfrac{1}{\sqrt{2\pi\sigma}}\exp\left[-\dfrac{1}{2}\left(\dfrac{x-\mu}{\sigma}\right)^2\right]

    The final PDF is expressed as :math:`f(x)`.
    """
    exponent_factor = (x - mean)**2 / (2 * std**2)
    exponent_factor = np.exp(-exponent_factor)
    normalization_factor = std * np.sqrt(2 * np.pi)

    pdf_ = exponent_factor / normalization_factor

    if not normalize:
        pdf_ = pdf_ / np.max(pdf_)
        pdf_ *= amplitude

    return pdf_


@doc_inherit(parent=gaussian_pdf_, style=doc_style)
def gaussian_cdf_(x: np.array,
                  amplitude: float = 1.0, mean: float = 0.0, std: float = 1.0,
                  normalize: bool = False) -> np.array:
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
    num_ = x - mean
    den_ = std * np.sqrt(2)
    return 0.5 * (1 + erf(num_ / den_))


def half_normal_pdf_(x: np.array,
                     amplitude: float = 1.0, scale: float = 1.0, normalize: bool = False) -> np.array:
    """
    Compute the half-normal distribution.

    The half-normal distribution is a special case of the folded half-normal distribution where the mean (`mu`)  is set to 0.
    It describes the absolute value of a standard normal variable scaled by a specified factor.

    Parameters
    ----------
    x : np.array
        The input values at which to evaluate the half-normal distribution.
    amplitude : float, optional
        The amplitude of the distribution. Defaults to 1. Ignored if `normalize` is set to True.
    scale : float, optional
        The scale parameter (`std`) of the distribution, corresponding to the standard deviation of the original normal distribution.
        Defaults to 1.0.
    normalize : bool, optional
        If True, the distribution will be normalized so that the PDF is at most 1. Defaults to False.

    Returns
    -------
    np.array
        The computed half-normal distribution values for the input `x`.

    Notes
    -----
    The half-normal distribution is defined as the absolute value of a normal distribution with zero mean and specified standard deviation.
    The probability density function (PDF) is:

    .. math::
        f(x; \\std) = \\sqrt{2/\\pi} \\frac{1}{\\std} e^{-x^2 / (2\\std^2)}

    where `x >= 0` and `\\std` is the scale parameter.

    The half-normal distribution is a special case of the folded half-normal distribution with `mu = 0`.
    """
    return folded_normal_pdf_(x, amplitude=amplitude, mu=0, variance=scale, normalize=normalize)


def half_normal_cdf_(x: np.array,
                     amplitude: float = 1.0, scale: float = 1.0, normalize: bool = False) -> np.array:
    return folded_normal_cdf_(x=x, amplitude=amplitude, mean=0, std=scale, normalize=normalize)


def laplace_pdf_(x: np.array,
                 amplitude: float = 1., mean: float = 0., diversity: float = 1.,
                 normalize: bool = False) -> np.array:
    r"""
    Compute PDF for :class:`~pymultifit.distributions.laplace_d.LaplaceDistribution`.

    Parameters
    ----------
    x : np.array
        Input array of values.
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    mean : float, optional
        The mean parameter, :math:`\mu`. Defaults to 0.0.
    diversity : float, optional
        The diversity parameter, :math:`b`. Defaults to 1.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.array
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The Laplace PDF is defined as:

    .. math:: f(x\ |\ \mu, b) = \dfrac{1}{2b}\exp\left(-\dfrac{|x - \mu|}{b}\right)

    The final PDF is expressed as :math:`f(x)`.
    """
    exponent_factor = abs(x - mean) / diversity
    exponent_factor = np.exp(-exponent_factor)
    normalization_factor = 2 * diversity

    pdf_ = exponent_factor / normalization_factor

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_


def _pdf_scaling(pdf_, amplitude):
    return amplitude * (pdf_ / np.max(pdf_))


@doc_inherit(parent=laplace_pdf_, style=doc_style)
def laplace_cdf_(x: np.array,
                 amplitude: float = 1.0, mean: float = 0.0, diversity: float = 1.0,
                 normalize: bool = False) -> np.array:
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
    np.array
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


def log_normal_pdf_(x: np.array,
                    amplitude: float = 1., mean: float = 0., std: float = 1.,
                    loc: float = 0., normalize: bool = False) -> np.array:
    r"""
    Compute PDF for :class:`~pymultifit.distributions.logNormal_d.LogNormalDistribution`.

    Parameters
    ----------
    x : np.array
        Input array of values.
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    mean : float, optional
        The mean parameter, :math:`\mu`. Defaults to 0.0.
    std : float, optional
        The standard deviation parameter, :math:`\sigma`. Defaults to 1.0.
    loc : float, optional
        The location parameter, :math:`-` shifting. Defaults to 0.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.array
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The LogNormal PDF is defined as:

    .. math::
        f(y\ |\ \mu, \sigma) = \dfrac{1}{\sigma y\sqrt{2\pi}}\exp\left(-\dfrac{(\ln y - \mu)^2}{2\sigma^2}\right)

    where, :math:`y` is a transformed value of :math:`x`, defined as:

    .. math::
        y = x - \text{loc}

    The final PDF is expressed as :math:`f(y)`.
    """
    y = x - loc

    exponent_factor = (np.log(y) - mean)**2 / (2 * std**2)
    exponent_factor = np.exp(-exponent_factor)
    normalization_factor = std * y * np.sqrt(2 * np.pi)

    pdf_ = exponent_factor / normalization_factor

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return _remove_nans(pdf_)


@doc_inherit(parent=log_normal_pdf_, style=doc_style)
def log_normal_cdf_(x: np.array,
                    amplitude: float = 1.0, mean: float = 0.0, std=1.0,
                    loc: float = 0.0, normalize: bool = False) -> np.array:
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
    np.array
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The LogNormal CDF is defined as:

    .. math::
        F(x) = \Phi\left(\dfrac{\ln x - \mu}{\sigma}\right)
    """
    return _remove_nans(gaussian_cdf_(x=np.log(x - loc), mean=mean, std=std))


def norris2005(x: np.array,
               amplitude: float = 1., rise_time: float = 1., decay_time: float = 1.,
               normalize: bool = False) -> np.array:
    # """
    # Computes the Norris 2005 light curve model.
    #
    # The Norris 2005 model describes a light curve with an asymmetric shape characterized by exponential rise and decay times.
    #
    # Parameters
    # ----------
    # x : np.array
    #     The input time array at which to evaluate the light curve model.
    # amplitude : float, optional
    #     The amplitude of the light curve peak. Default is 1.0.
    # rise_time : float, optional
    #     The characteristic rise time of the light curve. Default is 1.0.
    # decay_time : float, optional
    #     The characteristic decay time of the light curve. Default is 1.0.
    # normalize : bool, optional
    #     Included for consistency with other distributions in the library.
    #     This parameter does not affect the output since normalization is not required for the Norris 2005 model. Default is False.
    #
    # Returns
    # -------
    # np.array
    #     The evaluated Norris 2005 model at the input times `x`.
    #
    # References
    # ----------
    #     Norris, J. P. (2005). ApJ, 627, 324–345.
    #     Robert, J. N. (2011). MNRAS, 419, 2, 1650-1659.
    # """
    tau = np.sqrt(rise_time * decay_time)
    xi = np.sqrt(rise_time / decay_time)

    return norris2011(x, amplitude=amplitude, tau=tau, xi=xi)


def norris2011(x: np.array,
               amplitude: float = 1., tau: float = 1., xi: float = 1.,
               normalize: bool = False) -> np.array:
    # """
    # Computes the Norris 2011 light curve model.
    #
    # The Norris 2011 model is a reformulation of the original Norris 2005 model, expressed in terms of different parameters to facilitate better
    # scaling across various energy bands in gamma-ray burst (GRB) light curves. The light curve is modeled as:
    #
    #     P(t) = A * exp(-ξ * (t / τ + τ / t))
    #
    # where τ and ξ are derived from the rise and decay times of the pulse.
    #
    # Parameters
    # ----------
    # x : np.array
    #     The input time array at which to evaluate the light curve model.
    # amplitude : float, optional
    #     The amplitude of the light curve peak (A in the formula). Default is 1.0.
    # tau : float, optional
    #     The pulse timescale parameter (τ in the formula). Default is 1.0.
    # xi : float, optional
    #     The asymmetry parameter (ξ in the formula). Default is 1.0.
    # normalize : bool, optional
    #     Included for consistency with other distributions in the library.
    #     This parameter does not affect the output since normalization is not required for the Norris 2011 model. Default is False.
    #
    # Returns
    # -------
    # np.array
    #     The evaluated Norris 2011 model at the input times `x`.
    #
    # Notes
    # -----
    # - In this parameterization, the pulse peak occurs at t_peak = τ.
    #
    # References
    # ----------
    #     Norris, J. P. (2005). ApJ, 627, 324–345.
    #     Norris, J. P. (2011). MNRAS, 419, 2, 1650–1659.
    # """
    fraction1 = x / tau
    fraction2 = tau / x
    return amplitude * np.exp(-xi * (fraction1 + fraction2))


def power_law_(x: np.array,
               amplitude: float = 1.0, alpha: float = -1.0,
               normalize: bool = False) -> np.array:
    r"""
    Compute power-law function.

    Parameters
    ----------
    x : np.array
        Input array of values where PDF is evaluated.
    amplitude : float, optional
        The amplitude or scaling factor of the power law.
        Defaults to 1.0.
    alpha : float, optional
        The exponent factor, :math:`\alpha`.
        Defaults to -1.0.
    normalize : bool, optional
        For API consistency only.

    Returns
    -------
    np.array
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The power-law function is defined as:

    .. math:: f(E\ |\ A, \alpha) = A\left(\dfrac{E}{E_\text{piv}}\right)^{-\alpha}

    where :math:`E_\text{piv}` is the pivot energy, fixed at :math:`100\,\text{keV}`.
    """
    e_pivot = 100.
    return amplitude * (x / e_pivot)**-alpha


def uniform_pdf_(x: np.array,
                 amplitude: float = 1.0, low: float = 0.0, high: float = 1.0,
                 normalize: bool = False) -> np.array:
    r"""
    Compute PDF of :class:`~pymultifit.distributions.uniform_d.UniformDistribution`.

    Parameters
    ----------
    x : np.array
        Input array of values.
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    low : float, optional
        The lower bound, :math:`a`. Defaults to 0.0.
    high : float, optional
        The upper bound, :math:`b`. Defaults to 1.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.array
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The Uniform PDF is defined as:

    .. math:: f(x\ |\ a, b) = \dfrac{1}{b-a}
    """
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
def uniform_cdf_(x: np.array,
                 amplitude: float = 1.0, low: float = 0.0, high: float = 1.0,
                 normalize: bool = False) -> np.array:
    r"""
    Compute CDF of :class:`pymultifit.distributions.uniform_d.UniformDistribution`.

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
    high = high + low

    if low == high == 0:
        return np.full(shape=x.size, fill_value=np.nan)

    cdf_values = np.zeros_like(a=x, dtype=float)
    within_bounds = (x >= low) & (x <= high)
    cdf_values[within_bounds] = (x[within_bounds] - low) / (high - low)
    cdf_values[x > high] = 1

    return cdf_values


def skew_normal_pdf_(x: np.array,
                     amplitude: float = 1.0, shape: float = 0.0, loc: float = 0.0, scale: float = 1.0,
                     normalize: bool = False) -> np.array:
    r"""
    Compute PDF of :class:`~pymultifit.distributions.skewNormal_d.SkewNormalDistribution`.

    Parameters
    ----------
    x : np.array
        Input array of values.
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
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
    np.array
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The SkewNormal PDF is defined as:

    .. math:: f(y\ |\ \alpha, \xi, \omega) =
             2\phi(y)\Phi(\alpha y)

    where, :math:`\phi(y)` and :math:`\Phi(\alpha y)` are the :class:`~pymultifit.distributions.GaussianDistribution`
    PDF and CDF defined at :math:`y` and :math:`\alpha y` respectively. Additionally, :math:`y` is a transformed value of :math:`x`, defined as:

    .. math:: y = \dfrac{x - \xi}{\omega}

    The final PDF is expressed as :math:`f(y)/\omega`.
    """
    y = (x - loc) / scale
    g_pdf_ = gaussian_pdf_(x=y, normalize=True)
    g_cdf_ = gaussian_cdf_(x=shape * y, normalize=True)

    pdf_ = (2 / scale) * g_pdf_ * g_cdf_

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return _remove_nans(pdf_)


@doc_inherit(parent=skew_normal_pdf_, style=doc_style)
def skew_normal_cdf_(x: np.array,
                     amplitude: float = 1.0, shape: float = 0.0, loc: float = 0.0, scale: float = 1.0,
                     normalize: bool = False):
    r"""
    Compute CDF of :class:`~pymultifit.distributions.SkewNormalDistribution`.

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

    where, :math:`T` is the Owen's T function, see :obj:`scipy.specials.owens_t`, and
    :math:`\Phi(\cdot)` is the :class:`~pymultifit.distributions.gaussian_d.GaussianDistribution` CDF function.
    """
    y = (x - loc) / scale
    return gaussian_cdf_(x=y, normalize=True) - 2 * owens_t(y, shape)