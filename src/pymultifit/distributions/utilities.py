"""Created on Aug 03 17:13:21 2024"""

__all__ = ['arc_sine_pdf_', 'arc_sine_cdf_', 'arc_sine_logpdf_',
           'beta_pdf_', 'beta_cdf_', 'beta_logpdf_',
           'chi_square_pdf_',
           'exponential_pdf_',
           'folded_normal_pdf_',
           'gamma_sr_pdf_', 'gamma_sr_cdf_',
           'gamma_ss_pdf_',
           'gaussian_pdf_', 'gaussian_cdf_',
           'half_normal_',
           'integral_check',
           'laplace_pdf_', 'laplace_cdf_',
           'log_normal_pdf_', 'log_normal_cdf_',
           'norris2005',
           'norris2011',
           'power_law_',
           'uniform_pdf_', 'uniform_cdf_']

import numpy as np
from custom_inherit import doc_inherit
from scipy.special import betainc, erf, gamma, gammainc, gammaln

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
                  amplitude: float = 1.0, loc: float = 0.0, scale: float = 1.0,
                  normalize: bool = False) -> np.array:
    r"""
    Compute PDF of the ArcSine distribution.

    Parameters
    ----------
    x : np.array
        Input array of values where the PDF is evaluated.
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


@doc_inherit(parent=arc_sine_pdf_, style=doc_style)
def arc_sine_cdf_(x: np.array,
                  amplitude: float = 1.0, loc: float = 0.0, scale: float = 1.0,
                  normalize: bool = False) -> np.array:
    r"""
    Compute CDF of the ArcSine distribution.

    Parameters
    ----------
    amplitude: float, optional
        For API consistency only.
    normalize: bool, optional
        For API consistency only.

    Returns
    -------
    np.array
        Array of the same size as :math:`x` containing the evaluated CDF values.

    Notes
    -----
    The ArcSine CDF is defined as:

    .. math:: F(y) = \frac{2}{\pi}\arcsin\left[\sqrt{y}\right]

    where, :math:`y` is a transformed value of :math:`x`, defined as:

    .. math:: y = \frac{x - \text{loc}}{\text{scale}}
    """
    return beta_cdf_(x=x, amplitude=amplitude, alpha=0.5, beta=0.5, loc=loc, scale=scale, normalize=normalize)


@doc_inherit(parent=arc_sine_pdf_, style=doc_style)
def arc_sine_logpdf_(x: np.array,
                     amplitude: float = 1.0, loc: float = 0.0, scale: float = 1.0,
                     normalize: bool = False) -> np.array:
    r"""
    Compute logPDF of the ArcSine Distribution.

    Returns
    -------
    np.array
        Array of the same shape as `x`, containing the evaluated log-PDF values.
    """
    return beta_logpdf_(x=x, amplitude=amplitude, alpha=0.5, beta=0.5, loc=loc, scale=scale, normalize=normalize)


def beta_pdf_(x: np.array,
              amplitude: float = 1.0, alpha: float = 1.0, beta: float = 1.0, loc: float = 0.0, scale: float = 1.0,
              normalize: bool = False) -> np.array:
    r"""
    Compute PDF of the :class:`~pymultifit.distributions.beta_d.BetaDistribution`.

    Parameters
    ----------
    x : np.array
        The input array for which to compute the Beta PDF.
    amplitude : float, optional
        A scaling factor applied to the PDF. Default is 1.0.
        Ignored if **normalize** is `True`.
    alpha : float, optional
        The :math:`\alpha` parameter. Default is 1.0.
    beta : float, optional
        The :math:`\beta` parameter. Default is 1.0.
    loc : float, optional
        The location parameter :math:`-` for shifting. Default is 0.0.
    scale : float, optional
        The scale parameter - scaling. Default is 1.0.
    normalize : bool, optional
        If True, the distribution will be normalized such that the total area under the PDF equals 1.
        Defaults to False.

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
    pdf_ = np.zeros_like(y, dtype=float)

    numerator = y**(alpha - 1) * (1 - y)**(beta - 1)

    if normalize:
        normalization_factor = gamma(alpha) * gamma(beta)
        normalization_factor /= gamma(alpha + beta)
        amplitude = 1.0
    else:
        normalization_factor = 1.0

    mask_ = _beta_masking(alpha=alpha, beta=beta, y=y)

    pdf_[~mask_] = amplitude * (numerator[~mask_] / normalization_factor)

    if alpha <= 1:
        pdf_[y == 0] = np.inf
    if beta <= 1:
        pdf_[y == 1] = np.inf

    # handle the cases where nans can occur with nan_to_num
    # np.inf and -np.inf to not affect the infinite values
    return np.nan_to_num(x=pdf_ / scale, copy=False, nan=0, posinf=np.inf, neginf=-np.inf)


@doc_inherit(parent=beta_pdf_, style=doc_style)
def beta_logpdf_(x: np.array,
                 amplitude: float = 1.0, alpha: float = 1.0, beta: float = 1.0,
                 loc: float = 0.0, scale: float = 1.0,
                 normalize: bool = False) -> np.array:
    r"""
    Compute logPDF of the Beta distribution.

    Returns
    -------
    np.array
        Array of the same shape as :math:`x`, containing the evaluated log-PDF values.

    Notes
    -----

    """
    if x.size == 0:
        return np.array([])

    y = (x - loc) / scale
    logpdf_ = np.full_like(y, -np.inf, dtype=float)  # Default to -inf for invalid regions

    mask_ = _beta_masking(alpha=alpha, beta=beta, y=y)

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


def _beta_masking(alpha, beta, y):
    out_of_range_mask = np.logical_or(y < 0, y > 1)
    undefined_mask = np.zeros_like(y, dtype=bool)
    if alpha <= 1:
        undefined_mask = np.logical_or(undefined_mask, y == 0)
    if beta <= 1:
        undefined_mask = np.logical_or(undefined_mask, y == 1)
    mask_ = np.logical_or(out_of_range_mask, undefined_mask)
    return mask_


def beta_cdf_(x: np.array,
              amplitude: float = 1.0, alpha: float = 1.0, beta: float = 1.0, loc: float = 0.0, scale: float = 1.0,
              normalize: bool = False) -> np.array:
    r"""
    Compute CDF for :class:`~pymultifit.distributions.beta_d.BetaDistribution`.

    Parameters
    ----------
    x : np.array
        The input array for which to compute the Beta PDF.
    amplitude : float, optional
        For API consistency only.
    alpha : float, optional
        The shape parameter, :math:`\alpha`. Default is 1.0.
    beta : float, optional
        The shape parameter, :math:`\beta`. Default is 1.0.
    loc : float, optional
        The location parameter :math:`-` for shifting. Default is 0.0.
    scale : float, optional
        The scale parameter - scaling. Default is 1.0.
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
    cdf_ = np.zeros_like(y, dtype=float)

    mask_ = np.logical_and(y > 0, y < 1)
    cdf_[mask_] = betainc(alpha, beta, y[mask_])
    cdf_[y >= 1] = 1
    return cdf_


def chi_square_pdf_(x, amplitude: float = 1., degree_of_freedom: float = 1., normalize: bool = False):
    r"""
    Compute PDF for :mod:`~pymultifit.distributions.chiSquare_d.ChiSquareDistribution`.

    Parameters
    ----------
    x : np.array
        Input array of values where the PDF is evaluated.
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    degree_of_freedom : float, optional
        The degrees of freedom parameter. Defaults to 1.
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

    .. math:: f(y; k) = \dfrac{y^{(k/2) - 1} e^{-y/2}}{2^{k/2} \Gamma(k/2)}

    where :math:`k` is the degrees of freedom, and :math:`\Gamma(k)` is the :obj:`~scipy.special.gamma` function.
    Additionally :math:`y` is a transformed value of :math:`x`, defined as:

    .. math:: y = \dfrac{x - \text{loc}}{\text{scale}}

    The final PDF is expressed as :math:`f(y)/\text{scale}`
    """
    return gamma_sr_pdf_(x=x, amplitude=amplitude, shape=degree_of_freedom / 2., rate=0.5, normalize=normalize)


def exponential_pdf_(x: np.array,
                     amplitude: float = 1., scale: float = 1.,
                     normalize: bool = False) -> np.array:
    r"""
    Compute the PDF of :mod:`~pymultifit.distributions.exponential_d.ExponentialDistribution`.

    Parameters
    ----------
    x : np.array
        Input array of values where the PDF is evaluated, supports :math:`x \geq 0`.
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    scale : float
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
        \lambda \exp\left[-\lambda y\right] &; x \geq 0, \\
        0 &; y < 0.
        \end{cases}

    where, :math:`y` is a transformed value of :math:`x`, defined as:

    .. math:: y = \frac{x - \text{loc}}{\text{scale}}

    The final PDF is expressed as :math:`f(y)/\text{scale}`
    """
    return gamma_sr_pdf_(x, amplitude=amplitude, shape=1., rate=scale, normalize=normalize)


def folded_normal_pdf_(x: np.array,
                       amplitude: float = 1., mu: float = 0.0, variance: float = 1.0,
                       normalize: bool = False) -> np.array:
    r"""
    Compute PDF for :class:`~pymultifit.distributions.foldedNormal_d.FoldedNormalDistribution`.

    Parameters
    ----------
    x : np.array
        Input array of values where the PDF is evaluated.
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    mu : float, optional
        The mean parameter, :math:`\mu`. Defaults to 0.0.
    variance : float, optional
        The variance parameter, :math:`\sigma^2`. Defaults to 1.0.
    normalize : bool, optional
        If ``True``, the distribution will be normalized to integrate to 1.
        Defaults to False.

    Returns
    -------
    np.array
        The computed folded normal distribution values for the input `x`.

    Notes
    -----
    The PDF of :class:`~pymultifit.distributions.foldedNormal_d.FoldedNormalDistribution` is given by:

    .. math:: f(x; \mu, \sigma^2) = \phi(x; \mu, \sigma) + \phi(x; -\mu, \sigma),

    where :math:`\phi` is the PDF of :class:`~pymultifit.distributions.gaussian_d.GaussianDistribution`.
    """
    sigma = np.sqrt(variance)
    result = np.zeros_like(a=x, dtype=float)
    mask = x >= 0
    g1 = gaussian_pdf_(x[mask], amplitude=amplitude, mu=mu, sigma=sigma, normalize=normalize)
    g2 = gaussian_pdf_(x[mask], amplitude=amplitude, mu=-mu, sigma=sigma, normalize=normalize)
    result[mask] = g1 + g2

    return result


def gamma_sr_pdf_(x: np.array,
                  amplitude: float = 1.0, shape: float = 1.0, rate: float = 1.0,
                  loc: float = 0.0, normalize: bool = False) -> np.array:
    r"""
    Compute PDF for :class:`~pymultifit.distributions.gamma_d.GammaDistributionSR` with :math:`\alpha` (shape) and :math:`\lambda` (rate) parameters.

    Parameters
    ----------
    x : np.array
        Input array of values where the PDF is evaluated.
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    shape : float, optional
        The shape parameter, :math:`\alpha`. Defaults to 1.0.
    rate : float, optional
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
        \dfrac{\lambda^\alpha}{\Gamma(\alpha)} y^{\alpha - 1} e^{-\lambda y}, & y > \text{loc}, \\
        0, & y \leq \text{loc}.
        \end{cases}

    where :math:`y` is a transformed value of :math:`x`, defined as:

    .. math:: y = x - \text{loc}

    The final PDF is expressed as :math:`f(y)`.

    .. important::
        The :obj:`scipy.stats.gamma` distribution is related to this function as

        .. math:: \lambda = \frac{1}{\text{scale}},

        thus the final PDF doesn't need re-transformation by :math:`\text{scale}` division.
    """
    y = x - loc
    numerator = y**(shape - 1) * np.exp(-rate * y)

    if normalize:
        normalization_factor = gamma(shape)
        normalization_factor /= rate**shape
        amplitude = 1
    else:
        normalization_factor = 1

    pdf_ = amplitude * (numerator / normalization_factor)
    pdf_[x < loc] = 0

    return pdf_


@doc_inherit(parent=gamma_sr_pdf_, style=doc_style)
def gamma_sr_cdf_(x: np.array,
                  amplitude: float = 1., shape: float = 1., rate: float = 1.,
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
    return np.nan_to_num(x=gammainc(shape, rate * x), copy=False, nan=0)


def gamma_ss_pdf_(x: np.array,
                  amplitude: float = 1., alpha: float = 1., theta: float = 1.,
                  normalize: bool = False) -> np.array:
    r"""
    Compute PDF for :class:`~pymultifit.distributions.gamma_d.GammaDistributionSS` with :math:`\alpha` (shape) and :math:`\theta` (scale) parameters.

    Parameters
    ----------
    x : np.array
        The input values at which to evaluate the Gamma distribution.
    amplitude : float, optional
        The amplitude of the distribution. Defaults to 1. Ignored if `normalize` is set to True.
    alpha : float, optional
        The shape parameter (`k`) of the Gamma distribution. Defaults to 1.
    theta : float, optional
        The scale parameter (`\theta`) of the Gamma distribution. Defaults to 1. The rate parameter is computed as `1 / scale`.
    normalize : bool, optional
        If True, the distribution will be normalized so that the PDF is at most 1. Defaults to False.

    Returns
    -------
    np.array
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The Gamma distribution parameterized by shape and scale is defined as:
    .. math::
        f(x; k, \theta) = \frac{x^{k-1} e^{-x / \theta}}{\theta^k \Gamma(k)}

    where:
        - `k` is the shape parameter
        - `\theta` is the scale parameter
        - `\Gamma(k)` is the Gamma function.

    This function computes the equivalent Gamma distribution using the relationship between scale (`\theta`) and rate (`\beta`) where:
    .. math::
        \beta = \frac{1}{\theta}.
    """
    return gamma_sr_pdf_(x, amplitude=amplitude, shape=alpha, rate=1 / theta, normalize=normalize)


@doc_inherit(parent=gamma_ss_pdf_, style=doc_style)
def gamma_ss_cdf_(x: np.array,
                  amplitude: float = 1., alpha: float = 1.0, theta: float = 1.0,
                  normalize: bool = False) -> np.array:
    r"""
    Compute CDF for :class:`~pymultifit.distributions.gamma_d.GammaDistributionSS` with :math:`\alpha` (shape) and :math:`\theta` (scale) parmaeters.

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


def gaussian_pdf_(x: np.array,
                  amplitude: float = 1., mu: float = 0., sigma: float = 1.,
                  normalize: bool = False) -> np.array:
    r"""
    Compute PDF for the :mod:`~pymultifit.distributions.gaussian_d.GaussianDistribution`

    Parameters
    ----------
    x : np.array
        Input array of values where the PDF is evaluated.
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    mu : float
        The mean parameter, :math:`\mu`. Defaults to 0.
    sigma : float
        The standard deviation parameter, :math:`\std_`. Defaults to 1.
    normalize : bool, optional
        If True, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.array
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The Gaussian PDF is defined as:

    .. math::
        f(x; \mu, \std_) = \phi\left(\dfrac{x-\mu}{\std_}\right) =
        \dfrac{1}{\sqrt{2\pi\std_}}\exp\left[-\dfrac{1}{2}\left(\dfrac{x-\mu}{\std_}\right)^2\right]

    The final PDF is expressed as :math:`f(x)`.
    """
    exponent_factor = (x - mu)**2 / (2 * sigma**2)

    if normalize:
        normalization_factor = sigma * np.sqrt(2 * np.pi)
        amplitude = 1
    else:
        normalization_factor = 1

    return amplitude * (np.exp(-exponent_factor) / normalization_factor)


@doc_inherit(parent=gaussian_pdf_, style=doc_style)
def gaussian_cdf_(x: np.array,
                  amplitude: float = 1., mu: float = 0., sigma: float = 1.,
                  normalize: bool = False) -> np.array:
    r"""
    Compute CDF for the :mod:`~pymultifit.distributions.gaussian_d.GaussianDistribution`

    Parameters
    ----------
    amplitude: float, optional
        For API consistency only.
    normalize: float, optional
        For API consistency only.

    Returns
    -------
    np.array
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The Gaussian CDF is defined as:

    .. math::
        F(x) = \Phi\left(\dfrac{x-\mu}{\sigma}\right) =
        \dfrac{1}{2} \left[1 + \text{erf}\left(\dfrac{x - \mu}{\sigma\sqrt{2}}\right)\right]
    """
    num_ = x - mu
    den_ = sigma * np.sqrt(2)
    return 0.5 * (1 + erf(num_ / den_))


def half_normal_(x: np.array,
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


def laplace_pdf_(x: np.array,
                 amplitude: float = 1., mean: float = 0., diversity: float = 1.,
                 normalize: bool = False) -> np.array:
    r"""
    Compute PDF for :class:`~pymultifit.distributions.laplace_d.LaplaceDistribution`.

    Parameters
    ----------
    x : np.array
        Input array of values where the PDF is evaluated.
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

    if normalize:
        normalization_factor = 2 * diversity
        amplitude = 1
    else:
        normalization_factor = 1

    return amplitude * (np.exp(-exponent_factor) / normalization_factor)


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
        Input array of values where the PDF is evaluated.
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

    if normalize:
        normalization_factor = std * y * np.sqrt(2 * np.pi)
        amplitude = 1
    else:
        normalization_factor = 1

    pdf_ = amplitude * np.exp(-exponent_factor) / normalization_factor

    return np.nan_to_num(pdf_, False, 0, posinf=np.inf, neginf=-np.inf)


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
    y = x - loc
    num_ = np.log(y) - mean
    den_ = std * np.sqrt(2)
    pdf_ = 0.5 * (1 + erf(num_ / den_))
    return np.nan_to_num(pdf_, False, 0, np.inf, -np.inf)


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


def power_law_(x: np.array, amplitude: float = 1, alpha: float = -1, normalize: bool = False) -> np.array:
    """
    Compute a power-law function for a given set of x values.

    This function is designed with a `normalize` parameter for consistency with other probability density functions (PDFs).
    However, the `normalize` parameter has no effect in this function, as normalization of the power law is not handled here.

    Parameters
    ----------
    x : np.array
        Input values for which the power-law function will be evaluated.
    amplitude : float, optional
        The amplitude or scaling factor of the power law, by default 1.
    alpha : float, optional
        The exponent of the power law, by default -1.
    normalize : bool, optional
        Included for consistency with other PDF functions. Has no effect on the output. Defaults to False.

    Returns
    -------
    np.array
        Computed power-law values for each element in x.
    """
    return amplitude * x**-alpha


def uniform_pdf_(x: np.array,
                 amplitude: float = 1.0, low: float = 0.0, high: float = 1.0,
                 normalize: bool = False) -> np.array:
    r"""
    Compute PDF of :class:`~pymultifit.distributions.uniform_d.UniformDistribution`.

    Parameters
    ----------
    x : np.array
        Input array of values where the PDF is evaluated.
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
    pdf_values = np.zeros_like(a=x, dtype=float)

    if high_ == low:
        return np.full(x.size, np.nan)

    amplitude = 1.0 if normalize else amplitude
    mask_ = np.logical_and(x >= low, x <= high_)
    pdf_values[mask_] = amplitude / (high_ - low)

    return np.nan_to_num(x=pdf_values, copy=False, nan=0, posinf=np.inf, neginf=-np.inf)


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

    Returns
    -------
    np.array
        Array of the same shape as :math:`x`, containing the evaluated values.

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
