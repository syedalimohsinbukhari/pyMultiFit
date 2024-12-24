"""Created on Aug 03 17:13:21 2024"""

__all__ = ['arc_sine_pdf_', 'arc_sine_cdf_', 'arc_sine_logpdf_',
           'beta_pdf_', 'beta_cdf_', 'beta_logpdf_',
           'chi_square_pdf_',
           'exponential_pdf_',
           'folded_normal_',
           'gamma_sr_pdf_', 'gamma_sr_cdf_',
           'gamma_ss_',
           'gaussian_pdf_',
           'half_normal_',
           'integral_check',
           'laplace_',
           'log_normal_',
           'norris2005',
           'norris2011',
           'power_law_',
           'uniform_']

import numpy as np
from custom_inherit import doc_inherit
from scipy.special import betainc, gamma, gammainc, gammaln

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

    .. math:: f(y; \\alpha, \\beta) = \\frac{y^{\\alpha - 1} (1 - y)^{\\beta - 1}}{B(\\alpha, \\beta)}

    where :math:`B(\\alpha, \\beta)` is the Beta function, and :math:`y` is a transformed value of :math:`x` such that:

    .. math:: y = \\frac{x - \\text{loc}}{\\text{scale}}

    see, :obj:`scipy.special.beta`.
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


def folded_normal_(x: np.array,
                   amplitude: float = 1., mu: float = 0.0, variance: float = 1.0,
                   normalize: bool = False) -> np.array:
    # """
    # Compute the folded half-normal distribution.
    #
    # The folded half-normal distribution is the sum of two Gaussian distributions mirrored around `mu`.
    # It is defined as the sum of a normal distribution centered at `mu` and another mirrored at `-mu`.
    #
    # Parameters
    # ----------
    # x : np.array
    #     Input array where the folded half-normal distribution is evaluated.
    # amplitude : float, optional
    #     The amplitude of the distribution. Defaults to 1. Ignored if `normalize` is set to True.
    # mu : float, optional
    #     The mean (`mu`) of the original normal distribution. Defaults to 0.0.
    # variance : float, optional
    #     The standard deviation (`sigma`) of the original normal distribution. Defaults to 1.0.
    # normalize : bool, optional
    #     If True, the distribution will be normalized so that the PDF is at most 1. Defaults to False.
    #
    # Returns
    # -------
    # np.array
    #     The computed folded half-normal distribution values for the input `x`.
    #
    # Notes
    # -----
    # The folded half-normal distribution is defined as the sum of two Gaussian
    # distributions:
    # .. math::
    #     f(x; \\mu, \\sigma) = g_1(x; \\mu, \\sigma) + g_2(x; -\\mu, \\sigma)
    #
    # where `g_1` and `g_2` are the Gaussian distributions with the specified parameters.
    # """
    sigma = np.sqrt(variance)
    mask = x >= 0
    g1 = gaussian_pdf_(x[mask], amplitude=amplitude, mu=mu, sigma=sigma, normalize=normalize)
    g2 = gaussian_pdf_(x[mask], amplitude=amplitude, mu=-mu, sigma=sigma, normalize=normalize)
    result = np.zeros_like(x)
    result[mask] = g1 + g2

    return result


def gamma_sr_pdf_(x: np.array,
                  amplitude: float = 1.0, shape: float = 1.0, rate: float = 1.0,
                  loc: float = 0.0, normalize: bool = False) -> np.array:
    r"""
    Compute the PDF for :class:`~pymultifit.distributions.gamma_d.GammaDistributionSR` with :math:`\alpha` (shape) and :math:`\lambda` (rate) parameters.

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
        f(x; \alpha, \lambda) =
        \begin{cases}
        \frac{\lambda^\alpha}{\Gamma(\alpha)} (x - \text{loc})^{\alpha - 1} e^{-\lambda (x - \text{loc})}, & x > \text{loc}, \\
        0, & x \leq \text{loc}.
        \end{cases}
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
    """
    return np.nan_to_num(x=gammainc(shape, rate * x), copy=False, nan=0)


def gamma_ss_(x: np.array,
              amplitude: float = 1., shape: float = 1., scale: float = 1.,
              normalize: bool = False) -> np.array:
    # """Compute the Gamma distribution using the shape and scale parameterization.
    #
    # This function wraps the Gamma distribution parameterized by `shape` and `rate` and provides an interface for  `shape` and `scale`.
    # The relationship between scale and rate is defined as:
    #
    # Parameters
    # ----------
    # x : np.array
    #     The input values at which to evaluate the Gamma distribution.
    # amplitude : float, optional
    #     The amplitude of the distribution. Defaults to 1. Ignored if `normalize` is set to True.
    # shape : float, optional
    #     The shape parameter (`k`) of the Gamma distribution. Defaults to 1.
    # scale : float, optional
    #     The scale parameter (`\theta`) of the Gamma distribution. Defaults to 1. The rate parameter is computed as `1 / scale`.
    # normalize : bool, optional
    #     If True, the distribution will be normalized so that the PDF is at most 1. Defaults to False.
    #
    # Returns
    # -------
    # np.array
    #     The computed Gamma distribution values for the input `x`.
    #
    # Notes
    # -----
    # The Gamma distribution parameterized by shape and scale is defined as:
    # .. math::
    #     f(x; k, \theta) = \frac{x^{k-1} e^{-x / \theta}}{\theta^k \Gamma(k)}
    #
    # where:
    #     - `k` is the shape parameter
    #     - `\theta` is the scale parameter
    #     - `\Gamma(k)` is the Gamma function.
    #
    # This function computes the equivalent Gamma distribution using the relationship between scale (`\theta`) and rate (`\beta`) where:
    # .. math::
    #     \beta = \frac{1}{\theta}.
    # """
    return gamma_sr_pdf_(x, amplitude=amplitude, shape=shape, rate=1 / scale, normalize=normalize)


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
        The standard deviation parameter, :math:`\sigma`. Defaults to 1.
    normalize : bool, optional
        If True, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Returns
    -------
    np.array
        Array of the same shape as :math:`x`, containing the evaluated values.

    Notes
    -----
    The input `x` should be a NumPy array. If `x` is a scalar, it will be treated as a single-element array.
    """
    exponent_factor = (x - mu)**2 / (2 * sigma**2)

    if normalize:
        normalization_factor = sigma * np.sqrt(2 * np.pi)
        amplitude = 1
    else:
        normalization_factor = 1

    return amplitude * (np.exp(-exponent_factor) / normalization_factor)


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
        The scale parameter (`sigma`) of the distribution, corresponding to the standard deviation of the original normal distribution.
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
        f(x; \\sigma) = \\sqrt{2/\\pi} \\frac{1}{\\sigma} e^{-x^2 / (2\\sigma^2)}

    where `x >= 0` and `\\sigma` is the scale parameter.

    The half-normal distribution is a special case of the folded half-normal distribution with `mu = 0`.
    """
    return folded_normal_(x, amplitude=amplitude, mu=0, variance=scale, normalize=normalize)


def laplace_(x: np.array,
             amplitude: float = 1., mean: float = 0., diversity: float = 1.,
             normalize: bool = False) -> np.array:
    """Compute the Laplace distribution's probability density function (PDF).

    Parameters
    ----------
    x : np.array
        Points at which to evaluate the PDF.
    amplitude : float
        The amplitude (scale) of the distribution. Defaults to 1.
    mean : float
        The mean (location parameter) of the distribution. Defaults to 0.
    diversity : float
        The diversity (scale parameter) of the distribution. Defaults to 1.
    normalize : bool, optional
        Whether to normalize the PDF. Defaults to True.

    Returns
    -------
    np.array
        The PDF values at the given points.
    """
    exponent_factor = abs(x - mean) / diversity

    if normalize:
        normalization_factor = 2 * diversity
        amplitude = 1
    else:
        normalization_factor = 1

    return amplitude * (np.exp(-exponent_factor) / normalization_factor)


def log_normal_(x: np.array,
                amplitude: float = 1., mean: float = 0., standard_deviation: float = 1.,
                normalize: bool = False) -> np.array:
    """
    Compute the Log-Normal distribution probability density function (PDF).

    The Log-Normal PDF is given by:

    f(x) = (1 / (x * sigma * sqrt(2 * pi))) * exp(- (log(x) - mu)**2 / (2 * sigma**2))

    Parameters
    ----------
    x : np.array
        The input values at which to evaluate the Log-Normal PDF. Must be positive.
    amplitude : float
        The amplitude (scale) of the distribution. Defaults to 1.
    mean : float
        The mean of the logarithm of the distribution (i.e., mu of the normal distribution in log-space). Defaults to 0.
    standard_deviation : float
        The standard deviation of the logarithm of the distribution (i.e., sigma of the normal distribution in
        log-space). Defaults to 1.
    normalize : bool
        If True, the function returns the normalized value of the PDF. Defaults to True.

    Returns
    -------
    np.array
        The probability density function values for the input values.

    Raises
    ------
    ValueError
        If any value in `x` is less than or equal to zero.

    Notes
    -----
    The input `x` must be positive because the logarithm of zero or negative numbers is undefined.
    """
    if np.any(x <= 0):
        raise ValueError("x must be positive for the log-normal distribution.")

    exponent_factor = (np.log(x) - mean)**2 / (2 * standard_deviation**2)

    if normalize:
        normalization_factor = standard_deviation * x * np.sqrt(2 * np.pi)
        amplitude = 1
    else:
        normalization_factor = 1

    return amplitude * np.exp(-exponent_factor) / normalization_factor


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


def uniform_(x: np.array,
             amplitude: float = 1.0, low: float = 0.0, high: float = 1.0,
             normalize: bool = False) -> np.array:
    # """
    # Compute the Uniform distribution probability density function (PDF).
    #
    # The Uniform PDF is given by:
    # f(x) = amplitude / (high - low) for x ∈ [low, high]
    #        0 otherwise
    #
    # Parameters
    # ----------
    # x : np.array
    #     The input values at which to evaluate the Uniform PDF.
    # low : float
    #     The lower bound of the Uniform distribution. Defaults to 0.
    # high : float
    #     The upper bound of the Uniform distribution. Defaults to 1.
    # amplitude : float
    #     The amplitude of the Uniform distribution. Defaults to 1.
    # normalize : bool
    #     If True, the function returns the normalized PDF (amplitude = 1 / (high - low)).
    #     Defaults to False.
    #
    # Returns
    # -------
    # np.array
    #     The probability density function values for the input values.
    #
    # Notes
    # -----
    # - The input `x` should be a NumPy array. If `x` is a scalar, it will be treated as a single-element array.
    # - If `normalize=True`, the amplitude is overridden to ensure the PDF integrates to 1.
    # """
    if low >= high:
        raise ValueError("`low` must be less than `high`.")

    if normalize:
        amplitude = 1.0

    # Compute the PDF values
    pdf_values = np.where((x >= low) & (x <= high), amplitude / (high - low), 0.0)

    return pdf_values
