"""Created on Aug 03 17:13:21 2024"""

__all__ = ['beta_', 'chi_squared_', 'exponential_', 'folded_half_normal_', 'gamma_sr_', 'gamma_ss_', 'gaussian_', 'half_normal_', 'integral_check',
           'laplace_', 'log_normal_', 'norris2005', 'norris2011', 'power_law_']

import numpy as np
from scipy.special import beta as beta_f, gamma


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


def beta_(x: np.ndarray,
          amplitude: float = 1., alpha: float = 1., beta: float = 1.,
          normalize: bool = False) -> np.ndarray:
    """
    Compute the beta probability density function (PDF).

    Parameters
    ----------
    x : np.ndarray
        The input array for which to compute the PDF.
    amplitude : float
        The amplitude to apply to the PDF. Default is 1.
    alpha : float
        The alpha (shape) parameter of the beta distribution. Default is 1.
    beta : float
        The beta (shape) parameter of the beta distribution. Default is 1.
    normalize : bool
        If True, the PDF is normalized using the beta function. Default is True.

    Returns
    -------
    np.ndarray
        The probability density function values for the given input.
    """
    numerator = x**(alpha - 1) * (1 - x)**(beta - 1)

    if normalize:
        normalization_factor = beta_f(alpha, beta)
        amplitude = 1.0
    else:
        normalization_factor = 1.0

    return amplitude * (numerator / normalization_factor)


def chi_squared_(x, amplitude: float = 1., degree_of_freedom: float = 1., normalize: bool = False):
    """
    Compute the Chi-Squared distribution.

    The Chi-Squared distribution is a special case of the Gamma distribution with `alpha = degree_of_freedom / 2` and `beta = 1 / 2`.

    Parameters
    ----------
    x : array-like
        The input values at which to evaluate the Chi-Squared distribution.
    amplitude : float, optional
        The amplitude of the distribution. Defaults to 1.
        Ignored if `normalize` is set to True.
    degree_of_freedom : float, optional
        The degrees of freedom of the Chi-Squared distribution. Defaults to 1.
    normalize : bool, optional
        If True, the distribution will be normalized so that the PDF is at most 1. Defaults to False.

    Returns
    -------
    array-like
        The computed Chi-Squared distribution values for the input `x`.

    Notes
    -----
    The Chi-Squared distribution is related to the Gamma distribution by:
    .. math::
        f(x; k) = \\frac{x^{(k/2) - 1} e^{-x/2}}{2^{k/2} \\Gamma(k/2)}

    where `k` is the degrees of freedom, and `Γ(k)` is the Gamma function.

    If `normalize` is True, the distribution will be scaled such that the maximum value of the PDF is 1.
    """
    return gamma_ss_(x, amplitude=amplitude, shape=degree_of_freedom / 2., scale=2., normalize=normalize)


def exponential_(x: np.ndarray,
                 amplitude: float = 1., scale: float = 1.,
                 normalize: bool = False) -> np.ndarray:
    """
    Compute the Exponential distribution probability density function (PDF).

    The Exponential PDF is given by:
    f(x; lambda) = lambda * exp(-lambda * x), for x >= 0, otherwise 0.

    Parameters
    ----------
    x : np.ndarray
        The input values at which to evaluate the Exponential PDF. Should be non-negative.
    amplitude : float
        The amplitude of the exponential distribution. Defaults to 1.
    scale : float
        The scale parameter (lambda) of the exponential distribution. Defaults to 1.
    normalize : bool
        If True, the function normalizes the PDF. Otherwise, the amplitude scales the PDF. Defaults to True.

    Returns
    -------
    np.ndarray
        The probability density function values for the input values. For values of `x < 0`, the PDF is 0.

    Notes
    -----
    The input `x` should be a NumPy array. If `x` is a scalar, it will be treated as a single-element array.
    """
    return gamma_sr_(x, amplitude=amplitude, shape=1., rate=scale, normalize=normalize)


def folded_half_normal_(x: np.ndarray,
                        amplitude: float = 1., mu: float = 0.0, variance: float = 1.0,
                        normalize: bool = False) -> np.ndarray:
    """
    Compute the folded half-normal distribution.

    The folded half-normal distribution is the sum of two Gaussian distributions mirrored around `mu`.
    It is defined as the sum of a normal distribution centered at `mu` and another mirrored at `-mu`.

    Parameters
    ----------
    x : np.ndarray
        Input array where the folded half-normal distribution is evaluated.
    amplitude : float, optional
        The amplitude of the distribution. Defaults to 1. Ignored if `normalize` is set to True.
    mu : float, optional
        The mean (`mu`) of the original normal distribution. Defaults to 0.0.
    variance : float, optional
        The standard deviation (`sigma`) of the original normal distribution. Defaults to 1.0.
    normalize : bool, optional
        If True, the distribution will be normalized so that the PDF is at most 1. Defaults to False.

    Returns
    -------
    np.ndarray
        The computed folded half-normal distribution values for the input `x`.

    Notes
    -----
    The folded half-normal distribution is defined as the sum of two Gaussian
    distributions:
    .. math::
        f(x; \\mu, \\sigma) = g_1(x; \\mu, \\sigma) + g_2(x; -\\mu, \\sigma)

    where `g_1` and `g_2` are the Gaussian distributions with the specified parameters.
    """
    sigma = np.sqrt(variance)
    mask = x >= 0
    g1 = gaussian_(x[mask], amplitude=amplitude, mu=mu, sigma=sigma, normalize=normalize)
    g2 = gaussian_(x[mask], amplitude=amplitude, mu=-mu, sigma=sigma, normalize=normalize)
    result = np.zeros_like(x)
    result[mask] = g1 + g2

    return result


def gamma_sr_(x: np.ndarray,
              amplitude: float = 1., shape: float = 1., rate: float = 1.,
              normalize: bool = False) -> np.ndarray:
    """
    Computes the Gamma distribution PDF for given parameters.

    Parameters
    ----------
    x : np.ndarray
        Values at which to evaluate the PDF.
    amplitude : float
        The scaling factor for the distribution. Defaults to 1.
    shape : float
        The shape parameter of the Gamma distribution. Defaults to 1.
    rate : float
        The rate parameter of the Gamma distribution. Defaults to 1.
    normalize : bool
        Whether to normalize the distribution (i.e., set amplitude to 1). Defaults to True.

    Returns
    -------
    np.ndarray
        The probability density function evaluated at `x`.
    """
    numerator = x**(shape - 1) * np.exp(-rate * x)

    if normalize:
        normalization_factor = gamma(shape) / rate**shape
        amplitude = 1
    else:
        normalization_factor = 1

    return amplitude * (numerator / normalization_factor)


def gamma_ss_(x: np.ndarray,
              amplitude: float = 1., shape: float = 1., scale: float = 1.,
              normalize: bool = False) -> np.ndarray:
    """
    Compute the Gamma distribution using the shape and scale parameterization.

    This function wraps the Gamma distribution parameterized by `shape` and `rate` and provides an interface for  `shape` and `scale`.
    The relationship between scale and rate is defined as:
    .. math::
        \text{rate} = \frac{1}{\text{scale}}

    Parameters
    ----------
    x : np.ndarray
        The input values at which to evaluate the Gamma distribution.
    amplitude : float, optional
        The amplitude of the distribution. Defaults to 1. Ignored if `normalize` is set to True.
    shape : float, optional
        The shape parameter (`k`) of the Gamma distribution. Defaults to 1.
    scale : float, optional
        The scale parameter (`\theta`) of the Gamma distribution. Defaults to 1. The rate parameter is computed as `1 / scale`.
    normalize : bool, optional
        If True, the distribution will be normalized so that the PDF is at most 1. Defaults to False.

    Returns
    -------
    np.ndarray
        The computed Gamma distribution values for the input `x`.

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
    return gamma_sr_(x, amplitude=amplitude, shape=shape, rate=1 / scale, normalize=normalize)


def gaussian_(x: np.ndarray,
              amplitude: float = 1., mu: float = 0., sigma: float = 1.,
              normalize: bool = False) -> np.ndarray:
    """
    Compute the Gaussian (Normal) distribution probability density function (PDF).

    The Gaussian PDF is given by:
    f(x) = (1 / (sigma * sqrt(2 * pi))) * exp(- (x - mu)**2 / (2 * sigma**2))

    Parameters
    ----------
    x : np.ndarray
        The input values at which to evaluate the Gaussian PDF.
    amplitude: float
        The amplitude of the Gaussian distribution. Defaults to 1.
    mu : float
        The mean of the Gaussian distribution. Defaults to 0.
    sigma : float
        The standard deviation of the Gaussian distribution. Defaults to 1.
    normalize : bool
        If True, the function returns the normalized value of the PDF else the PDF is not normalized. Default is True.


    Returns
    -------
    np.ndarray
        The probability density function values for the input values.

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


def half_normal_(x: np.ndarray,
                 amplitude: float = 1.0, scale: float = 1.0, normalize: bool = False) -> np.ndarray:
    """
    Compute the half-normal distribution.

    The half-normal distribution is a special case of the folded half-normal distribution where the mean (`mu`)  is set to 0.
    It describes the absolute value of a standard normal variable scaled by a specified factor.

    Parameters
    ----------
    x : np.ndarray
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
    np.ndarray
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
    return folded_half_normal_(x, amplitude=amplitude, mu=0, variance=scale, normalize=normalize)


def laplace_(x: np.ndarray,
             amplitude: float = 1., mean: float = 0., diversity: float = 1.,
             normalize: bool = False) -> np.ndarray:
    """Compute the Laplace distribution's probability density function (PDF).

    Parameters
    ----------
    x : np.ndarray
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
    np.ndarray
        The PDF values at the given points.
    """
    exponent_factor = abs(x - mean) / diversity

    if normalize:
        normalization_factor = 2 * diversity
        amplitude = 1
    else:
        normalization_factor = 1

    return amplitude * (np.exp(-exponent_factor) / normalization_factor)


def log_normal_(x: np.ndarray,
                amplitude: float = 1., mean: float = 0., standard_deviation: float = 1.,
                normalize: bool = False) -> np.ndarray:
    """
    Compute the Log-Normal distribution probability density function (PDF).

    The Log-Normal PDF is given by:

    f(x) = (1 / (x * sigma * sqrt(2 * pi))) * exp(- (log(x) - mu)**2 / (2 * sigma**2))

    Parameters
    ----------
    x : np.ndarray
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
    np.ndarray
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


def norris2005(x: np.ndarray,
               amplitude: float = 1., rise_time: float = 1., decay_time: float = 1.,
               normalize: bool = False) -> np.ndarray:
    """
    Computes the Norris 2005 light curve model.

    The Norris 2005 model describes a light curve with an asymmetric shape characterized by exponential rise and decay times.

    Parameters
    ----------
    x : np.ndarray
        The input time array at which to evaluate the light curve model.
    amplitude : float, optional
        The amplitude of the light curve peak. Default is 1.0.
    rise_time : float, optional
        The characteristic rise time of the light curve. Default is 1.0.
    decay_time : float, optional
        The characteristic decay time of the light curve. Default is 1.0.
    normalize : bool, optional
        Included for consistency with other distributions in the library.
        This parameter does not affect the output since normalization is not required for the Norris 2005 model. Default is False.

    Returns
    -------
    np.ndarray
        The evaluated Norris 2005 model at the input times `x`.

    References
    ----------
        Norris, J. P. (2005). ApJ, 627, 324–345.
        Robert, J. N. (2011). MNRAS, 419, 2, 1650-1659.
    """
    tau = np.sqrt(rise_time * decay_time)
    xi = np.sqrt(rise_time / decay_time)

    return norris2011(x, amplitude=amplitude, tau=tau, xi=xi)


def norris2011(x: np.ndarray,
               amplitude: float = 1., tau: float = 1., xi: float = 1.,
               normalize: bool = False) -> np.ndarray:
    """
    Computes the Norris 2011 light curve model.

    The Norris 2011 model is a reformulation of the original Norris 2005 model, expressed in terms of different parameters to facilitate better
    scaling across various energy bands in gamma-ray burst (GRB) light curves. The light curve is modeled as:

        P(t) = A * exp(-ξ * (t / τ + τ / t))

    where τ and ξ are derived from the rise and decay times of the pulse.

    Parameters
    ----------
    x : np.ndarray
        The input time array at which to evaluate the light curve model.
    amplitude : float, optional
        The amplitude of the light curve peak (A in the formula). Default is 1.0.
    tau : float, optional
        The pulse timescale parameter (τ in the formula). Default is 1.0.
    xi : float, optional
        The asymmetry parameter (ξ in the formula). Default is 1.0.
    normalize : bool, optional
        Included for consistency with other distributions in the library.
        This parameter does not affect the output since normalization is not required for the Norris 2011 model. Default is False.

    Returns
    -------
    np.ndarray
        The evaluated Norris 2011 model at the input times `x`.

    Notes
    -----
    - In this parameterization, the pulse peak occurs at t_peak = τ.

    References
    ----------
        Norris, J. P. (2005). ApJ, 627, 324–345.
        Norris, J. P. (2011). MNRAS, 419, 2, 1650–1659.
    """
    fraction1 = x / tau
    fraction2 = tau / x
    return amplitude * np.exp(-xi * (fraction1 + fraction2))


def power_law_(x: np.ndarray, amplitude: float = 1, alpha: float = -1, normalize: bool = False) -> np.ndarray:
    """
    Compute a power-law function for a given set of x values.

    This function is designed with a `normalize` parameter for consistency with other probability density functions (PDFs).
    However, the `normalize` parameter has no effect in this function, as normalization of the power law is not handled here.

    Parameters
    ----------
    x : np.ndarray
        Input values for which the power-law function will be evaluated.
    amplitude : float, optional
        The amplitude or scaling factor of the power law, by default 1.
    alpha : float, optional
        The exponent of the power law, by default -1.
    normalize : bool, optional
        Included for consistency with other PDF functions. Has no effect on the output. Defaults to False.

    Returns
    -------
    np.ndarray
        Computed power-law values for each element in x.
    """
    return amplitude * x**-alpha
