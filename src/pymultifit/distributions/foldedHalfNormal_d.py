"""Created on Dec 04 03:42:42 2024"""

from typing import Any, Dict

import numpy as np
from scipy.special import erf

from .backend import BaseDistribution
from .gaussian_d import gaussian_, GaussianDistribution


class FoldedHalfNormalDistribution(BaseDistribution):
    """Class for folded half-normal distribution."""

    def __init__(self, amplitude: float = 1.0, mean: float = 0.0, variance: float = 1., normalize: bool = False):
        if variance < 0:
            raise ValueError(f"Variance for {self.__class__.__name__} cannot be < 0.")
        self.amplitude = 1. if normalize else amplitude
        self.mean = mean
        self.var_ = variance

        self.norm = normalize

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return folded_half_normal_(x, amplitude=self.amplitude, mu=self.mean, variance=self.var_, normalize=self.norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        result = np.zeros_like(x)
        mask = x >= 0
        g1_cdf = GaussianDistribution(amplitude=self.amplitude, mean=self.mean,
                                      standard_deviation=np.sqrt(self.var_), normalize=self.norm).cdf(x[mask])
        g2_cdf = GaussianDistribution(amplitude=self.amplitude, mean=-self.mean,
                                      standard_deviation=np.sqrt(self.var_), normalize=self.norm).cdf(x[mask])
        result[mask] = g1_cdf + g2_cdf
        return result

    def stats(self) -> Dict[str, Any]:
        mean_, std_ = self.mean, np.sqrt(self.var_)

        f1 = std_ * np.sqrt(2 / np.pi) * np.exp(-mean_**2 / (2 * std_**2))
        f2 = mean_ * erf(mean_ / (np.sqrt(2 * np.pi)))

        mu_y = f1 + f2
        var_y = mean_**2 + std_**2 - mu_y**2

        return {'mean': mu_y,
                'variance': var_y}


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
