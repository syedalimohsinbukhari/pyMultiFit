"""Created on Aug 03 21:35:28 2024"""

from typing import Dict, Optional

import numpy as np
from scipy.stats import skewnorm

from ._backend import BaseDistribution
from .gaussian_d import GaussianDistribution


# TODO:
#   See if normalization factor can be used to un-normalize the pdf.

class SkewedNormalDistribution(BaseDistribution):
    """Class for Skewed Normal distribution."""

    def __init__(self,
                 shape: Optional[float] = 1,
                 location: Optional[float] = 0,
                 scale: Optional[float] = 1):
        """
        Initialize a Skewed Normal Distribution.

        Parameters
        ----------
        shape : float
            The shape parameter (alpha) controlling the skewness of the distribution.
        location : float
            The location parameter (epsilon) of the skew-normal distribution.
        scale : float
            The scale parameter (omega) of the skew-normal distribution.
        """
        self.shape = shape
        self.location = location
        self.scale = scale

    @classmethod
    def from_standard_notation(cls,
                               alpha: Optional[float] = 1,
                               epsilon: Optional[float] = 0,
                               omega: Optional[float] = 1):
        """
        Create an instance of SkewedNormalDistribution using standard notation parameters.

        This class method allows for creating an instance of the SkewedNormalDistribution
        using the parameters in the standard notation where:
        - alpha (shape parameter)
        - epsilon (location parameter)
        - omega (scale parameter)

        Parameters
        ----------
        alpha : float, optional
            The shape parameter (alpha) controlling the skewness of the distribution. Default is 1.
        epsilon : float, optional
            The location parameter (epsilon) of the skew-normal distribution. Default is 0.
        omega : float, optional
            The scale parameter (omega) of the skew-normal distribution. Default is 1.

        Returns
        -------
        SkewedNormalDistribution
            An instance of SkewedNormalDistribution initialized with the given parameters.
        """
        return cls(shape=alpha, location=epsilon, scale=omega)

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the Skew-Normal distribution probability density function (PDF).

        Parameters
        ----------
        x : np.ndarray
            The input values at which to evaluate the Skew-Normal PDF.

        Returns
        -------
        np.ndarray
            The probability density function values for the input values.
        """
        return _skew_normal(x, self.shape, self.location, self.scale)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Public method to compute the PDF.

        Parameters
        ----------
        x : np.ndarray
            The input values at which to evaluate the Skew-Normal PDF.

        Returns
        -------
        np.ndarray
            The probability density function values for the input values.
        """
        return self._pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the cumulative distribution function (CDF) of the skew-normal distribution.

        Parameters
        ----------
        x : np.ndarray
            The input values at which to evaluate the CDF.

        Returns
        -------
        np.ndarray
            The cumulative distribution function values for the input values.
        """
        return skewnorm(s=self.shape, loc=self.location, scale=self.scale).cdf(x)

    def stats(self) -> Dict[str, float]:
        """
        Compute and return the mean, mode, and variance of the skew-normal distribution.

        Returns
        -------
        dict
            A dictionary containing the mean, mode, and variance of the distribution.
        """
        alpha, omega, epsilon = self.shape, self.scale, self.location
        delta = alpha / np.sqrt(1 + alpha**2)

        mean = epsilon + omega * delta * np.sqrt(2 / np.pi)

        def _m0(alpha_):
            m0 = np.sqrt(2 / np.pi) * delta
            m0 -= ((1 - np.pi / 4) * (np.sqrt(2 / np.pi) * delta)**3) / (1 - (2 / np.pi) * delta**2)
            m0 -= (2 * np.pi / abs(alpha_)) * np.exp(-(2 * np.pi / abs(alpha_))) * np.sign(alpha_)
            return m0

        mode = epsilon + omega * _m0(alpha)
        variance = omega**2 * (1 - (2 * delta**2 / np.pi))

        return {'mean': mean, 'mode': mode, 'variance': variance}


def _skew_normal(x: np.ndarray,
                 shape: Optional[float] = 0,
                 location: Optional[float] = 0,
                 scale: Optional[float] = 1) -> np.ndarray:
    """
    Compute the Skew-Normal distribution probability density function (PDF).

    The Skew-Normal PDF is given by:
    f(x) = (2 / sigma * sqrt(2 * pi)) * phi((x - mu) / sigma) * Phi(alpha * (x - mu) / sigma)
    where:
    - phi is the standard normal PDF.
    - Phi is the standard normal cumulative distribution function (CDF).

    Parameters
    ----------
    x : np.ndarray
        The input values at which to evaluate the Skew-Normal PDF.
    location : float, optional
        The location parameter (mean) of the Skew-Normal distribution.
    scale : float, optional
        The scale parameter (standard deviation) of the Skew-Normal distribution.
    shape : float, optional
        The shape parameter that controls the skewness of the distribution.

    Returns
    -------
    np.ndarray
        The probability density function values for the input values.

    Raises
    ------
    ValueError
        If `scale` is non-positive.

    Notes
    -----
    - The Skew-Normal distribution is a generalization of the normal distribution that allows for skewness.
    - The input `x` can be any real number, but `scale` must be positive.
    """
    if scale <= 0:
        raise ValueError("scale must be positive.")

    z = (x - location) / scale
    gd = GaussianDistribution()
    phi_z = gd.pdf(z)

    def _phi(z_value):
        return gd.cdf(z_value)

    return 2 * scale * phi_z * _phi(shape * z)
