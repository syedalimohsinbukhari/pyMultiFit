"""Created on Jan 29 15:42:23 2025"""

import numpy as np
from scipy.special.cython_special import gammaincc, gammainc
from scipy.stats import gennorm

from pymultifit import EPSILON
from pymultifit.distributions.backend import BaseDistribution
from pymultifit.distributions.utilities_d import _pdf_scaling


class SymmetricGeneralizedGaussianDistribution(BaseDistribution):

    def __init__(self, amplitude: float = 1.0, shape: float = 1.0, loc: float = 0.0, scale: float = 1.0,
                 normalize: bool = False):
        self.amplitude = 1. if normalize else amplitude
        self.mu = loc
        self.alpha1 = scale
        self.beta = shape

        self.norm = normalize

    @classmethod
    def scipy_like(cls, beta, loc: float = 0.0, scale: float = 1.0):
        instance = cls(shape=beta, loc=loc, scale=scale, normalize=True)
        return instance

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return gen_gaussian_pdf_(x,
                                 amplitude=self.amplitude, loc=self.mu, scale=self.alpha1, shape=self.beta,
                                 skew=self.alpha1, normalize=self.norm)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return gen_gaussian_cdf_(x,
                                 amplitude=self.amplitude, loc=self.mu, scale=self.alpha1, shape=self.beta,
                                 skew=self.alpha1, normalize=self.norm)


def gen_gaussian_pdf_(x: np.ndarray,
                      amplitude: float = 1.0, loc: float = 0.0, scale: float = 1.0, shape: float = 1.0,
                      skew: float = 1.0, normalize: bool = False) -> np.ndarray:
    # taken from https://ieeexplore.ieee.org/document/8959693
    mu, alpha1, beta, alpha2 = loc, scale, shape, skew

    f1 = (-x + mu) / alpha1
    f2 = (x - mu) / alpha2
    denominator = (alpha1 + alpha2) * gamma(1 / beta)
    numerator = np.where(x < mu, np.exp(-f1**beta), np.exp(-f2**beta))
    pdf_ = (beta * numerator) / denominator

    if not normalize:
        pdf_ = _pdf_scaling(pdf_=pdf_, amplitude=amplitude)

    return pdf_


import numpy as np
from scipy.special import gamma, gammaincc, gammainc


def gen_gaussian_cdf_(x: np.ndarray, amplitude: float = 1.0, loc: float = 0.0,
                      scale: float = 1.0, shape: float = 1.0, skew: float = 1.0, normalize: bool = False) -> np.ndarray:
    mu, alpha1, beta, alpha2 = loc, scale, shape, skew

    normalization_factor = (alpha1 + alpha2) * gamma(1 / beta)

    # Compute masks
    mask_left = x < mu
    mask_right = x >= mu

    # Compute f1 and f2 selectively using masks
    f1 = np.zeros_like(x)
    f1[mask_left] = (mu - x[mask_left]) / alpha1

    f2 = np.zeros_like(x)
    f2[mask_right] = (x[mask_right] - mu) / alpha2

    # Compute left and right CDF components
    cdf_left = np.zeros_like(x)
    cdf_right = np.zeros_like(x)

    cdf_left[mask_left] = (alpha1 / normalization_factor) * gammaincc(1 / beta, f1[mask_left]**beta)

    cdf_right[mask_right] = (
            (alpha2 / normalization_factor) * gammainc(1 / beta, f2[mask_right]**beta)
            + alpha1 / (alpha1 + alpha2)
    )

    cdf = cdf_left + cdf_right

    print(cdf.size)

    return cdf


x = np.linspace(start=-500, stop=500, num=100)

a = np.random.uniform(low=-100, high=100, size=10_000)
b = np.random.uniform(low=EPSILON, high=100, size=10_000)

for a_, b_ in zip(a, b):
    print(a_, b_)
    gen_scipy = gennorm(beta=2, loc=a_, scale=b_).cdf(x)
    gen_custom = SymmetricGeneralizedGaussianDistribution.scipy_like(beta=1, loc=a_, scale=b_).cdf(x)
    print(np.allclose(gen_scipy, gen_custom))

    # gen_scipy = gennorm(beta=2, loc=a_, scale=b_).pdf(x)
    # gen_custom = SymmetricGeneralizedGaussianDistribution.scipy_like(beta=1, loc=a_, scale=b_).pdf(x)
    # print(np.allclose(gen_scipy, gen_custom))
