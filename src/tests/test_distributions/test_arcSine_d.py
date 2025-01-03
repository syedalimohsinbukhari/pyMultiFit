"""Created on Dec 15 19:24:18 2024"""

import numpy as np
from scipy.stats import arcsine, beta

from ...pymultifit import EPSILON
from ...pymultifit.distributions.arcSine_d import ArcSineDistribution


class TestArcSineDistribution:

    @staticmethod
    def test_initialization():
        dist_ = ArcSineDistribution(amplitude=2.0, normalize=False)
        assert dist_.amplitude == 2.0
        assert dist_.loc == 0.0
        assert dist_.scale == 1.0
        assert not dist_.norm

        x = np.linspace(start=0, stop=1, num=100)
        _distribution1 = ArcSineDistribution(normalize=True)
        _distribution2 = beta.pdf(x, a=0.5, b=0.5)

        np.testing.assert_allclose(actual=_distribution1.pdf(x), desired=_distribution2, rtol=1e-5, atol=1e-8)

    @staticmethod
    def test_stats():
        loc_ = np.random.uniform(low=-5, high=10, size=10)
        scale_ = np.random.uniform(low=EPSILON, high=10, size=10)
        stack_ = np.column_stack([loc_, scale_])

        for loc, scale in stack_:
            _distribution = ArcSineDistribution.scipy_like(loc=loc, scale=scale)
            d_stats = _distribution.stats()

            # Scipy calculations
            scipy_mean, scipy_variance = arcsine.stats(loc=loc, scale=scale, moments='mv')
            scipy_median = arcsine.median(loc=loc, scale=scale)
            scipy_stddev = np.sqrt(scipy_variance)

            # Assertions for mean and variance
            np.testing.assert_allclose(actual=scipy_mean, desired=d_stats['mean'], rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=scipy_variance, desired=d_stats['variance'], rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=scipy_median, desired=d_stats['median'], rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=scipy_stddev, desired=d_stats['std'], rtol=1e-5, atol=1e-8)

    @staticmethod
    def test_pdf_cdf():
        x1 = np.linspace(start=-0.5, stop=1.5, num=10)
        x2 = np.array([0, 1])

        for x in [x1, x2]:
            distribution = ArcSineDistribution(normalize=True)
            pdf_custom = distribution.pdf(x)
            cdf_custom = distribution.cdf(x)

            pdf_scipy = arcsine.pdf(x)
            cdf_scipy = arcsine.cdf(x)

            np.testing.assert_allclose(actual=pdf_custom, desired=pdf_scipy, rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=cdf_custom, desired=cdf_scipy, rtol=1e-5, atol=1e-8)
