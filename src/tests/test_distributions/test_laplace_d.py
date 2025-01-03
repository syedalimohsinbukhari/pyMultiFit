"""Created on Dec 10 16:33:01 2024"""

import numpy as np
import pytest
from scipy.stats import laplace

from ...pymultifit import EPSILON
from ...pymultifit.distributions import LaplaceDistribution
from ...pymultifit.distributions.backend import errorHandling as erH


class TestLaplaceDistribution:

    @staticmethod
    def test_initialization():
        dist = LaplaceDistribution(amplitude=2.0, mean=1.0, diversity=0.5, normalize=False)
        assert dist.amplitude == 2.0
        assert dist.mu == 1.0
        assert dist.b == 0.5
        assert not dist.norm

        dist_normalized = LaplaceDistribution(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1.0

    @staticmethod
    def test_constraints():
        with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
            LaplaceDistribution(amplitude=-1.0, normalize=False)

        # amplitude should be internally updated to 1.0 if `normalize` is called
        distribution = LaplaceDistribution(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

        with pytest.raises(erH.NegativeScaleError, match=f"Diversity {erH.neg_message}"):
            LaplaceDistribution(amplitude=1.0, diversity=-3.0, normalize=True)

    @staticmethod
    def test_edge_case():
        dist = LaplaceDistribution()
        x = np.array([])
        result = dist.pdf(x)
        assert result.size == 0  # Should return an empty array

    @staticmethod
    def test_stats():
        loc_ = np.random.uniform(low=-5, high=10, size=10)
        scale_ = np.random.uniform(low=EPSILON, high=10, size=10)
        stack_ = np.column_stack([loc_, scale_])

        for loc, scale in stack_:
            _distribution = LaplaceDistribution.scipy_like(loc=loc, scale=scale)
            d_stats = _distribution.stats()

            # Scipy calculations
            scipy_mean, scipy_variance = laplace.stats(loc=loc, scale=scale, moments='mv')
            scipy_median = laplace.median(loc=loc, scale=scale)
            scipy_stddev = np.sqrt(scipy_variance)

            # Assertions for mean and variance
            np.testing.assert_allclose(actual=scipy_mean, desired=d_stats['mean'], rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=scipy_variance, desired=d_stats['variance'], rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=scipy_median, desired=d_stats['median'], rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=scipy_stddev, desired=d_stats['std'], rtol=1e-5, atol=1e-8)

    @staticmethod
    def test_pdf_cdf():
        def _cdf_pdf_custom(x_, dist_, what='cdf'):
            return dist_.cdf(x_) if what == 'cdf' else dist_.pdf(x_)

        def _cdf_pdf_scipy(x_, loc, scale, what='cdf'):
            return laplace.cdf(x_, loc=loc, scale=scale) if what == 'cdf' else laplace.pdf(x_, loc=loc, scale=scale)

        for i in ['cdf', 'pdf']:
            for _ in range(50):  # Run 50 random tests
                loc_ = np.random.uniform(low=-10, high=10)
                scale_ = np.random.uniform(low=EPSILON, high=5)
                x = np.random.uniform(low=loc_ - 10, high=loc_ + 10, size=10)
                distribution = LaplaceDistribution(amplitude=1.0, mean=loc_, diversity=scale_, normalize=True)
                expected = _cdf_pdf_scipy(x_=x, loc=loc_, scale=scale_, what=i)
                np.testing.assert_allclose(actual=_cdf_pdf_custom(x_=x, dist_=distribution, what=i),
                                           desired=expected, rtol=1e-5, atol=1e-8)
