"""Created on Dec 14 04:10:02 2024"""

import numpy as np
import pytest
from scipy.stats import lognorm

from ...pymultifit import EPSILON
from ...pymultifit.distributions import LogNormalDistribution
from ...pymultifit.distributions.backend import errorHandling as erH


class TestLogNormalDistribution:

    @staticmethod
    def test_initialization():
        dist = LogNormalDistribution(amplitude=2.0, mu=1.0, std=0.5, normalize=False)
        assert dist.amplitude == 2.0
        assert dist.mu == np.log(1.0)
        assert dist.std == 0.5
        assert not dist.norm

        dist_normalized = LogNormalDistribution(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1.0

    @staticmethod
    def test_constraints():
        with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
            LogNormalDistribution(amplitude=-1.0, normalize=False)

        # amplitude should be internally updated to 1.0 if `normalize` is called
        distribution = LogNormalDistribution(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

        with pytest.raises(erH.NegativeStandardDeviationError, match=f"Standard deviation {erH.neg_message}"):
            LogNormalDistribution(std=-3.0)

    @staticmethod
    def test_edge_case():
        dist = LogNormalDistribution()
        x = np.array([])
        result = dist.pdf(x)
        assert result.size == 0  # Should return an empty array

    @staticmethod
    def test_stats():
        shape_ = np.random.uniform(low=EPSILON, high=10, size=100)
        loc_ = np.random.uniform(low=-5, high=10, size=100)
        scale_ = np.random.uniform(low=EPSILON, high=10, size=100)
        stack_ = np.column_stack([shape_, loc_, scale_])

        for shape, loc, scale in stack_:
            _distribution = LogNormalDistribution.scipy_like(s=shape, loc=loc, scale=scale)
            d_stats = _distribution.stats()

            # Scipy calculations
            scipy_mean, scipy_variance = lognorm.stats(s=shape, loc=loc, scale=scale, moments='mv')
            scipy_median = lognorm.median(s=shape, loc=loc, scale=scale)
            scipy_stddev = np.sqrt(scipy_variance)

            # Assertions for mean and variance
            np.testing.assert_allclose(actual=scipy_mean, desired=d_stats['mean'], rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=scipy_variance, desired=d_stats['variance'], rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=scipy_median, desired=d_stats['median'], rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=scipy_stddev, desired=d_stats['std'], rtol=1e-5, atol=1e-8)

    @staticmethod
    def test_pdf_cdf_log_norm():
        def _cdf_pdf_custom(x_, dist_, what='cdf'):
            """Evaluate the CDF or PDF for the custom LogNormalDistribution."""
            return dist_.cdf(x_) if what == 'cdf' else dist_.pdf(x_)

        def _cdf_pdf_scipy(x_, s, loc, scale, what='cdf'):
            return lognorm.cdf(x_, s=s, scale=scale, loc=loc) if what == 'cdf' else lognorm.pdf(x_, s=s, scale=scale, loc=loc)

        for i in ['cdf', 'pdf']:
            for _ in range(50):  # Run 50 random tests
                shape_ = np.random.uniform(low=EPSILON, high=10)
                loc_ = np.random.uniform(low=-10, high=10)
                scale_ = np.random.uniform(low=EPSILON, high=10.0)

                x = np.random.uniform(low=EPSILON, high=200.0, size=50)
                distribution = LogNormalDistribution.scipy_like(s=shape_, loc=loc_, scale=scale_)

                expected = _cdf_pdf_scipy(x_=x, s=shape_, loc=loc_, scale=scale_, what=i)
                actual = _cdf_pdf_custom(x_=x, dist_=distribution, what=i)

                np.testing.assert_allclose(actual=actual, desired=expected, rtol=1e-5, atol=1e-8)
