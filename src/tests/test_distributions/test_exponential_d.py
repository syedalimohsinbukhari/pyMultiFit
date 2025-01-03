"""Created on Dec 14 06:40:29 2024"""

import numpy as np
import pytest
from scipy.stats import expon

from ...pymultifit import EPSILON
from ...pymultifit.distributions import ExponentialDistribution
from ...pymultifit.distributions.backend import errorHandling as erH


class TestExponentialDistribution:

    @staticmethod
    def test_initialization():
        dist_ = ExponentialDistribution(amplitude=2.0, scale=1.0, normalize=False)
        assert dist_.amplitude == 2.0
        assert dist_.scale == 1.0
        assert not dist_.norm

        dist_normalized = ExponentialDistribution(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1.0

    @staticmethod
    def test_constraints():
        with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
            ExponentialDistribution(amplitude=-1.0, normalize=False)

        # amplitude should be internally updated to 1.0 if `normalize` is called
        distribution = ExponentialDistribution(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

        with pytest.raises(erH.NegativeScaleError, match=f"Scale {erH.neg_message}"):
            ExponentialDistribution(scale=-3.0)

    @staticmethod
    def test_edge_case():
        dist = ExponentialDistribution()
        x = np.array([])
        result = dist.pdf(x)
        assert result.size == 0  # Should return an empty array

    @staticmethod
    def test_stats():
        loc_values = np.linspace(start=EPSILON, stop=10, num=500)
        scale_values = np.linspace(start=EPSILON, stop=10, num=500)
        stack_ = np.column_stack([loc_values, scale_values])

        for loc, scale_scipy in stack_:
            # Custom distribution parameters
            scale_custom = scale_scipy
            _distribution = ExponentialDistribution(amplitude=1.0, scale=scale_custom, loc=loc, normalize=True)
            d_stats = _distribution.stats()

            # Scipy calculations
            scipy_mean, scipy_variance = expon.stats(loc=loc, scale=1 / scale_scipy, moments='mv')
            scipy_median = expon.median(loc=loc, scale=1 / scale_scipy)
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
            return expon.cdf(x_, loc=loc, scale=scale) if what == 'cdf' else expon.pdf(x_, loc=loc, scale=scale)

        for i in ['pdf', 'cdf']:
            loc_ = np.random.uniform(low=EPSILON, high=20, size=500)
            lambda_ = np.random.uniform(low=EPSILON, high=2.0, size=500)
            stack_ = np.column_stack([loc_, lambda_])
            for loc, scale in stack_:
                x = np.random.uniform(low=-20, high=20.0, size=10)
                distribution = ExponentialDistribution(scale=scale, loc=loc, normalize=True)

                expected = _cdf_pdf_scipy(x_=x, loc=loc, scale=1 / scale, what=i)  # Scipy uses scale=1/lambda
                actual = _cdf_pdf_custom(x_=x, dist_=distribution, what=i)

                np.testing.assert_allclose(actual=actual, desired=expected, rtol=1e-5, atol=1e-8)
