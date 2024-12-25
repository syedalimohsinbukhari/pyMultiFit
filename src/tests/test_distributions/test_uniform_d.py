"""Created on Dec 14 03:28:42 2024"""

import numpy as np
import pytest
from scipy.stats import beta, uniform

from ...pymultifit import EPSILON
from ...pymultifit.distributions import UniformDistribution
from ...pymultifit.distributions.backend import errorHandling as erH

edge_cases = [
    (0.0, 0.0),  # Degenerate
    (0.0, EPSILON),  # Small range
    (-1e6, 1e6),  # Large range
    (3, 3)  # Another degenerate case
]


class TestUniformDistribution:

    @staticmethod
    def test_initialization():
        dist = UniformDistribution(amplitude=2.0, normalize=False)
        assert dist.amplitude == 2.0
        assert dist.low == 0.0
        assert dist.high == 1.0
        assert not dist.norm

        # normalization should make amplitude = 1
        dist_normalized = UniformDistribution(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1.0

    @staticmethod
    def test_constraints():
        with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
            UniformDistribution(amplitude=-1.0, normalize=False)

        # amplitude should be internally updated to 1.0 if `normalize` is called
        distribution = UniformDistribution(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

    @staticmethod
    def test_edge_cases():
        dist = UniformDistribution()
        x = np.array([])
        result = dist.pdf(x)
        assert result.size == 0  # Should return an empty array

    @staticmethod
    def test_stats():
        def check_stats(dist_, scipy_dist):
            """Check stats for the custom Uniform distribution against scipy."""
            stats = dist_.stats()
            scipy_mean, scipy_var = scipy_dist.stats(moments='mv')
            scipy_median = scipy_dist.median()

            np.testing.assert_allclose(actual=stats['mean'], desired=scipy_mean)
            np.testing.assert_allclose(actual=stats['median'], desired=scipy_median)
            np.testing.assert_allclose(actual=stats['variance'], desired=scipy_var)
            assert stats.get('mode', []) == []

        def _tester(low_, high_):
            distribution = UniformDistribution(amplitude=1.0, low=low_, high=high_, normalize=True)
            scipy_uniform = uniform(loc=distribution.low, scale=distribution.high)
            scipy_beta = beta(1, 1, loc=distribution.low, scale=distribution.high)
            check_stats(dist_=distribution, scipy_dist=scipy_uniform)
            check_stats(dist_=distribution, scipy_dist=scipy_beta)

        for _ in range(50):
            low_val = np.random.uniform(low=0.0, high=10.0)
            high_val = np.random.uniform(low=0.0, high=10.0)

            _tester(low_=low_val, high_=high_val)

        for low_val, high_val in edge_cases:
            print(low_val, high_val)
            _tester(low_=low_val, high_=high_val)

    @staticmethod
    def test_pdf_cdf_uniform():
        def _cdf_pdf_custom(x_, dist_, what='cdf'):
            """Evaluate the CDF or PDF for the custom UniformDistribution."""
            return dist_.cdf(x_) if what == 'cdf' else dist_.pdf(x_)

        def _cdf_pdf_scipy_u(x_, loc, scale, what='cdf'):
            """Evaluate the CDF or PDF for SciPy uniform distribution."""
            return uniform.cdf(x_, loc=loc, scale=scale) if what == 'cdf' else uniform.pdf(x_, loc=loc, scale=scale)

        def _cdf_pdf_scipy_b(x_, loc, scale, what='cdf'):
            """Evaluate the CDF or PDF for SciPy beta distribution."""
            return beta.cdf(x_, 1, 1, loc=loc, scale=scale) if what == 'cdf' else beta.pdf(x_, 1, 1, loc=loc, scale=scale)

        def _tester(low_, high_, what_):
            x = np.linspace(start=-5, stop=5, num=10)
            distribution = UniformDistribution(low=low_, high=high_, normalize=True)
            actual = _cdf_pdf_custom(x_=x, dist_=distribution, what=what_)
            expected1 = _cdf_pdf_scipy_u(x_=x, loc=low_, scale=high_, what=what_)
            expected2 = _cdf_pdf_scipy_b(x_=x, loc=low_, scale=high_, what=what_)
            np.testing.assert_allclose(actual=actual, desired=expected1, rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=actual, desired=expected2, rtol=1e-5, atol=1e-8)

        for i in ['pdf', 'cdf']:
            for _ in range(50):
                low_value = np.random.uniform(low=0.0, high=10.0)
                high_value = np.random.uniform(low=0.0, high=10.0)

                _tester(low_=low_value, high_=high_value, what_=i)

            for low_value, high_value in edge_cases:
                _tester(low_=low_value, high_=high_value, what_=i)
