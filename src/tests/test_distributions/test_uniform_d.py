"""Created on Dec 14 03:28:42 2024"""

import numpy as np
import pytest

from ...pymultifit.distributions import UniformDistribution
from ...pymultifit.distributions.backend import errorHandling as erH


class TestGaussianDistribution:

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
        def check_stats(dist_):
            """check stats for the Uniform distribution"""
            stats = dist_.stats()
            assert stats['mean'] == 0.5 * (dist_.low + dist_.high)
            # no median value available so check for []
            assert stats['median'] == 0.5 * (dist_.low + dist_.high)
            assert stats['variance'] == (1 / 12.) * (dist_.high - dist_.low)**2
            assert stats.get('mode', []) == []

        distribution1 = UniformDistribution(amplitude=1.0, normalize=True)
        check_stats(distribution1)
        distribution2 = UniformDistribution(amplitude=1.0, low=-3.0, high=3.0, normalize=True)
        check_stats(distribution2)

    @staticmethod
    def test_pdf_cdf_uniform():
        def _cdf_pdf_custom(x_, dist_, what='cdf'):
            """Evaluate the CDF or PDF for the custom UniformDistribution."""
            return dist_.cdf(x_) if what == 'cdf' else dist_.pdf(x_)

        def _cdf_pdf_scipy(x_, loc, scale, what='cdf'):
            """Evaluate the CDF or PDF for SciPy's uniform distribution."""
            from scipy.stats import uniform
            return uniform.cdf(x_, loc=loc, scale=scale) if what == 'cdf' else uniform.pdf(x_, loc=loc, scale=scale)

        for i in ['cdf', 'pdf']:
            for _ in range(50):
                low_ = np.random.uniform(low=0.0, high=10.0)
                high_ = np.random.uniform(low=low_ + 0.1, high=low_ + 10.0)  # Ensure high > low

                x = np.random.uniform(low=low_ - 5, high=high_ + 5, size=50)

                distribution = UniformDistribution(low=low_, high=high_)
                scale_ = high_ - low_

                expected = _cdf_pdf_scipy(x_=x, loc=low_, scale=scale_, what=i)
                actual = _cdf_pdf_custom(x_=x, dist_=distribution, what=i)

                np.testing.assert_allclose(actual=actual, desired=expected, rtol=1e-5, atol=1e-8)
