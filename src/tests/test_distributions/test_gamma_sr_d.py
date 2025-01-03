"""Created on Dec 10 00:56:43 2024"""

import numpy as np
import pytest
from scipy.stats import gamma

from ...pymultifit import EPSILON
from ...pymultifit.distributions import GammaDistributionSR, GammaDistributionSS
from ...pymultifit.distributions.backend import errorHandling as erH


class TestGammaDistributionSR:
    @staticmethod
    def test_initialization():
        dist = GammaDistributionSR(amplitude=1.0, shape=1.0, rate=1.0, normalize=False)
        assert dist.amplitude == 1.0
        assert dist.shape == 1.0
        assert dist.rate == 1.0
        assert not dist.norm

        # normalization should make amplitude = 1
        dist_normalized = GammaDistributionSR(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1

    @staticmethod
    def test_constraints():
        with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
            GammaDistributionSR(amplitude=-1.0)

        # amplitude should be internally updated to 1.0 if `normalize` is called
        distribution = GammaDistributionSR(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

        with pytest.raises(erH.NegativeShapeError, match=f"Shape {erH.neg_message}"):
            GammaDistributionSR(shape=-1.0)

        with pytest.raises(erH.NegativeRateError, match=f"Rate {erH.neg_message}"):
            GammaDistributionSR(rate=-3.0)

    @staticmethod
    def test_edge_cases():
        dist = GammaDistributionSR()
        x = np.array([])
        result = dist.pdf(x)
        assert result.size == 0  # Should return an empty array

    @staticmethod
    def test_stats():
        def check_stats_sr(custom_, scipy_):
            stats = custom_.stats()

            scipy_mean = scipy_.mean()
            assert np.isclose(stats['mean'], scipy_mean, rtol=1e-6), f"Mean mismatch: {stats['mean']} vs {scipy_mean}"

            scipy_variance = scipy_.var()
            assert np.isclose(stats['variance'], scipy_variance,
                              rtol=1e-6), f"Variance mismatch: {stats['variance']} vs {scipy_variance}"

            # scipy doesn't provide mode for the distribution so I'm not doing that test

            assert stats.get('median', []) == [], "Median check failed (should be empty)."

        for _ in range(50):
            shape_ = np.random.uniform(low=EPSILON, high=10.0)
            scale_ = np.random.uniform(low=EPSILON, high=10.0)
            loc_ = np.random.uniform(low=-10.0, high=10.0)

            dist1 = GammaDistributionSR(shape=shape_, rate=scale_, loc=loc_, normalize=True)
            dist2 = GammaDistributionSS(shape=shape_, scale=1 / scale_, loc=loc_, normalize=True)

            scipy_dist = gamma(a=shape_, scale=1 / scale_, loc=loc_)

            check_stats_sr(custom_=dist1, scipy_=scipy_dist)
            check_stats_sr(custom_=dist2, scipy_=scipy_dist)

    @staticmethod
    def test_pdf_cdf():
        def _cdf_pdf_custom(x_, dist_, what='cdf'):
            return dist_.cdf(x_) if what == 'cdf' else dist_.pdf(x_)

        def _cdf_pdf_scipy(x_, shape, rate, loc, what='cdf'):
            return [gamma.cdf(x_, a=shape, scale=rate, loc=loc) if what == 'cdf' else
                    gamma.pdf(x_, a=shape, scale=rate, loc=loc)][0]

        for i in ['pdf', 'cdf']:
            for _ in range(50):  # Run 50 random tests
                loc_ = np.random.uniform(low=EPSILON, high=5)
                shape_ = np.random.uniform(low=EPSILON, high=10)
                rate_ = np.random.uniform(low=EPSILON, high=10)
                scale_ = 1 / rate_

                x = np.linspace(start=0, stop=5, num=10)  # Generate valid x values
                distribution_sr = GammaDistributionSR(shape=shape_, rate=rate_, loc=loc_, normalize=True)
                distribution_ss = GammaDistributionSS(shape=shape_, scale=scale_, loc=loc_, normalize=True)

                expected = _cdf_pdf_scipy(x_=x, shape=shape_, rate=scale_, loc=loc_, what=i)
                custom1 = _cdf_pdf_custom(x_=x, dist_=distribution_sr, what=i)
                custom2 = _cdf_pdf_custom(x_=x, dist_=distribution_ss, what=i)

                np.testing.assert_allclose(actual=custom1, desired=expected, rtol=1e-6, atol=1e-8)
                np.testing.assert_allclose(actual=custom2, desired=expected, rtol=1e-6, atol=1e-8)
