"""Created on Dec 10 00:56:43 2024"""

import numpy as np
import pytest
from scipy.stats import gamma

from ...pymultifit import EPSILON
from ...pymultifit.distributions import GammaDistributionSR, GammaDistributionSS
from ...pymultifit.distributions.backend import errorHandling as erH


class TestGammaDistributionSS:
    @staticmethod
    def test_initialization():
        dist = GammaDistributionSS(amplitude=1.0, shape=1.0, scale=1.0, normalize=False)
        assert dist.amplitude == 1.0
        assert dist.shape == 1.0
        assert dist.scale == 1.0
        assert not dist.norm

        # normalization should make amplitude = 1
        dist_normalized = GammaDistributionSS(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1

    @staticmethod
    def test_constraints():
        with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
            GammaDistributionSS(amplitude=-1.0)

        # amplitude should be internally updated to 1.0 if `normalize` is called
        distribution = GammaDistributionSS(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

        with pytest.raises(erH.NegativeShapeError, match=f"Shape {erH.neg_message}"):
            GammaDistributionSS(shape=-1.0)

        with pytest.raises(erH.NegativeScaleError, match=f"Scale {erH.neg_message}"):
            GammaDistributionSS(scale=-3.0)

    @staticmethod
    def test_edge_cases():
        dist = GammaDistributionSS()
        x = np.array([])
        result = dist.pdf(x)
        assert result.size == 0  # Should return an empty array

    @staticmethod
    def test_stats():
        def check_stats_ss(dist_):
            """check stats for the gammaSS distribution"""
            stats = dist_.stats()
            assert stats['mean'] == dist_.shape * dist_.scale
            # no median value available so check for []
            assert stats.get('median', []) == []
            if dist_.shape >= 1:
                assert stats['mode'] == (dist_.shape - 1) * dist_.scale
            else:
                assert stats['mode'] == 0
            assert stats['variance'] == dist_.shape * dist_.scale

        dist1 = GammaDistributionSS(amplitude=1.0, shape=1.0, scale=1.0, normalize=True)
        check_stats_ss(dist1)
        dist2 = GammaDistributionSS(amplitude=1.0, shape=0.5, scale=1.0, normalize=True)
        check_stats_ss(dist2)

    @staticmethod
    def test_pdf_cdf():
        def _cdf_pdf_custom(x_, dist_, what='cdf'):
            return dist_.cdf(x_) if what == 'cdf' else dist_.pdf(x_)

        def _cdf_pdf_scipy(x_, loc, scale, what='cdf'):
            return gamma.cdf(x_, a=loc, scale=scale) if what == 'cdf' else gamma.pdf(x_, a=loc, scale=scale)

        for i in ['cdf', 'pdf']:
            for _ in range(50):  # Run 50 random tests
                loc_ = np.random.uniform(low=EPSILON, high=10)
                scale_ = np.random.uniform(low=EPSILON, high=5)
                x = np.random.uniform(low=loc_ - 10, high=loc_ + 10, size=50)
                distribution = GammaDistributionSS(shape=loc_, scale=scale_, normalize=True)
                expected = _cdf_pdf_scipy(x_=x, loc=loc_, scale=scale_, what=i)
                np.testing.assert_allclose(actual=_cdf_pdf_custom(x_=x, dist_=distribution, what=i),
                                           desired=expected, rtol=1e-5, atol=1e-8)


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
        distribution = GammaDistributionSR(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

        with pytest.raises(erH.NegativeShapeError, match=f"Shape {erH.neg_message}"):
            GammaDistributionSR(amplitude=1.0, shape=-1.0, normalize=True)

        with pytest.raises(erH.NegativeRateError, match=f"Rate {erH.neg_message}"):
            GammaDistributionSR(amplitude=1.0, shape=1.0, rate=-3.0, normalize=True)

    @staticmethod
    def test_edge_cases():
        dist = GammaDistributionSR()
        x = np.array([])
        result = dist.pdf(x)
        assert result.size == 0  # Should return an empty array

    @staticmethod
    def test_stats():
        def check_stats_sr(distribution):
            """check stats for the gammaSS distribution"""
            stats = distribution.stats()
            assert stats['mean'] == distribution.shape / distribution.rate
            # no median value available so check for []
            assert stats.get('median', []) == []
            if distribution.shape >= 1:
                assert stats['mode'] == (distribution.shape - 1) / distribution.rate
            else:
                assert stats['mode'] == 0
            assert stats['variance'] == distribution.shape / distribution.rate

        dist1 = GammaDistributionSR(amplitude=1.0, shape=1.0, rate=1.0, normalize=True)
        check_stats_sr(dist1)
        dist2 = GammaDistributionSR(amplitude=1.0, shape=0.5, rate=1.0, normalize=True)
        check_stats_sr(dist2)

    @staticmethod
    def test_cdf():
        # Test systematic parameter coverage
        x = np.linspace(EPSILON, 5, 11)  # Test CDF evaluation over a fixed range
        for mean_ in [EPSILON, 1, 2, 3, 4, 5, 6]:
            for diversity_ in np.linspace(EPSILON, 5, 21):
                distribution = GammaDistributionSS(amplitude=1.0, shape=mean_, scale=diversity_, normalize=True)
                expected = gamma.cdf(x, a=mean_, scale=diversity_)
                np.testing.assert_allclose(actual=distribution.cdf(x), desired=expected, rtol=1e-5)

        # Test edge cases with specific `x` values
        for mean_ in [EPSILON, 10, 100]:
            for diversity_ in [EPSILON, 0.5, 2.0]:
                x = np.array([mean_, mean_ + 10])  # Extreme values
                distribution = GammaDistributionSS(amplitude=1.0, shape=mean_, scale=diversity_, normalize=True)
                expected = gamma.cdf(x, a=mean_, scale=diversity_)
                np.testing.assert_allclose(actual=distribution.cdf(x), desired=expected, rtol=1e-5)

        # Test randomized inputs
        for _ in range(10):  # Run 10 random tests
            mean_ = np.random.uniform(EPSILON, 10)
            diversity_ = np.random.uniform(EPSILON, 5)
            x = np.random.uniform(EPSILON, mean_ + 10, 10)
            distribution = GammaDistributionSS(amplitude=1.0, shape=mean_, scale=diversity_, normalize=True)
            expected = gamma.cdf(x, a=mean_, scale=diversity_)
            np.testing.assert_allclose(actual=distribution.cdf(x), desired=expected, rtol=1e-5)

        # Test invalid inputs
        with pytest.raises(erH.NegativeShapeError, match=f"Shape {erH.neg_message}"):
            GammaDistributionSS(amplitude=1.0, shape=0, normalize=True)

        with pytest.raises(erH.NegativeScaleError, match=f"Scale {erH.neg_message}"):
            GammaDistributionSS(amplitude=1.0, scale=0, normalize=True)
