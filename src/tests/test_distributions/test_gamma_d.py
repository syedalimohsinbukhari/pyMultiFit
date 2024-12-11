"""Created on Dec 10 00:56:43 2024"""

import numpy as np
import pytest
from scipy.stats import gamma

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
        distribution = GammaDistributionSS(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

        with pytest.raises(erH.NegativeShapeError, match=f"Shape {erH.neg_message}"):
            GammaDistributionSS(amplitude=1.0, shape=-1.0, normalize=True)

        with pytest.raises(erH.NegativeScaleError, match=f"Scale {erH.neg_message}"):
            GammaDistributionSS(amplitude=1.0, shape=1.0, scale=-3.0, normalize=True)

    @staticmethod
    def test_edge_cases():
        dist = GammaDistributionSS()
        x = np.array([])
        result = dist.pdf(x)
        assert result.size == 0  # Should return an empty array

    @staticmethod
    def test_stats():
        def check_stats_ss(distribution):
            """check stats for the gammaSS distribution"""
            stats = distribution.stats()
            assert stats['mean'] == distribution.shape * distribution.scale
            # no median value available so check for []
            assert stats.get('median', []) == []
            if distribution.shape >= 1:
                assert stats['mode'] == (distribution.shape - 1) * distribution.scale
            else:
                assert stats['mode'] == 0
            assert stats['variance'] == distribution.shape * distribution.scale

        dist1 = GammaDistributionSS(amplitude=1.0, shape=1.0, scale=1.0, normalize=True)
        check_stats_ss(dist1)
        dist2 = GammaDistributionSS(amplitude=1.0, shape=0.5, scale=1.0, normalize=True)
        check_stats_ss(dist2)

    @staticmethod
    def test_cdf():
        x = np.array([0, 1, 2])
        distribution = GammaDistributionSS(amplitude=1.0, shape=1.0, scale=1.0, normalize=True)
        expected = gamma.cdf(x, a=1.0, scale=1.0)
        np.testing.assert_allclose(distribution.cdf(x), expected, rtol=1e-5)


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
        x = np.array([0, 1, 2])
        distribution = GammaDistributionSR(amplitude=1.0, shape=1.0, rate=2.0, normalize=True)
        expected = gamma.cdf(x, a=1.0, scale=1 / 2.0)
        np.testing.assert_allclose(distribution.cdf(x), expected, rtol=1e-5)
