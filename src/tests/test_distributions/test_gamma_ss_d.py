"""Created on Dec 10 00:56:43 2024"""

import numpy as np
import pytest
from scipy.stats import gamma

from ...pymultifit import EPSILON
from ...pymultifit.distributions import GammaDistributionSS
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

        def _cdf_pdf_scipy(x_, shape, scale, what='cdf'):
            return gamma.cdf(x_, a=shape, scale=scale) if what == 'cdf' else gamma.pdf(x_, a=shape, scale=scale)

        for i in ['cdf', 'pdf']:
            for _ in range(100):  # Run 50 random tests
                shape_ = np.random.uniform(low=EPSILON, high=50)  # Shape parameter (alpha > 0)
                scale_ = np.random.uniform(low=EPSILON, high=30)  # Scale parameter (scale > 0)

                x = np.random.uniform(low=0, high=15, size=50)  # Generate valid x values
                distribution = GammaDistributionSS(shape=shape_, scale=scale_, normalize=True)

                expected = _cdf_pdf_scipy(x_=x, shape=shape_, scale=scale_, what=i)
                actual = _cdf_pdf_custom(x_=x, dist_=distribution, what=i)

                np.testing.assert_allclose(actual=actual, desired=expected, rtol=1e-5, atol=1e-8)
