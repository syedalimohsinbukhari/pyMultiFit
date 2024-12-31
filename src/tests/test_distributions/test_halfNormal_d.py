"""Created on Dec 16 14:36:17 2024"""

import numpy as np
import pytest
from scipy.stats import halfnorm

from ...pymultifit import EPSILON
from ...pymultifit.distributions import FoldedNormalDistribution, HalfNormalDistribution
from ...pymultifit.distributions.backend import errorHandling as erH


class TestHalfNormalDistribution:

    @staticmethod
    def test_initialization():
        dist = HalfNormalDistribution(amplitude=2.0, scale=1.0, normalize=False)
        assert dist.amplitude == 2.0
        assert dist.scale == 1.0
        assert not dist.norm

        dist_normalized = FoldedNormalDistribution(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1.0

    @staticmethod
    def test_constraints():
        with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
            HalfNormalDistribution(amplitude=-1.0, normalize=False)

        # amplitude should be internally updated to 1.0 if `normalize` is called
        distribution = HalfNormalDistribution(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

    @staticmethod
    def test_edge_case():
        dist = HalfNormalDistribution()
        x = np.array([])
        result = dist.pdf(x)
        assert result.size == 0  # Should return an empty array

    @staticmethod
    def test_pdf_cdf():
        def _cdf_pdf_custom(x_, dist_, what='cdf'):
            return dist_.cdf(x_) if what == 'cdf' else dist_.pdf(x_)

        def _cdf_pdf_scipy(x_, loc, scale, what='cdf'):
            c = abs(loc) / scale
            return halfnorm.cdf(x_, loc=0, scale=scale) if what == 'cdf' else halfnorm.pdf(x_, loc=0, scale=scale)

        for i in ['pdf', 'cdf']:
            for _ in range(50):  # Run 50 random tests
                loc_ = np.random.uniform(low=-10, high=10)
                scale_ = np.random.uniform(low=0.1, high=5)
                x = np.random.uniform(low=EPSILON, high=loc_ + 10, size=50)
                distribution = HalfNormalDistribution(scale=scale_, normalize=True)
                expected = _cdf_pdf_scipy(x_=x, loc=loc_, scale=scale_, what=i)
                expected = np.nan_to_num(expected, True, 0)
                np.testing.assert_allclose(actual=_cdf_pdf_custom(x_=x, dist_=distribution, what=i),
                                           desired=expected, rtol=1e-5, atol=1e-8)
