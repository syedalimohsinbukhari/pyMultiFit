"""Created on Dec 15 23:23:05 2024"""

import numpy as np
import pytest
from scipy.stats import foldnorm

from ...pymultifit import EPSILON
from ...pymultifit.distributions import FoldedNormalDistribution
from ...pymultifit.distributions.backend import errorHandling as erH


class TestFoldedNormalDistribution:

    @staticmethod
    def test_initialization():
        dist = FoldedNormalDistribution(amplitude=2.0, mean=1.0, sigma=0.5, normalize=False)
        assert dist.amplitude == 2.0
        assert dist.mean == 1.0
        assert dist.sigma == 0.5
        assert not dist.norm

        dist_normalized = FoldedNormalDistribution(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1.0

    @staticmethod
    def test_constraints():
        with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
            FoldedNormalDistribution(amplitude=-1.0, normalize=False)

        # amplitude should be internally updated to 1.0 if `normalize` is called
        distribution = FoldedNormalDistribution(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

    @staticmethod
    def test_edge_case():
        dist = FoldedNormalDistribution()
        x = np.array([])
        result = dist.pdf(x)
        assert result.size == 0  # Should return an empty array

    @staticmethod
    def test_pdf_cdf():
        np.random.seed(43)

        def _cdf_pdf_custom(x_, dist_, what='cdf'):
            return dist_.cdf(x_) if what == 'cdf' else dist_.pdf(x_)

        def _cdf_pdf_scipy(x_, c_, loc_, scale_, what='cdf'):
            return foldnorm.cdf(x_, c=c_, loc=loc_, scale=scale_) if what == 'cdf' else foldnorm.pdf(x_, c=c_, loc=loc_, scale=scale_)

        for test_type in ['cdf']:
            for _ in range(50):
                mean = np.random.uniform(low=EPSILON, high=10)
                std = np.random.uniform(low=EPSILON, high=10)
                loc = np.random.uniform(low=-10, high=10)

                x = np.linspace(-20, 20, 50)
                distribution = FoldedNormalDistribution(mean=mean, sigma=std, loc=loc, normalize=True)

                actual = _cdf_pdf_custom(x_=x, dist_=distribution, what=test_type)
                desired = _cdf_pdf_scipy(x_=x, c_=mean, loc_=loc, scale_=std, what=test_type)

                np.testing.assert_allclose(actual=actual, desired=desired, rtol=1e-5, atol=1e-8)
