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
        dist = LogNormalDistribution(amplitude=2.0, mean=1.0, std=0.5, normalize=False)
        assert dist.amplitude == 2.0
        assert dist.mean == np.log(1.0)
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
        dist_ = LogNormalDistribution(amplitude=1.0, mean=2.0, std=3.0)
        d_stats = dist_.stats()
        assert d_stats["mean"] == np.exp(dist_.mean + (dist_.std**2 / 2))
        assert d_stats["median"] == np.exp(dist_.mean)
        assert d_stats["mode"] == np.exp(dist_.mean - dist_.std**2)
        assert d_stats["variance"] == (np.exp(dist_.std**2) - 1) * np.exp(2 * dist_.mean + dist_.std**2)

    @staticmethod
    def test_pdf_cdf_log_norm():
        def _cdf_pdf_custom(x_, dist_, what='cdf'):
            """Evaluate the CDF or PDF for the custom LogNormalDistribution."""
            return dist_.cdf(x_) if what == 'cdf' else dist_.pdf(x_)

        def _cdf_pdf_scipy(x_, s, loc, scale, what='cdf'):
            return lognorm.cdf(x_, s=s, scale=scale, loc=loc) if what == 'cdf' else lognorm.pdf(x_, s=s, scale=scale, loc=loc)

        for i in ['cdf', 'pdf']:
            for _ in range(50):  # Run 50 random tests
                shape_ = np.random.uniform(low=0.1, high=2.0)
                loc_ = np.random.uniform(low=0.1, high=2.0)
                scale_ = np.random.uniform(low=0.1, high=10.0)

                x = np.random.uniform(low=EPSILON, high=200.0, size=50)
                distribution = LogNormalDistribution(mean=scale_, std=shape_, loc=loc_, normalize=True)

                expected = _cdf_pdf_scipy(x_=x, s=shape_, loc=loc_, scale=scale_, what=i)
                actual = _cdf_pdf_custom(x_=x, dist_=distribution, what=i)

                np.testing.assert_allclose(actual=actual, desired=expected, rtol=1e-5, atol=1e-8)