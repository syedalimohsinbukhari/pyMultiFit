"""Created on Dec 14 06:40:29 2024"""

import numpy as np
import pytest
from scipy.stats import expon

from ...pymultifit.distributions import ExponentialDistribution
from ...pymultifit.distributions.backend import errorHandling as erH


class TestGaussianDistribution:

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
        dist_ = ExponentialDistribution(amplitude=1.0, scale=2.0)
        d_stats = dist_.stats()
        assert d_stats["mean"] == 1 / dist_.scale
        assert d_stats["median"] == np.log(2) / dist_.scale
        assert d_stats["mode"] == 0
        assert d_stats["variance"] == 1 / dist_.scale**2

    @staticmethod
    def test_pdf_cdf_exponential():
        def _cdf_pdf_custom(x_, dist_, what='cdf'):
            return dist_.cdf(x_) if what == 'cdf' else dist_.pdf(x_)

        def _cdf_pdf_scipy(x_, scale, what='cdf'):
            return expon.cdf(x_, scale=scale) if what == 'cdf' else expon.pdf(x_, scale=scale)

        for i in ['cdf', 'pdf']:
            for _ in range(50):  # Run 50 random tests
                rate_ = np.random.uniform(low=0.1, high=2.0)

                x = np.random.uniform(low=0.01, high=20.0, size=50)
                distribution = ExponentialDistribution(scale=rate_, normalize=True)

                expected = _cdf_pdf_scipy(x_=x, scale=1 / rate_, what=i)  # Scipy uses scale=1/lambda
                actual = _cdf_pdf_custom(x_=x, dist_=distribution, what=i)

                np.testing.assert_allclose(actual=actual, desired=expected, rtol=1e-5, atol=1e-8)
