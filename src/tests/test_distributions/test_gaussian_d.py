"""Created on Dec 08 11:48:14 2024"""

import numpy as np
import pytest
from scipy.stats import norm

from ...pymultifit.distributions import GaussianDistribution
from ...pymultifit.distributions.backend import errorHandling as erH


class TestGaussianDistribution:

    @staticmethod
    def test_initialization():
        dist = GaussianDistribution(amplitude=2.0, mean=1.0, standard_deviation=0.5, normalize=False)
        assert dist.amplitude == 2.0
        assert dist.mean == 1.0
        assert dist.std_ == 0.5
        assert not dist.norm

        dist_normalized = GaussianDistribution(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1.0

    @staticmethod
    def test_constraints():
        with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
            GaussianDistribution(amplitude=-1.0, normalize=True)

        with pytest.raises(erH.NegativeStandardDeviationError, match=f"Standard deviation {erH.neg_message}"):
            GaussianDistribution(amplitude=1.0, standard_deviation=-3.0, normalize=True)

    @staticmethod
    def test_pdf():
        x = np.array([0, 1, 2])
        dist = GaussianDistribution(amplitude=1.0, mean=1.0, standard_deviation=0.5, normalize=True)
        expected = norm.pdf(x, loc=1.0, scale=0.5)
        np.testing.assert_allclose(actual=dist._pdf(x), desired=expected)

    @staticmethod
    def test_edge_cases():
        dist = GaussianDistribution()
        x = np.array([])
        result = dist.pdf(x)
        assert result.size == 0  # Should return an empty array

    @staticmethod
    def test_stats():
        dist = GaussianDistribution(amplitude=1.0, mean=2.0, standard_deviation=3.0)
        stats = dist.stats()
        assert stats["mean"] == 2.0
        assert stats["median"] == 2.0
        assert stats["mode"] == 2.0
        assert stats["variance"] == 9.0  # Variance = std_dev^2

    @staticmethod
    def test_cdf():
        x = np.array([0, 1, 2])
        dist = GaussianDistribution(amplitude=1.0, mean=1.0, standard_deviation=0.5, normalize=True)
        expected = norm.cdf(x, loc=1.0, scale=0.5)  # Use scipy's implementation for comparison
        np.testing.assert_allclose(dist.cdf(x), expected, rtol=1e-5)
