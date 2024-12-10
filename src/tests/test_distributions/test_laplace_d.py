"""Created on Dec 10 16:33:01 2024"""

import numpy as np
import pytest
from scipy.stats import laplace

from ...pymultifit import EPSILON
from ...pymultifit.distributions import LaplaceDistribution
from ...pymultifit.distributions.backend import errorHandling as erH
from ...pymultifit.distributions.backend.errorHandling import neg_message


class TestLaplaceDistribution:

    @staticmethod
    def test_initialization():
        dist = LaplaceDistribution(amplitude=2.0, mean=1.0, diversity=0.5, normalize=False)
        assert dist.amplitude == 2.0
        assert dist.mu == 1.0
        assert dist.b == 0.5
        assert not dist.norm

        dist_normalized = LaplaceDistribution(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1.0

    @staticmethod
    def test_constraints():
        with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
            LaplaceDistribution(amplitude=-1.0, normalize=False)

        distribution = LaplaceDistribution(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

        with pytest.raises(erH.NegativeScaleError, match=f"Diversity {erH.neg_message}"):
            LaplaceDistribution(amplitude=1.0, diversity=-3.0, normalize=True)

    @staticmethod
    def test_pdf():
        x = np.array([0, 1, 2])
        dist = LaplaceDistribution(amplitude=1.0, mean=1.0, diversity=0.5, normalize=True)
        expected = laplace.pdf(x, loc=1.0, scale=0.5)
        np.testing.assert_allclose(actual=dist._pdf(x), desired=expected)

    @staticmethod
    def test_edge_cases():
        dist = LaplaceDistribution()
        x = np.array([])
        result = dist.pdf(x)
        assert result.size == 0  # Should return an empty array

    @staticmethod
    def test_stats():
        distribution = LaplaceDistribution(amplitude=1.0, mean=2.0, diversity=3.0)
        d_stats = distribution.stats()
        assert d_stats["mean"] == distribution.mu
        assert d_stats["median"] == distribution.mu
        assert d_stats["mode"] == distribution.mu
        assert d_stats["variance"] == 2 * distribution.b**2

    @staticmethod
    def test_cdf():
        # Test systematic parameter coverage
        x = np.linspace(EPSILON, 5, 11)  # Test CDF evaluation over a fixed range
        for mean_ in [-3, -2, -1, 0, 1, 2, 3]:
            for diversity_ in np.linspace(EPSILON, 5, 21):
                distribution = LaplaceDistribution(amplitude=1.0, mean=mean_, diversity=diversity_, normalize=True)
                expected = laplace.cdf(x, loc=mean_, scale=diversity_)
                np.testing.assert_allclose(actual=distribution.cdf(x), desired=expected, rtol=1e-5)

        # Test edge cases with specific `x` values
        for mean_ in [-10, 0, 10]:
            for diversity_ in [EPSILON, 0.5, 2.0]:
                x = np.array([mean_ - 10, mean_, mean_ + 10])  # Extreme values
                distribution = LaplaceDistribution(amplitude=1.0, mean=mean_, diversity=diversity_, normalize=True)
                expected = laplace.cdf(x, loc=mean_, scale=diversity_)
                np.testing.assert_allclose(actual=distribution.cdf(x), desired=expected, rtol=1e-5)

        # Test randomized inputs
        for _ in range(10):  # Run 10 random tests
            mean_ = np.random.uniform(-10, 10)
            diversity_ = np.random.uniform(EPSILON, 5)
            x = np.random.uniform(mean_ - 10, mean_ + 10, 10)
            distribution = LaplaceDistribution(amplitude=1.0, mean=mean_, diversity=diversity_, normalize=True)
            expected = laplace.cdf(x, loc=mean_, scale=diversity_)
            np.testing.assert_allclose(actual=distribution.cdf(x), desired=expected, rtol=1e-5)

        # Test invalid inputs
        with pytest.raises(erH.NegativeScaleError, match=f"Diversity {neg_message}"):
            LaplaceDistribution(amplitude=1.0, mean=0, diversity=0, normalize=True)
