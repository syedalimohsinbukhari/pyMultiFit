"""Created on Dec 14 06:55:53 2024"""

import numpy as np
import pytest
from scipy.special import betaincinv
from scipy.stats import beta

from ...pymultifit.distributions import BetaDistribution
from ...pymultifit.distributions.backend import errorHandling as erH


class TestBetaDistribution:

    @staticmethod
    def test_initialization():
        dist_ = BetaDistribution(amplitude=2.0, normalize=False)
        assert dist_.amplitude == 2.0
        assert dist_.alpha == 1.0
        assert dist_.beta == 1.0
        assert not dist_.norm

        dist_normalized = BetaDistribution(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1.0

    @staticmethod
    def test_constraints():
        with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
            BetaDistribution(amplitude=-1.0, normalize=False)

        # amplitude should be internally updated to 1.0 if `normalize` is called
        distribution = BetaDistribution(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

        with pytest.raises(erH.NegativeAlphaError, match=f"Alpha {erH.neg_message}"):
            BetaDistribution(alpha=-3.0)

        with pytest.raises(erH.NegativeBetaError, match=f"Beta {erH.neg_message}"):
            BetaDistribution(beta=-3.0)

        with pytest.raises(erH.XOutOfRange, match="X out of range."):
            BetaDistribution().pdf(np.array([-1, 0, 1, 2]))

    @staticmethod
    def test_edge_case():
        dist = BetaDistribution()
        x = np.array([])
        result = dist.pdf(x)
        assert result.size == 0  # Should return an empty array

    @staticmethod
    def test_stats():
        dist_ = BetaDistribution(amplitude=1.0, alpha=1.0, beta=2.0, normalize=True)
        a, b = dist_.alpha, dist_.beta
        d_stats = dist_.stats()
        assert d_stats["mean"] == a / (a + b)
        assert d_stats["median"] == betaincinv(a, b, 0.5)
        if np.logical_and(a > 1, b > 1):
            assert d_stats["mode"] == (a - 1) / (a + b - 2)
        elif np.logical_or(a < 0, b < 0):
            assert d_stats.get('mode', []) == []
        assert d_stats["variance"] == (a * b) / ((a + b)**2 * (a + b + 1))

    @staticmethod
    def test_stats_scipy():
        for _ in range(50):  # Run 50 random tests
            # Generate random alpha and beta values
            a_ = np.random.uniform(0.1, 5.0)  # Alpha (shape1) > 0
            b_ = np.random.uniform(0.1, 5.0)  # Beta (shape2) > 0

            dist_ = BetaDistribution(amplitude=1.0, alpha=a_, beta=b_, normalize=True)
            a, b = dist_.alpha, dist_.beta

            # Get custom stats
            d_stats = dist_.stats()

            # Get stats from scipy
            scipy_mean, scipy_variance = beta.stats(a=a, b=b, moments="mv")
            scipy_median = beta.median(a=a, b=b)
            scipy_mode = None
            if a > 1 and b > 1:
                scipy_mode = (a - 1) / (a + b - 2)

            # Perform assertions
            assert np.isclose(d_stats["mean"], scipy_mean, rtol=1e-5,
                              atol=1e-8), f"Mean mismatch: {d_stats['mean']} != {scipy_mean} for alpha={a}, beta={b}"
            assert np.isclose(d_stats["median"], scipy_median, rtol=1e-5,
                              atol=1e-8), f"Median mismatch: {d_stats['median']} != {scipy_median} for alpha={a}, beta={b}"
            if a > 1 and b > 1:
                assert np.isclose(d_stats["mode"], scipy_mode, rtol=1e-5,
                                  atol=1e-8), f"Mode mismatch: {d_stats['mode']} != {scipy_mode} for alpha={a}, beta={b}"
            elif np.logical_or(a <= 0, b <= 0):
                assert d_stats.get("mode", []) == [], f"Unexpected mode for invalid parameters: {d_stats['mode']} for alpha={a}, beta={b}"
            assert np.isclose(d_stats["variance"], scipy_variance, rtol=1e-5,
                              atol=1e-8), f"Variance mismatch: {d_stats['variance']} != {scipy_variance} for alpha={a}, beta={b}"
