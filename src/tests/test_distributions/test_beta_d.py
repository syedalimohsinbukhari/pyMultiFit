"""Created on Dec 14 06:55:53 2024"""

import numpy as np
import pytest
from scipy.stats import beta

from ...pymultifit import EPSILON
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

    @staticmethod
    def test_edge_case():
        dist = BetaDistribution()
        x = np.array([])
        result = dist.pdf(x)
        assert result.size == 0  # Should return an empty array

    @staticmethod
    def test_stats():
        alpha_ = np.random.uniform(low=EPSILON, high=10, size=10)
        _beta = np.random.uniform(low=EPSILON, high=10, size=10)
        loc_ = np.random.uniform(low=-5, high=10, size=10)
        scale_ = np.random.uniform(low=EPSILON, high=10, size=10)
        stack_ = np.column_stack([alpha_, _beta, loc_, scale_])

        for a, b, loc, scale in stack_:
            _distribution = BetaDistribution.scipy_like(a=a, b=b, scale=scale, loc=loc)
            d_stats = _distribution.stats()

            # Scipy calculations
            scipy_mean, scipy_variance = beta.stats(a=a, b=b, loc=loc, scale=scale, moments='mv')
            scipy_median = beta.median(a=a, b=b, loc=loc, scale=scale)
            scipy_stddev = np.sqrt(scipy_variance)

            # Assertions for mean and variance
            np.testing.assert_allclose(actual=scipy_mean, desired=d_stats['mean'], rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=scipy_variance, desired=d_stats['variance'], rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=scipy_median, desired=d_stats['median'], rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=scipy_stddev, desired=d_stats['std'], rtol=1e-5, atol=1e-8)

    @staticmethod
    def test_pdf_cdf_logpdf_scipy():
        x_ = np.linspace(start=-3, stop=3, num=50)
        loc, scale = 0, 1

        def _multipy(dist, func_type):
            return dist.pdf if func_type == 'pdf' else dist.cdf

        def _scipy(a_, b_, loc_, scale_, func_type):
            return [beta(a=a_, b=b_, loc=loc_, scale=scale_).pdf(x_) if func_type == 'pdf' else
                    beta(a=a_, b=b_, loc=loc_, scale=scale_).cdf(x_)][0]

        for func_ in ['pdf', 'cdf']:
            for _ in range(50):  # Run 50 random tests
                alpha_ = np.random.uniform(low=EPSILON, high=5.0)

                for case in [1, 2, 3]:
                    if case == 1:
                        beta_ = alpha_
                    elif case == 2:
                        beta_ = np.random.uniform(low=-EPSILON, high=5.0)
                    else:
                        beta_ = np.random.uniform(low=EPSILON, high=5.0)
                        loc = np.random.uniform(low=-5.0, high=2.0)
                        scale = np.random.uniform(low=-5.0, high=5.0)

                    dist_ = BetaDistribution(amplitude=1.0, alpha=alpha_, beta=beta_, loc=loc, scale=scale,
                                             normalize=True)

                    pymul_ = _multipy(dist=dist_, func_type=func_)(x_)
                    scipy_vals = _scipy(a_=alpha_, b_=beta_, loc_=loc, scale_=scale, func_type=func_)

                    np.testing.assert_allclose(actual=pymul_, desired=scipy_vals, rtol=1e-5, atol=1e-8)

    @staticmethod
    def test_beta_edge_cases():
        x_ = np.linspace(start=-0.5, stop=1.5, num=100)
        loc, scale = 0, 1

        # Edge cases
        edge_cases = [
            (1, 1),  # Uniform distribution
            (0.5, 0.5),  # U-shaped distribution
            (2, 2),  # Symmetric uni-modal distribution
            (5, 1),  # Sharply peaked near 1
            (1, 5),  # Sharply peaked near 0
        ]

        # Extreme cases
        extreme_cases = [
            (1e-5, 1e-5),  # Almost degenerate at boundaries
            (1e5, 1e5),  # Extremely large and symmetric
            (1e-5, 5),  # Extremely sharp near 0
            (5, 1e-5),  # Extremely sharp near 1
            (1e-5, 1e5),  # Near-degenerate behavior with extreme asymmetry - 1
            (1e5, 1e-5),  # Near-degenerate behavior with extreme asymmetry - 2
        ]

        test_cases = edge_cases + extreme_cases

        for a_, b_ in test_cases:
            dist_ = BetaDistribution(amplitude=1.0, alpha=a_, beta=b_, loc=loc, scale=scale, normalize=True)

            scipy_pdf = beta(a_, b_, loc=loc, scale=scale).pdf(x_)
            scipy_cdf = beta(a_, b_, loc=loc, scale=scale).cdf(x_)

            np.testing.assert_allclose(actual=dist_.pdf(x_), desired=scipy_pdf, rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=dist_.cdf(x_), desired=scipy_cdf, rtol=1e-5, atol=1e-8)
