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
            GaussianDistribution(amplitude=-1.0, normalize=False)

        # amplitude should be internally updated to 1.0 if `normalize` is called
        distribution = GaussianDistribution(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

        with pytest.raises(erH.NegativeStandardDeviationError, match=f"Standard deviation {erH.neg_message}"):
            GaussianDistribution(standard_deviation=-3.0)

    @staticmethod
    def test_edge_case():
        dist = GaussianDistribution()
        x = np.array([])
        result = dist.pdf(x)
        assert result.size == 0  # Should return an empty array

    @staticmethod
    def test_stats():
        distribution = GaussianDistribution(amplitude=1.0, mean=2.0, standard_deviation=3.0)
        d_stats = distribution.stats()
        assert d_stats["mean"] == distribution.mean
        assert d_stats["median"] == distribution.mean
        assert d_stats["mode"] == distribution.mean
        assert d_stats["variance"] == distribution.std_**2

    @staticmethod
    def test_pdf_cdf():
        def _cdf_pdf_custom(x_, dist_, what='cdf'):
            return dist_.cdf(x_) if what == 'cdf' else dist_.pdf(x_)

        def _cdf_pdf_scipy(x_, loc, scale, what='cdf'):
            return norm.cdf(x_, loc=loc, scale=scale) if what == 'cdf' else norm.pdf(x_, loc=loc, scale=scale)

        for i in ['cdf', 'pdf']:
            for _ in range(50):  # Run 50 random tests
                loc_ = np.random.uniform(low=-10, high=10)
                scale_ = np.random.uniform(low=0.1, high=5)
                x = np.linspace(start=loc_ - 10, stop=loc_ + 10, num=50)
                distribution = GaussianDistribution(mean=loc_, standard_deviation=scale_, normalize=True)
                expected = _cdf_pdf_scipy(x_=x, loc=loc_, scale=scale_, what=i)
                np.testing.assert_allclose(actual=_cdf_pdf_custom(x_=x, dist_=distribution, what=i),
                                           desired=expected, rtol=1e-5, atol=1e-8)

    @staticmethod
    def test_gaussian_edge_cases():
        x_ = np.linspace(start=-10, stop=10, num=100)

        # Edge cases
        edge_cases = [
            (0, 1),  # Standard normal distribution (mean=0, std=1)
            (0, 0.1),  # Narrow peak at center (small std)
            (0, 10),  # Broad distribution (large std)
            (5, 1),  # Mean shifted to the right
            (-5, 1),  # Mean shifted to the left
        ]

        # Extreme cases
        extreme_cases = [
            (0, 1e-5),  # Very narrow peak (almost a delta function)
            (0, 1e5),  # Extremely broad distribution
            (1e5, 1),  # Very large mean
            (-1e5, 1),  # Very large negative mean
            (1e5, 1e-5),  # Extreme mean and narrow std
            (-1e5, 1e-5),  # Extreme negative mean and narrow std
        ]

        test_cases = edge_cases + extreme_cases

        for mean_, std_ in test_cases:
            dist_ = GaussianDistribution(amplitude=1.0, mean=mean_, standard_deviation=std_, normalize=True)

            # SciPy Gaussian distribution for comparison
            scipy_pdf = norm(loc=mean_, scale=std_).pdf(x_)
            # scipy_logpdf = norm(loc=mean_, scale=std).logpdf(x_)
            scipy_cdf = norm(loc=mean_, scale=std_).cdf(x_)

            # Assert PDF, logPDF, and CDF are close to SciPy's implementation
            np.testing.assert_allclose(actual=dist_.pdf(x_), desired=scipy_pdf, rtol=1e-5, atol=1e-8)
            # np.testing.assert_allclose(actual=dist_.logpdf(x_), desired=scipy_logpdf, rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=dist_.cdf(x_), desired=scipy_cdf, rtol=1e-5, atol=1e-8)
