"""Created on Jan 03 16:13:58 2025"""

import numpy as np
import pytest
from scipy.stats import skewnorm

from ...pymultifit import EPSILON
from ...pymultifit.distributions import SkewNormalDistribution
from ...pymultifit.distributions.backend import errorHandling as erH


class TestGaussianDistribution:

    @staticmethod
    def test_initialization():
        dist = SkewNormalDistribution(amplitude=2.0, shape=1.0, location=0.5, scale=1.0, normalize=False)
        assert dist.amplitude == 2.0
        assert dist.location == 0.5
        assert dist.scale == 1.0
        assert not dist.norm

        dist_normalized = SkewNormalDistribution(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1.0

    @staticmethod
    def test_constraints():
        with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
            SkewNormalDistribution(amplitude=-1.0, normalize=False)

        # amplitude should be internally updated to 1.0 if `normalize` is called
        distribution = SkewNormalDistribution(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

        with pytest.raises(erH.NegativeScaleError, match=f"Scale {erH.neg_message}"):
            SkewNormalDistribution(scale=-3.0)

    @staticmethod
    def test_edge_case():
        dist = SkewNormalDistribution()
        x = np.array([])
        result = dist.pdf(x)
        assert result.size == 0  # Should return an empty array

    @staticmethod
    def test_stats():
        shape_ = np.random.uniform(low=EPSILON, high=10, size=10)
        loc_ = np.random.uniform(low=-5, high=10, size=10)
        scale_ = np.random.uniform(low=EPSILON, high=10, size=10)
        stack_ = np.column_stack([shape_, loc_, scale_])

        for shape, loc, scale in stack_:
            _distribution = SkewNormalDistribution.scipy_like(a=shape, loc=loc, scale=scale)
            d_stats = _distribution.stats()

            # Scipy calculations
            scipy_mean, scipy_variance = skewnorm.stats(a=shape, loc=loc, scale=scale, moments='mv')
            scipy_stddev = np.sqrt(scipy_variance)

            # Assertions for mean and variance
            np.testing.assert_allclose(actual=scipy_mean, desired=d_stats['mean'], rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=scipy_variance, desired=d_stats['variance'], rtol=1e-5, atol=1e-8)
            assert d_stats['median'] is None
            np.testing.assert_allclose(actual=scipy_stddev, desired=d_stats['std'], rtol=1e-5, atol=1e-8)

    @staticmethod
    def test_pdf_cdf():
        def _cdf_pdf_custom(x_, dist_, what='cdf'):
            return dist_.cdf(x_) if what == 'cdf' else dist_.pdf(x_)

        def _cdf_pdf_scipy(x_, shape, loc, scale, what='cdf'):
            return [skewnorm.cdf(x_, a=shape, loc=loc, scale=scale) if what == 'cdf' else
                    skewnorm.pdf(x_, a=shape, loc=loc, scale=scale)][0]

        for i in ['cdf', 'pdf']:
            for _ in range(50):  # Run 50 random tests
                shape_ = np.random.uniform(low=EPSILON, high=10)
                loc_ = np.random.uniform(low=-10, high=10)
                scale_ = np.random.uniform(low=EPSILON, high=10)
                x = np.linspace(start=loc_ - 10, stop=loc_ + 10, num=50)
                distribution = SkewNormalDistribution.scipy_like(a=shape_, loc=loc_, scale=scale_)
                expected = _cdf_pdf_scipy(x_=x, shape=shape_, loc=loc_, scale=scale_, what=i)
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
            dist_ = SkewNormalDistribution.scipy_like(a=1, loc=mean_, scale=std_)

            # SciPy Gaussian distribution for comparison
            scipy_pdf = skewnorm(a=1, loc=mean_, scale=std_).pdf(x_)
            # scipy_logpdf = norm(loc=mean_, scale=std).logpdf(x_)
            scipy_cdf = skewnorm(a=1, loc=mean_, scale=std_).cdf(x_)

            # Assert PDF, logPDF, and CDF are close to SciPy's implementation
            np.testing.assert_allclose(actual=dist_.pdf(x_), desired=scipy_pdf, rtol=1e-5, atol=1e-8)
            # np.testing.assert_allclose(actual=dist_.logpdf(x_), desired=scipy_logpdf, rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=dist_.cdf(x_), desired=scipy_cdf, rtol=1e-5, atol=1e-8)
