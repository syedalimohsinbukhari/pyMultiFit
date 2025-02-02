"""Created on Feb 02 17:13:40 2025"""

import numpy as np
import pytest
from scipy.stats import invgamma

from ...pymultifit import EPSILON
from ...pymultifit.distributions import ScaledInverseChiSquareDistribution
from ...pymultifit.distributions.backend import errorHandling as erH


class TestGaussianDistribution:

    @staticmethod
    def test_initialization():
        dist = ScaledInverseChiSquareDistribution(amplitude=2.0, df=1.0, scale=0.5, normalize=False)
        assert dist.amplitude == 2.0
        assert dist.df == 1.0
        assert dist.scale == 0.5
        assert dist.tau2 == dist.scale / dist.df
        assert not dist.norm

        dist_normalized = ScaledInverseChiSquareDistribution(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1.0

    @staticmethod
    def test_constraints():
        with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
            ScaledInverseChiSquareDistribution(amplitude=-1.0, normalize=False)

        # amplitude should be internally updated to 1.0 if `normalize` is called
        distribution = ScaledInverseChiSquareDistribution(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

        with pytest.raises(erH.NegativeScaleError, match=f"Scale {erH.neg_message}"):
            ScaledInverseChiSquareDistribution(scale=-3.0)

    @staticmethod
    def test_edge_case():
        dist = ScaledInverseChiSquareDistribution()
        x = np.array([])
        result = dist.pdf(x)
        assert result.size == 0

        result = dist.cdf(x)
        assert result.size == 0

        result = dist.logpdf(x)
        assert result.size == 0

        result = dist.logcdf(x)
        assert result.size == 0

    @staticmethod
    def test_stats():
        beta_ = np.random.uniform(low=EPSILON, high=10, size=10)
        loc_ = np.random.uniform(low=-5, high=10, size=10)
        scale_ = np.random.uniform(low=EPSILON, high=10, size=10)
        stack_ = np.column_stack([beta_, loc_, scale_])

        for beta, loc, scale in stack_:
            _distribution = ScaledInverseChiSquareDistribution.scipy_like(a=beta, loc=loc, scale=scale)
            d_stats = _distribution.stats()

            # Scipy calculations
            scipy_mean, scipy_variance = invgamma.stats(a=beta / 2, loc=loc, scale=scale / 2, moments='mv')
            scipy_stddev = np.sqrt(scipy_variance)

            # Assertions for mean and variance
            np.testing.assert_allclose(actual=scipy_mean, desired=d_stats['mean'], rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=scipy_variance, desired=d_stats['variance'], rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=scipy_stddev, desired=d_stats['std'], rtol=1e-5, atol=1e-8)

    @staticmethod
    def test_pdf_cdf():
        df = np.random.uniform(low=1, high=100, size=1000)
        loc = np.random.uniform(low=-100, high=100, size=1000)
        scale = np.random.uniform(low=1, high=100, size=1000)

        pars = np.column_stack([df, loc, scale])

        x = np.linspace(start=1, stop=100, num=1000)

        for df_, loc_, scale_ in pars:
            _dist = ScaledInverseChiSquareDistribution.scipy_like(a=df_, loc=loc_, scale=scale_)

            p1 = _dist.pdf(x)
            p11 = _dist.logpdf(x)
            p3 = _dist.cdf(x)
            p31 = _dist.logcdf(x)

            p2 = invgamma.pdf(x, a=df_ / 2, loc=loc_, scale=scale_ / 2)
            p22 = invgamma.logpdf(x, a=df_ / 2, loc=loc_, scale=scale_ / 2)
            p4 = invgamma.cdf(x, a=df_ / 2, loc=loc_, scale=scale_ / 2)
            p41 = invgamma.logcdf(x, a=df_ / 2, loc=loc_, scale=scale_ / 2)

            np.testing.assert_allclose(actual=p1, desired=p2, rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=p11, desired=p22, rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=p3, desired=p4, rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=p31, desired=p41, rtol=1e-5, atol=1e-8)

    @staticmethod
    def test_edge_cases():
        x_ = np.linspace(1, 100, 1000)

        # Edge cases
        edge_cases = [
            (0, 1),  # Standard distribution (loc=0, scale=1)
            (0, 0.1),  # Narrow peak at center (small scale)
            (0, 10),  # Broad distribution (large scale)
            (5, 1),  # Mean shifted to the right
            (-5, 1),  # Mean shifted to the left
        ]

        # Extreme cases
        extreme_cases = [
            (0, 1e-5),  # Very narrow peak
            (0, 1e5),  # Extremely broad distribution
            (1e5, 1),  # Very large loc
            (-1e5, 1),  # Very large negative loc
            (1e5, 1e-5),  # Extreme loc and narrow scale
            (-1e5, 1e-5),  # Extreme negative loc and narrow scale
        ]

        test_cases = edge_cases + extreme_cases

        for mean_, std_ in test_cases:
            dist_ = ScaledInverseChiSquareDistribution.scipy_like(a=1, loc=mean_, scale=std_)

            scipy_pdf = invgamma(a=0.5, loc=mean_, scale=std_ / 2).pdf(x_)
            scipy_logpdf = invgamma(a=0.5, loc=mean_, scale=std_ / 2).logpdf(x_)
            scipy_cdf = invgamma(a=0.5, loc=mean_, scale=std_ / 2).cdf(x_)
            scipy_logcdf = invgamma(a=0.5, loc=mean_, scale=std_ / 2).logcdf(x_)

            np.testing.assert_allclose(actual=dist_.pdf(x_), desired=scipy_pdf, rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=dist_.logpdf(x_), desired=scipy_logpdf, rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=dist_.cdf(x_), desired=scipy_cdf, rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=dist_.logcdf(x_), desired=scipy_logcdf, rtol=1e-5, atol=1e-8)
