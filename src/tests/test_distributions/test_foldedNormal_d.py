"""Created on Dec 15 23:23:05 2024"""

import numpy as np
import pytest
from scipy.stats import foldnorm

from ...pymultifit import EPSILON
from ...pymultifit.distributions import FoldedNormalDistribution
from ...pymultifit.distributions.backend import errorHandling as erH

np.random.seed(45)


class TestFoldedNormalDistribution:

    @staticmethod
    def test_initialization():
        dist = FoldedNormalDistribution(amplitude=2.0, mu=1.0, sigma=0.5, normalize=False)
        assert dist.amplitude == 2.0
        assert dist.mu == 1.0
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

    # @staticmethod
    # def test_stats():
    #     mean_values = np.linspace(start=EPSILON, stop=10, num=50)
    #     loc_values = np.linspace(start=-10, stop=10, num=50)
    #     sigma_values = np.linspace(start=EPSILON, stop=10, num=50)
    #     stack_ = np.column_stack([mean_values, loc_values, sigma_values])
    #
    #     for mean_, loc_, scale_ in stack_:
    #         # Custom distribution parameters
    #         _distribution = FoldedNormalDistribution(mu=mean_, sigma=scale_, loc=loc_, normalize=True)
    #         d_stats = _distribution.stats()
    #
    #         # Scipy calculations
    #         scipy_mean, scipy_variance = foldnorm.stats(loc=loc_, c=mean_, scale=scale_, moments='mv')
    #         scipy_median = foldnorm.median(c=mean_, loc=loc_, scale=scale_)
    #         scipy_stddev = np.sqrt(scipy_variance)
    #
    #         # Assertions for mean and variance
    #         np.testing.assert_allclose(desired=scipy_mean, actual=d_stats['mean'], rtol=1e-5, atol=1e-8)
    #        np.testing.assert_allclose(desired=scipy_variance, actual=d_stats['variance'], rtol=1e-5, atol=1e-8)
    #        np.testing.assert_allclose(actual=scipy_median, desired=d_stats['median'], rtol=1e-5, atol=1e-8)
    #        np.testing.assert_allclose(actual=scipy_stddev, desired=d_stats['std'], rtol=1e-5, atol=1e-8)

    @staticmethod
    def test_pdf_cdf():
        np.random.seed(43)

        def _cdf_pdf_custom(x_, dist_, what='cdf'):
            return dist_.cdf(x_) if what == 'cdf' else dist_.pdf(x_)

        def _cdf_pdf_scipy(x_, c_, loc_, scale_, what='cdf'):
            return foldnorm.cdf(x_, c=c_, loc=loc_, scale=scale_) if what == 'cdf' else foldnorm.pdf(x_, c=c_, loc=loc_, scale=scale_)

        for test_type in ['cdf']:
            for _ in range(50):
                shape = np.random.uniform(low=EPSILON, high=10)
                std = np.random.uniform(low=EPSILON, high=10)
                loc = np.random.uniform(low=-10, high=10)

                x = np.linspace(-20, 20, 50)
                distribution = FoldedNormalDistribution(mu=shape, sigma=std, loc=loc, normalize=True)

                actual = _cdf_pdf_custom(x_=x, dist_=distribution, what=test_type)
                desired = _cdf_pdf_scipy(x_=x, c_=shape, loc_=loc, scale_=std, what=test_type)

                np.testing.assert_allclose(actual=actual, desired=desired, rtol=1e-5, atol=1e-8)
