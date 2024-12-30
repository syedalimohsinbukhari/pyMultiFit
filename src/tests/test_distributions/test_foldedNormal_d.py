"""Created on Dec 15 23:23:05 2024"""

import numpy as np
import pytest
from scipy.stats import foldnorm

from ...pymultifit import EPSILON
from ...pymultifit.distributions import FoldedNormalDistribution
from ...pymultifit.distributions.backend import errorHandling as erH


class TestFoldedNormalDistribution:

    @staticmethod
    def test_initialization():
        dist = FoldedNormalDistribution(amplitude=2.0, mean=1.0, sigma=0.5, normalize=False)
        assert dist.amplitude == 2.0
        assert dist.mean == 1.0
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

        with pytest.raises(erH.NegativeStandardDeviationError, match=f"Standard deviation {erH.neg_message}"):
            FoldedNormalDistribution(sigma=-3.0)

    @staticmethod
    def test_edge_case():
        dist = FoldedNormalDistribution()
        x = np.array([])
        result = dist.pdf(x)
        assert result.size == 0  # Should return an empty array

    @staticmethod
    def test_pdf_cdf():
        np.random.seed(43)
        def _cdf_pdf_custom(x_, dist_, what='cdf'):
            return dist_.cdf(x_) if what == 'cdf' else dist_.pdf(x_)

        def _cdf_pdf_scipy(x_, mean_, std_, what='cdf'):
            c = abs(mean_) / std_  # Adjust for scipy's foldnorm input
            return foldnorm.cdf(x_, c=c, loc=mean_, scale=std_) if what == 'cdf' else foldnorm.pdf(x_, c=c, loc=mean_, scale=std_)

        for test_type in ['pdf']:  # Run tests for both PDF and CDF
            for _ in range(50):  # Run 50 random tests
                # Randomly generate test parameters
                mean = np.random.uniform(low=-10, high=10)
                std = np.random.uniform(low=0.1, high=5)  # Std deviation instead of scale
                x = np.random.uniform(low=EPSILON, high=mean + 10, size=10)

                # Initialize the custom distribution
                distribution = FoldedNormalDistribution(mean=mean, sigma=std, normalize=True)

                # Compute expected results from scipy
                expected = _cdf_pdf_scipy(x_=x, mean_=mean, std_=std, what=test_type)
                expected = np.nan_to_num(expected, True, 0)  # Convert NaNs to 0 if present

                # Compare custom distribution output with scipy's
                np.testing.assert_allclose(actual=_cdf_pdf_custom(x_=x, dist_=distribution, what=test_type),
                                           desired=expected, rtol=1e-5, atol=1e-8)
