"""Created on Dec 08 11:48:14 2024"""

import numpy as np
import pytest
from scipy.stats import gennorm

from ...pymultifit import EPSILON
from ...pymultifit.distributions import SymmetricGeneralizedNormalDistribution
from ...pymultifit.distributions.backend import errorHandling as erH


class TestSymNormalDistribution:

    @staticmethod
    def test_initialization():
        dist = SymmetricGeneralizedNormalDistribution(normalize=False)
        assert dist.amplitude == 1.0
        assert dist.shape == 1.0
        assert dist.loc == 0.0
        assert dist.scale == 1.0
        assert not dist.norm

        dist_normalized = SymmetricGeneralizedNormalDistribution(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1.0

    @staticmethod
    def test_constraints():
        with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
            SymmetricGeneralizedNormalDistribution(amplitude=-1.0, normalize=False)

        # amplitude should be internally updated to 1.0 if `normalize` is called
        distribution = SymmetricGeneralizedNormalDistribution(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

        with pytest.raises(erH.NegativeShapeError, match=f"Shape {erH.neg_message}"):
            SymmetricGeneralizedNormalDistribution(shape=-3.0)

    @staticmethod
    def test_edge_case():
        dist = SymmetricGeneralizedNormalDistribution()
        x = np.array([])

        result = dist.pdf(x)
        assert result.size == 0

        result = dist.cdf(x)
        assert result.size == 0

    @staticmethod
    def test_stats():
        shape_ = np.random.uniform(low=EPSILON, high=10, size=100)
        loc_ = np.random.uniform(low=-10, high=10, size=100)
        scale_ = np.random.uniform(low=EPSILON, high=10, size=100)
        stack_ = np.column_stack([shape_, loc_, scale_])

        for shape, loc, scale in stack_:
            _distribution = SymmetricGeneralizedNormalDistribution.from_scipy_params(beta=shape, loc=loc, scale=scale)
            d_stats = _distribution.stats()

            # Scipy calculations
            scipy_mean, scipy_variance = gennorm.stats(beta=shape, loc=loc, scale=scale, moments="mv")
            scipy_median = gennorm.median(beta=shape, loc=loc, scale=scale)
            scipy_stddev = np.sqrt(scipy_variance)

            # Assertions for mean and variance
            np.testing.assert_allclose(actual=scipy_mean, desired=d_stats["mean"], rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=scipy_variance, desired=d_stats["variance"], rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=scipy_median, desired=d_stats["median"], rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=scipy_stddev, desired=d_stats["std"], rtol=1e-5, atol=1e-8)

    @staticmethod
    def test_pdf_cdf():
        def _cdf_pdf_custom(x_, dist_, what="cdf"):
            return dist_.cdf(x_) if what == "cdf" else dist_.pdf(x_)

        def _cdf_pdf_scipy(x_, shape, loc, scale, what="cdf"):
            return [
                (
                    gennorm.cdf(x_, beta=shape, loc=loc, scale=scale)
                    if what == "cdf"
                    else gennorm.pdf(x_, beta=shape, loc=loc, scale=scale)
                )
            ][0]

        for i in ["cdf", "pdf"]:
            for _ in range(500):  # Run 50 random tests
                shape_ = np.random.uniform(low=EPSILON, high=100)
                loc_ = np.random.uniform(low=-100, high=100)
                scale_ = np.random.uniform(low=EPSILON, high=100)
                x = np.linspace(start=loc_ - 10, stop=loc_ + 10, num=50)
                distribution = SymmetricGeneralizedNormalDistribution.from_scipy_params(
                    beta=shape_, loc=loc_, scale=scale_
                )
                expected = _cdf_pdf_scipy(x_=x, shape=shape_, loc=loc_, scale=scale_, what=i)
                np.testing.assert_allclose(
                    actual=_cdf_pdf_custom(x_=x, dist_=distribution, what=i), desired=expected, rtol=1e-5, atol=1e-8
                )
