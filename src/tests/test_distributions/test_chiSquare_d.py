"""Created on Dec 14 12:49:44 2024"""

import numpy as np
import pytest
from scipy.stats import chi2

from ...pymultifit import EPSILON
from ...pymultifit.distributions import ChiSquareDistribution
from ...pymultifit.distributions.backend import errorHandling as erH


class TestChiSquareDistribution:

    @staticmethod
    def test_initialization():
        dist_ = ChiSquareDistribution(amplitude=2.0, degree_of_freedom=1, normalize=False)
        assert dist_.amplitude == 2.0
        assert dist_.dof == 1
        assert not dist_.norm

        dist_normalized = ChiSquareDistribution(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1.0

    @staticmethod
    def test_constraints():
        with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
            ChiSquareDistribution(amplitude=-1.0, normalize=False)

        # amplitude should be internally updated to 1.0 if `normalize` is called
        distribution = ChiSquareDistribution(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

    @staticmethod
    def test_edge_case():
        dist = ChiSquareDistribution()
        x = np.array([])
        result = dist.pdf(x)
        assert result.size == 0  # Should return an empty array

    @staticmethod
    def test_stats():
        df_ = np.random.randint(low=EPSILON, high=100, size=10)
        loc_ = np.random.uniform(low=-5, high=10, size=10)
        scale_ = np.random.uniform(low=EPSILON, high=10, size=10)
        stack_ = np.column_stack([df_, loc_, scale_])

        for df, loc, scale in stack_:
            _distribution = ChiSquareDistribution.scipy_like(df=df, loc=loc, scale=scale)
            d_stats = _distribution.stats()

            # Scipy calculations
            scipy_mean, scipy_variance = chi2.stats(df=df, loc=loc, scale=scale, moments='mv')
            scipy_stddev = np.sqrt(scipy_variance)

            # Assertions for mean and variance
            np.testing.assert_allclose(actual=scipy_mean, desired=d_stats['mean'], rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=scipy_variance, desired=d_stats['variance'], rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=scipy_stddev, desired=d_stats['std'], rtol=1e-5, atol=1e-8)
            assert d_stats['median'] is None

    @staticmethod
    def test_pdf_cdf():
        def _cdf_pdf_custom(x_, dist_, what='cdf'):
            """Evaluate the CDF or PDF for the custom ChiSquareDistribution."""
            return dist_.cdf(x_) if what == 'cdf' else dist_.pdf(x_)

        def _cdf_pdf_scipy(x_, degree_of_freedom, loc, scale, what='cdf'):
            """Evaluate the CDF or PDF using scipy.stats.chi2."""
            return [chi2.cdf(x_, df=degree_of_freedom, loc=loc, scale=scale) if what == 'cdf' else
                    chi2.pdf(x_, df=degree_of_freedom, loc=loc, scale=scale)][0]

        for i in ['pdf']:
            for _ in range(100):
                dof = np.random.randint(1, 21)  # Degrees of freedom, always a positive integer
                loc = np.random.uniform(-10, 10)
                scale = np.random.uniform(EPSILON, 10)
                x = np.linspace(start=EPSILON, stop=50.0, num=10)

                distribution = ChiSquareDistribution(degree_of_freedom=dof, loc=loc, scale=scale, normalize=True)
                expected = _cdf_pdf_scipy(x_=x, degree_of_freedom=dof, loc=loc, scale=scale, what=i)
                actual = _cdf_pdf_custom(x_=x, dist_=distribution, what=i)

                np.testing.assert_allclose(actual=actual, desired=expected, rtol=1e-5, atol=1e-8)
