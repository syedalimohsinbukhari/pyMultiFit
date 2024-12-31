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

        with pytest.raises(erH.DegreeOfFreedomError, match=r"DOF can only be integer, N+"):
            ChiSquareDistribution(degree_of_freedom=-3)

    @staticmethod
    def test_edge_case():
        dist = ChiSquareDistribution()
        x = np.array([])
        result = dist.pdf(x)
        assert result.size == 0  # Should return an empty array

    @staticmethod
    def test_stats():
        dist_ = ChiSquareDistribution(amplitude=1.0, degree_of_freedom=2)
        d_stats = dist_.stats()
        assert d_stats["mean"] == dist_.dof
        np.testing.assert_allclose(actual=d_stats["median"],
                                   desired=dist_.dof * (1 - (2 / (9 * dist_.dof)))**3)
        assert d_stats["mode"] == max(dist_.dof - 2, 0)
        assert d_stats["variance"] == 2 * dist_.dof

    np.random.seed(42)

    @staticmethod
    def test_pdf_cdf_chi_square():
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

                distribution = ChiSquareDistribution(degree_of_freedom=2, loc=-3, scale=2, normalize=True)
                expected = _cdf_pdf_scipy(x_=x, degree_of_freedom=2, loc=-3, scale=2, what=i)
                actual = _cdf_pdf_custom(x_=x, dist_=distribution, what=i)

                np.testing.assert_allclose(actual=actual, desired=expected, rtol=1e-5, atol=1e-8)

    @staticmethod
    def test_stats_scipy():
        for _ in range(50):  # Run 50 random tests
            # Generate random degrees of freedom (dof), ensuring it's a positive integer
            dof_ = np.random.randint(1, 21)  # Degrees of freedom between 1 and 20

            # Create an instance of the custom Chi-Square distribution
            dist_ = ChiSquareDistribution(degree_of_freedom=dof_, normalize=True)
            dof = dist_.dof

            d_stats = dist_.stats()
            scipy_mean, scipy_variance = chi2.stats(df=dof, moments="mv")
            scipy_mode = max(dof - 2, 0)

            assert np.isclose(d_stats["mean"], scipy_mean, rtol=1e-5,
                              atol=1e-8), f"Mean mismatch: {d_stats['mean']} != {scipy_mean} for dof={dof}"
            assert np.isclose(d_stats['mode'], scipy_mode, rtol=1e-5,
                              atol=1e-8), f"Median mismatch: {d_stats['median']} != {scipy_mode} for dof={dof}"
            assert np.isclose(d_stats["variance"], scipy_variance, rtol=1e-5,
                              atol=1e-8), f"Variance mismatch: {d_stats['variance']} != {scipy_variance} for dof={dof}"
            # skipped median testing
