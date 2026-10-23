"""Created on Dec 08 11:48:14 2024"""

import numpy as np
import pytest
from scipy.stats import gennorm

from ...pymultifit import EPSILON
from ...pymultifit.distributions import SymmetricGeneralizedNormalDistribution
from ...pymultifit.distributions.utilities_d import sym_gen_normal_log_cdf_
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
        distribution = SymmetricGeneralizedNormalDistribution(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

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


@pytest.mark.parametrize("shape", [0.3, 0.5, 1.0, 2.0, 8.0, 30.0])
def test_cdf_and_logcdf_keep_the_left_tail(shape):
    # 0.5 - 0.5 * P is exactly 0 as soon as P rounds to 1 (x < -8.4 for shape 2), the true value is 1e-45 at x = -10
    x = np.array([-1e3, -100.0, -30.0, -10.0, -8.5, -3.0, -0.1, 0.0, 0.1, 3.0, 30.0])
    dist = SymmetricGeneralizedNormalDistribution(shape=shape, normalize=True)
    expected = gennorm.cdf(x, shape)

    nonzero = expected > 0
    np.testing.assert_allclose(dist.cdf(x)[nonzero], expected[nonzero], rtol=1e-12)
    np.testing.assert_array_equal(dist.cdf(x)[~nonzero], 0.0)
    log_cdf = sym_gen_normal_log_cdf_(x, shape=shape, normalize=True)
    # atol: log(F) of an F within 1e-13 of 1 is one rounding step of F, which scipy and this code do not share
    np.testing.assert_allclose(log_cdf[nonzero], np.log(expected[nonzero]), rtol=1e-12, atol=2e-16)


def test_cdf_left_tail_is_not_zero_where_the_value_is_representable():
    dist = SymmetricGeneralizedNormalDistribution(shape=2.0, normalize=True)
    assert dist.cdf(np.array([-10.0]))[0] == pytest.approx(1.0442437918812844e-45, rel=1e-12)
    log_cdf = sym_gen_normal_log_cdf_(np.array([-10.0]), shape=2.0, normalize=True)
    assert log_cdf[0] == pytest.approx(-103.57303620540482, rel=1e-12)


def test_cdf_scalars_nan_and_infinities():
    dist = SymmetricGeneralizedNormalDistribution(shape=2.0, normalize=True)
    assert dist.cdf(-3.0) == pytest.approx(gennorm.cdf(-3.0, 2.0), rel=1e-12)
    assert dist.cdf(3.0) == pytest.approx(gennorm.cdf(3.0, 2.0), rel=1e-12)
    assert np.isnan(dist.cdf(np.nan))
    np.testing.assert_array_equal(dist.cdf(np.array([-np.inf, 0.0, np.inf])), [0.0, 0.5, 1.0])


@pytest.mark.parametrize("shape, loc, scale", [(0.5, 0.0, 1.0), (1.0, -1.0, 2.0), (2.0, 3.0, 0.5), (8.0, 0.0, 10.0)])
def test_logpdf_and_logcdf_are_defined(shape, loc, scale):
    # the class used to inherit the BaseDistribution stubs, which return None
    x = np.linspace(loc - 5 * scale, loc + 5 * scale, 101)
    dist = SymmetricGeneralizedNormalDistribution(shape=shape, loc=loc, scale=scale, normalize=True)

    np.testing.assert_allclose(dist.logpdf(x), gennorm.logpdf(x, shape, loc=loc, scale=scale), rtol=1e-12)
    np.testing.assert_allclose(dist.logcdf(x), gennorm.logcdf(x, shape, loc=loc, scale=scale), rtol=1e-12, atol=2e-16)


def test_logpdf_of_the_unnormalised_distribution_is_the_log_of_its_pdf():
    x = np.linspace(-4, 4, 81)
    dist = SymmetricGeneralizedNormalDistribution(amplitude=3.0, shape=1.5, loc=0.5, scale=1.2, normalize=False)

    np.testing.assert_allclose(dist.logpdf(x), np.log(dist.pdf(x)), rtol=1e-12)
    assert dist.logpdf(x).max() == pytest.approx(np.log(3.0))


def test_logpdf_and_logcdf_are_empty_for_empty_input():
    dist = SymmetricGeneralizedNormalDistribution()
    assert dist.logpdf(np.array([])).size == 0
    assert dist.logcdf(np.array([])).size == 0


@pytest.mark.parametrize("amplitude", [0.5, 1.0, 3.0])
def test_unnormalised_pdf_peaks_at_the_amplitude(amplitude):
    # sym_gen_normal_pdf_ used to apply the log scaling to the pdf: log(amplitude) + pdf - max(pdf)
    x = np.linspace(-4, 4, 801)
    pdf = SymmetricGeneralizedNormalDistribution(amplitude=amplitude, shape=1.5, normalize=False).pdf(x)

    assert pdf.max() == pytest.approx(amplitude)
    assert pdf.min() >= 0.0
    np.testing.assert_allclose(pdf / pdf.max(), gennorm.pdf(x, 1.5) / gennorm.pdf(0.0, 1.5), rtol=1e-10)
