"""Created on Oct 13 2026"""

import numpy as np
import pytest
from scipy.stats import genpareto

from ...pymultifit.distributions import QExponentialDistribution

# (q, rate, loc): q < 1 has a finite upper bound, q = 1 is the exponential, q > 1 is heavy tailed (q >= 1.5: no mean)
CASES = [
    (-1.0, 1.0, 0.0),
    (0.5, 2.0, -1.5),
    (1.0, 1.0, 0.0),
    (1.0, 0.3, 4.0),
    (1.3, 1.3, -3.3),
    (1.5, 3.0, 3.0),
    (1.9, 0.7, 0.5),
]


def _grid(loc, scale, c):
    """Sorted points left of the support, inside it (not on the finite upper bound) and beyond it."""
    upper = loc - scale / c if c < 0 else loc + 40 * scale
    inside = np.linspace(loc, upper - 1e-6 * scale if c < 0 else upper, 400)
    return np.concatenate([np.linspace(loc - 5, loc - 1e-9, 25), inside, [upper + 1.0, upper + 50.0]])


class TestQExponentialDistribution:

    @staticmethod
    def test_initialization():
        dist = QExponentialDistribution()
        assert dist.amplitude == 1.0
        assert dist.q == 1.0
        assert dist.rate == 1.0
        assert dist.loc == 0.0
        assert not dist.normalize

    @staticmethod
    @pytest.mark.parametrize("q, rate, loc", CASES)
    def test_to_scipy_params(q, rate, loc):
        params = QExponentialDistribution(q=q, rate=rate, loc=loc).to_scipy_params()
        assert set(params) == {"c", "loc", "scale"}
        np.testing.assert_allclose(params["c"], (q - 1) / (2 - q))
        np.testing.assert_allclose(params["scale"], 1 / ((2 - q) * rate))
        assert params["loc"] == loc

    @staticmethod
    def test_to_scipy_params_q_one_is_exponential():
        params = QExponentialDistribution(q=1.0, rate=2.0, loc=3.0).to_scipy_params()
        assert params == {"c": 0.0, "loc": 3.0, "scale": 0.5}

    @staticmethod
    @pytest.mark.parametrize("q, rate", [(2.0, 1.0), (2.5, 1.0), (1.0, 0.0), (1.0, -1.0)])
    def test_to_scipy_params_invalid(q, rate):
        with pytest.raises(ValueError):
            QExponentialDistribution(q=q, rate=rate).to_scipy_params()

    @staticmethod
    @pytest.mark.parametrize("q, rate, loc", CASES)
    def test_against_scipy_genpareto(q, rate, loc):
        dist = QExponentialDistribution(q=q, rate=rate, loc=loc, normalize=True)
        ref = genpareto(**dist.to_scipy_params())
        x = _grid(loc, ref.kwds["scale"], ref.kwds["c"])

        np.testing.assert_allclose(dist.pdf(x), ref.pdf(x), rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(dist.cdf(x), ref.cdf(x), rtol=1e-8, atol=1e-12)

        inside = ref.pdf(x) > 0
        np.testing.assert_allclose(dist.logpdf(x)[inside], ref.logpdf(x)[inside], rtol=1e-8, atol=1e-10)
        assert np.all(np.isneginf(dist.logpdf(x)[~inside]))

        positive = ref.cdf(x) > 1e-300
        np.testing.assert_allclose(dist.logcdf(x)[positive], ref.logcdf(x)[positive], rtol=1e-8, atol=1e-10)

    @staticmethod
    @pytest.mark.parametrize("q, rate, loc", CASES)
    def test_cdf_properties(q, rate, loc):
        dist = QExponentialDistribution(q=q, rate=rate, loc=loc, normalize=True)
        params = dist.to_scipy_params()
        x = _grid(loc, params["scale"], params["c"])
        cdf = dist.cdf(x)

        assert np.all(np.diff(cdf) >= -1e-12)
        assert cdf[0] == 0.0
        assert 0.0 <= cdf.min() and cdf.max() <= 1.0 + 1e-12

    @staticmethod
    def test_q_one_matches_exponential():
        x = np.linspace(-2, 20, 300)
        dist = QExponentialDistribution(q=1.0, rate=0.8, loc=1.0, normalize=True)
        ref = genpareto(c=0.0, loc=1.0, scale=1 / 0.8)
        np.testing.assert_allclose(dist.pdf(x), ref.pdf(x), rtol=1e-10, atol=1e-14)

    @staticmethod
    @pytest.mark.parametrize("q, rate, loc", CASES)
    def test_zero_outside_support(q, rate, loc):
        dist = QExponentialDistribution(q=q, rate=rate, loc=loc, normalize=True)
        left = np.array([loc - 1e-9, loc - 1.0, loc - 100.0])
        np.testing.assert_array_equal(dist.pdf(left), 0.0)
        np.testing.assert_array_equal(dist.cdf(left), 0.0)

    @staticmethod
    @pytest.mark.parametrize("amplitude", [0.5, 1.0, 7.3])
    def test_amplitude_is_the_peak(amplitude):
        dist = QExponentialDistribution(amplitude=amplitude, q=1.3, rate=1.2, loc=-1.0)
        x = np.linspace(-1.0, 10.0, 500)
        y = dist.pdf(x)
        np.testing.assert_allclose(y.max(), amplitude)
        np.testing.assert_allclose(y[0], amplitude)  # the mode sits at ``loc``

    @staticmethod
    def test_edge_case_empty():
        dist = QExponentialDistribution()
        x = np.array([])
        for method in (dist.pdf, dist.cdf, dist.logpdf, dist.logcdf):
            assert method(x).size == 0

    @staticmethod
    @pytest.mark.parametrize("q, rate, loc", [c for c in CASES if c[0] < 4 / 3])
    def test_stats_against_scipy(q, rate, loc):
        dist = QExponentialDistribution(q=q, rate=rate, loc=loc)
        ref = genpareto(**dist.to_scipy_params())
        stats = dist.stats()

        np.testing.assert_allclose(stats["mean"], ref.mean(), rtol=1e-9)
        np.testing.assert_allclose(stats["variance"], ref.var(), rtol=1e-9)
        np.testing.assert_allclose(stats["std"], ref.std(), rtol=1e-9)
        assert stats["mode"] == loc

    @staticmethod
    def test_stats_moments_that_do_not_exist():
        assert np.isinf(QExponentialDistribution(q=1.4).stats()["variance"])
        assert np.isinf(QExponentialDistribution(q=1.6).stats()["mean"])
        assert np.isnan(QExponentialDistribution(q=1.6).stats()["variance"])
        assert all(np.isnan(v) for v in QExponentialDistribution(rate=0.0).stats().values())
