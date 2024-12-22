"""Created on Dec 15 19:24:18 2024"""

import numpy as np
from scipy.stats import beta

from ...pymultifit.distributions.arcSine_d import ArcSineDistribution


class TestArcSineDistribution:

    @staticmethod
    def test_initialization():
        dist_ = ArcSineDistribution(amplitude=2.0, normalize=False)
        assert dist_.amplitude == 2.0
        assert dist_.loc == 0.
        assert dist_.scale == 1.0
        assert not dist_.norm

        x = np.linspace(0, 1, 100)
        _distribution1 = ArcSineDistribution(normalize=True)
        _distribution2 = beta.pdf(x, a=0.5, b=0.5)

        np.testing.assert_allclose(_distribution1.pdf(x), _distribution2, atol=1e-8, rtol=1e-5)

    @staticmethod
    def test_pdf_cdf():
        x1 = np.linspace(-0.5, 1.5, 1000)
        x2 = np.array([0, 1])

        for x in [x1, x2]:
            distribution = ArcSineDistribution(normalize=True)
            pdf_custom = distribution.pdf(x)
            cdf_custom = distribution.cdf(x)

            pdf_scipy = beta.pdf(x, a=0.5, b=0.5)
            cdf_scipy = beta.cdf(x, a=0.5, b=0.5)

            np.testing.assert_allclose(pdf_custom, pdf_scipy, rtol=1e-5, atol=1e-8,
                                       err_msg="PDF does not match SciPy Beta(0.5, 0.5).")
            np.testing.assert_allclose(cdf_custom, cdf_scipy, rtol=1e-5, atol=1e-8,
                                       err_msg="CDF does not match SciPy Beta(0.5, 0.5).")
