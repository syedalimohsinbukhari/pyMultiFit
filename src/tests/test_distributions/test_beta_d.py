"""Created on Dec 14 06:55:53 2024"""

import pytest
from scipy.stats import beta

from .base_test_functions import edge_cases, scale_parameter, value_functions, loc_parameter, stats, shape_parameter
from ...pymultifit.distributions import BetaDistribution
from ...pymultifit.distributions.backend import errorHandling as erH


class TestBetaDistribution:

    @staticmethod
    def test_initialization():
        dist_ = BetaDistribution(amplitude=2.0, normalize=False)
        assert dist_.amplitude == 2.0
        assert dist_.alpha == 1.0
        assert dist_.beta == 1.0
        assert not dist_.norm

        dist_normalized = BetaDistribution(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1.0

    @staticmethod
    def test_constraints():
        with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
            BetaDistribution(amplitude=-1.0, normalize=False)

        # amplitude should be internally updated to 1.0 if `normalize` is called
        distribution = BetaDistribution(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

        with pytest.raises(erH.NegativeAlphaError, match=f"Alpha {erH.neg_message}"):
            BetaDistribution(alpha=-3.0)

        with pytest.raises(erH.NegativeBetaError, match=f"Beta {erH.neg_message}"):
            BetaDistribution(beta=-3.0)

    @staticmethod
    def test_edge_cases():
        edge_cases(BetaDistribution())

    @staticmethod
    def test_stats():
        stats(custom_distribution=BetaDistribution.scipy_like,
              scipy_distribution=beta, parameters=[shape_parameter, shape_parameter, loc_parameter, scale_parameter])

    @staticmethod
    def test_pdfs():
        value_functions(custom_distribution=BetaDistribution.scipy_like, scipy_distribution=beta,
                        parameters=[shape_parameter, shape_parameter, loc_parameter, scale_parameter])
