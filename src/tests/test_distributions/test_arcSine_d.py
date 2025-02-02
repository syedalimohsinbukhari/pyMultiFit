"""Created on Dec 15 19:24:18 2024"""

import pytest
from scipy.stats import arcsine

from .base_test_functions import edge_cases, scale_parameter, value_functions, loc_parameter, stats
from ...pymultifit.distributions.arcSine_d import ArcSineDistribution
from ...pymultifit.distributions.backend import errorHandling as erH


class TestArcSineDistribution:

    @staticmethod
    def test_constraints():
        with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
            ArcSineDistribution(amplitude=-1.0, normalize=False)

        distribution = ArcSineDistribution(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

    @staticmethod
    def test_edge_cases():
        edge_cases(ArcSineDistribution())

    @staticmethod
    def test_stats():
        stats(custom_distribution=ArcSineDistribution.scipy_like,
              scipy_distribution=arcsine, parameters=[loc_parameter, scale_parameter])

    @staticmethod
    def test_pdfs():
        value_functions(custom_distribution=ArcSineDistribution.scipy_like, scipy_distribution=arcsine,
                        parameters=[loc_parameter, scale_parameter])
