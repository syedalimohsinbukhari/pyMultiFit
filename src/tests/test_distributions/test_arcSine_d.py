"""Created on Dec 15 19:24:18 2024"""

import pytest
from scipy.stats import arcsine

from . import base_test_functions as btf
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
        btf.edge_cases(distribution=ArcSineDistribution(), log_check=True)

    @staticmethod
    def test_stats():
        btf.stats(
            custom_distribution=ArcSineDistribution.from_scipy_params,
            scipy_distribution=arcsine,
            parameters=[btf.loc_parameter, btf.scale_parameter],
        )

    @staticmethod
    def test_pdfs():
        btf.value_functions(
            custom_distribution=ArcSineDistribution.from_scipy_params,
            scipy_distribution=arcsine,
            parameters=[btf.loc_parameter, btf.scale_parameter],
            log_check=True,
        )

    @staticmethod
    def test_single_values():
        btf.single_input_n_variables(
            custom_distribution=ArcSineDistribution.from_scipy_params,
            scipy_distribution=arcsine,
            parameters=[btf.loc_parameter, btf.scale_parameter],
            log_check=True,
        )
