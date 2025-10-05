"""Created on Dec 14 06:40:29 2024"""

import pytest
from scipy.stats import expon

from . import base_test_functions as btf
from ...pymultifit.distributions import ExponentialDistribution
from ...pymultifit.distributions.backend import errorHandling as erH


class TestChiSquareDistribution:

    @staticmethod
    def test_initialization():
        dist_ = ExponentialDistribution(amplitude=2.0, scale=1.0, normalize=False)
        assert dist_.amplitude == 2.0
        assert dist_.scale == 1.0
        assert not dist_.norm

        dist_normalized = ExponentialDistribution(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1.0

    @staticmethod
    def test_constraints():
        with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
            ExponentialDistribution(amplitude=-1.0, normalize=False)

        # amplitude should be internally updated to 1.0 if `normalize` is called
        distribution = ExponentialDistribution(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

        with pytest.raises(erH.NegativeScaleError, match=f"Scale {erH.neg_message}"):
            ExponentialDistribution(scale=-3.0)

    @staticmethod
    def test_edge_cases():
        btf.edge_cases(distribution=ExponentialDistribution(), log_check=True)

    @staticmethod
    def test_stats():
        btf.stats(custom_distribution=ExponentialDistribution.from_scipy_params, scipy_distribution=expon,
                  parameters=[btf.loc_parameter, btf.scale_parameter], is_expon=True)

    @staticmethod
    def test_pdfs():
        btf.value_functions(custom_distribution=ExponentialDistribution.from_scipy_params, scipy_distribution=expon,
                            parameters=[btf.loc_parameter, btf.scale_parameter], log_check=True, is_expon=True)

    @staticmethod
    def test_single_values():
        btf.single_input_n_variables(custom_distribution=ExponentialDistribution.from_scipy_params, scipy_distribution=expon,
                                     parameters=[btf.loc_parameter, btf.scale_parameter],
                                     log_check=True, is_expon=True)
