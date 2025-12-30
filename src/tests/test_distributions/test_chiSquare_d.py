"""Created on Dec 14 12:49:44 2024"""

import pytest
from scipy.stats import chi2

from . import base_test_functions as btf
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
    def test_edge_cases():
        btf.edge_cases(distribution=ChiSquareDistribution(), log_check=True)

    @staticmethod
    def test_stats():
        btf.stats(
            custom_distribution=ChiSquareDistribution.from_scipy_params,
            scipy_distribution=chi2,
            parameters=[btf.shape1_parameter, btf.loc1_parameter, btf.scale_parameter],
            median=False,
        )

    @staticmethod
    def test_pdfs():
        btf.value_functions(
            custom_distribution=ChiSquareDistribution.from_scipy_params,
            scipy_distribution=chi2,
            parameters=[btf.shape1_parameter, btf.loc1_parameter, btf.scale_parameter],
            log_check=True,
        )

    @staticmethod
    def test_single_values():
        btf.single_input_n_variables(
            custom_distribution=ChiSquareDistribution.from_scipy_params,
            scipy_distribution=chi2,
            parameters=[btf.shape1_parameter, btf.loc1_parameter, btf.scale_parameter],
            log_check=True,
        )
