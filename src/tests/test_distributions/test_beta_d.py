"""Created on Dec 14 06:55:53 2024"""

from scipy.stats import beta

from . import base_test_functions as btf
from ...pymultifit.distributions import BetaDistribution


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
    def test_edge_cases():
        btf.edge_cases(distribution=BetaDistribution(), log_check=True)

    @staticmethod
    def test_stats():
        btf.stats(
            custom_distribution=BetaDistribution.from_scipy_params,
            scipy_distribution=beta,
            parameters=[btf.shape1_parameter, btf.shape2_parameter, btf.loc_parameter, btf.scale_parameter],
        )

    @staticmethod
    def test_pdfs():
        btf.value_functions(
            custom_distribution=BetaDistribution.from_scipy_params,
            scipy_distribution=beta,
            parameters=[btf.shape1_parameter, btf.shape2_parameter, btf.loc_parameter, btf.scale_parameter],
            log_check=True,
        )

    @staticmethod
    def test_single_values():
        btf.single_input_n_variables(
            custom_distribution=BetaDistribution.from_scipy_params,
            scipy_distribution=beta,
            parameters=[btf.shape1_parameter, btf.shape2_parameter, btf.loc_parameter, btf.scale_parameter],
            log_check=True,
        )
