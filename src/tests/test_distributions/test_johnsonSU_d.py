"""Created on Dec 16 14:36:17 2024"""

import numpy as np
from scipy.stats import johnsonsu

from . import base_test_functions as btf
from ...pymultifit.distributions import JohnsonSUDistribution

np.random.seed(42)


class TestHalfNormalDistribution:

    @staticmethod
    def test_initialization():
        dist = JohnsonSUDistribution(amplitude=2.0, lambda_=1.0, normalize=False)
        assert dist.amplitude == 2.0
        assert dist.lambda_ == 1.0
        assert not dist.norm

        dist_normalized = JohnsonSUDistribution(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1.0

    @staticmethod
    def test_edge_cases():
        btf.edge_cases(distribution=JohnsonSUDistribution(), log_check=True)

    @staticmethod
    def test_stats():
        btf.stats(
            custom_distribution=JohnsonSUDistribution.from_scipy_params,
            scipy_distribution=johnsonsu,
            parameters=[btf.shape1_parameter, btf.shape2_parameter, btf.loc_parameter, btf.scale_parameter],
            median=False,
        )

    @staticmethod
    def test_pdfs():
        btf.value_functions(
            custom_distribution=JohnsonSUDistribution.from_scipy_params,
            scipy_distribution=johnsonsu,
            parameters=[btf.shape1_parameter, btf.shape2_parameter, btf.loc_parameter, btf.scale_parameter],
            log_check=True,
        )

    @staticmethod
    def test_single_values():
        btf.single_input_n_variables(
            custom_distribution=JohnsonSUDistribution.from_scipy_params,
            scipy_distribution=johnsonsu,
            parameters=[btf.shape1_parameter, btf.shape2_parameter, btf.loc_parameter, btf.scale_parameter],
            log_check=True,
        )
