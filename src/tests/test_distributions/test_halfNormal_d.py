"""Created on Dec 16 14:36:17 2024"""

import numpy as np
import pytest
from scipy.stats import halfnorm

from . import base_test_functions as btf
from ...pymultifit.distributions import FoldedNormalDistribution, HalfNormalDistribution
from ...pymultifit.distributions.backend import errorHandling as erH

np.random.seed(42)


class TestHalfNormalDistribution:

    @staticmethod
    def test_initialization():
        dist = HalfNormalDistribution(amplitude=2.0, scale=1.0, normalize=False)
        assert dist.amplitude == 2.0
        assert dist.scale == 1.0
        assert not dist.norm

        dist_normalized = FoldedNormalDistribution(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1.0

    @staticmethod
    def test_constraints():
        with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
            HalfNormalDistribution(amplitude=-1.0, normalize=False)

        # amplitude should be internally updated to 1.0 if `normalize` is called
        distribution = HalfNormalDistribution(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

    @staticmethod
    def test_edge_cases():
        btf.edge_cases(distribution=HalfNormalDistribution(), log_check=True)

    @staticmethod
    def test_stats():
        btf.stats(
            custom_distribution=HalfNormalDistribution.from_scipy_params,
            scipy_distribution=halfnorm,
            parameters=[btf.loc1_parameter, btf.scale_parameter],
            median=False,
        )

    @staticmethod
    def test_pdfs():
        btf.value_functions(
            custom_distribution=HalfNormalDistribution.from_scipy_params,
            scipy_distribution=halfnorm,
            parameters=[btf.loc1_parameter, btf.scale_parameter],
            log_check=True,
        )

    @staticmethod
    def test_single_values():
        btf.single_input_n_variables(
            custom_distribution=HalfNormalDistribution.from_scipy_params,
            scipy_distribution=halfnorm,
            parameters=[btf.loc1_parameter, btf.scale_parameter],
            log_check=True,
        )
