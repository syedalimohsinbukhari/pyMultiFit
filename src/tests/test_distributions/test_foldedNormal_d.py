"""Created on Dec 15 23:23:05 2024"""

import numpy as np
import pytest
from scipy.stats import foldnorm

from . import base_test_functions as btf
from ...pymultifit.distributions import FoldedNormalDistribution
from ...pymultifit.distributions.backend import errorHandling as erH

np.random.seed(45)


class TestChiSquareDistribution:

    @staticmethod
    def test_initialization():
        dist = FoldedNormalDistribution(amplitude=2.0, mu=1.0, sigma=0.5, normalize=False)
        assert dist.amplitude == 2.0
        assert dist.mu == 1.0
        assert dist.sigma == 0.5
        assert not dist.norm

        dist_normalized = FoldedNormalDistribution(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1.0

    @staticmethod
    def test_constraints():
        with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
            FoldedNormalDistribution(amplitude=-1.0, normalize=False)

        # amplitude should be internally updated to 1.0 if `normalize` is called
        distribution = FoldedNormalDistribution(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

    @staticmethod
    def test_edge_cases():
        btf.edge_cases(distribution=FoldedNormalDistribution(), log_check=True)

    @staticmethod
    def test_stats():
        btf.stats(
            custom_distribution=FoldedNormalDistribution.from_scipy_params,
            scipy_distribution=foldnorm,
            parameters=[btf.shape_parameter, btf.loc_parameter, btf.scale_parameter],
            median=False,
        )

    @staticmethod
    def test_pdfs():
        btf.value_functions(
            custom_distribution=FoldedNormalDistribution.from_scipy_params,
            scipy_distribution=foldnorm,
            parameters=[btf.shape_parameter, btf.loc_parameter, btf.scale_parameter],
            log_check=True,
        )

    @staticmethod
    def test_single_values():
        btf.single_input_n_variables(
            custom_distribution=FoldedNormalDistribution.from_scipy_params,
            scipy_distribution=foldnorm,
            parameters=[btf.shape_parameter, btf.loc_parameter, btf.scale_parameter],
            log_check=True,
        )
