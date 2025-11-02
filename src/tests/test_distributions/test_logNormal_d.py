"""Created on Dec 14 04:10:02 2024"""

import numpy as np
import pytest
from scipy.stats import lognorm

from . import base_test_functions as btf
from ...pymultifit.distributions import LogNormalDistribution
from ...pymultifit.distributions.backend import errorHandling as erH

np.random.seed(42)


class TestLogNormalDistribution:

    @staticmethod
    def test_initialization():
        dist = LogNormalDistribution(amplitude=2.0, mu=1.0, std=0.5, normalize=False)
        assert dist.amplitude == 2.0
        assert dist.mu == np.log(1)
        assert dist.std == 0.5
        assert not dist.norm

        dist_normalized = LogNormalDistribution(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1.0

    @staticmethod
    def test_constraints():
        with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
            LogNormalDistribution(amplitude=-1.0, normalize=False)

        # amplitude should be internally updated to 1.0 if `normalize` is called
        distribution = LogNormalDistribution(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

        with pytest.raises(erH.NegativeStandardDeviationError, match=f"Standard deviation {erH.neg_message}"):
            LogNormalDistribution(std=-3.0)

    @staticmethod
    def test_edge_cases():
        btf.edge_cases(distribution=LogNormalDistribution(), log_check=True)

    @staticmethod
    def test_stats():
        btf.stats(
            custom_distribution=LogNormalDistribution.from_scipy_params,
            scipy_distribution=lognorm,
            parameters=[btf.shape_parameter, btf.loc_parameter, btf.scale_parameter],
            median=False,
        )

    @staticmethod
    def test_pdfs():
        btf.value_functions(
            custom_distribution=LogNormalDistribution.from_scipy_params,
            scipy_distribution=lognorm,
            parameters=[btf.shape_parameter, btf.loc_parameter, btf.scale_parameter],
            log_check=True,
        )

    @staticmethod
    def test_single_values():
        btf.single_input_n_variables(
            custom_distribution=LogNormalDistribution.from_scipy_params,
            scipy_distribution=lognorm,
            parameters=[btf.shape_parameter, btf.loc_parameter, btf.scale_parameter],
            log_check=True,
        )
