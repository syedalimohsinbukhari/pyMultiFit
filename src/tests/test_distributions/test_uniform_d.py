"""Created on Dec 14 03:28:42 2024"""

import numpy as np
import pytest
from scipy.stats import uniform

from . import base_test_functions as btf
from ...pymultifit.distributions import UniformDistribution
from ...pymultifit.distributions.backend import errorHandling as erH

np.random.seed(42)


class TestUniformDistribution:

    @staticmethod
    def test_initialization():
        dist = UniformDistribution(amplitude=2.0, normalize=False)
        assert dist.amplitude == 2.0
        assert dist.low == 0.0
        assert dist.high == 1.0
        assert not dist.norm

        # normalization should make amplitude = 1
        dist_normalized = UniformDistribution(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1.0

    @staticmethod
    def test_constraints():
        with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
            UniformDistribution(amplitude=-1.0, normalize=False)

        # amplitude should be internally updated to 1.0 if `normalize` is called
        distribution = UniformDistribution(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

    @staticmethod
    def test_edge_cases():
        btf.edge_cases(distribution=UniformDistribution(), log_check=True)

    @staticmethod
    def test_stats():
        btf.stats(
            custom_distribution=UniformDistribution.from_scipy_params,
            scipy_distribution=uniform,
            parameters=[btf.loc1_parameter, btf.scale_parameter],
            median=False,
            equal_case=True,
            equal_params=np.array([1, 1]),
        )

    @staticmethod
    def test_pdfs():
        btf.value_functions(
            custom_distribution=UniformDistribution.from_scipy_params,
            scipy_distribution=uniform,
            parameters=[btf.loc1_parameter, btf.scale_parameter],
            log_check=True,
        )

    @staticmethod
    def test_single_values():
        btf.single_input_n_variables(
            custom_distribution=UniformDistribution.from_scipy_params,
            scipy_distribution=uniform,
            parameters=[btf.loc1_parameter, btf.scale_parameter],
            log_check=True,
        )
