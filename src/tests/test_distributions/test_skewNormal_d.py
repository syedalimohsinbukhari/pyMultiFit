"""Created on Jan 03 16:13:58 2025"""

import numpy as np
import pytest
from scipy.stats import skewnorm

from ...pymultifit.distributions import SkewNormalDistribution
from ...pymultifit.distributions.backend import errorHandling as erH
from . import base_test_functions as btf

np.random.seed(42)


class TestSkewNormalDistribution:

    @staticmethod
    def test_initialization():
        dist = SkewNormalDistribution(amplitude=2.0, shape=1.0, location=0.5, scale=1.0, normalize=False)
        assert dist.amplitude == 2.0
        assert dist.location == 0.5
        assert dist.scale == 1.0
        assert not dist.norm

        dist_normalized = SkewNormalDistribution(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1.0

    @staticmethod
    def test_constraints():
        distribution = SkewNormalDistribution(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

    @staticmethod
    def test_edge_cases():
        btf.edge_cases(distribution=SkewNormalDistribution())

    @staticmethod
    def test_stats():
        btf.stats(
            custom_distribution=SkewNormalDistribution.from_scipy_params,
            scipy_distribution=skewnorm,
            parameters=[btf.shape1_parameter, btf.loc1_parameter, btf.scale_parameter],
            median=False,
        )

    @staticmethod
    def test_pdfs():
        btf.value_functions(
            custom_distribution=SkewNormalDistribution.from_scipy_params,
            scipy_distribution=skewnorm,
            parameters=[btf.shape1_parameter, btf.loc1_parameter, btf.scale_parameter],
        )

    @staticmethod
    def test_single_values():
        btf.single_input_n_variables(
            custom_distribution=SkewNormalDistribution.from_scipy_params,
            scipy_distribution=skewnorm,
            parameters=[btf.shape1_parameter, btf.loc1_parameter, btf.scale_parameter],
        )
