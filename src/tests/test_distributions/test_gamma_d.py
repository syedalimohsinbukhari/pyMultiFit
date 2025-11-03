"""Created on Dec 10 00:56:43 2024"""

import numpy as np
import pytest
from scipy.stats import gamma

from . import base_test_functions as btf
from ...pymultifit.distributions import GammaDistribution
from ...pymultifit.distributions.backend import errorHandling as erH

np.random.seed(45)


class TestGammaDistribution:

    @staticmethod
    def test_initialization():
        dist = GammaDistribution(amplitude=1.0, shape=1.0, scale=1.0, normalize=False)
        assert dist.amplitude == 1.0
        assert dist.shape == 1.0
        assert dist.scale == 1.0
        assert not dist.norm

        # normalization should make amplitude = 1
        dist_normalized = GammaDistribution(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1

    @staticmethod
    def test_constraints():
        with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
            GammaDistribution(amplitude=-1.0)

        # amplitude should be internally updated to 1.0 if `normalize` is called
        distribution = GammaDistribution(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

        with pytest.raises(erH.NegativeShapeError, match=f"Shape {erH.neg_message}"):
            GammaDistribution(shape=-1.0)

        with pytest.raises(erH.NegativeScaleError, match=f"Scale {erH.neg_message}"):
            GammaDistribution(scale=-3.0)

    @staticmethod
    def test_edge_cases():
        btf.edge_cases(distribution=GammaDistribution(), log_check=True)

    @staticmethod
    def test_stats():
        btf.stats(
            custom_distribution=GammaDistribution.from_scipy_params,
            scipy_distribution=gamma,
            parameters=[btf.shape1_parameter, btf.loc_parameter, btf.scale_parameter],
            median=False,
        )

    @staticmethod
    def test_pdfs():
        btf.value_functions(
            custom_distribution=GammaDistribution.from_scipy_params,
            scipy_distribution=gamma,
            parameters=[btf.shape1_parameter, btf.loc_parameter, btf.scale_parameter],
            log_check=True,
        )

    @staticmethod
    def test_single_values():
        btf.single_input_n_variables(
            custom_distribution=GammaDistribution.from_scipy_params,
            scipy_distribution=gamma,
            parameters=[btf.shape1_parameter, btf.loc_parameter, btf.scale_parameter],
            log_check=True,
        )
