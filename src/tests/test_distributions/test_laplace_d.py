"""Created on Dec 10 16:33:01 2024"""

import numpy as np
import pytest
from scipy.stats import laplace

from . import base_test_functions as btf
from ...pymultifit.distributions import LaplaceDistribution
from ...pymultifit.distributions.backend import errorHandling as erH

np.random.seed(42)


class TestLaplaceDistribution:

    @staticmethod
    def test_initialization():
        dist = LaplaceDistribution(amplitude=2.0, mean=1.0, diversity=0.5, normalize=False)
        assert dist.amplitude == 2.0
        assert dist.mu == 1.0
        assert dist.b == 0.5
        assert not dist.norm

        dist_normalized = LaplaceDistribution(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1.0

    @staticmethod
    def test_constraints():
        with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
            LaplaceDistribution(amplitude=-1.0, normalize=False)

        # amplitude should be internally updated to 1.0 if `normalize` is called
        distribution = LaplaceDistribution(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

        with pytest.raises(erH.NegativeScaleError, match=f"Diversity {erH.neg_message}"):
            LaplaceDistribution(amplitude=1.0, diversity=-3.0, normalize=True)

    @staticmethod
    def test_edge_cases():
        btf.edge_cases(distribution=LaplaceDistribution(), log_check=True)

    @staticmethod
    def test_stats():
        btf.stats(custom_distribution=LaplaceDistribution.scipy_like, scipy_distribution=laplace,
                  parameters=[btf.loc_parameter, btf.scale_parameter])

    @staticmethod
    def test_pdfs():
        btf.value_functions(custom_distribution=LaplaceDistribution.scipy_like, scipy_distribution=laplace,
                            parameters=[btf.loc_parameter, btf.scale_parameter], log_check=True)

    @staticmethod
    def test_single_values():
        btf.single_input_n_variables(custom_distribution=LaplaceDistribution.scipy_like,
                                     scipy_distribution=laplace,
                                     parameters=[btf.loc_parameter, btf.scale_parameter],
                                     log_check=True)
