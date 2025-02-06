"""Created on Dec 08 11:48:14 2024"""

import numpy as np
import pytest
from scipy.stats import norm

from . import base_test_functions as btf
from ...pymultifit.distributions import GaussianDistribution
from ...pymultifit.distributions.backend import errorHandling as erH

np.random.seed(45)


class TestGaussianDistribution:

    @staticmethod
    def test_initialization():
        dist = GaussianDistribution(amplitude=2.0, mu=1.0, std=0.5, normalize=False)
        assert dist.amplitude == 2.0
        assert dist.mu == 1.0
        assert dist.std_ == 0.5
        assert not dist.norm

        dist_normalized = GaussianDistribution(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1.0

    @staticmethod
    def test_constraints():
        with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
            GaussianDistribution(amplitude=-1.0, normalize=False)

        # amplitude should be internally updated to 1.0 if `normalize` is called
        distribution = GaussianDistribution(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

        with pytest.raises(erH.NegativeStandardDeviationError, match=f"Standard deviation {erH.neg_message}"):
            GaussianDistribution(std=-3.0)

    @staticmethod
    def test_edge_cases():
        btf.edge_cases(distribution=GaussianDistribution(), log_check=True)

    @staticmethod
    def test_stats():
        btf.stats(custom_distribution=GaussianDistribution.scipy_like, scipy_distribution=norm,
                  parameters=[btf.loc_parameter, btf.scale_parameter])

    @staticmethod
    def test_pdfs():
        btf.value_functions(custom_distribution=GaussianDistribution.scipy_like, scipy_distribution=norm,
                            parameters=[btf.loc_parameter, btf.scale_parameter], log_check=True)

    @staticmethod
    def test_single_values():
        btf.single_input_n_variables(custom_distribution=GaussianDistribution.scipy_like,
                                     scipy_distribution=norm,
                                     parameters=[btf.loc_parameter, btf.scale_parameter],
                                     log_check=True)
