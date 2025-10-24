"""Created on Feb 02 17:13:40 2025"""

import numpy as np
import pytest
from scipy.stats import invgamma

from . import base_test_functions as btf
from ...pymultifit.distributions import ScaledInverseChiSquareDistribution
from ...pymultifit.distributions.backend import errorHandling as erH

np.random.seed(42)


class TestScaledInvChiSquareDistribution:

    @staticmethod
    def test_initialization():
        dist = ScaledInverseChiSquareDistribution(amplitude=2.0, df=1.0, scale=0.5, normalize=False)
        assert dist.amplitude == 2.0
        assert dist.df == 1.0
        assert dist.scale == 0.5
        assert dist.tau2 == dist.scale / dist.df
        assert not dist.norm

        dist_normalized = ScaledInverseChiSquareDistribution(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1.0

    @staticmethod
    def test_constraints():
        with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
            ScaledInverseChiSquareDistribution(amplitude=-1.0, normalize=False)

        # amplitude should be internally updated to 1.0 if `normalize` is called
        distribution = ScaledInverseChiSquareDistribution(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

        with pytest.raises(erH.NegativeScaleError, match=f"Scale {erH.neg_message}"):
            ScaledInverseChiSquareDistribution(scale=-3.0)

    @staticmethod
    def test_edge_cases():
        btf.edge_cases(distribution=ScaledInverseChiSquareDistribution(), log_check=True)

    @staticmethod
    def test_stats():
        btf.stats(
            custom_distribution=ScaledInverseChiSquareDistribution.from_scipy_params,
            scipy_distribution=invgamma,
            parameters=[btf.shape_parameter, btf.loc_parameter, btf.scale_parameter],
            median=False,
            is_scaled_inv_chi=True,
        )

    @staticmethod
    def test_pdfs():
        btf.value_functions(
            custom_distribution=ScaledInverseChiSquareDistribution.from_scipy_params,
            scipy_distribution=invgamma,
            parameters=[btf.shape_parameter, btf.loc_parameter, btf.scale_parameter],
            log_check=True,
            is_scaled_inv_chi=True,
        )

    @staticmethod
    def test_single_values():
        btf.single_input_n_variables(
            custom_distribution=ScaledInverseChiSquareDistribution.from_scipy_params,
            scipy_distribution=invgamma,
            parameters=[btf.shape_parameter, btf.loc_parameter, btf.scale_parameter],
            log_check=True,
            is_scaled_inv_chi=True,
        )
