"""Created on Dec 14 06:55:53 2024"""
import pytest
from scipy.stats import beta

from . import base_test_functions as btf
from ...pymultifit.distributions import BetaDistribution
from ...pymultifit.distributions.backend import errorHandling as erH


class TestBetaDistribution:

    @staticmethod
    def test_initialization():
        dist_ = BetaDistribution(amplitude=2.0, normalize=False)
        assert dist_.amplitude == 2.0
        assert dist_.alpha == 1.0
        assert dist_.beta == 1.0
        assert not dist_.norm

        dist_normalized = BetaDistribution(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1.0

    @staticmethod
    def test_constraints():
        with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
            BetaDistribution(amplitude=-1.0, normalize=False)

        # amplitude should be internally updated to 1.0 if `normalize` is called
        distribution = BetaDistribution(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0

        with pytest.raises(erH.NegativeAlphaError, match=f"Alpha {erH.neg_message}"):
            BetaDistribution(alpha=-3.0)

        with pytest.raises(erH.NegativeBetaError, match=f"Beta {erH.neg_message}"):
            BetaDistribution(beta=-3.0)

    @staticmethod
    def test_edge_cases():
        btf.edge_cases(distribution=BetaDistribution(), log_check=True)

    @staticmethod
    def test_stats():
        btf.stats(custom_distribution=BetaDistribution.scipy_like,
                  scipy_distribution=beta,
                  parameters=[btf.shape_parameter, btf.shape_parameter, btf.loc_parameter, btf.scale_parameter])

    @staticmethod
    def test_pdfs():
        btf.value_functions(custom_distribution=BetaDistribution.scipy_like, scipy_distribution=beta,
                            parameters=[btf.shape_parameter, btf.shape_parameter, btf.loc_parameter,
                                        btf.scale_parameter], log_check=True)

    @staticmethod
    def test_single_values():
        btf.single_input_n_variables(custom_distribution=BetaDistribution.scipy_like, scipy_distribution=beta,
                                     parameters=[btf.shape_parameter, btf.shape_parameter,
                                                 btf.loc_parameter, btf.scale_parameter], log_check=True)
