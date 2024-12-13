"""Created on Dec 14 03:28:42 2024"""

import pytest

from ...pymultifit.distributions.backend import errorHandling as erH
from ...pymultifit.distributions.uniform_d import UniformDistribution


class TestGaussianDistribution:

    @staticmethod
    def test_initialization():
        dist = UniformDistribution(amplitude=2.0, normalize=False)
        assert dist.amplitude == 2.0
        assert dist.low == 0.0
        assert dist.high == 1.0
        assert not dist.norm

        dist_normalized = UniformDistribution(amplitude=2.0, normalize=True)
        assert dist_normalized.amplitude == 1.0

    @staticmethod
    def test_constraints():
        with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
            UniformDistribution(amplitude=-1.0, normalize=False)

        # amplitude should be internally updated to 1.0 if `normalize` is called
        distribution = UniformDistribution(amplitude=-1.0, normalize=True)
        assert distribution.amplitude == 1.0
