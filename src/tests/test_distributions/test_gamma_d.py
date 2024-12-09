"""Created on Dec 10 00:56:43 2024"""

import numpy as np
import pytest

from ...pymultifit.distributions import GammaDistributionSS
from ...pymultifit.distributions.backend import errorHandling as erH


def test_initialization():
    amp, shp, scl = 1, 1, 1
    dist = GammaDistributionSS(amplitude=amp, shape=shp, scale=scl, normalize=False)
    assert dist.amplitude == amp
    assert dist.shape == shp
    assert dist.rate == 1 / scl
    assert not dist.norm

    # normalization should make amplitude = 1
    dist_normalized = GammaDistributionSS(amplitude=2.0, normalize=True)
    assert dist_normalized.amplitude == 1


def test_constraints():
    with pytest.raises(erH.NegativeAmplitudeError, match=f"Amplitude {erH.neg_message}"):
        GammaDistributionSS(amplitude=-1.0, normalize=True)

    with pytest.raises(erH.NegativeShapeError, match=f"Shape {erH.neg_message}"):
        GammaDistributionSS(amplitude=1.0, shape=-1.0, normalize=True)


def test_edge_cases():
    dist = GammaDistributionSS()
    x = np.array([])
    result = dist.pdf(x)
    assert result.size == 0  # Should return an empty array
