"""Created on Aug 03 20:07:36 2024"""

from typing import Optional

from .arcSine_d import ArcSineDistribution
from .beta_d import BetaDistribution
from .chiSquare_d import ChiSquareDistribution
from .exponential_d import ExponentialDistribution
from .foldedNormal_d import FoldedNormalDistribution
from .gamma_d import GammaDistributionSR, GammaDistributionSS
from .gaussian_d import GaussianDistribution
from .halfNormal_d import HalfNormalDistribution
from .laplace_d import LaplaceDistribution
from .logNormal_d import LogNormalDistribution
from .others import cubic, line, linear, nth_polynomial, quadratic
from .skewNormal_d import SkewNormalDistribution
from .uniform_d import UniformDistribution

oFloat = Optional[float]
