"""Created on Aug 03 20:07:36 2024"""

from typing import Optional

from .beta_d import BetaDistribution
from .chiSquare_d import ChiSquareDistribution
from .exponential_d import ExponentialDistribution
from .foldedHalfNormal_d import FoldedHalfNormalDistribution
from .gamma_d import GammaDistributionSR, GammaDistributionSS
from .gaussian_d import GaussianDistribution
from .halfNormal_d import HalfNormalDistribution
from .laplace_d import LaplaceDistribution
from .logNorm_d import LogNormalDistribution
from .norris_d import Norris2005Distribution, Norris2011Distribution
from .others import cubic, line, linear, nth_polynomial, quadratic
from .powerLaw_d import PowerLawDistribution
from .skewNorm_d import SkewedNormalDistribution

oFloat = Optional[float]
