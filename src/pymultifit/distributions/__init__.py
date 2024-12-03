"""Created on Aug 03 20:07:36 2024"""

from typing import Optional

from .beta_d import _beta, BetaDistribution
from .chiSquare_d import chi_squared_, ChiSquareDistribution
from .exponential_d import exponential_, ExponentialDistribution
from .foldedHalfNormal_d import folded_half_normal_, FoldedHalfNormalDistribution
from .gamma_d import gamma_, GammaDistributionSR, GammaDistributionSS
from .gaussian_d import gaussian_, GaussianDistribution
from .halfNormal_d import half_normal_, HalfNormalDistribution
from .laplace_d import laplace_, LaplaceDistribution
from .logNorm_d import log_normal_, LogNormalDistribution
from .norris_d import norris2005, Norris2005Distribution, norris2011, Norris2011Distribution
from .others import cubic, line, linear, nth_polynomial, quadratic
from .powerLaw_d import power_law_, PowerLawDistribution
from .skewNorm_d import SkewedNormalDistribution

oFloat = Optional[float]
