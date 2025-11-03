"""Created on Aug 03 20:07:36 2024"""

from typing import Optional

from .arcSine_d import ArcSineDistribution
from .backend import LineFunction, CubicFunction, QuadraticFunction
from .betaPrime_d import BetaPrimeDistribution
from .beta_d import BetaDistribution
from .chiSquare_d import ChiSquareDistribution
from .exponential_d import ExponentialDistribution
from .foldedNormal_d import FoldedNormalDistribution
from .gamma_d import GammaDistribution
from .gaussian_d import GaussianDistribution
from .generalized import SymmetricGeneralizedNormalDistribution, ScaledInverseChiSquareDistribution
from .halfNormal_d import HalfNormalDistribution
from .johnsonSU_d import JohnsonSUDistribution
from .laplace_d import LaplaceDistribution
from .logNormal_d import LogNormalDistribution
from .skewNormal_d import SkewNormalDistribution
from .uniform_d import UniformDistribution

OptionalFloat = Optional[float]
