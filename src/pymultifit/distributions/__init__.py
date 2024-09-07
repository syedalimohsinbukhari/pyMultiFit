"""Created on Aug 03 20:07:36 2024"""

from typing import Optional

from .gaussian_d import GaussianDistribution
from .laplace_d import LaplaceDistribution
from .logNorm_d import LogNormalDistribution
from .others import cubic, line, linear, nth_polynomial, quadratic
from .skewNorm_d import SkewedNormalDistribution

oFloat = Optional[float]
