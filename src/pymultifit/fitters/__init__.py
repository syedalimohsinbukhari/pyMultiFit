"""Created on Aug 03 20:34:39 2024"""

import numpy as np

ArrayLike = np.typing.ArrayLike
NDArray = np.typing.NDArray[np.floating]

from .chiSquare_f import ChiSquareFitter
from .exponential_f import ExponentialFitter
from .foldedNormal_f import FoldedNormalFitter
from .gamma_f import GammaFitter
from .gaussian_f import GaussianFitter
from .halfNormal_f import HalfNormalFitter
from .laplace_f import LaplaceFitter
from .logNormal_f import LogNormalFitter
from .mixed_f import MixedDataFitter
from .polynomial_f import CubicFitter, LineFitter, QuadraticFitter
from .skewNormal_f import SkewNormalFitter
