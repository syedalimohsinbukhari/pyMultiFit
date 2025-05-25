"""Created on Jul 18 00:15:42 2024"""

from typing import Union, Sequence

import numpy as np
from numpy._typing import NDArray

from .version import (
    __author__,
    __copyright__,
    __description__,
    __email__,
    __license__,
    __url__,
    __version__,
)

doc_style = "numpy_napoleon_with_merge"

fArray = Union[float, NDArray[np.floating]]
# taken from https://stackoverflow.com/a/19141711
EPSILON = np.finfo(float).eps
epsilon = np.sqrt(EPSILON)

SeqFloat = Sequence[Sequence[float]]
Sequences_ = Union[NDArray[np.floating], SeqFloat]

GAUSSIAN = "gaussian"
NORMAL = GAUSSIAN

ARC_SINE = "arc_sine"
BETA = "beta"
CHI_SQUARE = "chi_square"
EXPONENTIAL = "exponential"
FOLDED_NORMAL = "folded_normal"
GAMMA_SR = "gamma_sr"
GAMMA_SS = "gamma_ss"
HALF_NORMAL = "half_normal"
LAPLACE = "laplace"
LOG_NORMAL = "log_normal"
SKEW_NORMAL = "skew_normal"

LINE = "line"
LINEAR = LINE
QUADRATIC = "quad"
CUBIC = "cubic"
