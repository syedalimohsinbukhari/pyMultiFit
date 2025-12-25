"""Created on Jul 18 00:15:42 2024"""

from typing import Union, Tuple, List, Annotated

import numpy as np
import scipy.special as ssp
from deprecated.sphinx import deprecated
from numpy.typing import NDArray

from .version import __author__, __copyright__, __description__, __email__, __license__, __url__, __version__


def mark_deprecated(ver_: str, new: str):
    """Decorator that marks a scipy_like-style method as deprecated.

    Automatically extracts the method name and constructs a standardized warning.

    Parameters
    ----------
    ver_ : str
        The version where the method is deprecated.
    new : str
        The name of the method to use instead.
    """

    def decorator(func):
        method_name = func.__name__
        reason = f"Use `{new}` instead of `{method_name}`. `{method_name}` will be removed in a future release."
        return deprecated(version=ver_, reason=reason)(func)

    return decorator


def md_scipy_like(ver_: str, new: str = "from_scipy_params"):
    return mark_deprecated(ver_, new)


doc_style = "numpy_napoleon_with_merge"

INF = np.inf
LOG = np.log

# taken from https://stackoverflow.com/a/19141711
EPSILON = np.finfo(float).eps
epsilon = np.sqrt(EPSILON)

TWO = 2.0
SQRT_TWO = np.sqrt(TWO)
LOG_TWO = LOG(TWO)
LOG_SQRT_TWO = ssp.xlogy(0.5, TWO)

PI = np.pi
SQRT_PI = np.sqrt(PI)
LOG_PI = LOG(PI)
LOG_SQRT_PI = ssp.xlogy(0.5, PI)

TWO_PI = 2 * PI
SQRT_TWO_PI = np.sqrt(TWO_PI)
LOG_TWO_PI = LOG(TWO_PI)
LOG_SQRT_TWO_PI = ssp.xlogy(0.5, TWO_PI)

INV_PI = 1.0 / PI
TWO_BY_PI = 2.0 * INV_PI
SQRT_TWO_BY_PI = np.sqrt(TWO_BY_PI)
LOG_TWO_BY_PI = LOG(TWO_BY_PI)
LOG_SQRT_TWO_BY_PI = ssp.xlogy(0.5, TWO_BY_PI)

ListOrNdArray = Union[List[int | float], np.ndarray]
OneDArray = Annotated[NDArray[np.float64], "1D array"]

ParamTuple = Tuple[int | float, ...]
Params_ = Union[List[ParamTuple], np.ndarray]

GAUSSIAN = "gaussian"
NORMAL = GAUSSIAN

ARC_SINE = "arc_sine"
BETA = "beta"
CHI_SQUARE = "chi_square"
EXPONENTIAL = "exponential"
FOLDED_NORMAL = "folded_normal"
GAMMA = "gamma"
HALF_NORMAL = "half_normal"
LAPLACE = "laplace"
LOG_NORMAL = "log_normal"
SKEW_NORMAL = "skew_normal"

LINE = "line"
LINEAR = LINE
QUADRATIC = "quad"
CUBIC = "cubic"
