"""Created on Jul 18 00:15:42 2024"""

import functools
from functools import wraps
from typing import Union, Tuple, List, Annotated

import numpy as np
import scipy.special as ssp
from deprecated.sphinx import deprecated
from numpy.typing import NDArray

from .version import __author__, __copyright__, __description__, __email__, __license__, __url__, __version__


def check_scale_positive(func):
    """Decorator that returns NaN array if scale < 0."""

    @wraps(func)
    def wrapper(x, *args, **kwargs):
        scale = args[-2]
        if scale is not None and scale < 0:
            # Return NaNs of the same shape as x
            return np.full_like(x, np.nan, dtype=float)
        return func(x, *args, **kwargs)

    return wrapper


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


def suppress_numpy_warnings():
    """
    A decorator that suppresses NumPy warnings using np.errstate.

    Parameters (all optional):
        divide, over, under, invalid: Can be 'ignore', 'warn', 'raise', 'call', 'print', or 'log'
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with np.errstate(all="ignore"):
                return func(*args, **kwargs)

        return wrapper

    return decorator


doc_style = "numpy_napoleon_with_merge"

INF = np.inf
LOG = np.log
SQRT = np.sqrt

# taken from https://stackoverflow.com/a/19141711
EPSILON = np.finfo(float).eps
epsilon = SQRT(EPSILON)

TWO = 2.0
SQRT_TWO = SQRT(TWO)
LOG_TWO = LOG(TWO)
LOG_SQRT_TWO = ssp.xlogy(0.5, TWO)

PI = np.pi
SQRT_PI = SQRT(PI)
LOG_PI = LOG(PI)
LOG_SQRT_PI = ssp.xlogy(0.5, PI)

TWO_PI = 2 * PI
SQRT_TWO_PI = SQRT(TWO_PI)
LOG_TWO_PI = LOG(TWO_PI)
LOG_SQRT_TWO_PI = ssp.xlogy(0.5, TWO_PI)

INV_PI = 1.0 / PI
TWO_BY_PI = 2.0 * INV_PI
SQRT_TWO_BY_PI = SQRT(TWO_BY_PI)
LOG_TWO_BY_PI = LOG(TWO_BY_PI)
LOG_SQRT_TWO_BY_PI = ssp.xlogy(0.5, TWO_BY_PI)

ListOrNdArray = Union[List[int or float], np.ndarray]
OneDArray = Annotated[NDArray[np.float64], "1D array"]

ParamTuple = Tuple[int or float, ...]
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
