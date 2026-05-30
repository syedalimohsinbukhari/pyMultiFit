"""Created on Jul 18 00:15:42 2024"""

import functools

import numpy as np
import scipy.special as ssp
from deprecation import deprecated as _deprecated

from .version import __author__, __copyright__, __description__, __email__, __license__, __url__, __version__


def mark_deprecated(ver_: str, new: str):
    """Decorator that marks a `scipy_like`-style method as deprecated.

    Parameters
    ----------
    ver_ :
        The version where the method is deprecated.
    new :
        The name of the method to use instead.
    """

    def _decorator(func):
        method_name = func.__name__
        reason = f"Use ``{new}`` instead of ``{method_name}``. ``{method_name}`` will be removed in a future release."
        return _deprecated(deprecated_in=ver_, removed_in=None, details=reason)(func)

    return _decorator


def _md_scipy_like(ver_: str, new: str = "from_scipy_params"):
    return mark_deprecated(ver_=ver_, new=new)


def suppress_numpy_warnings():
    """A decorator that suppresses NumPy warnings using ``np.errstate``."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with np.errstate(all="ignore"):
                return func(*args, **kwargs)

        return wrapper

    return decorator


doc_style = "numpy_napoleon_with_merge"

_UNSET = object()

INF = np.inf
LOG = np.log
SQRT = np.sqrt
EXP = np.exp
NAN = np.nan

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

NAN_DICT = {"mean": NAN, "median": NAN, "variance": NAN, "std": NAN}
