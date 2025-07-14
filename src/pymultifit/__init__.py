"""Created on Jul 18 00:15:42 2024"""

from typing import Union, Sequence, Tuple, List

import numpy as np
from deprecated.sphinx import deprecated

from .version import (
    __author__,
    __copyright__,
    __description__,
    __email__,
    __license__,
    __url__,
    __version__,
)


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


def md_scipy_like(ver_: str, new: str = 'from_scipy_params'):
    return mark_deprecated(ver_, new)


doc_style = "numpy_napoleon_with_merge"

INF = np.inf
LOG = np.log

# taken from https://stackoverflow.com/a/19141711
EPSILON = np.finfo(float).eps
epsilon = np.sqrt(EPSILON)

listOrNdArray = Union[List[int | float], np.ndarray]

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
