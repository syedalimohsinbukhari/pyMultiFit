"""Created on Jul 18 00:15:42 2024"""

from typing import Tuple, Union, List

import numpy as np
from matplotlib import pyplot as plt

from .version import __author__, __copyright__, __description__, __email__, __license__, __url__, __version__

doc_style = 'numpy_napoleon_with_merge'

MPL_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

fArray = Union[float, np.ndarray]
# taken from https://stackoverflow.com/a/19141711
EPSILON = np.finfo(float).eps
epsilon = np.sqrt(EPSILON)

listOfFloats = List[float]
listOfTuples = List[Tuple[float, ...]]
listOfTuplesOrArray = Union[listOfTuples, np.ndarray]
listOfTuplesOrFloatsOrArray = Union[listOfFloats, listOfTuplesOrArray]

GAUSSIAN = 'gaussian'
NORMAL = GAUSSIAN

ARC_SINE = 'arc_sine'
BETA = 'beta'
CHI_SQUARE = 'chi_square'
EXPONENTIAL = 'exponential'
FOLDED_NORMAL = 'folded_normal'
GAMMA_SR = 'gamma_sr'
GAMMA_SS = 'gamma_ss'
HALF_NORMAL = 'half_normal'
LAPLACE = 'laplace'
LOG_NORMAL = 'log_normal'
SKEW_NORMAL = 'skew_normal'

LINE = 'line'
LINEAR = LINE
QUADRATIC = 'quad'
CUBIC = 'cubic'
