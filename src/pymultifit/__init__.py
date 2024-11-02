"""Created on Jul 18 00:15:42 2024"""

import numpy as np

from .version import __author__, __copyright__, __description__, __email__, __license__, __url__, __version__

EPSILON = np.finfo(float).eps

GAUSSIAN = 'gaussian'
NORMAL = GAUSSIAN

LOG_NORMAL = 'log_normal'
SKEW_NORMAL = 'skew_normal'
LAPLACE = 'laplace'
GAMMA = 'gamma'
BETA = 'beta'
ARCSINE = 'arcSine'

LINE = 'line'
LINEAR = LINE
QUADRATIC = 'quad'
CUBIC = 'cubic'
