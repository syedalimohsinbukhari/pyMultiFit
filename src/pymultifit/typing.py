"""Created on Apr 07 13:24:05 2026"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike as _ArrayLike, NDArray as _NDArray

NDArray = _NDArray[np.floating]
ArrayLike = _ArrayLike

RaggedParams = list[tuple[int | float, ...]]  # mixed-length components
Params_ = RaggedParams | ArrayLike
