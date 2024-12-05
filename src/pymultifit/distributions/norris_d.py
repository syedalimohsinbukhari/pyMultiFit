"""Created on Nov 30 04:23:27 2024"""

from typing import Any, Dict

import numpy as np

from .backend import BaseDistribution
from .utilities import norris2011


class Norris2011Distribution(BaseDistribution):

    def __init__(self, amplitude: float = 1., tau: float = 1., xi: float = 1., normalize: bool = False):
        self.amplitude = 1. if normalize else amplitude
        self.tau = tau
        self.xi = xi

        self.norm = normalize

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return norris2011(x, amplitude=self.amplitude, tau=self.tau, xi=self.xi, normalize=self.norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("The Norris2011 function doesn't has CDF implemented yet.")

    def stats(self) -> Dict[str, Any]:
        raise NotImplementedError("The Norris2011 function doesn't has stats implemented yet.")


class Norris2005Distribution(Norris2011Distribution):

    def __init__(self, amplitude: float = 1., rise_time: float = 1., decay_time: float = 1., normalize: bool = False):
        tau = np.sqrt(rise_time * decay_time)
        xi = np.sqrt(rise_time / decay_time)
        super().__init__(amplitude=amplitude, tau=tau, xi=xi, normalize=normalize)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("The Norris2005 function doesn't has CDF implemented yet.")

    def stats(self) -> Dict[str, Any]:
        raise NotImplementedError("The Norris2005 function doesn't has stats implemented yet.")
