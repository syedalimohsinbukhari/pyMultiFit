"""Created on Jul 12 05:01:19 2025"""

import numpy as np
from numpy import ndarray

from . import BaseDistribution
from ..utilities_d import line, quadratic, cubic


class LineFunction(BaseDistribution):
    def __init__(self, slope: float = 1.0, intercept: float = 1.0, normalize: bool = False):
        self.slope = slope
        self.intercept = intercept

        self.norm = normalize

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Calculates the line function.

        Parameters
        ----------
        x : np.ndarray
            Input array of values.

        Returns
        -------
        np.ndarray
            Array of the same shape as :math:`x`, containing the evaluated values.
        """
        return line(x, slope=self.slope, intercept=self.intercept)


class QuadraticFunction(BaseDistribution):
    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 1.0, normalize: bool = False):
        self.a = a
        self.b = b
        self.c = c

        self.norm = normalize

    def pdf(self, x: ndarray) -> ndarray:
        """Calculates the quadratic function.

        Parameters
        ----------
        x : np.ndarray
            Input array of values.

        Returns
        -------
        np.ndarray
            Array of the same shape as :math:`x`, containing the evaluated values.
        """
        return quadratic(x, a=self.a, b=self.b, c=self.c)


class CubicFunction(BaseDistribution):
    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 1.0, d: float = 1.0, normalize: bool = False):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

        self.norm = normalize

    def pdf(self, x: ndarray) -> ndarray:
        """Calculates the cubic function.

        Parameters
        ----------
        x : np.ndarray
            Input array of values.

        Returns
        -------
        np.ndarray
            Array of the same shape as :math:`x`, containing the evaluated values.
        """
        return cubic(x, a=self.a, b=self.b, c=self.c, d=self.d)
