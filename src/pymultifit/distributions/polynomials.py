"""Created on Aug 10 23:37:54 2024"""

from typing import List

import numpy as np
from numpy.typing import NDArray

from .backend import BaseDistribution
from .utilities_d import line, quadratic, cubic, nth_polynomial


class Line(BaseDistribution):
    def __init__(
        self,
        slope: float = 1.0,
        intercept: float = 1.0,
        normalize: bool = False,
    ):
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
        return line(
            x,
            slope=self.slope,
            intercept=self.intercept,
        )


class Quadratic(BaseDistribution):
    def __init__(
        self,
        a: float = 1.0,
        b: float = 1.0,
        c: float = 1.0,
        normalize: bool = False,
    ):
        self.a = a
        self.b = b
        self.c = c

        self.norm = normalize

    def pdf(self, x: NDArray) -> NDArray:
        return quadratic(
            x,
            a=self.a,
            b=self.b,
            c=self.c,
        )


class Cubic(BaseDistribution):
    def __init__(
        self,
        a: float = 1.0,
        b: float = 1.0,
        c: float = 1.0,
        d: float = 1.0,
        normalize: bool = False,
    ):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

        self.norm = normalize

    def pdf(self, x: NDArray):
        return cubic(
            x,
            a=self.a,
            b=self.b,
            c=self.c,
            d=self.d,
        )


class Polynomial(BaseDistribution):
    def __init__(self, degree: List[float], normalize: bool = False):
        self.degree = degree

        self.norm = normalize

    def pdf(self, x: NDArray):
        return nth_polynomial(
            x,
            coefficients=self.degree,
        )
