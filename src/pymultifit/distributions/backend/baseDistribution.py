"""Created on Aug 03 22:06:29 2024"""

from typing import Any, Dict

import numpy as np


class BaseDistribution:
    """
    Bare-bones class for statistical distributions to provide consistent methods.

    This class serves as a template for other distribution classes, defining the common interface
    for probability density function (PDF), cumulative distribution function (CDF), and statistics.
    """

    def pdf(self, x: np.array) -> np.array:
        """
        Compute the probability density function (PDF) for the distribution.

        Parameters
        ----------
        x : np.array
            Input values at which to evaluate the PDF.

        Returns
        -------
        np.array
            The PDF values at the input values.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def cdf(self, x: np.array) -> np.array:
        """
        Compute the cumulative density function (CDF) for the distribution.

        Parameters
        ----------
        x : np.array
            Input values at which to evaluate the PDF.

        Returns
        -------
        np.array
            The CDF values at the input values.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def stats(self) -> Dict[str, Any]:
        r"""
        Computes and returns the statistical properties of the distribution, including,

        #. mean,
        #. median,
        #. mode, and
        #. variance

        If any of the parameter is not computable for a distribution, this method returns None.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing statistical properties such as mean, variance, etc.
        """
        raise NotImplementedError("Subclasses should implement this method.")
