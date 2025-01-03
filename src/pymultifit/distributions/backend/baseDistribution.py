"""Created on Aug 03 22:06:29 2024"""

from typing import Any, Dict

import numpy as np


class BaseDistribution:
    r"""
    Bare-bones class for statistical distributions to provide consistent methods.

    This class serves as a template for other distribution classes, defining the common interface
    for probability density function (PDF), cumulative distribution function (CDF), and statistics.
    """

    def cdf(self, x: np.array) -> np.array:
        """
        Compute the cumulative density function (CDF) for the distribution.

        :param x: input values at which to evaluate the CDF.
        :type x: np.array

        :raise NotImplementedError:
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def pdf(self, x: np.array) -> np.array:
        r"""
        Compute the probability density function (PDF) for the distribution.

        :param x: input values at which to evaluate the PDF.
        :type x: np.array

        :raise NotImplementedError:
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

        :returns: A dictionary containing statistical properties such as mean, variance, etc.
        :rtype: Dict[str, Any]

        :raise BaseDistributionError: If the statistical properties are not computed.
        """
        raise NotImplementedError("Subclasses should implement this method.")
