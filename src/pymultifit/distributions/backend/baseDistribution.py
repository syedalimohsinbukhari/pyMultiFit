"""Created on Aug 03 22:06:29 2024"""

from typing import Dict

from ... import fArray


class BaseDistribution:
    r"""
    Bare-bones class for statistical distributions to provide consistent methods.

    This class serves as a template for other distribution classes, defining the common interface
    for probability density function (PDF), cumulative distribution function (CDF), and statistics.
    """

    def pdf(self, x: fArray) -> fArray:
        r"""
        Compute the probability density function (PDF) for the distribution.

        :param x: Input array at which to evaluate the PDF.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def logpdf(self, x: fArray) -> fArray:
        r"""
        Compute the log probability density function (logPDF) for the distribution.

        :param x: Input array at which to evaluate the logPDF.
        """
        pass

    def cdf(self, x: fArray) -> fArray:
        """
        Compute the cumulative density function (CDF) for the distribution.

        :param x: Input array at which to evaluate the CDF.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def logcdf(self, x: fArray) -> fArray:
        r"""
        Compute the log cumulative density function (logCDF) for the distribution.

        :param x: Input array at which to evaluate the logCDF.
        """
        pass

    def stats(self) -> Dict[str, float]:
        r"""
        Computes and returns the statistical properties of the distribution, including,

        #. mean,
        #. median,
        #. variance, and
        #. standard deviation.

        :returns: A dictionary containing statistical properties such as mean, variance, etc.
        :rtype: Dict[str, float]

        Notes
        -----
        If any of the parameter is not computable for a distribution, this method returns None.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @property
    def mean(self):
        """The mean of the distribution."""
        return self.stats().get('mean', None)

    @property
    def median(self):
        """The median of the distribution."""
        return self.stats().get('median', None)

    @property
    def mode(self):
        """The mode of the distribution."""
        return self.stats().get('mode', None)

    @property
    def variance(self):
        """The variance of the distribution."""
        return self.stats().get('variance', None)

    @property
    def stddev(self):
        """The standard deviation of the distribution."""
        return self.stats().get('std', None)
