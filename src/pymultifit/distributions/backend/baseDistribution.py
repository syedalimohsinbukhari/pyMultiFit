"""Created on Aug 03 22:06:29 2024"""

from typing import Dict, Optional

from ... import OneDArray


class BaseDistribution:
    r"""
    Bare-bones class for statistical distributions to provide consistent methods.

    This class serves as a template for other distribution classes, defining the common interface
    for probability density function (PDF), cumulative distribution function (CDF), and statistics.
    """

    def pdf(self, x: OneDArray) -> OneDArray:  # type: ignore[empty-body]
        r"""
        Compute the probability density function (PDF) for the distribution.

        :param x: Input array at which to evaluate the PDF.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def logpdf(self, x: OneDArray) -> OneDArray:  # type: ignore[empty-body]
        r"""
        Compute the log probability density function (logPDF) for the distribution.

        :param x: Input array at which to evaluate the logPDF.
        """

    def cdf(self, x: OneDArray) -> OneDArray:  # type: ignore[empty-body]
        """
        Compute the cumulative density function (CDF) for the distribution.

        :param x: Input array at which to evaluate the CDF.
        """

    def logcdf(self, x: OneDArray) -> OneDArray:  # type: ignore[empty-body]
        r"""
        Compute the log cumulative density function (logCDF) for the distribution.

        :param x: Input array at which to evaluate the logCDF.
        """

    def stats(self) -> Dict[str, float]:  # type: ignore[empty-body]
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
        If any of the parameters is not computable for a distribution, this method returns None.
        """

    def _get_stats(self, key: str) -> Optional[float]:
        stats = self.stats()
        return stats.get(key) if stats else None

    @property
    def mean(self) -> Optional[float]:
        """The mean of the distribution."""
        return self._get_stats("mean")

    @property
    def median(self) -> Optional[float]:
        """The median of the distribution."""
        return self._get_stats("median")

    @property
    def mode(self) -> Optional[float]:
        """The mode of the distribution."""
        return self._get_stats("mode")

    @property
    def variance(self) -> Optional[float]:
        """The variance of the distribution."""
        return self._get_stats("variance")

    @property
    def stddev(self) -> Optional[float]:
        """The standard deviation of the distribution."""
        return self._get_stats("std")
