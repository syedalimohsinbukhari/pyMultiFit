"""Created on Aug 03 22:06:29 2024"""

from __future__ import annotations

from ...typing import ArrayLike, NDArray


class BaseDistribution:
    """Bare-bones class for statistical distributions to provide consistent methods."""

    def pdf(self, x: ArrayLike) -> NDArray:  # type: ignore[empty-body]
        """
        Compute the probability density function (PDF) for the distribution.

        Parameters
        ----------
        x :
            Input array at which to evaluate the PDF.

        Returns
        -------
        NDArray
            The probability density function evaluated at the input array.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def logpdf(self, x: ArrayLike) -> NDArray:  # type: ignore[empty-body]
        """
        Compute the log probability density function (logPDF) for the distribution.

        Parameters
        ----------
        x :
            Input array at which to evaluate the logPDF.

        Returns
        -------
        NDArray
            The log probability density function evaluated at the input array.
        """

    def cdf(self, x: ArrayLike) -> NDArray:  # type: ignore[empty-body]
        """
        Compute the cumulative density function (CDF) for the distribution.

        Parameters
        ----------
        x :
            Input array at which to evaluate the CDF.

        Returns
        -------
        NDArray
            The cumulative density function evaluated at the input array.
        """

    def logcdf(self, x: ArrayLike) -> NDArray:  # type: ignore[empty-body]
        """
        Compute the log cumulative density function (logCDF) for the distribution.

        Parameters
        ----------
        x :
            Input array at which to evaluate the logCDF.

        Returns
        -------
        NDArray
            The log cumulative density function evaluated at the input array.
        """

    def stats(self) -> dict[str, float]:  # type: ignore[empty-body]
        """
        Computes and returns the statistical properties of the distribution, including:

        #. mean,
        #. median,
        #. variance, and
        #. standard deviation.

        Returns
        -------
        dict
            A dictionary containing statistical properties such as mean, variance, etc.

        Notes
        -----
            If any of the parameters is not computable for a distribution, this method returns ``None``.
        """

    def _get_stats(self, key: str) -> float | None:
        stats = self.stats()
        return stats.get(key) if stats else None

    @property
    def mean(self) -> float | None:
        """
        The mean of the distribution.

        Returns
        -------
        float
            The mean of the distribution, or ``None`` if it cannot be computed.
        """
        return self._get_stats("mean")

    @property
    def median(self) -> float | None:
        """
        The median of the distribution.

        Returns
        -------
        float
            The median of the distribution, or ``None`` if it cannot be computed.
        """
        return self._get_stats("median")

    @property
    def mode(self) -> float | None:
        """
        The mode of the distribution.

        Returns
        -------
        float
            The mode of the distribution, or ``None`` if it cannot be computed.
        """
        return self._get_stats("mode")

    @property
    def variance(self) -> float | None:
        """
        The variance of the distribution.

        Returns
        -------
        float
            The variance of the distribution, or ``None`` if it cannot be computed.
        """
        return self._get_stats("variance")

    @property
    def stddev(self) -> float | None:
        """
        The standard deviation of the distribution.

        Returns
        -------
        float
            The standard deviation of the distribution, or ``None`` if it cannot be computed.
        """
        return self._get_stats("std")
