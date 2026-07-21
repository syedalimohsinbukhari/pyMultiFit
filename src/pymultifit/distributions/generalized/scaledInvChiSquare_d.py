"""Created on Feb 02 03:46:43 2025"""

from ..backend import BaseDistribution, errorHandling as erH
from ..utilities_d import (
    scaled_inv_chi_square_cdf_,
    scaled_inv_chi_square_log_cdf_,
    scaled_inv_chi_square_log_pdf_,
    scaled_inv_chi_square_pdf_,
)
from ... import _md_scipy_like, SQRT, INF, NAN_DICT
from ...typing import ArrayLike, NDArray


class ScaledInverseChiSquareDistribution(BaseDistribution):
    r"""
    Class for ScaledInverseChiSquareDistribution.

    Parameters
    ----------
    amplitude :
        The amplitude of the PDF. Defaults to 1.0. Ignored if **normalize** is ``True``.
    df :
        Degrees of freedom parameter, :math:`\nu`. Defaults to 1.0.
    scale :
        Scale parameter, :math:`s^2`. Defaults to 1.0.
    loc :
        Location/translation parameter, :math:`\mu`. Defaults to 0.0.
    normalize :
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Raises
    ------
    NegativeAmplitudeError
        If the provided value of amplitude is negative and **normalize** is ``False``.
    """

    def __init__(
        self, amplitude: float = 1.0, df: float = 1.0, scale: float = 1.0, loc: float = 0.0, normalize: bool = False
    ):
        if not normalize and amplitude < 0:
            raise erH.NegativeAmplitudeError()

        self.amplitude = 1 if normalize else amplitude
        self.df = df
        self.scale = scale
        self.tau2 = scale / df

        self.loc = loc
        self.norm = normalize

    @classmethod
    @_md_scipy_like("1.0.7")
    def scipy_like(cls, a: float, loc: float = 0.0, scale=1.0):
        """
        Instantiate ScaledInverseChiSquareDistribution with scipy parametrization.

        Parameters
        ----------
        a :
            The degrees of freedom parameter.
        loc :
            The location parameter. Defaults to 0.0.
        scale :
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        ScaledInverseChiSquareDistribution
            An instance of normalized ScaledInverseChiSquareDistribution.
        """
        return cls(df=a, loc=loc, scale=scale, normalize=True)

    @classmethod
    def from_scipy_params(cls, a: float, loc: float = 0.0, scale=1.0):
        """
        Instantiate ScaledInverseChiSquareDistribution with scipy parametrization.

        Parameters
        ----------
        a :
            The degrees of freedom parameter.
        loc :
            The location parameter. Defaults to 0.0.
        scale :
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        ScaledInverseChiSquareDistribution
            An instance of normalized ScaledInverseChiSquareDistribution.
        """
        return cls(df=a, loc=loc, scale=scale, normalize=True)

    def pdf(self, x: ArrayLike) -> NDArray:
        """
        Probability density function evaluated at x.

        Parameters
        ----------
        x :
            Quantiles where the PDF is evaluated.

        Returns
        -------
        NDArray
            Probability density function values evaluated at x.
        """
        return scaled_inv_chi_square_pdf_(
            x, amplitude=self.amplitude, df=self.df, scale=self.scale, loc=self.loc, normalize=self.norm
        )

    def logpdf(self, x: ArrayLike) -> NDArray:
        """
        Log of the probability density function evaluated at x.

        Parameters
        ----------
        x :
            Quantiles where the log-PDF is evaluated.

        Returns
        -------
        NDArray
            Logarithm of the probability density function values evaluated at x.
        """
        return scaled_inv_chi_square_log_pdf_(
            x, amplitude=self.amplitude, df=self.df, scale=self.scale, loc=self.loc, normalize=self.norm
        )

    def cdf(self, x: ArrayLike) -> NDArray:
        """
        Cumulative distribution function evaluated at x.

        Parameters
        ----------
        x :
            Quantiles where the CDF is evaluated.

        Returns
        -------
        NDArray
            Cumulative distribution function values evaluated at x.
        """
        return scaled_inv_chi_square_cdf_(
            x, amplitude=self.amplitude, df=self.df, scale=self.scale, loc=self.loc, normalize=self.norm
        )

    def logcdf(self, x: ArrayLike) -> NDArray:
        """
        Log of the cumulative distribution function evaluated at x.

        Parameters
        ----------
        x :
            Quantiles where the log-CDF is evaluated.

        Returns
        -------
        NDArray
            Logarithm of the cumulative distribution function values evaluated at x.
        """
        return scaled_inv_chi_square_log_cdf_(
            x, amplitude=self.amplitude, df=self.df, loc=self.loc, scale=self.scale, normalize=self.norm
        )

    def stats(self) -> dict[str, float]:
        r"""
        Compute descriptive summary statistics for the distribution.

        Returns
        -------
        dict
            A dictionary containing the calculated statistics.
            Key mappings:

            - "mean": Expected value (returns infinity if :math:`\nu \le 2`).
            - "mode": Mode value.
            - "variance": Population variance (returns infinity if :math:`\nu \le 4`).
            - "std": Standard deviation (returns infinity if :math:`\nu \le 4`).
        """
        v, tau2, loc = self.df, self.tau2, self.loc

        if any(param <= 0 for param in (v, tau2)):
            return NAN_DICT

        mean_ = (v * tau2) / (v - 2)
        mode_ = (v * tau2) / (v + 2)
        variance_ = (2 * v**2 * tau2**2) / ((v - 2) ** 2 * (v - 4))

        return {
            "mean": mean_ + loc if v > 2 else INF,
            "mode": mode_ + loc,
            "variance": variance_ if v > 4 else INF,
            "std": SQRT(variance_) if v > 4 else INF,
        }