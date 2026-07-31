"""Created on Feb 02 03:46:43 2025"""

from ..backend import BaseDistribution
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
    Class for :class:`~.ScaledInverseChiSquareDistribution`.

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

    Examples
    --------
    Importing libraries:

    .. literalinclude:: ../../../examples/basic/gaussian.py
       :language: python
       :linenos:
       :lineno-start: 3
       :lines: 3-7

    Generating a standard :class:`~.ScaledInverseChiSquaredDistribution` (:math:`\nu=1, \tau^2=1, \mu=0`)
     with ``pyMultiFit`` and ``scipy`` (where :math:`\nu=1` yields the inverse gamma parameterization :math:`a=\frac{\nu}{2}, \text{scale}=\frac{\nu \tau^2}{2}`):

    .. literalinclude:: ../../../examples/basic/scaled_inv_chi2.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/scaled_inv_chi2.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/scaled_inv_chi2_example1.png
       :alt: ScaledInvChi2(1, 1, 0)
       :align: center

    Generating a scaled and translated :class:`~.ScaledInverseChiSquaredDistribution` with Gaussian variance hyperparameters (:math:`\nu=5, \tau^2=2.5, \mu=-3`):

    .. literalinclude:: ../../../examples/basic/scaled_inv_chi2.py
       :language: python
       :lineno-start: 32
       :lines: 32

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/scaled_inv_chi2.py
       :language: python
       :lineno-start: 34
       :lines: 34-49

    .. image:: ../../../images/scaled_inv_chi2_example2.png
       :alt: ScaledInvChi2(5, 2.5, -3)
       :align: center
    """

    def __init__(
        self, amplitude: float = 1.0, df: float = 1.0, scale: float = 1.0, loc: float = 0.0, normalize: bool = False
    ):
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
        Instantiate :class:`~.ScaledInverseChiSquareDistribution` with scipy parametrization.

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
        :class:`~.ScaledInverseChiSquareDistribution`
            An instance of normalized :class:`~.ScaledInverseChiSquareDistribution`.
        """
        return cls(df=a, loc=loc, scale=scale, normalize=True)

    @classmethod
    def from_scipy_params(cls, a: float, loc: float = 0.0, scale=1.0):
        """
        Instantiate :class:`~.ScaledInverseChiSquareDistribution` with scipy parametrization.

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
        :class:`~.ScaledInverseChiSquareDistribution`
            An instance of normalized :class:`~.ScaledInverseChiSquareDistribution`.
        """
        return cls(df=a, loc=loc, scale=scale, normalize=True)

    def pdf(self, x: ArrayLike) -> NDArray:
        return scaled_inv_chi_square_pdf_(
            x, amplitude=self.amplitude, df=self.df, scale=self.scale, loc=self.loc, normalize=self.norm
        )

    def logpdf(self, x: ArrayLike) -> NDArray:
        return scaled_inv_chi_square_log_pdf_(
            x, amplitude=self.amplitude, df=self.df, scale=self.scale, loc=self.loc, normalize=self.norm
        )

    def cdf(self, x: ArrayLike) -> NDArray:
        return scaled_inv_chi_square_cdf_(
            x, amplitude=self.amplitude, df=self.df, scale=self.scale, loc=self.loc, normalize=self.norm
        )

    def logcdf(self, x: ArrayLike) -> NDArray:
        return scaled_inv_chi_square_log_cdf_(
            x, amplitude=self.amplitude, df=self.df, loc=self.loc, scale=self.scale, normalize=self.norm
        )

    def stats(self) -> dict[str, float]:
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