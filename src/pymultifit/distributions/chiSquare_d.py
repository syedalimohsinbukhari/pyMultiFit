"""Created on Dec 03 17:37:05 2024"""

from __future__ import annotations

from .. import NAN_DICT, SQRT, _md_scipy_like
from ..typing import ArrayLike, NDArray
from .backend import BaseDistribution
from .utilities_d import chi_square_cdf_, chi_square_log_cdf_, chi_square_log_pdf_, chi_square_pdf_


class ChiSquareDistribution(BaseDistribution):
    r"""
    Class for :class:`~.ChiSquareDistribution`.

    .. note::
        The :class:`~.ChiSquareDistribution` is a special case of the :class:`~pymultifit.distributions.gamma_d.GammaDistribution`,

        * :math:`\alpha\ (\text{shape}) = \text{dof} / 2`,
        * :math:`\theta\ (\text{scale}) = 2`.

    Parameters
    ----------
    amplitude :
        The amplitude of the PDF. Defaults to 1.0. Ignored if **normalize** is ``True``.
    degree_of_freedom :
        The degree of freedom for the chi-square distribution. Default is 1.0.
    loc :
        The location parameter, for shifting. Defaults to 0.0.
    normalize :
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Examples
    --------
    Importing libraries

    .. literalinclude:: ../../../examples/basic/chisquare.py
       :language: python
       :linenos:
       :lineno-start: 3
       :lines: 3-7

    Generating a standard :math:`\chi^2(1)` distribution with ``pyMultiFit`` and ``scipy``.

    .. literalinclude:: ../../../examples/basic/chisquare.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**

    .. literalinclude:: ../../../examples/basic/chisquare.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/chi2_example1.png
       :alt: Chi-Square distribution (df=1)
       :align: center

    Generating a translated :math:`\chi^2(1)` distribution with :math:`\text{loc} = 3`.

    .. literalinclude:: ../../../examples/basic/chisquare.py
       :language: python
       :lineno-start: 32
       :lines: 32

    Plotting **PDF** and **CDF**

    .. literalinclude:: ../../../examples/basic/chisquare.py
       :language: python
       :lineno-start: 34
       :lines: 34-49

    .. image:: ../../../images/chi2_example2.png
       :alt: Chi-Square distribution (shifted and translated)
       :align: center
    """

    def __init__(
        self,
        amplitude: float = 1.0,
        degree_of_freedom: int | float = 1,
        loc: float = 0.0,
        scale: float = 1.0,
        normalize: bool = False,
    ):
        self.amplitude = 1 if normalize else amplitude
        self.dof = degree_of_freedom
        self.loc = loc
        self.scale = scale

        self.norm = normalize

    @classmethod
    @_md_scipy_like("v1.0.7")
    def scipy_like(cls, df: int | float, loc: float = 0.0, scale: float = 1.0) -> "ChiSquareDistribution":
        """
        Instantiate :class:`~.ChiSquareDistribution` with ``scipy`` parameterization.

        Parameters
        ----------
        df :
            The degree of freedom for the :class:`~.ChiSquareDistribution`.
        loc :
            The location parameter. Defaults to 0.0.
        scale :
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        :class:`~.ChiSquareDistribution`
            An instance of normalized :class:`~.ChiSquareDistribution`.
        """
        return cls(degree_of_freedom=df, loc=loc, scale=scale, normalize=True)

    @classmethod
    def from_scipy_params(cls, df: int | float, loc: float = 0.0, scale: float = 1.0) -> "ChiSquareDistribution":
        """
        Instantiate :class:`~.ChiSquareDistribution` with ``scipy`` parameterization.

        Parameters
        ----------
        df :
            The degree of freedom for the :class:`~.ChiSquareDistribution`.
        loc :
            The location parameter. Defaults to 0.0.
        scale :
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        :class:`~.ChiSquareDistribution`
            An instance of normalized :class:`~.ChiSquareDistribution`.
        """
        return cls(degree_of_freedom=df, loc=loc, scale=scale, normalize=True)

    def pdf(self, x: ArrayLike) -> NDArray:
        return chi_square_pdf_(
            x, amplitude=self.amplitude, degree_of_freedom=self.dof, loc=self.loc, scale=self.scale, normalize=self.norm
        )

    def logpdf(self, x: ArrayLike) -> NDArray:
        return chi_square_log_pdf_(
            x, amplitude=self.amplitude, degree_of_freedom=self.dof, loc=self.loc, scale=self.scale, normalize=self.norm
        )

    def cdf(self, x: ArrayLike) -> NDArray:
        return chi_square_cdf_(
            x, amplitude=self.amplitude, degree_of_freedom=self.dof, loc=self.loc, scale=self.scale, normalize=self.norm
        )

    def logcdf(self, x: ArrayLike) -> NDArray:
        return chi_square_log_cdf_(
            x, amplitude=self.amplitude, degree_of_freedom=self.dof, loc=self.loc, scale=self.scale, normalize=self.norm
        )

    def stats(self) -> dict[str, float]:
        df = self.dof
        s, l_ = self.scale, self.loc

        if any(param <= 0 for param in (df, s)):
            return NAN_DICT

        mean_ = (s * df) + l_
        mode_ = max(df - 2, 0)
        variance_ = 2 * df * s**2

        return {"mean": mean_, "mode": mode_, "variance": variance_, "std": SQRT(variance_)}
