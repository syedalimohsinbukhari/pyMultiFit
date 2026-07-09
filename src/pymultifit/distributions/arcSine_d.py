"""Created on Aug 14 02:02:42 2024"""

from __future__ import annotations

from numpy import sqrt

from .. import NAN_DICT, _md_scipy_like
from ..typing import ArrayLike, NDArray
from .backend import BaseDistribution
from .backend import errorHandling as erH
from .utilities_d import arc_sine_cdf_, arc_sine_log_cdf_, arc_sine_log_pdf_, arc_sine_pdf_


class ArcSineDistribution(BaseDistribution):
    r"""
    Class for ArcSine distribution.

    .. note::
        The :class:`ArcSineDistribution` is a special case of :class:`~pymultifit.distributions.beta_d.BetaDistribution`,

        * :math:`\alpha_\text{beta} = 0.5`,
        * :math:`\lambda_\text{beta} = 0.5`.

    Parameters
    ----------
    amplitude :
        The amplitude of the PDF. Defaults to 1.0. Ignored if **normalize** is ``True``.
    loc :
        The location parameter, :math:`-` shifting. Defaults to 0.0.
    scale :
        The scale parameter, for shifting. Defaults to 1.0.
    normalize :
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Examples
    --------
    Importing libraries

    .. literalinclude:: ../../../examples/basic/arcSine.py
       :language: python
       :linenos:
       :lineno-start: 3
       :lines: 3-7

    Generating the ArcSine distribution with ``pyMultiFit`` and ``scipy``.

    .. literalinclude:: ../../../examples/basic/arcSine.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**

    .. literalinclude:: ../../../examples/basic/arcSine.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-27

    .. image:: ../../../images/arcsine_example.png
       :alt: ArcSine distribution
       :align: center
    """

    def __init__(self, amplitude: float = 1.0, loc: float = 0.0, scale: float = 1.0, normalize: bool = False):
        self.amplitude = 1 if normalize else amplitude
        self.loc = loc
        self.scale = scale

        self.norm = normalize

    @classmethod
    @_md_scipy_like("1.0.7")
    def scipy_like(cls, loc: float = 0.0, scale: float = 1.0) -> "ArcSineDistribution":
        """
        Instantiate `ArcSineDistribution` with scipy parameterization.

        Parameters
        ----------
        loc :
            The location parameter. Defaults to 0.0.
        scale :
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        ArcSineDistribution
            An instance of normalized ArcSineDistribution.
        """
        return cls(loc=loc, scale=scale, normalize=True)

    """Created on Aug 14 02:02:42 2024"""

    from __future__ import annotations

    from numpy import sqrt

    from .. import NAN_DICT, _md_scipy_like
    from ..typing import ArrayLike, NDArray
    from .backend import BaseDistribution
    from .backend import errorHandling as erH
    from .utilities_d import arc_sine_cdf_, arc_sine_log_cdf_, arc_sine_log_pdf_, arc_sine_pdf_

    class ArcSineDistribution(BaseDistribution):
        r"""
        Class for :class:`~.ArcSineDistribution`.

        .. note::
            The :class:`~.ArcSineDistribution` is a special case of :class:`~pymultifit.distributions.beta_d.BetaDistribution`,

            * :math:`\alpha_\text{beta} = 0.5`,
            * :math:`\lambda_\text{beta} = 0.5`.

        Parameters
        ----------
        amplitude :
            The amplitude of the PDF. Defaults to 1.0. Ignored if **normalize** is ``True``.
        loc :
            The location parameter, :math:`-` shifting. Defaults to 0.0.
        scale :
            The scale parameter, for shifting. Defaults to 1.0.
        normalize :
            If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
            Defaults to ``False``.

        Examples
        --------
        Importing libraries

        .. literalinclude:: ../../../examples/basic/arcSine.py
           :language: python
           :linenos:
           :lineno-start: 3
           :lines: 3-7

        Generating the :class:`~.ArcSineDistribution` with ``pyMultiFit`` and ``scipy``.

        .. literalinclude:: ../../../examples/basic/arcSine.py
           :language: python
           :linenos:
           :lineno-start: 9
           :lines: 9-12

        Plotting **PDF** and **CDF**

        .. literalinclude:: ../../../examples/basic/arcSine.py
           :language: python
           :linenos:
           :lineno-start: 14
           :lines: 14-27

        .. image:: ../../../images/arcsine_example.png
           :alt: ArcSine distribution
           :align: center
        """

        def __init__(self, amplitude: float = 1.0, loc: float = 0.0, scale: float = 1.0, normalize: bool = False):
            self.amplitude = 1 if normalize else amplitude
            self.loc = loc
            self.scale = scale

            self.norm = normalize

        @classmethod
        @_md_scipy_like("1.0.7")
        def scipy_like(cls, loc: float = 0.0, scale: float = 1.0) -> "ArcSineDistribution":
            """
            Instantiate :class:`~.ArcSineDistribution` with scipy parameterization.

            Parameters
            ----------
            loc :
                The location parameter. Defaults to 0.0.
            scale :
                The scale parameter. Defaults to 1.0.

            Returns
            -------
            :class:`~.ArcSineDistribution`
                An instance of normalized :class:`~.ArcSineDistribution`.
            """
            return cls(loc=loc, scale=scale, normalize=True)

        @classmethod
        def from_scipy_params(cls, loc: float = 0.0, scale: float = 1.0) -> "ArcSineDistribution":
            """
            Instantiate :class:`~.ArcSineDistribution` with scipy parameterization.

            Parameters
            ----------
            loc :
                The location parameter. Defaults to 0.0.
            scale :
                The scale parameter. Defaults to 1.0.

            Returns
            -------
            :class:`~.ArcSineDistribution`
                An instance of normalized :class:`~.ArcSineDistribution`.
            """
            return cls(loc=loc, scale=scale, normalize=True)

        def pdf(self, x: ArrayLike) -> NDArray:
            return arc_sine_pdf_(x, amplitude=self.amplitude, loc=self.loc, scale=self.scale, normalize=self.norm)

        def logpdf(self, x: ArrayLike) -> NDArray:
            return arc_sine_log_pdf_(x, amplitude=self.amplitude, loc=self.loc, scale=self.scale, normalize=self.norm)

        def cdf(self, x: ArrayLike) -> NDArray:
            return arc_sine_cdf_(x, amplitude=self.amplitude, loc=self.loc, scale=self.scale, normalize=self.norm)

        def logcdf(self, x: ArrayLike) -> NDArray:
            return arc_sine_log_cdf_(x, amplitude=self.amplitude, loc=self.loc, scale=self.scale, normalize=self.norm)

        def stats(self) -> dict[str, float]:
            s_, l_ = self.scale, self.loc

            if s_ <= 0:
                return NAN_DICT

            mean_ = (s_ * 0.5) + l_
            median_ = (s_ * 0.5) + l_
            variance_ = (1 / 8) * s_ ** 2

            return {"mean": mean_, "median": median_, "variance": variance_, "std": sqrt(variance_)}
    @classmethod
    def from_scipy_params(cls, loc: float = 0.0, scale: float = 1.0) -> "ArcSineDistribution":
        """
        Instantiate `ArcSineDistribution` with scipy parameterization.

        Parameters
        ----------
        loc :
            The location parameter. Defaults to 0.0.
        scale :
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        ArcSineDistribution
            An instance of normalized ArcSineDistribution.
        """
        return cls(loc=loc, scale=scale, normalize=True)

    def pdf(self, x: ArrayLike) -> NDArray:
        return arc_sine_pdf_(x, amplitude=self.amplitude, loc=self.loc, scale=self.scale, normalize=self.norm)

    def logpdf(self, x: ArrayLike) -> NDArray:
        return arc_sine_log_pdf_(x, amplitude=self.amplitude, loc=self.loc, scale=self.scale, normalize=self.norm)

    def cdf(self, x: ArrayLike) -> NDArray:
        return arc_sine_cdf_(x, amplitude=self.amplitude, loc=self.loc, scale=self.scale, normalize=self.norm)

    def logcdf(self, x: ArrayLike) -> NDArray:
        return arc_sine_log_cdf_(x, amplitude=self.amplitude, loc=self.loc, scale=self.scale, normalize=self.norm)

    def stats(self) -> dict[str, float]:
        s_, l_ = self.scale, self.loc

        if s_ <= 0:
            return NAN_DICT

        mean_ = (s_ * 0.5) + l_
        median_ = (s_ * 0.5) + l_
        variance_ = (1 / 8) * s_**2

        return {"mean": mean_, "median": median_, "variance": variance_, "std": sqrt(variance_)}
