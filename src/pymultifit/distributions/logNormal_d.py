"""Created on Aug 03 21:02:45 2024"""

from __future__ import annotations

from .. import EXP, NAN_DICT, SQRT, _md_scipy_like, suppress_numpy_warnings
from ..typing import ArrayLike, NDArray
from .backend import BaseDistribution
from .utilities_d import log_normal_cdf_, log_normal_log_cdf_, log_normal_log_pdf_, log_normal_pdf_


class LogNormalDistribution(BaseDistribution):
    r"""
    Class for :class:`~.LogNormalDistribution`.

    Parameters
    ----------
    amplitude :
        The amplitude of the PDF. Defaults to 1.0. Ignored if ``normalize`` is ``True``.
    mu :
        The mean parameter, :math:`\mu`. Defaults to 1.0.
    std :
        The standard deviation parameter, :math:`\sigma`. Defaults to 1.0.
    loc :
        The location parameter, for shifting. Defaults to 0.0.
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

    Generating a standard Gaussian(:math:`\mu=0, \sigma = 1`) distribution with ``pyMultiFit`` and ``scipy``:

    .. literalinclude:: ../../../examples/basic/gaussian.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/gaussian.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/gaussian_example1.png
       :alt: Gaussian(0, 1)
       :align: center

    Generating a translated Gaussian(:math:`\mu=3, \sigma=2`) distribution:

    .. literalinclude:: ../../../examples/basic/gaussian.py
       :language: python
       :lineno-start: 32
       :lines: 32

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/gaussian.py
       :language: python
       :lineno-start: 34
       :lines: 34-49

    .. image:: ../../../images/gaussian_example2.png
       :alt: Gaussian(3, 2)
       :align: center
    """

    def __init__(
        self, amplitude: float = 1.0, mu: float = 1.0, std: float = 1.0, loc: float = 0.0, normalize: bool = False
    ):
        self.amplitude = 1.0 if normalize else amplitude
        self.mu = mu
        self.std = std
        self.loc = loc

        self.norm = normalize

    @classmethod
    @_md_scipy_like("1.0.7")
    def scipy_like(cls, s: float, loc: float = 0.0, scale: float = 1.0) -> "LogNormalDistribution":
        r"""
        Instantiate :class:`~.LogNormalDistribution` with ``scipy`` parameterization.

        Parameters
        ----------
        s :
            The shape parameter.
        loc :
            The location parameter. Defaults to 0.0.
        scale :
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        :class:`~.LogNormalDistribution`
            An instance of normalized :class:`~.LogNormalDistribution`.
        """
        return cls(std=s, mu=scale, loc=loc, normalize=True)

    @classmethod
    def from_scipy_params(cls, s: float, loc: float = 0.0, scale: float = 1.0) -> "LogNormalDistribution":
        r"""
        Instantiate :class:`~.LogNormalDistribution` with ``scipy`` parameterization.

        Parameters
        ----------
        s :
            The shape parameter.
        loc :
            The location parameter. Defaults to 0.0.
        scale :
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        :class:`~.LogNormalDistribution`
            An instance of normalized :class:`~.LogNormalDistribution`.
        """
        return cls(std=s, mu=scale, loc=loc, normalize=True)

    def pdf(self, x: ArrayLike) -> NDArray:
        return log_normal_pdf_(
            x, amplitude=self.amplitude, mean=self.mu, std=self.std, loc=self.loc, normalize=self.norm
        )

    def logpdf(self, x: ArrayLike) -> NDArray:
        return log_normal_log_pdf_(
            x, amplitude=self.amplitude, mean=self.mu, std=self.std, loc=self.loc, normalize=self.norm
        )

    def cdf(self, x: ArrayLike) -> NDArray:
        return log_normal_cdf_(
            x, amplitude=self.amplitude, mean=self.mu, std=self.std, loc=self.loc, normalize=self.norm
        )

    def logcdf(self, x: ArrayLike) -> NDArray:
        return log_normal_log_cdf_(
            x, amplitude=self.amplitude, mean=self.mu, std=self.std, loc=self.loc, normalize=self.norm
        )

    @suppress_numpy_warnings()
    def stats(self) -> dict[str, float]:
        m, s, l_ = self.mu, self.std, self.loc

        if m <= 0 or s <= 0:
            return NAN_DICT

        # copied from scipy source-code,
        # simpler implementations give reasonable higher values > 10^100 but scipy gives np.inf,
        # so I'm shortcutting it by taking scipy implementation here directly.
        p = EXP(s * s)
        mean_ = SQRT(p)
        variance_ = p * (p - 1)
        variance_ *= m**2

        return {"mean": (m * mean_) + l_, "variance": variance_, "std": SQRT(variance_)}
