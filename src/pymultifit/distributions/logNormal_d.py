"""Created on Aug 03 21:02:45 2024"""

from __future__ import annotations

from .. import EXP, LOG, SQRT, md_scipy_like, suppress_numpy_warnings
from ..typing import ArrayLike, NDArray
from .backend import BaseDistribution
from .backend import errorHandling as erH
from .utilities_d import log_normal_cdf_, log_normal_log_cdf_, log_normal_log_pdf_, log_normal_pdf_


class LogNormalDistribution(BaseDistribution):
    r"""
    Class for LogNormal distribution.

    Parameters
    ----------
    amplitude
        The amplitude of the PDF. Defaults to 1.0. Ignored if **normalize** is ``True``.
    mu
        The mean parameter, :math:`\mu`. Defaults to 0.0.
    std
        The standard deviation parameter, :math:`\sigma`. Defaults to 1.0.
    normalize
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Raises
    ------
    NegativeAmplitudeError
        If the provided value of amplitude is negative.

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
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()

        self.amplitude = 1.0 if normalize else amplitude
        self.mu = LOG(mu)
        self.std = std
        self.loc = loc

        self.norm = normalize

    @classmethod
    @md_scipy_like("1.0.7")
    def scipy_like(cls, s: float, loc: float = 0.0, scale: float = 1.0) -> "LogNormalDistribution":
        """
        Instantiate LogNormalDistribution with scipy parametrization.

        Parameters
        ----------
        s
            The shape parameter.
        loc
            The location parameter. Defaults to 0.0.
        scale
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        LogNormalDistribution
            An instance of normalized LogNormalDistribution.
        """
        return cls(std=s, mu=scale, loc=loc, normalize=True)

    @classmethod
    def from_scipy_params(cls, s: float, loc: float = 0.0, scale: float = 1.0) -> "LogNormalDistribution":
        """
        Instantiate LogNormalDistribution with scipy parametrization.

        Parameters
        ----------
        s
            The shape parameter.
        loc
            The location parameter. Defaults to 0.0.
        scale
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        LogNormalDistribution
            An instance of normalized LogNormalDistribution.
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
        m, s, l_ = EXP(self.mu), self.std, self.loc

        # copied from scipy source-code,
        # simpler implementations give reasonable higher values > 10^100 but scipy gives np.inf,
        # so I'm shortcutting it by taking scipy implementation here directly.
        p = EXP(s * s)
        mean_ = SQRT(p)
        variance_ = p * (p - 1)
        variance_ *= m**2

        return {"mean": (m * mean_) + l_, "variance": variance_, "std": SQRT(variance_)}
