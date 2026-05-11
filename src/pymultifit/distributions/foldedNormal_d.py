"""Created on Dec 04 03:42:42 2024"""

from __future__ import annotations

from scipy.special import erf

from .. import EXP, NAN_DICT, SQRT, SQRT_TWO, SQRT_TWO_BY_PI, md_scipy_like
from ..typing import ArrayLike, NDArray
from .backend import BaseDistribution
from .backend import errorHandling as erH
from .utilities_d import folded_normal_cdf_, folded_normal_log_cdf_, folded_normal_log_pdf_, folded_normal_pdf_


class FoldedNormalDistribution(BaseDistribution):
    r"""
    Class for FoldedNormal distribution.

    Parameters
    ----------
    amplitude
        The amplitude of the PDF. Defaults to 1.0. Ignored if **normalize** is ``True``.
    mu
        The mean parameter, :math:`\mu`. Defaults to 0.0.
    sigma
        The standard deviation parameter, :math:`\sigma`. Defaults to 1.0.
    loc
        The location parameter, for shifting. Defaults to 0.0.
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

    .. literalinclude:: ../../../examples/basic/foldednorm.py
       :language: python
       :linenos:
       :lineno-start: 3
       :lines: 3-7

    Generating a standard Folded Normal(:math:`\mu=0, \sigma = 1`) distribution with ``pyMultiFit`` and ``scipy``:

    .. literalinclude:: ../../../examples/basic/foldednorm.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/foldednorm.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/folded_normal_example1.png
       :alt: Gaussian(0, 1)
       :align: center

    Generating a translated Gaussian(:math:`\mu=2, \sigma=3`) distribution with :math:`\text{loc}=3`:

    .. literalinclude:: ../../../examples/basic/foldednorm.py
       :language: python
       :lineno-start: 32
       :lines: 32

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/foldednorm.py
       :language: python
       :lineno-start: 34
       :lines: 34-49

    .. image:: ../../../images/folded_normal_example2.png
       :alt: Gaussian(3, 2)
       :align: center
    """

    def __init__(
        self, amplitude: float = 1.0, mu: float = 0.0, sigma: float = 1.0, loc: float = 0.0, normalize: bool = False
    ):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()

        self.amplitude = 1.0 if normalize else amplitude
        self.mu = mu
        self.sigma = sigma
        self.loc = loc

        self.norm = normalize

    @classmethod
    @md_scipy_like("1.0.7")
    def scipy_like(cls, c: float, loc: float = 0.0, scale: float = 1.0) -> "FoldedNormalDistribution":
        r"""
        Instantiate FoldedNormalDistribution with scipy parametrization.

        Parameters
        ----------
        c
            The shape parameter.
        loc
            The location parameter. Defaults to 0.0.
        scale
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        FoldedNormalDistribution
            An instance of normalized FoldedNormalDistribution.
        """
        return cls(mu=c, sigma=scale, loc=loc, normalize=True)

    @classmethod
    def from_scipy_params(cls, c: float, loc: float = 0.0, scale: float = 1.0) -> "FoldedNormalDistribution":
        r"""
        Instantiate FoldedNormalDistribution with scipy parametrization.

        Parameters
        ----------
        c
            The shape parameter.
        loc
            The location parameter. Defaults to 0.0.
        scale
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        FoldedNormalDistribution
            An instance of normalized FoldedNormalDistribution.
        """
        return cls(mu=c, sigma=scale, loc=loc, normalize=True)

    def pdf(self, x: ArrayLike) -> NDArray:
        return folded_normal_pdf_(
            x, amplitude=self.amplitude, mean=self.mu, sigma=self.sigma, loc=self.loc, normalize=self.norm
        )

    def logpdf(self, x: ArrayLike) -> NDArray:
        return folded_normal_log_pdf_(
            x, amplitude=self.amplitude, mean=self.mu, sigma=self.sigma, loc=self.loc, normalize=self.norm
        )

    def cdf(self, x: ArrayLike) -> NDArray:
        return folded_normal_cdf_(
            x, amplitude=self.amplitude, mean=self.mu, sigma=self.sigma, loc=self.loc, normalize=self.norm
        )

    def logcdf(self, x: ArrayLike) -> NDArray:
        return folded_normal_log_cdf_(
            x, amplitude=self.amplitude, mean=self.mu, sigma=self.sigma, loc=self.loc, normalize=self.norm
        )

    def stats(self) -> dict[str, float]:
        mean_, std_ = self.mu, self.sigma

        if std_ <= 0:
            return NAN_DICT

        f1 = SQRT_TWO_BY_PI * EXP(-0.5 * mean_**2)
        f2 = mean_ * erf(mean_ / SQRT_TWO)

        mu_y = f1 + f2
        var_y = mean_**2 + 1 - mu_y**2

        return {"mean": (std_ * mu_y) + self.loc, "variance": var_y * std_**2, "std": SQRT(var_y * std_**2)}
