"""Created on Jan 29 15:42:23 2025"""

from scipy.special import gammaln

from ..backend import BaseDistribution, errorHandling as erH
from ..utilities_d import sym_gen_normal_cdf_, sym_gen_normal_pdf_
from ... import _md_scipy_like, LOG, EXP, SQRT
from ...typing import ArrayLike, NDArray


class SymmetricGeneralizedNormalDistribution(BaseDistribution):
    r"""
    Class for :class:`~.SymmetricGeneralizedNormalDistribution`.

    Parameters
    ----------
    amplitude :
        The amplitude of the PDF. Defaults to 1.0. Ignored if **normalize** is ``True``.
    shape :
        The shape parameter, :math:`\beta`. Defaults to 1.0.
    loc :
        The shape parameter, :math:`\mu`. Defaults to 0.0.
    scale :
        The standard deviation parameter, :math:`\alpha`. Defaults to 1.0.
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

    Generating a standard :class:`~.SymmetricGeneralizedNormalDistribution` (:math:`\beta=1, \mu=0, \alpha = 1`)
     with ``pyMultiFit`` and ``scipy``:

    .. literalinclude:: ../../../examples/basic/gennorm.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/gennorm.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/gen_norm_example1.png
       :alt: GenNorm(1, 0, 1)
       :align: center

    Generating a scaled and translated :class:`~.SymmetricGeneralizedNormalDistribution` (:math:`\beta=2, \mu=-3, \alpha=5`):

    .. literalinclude:: ../../../examples/basic/gennorm.py
       :language: python
       :lineno-start: 32
       :lines: 32

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/gennorm.py
       :language: python
       :lineno-start: 34
       :lines: 34-49

    .. image:: ../../../images/gen_norm_example2.png
       :alt: GenNorm(2, -3, 5)
       :align: center
    """

    def __init__(
        self, amplitude: float = 1.0, shape: float = 1.0, loc: float = 0.0, scale: float = 1.0, normalize: bool = False
    ):
        self.amplitude = 1.0 if normalize else amplitude
        self.loc = loc
        self.scale = scale
        self.shape = shape

        self.norm = normalize

    @classmethod
    @_md_scipy_like("1.0.7")
    def scipy_like(cls, beta: float, loc: float = 0.0, scale: float = 1.0):
        """
        Instantiate :class:`~.SymmetricGeneralizedNormalDistribution` with scipy parametrization.

        Parameters
        ----------
        beta :
            The shape parameter.
        loc :
            The mean parameter. Defaults to 0.0.
        scale :
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        :class:`~.SymmetricGeneralizedNormalDistribution`
            An instance of normalized :class:`~.SymmetricGeneralizedNormalDistribution`.
        """
        return cls(shape=beta, loc=loc, scale=scale, normalize=True)

    @classmethod
    def from_scipy_params(cls, beta, loc: float = 0.0, scale: float = 1.0):
        """
        Instantiate :class:`~.SymmetricGeneralizedNormalDistribution` with scipy parametrization.

        Parameters
        ----------
        beta :
            The shape parameter.
        loc :
            The mean parameter. Defaults to 0.0.
        scale :
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        :class:`~.SymmetricGeneralizedNormalDistribution`
            An instance of normalized :class:`~.SymmetricGeneralizedNormalDistribution`.
        """
        return cls(shape=beta, loc=loc, scale=scale, normalize=True)

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
        return sym_gen_normal_pdf_(
            x, amplitude=self.amplitude, shape=self.shape, loc=self.loc, scale=self.scale, normalize=self.norm
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
        return sym_gen_normal_cdf_(
            x, amplitude=self.amplitude, shape=self.shape, loc=self.loc, scale=self.scale, normalize=self.norm
        )

    def stats(self) -> dict[str, float]:
        r"""
        Compute descriptive summary statistics for the distribution.

        Returns
        -------
        dict
            A dictionary containing the calculated statistics.
            Key mappings:

            - "mean": The expected value, :math:`\mu`.
            - "median": The median value, :math:`\mu`.
            - "mode": The mode value, :math:`\mu`.
            - "variance": The calculated variance, :math:`\sigma^2`.
            - "std": The calculated standard deviation, :math:`\sigma`.
        """
        mean_ = self.loc
        median_ = self.loc
        mode_ = self.loc

        variance_ = 2 * LOG(self.scale) + gammaln(3 / self.shape) - gammaln(1 / self.shape)
        variance_ = EXP(variance_)

        return {"mean": mean_, "median": median_, "mode": mode_, "variance": variance_, "std": SQRT(variance_)}