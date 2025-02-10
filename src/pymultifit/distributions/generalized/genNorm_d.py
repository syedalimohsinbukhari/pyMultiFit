"""Created on Jan 29 15:42:23 2025"""

import numpy as np
from scipy.special import gammaln

from ..backend import BaseDistribution, errorHandling as erH
from ..utilities_d import sym_gen_normal_pdf_, sym_gen_normal_cdf_


class SymmetricGeneralizedNormalDistribution(BaseDistribution):
    r"""
    Class for SymmetricGeneralizedNormalDistribution.

    :param amplitude: The amplitude of the PDF. Defaults to 1.0. Ignored if **normalize** is ``True``.
    :type amplitude: float, optional

    :param shape: The shape parameter, :math:`\beta`. Defaults to 1.0.
    :type shape: float, optional

    :param loc: The shape parameter, :math:`\mu`. Defaults to 0.0.
    :type loc: float, optional

    :param scale: The standard deviation parameter, :math:`\alpha`. Defaults to 1.0.
    :type scale: float, optional

    :param normalize: If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
     Defaults to ``False``.
    :type normalize: bool, optional

    :raise NegativeAmplitudeError: If the provided value of amplitude is negative.
    :raise NegativeScaleError: If the provided value of scale parameter is negative.

    Examples
    --------
    Importing libraries:

    .. literalinclude:: ../../../examples/basic/gaussian.py
       :language: python
       :linenos:
       :lineno-start: 3
       :lines: 3-7

    Generating a standard SymmetricGeneralizedNormalDistribution(:math:`\beta=1, \mu=0, \alpha = 1`)
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

    Generating a scaled and translated SymmetricGeneralizedNormalDistribution(:math:`\beta=2, \mu=-3, \alpha=5`):

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

    def __init__(self, amplitude: float = 1.0, shape: float = 1.0, loc: float = 0.0, scale: float = 1.0,
                 normalize: bool = False):
        if amplitude < 0 and not normalize:
            raise erH.NegativeAmplitudeError()
        if shape < 0:
            raise erH.NegativeShapeError()
        self.amplitude = 1. if normalize else amplitude
        self.loc = loc
        self.scale = scale
        self.shape = shape

        self.norm = normalize

    @classmethod
    def scipy_like(cls, beta, loc: float = 0.0, scale: float = 1.0):
        """
        Instantiate SymmetricGeneralizedNormalDistribution with scipy parametrization.

        Parameters
        ----------
        beta: float
            The shape parameter.
        loc: float, optional
            The mean parameter. Defaults to 0.0.
        scale: float, optional
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        SymmetricGeneralizedNormalDistribution
            An instance of normalized SymmetricGeneralizedNormalDistribution.
        """
        return cls(shape=beta, loc=loc, scale=scale, normalize=True)

    def pdf(self, x):
        return sym_gen_normal_pdf_(x, amplitude=self.amplitude, shape=self.shape, loc=self.loc, scale=self.scale,
                                   normalize=self.norm)

    def cdf(self, x):
        return sym_gen_normal_cdf_(x, amplitude=self.amplitude, shape=self.shape, loc=self.loc, scale=self.scale,
                                   normalize=self.norm)

    def stats(self):
        mean_ = self.loc
        median_ = self.loc
        mode_ = self.loc

        variance_ = 2 * np.log(self.scale) + gammaln(3 / self.shape) - gammaln(1 / self.shape)
        variance_ = np.exp(variance_)

        return {'mean': mean_,
                'median': median_,
                'mode': mode_,
                'variance': variance_,
                'std': np.sqrt(variance_)}
