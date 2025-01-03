"""Created on Aug 14 02:02:42 2024"""

from .backend import errorHandling as erH
from .beta_d import BetaDistribution


class ArcSineDistribution(BetaDistribution):
    r"""
    Class for ArcSine distribution.

    .. note::
        The :class:`ArcSineDistribution` is a special case of the
        :class:`~pymultifit.distributions.beta_d.BetaDistribution`,

        * :math:`\alpha_\text{beta} = 0.5`,
        * :math:`\lambda_\text{beta} = 0.5`.

    :param amplitude: The amplitude of the PDF. Defaults to 1.0. Ignored if **normalize** is ``True``.
    :type amplitude: float, optional

    :param loc: The location parameter, :math:`-` shifting. Defaults to 0.0.
    :type loc: float, optional

    :param scale: The scale parameter, for shifting. Defaults to 1.0.
    :type scale: float, optional

    :param normalize: If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.
    :type normalize: bool, optional


    :raise NegativeAmplitudeError: If the provided value of amplitude is negative.
    :raise NegativeScaleError: If the provided value of scale is negative.

    :example:

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
       :lines: 14-

    .. image:: ../../../images/arcsine_example.png
       :alt: ArcSine distribution
       :align: center
    """

    def __init__(self, amplitude: float = 1., loc: float = 0.0, scale: float = 1.0, normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        if scale < 0:
            raise erH.NegativeScaleError()

        self.amplitude = 1 if normalize else amplitude
        self.loc = loc
        self.scale = scale

        self.norm = normalize
        super().__init__(amplitude=self.amplitude, alpha=0.5, beta=0.5, loc=self.loc, scale=self.scale,
                         normalize=self.norm)

    @classmethod
    def scipy_like(cls, loc: float = 0.0, scale: float = 1.0):
        """
        Instantiate ArcSineDistribution with scipy parameterization.

        Parameters
        ----------
        loc: float, optional
            The location parameter. Defaults to 0.0.
        scale: float, optional
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        ArcSineDistribution
            An instance of normalized ArcSineDistribution.
        """
        return cls(loc=loc, scale=scale, normalize=True)
