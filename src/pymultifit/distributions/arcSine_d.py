"""Created on Aug 14 02:02:42 2024"""

from .backend import errorHandling as erH
from .beta_d import BetaDistribution


class ArcSineDistribution(BetaDistribution):
    r"""
    Class for ArcSine distribution.

    Parameters
    ----------
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    loc : float, optional
        The location parameter, :math:`-` shifting.
        Defaults to 0.0.
    scale: float, optional
        The scale parameter, :math:`-` scaling.
        Defaults to 1.0,
    normalize : bool, optional
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
       :lines: 14-

    .. image:: ../../../images/arcsine_example.png
       :alt: ArcSine distribution
       :align: center

    Raises
    ------
    :class:`~pymultifit.distributions.backend.errorHandling.NegativeAmplitudeError`
        If the provided value of amplitude is negative.
    :class:`~pymultifit.distributions.backend.errorHandling.NegativeScaleError`
        If the provided value of scale is negative.
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
