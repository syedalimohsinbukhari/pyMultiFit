"""Created on Aug 14 02:02:42 2024"""

from .beta_d import BetaDistribution


class ArcSineDistribution(BetaDistribution):
    """Class for ArcSince distribution, a special case of Beta distribution with alpha=0.5, and beta=0.5"""

    def __init__(self):
        super().__init__(alpha=0.5, beta=0.5)

    @classmethod
    def with_amplitude(cls, amplitude: float = 1., alpha: float = 0.5, beta: float = 0.5):
        """
        Create an instance of ArcSineDistribution with a specified amplitude.

        Parameters
        ----------
        amplitude : float
            The amplitude to apply to the PDF. Defaults to 1.
        alpha: float
            The alpha parameter of the ArcSine distribution. Defaults to 0.5.
        beta: float
            The beta parameter of the ArcSine distribution. Defaults to 0.5.

        Returns
        -------
        ArcSineDistribution
            An instance of ArcSineDistribution with the specified amplitude.

        Notes
        -----
        The `alpha` and `beta` parameters are only here to match the BetaDistribution class signatures.
        These parameters are internally overridden to produce the required ArcSine distribution.
        """
        instance = cls()
        instance.amplitude = amplitude
        instance.norm = False
        return instance
