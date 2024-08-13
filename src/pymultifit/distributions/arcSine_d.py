"""Created on Aug 14 02:02:42 2024"""

from typing import Optional

from .beta_d import BetaDistribution


class ArcSineDistribution(BetaDistribution):
    """
    Represents the arcsine distribution as a special case of the Beta distribution.

    The arcsine distribution is a special case of the Beta distribution with parameters
    alpha = 0.5 and beta = 0.5.
    """

    def __init__(self, amplitude: Optional[float] = 1., normalize: bool = True):
        """
        Initializes the arcsine distribution as a special case of the Beta distribution.

        Parameters
        ----------
        amplitude: float, optional
            Scaling factor for non-normalized arcsine distribution (default is 1).
        normalize : bool, optional
            Determines if the distribution should be normalized (default is True).

        Notes
        ------
            If normalize flag is set to True, it ignores the amplitude scaling.
        """
        super().__init__(amplitude, 0.5, 0.5, normalize)
