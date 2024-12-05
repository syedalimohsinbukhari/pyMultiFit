"""Created on Aug 14 02:02:42 2024"""

from .beta_d import BetaDistribution


class ArcSineDistribution(BetaDistribution):
    """
    Class for the ArcSine distribution, a special case of the Beta distribution with alpha=0.5 and beta=0.5.

    The ArcSine distribution is a Beta distribution where both the shape parameters are fixed at 0.5.
    The user can still provide an amplitude and normalization flag as options, but the alpha and beta parameters are overridden to always be 0.5.

    Parameters
    ----------
    amplitude : float, optional
        The scaling factor for the distribution. Default is 1.
    alpha : float, optional
        The alpha parameter for the Beta distribution. This value will be overridden to 0.5. Default is 0.5.
    beta : float, optional
        The beta parameter for the Beta distribution. This value will be overridden to 0.5. Default is 0.5.
    normalize : bool, optional
        Whether to normalize the distribution (i.e., set the PDF to integrate to 1). Default is False.

    Notes
    -----
    Although the `alpha` and `beta` parameters are accepted as inputs, they will always be set to 0.5 internally,
    as the ArcSine distribution is a special case of the Beta distribution with these fixed parameters.
    """

    def __init__(self, amplitude: float = 1., alpha: float = 0.5, beta: float = 0.5, normalize: bool = False):
        # respect the user passed values if they're equal to 0.5, otherwise overwrite them.
        alpha = alpha if alpha == 0.5 else 0.5
        beta = beta if beta == 0.5 else 0.5
        super().__init__(amplitude=amplitude, alpha=alpha, beta=beta, normalize=normalize)
