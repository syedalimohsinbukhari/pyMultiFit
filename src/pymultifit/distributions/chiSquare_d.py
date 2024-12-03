"""Created on Dec 03 17:37:05 2024"""

from . import gamma_, GammaDistributionSR


class ChiSquareDistribution(GammaDistributionSR):
    """Class for chi-squared distribution."""

    def __init__(self, amplitude: float = 1., degree_of_freedom: float = 1., normalize: bool = False):
        super().__init__(amplitude=amplitude, shape=degree_of_freedom / 2., rate=1 / 2., normalize=normalize)


def chi_squared_(x, amplitude: float = 1., degree_of_freedom: float = 1., normalize: bool = False):
    """
    Compute the Chi-Squared distribution.

    The Chi-Squared distribution is a special case of the Gamma distribution with `alpha = degree_of_freedom / 2` and `beta = 1 / 2`.

    Parameters
    ----------
    x : array-like
        The input values at which to evaluate the Chi-Squared distribution.
    amplitude : float, optional
        The amplitude of the distribution. Defaults to 1.
        Ignored if `normalize` is set to True.
    degree_of_freedom : float, optional
        The degrees of freedom of the Chi-Squared distribution. Defaults to 1.
    normalize : bool, optional
        If True, the distribution will be normalized so that the PDF is at most 1. Defaults to False.

    Returns
    -------
    array-like
        The computed Chi-Squared distribution values for the input `x`.

    Notes
    -----
    The Chi-Squared distribution is related to the Gamma distribution by:
    .. math::
        f(x; k) = \\frac{x^{(k/2) - 1} e^{-x/2}}{2^{k/2} \\Gamma(k/2)}

    where `k` is the degrees of freedom, and `Γ(k)` is the Gamma function.

    If `normalize` is True, the distribution will be scaled such that the maximum value of the PDF is 1.
    """
    return gamma_(x, amplitude=amplitude, alpha=degree_of_freedom / 2., beta=1 / 2., normalize=normalize)
