"""Created on Nov 10 00:12:52 2024"""

from typing import Any, Dict

import numpy as np

from ._backend import BaseDistribution


class PowerLawDistribution(BaseDistribution):
    
    def __init__(self, alpha):
        self.alpha = alpha
        
        self.norm = True
        self.amplitude = 1.
    
    @classmethod
    def with_amplitude(cls, amplitude: float = 1, alpha: float = -1):
        """
        Create an instance with a specified amplitude, without normalization.

        Parameters
        ----------
        amplitude : float
            The amplitude to apply to the PDF. Defaults to 1.
        alpha : float
            The decay of the spectrum. Defaults to -1.

        Returns
        -------
        PowerLawDistribution
            An instance of PowerLawDistribution with the specified parameters.
        """
        instance = cls(alpha=alpha)
        instance.amplitude = amplitude
        instance.norm = False
        
        return instance
    
    def _pdf(self, x: np.ndarray):
        return power_law_(x, amplitude=self.amplitude, alpha=self.alpha, normalize=self.norm)
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        pass
    
    def stats(self) -> Dict[str, Any]:
        pass


def power_law_(x: np.ndarray, amplitude: float = 1, alpha: float = -1, normalize: bool = True) -> np.ndarray:
    """
    Compute a power-law function for a given set of x values.

    This function is designed with a `normalize` parameter for consistency with other probability density functions (PDFs).
    However, the `normalize` parameter has no effect in this function, as normalization of the power law is not handled here.

    Parameters
    ----------
    x : np.ndarray
        Input values for which the power-law function will be evaluated.
    amplitude : float, optional
        The amplitude or scaling factor of the power law, by default 1.
    alpha : float, optional
        The exponent of the power law, by default -1.
    normalize : bool, optional
        Included for consistency with other PDF functions. Has no effect on the output. Defaults to True.

    Returns
    -------
    np.ndarray
        Computed power-law values for each element in x.
    """
    return amplitude * x**-alpha


powerLawWA = PowerLawDistribution.with_amplitude
