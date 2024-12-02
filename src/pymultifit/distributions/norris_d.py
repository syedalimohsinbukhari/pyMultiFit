"""Created on Nov 30 04:23:27 2024"""

from typing import Any, Dict

import numpy as np

from .backend import BaseDistribution


class Norris2005Distribution(BaseDistribution):

    def __init__(self, amplitude: float = 1., rise_time: float = 1., decay_time: float = 1., normalize: bool = False):
        self.amplitude = amplitude
        self.rise_time = rise_time
        self.decay_time = decay_time

        self.norm = normalize

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return norris2005(x, amplitude=self.amplitude, rise_time=self.rise_time, decay_time=self.decay_time, normalize=self.norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("The Norris2005 function doesn't has CDF implemented yet.")

    def stats(self) -> Dict[str, Any]:
        raise NotImplementedError("The Norris2005 function doesn't has stats implemented yet.")


class Norris2011Distribution(BaseDistribution):

    def __init__(self, amplitude: float = 1., tau: float = 1., xi: float = 1., normalize: bool = False):
        self.amplitude = amplitude
        self.tau = tau
        self.xi = xi

        self.norm = normalize

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return norris2011(x, amplitude=self.amplitude, tau=self.tau, xi=self.xi, normalize=self.norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("The Norris2005 function doesn't has CDF implemented yet.")

    def stats(self) -> Dict[str, Any]:
        raise NotImplementedError("The Norris2005 function doesn't has stats implemented yet.")


def norris2005(x: np.ndarray,
               amplitude: float = 1., rise_time: float = 1., decay_time: float = 1.,
               normalize: bool = False) -> np.ndarray:
    """
    Computes the Norris 2005 light curve model.

    The Norris 2005 model describes a light curve with an asymmetric shape characterized by exponential rise and decay times.

    Parameters
    ----------
    x : np.ndarray
        The input time array at which to evaluate the light curve model.
    amplitude : float, optional
        The amplitude of the light curve peak. Default is 1.0.
    rise_time : float, optional
        The characteristic rise time of the light curve. Default is 1.0.
    decay_time : float, optional
        The characteristic decay time of the light curve. Default is 1.0.
    normalize : bool, optional
        Included for consistency with other distributions in the library.
        This parameter does not affect the output since normalization is not required for the Norris 2005 model. Default is False.

    Returns
    -------
    np.ndarray
        The evaluated Norris 2005 model at the input times `x`.

    References
    ----------
        Norris, J. P. (2005). ApJ, 627, 324–345.
        Robert, J. N. (2011). MNRAS, 419, 2, 1650-1659.
    """
    xi = np.sqrt(rise_time / decay_time)
    tau = np.sqrt(rise_time * decay_time)

    return norris2011(x, amplitude, tau, xi)


def norris2011(x: np.ndarray,
               amplitude: float = 1., tau: float = 1., xi: float = 1.,
               normalize: bool = False) -> np.ndarray:
    """
    Computes the Norris 2011 light curve model.

    The Norris 2011 model is a reformulation of the original Norris 2005 model, expressed in terms of different parameters to facilitate better
    scaling across various energy bands in gamma-ray burst (GRB) light curves. The light curve is modeled as:

        P(t) = A * exp(-ξ * (t / τ + τ / t))

    where τ and ξ are derived from the rise and decay times of the pulse.

    Parameters
    ----------
    x : np.ndarray
        The input time array at which to evaluate the light curve model.
    amplitude : float, optional
        The amplitude of the light curve peak (A in the formula). Default is 1.0.
    tau : float, optional
        The pulse timescale parameter (τ in the formula). Default is 1.0.
    xi : float, optional
        The asymmetry parameter (ξ in the formula). Default is 1.0.
    normalize : bool, optional
        Included for consistency with other distributions in the library.
        This parameter does not affect the output since normalization is not required for the Norris 2011 model. Default is False.

    Returns
    -------
    np.ndarray
        The evaluated Norris 2011 model at the input times `x`.

    Notes
    -----
    - In this parameterization, the pulse peak occurs at t_peak = τ.

    References
    ----------
        Norris, J. P. (2005). ApJ, 627, 324–345.
        Norris, J. P. (2011). MNRAS, 419, 2, 1650–1659.
    """
    fraction1 = x / tau
    fraction2 = tau / x
    return amplitude * np.exp(-xi * (fraction1 + fraction2))
