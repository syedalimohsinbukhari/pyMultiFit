"""Created on Aug 29 10:15:58 2024"""

from typing import Any, Dict

import numpy as np
from scipy.special import erf

from . import BOLTZMANN_CONSTANT
from .._backend import BaseDistribution


class MaxwellBoltzmannDistribution(BaseDistribution):
    """Class for Maxwell-Boltzmann distribution."""

    def __init__(self, a: float = 0.):
        self.a = a

        self.norm = True
        self.amplitude = 1.

    @classmethod
    def with_amplitude(cls, a: float, amplitude: float):
        """
        Create an instance of the Maxwell-Boltzmann distribution with a specified amplitude.

        Parameters
        ----------
        a : float
            The scale factor for the distribution.
        amplitude : float
            The amplitude scaling factor.

        Returns
        -------
        MaxwellBoltzmannDistribution
            An instance of the Maxwell-Boltzmann distribution with the specified amplitude.
        """
        instance = cls(a=a)
        instance.amplitude = amplitude
        instance.norm = False
        return instance

    @classmethod
    def from_physical_parameters(cls, particle_mass: float, temperature: float):
        """
        Create an instance of Maxwell-Boltzmann distribution using physical parameters.

        This method computes the scale factor `a` based on the particle mass and temperature,
        defined as `a = sqrt(k_B * T / m)`, where `k_B` is the Boltzmann constant, `T` is the
        temperature, and `m` is the particle mass.

        Parameters
        ----------
        particle_mass : float
            Mass of the particle in kilograms (kg).
        temperature : float
            Temperature in Kelvin (K).

        Returns
        -------
        MaxwellBoltzmannDistribution
            An instance of the Maxwell-Boltzmann distribution with the computed scale factor.
        """
        scale_factor = np.sqrt(BOLTZMANN_CONSTANT * temperature / particle_mass)
        return cls(a=scale_factor)

    def _pdf(self, velocities: np.ndarray) -> np.ndarray:
        return _maxwell_boltzmann_distribution(velocities, self.a, self.norm) * self.amplitude

    def pdf(self, velocities: np.ndarray) -> np.ndarray:
        return self._pdf(velocities)

    def cdf(self, velocities: np.ndarray) -> np.ndarray:
        erf_part = erf(velocities / np.sqrt(2 * self.a))
        mbd_part = np.sqrt(2 / self.a) * (velocities / self.a) * np.exp(-velocities**2 / (2 * self.a**2))

        return erf_part - mbd_part

    def stats(self) -> Dict[str, Any]:
        mean_value = 2 * self.a * np.sqrt(2 / np.pi)
        mode_value = np.sqrt(2) * self.a
        variance_value = (self.a**2 * (3 * np.pi - 8)) / np.pi

        return {'mean': mean_value,
                'mode': mode_value,
                'variance': variance_value}


def _maxwell_boltzmann_distribution(velocities: np.ndarray,
                                    amplitude: float = 1., a: float = 0.,
                                    normalize: bool = True) -> np.ndarray:
    """
    Compute the Maxwell-Boltzmann distribution.

    Parameters
    ----------
    velocities : np.ndarray
        Array of velocities.
    a : float, optional
        The scale factor, defined as sqrt(k_B * T / m). Default is 0.
    normalize : bool, optional
        Flag to normalize the distribution. Default is True.

    Returns
    -------
    np.ndarray
        The Maxwell-Boltzmann distribution evaluated at the given velocities.
    """
    exponential_factor = velocities**3 * np.exp(-velocities**2 / (2 * a**2))

    if normalize:
        normalization_factor = np.sqrt(np.pi / 2) * a**3
        amplitude = 1
    else:
        normalization_factor = 1

    return amplitude * (exponential_factor / normalization_factor)
