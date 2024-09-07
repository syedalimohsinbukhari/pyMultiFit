"""Created on Aug 31 19:33:59 2024"""

from typing import Union

import numpy as np
from matplotlib import pyplot as plt

floatArr = Union[float, np.ndarray]


def norris_pulse_xi_tau(time_array: floatArr, amplitude: float, xi: float, tau: float) -> floatArr:
    """
    Calculate the Norris pulse profile using the xi and tau parameters.

    Parameters
    ----------
    time_array : float or np.ndarray
        The time at which to evaluate the pulse profile. This can be a single time or an array of time points.
    amplitude : float
        The amplitude of the pulse.
    xi : float
        The shape parameter of the pulse.
    tau : float
        The characteristic timescale of the pulse.

    Returns
    -------
    float or np.ndarray
        The value of the Norris pulse profile at the given time(s).
    """
    exp_term = time_array / tau
    exp_term += tau / time_array
    return amplitude * np.exp(-xi * exp_term)


def norris_pulse_rise_decay(time_array: floatArr, amplitude: float, rise_time: float, decay_time: float) -> floatArr:
    """
    Calculate the Norris pulse profile using the rise and decay times.

    Parameters
    ----------
    time_array : float or np.ndarray
        The time at which to evaluate the pulse profile. This can be a single time or an array of time points.
    amplitude : float
        The amplitude of the pulse.
    rise_time : float
        The rise time of the pulse.
    decay_time : float
        The decay time of the pulse.

    Returns
    -------
    float or np.ndarray
        The value of the Norris pulse profile at the given time(s).
    """
    tau = np.sqrt(rise_time * decay_time)
    xi = np.sqrt(rise_time / decay_time)
    return norris_pulse_xi_tau(time_array=time_array, amplitude=amplitude, xi=xi, tau=tau)


def power_law(energy: floatArr, amplitude: float, alpha: float, pivot_energy: float = 100.0) -> floatArr:
    """
    Evaluate a power-law function at a given energy or array of energies.

    Parameters
    ----------
    energy : float or np.ndarray
        The energy at which to evaluate the power-law function.
        Can be a single energy value or an array of energy values.
    amplitude : float
        The amplitude (normalization) of the power-law function.
    alpha : float
        The spectral index of the power-law function.
    pivot_energy : float, optional
        The pivot energy at which the amplitude is defined, by default 100.0.

    Returns
    -------
    float or np.ndarray
        The value of the power-law function at the given energy or energies.
    """
    return amplitude * (energy / pivot_energy)**alpha


from typing import Union
import numpy as np

floatArr = Union[float, np.ndarray]


def broken_power_law(energy: floatArr, amplitude: float, pivot_energy: float,
                     index_low: float, break_energy: float, index_high: float) -> floatArr:
    """
    Evaluate a broken power-law function using np.piecewise.

    Parameters
    ----------
    energy : float or np.ndarray
        The energy at which to evaluate the broken power-law function. Can be a single energy value or an array of energy values.
    amplitude : float
        The amplitude (normalization) of the broken power-law function.
    pivot_energy : float
        The pivot energy in keV, at which the amplitude is defined.
    index_low : float
        The spectral index below the break energy.
    break_energy : float
        The break energy in keV, where the power-law changes slope.
    index_high : float
        The spectral index above the break energy.

    Returns
    -------
    float or np.ndarray
        The value of the broken power-law function at the given energy or energies.
    """

    def low_energy(energy):
        return amplitude * (energy / pivot_energy)**index_low

    def high_energy(energy):
        return amplitude * (break_energy / pivot_energy)**index_low * (energy / break_energy)**index_high

    return np.piecewise(energy, [energy <= break_energy, energy > break_energy],
                        [low_energy, high_energy])


t = np.linspace(0.1, 1500, 10_000)
y = norris_pulse_xi_tau(t, 600, 1, 50)

plt.plot(t, y)
plt.show()
