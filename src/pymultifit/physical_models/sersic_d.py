"""Created on Nov 15 10:38:42 2024"""

from typing import Any, Dict

import numpy as np
from scipy.special import gamma as g_

from ..distributions._backend import BaseDistribution


def b_n(n):
    f1 = 2 * n
    f2 = 1 / 3
    f3 = 4 / (405 * n)
    f4 = 46 / (25515 * n**2)
    f5 = 131 / (11487175 * n**3)
    f6 = 2194697 / (30690717750 * n**4)
    return f1 - f2 + f3 + f4 + f5 - f6


def p_n(n):
    f1 = 1
    f2 = 0.6097 / n
    f3 = 0.05563 / n**2
    return f1 - f2 + f3


class SersicDistribution(BaseDistribution):
    
    def __init__(self, eff_radius, eff_density, n):
        self.eff_radius = eff_radius
        self.eff_density = eff_density
        self.n = n
        self.p = p_n(n)
        
        self.norm = True,
        self.amplitude = 1.
    
    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return p_dehnen(x, self.eff_radius, self.eff_density, self.n)
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        pass
    
    def stats(self) -> Dict[str, Any]:
        pass


def sersic_(x: np.ndarray,
            amplitude, eff_radius, eff_density, n, normalize: bool = True):
    r_ratio = x / eff_radius
    f1 = eff_density * pow(r_ratio, -p_n(n))
    f2 = np.exp(-b_n(n) * pow(r_ratio, 1 / n))
    
    return amplitude * f1 * f2


def sersic_M(x: np.ndarray,
             amplitude, eff_radius, mass, n, normalize: bool = True):
    b = b_n(n)
    p = p_n(n)
    po = (mass / (4.0 * 3.14 * (eff_radius**3.0) * n * np.power(b, n * (p - 3.0)) * g_(n * (3.0 - p))))
    return po * (np.power(x / eff_radius, -p)) * (np.exp(-b * (np.power(x / eff_radius, 1.0 / n))))


def m_dehnen(x, M, rs, gamma):
    return M * ((x / (x + rs))**(3 - gamma))


def p_dehnen(r, M, rs, gamma):
    # rs = rh * ((2**(1 / (3 - gamma))) - 1)
    f1 = (3 - gamma) * M * rs
    f2 = 4 * np.pi * (r**gamma) * ((r + rs)**(4 - gamma))
    return f1 / f2

# class SphericalDistribution(BaseDistribution):
#
#     def __init__(self, _alpha, _beta, _gamma):
#         self.alpha = _alpha
#         self.beta = _beta
#         self.gamma = _gamma
#
#         self.norm = True,
#         self.amplitude = 1.
#
#     def _pdf(self, x: np.ndarray) -> np.ndarray:
#         return spherical(x, self.alpha, self.beta, self.gamma)
#
#     def pdf(self, x: np.ndarray) -> np.ndarray:
#         return self._pdf(x)
#
#     def cdf(self, x: np.ndarray) -> np.ndarray:
#         pass
#
#     def stats(self) -> Dict[str, Any]:
#         pass
#
#
# def spherical(r: np.ndarray, a: float, b: float, g: float):
#     beta_ = betainc(a * (3 - g), a * (b - 3), 1)
#     print(beta_)
#     num = 1 / (4 * np.pi * a * beta_)
#     den = r**g * pow(1 + pow(r, 1 / a), (b - g) * a)
#
#     return num / den
