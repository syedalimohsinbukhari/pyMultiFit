"""Created on Feb 02 03:46:43 2025"""

import numpy as np

from .backend import BaseDistribution, errorHandling as erH
from .utilities_d import (scaled_inv_chi_square_pdf_, scaled_inv_chi_square_log_pdf_, scaled_inv_chi_square_cdf_,
                          scaled_inv_chi_square_log_cdf_)


class ScaledInverseChiSquareDistribution(BaseDistribution):

    def __init__(self, amplitude: float = 1.0, df: float = 1.0, scale: float = 1.0, loc: float = 0.0,
                 normalize: bool = False):
        if not normalize and amplitude < 0:
            raise erH.NegativeAmplitudeError()
        if scale < 0:
            raise erH.NegativeScaleError()

        self.amplitude = 1 if normalize else amplitude
        self.df = df
        self.scale = scale
        self.tau2 = scale / df

        self.loc = loc
        self.norm = normalize

    @classmethod
    def scipy_like(cls, a, loc: float = 0.0, scale=1.0):
        return cls(df=a, loc=loc, scale=scale, normalize=True)

    def pdf(self, x):
        return scaled_inv_chi_square_pdf_(x, amplitude=self.amplitude, df=self.df, scale=self.scale, loc=self.loc,
                                          normalize=self.norm)

    def logpdf(self, x):
        return scaled_inv_chi_square_log_pdf_(x, amplitude=self.amplitude, df=self.df, scale=self.scale, loc=self.loc,
                                              normalize=self.norm)

    def cdf(self, x):
        return scaled_inv_chi_square_cdf_(x, amplitude=self.amplitude, df=self.df, scale=self.scale, loc=self.loc,
                                          normalize=self.norm)

    def logcdf(self, x):
        return scaled_inv_chi_square_log_cdf_(x, amplitude=self.amplitude, df=self.df, loc=self.loc, scale=self.scale,
                                              normalize=self.norm)

    def stats(self):
        v, tau2, loc = self.df, self.tau2, self.loc
        mean_ = (v * tau2) / (v - 2)
        median_ = None
        mode_ = (v * tau2) / (v + 2)
        variance_ = (2 * v**2 * tau2**2) / ((v - 2)**2 * (v - 4))

        return {'mean': mean_ + loc if v > 2 else np.inf,
                'median': median_,
                'mode': mode_ + loc,
                'variance': variance_ if v > 4 else np.inf,
                'std': np.sqrt(variance_) if v > 4 else np.inf}
