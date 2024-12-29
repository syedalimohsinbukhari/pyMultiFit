"""Created on Aug 14 00:45:37 2024"""

from typing import Dict

import numpy as np
from scipy.special import betaincinv

from .backend import BaseDistribution, errorHandling as erH
from .utilities_d import beta_cdf_, beta_pdf_


class BetaDistribution(BaseDistribution):
    r"""
    Class for Beta distribution.

    Parameters
    ----------
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    alpha : float, optional
        The :math:`\alpha` parameter.
        Default is 1.0.
    beta : float, optional
        The :math:`\beta` parameter.
        Default is 1.0.
    loc : float, optional
        The location parameter, :math:`-` shifting.
        Defaults to 0.0.
    scale: float, optional
        The scale parameter, :math:`-` scaling.
        Defaults to 1.0,
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Examples
    --------
    Importing libraries

    .. literalinclude:: ../../../examples/basic/beta1.py
       :language: python
       :linenos:
       :lineno-start: 3
       :lines: 3-7

    Generating a standard :math:`\text{Beta}(2, 30)` distribution with ``pyMultiFit`` and ``scipy``.

    .. literalinclude:: ../../../examples/basic/beta1.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**

    .. literalinclude:: ../../../examples/basic/beta1.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/beta_example1.png
       :alt: Beta distribution (5, 30)
       :align: center

    Generating a shifted and translated :math:`\text{Beta}(2, 30)` distribution.

    .. literalinclude:: ../../../examples/basic/beta2.py
       :language: python
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**

    .. literalinclude:: ../../../examples/basic/beta2.py
       :language: python
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/beta_example2.png
       :alt: Beta distribution (shifted and translated)
       :align: center
    """

    def __init__(self,
                 amplitude: float = 1.0, alpha: float = 1.0, beta: float = 1.0,
                 loc: float = 0.0, scale: float = 1.0, normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        elif alpha <= 0:
            raise erH.NegativeAlphaError()
        elif beta <= 0:
            raise erH.NegativeBetaError()
        self.amplitude = 1. if normalize else amplitude
        self.alpha = alpha
        self.beta = beta
        self.loc = loc
        self.scale = scale

        self.norm = normalize

    def pdf(self, x: np.ndarray) -> np.ndarray:
        if self.scale > 0:
            return beta_pdf_(x=x, amplitude=self.amplitude, alpha=self.alpha, beta=self.beta, loc=self.loc, scale=self.scale, normalize=self.norm)
        else:
            return np.full(shape=x.shape, fill_value=np.nan)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        if self.scale > 0:
            return beta_cdf_(x=x, alpha=self.alpha, beta=self.beta, loc=self.loc, scale=self.scale)
        else:
            return np.full(shape=x.shape, fill_value=np.nan)

    # def logpdf(self, x: np.array) -> np.array:
    #     return beta_logpdf_(x=x, alpha=self.alpha, beta=self.beta, loc=self.loc, scale=self.scale, normalize=self.norm)

    def stats(self) -> Dict[str, float]:
        a, b = self.alpha, self.beta

        mean_ = a / (a + b)
        median_ = betaincinv(a, b, 0.5)
        mode_ = []
        if np.logical_and(a > 1, b > 1):
            mode_ = (a - 1) / (a + b - 2)

        variance_ = (a * b) / ((a + b)**2 * (a + b + 1))

        return {'mean': mean_,
                'median': median_,
                'mode': mode_,
                'variance': variance_}
