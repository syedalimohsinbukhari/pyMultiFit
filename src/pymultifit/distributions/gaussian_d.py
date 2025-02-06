"""Created on Aug 03 20:07:50 2024"""

from .backend import BaseDistribution, errorHandling as erH
from .utilities_d import gaussian_cdf_, gaussian_pdf_, gaussian_log_pdf_, gaussian_log_cdf_


class GaussianDistribution(BaseDistribution):
    r"""
    Class for Gaussian distribution.

    :param amplitude: The amplitude of the PDF. Defaults to 1.0. Ignored if **normalize** is ``True``.
    :type amplitude: float, optional

    :param mu: The mean parameter, :math:`\mu`. Defaults to 0.0.
    :type mu: float, optional

    :param std: The standard deviation parameter, :math:`\sigma`. Defaults to 1.0.
    :type std: float, optional

    :param normalize: If ``True``, the distribution is normalized so that the total area under the PDF equals 1. Defaults to ``False``.
    :type normalize: bool, optional

    :raise NegativeAmplitudeError: If the provided value of amplitude is negative.
    :raise NegativeStandardDeviationError: If the provided value of standard deviation is negative.

    Examples
    --------
    Importing libraries:

    .. literalinclude:: ../../../examples/basic/gaussian.py
       :language: python
       :linenos:
       :lineno-start: 3
       :lines: 3-7

    Generating a standard Gaussian(:math:`\mu=0, \sigma = 1`) distribution with ``pyMultiFit`` and ``scipy``:

    .. literalinclude:: ../../../examples/basic/gaussian.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/gaussian.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/gaussian_example1.png
       :alt: Gaussian(0, 1)
       :align: center

    Generating a translated Gaussian(:math:`\mu=3, \sigma=2`) distribution:

    .. literalinclude:: ../../../examples/basic/gaussian.py
       :language: python
       :lineno-start: 32
       :lines: 32

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/gaussian.py
       :language: python
       :lineno-start: 34
       :lines: 34-49

    .. image:: ../../../images/gaussian_example2.png
       :alt: Gaussian(3, 2)
       :align: center
    """

    def __init__(self, amplitude: float = 1.0, mu: float = 0., std: float = 1., normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        elif std <= 0:
            raise erH.NegativeStandardDeviationError()

        self.amplitude = 1. if normalize else amplitude
        self.mu = mu
        self.std_ = std
        self.norm = normalize

    @classmethod
    def scipy_like(cls, loc: float = 0.0, scale: float = 1.0):
        """
        Instantiate GaussianDistribution with scipy parametrization.

        Parameters
        ----------
        loc: float, optional
            The mean parameter. Defaults to 0.0.
        scale: float, optional
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        GaussianDistribution
            An instance of normalized GaussianDistribution.
        """
        return cls(mu=loc, std=scale, normalize=True)

    def pdf(self, x):
        return gaussian_pdf_(x, amplitude=self.amplitude, mean=self.mu, std=self.std_, normalize=self.norm)

    def logpdf(self, x):
        return gaussian_log_pdf_(x, amplitude=self.amplitude, mean=self.mu, std=self.std_, normalize=self.norm)

    def cdf(self, x):
        return gaussian_cdf_(x, amplitude=self.amplitude, mean=self.mu, std=self.std_, normalize=self.norm)

    def logcdf(self, x):
        return gaussian_log_cdf_(x, amplitude=self.amplitude, mean=self.mu, std=self.std_, normalize=self.norm)

    def stats(self):
        m, s = self.mu, self.std_

        return {'mean': m,
                'median': m,
                'mode': m,
                'variance': s**2,
                'std': s}
