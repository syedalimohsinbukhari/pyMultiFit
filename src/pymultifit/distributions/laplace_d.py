"""Created on Aug 03 21:12:13 2024"""

from .backend import BaseDistribution, errorHandling as erH
from .utilities_d import laplace_cdf_, laplace_pdf_, laplace_log_pdf_, laplace_log_cdf_


class LaplaceDistribution(BaseDistribution):
    r"""
    Class for Laplace distribution.

    :param amplitude: The amplitude of the PDF. Defaults to 1.0. Ignored if **normalize** is ``True``.
    :type amplitude: float, optional

    :param mean: The mean parameter, :math:`\mu`. Defaults to 0.0.
    :type mean: float, optional

    :param diversity: The diversity parameter, :math:`b`. Defaults to 1.0.
    :type diversity: float, optional

    :param normalize: If ``True``, the distribution is normalized so that the total area under the PDF equals 1. Defaults to ``False``.
    :type normalize: bool, optional

    :raise NegativeAmplitudeError: If the provided value of amplitude is negative.
    :raise NegativeScaleError: If the provided value of diversity is negative.

    Examples
    --------
    Importing libraries:

    .. literalinclude:: ../../../examples/basic/laplace.py
       :language: python
       :linenos:
       :lineno-start: 3
       :lines: 3-7

    Generating a standard Laplace(:math:`\mu=0, b = 1`) distribution with ``pyMultiFit`` and ``scipy``:

    .. literalinclude:: ../../../examples/basic/laplace.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/laplace.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/laplace_example1.png
       :alt: Laplace(0, 1)
       :align: center

    Generating a translated Laplace(:math:`\mu=3, b=2`) distribution:

    .. literalinclude:: ../../../examples/basic/laplace.py
       :language: python
       :lineno-start: 32
       :lines: 32

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/laplace.py
       :language: python
       :lineno-start: 34
       :lines: 34-49

    .. image:: ../../../images/laplace_example2.png
       :alt: Laplace(3, 2)
       :align: center
    """

    def __init__(self, amplitude: float = 1., mean: float = 0, diversity: float = 1, normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        elif diversity <= 0:
            raise erH.NegativeScaleError('diversity')
        self.amplitude = 1. if normalize else amplitude
        self.mu = mean
        self.b = diversity

        self.norm = normalize

    @classmethod
    def scipy_like(cls, loc: float = 0.0, scale: float = 1.0):
        """
        Instantiate LaplaceDistribution with scipy parametrization.

        Parameters
        ----------
        loc: float, optional
            The location parameter. Defaults to 0.0.
        scale: float, optional
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        LaplaceDistribution
            An instance of normalized LaplaceDistribution.
        """
        return cls(mean=loc, diversity=scale, normalize=True)

    def pdf(self, x):
        return laplace_pdf_(x,
                            amplitude=self.amplitude, mean=self.mu, diversity=self.b, normalize=self.norm)

    def logpdf(self, x):
        return laplace_log_pdf_(x,
                                amplitude=self.amplitude, mean=self.mu, diversity=self.b, normalize=self.norm)

    def cdf(self, x):
        return laplace_cdf_(x,
                            amplitude=self.amplitude, mean=self.mu, diversity=self.b, normalize=self.norm)

    def logcdf(self, x):
        return laplace_log_cdf_(x,
                                amplitude=self.amplitude, mean=self.mu, diversity=self.b, normalize=self.norm)

    def stats(self):
        m, b = self.mu, self.b

        variance_ = 2 * b**2

        return {'mean': m,
                'median': m,
                'mode': m,
                'variance': variance_,
                'std': variance_**0.5}
