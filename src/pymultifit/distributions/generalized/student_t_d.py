"""Created on July 28 11:15:23 2026"""

from ..backend import BaseDistribution
from ... import INF, NAN, NAN_DICT, SQRT
from ...typing import ArrayLike, NDArray
from ..utilities_d import (
    students_t_cdf_,
    students_t_log_cdf_,
    students_t_log_pdf_,
    students_t_pdf_,
)


class StudentsTDistribution(BaseDistribution):
    r"""
    Class for :class:`~.StudentsTDistribution`.

    Parameters
    ----------
    amplitude : float, default=1.0
        The amplitude or scaling factor of the distribution. Defaults to 1.0.
    v : float, default=1.0
        Degrees of freedom parameter, :math:`v`. Must be strictly positive (:math:`v > 0`).
    scale : float, default=1.0
        The scale parameter, :math:`\sigma`. Defaults to 1.0. Must be strictly positive.
    loc : float, default=0.0
        The location parameter, :math:`\mu`. Defaults to 0.0.
    normalize : bool, default=False
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Examples
    --------
    Importing libraries:

    .. literalinclude:: ../../../examples/basic/gaussian.py
       :language: python
       :linenos:
       :lineno-start: 3
       :lines: 3-7

    Generating a heavy-tailed Student's t-distribution (:math:`v=1, \mu=0, \sigma=1`), equivalent to a Cauchy distribution,
    with ``pyMultiFit`` and ``scipy``:

    .. literalinclude:: ../../../examples/basic/student_t_d.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/student_t_d.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/student_t_example1.png
       :alt: StudentsT(1, 0, 1)
       :align: center

    Generating a scaled and translated Student's t-distribution approaching a Gaussian (:math:`v=100, \mu=-3, \sigma=2.5, A=2.0`):

    .. literalinclude:: ../../../examples/basic/student_t_d.py
       :language: python
       :lineno-start: 32
       :lines: 32

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/student_t_d.py
       :language: python
       :lineno-start: 34
       :lines: 34-49

    .. image:: ../../../images/student_t_example2.png
       :alt: StudentsT(100, -3, 2.5, A=2.0)
       :align: center
    """

    def __init__(
        self,
        amplitude: float = 1.0,
        v: float = 1.0,
        scale: float = 1.0,
        loc: float = 0.0,
        normalize: bool = False,
    ):
        if v <= 0:
            raise ValueError(f"Degrees of freedom v must be > 0, got {v}")
        if scale <= 0:
            raise ValueError(f"Scale must be > 0, got {scale}")

        self.amplitude = float(amplitude)
        self.v = float(v)
        self.scale = float(scale)
        self.loc = float(loc)
        self.normalize = bool(normalize)

    def logpdf(self, x: ArrayLike) -> NDArray:
        return students_t_log_pdf_(
            x,
            amplitude=self.amplitude,
            v=self.v,
            scale=self.scale,
            loc=self.loc,
            normalize=self.normalize,
        )

    def pdf(self, x: ArrayLike) -> NDArray:
        return students_t_pdf_(
            x,
            amplitude=self.amplitude,
            v=self.v,
            scale=self.scale,
            loc=self.loc,
            normalize=self.normalize,
        )

    def cdf(self, x: ArrayLike) -> NDArray:
        return students_t_cdf_(
            x,
            amplitude=self.amplitude,
            v=self.v,
            scale=self.scale,
            loc=self.loc,
            normalize=self.normalize,
        )

    def logcdf(self, x: ArrayLike) -> NDArray:
        return students_t_log_cdf_(
            x,
            amplitude=self.amplitude,
            v=self.v,
            scale=self.scale,
            loc=self.loc,
            normalize=self.normalize,
        )

    def stats(self) -> dict[str, float]:
        v, scale, loc = self.v, self.scale, self.loc

        if any(param <= 0 for param in (v, scale)):
            return NAN_DICT

        mean_ = loc if v > 1 else NAN
        mode_ = loc
        variance_ = (scale**2 * (v / (v - 2.0))) if v > 2 else (INF if v > 1 else NAN)

        return {
            "mean": mean_,
            "mode": mode_,
            "variance": variance_,
            "std": SQRT(variance_) if v > 2 else NAN,
        }