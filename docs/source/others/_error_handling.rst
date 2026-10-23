``errorHandling``
=================

.. py:currentmodule:: pymultifit.distributions.backend.errorHandling

.. autoexception:: BaseDistributionError
   :show-inheritance:

.. autoexception:: DegreeOfFreedomError
   :show-inheritance:

.. autoexception:: InvalidUniformParameters
   :show-inheritance:

.. autoexception:: NegativeAlphaError
   :show-inheritance:

.. autoexception:: NegativeAmplitudeError
   :show-inheritance:

.. autoexception:: NegativeBetaError
   :show-inheritance:

.. autoexception:: NegativeRateError
   :show-inheritance:

.. autoexception:: NegativeScaleError
   :show-inheritance:

.. autoexception:: NegativeShapeError
   :show-inheritance:

.. autoexception:: NegativeStandardDeviationError
   :show-inheritance:

.. autoexception:: NegativeVarianceError
   :show-inheritance:

.. autoexception:: XOutOfRange
   :show-inheritance:


``exceptions``
==============

The exceptions raised by ``pymultifit`` itself, outside the distribution classes, derive from one base class so that all of them can be caught together.
They are available from the package root (``from pymultifit import AxesError``) and from :mod:`pymultifit.exceptions`.

.. autoexception:: pymultifit.pyMultiFitErrors
   :show-inheritance:

.. autoexception:: pymultifit.AxesError
   :show-inheritance:
