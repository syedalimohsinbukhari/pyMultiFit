``result``
==========

.. py:currentmodule:: pymultifit.result

A :class:`FitResult` is an immutable description of a (possibly not yet performed) fit: the data, the fitted parameters and their covariance, the additive components of the model, and a label for every parameter.
It is everything the plotting and confidence-interval code needs, which is why :mod:`pymultifit.plot` and :mod:`pymultifit.ci` never have to import a fitter.

Obtain one from any fitter with :meth:`~pymultifit.fitters.backend.baseFitter.BaseFitter.to_result`:

.. code-block:: python

   result = fitter.to_result()
   result.model()               # total fitted curve at the data's x values
   result.model(x_new)          # ... or at any other points
   result.component_curve(0)    # first component alone
   result.residuals()           # data - model
   result.errors                # sqrt(diag(covariance))

Calling ``to_result()`` before ``fit()`` is allowed; the result then has no parameters or components, and the methods that need a fit raise ``RuntimeError``.
A result is a snapshot: a later ``fit()`` does not change an existing result.

.. list-table::
   :align: center
   :header-rows: 1

   * - Name
     - Description
   * - :class:`FitResult`
     - Immutable description of a fit.
   * - :class:`Component`
     - One additive component of the model: label, model function and parameter count.

.. autoclass:: FitResult
   :members:
   :class-doc-from: class

.. autoclass:: Component
   :members:
   :class-doc-from: class
