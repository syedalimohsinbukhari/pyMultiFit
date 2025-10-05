Polynomial Functions
====================

This section documents **functional components** included within the distribution module, which do not represent true probability distributions, but are provided for convenience when modeling or fitting simple mathematical functions.

These function-like classes implement a `pdf(x)` method that evaluates the corresponding mathematical expression at the input values `x`, returning a NumPy array of outputs.
This allows them to be used interchangeably with other distribution-like objects in modeling pipelines where only functional evaluation is needed.
They are especially useful in scenarios where you wish to fit or visualize basic curves (e.g., lines, quadratics) alongside or in place of statistical distributions.

The following classes are currently available:

.. autoclass:: pymultifit.distributions.backend.polynomial_d.LineFunction
   :members: pdf


.. autoclass:: pymultifit.distributions.backend.polynomial_d.QuadraticFunction
   :members: pdf


.. autoclass:: pymultifit.distributions.backend.polynomial_d.CubicFunction
   :members: pdf
