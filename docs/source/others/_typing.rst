``typing``
==========

.. py:currentmodule:: pymultifit.typing

The type aliases used in the signatures of ``pymultifit``.
They are plain aliases, so they can be used in your own annotations as well:

.. code-block:: python

   from pymultifit.typing import ArrayLike, NDArray, Params_

.. py:data:: ArrayLike
   :value: numpy.typing.ArrayLike

   Anything NumPy can turn into an array: a scalar, a (nested) sequence or an array.

.. py:data:: NDArray
   :value: numpy.typing.NDArray[numpy.floating]

   A NumPy array of floating point numbers.

.. py:data:: RaggedParams
   :value: list[tuple[float, ...]]

   The parameters of several components, one tuple per component, as in ``[(1.0, 5.0, 1.0), (2.0, 8.0, 0.5)]``.
   The tuples may have different lengths, which is why it is called ragged: a model of mixed components has a different number of parameters for each one.

.. py:data:: Params_
   :value: RaggedParams | NDArray

   The parameters of a fit, either as :data:`RaggedParams` or as an array.
