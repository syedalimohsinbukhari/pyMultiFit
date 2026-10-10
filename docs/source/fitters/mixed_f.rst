Mixed Data Fitter
==================

.. autoclass:: pymultifit.fitters.mixed_f.MixedDataFitter
   :members:
   :show-inheritance:
   :undoc-members:
   :class-doc-from: class


``MixedDataFitter`` fits a sum of different models to the same data.
The models are given by name in ``model_list``, and ``p0`` is a list with one initial guess per component:

.. code-block:: python

   from pymultifit import GAUSSIAN, LAPLACE, LINE
   from pymultifit.fitters import MixedDataFitter

   fitter = MixedDataFitter(x, y, model_list=[LINE, GAUSSIAN, LAPLACE])
   fitter.fit(p0=[(0.1, 2), (10, -5, 1.5), (4, 3, 1)])

The following models are available out of the box:

* :class:`~pymultifit.fitters.chiSquare_f.ChiSquareFitter` (``"chi_square"``)
* :class:`~pymultifit.fitters.exponential_f.ExponentialFitter` (``"exponential"``)
* :class:`~pymultifit.fitters.foldedNormal_f.FoldedNormalFitter` (``"folded_normal"``)
* :class:`~pymultifit.fitters.gamma_f.GammaFitter` (``"gamma"``)
* :class:`~pymultifit.fitters.gaussian_f.GaussianFitter` (``"gaussian"``, alias ``"normal"``)
* :class:`~pymultifit.fitters.halfNormal_f.HalfNormalFitter` (``"half_normal"``)
* :class:`~pymultifit.fitters.laplace_f.LaplaceFitter` (``"laplace"``)
* :class:`~pymultifit.fitters.polynomial_f.LineFitter` (``"line"``)
* :class:`~pymultifit.fitters.logNormal_f.LogNormalFitter` (``"log_normal"``)
* :class:`~pymultifit.fitters.skewNormal_f.SkewNormalFitter` (``"skew_normal"``)

Custom models can be supplied through ``model_dictionary``, a mapping from model name to a fitter class; when ``model_list`` is omitted it is taken from the dictionary's keys.

Parameters can be frozen per component by passing a sparse dictionary that maps zero-based component indices to a boolean mask, for example ``fitter.fit(p0, frozen={0: [False, True]})``.

Recommended Import
------------------

.. code-block:: python

   from pymultifit.fitters import MixedDataFitter

Full Import
-----------

.. code-block:: python

   from pymultifit.fitters.mixed_f import MixedDataFitter
