``generators``
==============

Template function
-----------------

.. list-table::
   :align: center
   :header-rows: 1

   * - Name
     - Description
   * - :meth:`~pymultifit.generators.generators.multi_base`
     - The template function for the generator functions.

Standalone generators
---------------------

.. list-table::
   :align: center
   :header-rows: 1

   * - Name
     - Description
   * - :meth:`~pymultifit.generators.generators.multiple_models`
     - Generator for data based on a combination of different models.

Derived generators
------------------

.. list-table::
   :align: center
   :header-rows: 1

   * - Name
     - Description
   * - :meth:`~pymultifit.generators.generators.multi_chi_squared`
     - Chi-Squared distribution generator.
   * - :meth:`~pymultifit.generators.generators.multi_exponential`
     - Exponential distribution generator.
   * - :meth:`~pymultifit.generators.generators.multi_folded_normal`
     - Folded Normal distribution generator.
   * - :meth:`~pymultifit.generators.generators.multi_gaussian`
     - Gaussian distribution generator.
   * - :meth:`~pymultifit.generators.generators.multi_half_normal`
     - Half-Normal distribution generator.
   * - :meth:`~pymultifit.generators.generators.multi_laplace`
     - Laplace distribution generator.
   * - :meth:`~pymultifit.generators.generators.multi_log_normal`
     - Log-Normal distribution generator.
   * - :meth:`~pymultifit.generators.generators.multi_skew_normal`
     - Skewed Normal distribution generator.

Reproducible noise
------------------

Every generator takes the ``noise_level`` of the noise it adds and an optional ``rng_engine``, a NumPy :class:`~numpy.random.Generator` (or an integer seed for a new one) to draw it from:

.. code-block:: python

   rng = np.random.default_rng(0)

   y_1 = multi_gaussian(x, params, noise_level=0.4, rng_engine=rng)
   y_2 = multi_gaussian(x, params, noise_level=0.4, rng_engine=rng)   # different noise, the same script is still reproducible

``rng_engine=None`` (the default) draws from NumPy's global state, which ``np.random.seed`` seeds, as it always did.
See :doc:`/reproducibility` for the rule that keeps a script reproducible and its draws independent.


.. toctree::
   :hidden:

   GeneratorFunctions         <generators>
