Beta Distribution
====================

.. autoclass:: pymultifit.distributions.beta_d.BetaDistribution
   :members: scipy_like
   :show-inheritance:

.. note::

   The :class:`~pymultifit.distributions.beta_d.BetaDistribution` encompasses the following specific cases:

   #. :class:`~pymultifit.distributions.arcSine_d.ArcSineDistribution`:
        - :math:`\alpha = \beta = 0.5`

   - :class:`~pymultifit.distributions.uniform_d.UniformDistribution`:
        - :math:`\alpha = \beta = 1`

This class internally utilizes the following functions from :mod:`~pymultifit.distributions.utilities_d` module:

* :func:`~pymultifit.distributions.utilities_d.beta_pdf_`
* :func:`~pymultifit.distributions.utilities_d.beta_cdf_`

..
    * :func:`~pymultifit.distributions.utilities_d.beta_logpdf_`

Recommended Import
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pymultifit.distributions import BetaDistribution

Full Import
^^^^^^^^^^^

.. code-block:: python

   from pymultifit.distributions.beta_d import BetaDistribution
