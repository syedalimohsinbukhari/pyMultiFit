Beta Distribution
====================

The parent module for this distribution is: :mod:`pymultifit.distributions.beta_d`.

.. autoclass:: pymultifit.distributions.beta_d.BetaDistribution
   :no-members:

.. note::

   The :class:`~pymultifit.distributions.beta_d.BetaDistribution` encompasses the following specific cases:

   - :class:`~pymultifit.distributions.arcSine_d.ArcSineDistribution` with parameters :math:`\alpha = 0.5` and :math:`\beta = 0.5`.
   - :class:`~pymultifit.distributions.uniform_d.UniformDistribution` with parameters :math:`\alpha = 1` and :math:`\beta = 1`.

This class internally utilizes the following functions from the :mod:`~pymultifit.distributions.utilities` module:

* :mod:`~pymultifit.distributions.utilities.beta_pdf_`
* :mod:`~pymultifit.distributions.utilities.beta_cdf_`

..
    * :mod:`~pymultifit.distributions.utilities.beta_logpdf_`

Recommended Import
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pymultifit.distributions import BetaDistribution

Full Import
^^^^^^^^^^^

.. code-block:: python

   from pymultifit.distributions.beta_d import BetaDistribution
