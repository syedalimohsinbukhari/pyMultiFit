ChiSquare Distribution
======================

The parent module for this distribution is: :mod:`~pymultifit.distributions.chiSquare_d`.

.. autoclass:: pymultifit.distributions.chiSquare_d.ChiSquareDistribution
   :show-inheritance:

.. note::
    The :class:`~pymultifit.distributions.chiSquare_d.ChiSquareDistribution` is a special case of the :class:`~pymultifit.distributions.gamma_d.GammaDistributionSR`,

    * :math:`\alpha_\text{gammaSR} = \text{dof} / 2`,
    * :math:`\lambda_\text{gammaSR} = 0.5`.

This class internally utilizes the following functions from the :mod:`~pymultifit.distributions.utilities` module:

* :mod:`~pymultifit.distributions.utilities.gamma_sr_pdf_`
* :mod:`~pymultifit.distributions.utilities.gamma_sr_cdf_`

Recommended Import
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pymultifit.distributions import ChiSquareDistribution

Full Import
^^^^^^^^^^^

.. code-block:: python

   from pymultifit.distributions.chiSquare_d import ChiSquareDistribution
