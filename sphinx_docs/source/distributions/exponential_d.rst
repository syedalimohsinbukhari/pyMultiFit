Exponential Distribution
========================

The parent module for this distribution is: :mod:`pymultifit.distributions.exponential_d`.

.. autoclass:: pymultifit.distributions.exponential_d.ExponentialDistribution
   :no-members:
   :show-inheritance:

.. note::
    The :class:`~pymultifit.distributions.exponential_d.ExponentialDistribution` is a special case of the :class:`~pymultifit.distributions.gamma_d.GammaDistributionSR`,

    * :math:`\alpha_\text{gammaSR} = 1`,
    * :math:`\lambda_\text{gammaSR} = \lambda_\text{expon}`.

This class internally utilizes the following functions from the :mod:`~pymultifit.distributions.utilities` module:

* :mod:`~pymultifit.distributions.utilities.gamma_sr_pdf_`
* :mod:`~pymultifit.distributions.utilities.gamma_sr_cdf_`

Recommended Import
------------------

.. code-block:: python

   from pymultifit.distributions import ExponentialDistribution

Full Import
-----------

.. code-block:: python

   from pymultifit.distributions.exponential_d import ExponentialDistribution
