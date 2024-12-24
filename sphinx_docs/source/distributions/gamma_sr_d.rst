Gamma Distribution (SR)
=======================

The parent module for this distribution is :mod:`~pymultifit.distributions.gamma_d`.

.. autoclass:: pymultifit.distributions.gamma_d.GammaDistributionSR
   :no-members:


.. note::

   The :class:`~pymultifit.distributions.gamma_d.GammaDistributionSR` encompasses the following specific cases:

   #. :class:`~pymultifit.distributions.exponential_d.ExponentialDistribution`:
        - :math:`\alpha = 1`, and
        - :math:`\lambda_\text{gammaSR} = \lambda_\text{expon}`.

   #. :class:`~pymultifit.distributions.gamma_d.GammaDistributionSS`:
        - :math:`\lambda = \theta^{-1}`.


This class internally utilizes the following functions from the :mod:`~pymultifit.distributions.utilities` module:

* :func:`~pymultifit.distributions.utilities.gamma_sr_pdf_`
* :func:`~pymultifit.distributions.utilities.gamma_sr_cdf_`

Recommended Import
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pymultifit.distributions import GammaDistributionSR


Full Import
^^^^^^^^^^^

.. code-block:: python

   from pymultifit.distributions.gamma_d import GammaDistributionSR
