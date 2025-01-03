Gamma Distribution (SR)
=======================

.. autoclass:: pymultifit.distributions.gamma_d.GammaDistributionSR
   :members:
   :inherited-members:
   :show-inheritance:


.. note::

   The :class:`~pymultifit.distributions.gamma_d.GammaDistributionSR` encompasses the following specific cases:

   #. :class:`~pymultifit.distributions.exponential_d.ExponentialDistribution`:
        - :math:`\alpha = 1`, and
        - :math:`\lambda_\text{gammaSR} = \lambda_\text{expon}`.

   #. :class:`~pymultifit.distributions.gamma_d.GammaDistributionSS`:
        - :math:`\alpha_\text{gammaSR} = \alpha_\text{gammaSS}`
        - :math:`\lambda = \theta^{-1}`.

   #. :class:`~pymultifit.distributions.uniform_d.UniformDistribution`:
       - :math:`\alpha = 1`, and
       - :math:`\lambda = 1`.


This class internally utilizes the following functions from :mod:`~pymultifit.distributions.utilities_d` module:

* :func:`~pymultifit.distributions.utilities_d.gamma_sr_pdf_`
* :func:`~pymultifit.distributions.utilities_d.gamma_sr_cdf_`

Recommended Import
------------------

.. code-block:: python

   from pymultifit.distributions import GammaDistributionSR


Full Import
-----------

.. code-block:: python

   from pymultifit.distributions.gamma_d import GammaDistributionSR
