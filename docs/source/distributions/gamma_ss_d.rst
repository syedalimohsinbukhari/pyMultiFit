Gamma Distribution (SS)
=======================

.. autoclass:: pymultifit.distributions.gamma_d.GammaDistributionSS
   :no-members:

.. note::
    The :class:`~pymultifit.distributions.gamma_d.GammaDistributionSS` is a special case of the :class:`~pymultifit.distributions.gamma_d.GammaDistributionSR` with  :math:`\lambda = \theta^{-1}`.

This class internally utilizes the following functions from :mod:`~pymultifit.distributions.utilities_d` module:

* :func:`~pymultifit.distributions.utilities_d.gamma_sr_pdf_`
* :func:`~pymultifit.distributions.utilities_d.gamma_sr_cdf_`

Recommended Import
------------------

.. code-block:: python

   from pymultifit.distributions import GammaDistributionSS

Full Import
-----------

.. code-block:: python

   from pymultifit.distributions.gamma_d import GammaDistributionSS
