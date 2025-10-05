Gamma Distribution
=======================

.. autoclass:: pymultifit.distributions.gamma_d.GammaDistribution
   :members:
   :inherited-members:
   :show-inheritance:
   :member-order: groupwise


.. note::

   The :class:`~pymultifit.distributions.gamma_d.GammaDistribution` encompasses the following specific cases:

   #. :class:`~pymultifit.distributions.exponential_d.ExponentialDistribution`:
        - :math:`\alpha = 1`, and
        - :math:`\theta_\text{gamma} = \dfrac{1}{\lambda_\text{expon}}`.

   #. :class:`~pymultifit.distributions.uniform_d.UniformDistribution`:
       - :math:`\alpha = 1`, and
       - :math:`\theta = 1`.


This class internally utilizes the following functions from :mod:`~pymultifit.distributions.utilities_d` module:

* :func:`~pymultifit.distributions.utilities_d.gamma_pdf_`
* :func:`~pymultifit.distributions.utilities_d.gamma_cdf_`

Recommended Import
------------------

.. code-block:: python

   from pymultifit.distributions import GammaDistribution


Full Import
-----------

.. code-block:: python

   from pymultifit.distributions.gamma_d import GammaDistribution
