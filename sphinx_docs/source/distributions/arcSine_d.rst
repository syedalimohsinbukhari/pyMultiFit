ArcSine Distribution
====================

The parent module for this distribution is: :mod:`pymultifit.distributions.arcSine_d`.

.. autoclass:: pymultifit.distributions.arcSine_d.ArcSineDistribution
   :no-members:

This class internally utilizes the following functions from the :mod:`~pymultifit.distributions.utilities` module:

* :mod:`~pymultifit.distributions.utilities.beta_pdf_`
* :mod:`~pymultifit.distributions.utilities.beta_cdf_`

..
    * :mod:`~pymultifit.distributions.utilities.beta_logpdf_`


.. important::
    The :class:`~pymultifit.distributions.arcSine_d.ArcSineDistribution` is a special case of the :class:`~pymultifit.distributions.beta_d.BetaDistribution`,

    * :math:`\alpha_\text{beta} = 0.5`,
    * :math:`\lambda_\text{beta} = 0.5`.

Full Import
-----------

.. code-block:: python

   from pymultifit.distributions.arcSine_d import ArcSineDistribution

Recommended Import
------------------

.. code-block:: python

   from pymultifit.distributions import ArcSineDistribution
