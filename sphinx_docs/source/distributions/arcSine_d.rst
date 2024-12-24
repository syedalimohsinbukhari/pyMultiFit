ArcSine Distribution
====================

The parent module for this distribution is :mod:`~pymultifit.distributions.arcSine_d`.

.. autoclass:: pymultifit.distributions.arcSine_d.ArcSineDistribution
   :no-members:

.. note::
    The :class:`~pymultifit.distributions.arcSine_d.ArcSineDistribution` is a special case of the :class:`~pymultifit.distributions.beta_d.BetaDistribution`,

    * :math:`\alpha_\text{beta} = 0.5`,
    * :math:`\lambda_\text{beta} = 0.5`.

This class internally utilizes the following functions from the :mod:`~pymultifit.distributions.utilities` module:

* :func:`~pymultifit.distributions.utilities.beta_pdf_`
* :func:`~pymultifit.distributions.utilities.beta_cdf_`

..
    * :func:`~pymultifit.distributions.utilities.beta_logpdf_`

Recommended Import
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pymultifit.distributions import ArcSineDistribution


Full Import
^^^^^^^^^^^

.. code-block:: python

   from pymultifit.distributions.arcSine_d import ArcSineDistribution
