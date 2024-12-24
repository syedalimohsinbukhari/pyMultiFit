.. include:: ./../substitutions.rst

Gaussian Distribution
=====================

The parent module for this distribution is: :mod:`pymultifit.distributions.gaussian_d`.

.. autoclass:: pymultifit.distributions.gaussian_d.GaussianDistribution
   :no-members:

This class internally utilizes the following functions from the |utilities| module:

* |gaussian_pdf_|
* |gaussian_cdf_|

Full Import
-----------

.. code-block:: python

   from pymultifit.distributions.gaussian_d import GaussianDistribution

Recommended Import
------------------

.. code-block:: python

   from pymultifit.distributions import GaussianDistribution
