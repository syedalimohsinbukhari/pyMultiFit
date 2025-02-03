Distribution utilities
======================

.. py:currentmodule:: pymultifit.distributions.utilities_d

.. automodule:: pymultifit.distributions.utilities_d

The :mod:`~pymultifit.distributions.utilities_d` module serves as a cornerstone of the ``pyMultiFit`` library, containing majority of the core utility functions utilized across various classes and components.
It provides a comprehensive suite of mathematical and statistical tools, including probability distribution functions (PDFs), cumulative distribution functions (CDFs), some internal scaling and masking utilities, and data preprocessing methods.

Available for user
------------------

.. autofunction:: arc_sine_pdf_
.. autofunction:: arc_sine_log_pdf_
.. autofunction:: arc_sine_cdf_
.. autofunction:: arc_sine_log_cdf_
.. autofunction:: beta_pdf_
.. autofunction:: beta_log_pdf_
.. autofunction:: beta_cdf_
.. autofunction:: beta_log_cdf_
.. autofunction:: chi_square_pdf_
.. autofunction:: exponential_pdf_
.. autofunction:: exponential_cdf_
.. autofunction:: folded_normal_pdf_
.. autofunction:: folded_normal_cdf_
.. autofunction:: gamma_sr_pdf_
.. autofunction:: gamma_sr_cdf_
.. autofunction:: gamma_ss_pdf_
.. autofunction:: gaussian_pdf_
.. autofunction:: gaussian_cdf_
.. autofunction:: half_normal_pdf_
.. autofunction:: half_normal_cdf_
.. autofunction:: laplace_pdf_
.. autofunction:: laplace_cdf_
.. autofunction:: log_normal_pdf_
.. autofunction:: log_normal_cdf_
.. autofunction:: scaled_inv_chi_square_pdf_
.. autofunction:: scaled_inv_chi_square_log_pdf_
.. autofunction:: scaled_inv_chi_square_cdf_
.. autofunction:: scaled_inv_chi_square_log_cdf_
.. autofunction:: skew_normal_pdf_
.. autofunction:: skew_normal_cdf_
.. autofunction:: sym_gen_normal_pdf_
.. autofunction:: sym_gen_normal_cdf_
.. autofunction:: uniform_pdf_
.. autofunction:: uniform_cdf_

Internal functions
------------------

.. autofunction:: _beta_masking
.. autofunction:: _folded
.. autofunction:: _pdf_scaling
.. autofunction:: _remove_nans
