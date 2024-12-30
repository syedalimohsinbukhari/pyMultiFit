Overview
========

**pyMultiFit** is an open-source Python library designed to simplify **fitting multiple models** or a **mixture of models** to data with ease. It is particularly useful for researchers working with signals, spectra, and experimental datasets.

Why pyMultiFit?
---------------

Data fitting is the backbone of scientific analysis, serving as the bread-and-butter for any researcher dealing with experimental or simulated data.
While popular libraries like **NumPy** and **SciPy** offer functions such as :obj:`~numpy.polyfit` and :obj:`~scipy.optimize.curve_fit` for polynomial and generic curve fittings, extending these tools for **multi-model fitting** is often cumbersome and repetitive.

This is where **pyMultiFit** steps in.
It provides out-of-the-box support for common multi-fitters and allows seamless integration of **user-defined fitters** with minimal effort.

Key Features
------------

1. **Traditional Multi-Fitters**
   Built-in support for common fitting models such as:

   - :class:`~pymultifit.distributions.gaussian_d.GaussianDistribution`
   - :class:`~pymultifit.distributions.laplace_d.LaplaceDistribution`

2. **User-Defined Fitters**
   A :class:`~pymultifit.fitters.backend.baseFitter.BaseFitter` class serves as an anchor for users to create custom fitters with minimal boilerplate code.

3. **N-Modal Data Generation**
   Generate synthetic datasets with **N Gaussian components** or other statistical distributions, perfect for:

   - Testing fitters.
   - Simulating realistic multi-modal data for research.

4. **Statistical Distributions**
   Provides built-in statistical distributions that can be easily incorporated into your workflows.

Benefits
--------

- **Ease of Use**: Simplifies the process of fitting multiple models without requiring extensive redefinitions.
- **Customizable**: Offers flexibility through the :class:`~pymultifit.fitters.backend.baseFitter.BaseFitter` class for domain-specific models.
- **Synthetic Data Support**: Makes it easy to generate complex, multi-modal datasets for testing and validation.
- **Research Focused**: Tailored to meet the needs of researchers dealing with signals, spectra, and experimental data.

Get Started
-----------

With **pyMultiFit**, you can focus more on analyzing your data and less on redefining model fitters. Whether you're fitting a spectrum with **five Gaussian peaks** or creating your custom statistical models, **pyMultiFit** has got you covered.


.. toctree::
   :hidden:

   installation
   tutorials
   api_index
