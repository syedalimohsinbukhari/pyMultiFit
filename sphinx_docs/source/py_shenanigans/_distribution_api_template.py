import re


def convert_to_snake_case(input_string):
    # Use regular expression to find the capital letters and split the string
    parts = re.findall(r'[A-Z][a-z]*', input_string)

    # Join the parts with underscores and lowercase the entire string
    return '_'.join(parts).lower()


def generate_distribution_rst_simple(distribution_name):
    """
    Generate an RST template for documenting a distribution using a single argument.

    Parameters
    ----------
    distribution_name : str
        The name of the distribution (e.g., "ArcSine").

    Returns
    -------
    str
        The RST content as a string.
    """
    module_name = distribution_name[0].lower() + distribution_name[1:]
    function_name = convert_to_snake_case(distribution_name)
    class_name = f"{distribution_name}Distribution"
    parent_module = f"pymultifit.distributions.{module_name}_d"
    parent_class = f"{parent_module}.{class_name}"
    utilities_module = "pymultifit.distributions.utilities"
    pdf_function = f"{function_name}_pdf_"
    cdf_function = f"{function_name}_cdf_"
    recommended_import = "pymultifit.distributions"

    template = f"""{distribution_name} Distribution
====================

The parent module for this distribution is: :mod:`{parent_module}`.

.. autoclass:: {parent_class}
   :no-members:

This class internally utilizes the following functions from the :mod:`~{utilities_module}` module:

* :mod:`~{utilities_module}.{pdf_function}`
* :mod:`~{utilities_module}.{cdf_function}`

Full Import
-----------

.. code-block:: python

   from {parent_module} import {class_name}

Recommended Import
------------------

.. code-block:: python

   from {recommended_import} import {class_name}
"""
    return module_name, template
