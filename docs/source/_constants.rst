``constants``
=============

The constants serve as predefined and standardized identifiers to ensure consistency and prevent potential spelling mistakes when referencing key elements, such as distributions and models, throughout the project.
These constants act as safeguards, making the codebase more robust and less error-prone by eliminating the need for users to manually input or recall exact strings.

Instead of typing "gaussian" directly, users can use the predefined constant :obj:`GAUSSIAN` to avoid typos.
Aliases such as :obj:`NORMAL` for :obj:`GAUSSIAN` ensure semantic clarity while maintaining consistency.
This approach not only reduces the likelihood of errors but also improves code readability and maintainability.

.. py:data:: EPSILON
   :type: float

   A small value used to prevent division by zero.

   .. note::
       .. code-block:: python

          EPSILON = np.finfo(float).eps

.. py:data:: GAUSSIAN
   :type: str

   Specifies the Gaussian distribution type.

.. py:data:: NORMAL
   :type: str

   Alias for `GAUSSIAN`.

.. py:data:: LOG_NORMAL
   :type: str

   Specifies the Log-Normal distribution type.

.. py:data:: SKEW_NORMAL
   :type: str

   Specifies the Skew-Normal distribution type.

.. py:data:: LAPLACE
   :type: str

   Specifies the Laplace distribution type.

.. py:data:: GAMMA_SR
   :type: str

   Gamma distribution with shape-rate parameterization.

.. py:data:: GAMMA_SS
   :type: str

   Gamma distribution with shape-scale parameterization.

.. py:data:: BETA
   :type: str

   Specifies the Beta distribution type.

.. py:data:: ARC_SINE
   :type: str

   Specifies the ArcSine distribution type.

.. py:data:: POWERLAW
   :type: str

   Specifies the PowerLaw distribution type.

.. py:data:: LINE
   :type: str

   Specifies a linear model type.

.. py:data:: LINEAR
   :type: str

   Alias for `LINE`.

.. py:data:: QUADRATIC
   :type: str

   Specifies a quadratic model type.

.. py:data:: CUBIC
   :type: str

   Specifies a cubic model type.


How to use
----------

They are available for various uses cases, and can be called directly from ``pyMultiFit`` using

.. code-block:: python

    from pymultifit import EPSILON
    from pymultifit import GAUSSIAN, LAPLACE

