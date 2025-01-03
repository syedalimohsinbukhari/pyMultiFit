Benchmarking with ``scipy``
===========================

This benchmarking file compares the performance and accuracy of the custom distributions implemented in `your_library` against their counterparts in SciPy (`scipy.stats`).

The purpose of these benchmarks is to evaluate:

1. **Speed**:
    - Measure the execution time of key operations (e.g., PDF and CDF).
    - Determine whether the custom implementations are comparable in performance to SciPy's optimized implementations.

2. **Accuracy**:
    - Compare the results of key operations (e.g., PDF, CDF) against SciPy's outputs to ensure numerical correctness.

3. **Usability**:
    - Highlight edge cases and parameter ranges where the custom implementations deviate or fail compared to SciPy.

Each benchmark will output timing information and accuracy metrics, providing insights into the trade-offs between speed and precision.

Key Benchmarked Operations
--------------------------

1. **Probability Density Function (PDF):**
    - Tests the performance of evaluating the likelihood at a given point.
    - This is a critical operation for statistical modeling.

2. **Cumulative Distribution Function (CDF):**
    - Benchmarks the computation of cumulative probabilities.
    - Used widely in hypothesis testing and statistical applications.

Accuracy Metrics:
    - Mean Absolute Error (MAE) and Relative Error will be calculated between custom and SciPy distributions for PDF and CDF outputs.

Benchmark Workflow
-------------------

Each distribution (e.g., Gaussian, Laplace) will be benchmarked as follows:

1. Generate test inputs:
    - Use a range of inputs (e.g., :obj:`numpy.linspace` for PDF/CDF).
    - Ensure the input range covers typical and edge-case scenarios.

2. Compare Speed:
    - Measure execution times for PDF and CDF for both the custom and SciPy implementations.

3. Compare Accuracy:
    - Compute differences (e.g., Mean Absolute Error) between the outputs of custom and SciPy distributions for identical inputs.

4. Output Results:
    - Print or store results in an easily interpretable format (tables or plots).

Expected Results
-----------------

At the end of the benchmarks, the following insights will be available:

1. Timing comparisons (in seconds) for each operation (PDF, CDF).
2. Accuracy comparisons using numerical error metrics.
3. Identification of cases where custom distributions outperform or lag behind SciPy counterparts.
4. Recommendations for further optimization, if needed.

.. toctree::
   :hidden:
   :maxdepth: 1

   _bm_accuracy
   _bm_speed
