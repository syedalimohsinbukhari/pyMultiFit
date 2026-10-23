Benchmarking with ``scipy``
===========================

The benchmarks compare the distributions of ``pyMultiFit`` with their counterparts in SciPy (``scipy.stats``), for **speed** and for **accuracy**.
The notebooks below are run on a fixed machine setup, and what they report is summarised here.

.. toctree::
   :maxdepth: 1

   _bm_speed
   _bm_accuracy
   _bm_summary

Speed
-----

For every distribution, ``pdf`` and ``cdf`` are timed for a default and for a second, variable parameter set, on ``np.linspace(eps, 10, n)`` for 50 logarithmically spaced sizes ``n`` between 1 and :math:`10^6`.
Each timing is the median of 15 repetitions after 3 untimed warm-up calls, and the quantity that is compared is the ratio

.. math::

   \text{ratio} = \frac{t_\text{pyMultiFit}}{t_\text{SciPy}},

so a ratio below 1 means ``pyMultiFit`` is faster. It is read at large ``n``, as the median over the 12 largest sizes (:math:`n \gtrsim 5\times10^4`), where the cost per point dominates the fixed cost of a call.
At small ``n`` the time is dominated by the fixed cost of a call, and the ratio is closer to 1; the speed plots in the notebook show how it settles.

On an AMD Ryzen Threadripper PRO 5955WX (NumPy 2.2.6, SciPy 1.15.3, Python 3.10, one pinned core, the 2.0.0 development code), ``pyMultiFit`` is faster than SciPy for **all 48 combinations**, between 0.10 and 0.98 of SciPy's time:

.. list-table::
   :align: center
   :header-rows: 1

   * - Distribution
     - PDF, default
     - PDF, variable
     - CDF, default
     - CDF, variable
   * - Gaussian
     - 0.43
     - 0.56
     - 0.64
     - 0.62
   * - Laplace
     - 0.44
     - 0.44
     - 0.35
     - 0.35
   * - Skew-Normal
     - 0.67
     - 0.66
     - 0.10
     - 0.19
   * - Log-Normal
     - 0.49
     - 0.50
     - 0.71
     - 0.62
   * - Beta
     - 0.68
     - 0.55
     - 0.42
     - 0.72
   * - ArcSine
     - 0.32
     - 0.35
     - 0.27
     - 0.34
   * - Gamma
     - 0.41
     - 0.40
     - 0.87
     - 0.95
   * - Chi-Squared
     - 0.35
     - 0.38
     - 0.98
     - 0.89
   * - Folded Normal
     - 0.66
     - 0.78
     - 0.80
     - 0.91
   * - Half-Normal
     - 0.49
     - 0.50
     - 0.65
     - 0.70
   * - Exponential
     - 0.49
     - 0.49
     - 0.58
     - 0.54
   * - Uniform
     - 0.20
     - 0.38
     - 0.20
     - 0.32

An older Intel Core i7-3720QM (AVX only) gives the same picture, with all 48 ratios between 0.19 and 0.98.
Compare ratios and not absolute times: the time of one function moves by up to about 20 % between runs of the same code, while the ratio within a run is stable to a few percent.

.. note::

   The benchmark grid reaches only up to ``x = 10``, so for the distributions with a bounded support (ArcSine, Beta, Uniform) most of the grid lies *outside* their support.
   SciPy evaluates only the points inside the support, and the ArcSine and Beta functions of ``pyMultiFit`` do the same since 2.0.0, so the ratios of those rows depend on how much of the grid is inside the support.
   With every point inside the support ``pyMultiFit`` is still faster, for example 0.31 and 0.53 of SciPy's time for the PDF of ``Beta(1, 1)`` and ``Beta(5, 80)``, and 0.63 for the ArcSine PDF, at :math:`10^6` points.

Accuracy
--------

Both libraries are evaluated on the same inputs, a general logarithmic grid from :math:`10^{-10}` to :math:`10^{10}` and a set of edge cases, and the absolute differences of the ``pdf``, ``logpdf``, ``cdf`` and ``logcdf`` are plotted.
``pyMultiFit`` agrees with SciPy to better than :math:`10^{-15}` on average for all distributions except the ArcSine (:math:`\epsilon < 10^{-12}`) and the Chi-Squared (:math:`\epsilon < 10^{-11}`).

One difference is SciPy's: ``scipy.stats.laplace.logpdf`` takes the logarithm of the density, which is subnormal from :math:`x \approx 730` and zero from :math:`x \approx 745`, so it loses digits and then returns ``-inf``, whereas ``pyMultiFit`` evaluates :math:`-|x-\mu|/b - \ln(2b)` directly and is exact there.
The accuracy notebook shows this as a spike in the Laplace log-PDF plots, and ``benchmarks/laplace_logpdf_underflow.py`` in the repository compares both with the exact value.

Reproducing the benchmarks
--------------------------

The timings are only comparable when the conditions are the same every time, so the benchmark code in the repository (``benchmarks/``) is a protocol, not just a notebook:

- the library versions come from ``uv.lock`` (``uv sync --frozen``), the CPU governor is set to ``performance`` and turbo / boost is off, the process is pinned to one core and BLAS / OpenMP use one thread;
- ``run_benchmarks.py`` refuses to run on an unclean working tree, writes one folder per machine and commit under ``benchmarks/results/`` together with the captured environment (``env.json``), and ``compare_runs.py`` compares two runs with a verdict that takes the noise of SciPy's own time into account;
- ``--baseline-ref <commit>`` runs an older ``pyMultiFit`` with the current benchmark code, which is how a change is measured on its own.

The exact steps, the settings and the protocol are described in ``benchmarks/README.md``, and ``benchmarks/SPEED_SUMMARY.md`` summarises the last change that was measured this way.
The inputs of the timings are deterministic, so no random seed affects them (see :doc:`/reproducibility` for where random numbers do matter).
