Reproducibility
===============

Most of ``pyMultiFit`` is deterministic. Random numbers enter in two places only, and both are controlled by the same argument, ``rng_engine``.
This page explains what it accepts, the one rule that keeps a script both reproducible *and* statistically independent, and what reproducibility cannot promise.

Where random numbers are used
-----------------------------

.. list-table::
   :align: center
   :header-rows: 1

   * - Where
     - What is drawn
   * - :meth:`~pymultifit.fitters.backend.baseFitter.BaseFitter.confidence_intervals`, :func:`~pymultifit.ci.compute_ci_bounds`
     - The bootstrap parameter vectors, from the multivariate normal distribution of the fit.
   * - The data generators: :func:`~pymultifit.generators.generators.multiple_models`, :func:`~pymultifit.generators.generators.multi_gaussian` and the other ``multi_*`` functions
     - The noise, ``noise_level`` times standard-normal draws.

Nothing else draws random numbers. In particular fitting is deterministic for given data and starting values ``p0``, and so are the distributions (``pdf``, ``cdf``, ``stats``), the prediction intervals and the plots.

The ``rng_engine`` argument
---------------------------

.. list-table::
   :align: center
   :header-rows: 1

   * - ``rng_engine``
     - Meaning
   * - ``None`` (default)
     - Fresh entropy: every call differs. For the data generators, NumPy's global state is used instead, so ``np.random.seed(...)`` still reproduces the noise.
   * - an integer
     - The seed of a new :class:`numpy.random.Generator`: every call with the same integer gives the same draws.
   * - a :class:`numpy.random.Generator`
     - Used as it is. It advances with every draw, so a second call continues where the first one stopped.

The one-generator rule
----------------------

Create **one** generator at the top of a script and pass the same object to every call that draws random numbers.
The script is then reproducible from its single seed, and every call still gets *different* draws:

.. code-block:: python

   rng = np.random.default_rng(42)

   y = multi_gaussian(x, params=true_params, noise_level=0.4, rng_engine=rng)   # the noise of the data

   fitter = GaussianFitter(x, y)
   fitter.fit(p0)

   ci_1 = fitter.confidence_intervals(95, rng_engine=rng)   # draws that continue from the noise above
   ci_2 = fitter.confidence_intervals(95, rng_engine=rng)   # and new draws again

The tempting alternative repeats the draws. Passing ``rng_engine=42`` (or a new ``np.random.default_rng(42)``) to every call starts a fresh generator from the same state each time, so all the calls draw the same numbers.
That is harmless for one call, but it silently correlates anything that was meant to be independent:

.. code-block:: python

   for fitter in fitters:
       fitter.confidence_intervals(95, rng_engine=42)       # the same standard-normal draws for every fitter

   rng = np.random.default_rng(42)                          # instead, one generator for the whole loop
   for fitter in fitters:
       fitter.confidence_intervals(95, rng_engine=rng)

A single integer is fine when one call is all there is, for example ``fitter.confidence_intervals(95, rng_engine=42)`` to make one figure reproducible.

How many bootstrap samples
--------------------------

A fixed seed makes a figure reproducible, not accurate: it is one random draw of the band.
The edges of a band move from seed to seed by an amount that shrinks with the square root of ``n_bootstrap``, and ``ci.n_bootstrap`` records how many samples a result is based on.
The figure shows the upper edge of a 95 % band for eight different seeds, as the deviation from their average:

.. plot::
   :context: close-figs
   :include-source:

   import numpy as np
   from matplotlib import pyplot as plt

   from pymultifit.fitters import GaussianFitter
   from pymultifit.generators import multi_gaussian

   rng = np.random.default_rng(1)
   x = np.linspace(-15, 15, 300)
   y = multi_gaussian(x, params=[(10, -5, 2), (8, 5, 3)], noise_level=0.4, rng_engine=rng)

   fitter = GaussianFitter(x, y)
   fitter.fit([(8, -4, 1.5), (6, 4, 2)])

   fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
   for ax, n_bootstrap in zip(axes, (200, 5000)):
       upper = np.array([fitter.confidence_intervals(95, n_bootstrap=n_bootstrap, rng_engine=seed).overall[95.0].upper
                         for seed in range(8)])
       ax.plot(np.linspace(x.min(), x.max(), upper.shape[1]), (upper - upper.mean(axis=0)).T, lw=1)
       ax.set_title(f"n_bootstrap = {n_bootstrap}")
       ax.set_xlabel("x")
   axes[0].set_ylabel("upper edge of the 95 % band,\ndeviation from the mean over 8 seeds")
   fig.tight_layout()

Ask for enough samples that the edges are stable at the precision you need, and compute all the levels in one call: the extra levels cost almost nothing, while a new call draws again.

Running in several processes
----------------------------

``pyMultiFit`` draws all its random numbers in the calling process. If you run it from several worker processes, do not share one generator between them: a generator that is copied into each worker starts every worker from the same state, which repeats the draws.
Give every worker its own child stream instead:

.. code-block:: python

   children = np.random.SeedSequence(42).spawn(n_workers)           # independent, reproducible streams
   rngs = [np.random.default_rng(child) for child in children]      # hand one to each worker

What reproducibility does not cover
-----------------------------------

The same seed gives the same draws on the same software, not necessarily everywhere:

- NumPy only keeps the streams of a ``Generator`` stable within a version, so pin NumPy if you need the numbers to be identical in the long run.
- The multivariate normal draws use a matrix decomposition from LAPACK, and the optimiser behind ``fit()`` (:func:`scipy.optimize.curve_fit`) converges to its own tolerance. Both can differ in the last digits between platforms and BLAS builds; the test suite compares fits and intervals to ``1e-6``, not bit for bit.
- The timings of the benchmarks are a different matter, see :doc:`benchmarks/benchmarks`.
