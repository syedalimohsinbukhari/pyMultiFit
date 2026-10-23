# Speed summary: support masking of the beta and arcsine functions

Commit `2eaf7f8` (change in `44c9cde`) against its parent package `a9ffeec`, measured on two machines with identical benchmark code (`2eaf7f8`).

## What changed

Package code in `src/pymultifit/distributions/utilities_d.py`:

1. `_beta_expr` computes only the form that is asked for (power form for the PDF, log form for the logPDF); it used to compute both and drop one.
2. `_beta_expr`, `beta_cdf_`, `beta_log_cdf_`, `arc_sine_cdf_` and `arc_sine_log_cdf_` evaluate the expensive part (`power`, `betainc`, `arcsin`) only for the points inside the support and fill the rest with constants. Outputs are bit-identical to the previous code, including for `nan`, `inf` and the support boundaries.

Why it mattered: the benchmark grid is `linspace(eps, 10)`, of which only 10 % (Beta(1, 1)) to 26 % (Beta(5, 80, -3, 5.6)) lies inside the beta support. scipy evaluates the in-support points only; multifit evaluated all of them.

## Runs

| label | folder | package | benchmark code |
|---|---|---|---|
| T530 before | `thinkpad_t530_a9ffeec_on_2eaf7f8` | a9ffeec | 2eaf7f8 |
| T530 after | `thinkpad_t530_2eaf7f8` | 2eaf7f8 | 2eaf7f8 |
| GPU_WS before | `sarl_gpu_ws_1_a9ffeec_on_2eaf7f8` | a9ffeec | 2eaf7f8 |
| GPU_WS after | `sarl_gpu_ws_1_2eaf7f8` | 2eaf7f8 | 2eaf7f8 |

- T530: Intel i7-3720QM (AVX, no AVX2/FMA). GPU_WS: AMD Threadripper PRO 5955WX (AVX2/FMA).
- Same Python, library versions and `uv.lock` on both machines. Performance governor, boost/turbo off, one pinned core, single-threaded BLAS.
- All four runs finished with status `ok` on a clean tree.

## Result

Ratio = multifit time / scipy time, median over the 12 largest sizes (below 1: multifit faster).

| function | parameters | T530 before → after | GPU_WS before → after |
|---|---|---|---|
| beta PDF | default | 1.88 → 0.60 (-68 %) | 2.29 → 0.67 (-71 %) |
| beta PDF | variable | 1.88 → 0.58 (-69 %) | 1.87 → 0.55 (-71 %) |
| beta CDF | default | 0.96 → 0.48 (-50 %) | 1.11 → 0.42 (-62 %) |
| beta CDF | variable | 0.92 → 0.77 (-16 %) | 0.98 → 0.72 (-27 %) |
| arcsine CDF | default | 0.86 → 0.29 (-66 %) | 0.83 → 0.27 (-68 %) |
| arcsine CDF | variable | 0.87 → 0.38 (-56 %) | 0.85 → 0.34 (-60 %) |

- Beta PDF went from about 2x slower than scipy to about 1.5-2x faster on both machines.
- The compare reports mark all six changes as consistent (both parameter sets agree) on both machines. There is no consistently slower result.
- Every other function and parameter set stays within about ±6 % on both machines. The only flag, gamma PDF on GPU_WS (+6 % / +2 %), is noise: the gamma code did not change, and in the default case one point in the median window has scipy's own time halving.
- Beta CDF with variable parameters gains the least: Beta(5, 80) has a costly `betainc` that masking cannot avoid.

## Reading notes

- Benchmarks of beta, arcsine and similar bounded distributions depend on how much of the grid lies inside the support. With all points inside the support the old and the new code take the same time, and multifit was already faster than scipy there (0.3-0.6x for beta PDF).
- Not covered by the benchmark: beta logPDF and logCDF (they got the same change), and the other bounded distributions (uniform has nothing to mask, arcsine PDF/logPDF were already masked, beta-prime and the generalized ones were not examined).
