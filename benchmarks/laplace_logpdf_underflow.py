"""scipy's Laplace logPDF underflows in the tail, ``pyMultiFit``'s does not.

``scipy.stats.laplace.logpdf`` evaluates the log of the PDF, which is below the smallest subnormal float for ``x`` above about 745,
so the result is ``-inf`` there, and it loses digits from ``x`` of about 729 on (the PDF is subnormal). The exact value is
``-|x - loc| / scale - ln(2 scale)``, which ``pyMultiFit`` computes directly. This is why the logPDF accuracy plots of the Laplace
distribution in ``accuracy.ipynb`` show a spike of about 1e308 (``|finite - (-inf)| = inf``, turned into the largest float by
``np.nan_to_num``): the reference is wrong, not the package.

Usage (from ``benchmarks/``)::

    uv run python laplace_logpdf_underflow.py
"""

import warnings

import numpy as np
import scipy.stats as ss

import pymultifit.distributions as pd

X = np.array([1.0, 100.0, 700.0, 729.0, 735.0, 742.0, 745.0, 746.0, 750.0, 1e3, 1e5, 1e10])


def main() -> None:
    custom = pd.LaplaceDistribution.from_scipy_params()
    reference = ss.laplace()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        multifit = custom.logpdf(X)
        scipy = reference.logpdf(X)
        scipy_pdf = reference.pdf(X)

    exact = -X - np.log(2.0)  # for loc = 0, scale = 1 and x > 0, correct to about one ulp

    print("Laplace(0, 1) logPDF: exact value against pyMultiFit and scipy")
    print(f"{'x':>8} {'exact':>16} {'pyMultiFit':>16} {'scipy':>16} {'scipy pdf':>11}   {'|multifit - exact|':>19} {'|scipy - exact|':>16}")
    for x, e, m, s, p in zip(X, exact, multifit, scipy, scipy_pdf):
        print(f"{x:8.0f} {e:16.9f} {m:16.9f} {s:16.9f} {p:11.2e}   {abs(m - e):19.2e} {abs(s - e):16.2e}")

    smallest_subnormal = np.nextafter(0.0, 1.0)
    print(f"\nsmallest positive float: {smallest_subnormal:.3e}; the PDF 0.5 * exp(-x) is below it from x = {-np.log(2 * smallest_subnormal):.1f}")

    nan_to_num = np.nan_to_num(np.abs(multifit - scipy), False, 0)
    print(f"largest |multifit - scipy| after np.nan_to_num: {nan_to_num.max():.3e}  (the spike in accuracy.ipynb)")


if __name__ == "__main__":
    main()
