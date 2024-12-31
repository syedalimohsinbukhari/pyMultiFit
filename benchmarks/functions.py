"""Created on Dec 31 05:45:40 2024"""

import numpy as np
from matplotlib import pyplot as plt

from pymultifit import EPSILON


def test_and_plot(general_case, edge_case, custom_dist, scipy_dist, title):
    accuracy_g = compare_accuracy(general_case[0], custom_dist, scipy_dist)
    plot_accuracy(general_case[0], accuracy_g, f"General Case {title}")

    for x_ in edge_case:
        accuracy_e = compare_accuracy(x_, custom_dist, scipy_dist)
        plot_accuracy(x_, accuracy_e, f"Edge Case {title}")


def compare_accuracy(x, custom_dist, scipy_dist):
    pdf_custom = custom_dist.pdf(x)
    cdf_custom = custom_dist.cdf(x)
    pdf_scipy = scipy_dist.pdf(x)
    cdf_scipy = scipy_dist.cdf(x)

    pdf_abs_diff = np.abs(pdf_custom - pdf_scipy) + EPSILON

    cdf_abs_diff = np.abs(cdf_custom - cdf_scipy) + EPSILON

    return {"pdf_abs_diff": pdf_abs_diff,
            "cdf_abs_diff": cdf_abs_diff}


# Plotting Function
def plot_accuracy(x, results, title_suffix):
    plt.figure(figsize=(12, 4))

    # Absolute Difference (PDF)
    plt.subplot(1, 2, 1)
    plt.plot(x, results["pdf_abs_diff"], label="PDF Absolute Diff", marker='.')
    plt.xscale("log")
    plt.yscale("symlog")
    plt.xlabel("x")
    plt.ylabel("Absolute Difference (PDF)")
    plt.title(f"Absolute Difference:\nPDF {title_suffix}")
    plt.legend()
    plt.grid(True)

    # Absolute Difference (CDF)
    plt.subplot(1, 2, 2)
    plt.plot(x, results["cdf_abs_diff"], label="CDF Absolute Diff", marker='.')
    plt.xscale("log")
    plt.yscale("symlog")
    plt.xlabel("x")
    plt.ylabel("Absolute Difference (CDF)")
    plt.title(f"Absolute Difference:\nCDF {title_suffix}")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('plots/comparison {title_suffix}.png'.format(title_suffix=title_suffix))
