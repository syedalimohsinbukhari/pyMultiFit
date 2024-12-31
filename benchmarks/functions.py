"""Created on Dec 31 05:45:40 2024"""

from timeit import default_timer as timer

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

    plt.subplot(1, 2, 1)
    plt.plot(x, results["pdf_abs_diff"], label="PDF Absolute Diff", marker='.')
    plt.xscale("log")
    plt.yscale("symlog")
    plt.xlabel("x")
    plt.ylabel("Absolute Difference (PDF)")
    plt.title(f"Absolute Difference:\nPDF {title_suffix}")
    plt.legend()
    plt.grid(True)

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
    plt.savefig('plots/{title_suffix}.png'.format(title_suffix=title_suffix))


#####################################################################################################################################################
# SPEED
#####################################################################################################################################################


def evaluate_speed(custom_dist, scipy_dist, n_points_list, compute_cdf=False, repetitions=100):
    """
    Evaluate and compare the speed of custom and SciPy distributions for PDF/CDF computations.

    Parameters:
    - custom_dist: object
        Custom distribution instance.
    - scipy_dist: object
        SciPy distribution instance.
    - n_points_list: list
        List of number of points to evaluate (used for scalability testing).
    - compute_cdf: bool
        Whether to test the speed of CDF (True) or PDF (False) computations.
    - repetitions: int
        Number of repetitions to average the execution times.

    Returns:
    - avg_times_class: list of average execution times for the custom distribution.
    - avg_times_scipy: list of average execution times for the SciPy distribution.
    """
    avg_times_class = []
    avg_times_scipy = []

    for n_points in n_points_list:
        x = np.linspace(1e-10, 10, n_points)

        # Timing for custom distribution
        class_times = []
        for _ in range(repetitions):
            start_class = timer()
            _ = custom_dist.cdf(x) if compute_cdf else custom_dist.pdf(x)
            end_class = timer()
            class_times.append(end_class - start_class)
        avg_times_class.append(np.mean(class_times))

        # Timing for SciPy distribution
        scipy_times = []
        for _ in range(repetitions):
            start_scipy = timer()
            _ = scipy_dist.cdf(x) if compute_cdf else scipy_dist.pdf(x)
            end_scipy = timer()
            scipy_times.append(end_scipy - start_scipy)
        avg_times_scipy.append(np.mean(scipy_times))

    return avg_times_class, avg_times_scipy


def plot_speed_and_ratios(n_points_list, times_class, times_scipy, title_suffix, save_as="speed_comparison"):
    n_points_list = np.array(n_points_list)

    mean_c = np.array([np.mean(i) for i in times_class])
    mean_s = np.array([np.mean(i) for i in times_scipy])

    ratio_means = mean_c / mean_s

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(n_points_list, mean_c, 'o-', label='Custom (Mean)', color='blue')
    plt.plot(n_points_list, mean_s, 's-', label='SciPy (Mean)', color='orange')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Number of Points")
    plt.ylabel("Execution Time (s)")
    plt.title(f"Speed Comparison: Custom vs SciPy ({title_suffix})")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(n_points_list, ratio_means, 'x-', label='Ratio (Mean)', color='purple')
    plt.xscale('log')
    plt.xlabel("Number of Points")
    plt.ylabel("Speed Ratio (Custom/SciPy)")
    plt.axhline(y=1, color='r', linestyle='--', label='Ratio = 1')
    plt.title(f"Speed Ratio: Custom/SciPy ({title_suffix})")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"plots/{save_as}_{title_suffix}.png")
