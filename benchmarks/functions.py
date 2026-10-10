"""Created on Dec 31 05:45:40 2024"""

import os
import re
import time
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FixedLocator

from pymultifit import EPSILON

# ``run_benchmarks.py --smoke`` sets BENCH_SMOKE=1: the same code path with 2 repetitions instead of 15
DEFAULT_REPETITIONS = 2 if os.environ.get("BENCH_SMOKE") == "1" else 15


def slugify(text: str) -> str:
    """Lower-case snake_case file name part; ``-3`` becomes ``m3`` and ``2.2`` becomes ``2p2`` so no parameter is lost."""
    text = re.sub(r"(?<![A-Za-z0-9])-(?=\d)", "m", text)
    text = re.sub(r"(?<=\d)\.(?=\d)", "p", text)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


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

    with np.errstate(invalid="ignore"):
        pdf_abs_diff = np.nan_to_num(np.abs(pdf_custom - pdf_scipy), False, 0) + EPSILON
        cdf_abs_diff = np.nan_to_num(np.abs(cdf_custom - cdf_scipy), False, 0) + EPSILON

    logpdf_custom = custom_dist.logpdf(x)
    logcdf_custom = custom_dist.logcdf(x)
    logpdf_scipy = scipy_dist.logpdf(x)
    logcdf_scipy = scipy_dist.logcdf(x)
    with np.errstate(invalid="ignore"):
        logpdf_abs_diff = np.nan_to_num(np.abs(logpdf_custom - logpdf_scipy), False, 0) + EPSILON
        logcdf_abs_diff = np.nan_to_num(np.abs(logcdf_custom - logcdf_scipy), False, 0) + EPSILON

    return {
        "pdf_abs_diff": pdf_abs_diff,
        "log_pdf_abs_diff": logpdf_abs_diff,
        "cdf_abs_diff": cdf_abs_diff,
        "log_cdf_abs_diff": logcdf_abs_diff,
    }


# Plotting Function
def plot_accuracy(x, results, title_suffix):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(x, results["pdf_abs_diff"], label="PDF Absolute Diff", marker=".")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("x")
    plt.ylabel("Absolute Difference (PDF)")
    plt.title(f"Absolute Difference:\nPDF {title_suffix}")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(x, results["log_pdf_abs_diff"], label="log PDF Absolute Diff", marker=".")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("x")
    plt.ylabel("Absolute Difference (log PDF)")
    plt.title(f"Absolute Difference:\nlog PDF {title_suffix}")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(x, results["cdf_abs_diff"], label="CDF Absolute Diff", marker=".")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("x")
    plt.ylabel("Absolute Difference (CDF)")
    plt.title(f"Absolute Difference:\nCDF {title_suffix}")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(x, results["log_cdf_abs_diff"], label="log CDF Absolute Diff", marker=".")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("x")
    plt.ylabel("Absolute Difference (log CDF)")
    plt.title(f"Absolute Difference:\nlog CDF {title_suffix}")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()


#####################################################################################################################################################
# SPEED
#####################################################################################################################################################


def _median_time(func, x, repetitions, warmup):
    for _ in range(warmup):
        func(x)

    times = []
    for _ in range(repetitions):
        start = timer()
        func(x)
        times.append(timer() - start)
    return np.median(times)


def evaluate_speed(custom_dist, scipy_dist, n_points_list, compute_cdf=False, repetitions=100, warmup=3):
    """Median runtime per number of points for the custom and the scipy distribution (``warmup`` untimed calls first)."""
    method = "cdf" if compute_cdf else "pdf"
    median_times_class = []
    median_times_scipy = []

    for n_points in n_points_list:
        x = np.linspace(start=EPSILON, stop=10, num=n_points)
        median_times_class.append(_median_time(getattr(custom_dist, method), x, repetitions, warmup))
        median_times_scipy.append(_median_time(getattr(scipy_dist, method), x, repetitions, warmup))

    return median_times_class, median_times_scipy


def plot_speed_and_ratios(n_points_list, times_class, times_scipy, function, label):
    """Plot the timings and their ratio; ``function`` is ``"PDF"`` or ``"CDF"``, ``label`` names the distribution."""
    n_points_list = np.array(n_points_list)

    time_c = np.asarray(times_class, dtype=float)  # each entry is already a median over the repetitions
    time_s = np.asarray(times_scipy, dtype=float)

    ratio = time_c / time_s

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(n_points_list, time_c, "o-", ms=3, label="Custom (median)", color="blue")
    plt.plot(n_points_list, time_s, "s-", ms=3, label="SciPy (median)", color="orange")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of Points")
    plt.ylabel("Execution Time (s)")
    plt.title(f"{label}: {function} speed, custom vs SciPy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(n_points_list, ratio, "x-", ms=4, label="Ratio (median)", color="purple")
    plt.xscale("log")
    plt.xlabel("Number of Points")
    plt.ylabel("Speed Ratio (Custom/SciPy)")
    plt.axhline(y=1, color="k", linestyle=":", label="Ratio = 1")
    plt.title(f"{label}: {function} speed ratio, custom/SciPy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    Path("plots/speed").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"plots/speed/{slugify(label)}_{function.lower()}.png")


def _progress(message: str):
    """Append a line to the file named by BENCH_PROGRESS (set by ``run_benchmarks.py``); the notebook itself prints nothing."""
    path = os.environ.get("BENCH_PROGRESS")
    if path:
        with open(path, "a") as handle:
            handle.write(f"{time.strftime('%H:%M:%S')} {message}\n")


def cdf_pdf_plots(custom_dist, scipy_dist, n_points, save_as: str, repetitions: int = DEFAULT_REPETITIONS):
    start = timer()
    p_times_class, p_times_scipy = evaluate_speed(custom_dist, scipy_dist, n_points, False, repetitions)
    plot_speed_and_ratios(n_points, p_times_class, p_times_scipy, "PDF", save_as)
    _progress(f"{save_as}: PDF done ({timer() - start:.0f} s)")

    start = timer()
    c_times_class, c_times_scipy = evaluate_speed(custom_dist, scipy_dist, n_points, True, repetitions)
    plot_speed_and_ratios(n_points, c_times_class, c_times_scipy, "CDF", save_as)
    _progress(f"{save_as}: CDF done ({timer() - start:.0f} s)")

    return (p_times_class, c_times_class), (p_times_scipy, c_times_scipy)


def plot_distribution_comparison(data_dict, title_labels=("PDF", "CDF")):
    """
    Automates the plotting of PDF and CDF timing comparisons for multiple distributions.

    Parameters:
    - data_dict: Dictionary with keys as distribution names and values as tuples
                 [(custom_PDF, scipy_PDF), (custom_CDF, scipy_CDF)]
    - title_labels: List of subplot titles (e.g., ['PDF', 'CDF'])

    Example usage:
    plot_distribution_comparison({
        'Exp': ([pdf_custom_exp, pdf_scipy_exp], [cdf_custom_exp, cdf_scipy_exp]),
        'Unif': ([pdf_custom_unif, pdf_scipy_unif], [cdf_custom_unif, cdf_scipy_unif]),
        'Laplace': ([pdf_custom_laplace, pdf_scipy_laplace], [cdf_custom_laplace, cdf_scipy_laplace])
    })
    """

    f, ax = plt.subplots(nrows=len(title_labels), ncols=1, figsize=(18, 6 * len(title_labels)))

    if len(title_labels) == 1:
        ax = [ax]

    for i, title in enumerate(title_labels):
        log_data = []
        xtick_labels = []

        for dist_name, (pdf_cdf_pair) in data_dict.items():
            pdf_or_cdf = pdf_cdf_pair[i]
            log_data.append(np.log10(pdf_or_cdf[0]))
            log_data.append(np.log10(pdf_or_cdf[1]))
            xtick_labels.append(f"custom\n{dist_name}")
            xtick_labels.append(f"scipy\n{dist_name}")

        ax[i].boxplot(log_data, meanline=True, showmeans=True)
        ax[i].set_xticklabels(xtick_labels, rotation=60, ha="center")
        ax[i].set_title(title)
        ax[i].set_ylabel("Log[Time] [s]")

    plt.tight_layout()
    plt.show()


def generate_data_dict(data_list, label_list):
    """
    Generates a dictionary mapping distribution names to corresponding PDF and CDF timing data.

    Parameters:
    - data_list: List of tuples, where each tuple contains (custom, scipy) timing data.
                 Example: [(m_norm1, s_norm1), (m_asin1, s_asin1), ...]
    - label_list: List of distribution labels corresponding to each tuple in data_list.

    Returns:
    - A dictionary structured like:
        {
            'Distribution_Name': ([custom_PDF, scipy_PDF], [custom_CDF, scipy_CDF])
        }
    """
    data_dict = {}
    for i, label in enumerate(label_list):
        custom, scipy = data_list[i]
        data_dict[label] = ([custom[0], scipy[0]], [custom[1], scipy[1]])

    return data_dict


def describe_data(data_list, labels=None, caption="PDF"):
    """
    Provides a detailed summary of the data including mean, std, quartiles (Q1, Q2, Q3, Q4), and median
    for multiple datasets.

    Parameters:
    - data_list: A list of lists, numpy arrays, or pandas Series of numerical values.
    - labels: A list of labels corresponding to each dataset (optional).

    Returns:
    - A styled pandas DataFrame with formatted summary statistics for all datasets.
    """

    summary_list = []

    for data in data_list:
        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        summary_stats = {
            "N": data.size,
            "Mean": data.mean(),
            "Std": data.std(),
            "Min": data.min(),
            "Q1 (25%)": data.quantile(0.25),
            "Q2 (Median)": data.median(),
            "Q3 (75%)": data.quantile(0.75),
            "Max": data.max(),
        }

        summary_list.append(summary_stats)

    index_labels = labels if labels else [f"Dataset {i + 1}" for i in range(len(data_list))]
    summary_df = pd.DataFrame(data=summary_list, index=index_labels)

    styled_summary = (
        summary_df.style.set_caption(f"{caption} Statistics")
        .format({col: "{:.3E}" for col in summary_df.columns[1:]})
        .set_table_styles([{"selector": "th", "props": [("font-size", "12pt"), ("text-align", "center")]}])
    )

    return styled_summary


def boxplot_comparison(df1, df2, label, fig_size=(16, 6)):
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)

    pd.plotting.boxplot(data=np.log10(df1 / df2), ax=ax)
    ax.axhline(y=np.log10(1), color="r", ls="--")
    f.suptitle(f"Time ratio comparison between custom/scipy {label}")
    ticks = [i.get_text() for i in ax.get_yticklabels()]
    float_list = [float(s.replace("−", "-")) for s in ticks]
    ax.yaxis.set_major_locator(FixedLocator(ax.get_yticks()))
    ax.set_yticklabels([round(10**i, 3) for i in float_list])

    plt.tight_layout()
    plt.show()


def heatmap(m_df, s_df, label="PDF"):
    raw_ratios = m_df / s_df
    raw_ratios.index = raw_ratios.index + 1

    v_min = min(raw_ratios.min().min(), 1 - 1e-6)  # TwoSlopeNorm needs vmin < 1 < vmax, even when every ratio is below (or above) 1
    v_max = max(raw_ratios.max().max(), 1 + 1e-6)
    norm = TwoSlopeNorm(vcenter=1, vmin=v_min, vmax=v_max)

    plt.figure(figsize=(16, 6))
    sns.heatmap(
        data=raw_ratios.T,
        cmap="RdYlGn_r",
        annot=False,
        cbar_kws={"label": "multifit/scipy"},
        yticklabels=s_df.columns,
        robust=True,
        lw=1,
        linecolor="k",
        norm=norm,
    )
    plt.title(f"Heatmap of execution time ratio for {label} evaluations")
    plt.ylabel("Distributions")
    plt.xticks(rotation=0, ha="center")
    plt.tight_layout()
    plt.show()
