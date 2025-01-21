"""Created on Dec 31 05:45:40 2024"""

from timeit import default_timer as timer

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FixedLocator

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
    avg_times_class = []
    avg_times_scipy = []

    for n_points in n_points_list:
        x = np.linspace(start=EPSILON, stop=10, num=n_points)

        class_times = []
        for _ in range(repetitions):
            if compute_cdf:
                start_class = timer()
                __ = custom_dist.cdf(x)
                end_class = timer()
            else:
                start_class = timer()
                __ = custom_dist.pdf(x)
                end_class = timer()
            class_times.append(end_class - start_class)
        avg_times_class.append(np.mean(class_times))

        scipy_times = []
        for _ in range(repetitions):
            if compute_cdf:
                start_scipy = timer()
                __ = scipy_dist.cdf(x)
                end_scipy = timer()
            else:
                start_scipy = timer()
                __ = scipy_dist.pdf(x)
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
    plt.plot(n_points_list, mean_c, 'o-', ms=3, label='Custom (Mean)', color='blue')
    plt.plot(n_points_list, mean_s, 's-', ms=3, label='SciPy (Mean)', color='orange')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Number of Points")
    plt.ylabel("Execution Time (s)")
    plt.title(f"Speed Comparison: Custom vs SciPy ({title_suffix})")
    plt.legend()
    plt.grid(True)

    import statsmodels.api as sm

    lowess_results = sm.nonparametric.lowess(ratio_means, n_points_list, frac=0.3)
    x_smooth, y_smooth = lowess_results[:, 0], lowess_results[:, 1]

    plt.subplot(1, 2, 2)
    plt.plot(n_points_list, ratio_means, 'x-', ms=4, label='Ratio (Mean)', color='purple')
    plt.plot(x_smooth, y_smooth, 'r--', lw=2, alpha=0.75, label='LOESS Fit')
    plt.xscale('log')
    plt.xlabel("Number of Points")
    plt.ylabel("Speed Ratio (Custom/SciPy)")
    plt.axhline(y=1, color='k', linestyle=':', label='Ratio = 1')
    plt.title(f"Speed Ratio: Custom/SciPy ({'Example Title'})")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"plots/{save_as}_{title_suffix}.png")


def cdf_pdf_plots(custom_dist, scipy_dist, n_points, save_as: str, repetitions: int = 15):
    p_times_class, p_times_scipy = evaluate_speed(custom_dist=custom_dist, scipy_dist=scipy_dist,
                                                  n_points_list=n_points, compute_cdf=False, repetitions=repetitions)
    plot_speed_and_ratios(n_points_list=n_points, times_class=p_times_class, times_scipy=p_times_scipy,
                          title_suffix="PDF Computations", save_as=save_as)

    c_times_class, c_times_scipy = evaluate_speed(custom_dist=custom_dist, scipy_dist=scipy_dist,
                                                  n_points_list=n_points, compute_cdf=False, repetitions=repetitions)
    plot_speed_and_ratios(n_points_list=n_points, times_class=c_times_class, times_scipy=c_times_scipy,
                          title_suffix="CDF Computations", save_as=save_as)

    return (p_times_class, c_times_class), (p_times_scipy, c_times_scipy)


def plot_distribution_comparison(data_dict, title_labels=('PDF', 'CDF')):
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
            xtick_labels.append(f'custom\n{dist_name}')
            xtick_labels.append(f'scipy\n{dist_name}')

        ax[i].boxplot(log_data, meanline=True, showmeans=True)
        ax[i].set_xticklabels(xtick_labels, rotation=60, ha="center")
        ax[i].set_title(title)

    plt.xlabel('Log[Time] [s]')
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


def describe_data(data_list, labels=None, caption='PDF'):
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

    for idx, data in enumerate(data_list):
        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        summary_stats = {'N': data.size,
                         'Mean': data.mean(),
                         'Std': data.std(),
                         'Min': data.min(),
                         'Q1 (25%)': data.quantile(0.25),
                         'Q2 (Median)': data.median(),
                         'Q3 (75%)': data.quantile(0.75),
                         'Max': data.max()}

        summary_list.append(summary_stats)

    index_labels = labels if labels else [f"Dataset {i + 1}" for i in range(len(data_list))]
    summary_df = pd.DataFrame(data=summary_list, index=index_labels)

    styled_summary = summary_df.style.set_caption(f"{caption} Statistics") \
        .format({col: "{:.3E}" for col in summary_df.columns[1:]}) \
        .set_table_styles([{'selector': 'th',
                            'props': [('font-size', '12pt'),
                                      ('text-align', 'center')]}])

    return styled_summary


def boxplot_comparison(df1, df2, label, fig_size=(16, 6)):
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)

    pd.plotting.boxplot(data=np.log10(df1 / df2), ax=ax)
    ax.axhline(y=np.log10(1), color='r', ls='--')
    f.suptitle(f'Time ratio comparison between custom/scipy {label}')
    ticks = [i.get_text() for i in ax.get_yticklabels()]
    float_list = [float(s.replace("−", "-")) for s in ticks]
    ax.yaxis.set_major_locator(FixedLocator(ax.get_yticks()))
    ax.set_yticklabels([round(10**i, 3) for i in float_list])

    plt.tight_layout()
    plt.show()


def heatmap(m_df, s_df, label='PDF'):
    raw_ratios = m_df / s_df
    raw_ratios.index = raw_ratios.index + 1

    v_min = raw_ratios.min().min()
    v_max = raw_ratios.max().max()
    norm = TwoSlopeNorm(vcenter=1, vmin=v_min, vmax=v_max)

    plt.figure(figsize=(16, 6))
    sns.heatmap(data=raw_ratios.T, cmap='RdYlGn_r', annot=False, cbar_kws={'label': 'multifit/scipy'},
                yticklabels=s_df.columns, robust=True, lw=1, linecolor='k', norm=norm)
    plt.title(f'Heatmap of execution time ratio for {label} evaluations')
    plt.ylabel('Distributions')
    plt.xticks(rotation=0, ha='center')
    plt.tight_layout()
    plt.show()
