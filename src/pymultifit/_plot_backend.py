"""Created on May 07 09:38:52 2026"""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from plotez import ebc, lpc, plot_errorband, plot_xy, spc
from scipy.stats import norm, pearsonr, t
from statsmodels.graphics.gofplots import ProbPlot

from .typing import NDArray

if TYPE_CHECKING:
    from pymultifit._plot import FitPlotter
    from pymultifit.fitters import MixedDataFitter
    from pymultifit.fitters.backend import BaseFitter

FIT_COLOR = "#000000"

SCATTER_COLOR = "#666666"
SCATTER_SIZE = 15
SCATTER_ALPHA = 0.55

CI_BASE = "#009E73"
PI_BASE = "#E69F00"

QQ_LINE_COLOR = "#FF0000"
QQ_LINE_LS = "--"
QQ_LINE_LW = 1.5

GRID_COLOR = "#000000"
GRID_ALPHA = 0.25
GRID_LS = "--"


def _qq(
    plot_object: "FitPlotter",
    fitter_object: "BaseFitter | MixedDataFitter",
    plot_title: str = "QQ-Plot",
    axis: Axes | None = None,
) -> Axes:
    plot_object._validate_fitted()
    residual = fitter_object.get_residuals()

    if axis is None:
        _, axis = plt.subplots(figsize=(6, 6))

    axis: Axes

    pp = ProbPlot(data=residual, dist=norm, fit=True)
    quantiles = pp.theoretical_quantiles
    values = pp.sample_quantiles

    q25, q75 = np.percentile(values, q=[25, 75])
    t_q25, t_q75 = norm.ppf([0.25, 0.75])
    slope = (q75 - q25) / (t_q75 - t_q25)
    intercept = q25 - slope * t_q25

    r, _ = pearsonr(x=quantiles, y=values)

    plotter = plot_xy(
        x_data=quantiles,
        y_data=values,
        data_label="Residuals",
        is_scatter=True,
        plot_config=spc(s=SCATTER_SIZE, alpha=SCATTER_ALPHA, color=SCATTER_COLOR),
        axis=axis,
    )

    plotter: Axes

    plot_xy(
        x_data=quantiles,
        y_data=slope * quantiles + intercept,
        data_label=f"Normal fit (r = {r:.4f})",
        x_label="Theoretical Quantiles",
        y_label="Sample Quantiles",
        plot_title=plot_title,
        plot_config=lpc(c=QQ_LINE_COLOR, ls=QQ_LINE_LS, lw=QQ_LINE_LW),
        axis=plotter,
    )

    plotter.grid(ls=GRID_LS, alpha=GRID_ALPHA, color=GRID_COLOR)

    return plotter


def _param_correlation(
    plot_object: "FitPlotter",
    fitter_object: "BaseFitter | MixedDataFitter",
    plot_title: str = "Parameter Correlation Matrix",
    param_labels: list[str] | None = None,
    axis: Axes | None = None,
) -> Axes:
    plot_object._validate_fitted()
    params = fitter_object.params
    cov_matrix = fitter_object.covariance

    cov_matrix: NDArray
    params: NDArray

    std = np.sqrt(np.diag(cov_matrix))
    outer = np.outer(std, std)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(outer > 0, cov_matrix / outer, 0.0)
    np.clip(corr, -1.0, 1.0, out=corr)

    n_params = params.shape[0]
    labels = param_labels or plot_object._default_param_labels()
    fig_size = max(4, n_params)

    if axis is None:
        _, axis = plt.subplots(figsize=(fig_size, fig_size))

    axis: Axes

    im = axis.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=axis, label="Correlation")

    axis.set_xticks(range(n_params))
    axis.set_yticks(range(n_params))
    axis.set_xticklabels(labels=labels, rotation=45, ha="right", fontsize=8)
    axis.set_yticklabels(labels=labels, fontsize=8)

    for i in range(n_params):
        for j in range(n_params):
            text_color = "white" if abs(corr[i, j]) > 0.6 else "black"
            axis.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7, color=text_color)

    axis.set_title(plot_title)

    return axis


def _prediction_interval(
    plot_object: "FitPlotter",
    fitter_object: "BaseFitter | MixedDataFitter",
    pi_level: int | list[int] = 95,
    axis: Axes | None = None,
    **kwargs,
) -> Axes:
    plot_object._validate_fitted()
    params = fitter_object.params

    x, params = np.asarray(fitter_object.x_values), np.asarray(params)

    pi_levels = sorted(
        [pi_level] if isinstance(pi_level, int) else list(pi_level), reverse=True  # widest band drawn first
    )

    x_label, y_label, plot_title, _, _ = _resolve_kwargs(
        kwargs=kwargs, n_fits=fitter_object.n_fits, class_name=fitter_object.__class__.__name__
    )

    n, k = len(x), len(params)

    residuals = fitter_object.get_residuals()
    sigma = np.sqrt(np.sum(residuals**2) / max(n - k, 1))
    fitted = fitter_object._n_fitter(fitter_object.x_values, *params)

    if axis is None:
        _, axis = plt.subplots(figsize=(10, 6))

    axis: Axes

    pi_colors = [plt.get_cmap("YlOrBr")(v) for v in np.linspace(0.35, 0.75, len(pi_levels))]

    for pi, col_ in zip(pi_levels, pi_colors):
        alpha_stat = 1 - pi / 100
        t_crit = t.ppf(1 - alpha_stat / 2, df=max(n - k, 1))
        plot_errorband(
            x_data=fitter_object.x_values,
            y_data=fitted,
            y_lower=fitted - t_crit * sigma,
            y_upper=fitted + t_crit * sigma,
            line_config=lpc(c=FIT_COLOR, zorder=10),
            band_config=ebc(c=col_, label=f"{pi}% PI"),
            axis=axis,
        )

    plot_xy(
        x_data=fitter_object.x_values,
        y_data=fitter_object.y_values,
        plot_config=spc(s=SCATTER_SIZE, alpha=SCATTER_ALPHA, c=SCATTER_COLOR),
        is_scatter=True,
        x_label=x_label,
        y_label=y_label,
        plot_title=plot_title,
        axis=axis,
    )

    axis.legend()
    axis.grid(ls=GRID_LS, alpha=GRID_ALPHA, color=GRID_COLOR)

    return axis


def _ci(
    fitter_object: "BaseFitter | MixedDataFitter",
    results: dict,
    ci_levels: float | tuple[float] | list[float],
    overall_ci: bool,
    individual_ci: bool,
    axis: Axes | None,
) -> Axes:
    if axis is None:
        _, axis = plt.subplots(figsize=(10, 6))

    if isinstance(ci_levels, int):
        ci_levels = [ci_levels]

    axis: Axes
    ci_levels: list

    # Extract x_range from results
    x_range = results.get("x_range", fitter_object.x_values)

    # Create alpha values for multiple CI levels (lighter for wider intervals)
    alphas = np.linspace(0.45, 0.15, len(ci_levels))[::-1]  # Reverse so narrower is darker

    def _helper(given_ci: dict, data_label: str, alpha_val: float):
        """Helper to plot a single CI band."""
        plot_errorband(
            x_data=x_range,
            y_data=given_ci["median"],
            y_lower=given_ci["lower"],
            y_upper=given_ci["upper"],
            line=False,
            band_config=ebc(c=CI_BASE, alpha=alpha_val, label=data_label),
            axis=axis,
        )

    if overall_ci:
        for ci, alpha_val in zip(ci_levels, alphas):
            label = f"{ci}% CI (overall)"
            ci_key = f"overall_ci_{ci}"

            if ci_key not in results:
                raise KeyError(f"CI level {ci} not found in results. Available: {list(results.keys())}")

            ci_data = results[ci_key]
            _helper(given_ci=ci_data, data_label=label, alpha_val=alpha_val)

    if individual_ci:
        for ci, alpha_val in zip(ci_levels, alphas):
            ci_key = f"individual_ci_{ci}"

            if ci_key not in results:
                raise KeyError(f"CI level {ci} not found in results. Available: {list(results.keys())}")

            ci_data = results[ci_key]
            for idx, fit_ci in enumerate(ci_data):
                label = f"{ci}% CI (fit {idx + 1})" if idx == 0 else ""  # label first only
                _helper(given_ci=fit_ci, data_label=label, alpha_val=alpha_val)

    axis.grid(color=GRID_COLOR, ls=GRID_LS, alpha=GRID_ALPHA)
    axis.legend()

    return axis


def _fit_and_residual(
    plot_object: "FitPlotter",
    fitter_object: "BaseFitter | MixedDataFitter",
    show_individuals,
    x_label,
    y_label,
    data_label,
    fit_label,
    residual_label,
    plot_title,
) -> tuple[Figure, tuple[Axes, Axes]]:
    plot_object._validate_fitted()

    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    _plot(
        plot_object=plot_object,
        fitter_object=fitter_object,
        show_individuals=show_individuals,
        axis=ax1,
        x_label="",
        y_label=y_label,
        data_label=data_label,
        fit_label=fit_label,
        plot_title=plot_title,
    )

    _resid(
        plot_object=plot_object, fitter_object=fitter_object, axis=ax2, residual_label=residual_label, x_label=x_label
    )

    return fig, (ax1, ax2)


def _resid(
    plot_object: "FitPlotter", fitter_object: "BaseFitter | MixedDataFitter", axis: Axes | None = None, **kwargs
) -> Axes:
    plot_object._validate_fitted()
    x, y = np.asarray(fitter_object.x_values), np.asarray(fitter_object.y_values)

    x_label, y_label, plot_title, data_label, fit_label, residual_label = _resolve_kwargs(
        kwargs, get_residual_label=True
    )

    plotter = plot_xy(
        x_data=x, y_data=fitter_object.get_residuals(), axis=axis, plot_title="", plot_config=lpc(alpha=0.75)
    )

    ax = plot_object._unwrap_plotter(plotter)
    ax.axhline(y=0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(residual_label)
    ax.legend_ = None

    return ax


def _resolve_kwargs(kwargs, n_fits=None, class_name=None, get_residual_label=False):
    x_label = kwargs.get("x_label", "X")
    y_label = kwargs.get("y_label", "Y")
    plot_title = kwargs.get("plot_title", f"{n_fits} {class_name} fit")
    data_label = kwargs.get("data_label", "Data")
    fit_label = kwargs.get("fit_label", "Fit")

    if get_residual_label:
        residual_label = kwargs.get("residual_label", "Residuals")
        return x_label, y_label, plot_title, data_label, fit_label, residual_label

    return x_label, y_label, plot_title, data_label, fit_label


def _plot(
    plot_object: "FitPlotter",
    fitter_object: "BaseFitter | MixedDataFitter",
    show_individuals: bool = False,
    axis: Axes | None = None,
    is_scatter: bool = False,
    **kwargs,
) -> Axes:
    plot_object._validate_fitted()
    x, y = np.asarray(fitter_object.x_values), np.asarray(fitter_object.y_values)

    x_label, y_label, plot_title, data_label, fit_label = _resolve_kwargs(
        kwargs, fitter_object.n_fits, fitter_object.__class__.__name__
    )

    params = fitter_object.params
    dl, tt = plot_object._resolve_data_labels(data_label, fit_label)

    axis = plot_xy(x_data=x, y_data=y, data_label=dl, axis=axis, is_scatter=is_scatter, plot_config=lpc(alpha=0.75))
    # Plot combined fit and/or individual component fits.
    # Behavior: if show_individuals and there's only one model component, draw only the individual fit
    # (which is the same as the combined).
    # Otherwise, draw the combined fit and, when requested, overlay individual fits.
    if show_individuals and fitter_object.n_fits == 1:
        plot_object._plot_individual_fits(axis=axis)
    else:
        # draw combined fit
        plot_xy(
            x_data=x,
            y_data=fitter_object._n_fitter(x, *params),
            x_label=x_label,
            y_label=y_label,
            plot_title=plot_title,
            data_label=tt,
            plot_config=lpc(c="k"),
            axis=axis,
        )
        # optionally overlay individual fits when there are multiple components
        if show_individuals:
            plot_object._plot_individual_fits(axis=axis)

    ax = plot_object._unwrap_plotter(axis)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(plot_title)
    ax.grid(ls="--", alpha=0.25, color="k")

    return ax
