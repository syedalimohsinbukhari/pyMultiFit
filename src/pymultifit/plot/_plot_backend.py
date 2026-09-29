"""Created on May 07 09:38:52 2026"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from plotez import ebc, lpc, plot_errorband, plot_xy, spc
from scipy.stats import norm, pearsonr, t
from statsmodels.graphics.gofplots import ProbPlot

from ..exceptions import AxesError
from ..typing import NDArray

if TYPE_CHECKING:
    from ..fitters import MixedDataFitter
    from ..fitters.backend import BaseFitter
    from . import FitPlotter

FIG_SIZE = (10, 6)

FIT_COLOR = "#000000"

SCATTER_COLOR = "#666666"
SCATTER_SIZE = 15
SCATTER_ALPHA = 0.55

CI_BASE = "#D55E00"
PI_BASE = "#E69F00"

QQ_FIG_SIZE = (6, 6)
QQ_LINE_COLOR = "#FF0000"
QQ_LINE_LS = "--"
QQ_LINE_LW = 1.5

GRID_COLOR = "#000000"
GRID_ALPHA = 0.25
GRID_LS = "--"

RESID_FIG_SIZE = (12, 4)


def _ci_plotter(
    fitter_object: "BaseFitter | MixedDataFitter",
    results: dict,
    ci_levels: float | tuple[float] | list[float],
    overall_ci: bool,
    individual_ci: bool,
    axis: Axes | None,
) -> Axes:
    if isinstance(ci_levels, int):
        ci_levels = [ci_levels]

    ci_levels: list

    # Extract x_range from results
    x_range = results.get("x_range", fitter_object.x_values)

    # Create alpha values for multiple CI levels (lighter for wider intervals)
    alphas = np.linspace(0.45, 0.15, len(ci_levels))

    def _helper(given_ci: dict, data_label: str, alpha_val: float):
        """Helper to plot a single CI band."""
        plot_errorband(
            x_data=x_range,
            y_data=given_ci["median"],
            y_lower=given_ci["lower"],
            y_upper=given_ci["upper"],
            data_label=data_label,
            line=False,
            band_config=ebc(c=CI_BASE, alpha=alpha_val),
            axis=axis,
        )

    axis = _single_axis_sanitizer(axis=axis)

    if overall_ci:
        for ci, alpha_ in zip(ci_levels, alphas):
            label = f"{ci}% CI (overall)"
            ci_key = f"overall_ci_{ci}"

            if ci_key not in results:
                raise KeyError(f"CI level {ci} not found in results. Available: {list(results.keys())}")

            ci_data = results[ci_key]
            _helper(given_ci=ci_data, data_label=label, alpha_val=alpha_)

    if individual_ci:
        for ci, alpha_ in zip(ci_levels, alphas):
            ci_key = f"individual_ci_{ci}"

            if ci_key not in results:
                raise KeyError(f"CI level {ci} not found in results. Available: {list(results.keys())}")

            ci_data = results[ci_key]
            for idx, fit_ci in enumerate(ci_data):
                label = f"{ci}% CI (fit {idx + 1})" if idx == 0 else ""  # label first only
                _helper(given_ci=fit_ci, data_label=label, alpha_val=alpha_)

    axis.legend()
    _grid(axis)

    return axis


def _fit_and_residual(
    plot_object: "FitPlotter",
    show_individuals: bool = False,
    x_label: str = "X",
    y_label: str = "Y",
    data_label: str = "Data",
    fit_label: str = "Total Fit",
    plot_title: str = "Plot",
    is_scatter_plot: bool = False,
    is_scatter_residual: bool = False,
    axes: tuple[Axes, Axes] | None = None,
) -> tuple[Axes, Axes]:
    if axes is None:
        _, (ax1, ax2) = plt.subplots(
            nrows=2, ncols=1, figsize=FIG_SIZE, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
    elif not isinstance(axes, list | tuple) or len(axes) != 2:
        raise AxesError("There must be two axes for fitter and residuals to plot upon.")
    else:
        ax1, ax2 = axes

    _plot(
        plot_object=plot_object,
        show_individuals=show_individuals,
        x_label="",
        y_label=y_label,
        plot_title=plot_title,
        data_label=data_label,
        fit_label=fit_label,
        is_scatter=is_scatter_plot,
        axis=ax1,
    )

    # manually turn off the xticks on the fit plot
    ax1.tick_params(axis="x", bottom=False, labelbottom=False)

    _resid(plot_object=plot_object, x_label=x_label, axis=ax2, data_label="", is_scatter=is_scatter_residual)
    # no need for residual legend when it is used with the fitted plot
    ax2.legend_ = None

    return ax1, ax2


def _grid(axis: Axes):
    axis.grid(ls=GRID_LS, alpha=GRID_ALPHA, color=GRID_COLOR)


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


def _plot(
    plot_object: "FitPlotter",
    show_individuals: bool = False,
    x_label: str = "X",
    y_label: str = "Y",
    plot_title: str = "Plot",
    data_label: str = "Data",
    fit_label: str = "Total Fit",
    is_scatter: bool = False,
    axis: Axes | None = None,
) -> Axes:
    fitter_object = _validate(plot_object)
    x, y = np.asarray(fitter_object.x_values), np.asarray(fitter_object.y_values)

    axis = _single_axis_sanitizer(axis=axis)

    params = fitter_object.params
    dl, tt = (data_label or "Data"), (fit_label or "Total fit")

    # The first catch of axis is necessary, the user might not pass an axis object, so the return from `plot_xy` is
    # required to further working.
    plot_xy(x_data=x, y_data=y, data_label=dl, axis=axis, is_scatter=is_scatter, plot_config=lpc(alpha=0.75))

    # Plot combined fit and/or individual component fits.
    # Behavior: if show_individuals and there's only one model component, draw only the individual fit.
    # Otherwise, draw the combined fit and, when requested, overlay individual fits.
    if show_individuals and fitter_object.n_fits == 1:
        plot_object._plot_individual_fits(axis=axis)
    else:
        # draw combined fit
        plot_xy(x_data=x, y_data=fitter_object._n_fitter(x, *params), data_label=tt, plot_config=lpc(c="k"), axis=axis)
        # optionally overlay individual fits when there are multiple components
        if show_individuals:
            plot_object._plot_individual_fits(axis=axis)

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(plot_title)

    _grid(axis)

    return axis


def _prediction_interval(
    plot_object: "FitPlotter",
    fitter_object: "BaseFitter | MixedDataFitter",
    pi_level: int | list[int] = 95,
    x_label: str = "X",
    y_label: str = "Y",
    plot_title: str = "PI",
    axis: Axes | None = None,
    **kwargs,
) -> Axes:
    plot_object._validate_fitted()
    params = fitter_object.params

    x, params = np.asarray(fitter_object.x_values), np.asarray(params)

    pi_levels = sorted(
        [pi_level] if isinstance(pi_level, int) else list(pi_level), reverse=True  # widest band drawn first
    )

    n, k = len(x), len(params)

    residuals = fitter_object.get_residuals()
    sigma = np.sqrt(np.sum(residuals**2) / max(n - k, 1))
    fitted = fitter_object._n_fitter(fitter_object.x_values, *params)

    axis = _single_axis_sanitizer(axis=axis)

    pi_colors = [plt.get_cmap("YlOrBr")(v) for v in np.linspace(0.35, 0.75, len(pi_levels))]

    for pi, col_ in zip(pi_levels, pi_colors):
        alpha_stat = 1 - pi / 100
        t_crit = t.ppf(1 - alpha_stat / 2, df=max(n - k, 1))
        axis = plot_errorband(
            x_data=fitter_object.x_values,
            y_data=fitted,
            y_lower=fitted - t_crit * sigma,
            y_upper=fitted + t_crit * sigma,
            line_config=lpc(c=FIT_COLOR, zorder=10),
            band_config=ebc(c=col_, label=f"{pi}% PI"),
            axis=axis,
        )

    axis = plot_xy(
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
    _grid(axis)

    return axis


def _obj_resolver(
    plot_object: "FitPlotter | None" = None, fitter_object: "BaseFitter | MixedDataFitter | None" = None
) -> tuple["FitPlotter", "BaseFitter | MixedDataFitter"]:
    if plot_object is None and fitter_object is None:
        raise ValueError("At least one of plot_object or fitter_object must be provided.")

    if plot_object is None:
        assert fitter_object is not None  # guaranteed by the raise above
        plot_object = fitter_object.plotter
    if fitter_object is None:
        assert plot_object is not None  # guaranteed by the raise above
        fitter_object = plot_object.fitter

    return plot_object, fitter_object


def _qq(
    plot_object: "FitPlotter | None" = None,
    fitter_object: "BaseFitter | MixedDataFitter | None" = None,
    plot_title: str = "QQ-Plot",
    axis: Axes | None = None,
) -> Axes:
    plot_object, fitter_object = _obj_resolver(plot_object=plot_object, fitter_object=fitter_object)
    residual = fitter_object.get_residuals()

    pp = ProbPlot(data=residual, dist=norm, fit=True)
    quantiles = pp.theoretical_quantiles
    values = pp.sample_quantiles

    q25, q75 = np.percentile(values, q=[25, 75])
    t_q25, t_q75 = norm.ppf([0.25, 0.75])
    slope = (q75 - q25) / (t_q75 - t_q25)
    intercept = q25 - slope * t_q25

    r, _ = pearsonr(x=quantiles, y=values)

    axis = _single_axis_sanitizer(axis=axis, figsize=QQ_FIG_SIZE)

    plot_xy(
        x_data=quantiles,
        y_data=values,
        data_label="Residuals",
        is_scatter=True,
        plot_config=spc(s=SCATTER_SIZE, alpha=SCATTER_ALPHA, color=SCATTER_COLOR),
        axis=axis,
    )

    plot_xy(
        x_data=quantiles,
        y_data=slope * quantiles + intercept,
        data_label=f"Normal fit (r = {r:.4f})",
        x_label="Theoretical Quantiles",
        y_label="Sample Quantiles",
        plot_title=plot_title,
        plot_config=lpc(c=QQ_LINE_COLOR, ls=QQ_LINE_LS, lw=QQ_LINE_LW),
        axis=axis,
    )

    _grid(axis)

    return axis


def _qq_compare(
    fitter_left: "BaseFitter | MixedDataFitter",
    fitter_right: "BaseFitter | MixedDataFitter",
    label_left: str | None = None,
    label_right: str | None = None,
    plot_title: str = "Q-Q Plot Comparison",
    axes: tuple[Axes, Axes] | None = None,
) -> tuple[Axes, Axes]:
    if axes is None:
        _, (ax_l, ax_r) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
    elif not isinstance(axes, list | tuple) or len(axes) != 2:
        raise AxesError("There must be two axes for fitter and residuals to plot upon.")
    else:
        ax_l, ax_r = axes

    if label_left is None:
        label_left = f"Q-Q plot | {fitter_left.__class__.__name__}"
    if label_right is None:
        label_right = f"Q-Q plot | {fitter_right.__class__.__name__}"

    _qq(fitter_object=fitter_left, plot_title=label_left, axis=ax_l)
    _qq(fitter_object=fitter_right, plot_title=label_right, axis=ax_r)

    # manually mute the right plot for its y-axis label and ticks
    ax_r.tick_params(axis="y", left=False, labelleft=False)
    ax_r.set_ylabel("")

    ax_l.get_figure().suptitle(plot_title)

    return ax_l, ax_r


def _validate(plot_object: "FitPlotter") -> "BaseFitter | MixedDataFitter":
    # validate that the fitter exists and has been fit
    f_obj = plot_object.fitter
    if f_obj.params is None:
        raise RuntimeError("Fit not performed yet. Call fit() first.")

    return f_obj


def _resid(
    plot_object: "FitPlotter",
    x_label: str = "X",
    y_label: str = r"$y - \hat{y}$",
    data_label: str = "Residuals",
    plot_title: str = "",
    is_scatter: bool = False,
    axis: Axes | None = None,
) -> Axes:
    fitter_object = _validate(plot_object)
    x, y = np.asarray(fitter_object.x_values), np.asarray(fitter_object.y_values)

    axis = _single_axis_sanitizer(axis=axis, figsize=RESID_FIG_SIZE)

    plot_xy(
        x_data=x,
        y_data=fitter_object.get_residuals(),
        x_label=x_label,
        y_label=y_label,
        data_label=data_label,
        axis=axis,
        is_scatter=is_scatter,
        plot_title=plot_title,
        plot_config=lpc(alpha=0.75),
    )

    _grid(axis)

    return axis


def _single_axis_sanitizer(axis: Axes | None, figsize: tuple[float, float] = FIG_SIZE) -> Axes:
    if axis is None:
        _, axis = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    return axis
