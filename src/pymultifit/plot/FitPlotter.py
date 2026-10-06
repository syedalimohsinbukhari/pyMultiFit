"""Created on May 06 15:21:29 2026"""

from __future__ import annotations

from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from plotez import lpc, plot_xy

from ..ci import compute_ci_bounds
from ..result import FitResult
from ._plot_backend import _ci_plotter, _fit_and_residual, _param_correlation, _plot, _prediction_interval, _qq, _resid


class FitPlotter:
    """Centralized plotting class for fits.

    Parameters
    ----------
    result :
        A :class:`~pymultifit.result.FitResult`, typically obtained from ``fitter.to_result()``.
        It may be a pre-fit result, in which case only :meth:`dry_run` is usable.
    """

    def __init__(self, result: FitResult) -> None:
        self.result = result

    @staticmethod
    def _format_param(value, t_low: float = 0.001, t_high: float = 10_000.0) -> str:
        """Format a parameter value with adaptive scientific / fixed notation.

        Parameters
        ----------
        value :
            Numeric parameter value.
        t_low :
            Threshold below which scientific notation is used. Defaults to 0.001.
        t_high :
            Threshold above which scientific notation is used. Defaults to 10 000.

        Returns
        -------
        str
            Formatted string.
        """
        return f"{value:.3E}" if t_high < abs(value) or abs(value) < t_low else f"{value:.3f}"

    @staticmethod
    def _get_color_cycle() -> list:
        """Return the active matplotlib color cycle, skipping the first color."""
        return plt.rcParams["axes.prop_cycle"].by_key()["color"][1:]

    @staticmethod
    def _plot_component(x, y, label: str, color: str, axis) -> None:
        """Render a single model component on *axis* as a dashed line.

        Parameters
        ----------
        x :
            x-data array.
        y :
            y-data array (evaluated component).
        label :
            Legend label for this component.
        color :
            Line colour.
        axis :
            Target axes object.
        """
        plot_xy(
            x_data=x,
            y_data=y,
            x_label="",
            y_label="",
            plot_title="",
            data_label=label,
            plot_config=lpc(ls="--", c=color),
            axis=axis,
        )

    def _plot_individual_fits(self, axis) -> None:
        """Plot every component of the model as a dashed line on *axis*.

        Parameters
        ----------
        axis :
            Target axes object.
        """
        result = self.result
        colors = self._get_color_cycle()

        for i, (comp, pars) in enumerate(result.split(result.params)):
            self._plot_component(
                x=result.x,
                y=comp.func(result.x, list(pars)),
                label=f"{comp.label} {i + 1}({', '.join(self._format_param(v) for v in pars)})",
                color=colors[i % len(colors)],
                axis=axis,
            )

    @staticmethod
    def _unwrap_plotter(plotter) -> Axes:
        return plotter[0] if isinstance(plotter, list) else plotter

    def _validate_fitted(self) -> FitResult:
        self.result.require_fit()
        return self.result

    def dry_run(self, axis: Axes | None = None, is_scatter: bool = False) -> None:
        """Plot raw x / y data for quick inspection before fitting.

        Parameters
        ----------
        axis :
            Target axes.  A new figure is created when ``None``.
        is_scatter :
            When ``True``, the raw data is plotted as a scatter plot instead of a line
        """
        axis = plot_xy(x_data=self.result.x, y_data=self.result.y, axis=axis, is_scatter=is_scatter)
        axis.get_figure().tight_layout()

    def plot_confidence_intervals(
        self,
        ci_levels: float | tuple[float] | list[float],
        results: dict | None = None,
        n_bootstrap: int = 5_000,
        overall_ci: bool = True,
        individual_ci: bool = False,
        seed: int | None = None,
        rng_engine=None,
        x_range=None,
        get_value: bool = False,
        axis: Axes | None = None,
    ) -> tuple[dict, Axes] | Axes:
        """
        Plot bootstrap confidence interval bounds.

        Parameters
        ----------
        ci_levels :
            CI level(s) to plot, as percentages or decimals (e.g., 95, 0.95 or [68, 95, 99]).
        results :
            Pre-computed CI dictionary returned by :meth:`~pymultifit.fitters.backend.baseFitter.BaseFitter.confidence_intervals`.
            When None, the CI is computed internally, defaults to None.
        n_bootstrap :
            Number of bootstrap samples to use when computing the CI.
            Ignored if the keyword `results` is provided, defaults to 5,000.
        overall_ci :
            Whether to draw the overall composite CI band, defaults to True.
        individual_ci :
            Whether to draw per-component CI bands, defaults to False.
        seed :
            Random seed for reproducibility (mutually exclusive with *rng_engine*).
            Defaults to None.
        rng_engine :
            Instance of numpy random Generator (mutually exclusive with *seed*).
            Defaults to None.
        x_range :
            X-values at which to evaluate the CI.
            When None, 1,000 evenly spaced points are used.
        get_value :
            Whether to return the computed CI dictionary along with the axis for further inspection, defaults to False.
        axis :
            The matplotlib axes object on which the plot is to be drawn.
            If None, an axis object is generated and returned. Defaults to None.

        Returns
        -------
        tuple of (dict, matplotlib.axes._axes.Axes) or matplotlib.axes._axes.Axes
            If `get_value` is True, a tuple containing the CI dictionary and the axis is returned; otherwise,
            only the axis is returned.
        """
        if results is None:
            results = compute_ci_bounds(
                result=self.result,
                ci_levels=ci_levels,
                n_bootstrap=n_bootstrap,
                overall_ci=overall_ci,
                individual_ci=individual_ci,
                seed=seed,
                rng_engine=rng_engine,
                x_range=x_range,
            )

        axis = _ci_plotter(
            result=self.result,
            results=results,
            ci_levels=ci_levels,
            overall_ci=overall_ci,
            individual_ci=individual_ci,
            axis=axis,
        )

        if get_value:
            return results, axis
        else:
            return axis

    def plot_fit(
        self,
        show_individuals: bool = False,
        x_label: str = "X",
        y_label: str = "Y",
        plot_title: str = "Plot",
        data_label: str = "Data",
        fit_label: str = "Total Fit",
        is_scatter: bool = False,
        axis: Axes | None = None,
    ) -> Axes:
        """Plot the fitted composite model on top of the raw data.

        Parameters
        ----------
        show_individuals :
            When True, each component is plotted separately.
        x_label :
            The x-axis label, defaults to "X".
        y_label :
            The y-axis label, defaults to "Y".
        plot_title :
            The title for the plot, defaults to "Plot".
        data_label :
            The label for the plotted data, defaults to "Data".
        fit_label :
            The label for the fitted curve, defaults to "Total Fit".
        is_scatter :
            When True, the raw data is plotted as a scatter plot instead of a line, defaults to False.
        axis :
            The matplotlib axis object on which the plot is to be drawn.
            If None, an axis object is generated and returned, defaults to None.

        Returns
        -------
        Axes :
            The matplotlib axis object on which the plot was drawn.
        """
        return _plot(
            plot_object=self,
            show_individuals=show_individuals,
            x_label=x_label,
            y_label=y_label,
            plot_title=plot_title,
            data_label=data_label,
            fit_label=fit_label,
            is_scatter=is_scatter,
            axis=axis,
        )

    def plot_fit_and_residuals(
        self,
        show_individuals: bool = False,
        x_label: str = "X",
        y_label: str = "Y",
        plot_title: str = "Fit and Residuals",
        data_label: str = "Data",
        fit_label: str = "Total Fit",
        is_scatter: tuple[bool, bool] = (False, False),
        axes: tuple[Axes, Axes] | None = None,
    ) -> tuple[Axes, Axes]:
        """Plot the fitted model and residuals in a two-panel figure.

        Parameters
        ----------
        show_individuals :
            When True, each component is plotted separately.
        x_label :
            The x-axis label, defaults to "X".
        y_label :
            The y-axis label, defaults to "Y".
        plot_title :
            The title for the figure, defaults to "Fit and Residuals".
        data_label :
            The label for the plotted data, defaults to "Data".
        fit_label :
            The label for the fitted curve, defaults to "Total Fit".
        is_scatter :
            When True, the raw data is plotted as a scatter plot instead of a line.
            The tuple is shared with both fit plot and residual plots individually, both defaults to False.
        axes :
            The matplotlib axis objects on which the fit and residuals are to be drawn.
            If None, a 1x2 subplot will be generated with 3:1 height for fit and residuals.

        Returns
        -------
        tuple[Axis, Axis]
            The set of axes on which the figure and residuals were drawn.
        """
        is_scatter_plot, is_scatter_residual = is_scatter
        return _fit_and_residual(
            plot_object=self,
            show_individuals=show_individuals,
            x_label=x_label,
            y_label=y_label,
            data_label=data_label,
            fit_label=fit_label,
            plot_title=plot_title,
            is_scatter_plot=is_scatter_plot,
            is_scatter_residual=is_scatter_residual,
            axes=axes,
        )

    def plot_parameter_correlation(
        self,
        param_labels: list[str] | None = None,
        plot_title: str = "Parameter Correlation Matrix",
        axis: Axes | None = None,
    ) -> Axes:
        """Heatmap of the parameter correlation matrix from the covariance matrix.

        Parameters
        ----------
        param_labels :
            Custom axis labels.
            When ``None``, model-aware names are generated automatically.
        plot_title :
            The title of the correlation plot.
        axis :
            The matplotlib axes object on which the plot is to be drawn.
            If None, an axis object is generated and returned. Defaults to None.

        Returns
        -------
        Axes :
            The axes on which the plot was drawn.
        """
        return _param_correlation(result=self.result, param_labels=param_labels, plot_title=plot_title, axis=axis)

    def plot_prediction_intervals(
        self,
        pi_level: float | list[float] = 95,
        x_label: str | None = None,
        y_label: str | None = None,
        plot_title: str | None = None,
        axis: Axes | None = None,
    ) -> Axes:
        r"""Plot prediction intervals for new individual observations.

        Notes
        -----
        The interval is computed analytically:

        .. math::

            \hat{y} \pm t_{(\alpha/2,\,n-k)} \cdot \hat{\sigma}

        where :math:`\hat{\sigma} = \sqrt{RSS / (n - k)}` is the residual standard deviation, :math:`n` is the
        number of data points, and :math:`k` is the total number of fitted parameters.

        Parameters
        ----------
        pi_level :
            Prediction interval level(s), as percentages (``95``, ``95.0``) or decimals (``0.95``).
            Pass a single level or a list of levels for multiple bands, defaults to 95.
        x_label :
            The x-axis label. If ``None``, the axis' existing label is kept, or "X" for a new axis.
        y_label :
            The y-axis label. If ``None``, the axis' existing label is kept, or "Y" for a new axis.
        plot_title :
            The title of the plot. If ``None``, the axis' existing title is kept, or "PI" for a new axis.
        axis :
            The matplotlib axis object on which the plot is to be drawn.
            If None, an axis object is generated and returned, defaults to None.

        Returns
        -------
        Axes :
            The matplotlib axis object on which the plot was drawn.
        """
        return _prediction_interval(
            result=self.result,
            pi_level=pi_level,
            x_label=x_label,
            y_label=y_label,
            plot_title=plot_title,
            axis=axis,
        )

    def plot_qq_plot(self, plot_title: str = "Q-Q plot", axis: Axes | None = None) -> Axes:
        """
        Generates a Q-Q plot for the fitted data.

        Parameters
        ----------
        plot_title :
            The title of the Q-Q plot, defaults to "Q-Q plot".
        axis :
            Matplotlib Axes object to use for the Q-Q plot.
            If None, a new Axes object is created.

        Returns
        -------
        Axes
            The Matplotlib Axes object containing the Q-Q plot.
        """
        return _qq(result=self.result, plot_title=plot_title, axis=axis)

    def plot_residuals(
        self,
        x_label: str = "X",
        y_label: str = "Y",
        data_label: str = "Residuals",
        plot_title: str = "",
        is_scatter: bool = False,
        axis: Axes | None = None,
    ) -> Axes:
        """Plot residuals (data − fitted model).

        Parameters
        ----------
        x_label :
            Label for the x-axis, defaults to "X".
        y_label :
            Label for the y-axis, defaults to "Y".
        plot_title :
            Residual plot title, defaults to "".
        data_label :
            Data label for the residuals, defaults to "Residuals".
        is_scatter :
            When True, the raw data is plotted as a scatter plot instead of a line, defaults to False.
        axis :
            The matplotlib axis object on which the plot is to be drawn.
            If None, an axis object is generated and returned, defaults to None.

        Returns
        -------
        Axes :
            The matplotlib axis object on which the plot was drawn.
        """
        return _resid(
            plot_object=self,
            x_label=x_label,
            y_label=y_label,
            data_label=data_label,
            plot_title=plot_title,
            is_scatter=is_scatter,
            axis=axis,
        )

    @staticmethod
    def save_plot(filename: str, figure: plt.Figure | None = None, dpi: int = 150, **kwargs) -> str:
        """Save the current (or provided) figure with format auto-detection.

        The output format is inferred from the file extension.  When the path
        carries no extension, ``.png`` is appended automatically.

        Parameters
        ----------
        filename :
            Destination path.  The extension determines the format.
            Supported: ``png``, ``pdf``, ``svg``, ``eps``,
            ``jpg`` / ``jpeg``, ``tiff`` / ``tif``.
        figure :
            Figure to save.  Falls back to ``plt.gcf()`` when ``None``.
        dpi :
            Resolution in dots per inch.  Defaults to 150.
        **kwargs :
            Forwarded directly to :func:`matplotlib.figure.Figure.savefig`.

        Returns
        -------
        str
            Resolved path of the saved file.

        Raises
        ------
        ValueError
            If the extension is not in the supported set.
        """
        _supported = {"png", "pdf", "svg", "eps", "jpg", "jpeg", "tiff", "tif"}

        path = Path(filename)
        ext = path.suffix.lower().lstrip(".")

        if not ext:
            ext = "png"
            path = path.with_suffix(".png")

        if ext not in _supported:
            raise ValueError(f"Unsupported format '.{ext}'. Supported formats: {sorted(_supported)}")

        fig = figure or plt.gcf()
        fig.savefig(path, format=ext, dpi=dpi, bbox_inches="tight", **kwargs)
        return str(path)
