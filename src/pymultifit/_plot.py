"""Created on May 06 15:21:29 2026"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from plotez import lpc, plot_xy

from ._plot_backend import _ci, _fit_and_residual, _param_correlation, _plot, _prediction_interval, _qq, _resid


class FitPlotter:
    """Centralized plotting class for fitter objects.

    Uses composition — accepts any fitter instance that exposes the required
    attributes — and an internal strategy pattern to differentiate between
    ``BaseFitter`` subclasses and ``MixedDataFitter``.

    Parameters
    ----------
    fitter :
        A fitted (or pre-fit) fitter instance.  Must expose at minimum the
        attributes listed in ``_REQUIRED_ATTRS``.

    Raises
    ------
    TypeError
        If the supplied object is missing any required attribute.
    """

    _REQUIRED_ATTRS = ("x_values", "y_values", "params", "n_fits", "n_par", "_n_fitter")

    def __init__(self, fitter) -> None:
        missing = [a for a in self._REQUIRED_ATTRS if not hasattr(fitter, a)]
        if missing:
            raise TypeError(f"Fitter is missing required attributes: {missing}")
        self.fitter = fitter

    def _default_param_labels(self) -> list[str]:
        """Auto-generate parameter labels, model-aware for ``MixedDataFitter``.

        Returns
        -------
        list[str]
            Labels like ``["Gaussian_1_p1", "Gaussian_1_p2", "Line_2_p1"]`` for
            mixed fitters, or ``["p1", "p2", ...]`` for single-model fitters.
        """
        if self._is_mixed_fitter():
            labels = []
            fitter = self.fitter
            for i, model in enumerate(fitter.model_list):
                n_par = fitter._instantiate_n_par(model=model)
                for j in range(n_par):
                    labels.append(f"{model.capitalize()}_{i + 1}_p{j + 1}")
            return labels
        return [f"p{i + 1}" for i in range(len(self.fitter.params))]

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

    def _is_mixed_fitter(self) -> bool:
        return hasattr(self.fitter, "model_list")

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

    def _plot_individual_base(self, axis) -> None:
        """Plot individual component fits for a ``BaseFitter`` subclass.

        Parameters
        ----------
        axis :
            Target axes object.
        """
        fitter = self.fitter
        x = fitter.x_values
        params = np.reshape(fitter.params, (fitter.n_fits, fitter.n_par))
        colors = self._get_color_cycle()
        class_name = fitter.__class__.__name__.replace("Fitter", "")

        for i, par in enumerate(params):
            self._plot_component(
                x=x,
                y=fitter.fitter(x=x, params=list(par)),
                label=f"{class_name} {i + 1}({', '.join(self._format_param(v) for v in par)})",
                color=colors[i % len(colors)],
                axis=axis,
            )

    def _plot_individual_fits(self, axis) -> None:
        """Dispatch to the correct strategy based on the fitter type."""
        if self._is_mixed_fitter():
            self._plot_individual_mixed(axis)
        else:
            self._plot_individual_base(axis)

    def _plot_individual_mixed(self, axis) -> None:
        """Plot individual component fits for a ``MixedDataFitter``.

        Parameters
        ----------
        axis :
            Target axes object.
        """
        fitter = self.fitter
        x = fitter.x_values
        colors = self._get_color_cycle()
        param_index = 0

        for i, model in enumerate(fitter.model_list):
            class_model = fitter._instantiate_class(model=model)
            n_par = fitter._instantiate_n_par(model=model)
            pars = fitter.params[param_index : param_index + n_par]
            self._plot_component(
                x=x,
                y=class_model.fitter(x=x, params=list(pars)),
                label=f"{model.capitalize()} {i + 1}({', '.join(self._format_param(v) for v in pars)})",
                color=colors[i % len(colors)],
                axis=axis,
            )
            param_index += n_par

    @staticmethod
    def _resolve_data_labels(data_label: str, fit_label: str) -> tuple[str, str]:
        """Resolve raw-data and fit-line legend labels.

        Parameters
        ----------
        data_label :
            User-supplied data label (empty string means "use default").
        fit_label :
            User-supplied fit label (empty string means "use default").

        Returns
        -------
        tuple[str, str]
            ``(data_label, fit_label)`` with defaults filled in.
        """
        return (data_label or "Data"), (fit_label or "Total fit")

    @staticmethod
    def _unwrap_plotter(plotter) -> Axes:
        return plotter[0] if isinstance(plotter, list) else plotter

    def _validate_fitted(self) -> None:
        if self.fitter.params is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")

    def dry_run(self, axis: Axes | None = None) -> None:
        """Plot raw x / y data for quick inspection before fitting.

        Parameters
        ----------
        axis :
            Target axes.  A new figure is created when ``None``.
        """
        plot_xy(x_data=self.fitter.x_values, y_data=self.fitter.y_values, axis=axis)

    def plot_ci_bounds(
        self,
        results: dict,
        ci_levels: int | list[int],
        overall_ci: bool = True,
        individual_ci: bool = False,
        axis: Axes | None = None,
    ) -> Axes:
        """Plot bootstrap confidence interval bounds.

        Parameters
        ----------
        results :
            Dictionary produced by ``fitter.ci_bounds()`` containing:
            - ``"x_range"``: X-values for CI evaluation
            - ``"overall_ci_95"``: Overall CI dict with "lower", "median", "upper" keys
            - ``"individual_ci_95"``: List of per-fit CI dicts (if individual_ci was used)
        ci_levels :
            CI percentage level(s) to plot (e.g., 95 or [68, 95, 99]).
            Must match levels computed in *results*.
        overall_ci :
            When ``True``, overall composite CI bands are drawn. Defaults to ``True``.
        individual_ci :
            When ``True``, per-component CI bands are drawn. Defaults to ``False``.
        axis :
            Target axes. A new figure is created when ``None``.

        Returns
        -------
        Axes
            The axes on which the plot was drawn.

        Notes
        -----
        Multiple CI levels are rendered with varying transparency (alpha values),
        where narrower intervals appear darker for better visual hierarchy.
        """
        return _ci(
            fitter_object=self.fitter,
            results=results,
            ci_levels=ci_levels,
            overall_ci=overall_ci,
            individual_ci=individual_ci,
            axis=axis,
        )

    def plot_fit(
        self,
        show_individuals: bool = False,
        x_label: str = "X",
        y_label: str = "Y",
        plot_title: str = "Plot",
        data_label: str = "Data",
        fit_label: str = "Total Fit",
        axis: Axes | None = None,
    ) -> Axes:
        """Plot the fitted composite model on top of the raw data.

        Parameters
        ----------
        show_individuals :
            When ``True``, each component is plotted as a dashed line.
        x_label :
            Label for the x-axis.  Defaults to ``"X"``.
        y_label :
            Label for the y-axis.  Defaults to ``"Y"``.
        plot_title :
            Plot title.  Defaults to an auto-generated string.
        data_label :
            Legend label for the raw-data series.
        fit_label :
            Legend label for the total-fit series.
        axis :
            Target axes.  A new figure is created when ``None``.

        Returns
        -------
        Axes :
            The axes on which the plot was drawn.
        """
        return _plot(
            plot_object=self,
            fitter_object=self.fitter,
            show_individuals=show_individuals,
            axis=axis,
            x_label=x_label,
            y_label=y_label,
            plot_title=plot_title,
            data_label=data_label,
            fit_label=fit_label,
        )

    def plot_fit_and_residuals(
        self,
        show_individuals: bool = False,
        x_label: str = "X",
        y_label: str = "Y",
        plot_title: str = "",
        data_label: str = "Data",
        residual_label: str = "Residuals",
        fit_label: str = "Total Fit",
    ) -> tuple[Figure, tuple[Axes, Axes]]:
        """Plot the fitted model and residuals in a two-panel figure.

        Parameters
        ----------
        show_individuals :
            When ``True``, individual components are plotted in the top panel.
        x_label :
            Label for the shared x-axis (shown on the residuals panel).
        y_label :
            Label for the y-axis of the fit panel.
        plot_title :
            Title for the fit panel.
        data_label :
            Forwarded to :meth:`plot_fit`.
        residual_label :
            Labels for the residual plot.
        fit_label :
            Forwarded to :meth:`plot_fit`.

        Returns
        -------
        tuple[plt.Figure, tuple[Axes, Axes]]
            ``(fig, (ax_fit, ax_residuals))``.
        """
        return _fit_and_residual(
            plot_object=self,
            fitter_object=self.fitter,
            show_individuals=show_individuals,
            x_label=x_label,
            y_label=y_label,
            plot_title=plot_title,
            data_label=data_label,
            fit_label=fit_label,
            residual_label=residual_label,
        )

    def plot_parameter_correlation(
        self, param_labels: list[str] | None = None, plot_title="Parameter Correlation Matrix", axis: Axes | None = None
    ) -> Axes:
        """Heatmap of the parameter correlation matrix from the covariance matrix.

        Each cell shows the Pearson correlation between a pair of fitted
        parameters.  Values near ±1 indicate strong linear dependence, which
        may signal over-parameterisation or identifiability issues.

        Parameters
        ----------
        param_labels :
            Custom axis labels.  When ``None``, model-aware names are generated
            automatically (e.g. ``"Gaussian_1_p1"`` for ``MixedDataFitter``,
            or ``"p1", "p2", ...`` for single-model fitters).
        axis :
            Target axes.  A square figure is created when ``None``.

        Returns
        -------
        Axes
            The axes on which the plot was drawn.
        """
        return _param_correlation(
            plot_object=self, fitter_object=self.fitter, param_labels=param_labels, plot_title=plot_title, axis=axis
        )

    def plot_prediction_intervals(self, pi_level: int | list[int] = 95, axis: Axes | None = None, **kwargs) -> Axes:
        """Plot prediction intervals for new individual observations.

        Prediction intervals are **wider** than confidence intervals: they
        estimate the range where a new single observation will fall, not the
        uncertainty in the mean response.

        The interval is computed analytically:

        .. math::

            \\hat{y} \\pm t_{\\alpha/2,\\,n-k} \\cdot \\hat{\\sigma}

        where :math:`\\hat{\\sigma} = \\sqrt{RSS / (n - k)}` is the residual
        standard deviation, :math:`n` is the number of data points, and
        :math:`k` is the total number of fitted parameters.

        Parameters
        ----------
        pi_level :
            Prediction interval level(s) as percentages.  Pass a single integer
            or a list of integers for multiple bands.  Defaults to 95.
        axis :
            Target axes.  A new figure is created when ``None``.

        Returns
        -------
        Axes
            The axes on which the plot was drawn.
        """
        return _prediction_interval(plot_object=self, fitter_object=self.fitter, pi_level=pi_level, axis=axis, **kwargs)

    def plot_qq_plot(self, plot_title: str = "Q-Q plot", axis: Axes | None = None) -> Axes:
        """
        Generates a Q-Q plot for the fitted data.

        Parameters
        ----------
        plot_title :
            The title of the Q-Q plot.
            Defaults to ``Q-Q plot``.
        axis :
            Matplotlib Axes object to use for the Q-Q plot.
            If None, a new Axes object is created.

        Returns
        -------
        Axes
            The Matplotlib Axes object containing the Q-Q plot.
        """
        return _qq(plot_object=self, fitter_object=self.fitter, plot_title=plot_title, axis=axis)

    def plot_residuals(
        self, x_label: str = "X", y_label: str = "Y", plot_title: str = "Residuals", axis: Axes | None = None
    ) -> Axes:
        """Plot residuals (data − fitted model).

        Parameters
        ----------
        x_label :
            Label for the x-axis.  Defaults to ``"X"``.
        y_label :
            Label for the y-axis.  Defaults to ``"Residuals"``.
        plot_title :
            Plot title.  Defaults to an auto-generated string.
        axis :
            Target axes.  A new figure is created when ``None``.

        Returns
        -------
        Axes :
            The axes on which the plot was drawn.
        """
        return _resid(plot_object=self, fitter_object=self.fitter, axis=axis)

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

    # ------------------------------------------------------------------ individual fits (strategy)
