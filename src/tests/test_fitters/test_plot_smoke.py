"""Created on Oct 06 2026

Smoke tests for the plotting API (``FitPlotter`` and the fitter plot methods), plus regressions for bugs found in the
plot sweep: confidence / prediction interval levels given as floats, overlays overwriting axis text, ``FitResult``
hashing and snapshot semantics, and ``qq_compare`` inputs.
"""

import functools

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from ...pymultifit import GAUSSIAN, LINE
from ...pymultifit.fitters import GaussianFitter
from ...pymultifit.fitters.mixed_f import MixedDataFitter
from ...pymultifit.generators import multi_gaussian, multiple_models
from ...pymultifit.plot import FitPlotter, qq_compare
from ...pymultifit.result import FitResult

_G = [(10.0, -5.0, 2.0), (8.0, 5.0, 3.0)]
_MIXED = [(0.1, 2.0), (10.0, -5.0, 2.0), (8.0, 5.0, 3.0)]
_MODELS = [LINE, GAUSSIAN, GAUSSIAN]
_X = np.linspace(-15, 15, 300)
_N_BOOT = 50


@functools.cache
def _base() -> GaussianFitter:
    y = multi_gaussian(_X, params=_G, noise_level=0.4) + np.random.default_rng(1).normal(0, 0.2, _X.size)
    fitter = GaussianFitter(_X, y)
    fitter.fit(p0=[(8, -4, 1.5), (6, 4, 2)])
    return fitter


@functools.cache
def _mixed() -> MixedDataFitter:
    y = multiple_models(_X, params=_MIXED, model_list=_MODELS) + np.random.default_rng(2).normal(0, 0.2, _X.size)
    fitter = MixedDataFitter(_X, y, model_list=_MODELS)
    fitter.fit(p0=[(0.0, 1.0), (8, -4, 1.5), (6, 4, 2)])
    return fitter


@pytest.fixture(params=["base", "mixed"])
def fitter(request):
    return _base() if request.param == "base" else _mixed()


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _text(ax: Axes) -> tuple[str, str, str]:
    return ax.get_xlabel(), ax.get_ylabel(), ax.get_title()


# ---------------------------------------------------------------------------
# every plot method runs, for single and mixed fitters
# ---------------------------------------------------------------------------


class TestPlotMethods:
    @pytest.mark.parametrize("kwargs", [{}, {"show_individuals": True}, {"is_scatter": True}])
    def test_plot_fit(self, fitter, kwargs):
        assert isinstance(fitter.plotter.plot_fit(**kwargs), Axes)
        assert isinstance(fitter.plot_fit(**kwargs), Axes)

    def test_plot_fit_single_component_with_individuals(self):
        fitter = GaussianFitter(_X, multi_gaussian(_X, params=[_G[0]], noise_level=0.2))
        fitter.fit(p0=[(8, -4, 1.5)])
        assert isinstance(fitter.plot_fit(show_individuals=True), Axes)

    def test_plot_fit_on_given_axis(self, fitter):
        _, ax = plt.subplots()
        assert fitter.plot_fit(axis=ax, plot_title="T", x_label="x", y_label="y") is ax
        assert _text(ax) == ("x", "y", "T")

    @pytest.mark.parametrize("scatter", [False, True])
    def test_plot_residuals(self, fitter, scatter):
        assert isinstance(fitter.plotter.plot_residuals(is_scatter=scatter), Axes)

    def test_plot_fit_and_residuals(self, fitter):
        ax_fit, ax_res = fitter.plotter.plot_fit_and_residuals(show_individuals=True)
        assert isinstance(ax_fit, Axes) and isinstance(ax_res, Axes)

    def test_plot_fit_and_residuals_given_axes(self, fitter):
        _, (a, b) = plt.subplots(2, 1, sharex=True)
        out = fitter.plotter.plot_fit_and_residuals(axes=(a, b))
        assert out == (a, b)

    def test_plot_fit_and_residuals_needs_two_axes(self, fitter):
        _, ax = plt.subplots()
        with pytest.raises(Exception, match="two axes"):
            fitter.plotter.plot_fit_and_residuals(axes=(ax,))

    def test_qq_plot(self, fitter):
        assert isinstance(fitter.plotter.plot_qq_plot(), Axes)

    @pytest.mark.parametrize("labels", [None, "custom"])
    def test_parameter_correlation(self, fitter, labels):
        n = len(fitter.params)
        given = None if labels is None else [f"q{i}" for i in range(n)]
        ax = fitter.plotter.plot_parameter_correlation(param_labels=given)
        shown = [t.get_text() for t in ax.get_xticklabels()]
        assert len(shown) == n
        assert shown == (list(fitter.to_result().param_labels) if given is None else given)

    def test_prediction_intervals(self, fitter):
        assert isinstance(fitter.plotter.plot_prediction_intervals(pi_level=[68, 95]), Axes)

    @pytest.mark.parametrize("flags", [(True, False), (False, True), (True, True)])
    def test_confidence_intervals_plot(self, fitter, flags):
        overall, individual = flags
        results, ax = fitter.confidence_intervals(
            ci_levels=95, n_bootstrap=_N_BOOT, seed=1, overall_ci=overall, individual_ci=individual, plot=True
        )
        assert isinstance(ax, Axes)
        assert ("overall_ci_95" in results) is overall
        assert ("individual_ci_95" in results) is individual

    def test_confidence_intervals_without_plot_returns_dict(self, fitter):
        results = fitter.confidence_intervals(ci_levels=95, n_bootstrap=_N_BOOT, seed=1)
        assert isinstance(results, dict) and {"x_range", "overall_ci_95"} <= set(results)

    def test_plot_confidence_intervals_get_value(self, fitter):
        results, ax = fitter.plotter.plot_confidence_intervals(
            ci_levels=95, n_bootstrap=_N_BOOT, seed=1, get_value=True
        )
        assert isinstance(results, dict) and isinstance(ax, Axes)

    def test_dry_run(self, fitter):
        fitter.dry_run()
        fitter.dry_run(is_scatter=True)


class TestSavePlot:
    def test_default_extension_is_png(self, fitter, tmp_path):
        fitter.plot_fit()
        path = FitPlotter.save_plot(str(tmp_path / "fig"))
        assert path.endswith(".png") and (tmp_path / "fig.png").exists()

    def test_format_from_extension(self, fitter, tmp_path):
        fig, ax = plt.subplots()
        fitter.plot_fit(axis=ax)
        FitPlotter.save_plot(str(tmp_path / "fig.pdf"), figure=fig, dpi=50)
        assert (tmp_path / "fig.pdf").read_bytes().startswith(b"%PDF")

    def test_unsupported_extension(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported format"):
            FitPlotter.save_plot(str(tmp_path / "fig.xyz"))


class TestBeforeFit:
    def test_plots_that_need_a_fit_raise(self):
        fitter = GaussianFitter(_X, np.exp(-(_X**2) / 8))
        for call in (fitter.plot_fit, fitter.plotter.plot_residuals, fitter.plotter.plot_qq_plot):
            with pytest.raises(RuntimeError, match="Fit not performed"):
                call()
        with pytest.raises(RuntimeError, match="Fit not performed"):
            fitter.confidence_intervals(ci_levels=95, n_bootstrap=_N_BOOT, seed=1)


# ---------------------------------------------------------------------------
# regression: CI / PI levels in any accepted format
# ---------------------------------------------------------------------------


class TestIntervalLevelFormats:
    @pytest.mark.parametrize(
        "levels, expected",
        [
            (95, {95}),
            (95.0, {95}),
            (0.95, {95}),
            ((68, 95), {68, 95}),
            ([0.68, 0.95], {68, 95}),
            ([68, 0.95], {68, 95}),
            (np.int64(95), {95}),
        ],
        ids=str,
    )
    def test_confidence_interval_levels(self, fitter, levels, expected):
        results, ax = fitter.confidence_intervals(ci_levels=levels, n_bootstrap=_N_BOOT, seed=1, plot=True)
        assert {int(k.split("_")[-1]) for k in results if k.startswith("overall_ci_")} == expected
        legend = {t.get_text() for t in ax.get_legend().get_texts()}
        assert {f"{lvl}% CI (overall)" for lvl in expected} <= legend

    @pytest.mark.parametrize("levels", [95, 95.0, 0.95, [68, 95], [0.68, 0.95], (68.0, 95.0), np.int64(90)], ids=str)
    def test_prediction_interval_levels(self, fitter, levels):
        ax = fitter.plotter.plot_prediction_intervals(pi_level=levels)
        assert isinstance(ax, Axes) and ax.get_legend() is not None

    @pytest.mark.parametrize("bad", [0, 100, 150, -5, [95, 0]])
    def test_invalid_prediction_interval_levels(self, fitter, bad):
        with pytest.raises(ValueError):
            fitter.plotter.plot_prediction_intervals(pi_level=bad)


# ---------------------------------------------------------------------------
# regression: overlays must not clobber the user's axis text
# ---------------------------------------------------------------------------


class TestOverlaysKeepAxisText:
    CUSTOM = ("Energy [keV]", "Flux", "My fit")

    def _fitted_axis(self, fitter):
        _, ax = plt.subplots()
        x_label, y_label, title = self.CUSTOM
        fitter.plot_fit(x_label=x_label, y_label=y_label, plot_title=title, axis=ax)
        return ax

    def test_confidence_interval_overlay(self, fitter):
        ax = self._fitted_axis(fitter)
        fitter.plotter.plot_confidence_intervals(ci_levels=95, n_bootstrap=_N_BOOT, seed=1, axis=ax)
        assert _text(ax) == self.CUSTOM

    def test_confidence_interval_overlay_via_fitter(self, fitter):
        ax = self._fitted_axis(fitter)
        fitter.confidence_intervals(ci_levels=95, n_bootstrap=_N_BOOT, seed=1, plot=True, axis=ax)
        assert _text(ax) == self.CUSTOM

    def test_prediction_interval_overlay(self, fitter):
        ax = self._fitted_axis(fitter)
        fitter.plotter.plot_prediction_intervals(pi_level=95, axis=ax)
        assert _text(ax) == self.CUSTOM

    def test_explicit_prediction_interval_text_wins(self, fitter):
        ax = self._fitted_axis(fitter)
        fitter.plotter.plot_prediction_intervals(pi_level=95, x_label="a", y_label="b", plot_title="c", axis=ax)
        assert _text(ax) == ("a", "b", "c")

    def test_fresh_axis_has_no_plotez_default_title(self, fitter):
        ax = fitter.plotter.plot_confidence_intervals(ci_levels=95, n_bootstrap=_N_BOOT, seed=1)
        assert ax.get_title() != "XY ErrorBand"

    def test_fresh_axis_prediction_interval_defaults(self, fitter):
        ax = fitter.plotter.plot_prediction_intervals(pi_level=95)
        assert _text(ax) == ("X", "Y", "PI")


# ---------------------------------------------------------------------------
# regression: qq_compare inputs
# ---------------------------------------------------------------------------


class TestQQCompare:
    def test_with_fitters(self):
        ax_l, ax_r = qq_compare(fitter_left=_base(), fitter_right=_mixed())
        assert isinstance(ax_l, Axes) and isinstance(ax_r, Axes)

    def test_with_results(self):
        ax_l, ax_r = qq_compare(fitter_left=_base().to_result(), fitter_right=_mixed().to_result())
        assert isinstance(ax_l, Axes) and isinstance(ax_r, Axes)

    def test_mixed_inputs_and_given_axes(self):
        _, (a, b) = plt.subplots(1, 2)
        assert qq_compare(fitter_left=_base(), fitter_right=_mixed().to_result(), axes=(a, b)) == (a, b)


# ---------------------------------------------------------------------------
# regression: FitResult is hashable and a true snapshot
# ---------------------------------------------------------------------------


class TestFitResultSnapshot:
    def test_hashable_with_identity_equality(self):
        fitter = _base()
        a, b = fitter.to_result(), fitter.to_result()
        assert hash(a) == hash(a) and a == a
        assert a != b
        assert len({a, b}) == 2

    def test_arrays_are_read_only(self):
        result = _base().to_result()
        for name in ("x", "y", "params", "covariance"):
            assert not getattr(result, name).flags.writeable, name
        with pytest.raises(ValueError):
            result.x[0] = 0.0

    def test_unaffected_by_in_place_changes_to_the_fitter(self):
        y = multi_gaussian(_X, params=_G, noise_level=0.3)
        fitter = GaussianFitter(_X.copy(), y)
        fitter.fit(p0=[(8, -4, 1.5), (6, 4, 2)])
        result = fitter.to_result()
        x_before, params_before, model_before = result.x.copy(), result.params.copy(), result.model().copy()

        fitter.x_values[0] = 999.0
        fitter.y_values[:] = 0.0
        fitter.params[:] = 0.0

        np.testing.assert_array_equal(result.x, x_before)
        np.testing.assert_array_equal(result.params, params_before)
        np.testing.assert_allclose(result.model(), model_before)

    def test_prefit_result_with_missing_arrays(self):
        result = GaussianFitter(_X, np.exp(-(_X**2) / 8)).to_result()
        assert isinstance(result, FitResult) and result.params is None and result.covariance is None
