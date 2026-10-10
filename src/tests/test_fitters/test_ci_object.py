"""Created on Oct 10 2026

Tests of the confidence-interval pipeline: exact levels, the ``ConfidenceIntervals`` object, ``rng_engine``, ``x_range``
(evaluated together with the data), the single evaluation per bootstrap sample, the plotter that takes the object, and the
``rng_engine`` of the data generators.
"""

import dataclasses
import functools

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from ...pymultifit import GAUSSIAN, LINE
from ...pymultifit.ci import Band, ConfidenceIntervals, compute_ci_bounds, normalize_levels
from ...pymultifit.fitters import GaussianFitter
from ...pymultifit.generators import multi_gaussian, multiple_models
from ...pymultifit.result import Component

_X = np.linspace(-15, 15, 300)
_G = [(10.0, -5.0, 2.0), (8.0, 5.0, 3.0)]
_N = 100


@functools.cache
def _fitter() -> GaussianFitter:
    y = multi_gaussian(_X, params=_G, noise_level=0.0) + np.random.default_rng(1).normal(0, 0.2, _X.size)
    fitter = GaussianFitter(_X, y)
    fitter.fit(p0=[(8, -4, 1.5), (6, 4, 2)])
    return fitter


def _result():
    return _fitter().to_result()


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# levels
# ---------------------------------------------------------------------------


class TestNormalizeLevels:
    @pytest.mark.parametrize(
        "levels, expected",
        [
            (95, (95.0,)),
            (95.0, (95.0,)),
            (0.95, (95.0,)),
            (np.int64(90), (90.0,)),
            ([0.68, 95, 99.7, 0.995], (68.0, 95.0, 99.5, 99.7)),
            ((99.7, 68), (68.0, 99.7)),
            ([95, 0.95, 95.0], (95.0,)),
            (0.29, (29.0,)),
        ],
        ids=str,
    )
    def test_exact_ascending_and_unique(self, levels, expected):
        assert normalize_levels(levels) == expected

    @pytest.mark.parametrize("bad", [0, 100, 150, -5, 1, [95, 0]], ids=str)
    def test_invalid(self, bad):
        with pytest.raises(ValueError, match="Invalid confidence interval"):
            normalize_levels(bad)


# ---------------------------------------------------------------------------
# the object
# ---------------------------------------------------------------------------


class TestConfidenceIntervalsObject:
    def test_describes_itself(self):
        ci = compute_ci_bounds(_result(), [95, 68], n_bootstrap=_N, rng_engine=1, individual_ci=True)
        assert isinstance(ci, ConfidenceIntervals)
        assert ci.levels == (68.0, 95.0)  # ascending, whatever the order given
        assert ci.n_bootstrap == _N and ci.x.shape == (1000,)
        assert set(ci.overall) == {68.0, 95.0} and set(ci.individual) == {68.0, 95.0}
        assert all(isinstance(b, Band) for b in ci.overall.values())
        assert all(len(bands) == 2 for bands in ci.individual.values())

    def test_parts_that_were_not_computed_are_none(self):
        only_overall = compute_ci_bounds(_result(), 95, n_bootstrap=_N, rng_engine=1)
        only_individual = compute_ci_bounds(
            _result(), 95, n_bootstrap=_N, rng_engine=1, overall_ci=False, individual_ci=True
        )
        assert only_overall.individual is None and only_overall.overall is not None
        assert only_individual.overall is None and only_individual.individual is not None

    def test_is_read_only(self):
        ci = compute_ci_bounds(_result(), 95, n_bootstrap=_N, rng_engine=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ci.levels = (1.0,)
        with pytest.raises(TypeError):
            ci.overall[50.0] = ci.overall[95.0]
        for array in (ci.x, ci.overall[95.0].lower, ci.overall[95.0].median, ci.overall[95.0].upper):
            assert not array.flags.writeable

    def test_band_is_ordered(self):
        band = compute_ci_bounds(_result(), 95, n_bootstrap=_N, rng_engine=1).overall[95.0]
        assert np.all(band.lower <= band.median) and np.all(band.median <= band.upper)

    def test_nested_levels_are_nested(self):
        ci = compute_ci_bounds(_result(), [68, 95, 99.7], n_bootstrap=_N, rng_engine=1)
        narrow, wide = ci.overall[68.0], ci.overall[99.7]
        assert np.all(wide.lower <= narrow.lower) and np.all(narrow.upper <= wide.upper)

    def test_levels_that_round_to_the_same_whole_percent_do_not_collide(self):
        ci = compute_ci_bounds(_result(), [99.5, 99.7], n_bootstrap=_N, rng_engine=1)
        assert ci.levels == (99.5, 99.7)
        assert not np.array_equal(ci.overall[99.5].upper, ci.overall[99.7].upper)
        assert {"overall_ci_99.5", "overall_ci_99.7"} <= set(ci.as_dict())

    def test_as_dict_layout(self):
        ci = compute_ci_bounds(_result(), [68, 95], n_bootstrap=_N, rng_engine=1, individual_ci=True)
        flat = ci.as_dict()
        assert list(flat) == ["x_range", "overall_ci_68", "overall_ci_95", "individual_ci_68", "individual_ci_95"]
        assert flat["x_range"] is ci.x
        assert set(flat["overall_ci_95"]) == {"lower", "median", "upper"}
        assert len(flat["individual_ci_95"]) == 2 and set(flat["individual_ci_95"][0]) == {"lower", "median", "upper"}
        np.testing.assert_array_equal(flat["overall_ci_95"]["upper"], ci.overall[95.0].upper)

    def test_band_lookup(self):
        ci = compute_ci_bounds(_result(), [68, 95], n_bootstrap=_N, rng_engine=1, individual_ci=True)
        assert ci.band(95) is ci.overall[95.0] and ci.band(0.95) is ci.overall[95.0]
        assert ci.band(68, component=1) is ci.individual[68.0][1]

    def test_band_lookup_errors(self):
        ci = compute_ci_bounds(_result(), 95, n_bootstrap=_N, rng_engine=1)
        with pytest.raises(KeyError, match="was not computed"):
            ci.band(90)
        with pytest.raises(ValueError, match="individual interval was not computed"):
            ci.band(95, component=0)
        with pytest.raises(ValueError, match="exactly one level"):
            ci.band([68, 95])
        no_overall = compute_ci_bounds(
            _result(), 95, n_bootstrap=_N, rng_engine=1, overall_ci=False, individual_ci=True
        )
        with pytest.raises(ValueError, match="overall interval was not computed"):
            no_overall.band(95)

    def test_repr_is_short(self):
        text = repr(compute_ci_bounds(_result(), 95, n_bootstrap=_N, rng_engine=1))
        assert len(text) < 200 and "levels=(95.0,)" in text and "n_bootstrap=100" in text

    def test_needs_a_part_and_a_valid_sample_count(self):
        with pytest.raises(ValueError, match="At least one"):
            compute_ci_bounds(_result(), 95, overall_ci=False, individual_ci=False)
        with pytest.raises(ValueError, match="n_bootstrap"):
            compute_ci_bounds(_result(), 95, n_bootstrap=0)

    def test_needs_a_fit(self):
        unfitted = GaussianFitter(_X, np.exp(-(_X**2) / 8)).to_result()
        with pytest.raises(RuntimeError, match="Fit not performed"):
            compute_ci_bounds(unfitted, 95, n_bootstrap=_N, rng_engine=1)


# ---------------------------------------------------------------------------
# rng_engine
# ---------------------------------------------------------------------------


class TestRngEngine:
    @staticmethod
    def _upper(**kwargs):
        return compute_ci_bounds(_result(), 95, n_bootstrap=_N, **kwargs).overall[95.0].upper

    def test_no_seed_is_allowed_and_differs_between_calls(self):
        assert not np.array_equal(self._upper(), self._upper())

    def test_same_integer_repeats(self):
        np.testing.assert_array_equal(self._upper(rng_engine=3), self._upper(rng_engine=3))
        assert not np.array_equal(self._upper(rng_engine=3), self._upper(rng_engine=4))

    def test_integer_is_the_seed_of_a_new_generator(self):
        np.testing.assert_array_equal(self._upper(rng_engine=3), self._upper(rng_engine=np.random.default_rng(3)))

    def test_generator_is_consumed_so_calls_are_independent(self):
        rng = np.random.default_rng(5)
        first, second = self._upper(rng_engine=rng), self._upper(rng_engine=rng)
        assert not np.array_equal(first, second)
        again = np.random.default_rng(5)  # the same sequence is reproduced from the same start
        np.testing.assert_array_equal(first, self._upper(rng_engine=again))
        np.testing.assert_array_equal(second, self._upper(rng_engine=again))

    def test_levels_and_parts_do_not_change_the_draws(self):
        alone = compute_ci_bounds(_result(), [95], n_bootstrap=_N, rng_engine=7)
        more = compute_ci_bounds(_result(), [68, 95, 99], n_bootstrap=_N, rng_engine=7, individual_ci=True)
        np.testing.assert_array_equal(alone.overall[95.0].lower, more.overall[95.0].lower)
        np.testing.assert_array_equal(alone.overall[95.0].upper, more.overall[95.0].upper)

    def test_fitter_method_forwards_it(self):
        fitter = _fitter()
        a = fitter.confidence_intervals(ci_levels=95, n_bootstrap=_N, rng_engine=9)
        b = compute_ci_bounds(fitter.to_result(), 95, n_bootstrap=_N, rng_engine=9)
        np.testing.assert_array_equal(a.overall[95.0].upper, b.overall[95.0].upper)

    def test_invalid_engine(self):
        with pytest.raises(TypeError):
            compute_ci_bounds(_result(), 95, n_bootstrap=_N, rng_engine=3.5)


# ---------------------------------------------------------------------------
# x_range
# ---------------------------------------------------------------------------


class TestXRange:
    @pytest.mark.parametrize("order", ["ascending", "descending", "shuffled"])
    def test_default_range_spans_the_data_whatever_their_order(self, order):
        x = _X.copy()
        y = multi_gaussian(x, params=_G, noise_level=0.0) + np.random.default_rng(1).normal(0, 0.2, x.size)
        if order == "descending":
            x, y = x[::-1], y[::-1]
        elif order == "shuffled":
            idx = np.random.default_rng(0).permutation(x.size)
            x, y = x[idx], y[idx]
        fitter = GaussianFitter(x, y)
        fitter.fit(p0=[(8, -4, 1.5), (6, 4, 2)])
        ci = fitter.confidence_intervals(ci_levels=95, n_bootstrap=_N, rng_engine=1)
        assert (ci.x.min(), ci.x.max()) == (-15.0, 15.0) and ci.x.size == 1000

    def test_given_range_is_kept_as_floats(self):
        ci = compute_ci_bounds(_result(), 95, n_bootstrap=_N, rng_engine=1, x_range=[-1, 0, 1])
        assert ci.x.dtype == float and list(ci.x) == [-1.0, 0.0, 1.0]
        assert ci.overall[95.0].median.shape == (3,)

    @pytest.mark.parametrize("bad", [[], np.zeros((2, 2))], ids=["empty", "2-D"])
    def test_invalid_range(self, bad):
        with pytest.raises(ValueError, match="x_range"):
            compute_ci_bounds(_result(), 95, n_bootstrap=_N, rng_engine=1, x_range=bad)

    def test_premise_amplitude_is_the_peak_of_the_evaluated_array(self):
        # the library scales an unnormalised component so that its maximum on the evaluated array equals the amplitude: a value
        # depends on the other x evaluated with it. If this ever changes (pointwise scaling), the x_range handling of
        # compute_ci_bounds, which evaluates a window together with the data x, can be simplified.
        result = _result()
        amplitude = result.params[0]
        assert result.component_curve(0, np.array([3.0]))[0] == pytest.approx(amplitude)  # a tail point, alone
        assert result.component_curve(0, np.array([3.0, -5.0]))[0] < 0.01  # the same point next to the peak

    @pytest.mark.parametrize(
        "window",
        [(-15, 15), (-25, 25), (-8, 8), (0, 15), (8, 15), (-15, -8), (2, 4)],
        ids=lambda w: f"[{w[0]}, {w[1]}]",
    )
    def test_band_follows_the_fitted_curve_wherever_the_window_is(self, window):
        result = _result()
        xr = np.linspace(*window, 400)
        ci = compute_ci_bounds(result, 95, n_bootstrap=200, rng_engine=1, x_range=xr)
        # the fitted curve evaluated with the data x, as the fit saw it (np.interp reads the same values at the window)
        truth = np.interp(xr, np.concatenate([[-25], result.x, [25]]), np.concatenate([[0], result.model(), [0]]))
        inside = (xr >= -15) & (xr <= 15)
        error = np.abs(ci.overall[95.0].median - truth)[inside].max() if inside.any() else 0.0
        assert error < 0.15  # peak is 10; a rescaled tail would be off by several units

    def test_individual_components_follow_the_fit_in_a_window_without_their_peak(self):
        result = _result()
        xr = np.linspace(8, 15, 200)
        ci = compute_ci_bounds(
            result, 95, n_bootstrap=200, rng_engine=1, x_range=xr, individual_ci=True, overall_ci=False
        )
        first_peak = ci.individual[95.0][0].median  # the component centred at -5 has only a tail in [8, 15]
        assert first_peak.max() < 0.1  # not rescaled to its amplitude of about 10

    def test_default_grid_is_unchanged_by_the_data_x(self):
        # no data x are added to the default grid: the result is exactly what evaluating that grid alone gives
        result = _result()
        ci = compute_ci_bounds(result, 95, n_bootstrap=50, rng_engine=2)
        params = np.random.default_rng(2).multivariate_normal(result.params, result.covariance, size=50)
        curves = np.array([result.model(ci.x, p) for p in params])
        np.testing.assert_array_equal(ci.overall[95.0].median, np.quantile(curves, 0.5, axis=0))


# ---------------------------------------------------------------------------
# every component is evaluated once per bootstrap sample
# ---------------------------------------------------------------------------


def _counting(result):
    calls = [0] * result.n_fits

    def wrap(index, func):
        def counted(x, params):
            calls[index] += 1
            return func(x, params)

        return counted

    components = tuple(Component(c.label, wrap(i, c.func), c.n_par) for i, c in enumerate(result.components))
    return dataclasses.replace(result, components=components), calls


class TestSingleEvaluation:
    @pytest.mark.parametrize("flags", [(True, False), (False, True), (True, True)], ids=str)
    def test_one_call_per_component_and_sample(self, flags):
        overall, individual = flags
        result, calls = _counting(_result())
        compute_ci_bounds(result, [68, 95], n_bootstrap=_N, rng_engine=1, overall_ci=overall, individual_ci=individual)
        assert calls == [_N, _N]

    def test_overall_only_does_not_keep_the_component_curves(self):
        # peak memory of an overall-only call must not grow with the number of components: the per-component array is only
        # created when it is asked for, so asking for the total alone must work for a model of many components
        x = np.linspace(-30, 30, 400)
        params = [(5.0, c, 1.0) for c in range(-24, 25, 6)]
        y = multi_gaussian(x, params=params, noise_level=0.0) + np.random.default_rng(1).normal(0, 0.05, x.size)
        fitter = GaussianFitter(x, y)
        fitter.fit(p0=params)
        ci = fitter.confidence_intervals(ci_levels=95, n_bootstrap=20, rng_engine=1)
        assert ci.individual is None and ci.overall[95.0].median.shape == (1000,)


# ---------------------------------------------------------------------------
# the plotter takes the object
# ---------------------------------------------------------------------------


class TestPlotterTakesTheObject:
    @staticmethod
    def _legend(ax):
        return [t.get_text() for t in ax.get_legend().get_texts()]

    def test_draws_what_was_computed(self):
        fitter = _fitter()
        ci = fitter.confidence_intervals(ci_levels=[95, 68], n_bootstrap=_N, rng_engine=1, individual_ci=True)
        ax = fitter.plotter.plot_confidence_intervals(ci)
        assert isinstance(ax, Axes)
        assert self._legend(ax) == ["68% CI (overall)", "95% CI (overall)", "68% CI (fit 1)", "95% CI (fit 1)"]

    def test_selection_narrows_what_is_drawn(self):
        fitter = _fitter()
        ci = fitter.confidence_intervals(ci_levels=95, n_bootstrap=_N, rng_engine=1, individual_ci=True)
        assert self._legend(fitter.plotter.plot_confidence_intervals(ci, individual=False)) == ["95% CI (overall)"]
        assert self._legend(fitter.plotter.plot_confidence_intervals(ci, overall=False)) == ["95% CI (fit 1)"]

    def test_individual_only_object_is_drawn_by_default(self):
        fitter = _fitter()
        ci = fitter.confidence_intervals(
            ci_levels=95, n_bootstrap=_N, rng_engine=1, overall_ci=False, individual_ci=True
        )
        assert self._legend(fitter.plotter.plot_confidence_intervals(ci)) == ["95% CI (fit 1)"]

    def test_asking_for_a_part_that_was_not_computed(self):
        fitter = _fitter()
        ci = fitter.confidence_intervals(ci_levels=95, n_bootstrap=_N, rng_engine=1)
        with pytest.raises(ValueError, match="individual intervals were not computed"):
            fitter.plotter.plot_confidence_intervals(ci, individual=True)
        only = fitter.confidence_intervals(
            ci_levels=95, n_bootstrap=_N, rng_engine=1, overall_ci=False, individual_ci=True
        )
        with pytest.raises(ValueError, match="overall intervals were not computed"):
            fitter.plotter.plot_confidence_intervals(only, overall=True)

    def test_nothing_selected(self):
        fitter = _fitter()
        ci = fitter.confidence_intervals(ci_levels=95, n_bootstrap=_N, rng_engine=1, individual_ci=True)
        with pytest.raises(ValueError, match="Nothing to draw"):
            fitter.plotter.plot_confidence_intervals(ci, overall=False, individual=False)

    @pytest.mark.parametrize("not_an_object", [{"x_range": 1}, None, 95], ids=["dict", "None", "level"])
    def test_other_objects_are_refused_with_a_hint(self, not_an_object):
        with pytest.raises(TypeError, match="ConfidenceIntervals"):
            _fitter().plotter.plot_confidence_intervals(not_an_object)

    def test_one_object_can_be_drawn_twice_without_recomputing(self):
        fitter = _fitter()
        ci = fitter.confidence_intervals(ci_levels=[68, 95], n_bootstrap=_N, rng_engine=1)
        _, (a, b) = plt.subplots(1, 2)
        fitter.plotter.plot_confidence_intervals(ci, axis=a)
        fitter.plotter.plot_confidence_intervals(ci, axis=b)
        assert self._legend(a) == self._legend(b)

    def test_works_for_a_mixed_model(self):
        from ...pymultifit.fitters.mixed_f import MixedDataFitter

        models, pars = [LINE, GAUSSIAN, GAUSSIAN], [(0.1, 2.0), (10.0, -5.0, 2.0), (8.0, 5.0, 3.0)]
        y = multiple_models(_X, params=pars, model_list=models, noise_level=0.0) + np.random.default_rng(2).normal(
            0, 0.2, _X.size
        )
        fitter = MixedDataFitter(_X, y, model_list=models)
        fitter.fit(p0=[(0.0, 1.0), (8, -4, 1.5), (6, 4, 2)])
        ci = fitter.confidence_intervals(ci_levels=95, n_bootstrap=_N, rng_engine=1, individual_ci=True)
        assert len(ci.individual[95.0]) == 3
        assert isinstance(fitter.plotter.plot_confidence_intervals(ci), Axes)


# ---------------------------------------------------------------------------
# rng_engine of the data generators
# ---------------------------------------------------------------------------


class TestGeneratorNoise:
    @staticmethod
    def _data(**kwargs):
        return multi_gaussian(_X, params=_G, noise_level=0.5, **kwargs)

    def test_none_keeps_using_the_global_state(self):
        np.random.seed(11)
        first = self._data()
        np.random.seed(11)
        np.testing.assert_array_equal(first, self._data())

    def test_a_generator_gives_reproducible_noise_independent_of_the_global_state(self):
        np.random.seed(1)
        a = self._data(rng_engine=np.random.default_rng(4))
        np.random.seed(2)
        b = self._data(rng_engine=np.random.default_rng(4))
        np.testing.assert_array_equal(a, b)

    def test_integer_is_the_seed_of_a_new_generator(self):
        np.testing.assert_array_equal(self._data(rng_engine=4), self._data(rng_engine=np.random.default_rng(4)))

    def test_one_generator_over_several_calls_gives_independent_noise(self):
        rng = np.random.default_rng(6)
        assert not np.array_equal(self._data(rng_engine=rng), self._data(rng_engine=rng))

    def test_noise_scales_with_the_level_and_vanishes_at_zero(self):
        clean = multi_gaussian(_X, params=_G, noise_level=0.0, rng_engine=1)
        noisy = multi_gaussian(_X, params=_G, noise_level=0.5, rng_engine=1)
        assert np.std(noisy - clean) == pytest.approx(0.5, rel=0.2)
        np.testing.assert_array_equal(clean, multi_gaussian(_X, params=_G))

    def test_multiple_models(self):
        kwargs = dict(params=[(0.1, 2.0), (10.0, -5.0, 2.0)], model_list=[LINE, GAUSSIAN], noise_level=0.5)
        a = multiple_models(_X, rng_engine=3, **kwargs)
        np.testing.assert_array_equal(a, multiple_models(_X, rng_engine=np.random.default_rng(3), **kwargs))
        assert not np.array_equal(a, multiple_models(_X, rng_engine=4, **kwargs))
