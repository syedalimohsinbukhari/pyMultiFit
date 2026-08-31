"""Created on May 31 2026

Tests verifying that fitter.get_fitted_curve() always reflects the most-recent
fit result, not stale p0 values or a previous fit's frozen parameters.

The six scenarios follow the design notes in tests_for_fitterplotter_API.txt,
adapted to the real public API:

  * fitter.get_fitted_curve()   — replaces the hypothetical plotter.get_fit_curve()
  * _get_component_curves()     — replaces the hypothetical plotter.get_individual_curves()
  * fitter.params               — replaces the hypothetical params_as_list()
"""

import numpy as np

from ...pymultifit import GAUSSIAN, LINE
from ...pymultifit.fitters import GaussianFitter
from ...pymultifit.fitters.mixed_f import MixedDataFitter
from ...pymultifit.generators import multi_gaussian, multiple_models

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)

_X_WIDE = np.linspace(-50, 50, 2000)

_G1 = (10.0, -5.0, 1.5)
_G2 = (6.0, 3.0, 2.0)
_LINE = (0.5, 2.0)

_Y_MIXED = multiple_models(_X_WIDE, params=[_LINE, _G1, _G2], model_list=[LINE, GAUSSIAN, GAUSSIAN], noise_level=0.0)


def _make_mixed() -> MixedDataFitter:
    return MixedDataFitter(_X_WIDE, _Y_MIXED, model_list=[LINE, GAUSSIAN, GAUSSIAN])


def _get_component_curves(fitter: MixedDataFitter) -> list[np.ndarray]:
    """Return per-component y-arrays for a MixedDataFitter using fitter.params.

    This helper intentionally uses the same internal helpers (_instantiate_class, _instantiate_n_par) that
    FitPlotter._plot_individual_mixed() relies on.
    There is no public ``get_individual_curves()`` API, so we mirror the plotter's own logic here to keep the test a
    faithful proxy for what the plotter would render — if either the params or the model evaluators are stale,
    both the plotter and this helper will reflect that staleness in the same way.
    """
    x = fitter.x_values
    curves = []
    offset = 0
    for model in fitter.model_list:
        cls = fitter._instantiate_class(model)
        n = fitter._instantiate_n_par(model)
        pars = list(fitter.params[offset : offset + n])
        curves.append(cls.fitter(x=x, params=pars))
        offset += n
    return curves


# ===========================================================================
# 1. Sequential fits — frozen then unfrozen
# ===========================================================================


class TestSequentialFitsCurveUpdates:

    def test_curve_changes_after_releasing_frozen_component(self):
        """get_fitted_curve() after call 2 (free) must differ from call 1 (frozen)."""
        fitter = _make_mixed()
        p0 = [_LINE, _G1, _G2]

        # Call 1: freeze Line to deliberately wrong values
        fitter.fit(p0=[(0.0, 0.0), _G1, _G2], frozen={0: [True, True]})
        curve_1 = fitter.get_fitted_curve().copy()

        # Call 2: fully free, correct p0
        fitter.fit(p0=p0)
        curve_2 = fitter.get_fitted_curve()

        assert not np.allclose(curve_1, curve_2), "curve after releasing frozen Line must differ from the frozen curve"

    def test_curve_tracks_convergence_not_p0(self):
        """get_fitted_curve() must reflect converged params, not starting p0."""
        fitter = _make_mixed()
        p0_wrong = [(_LINE[0] + 10, _LINE[1] + 10), _G1, _G2]  # deliberately off

        fitter.fit(p0=p0_wrong)
        curve = fitter.get_fitted_curve()

        # Evaluate model at the bad p0 directly
        flat_p0 = np.array([v for tup in p0_wrong for v in tup])
        curve_at_p0 = fitter._n_fitter(fitter.x_values, *flat_p0)

        assert not np.allclose(curve, curve_at_p0, atol=0.1), "fitted curve must not equal the curve at (wrong) p0"


# ===========================================================================
# 2. Different frozen dicts across calls
# ===========================================================================


class TestDifferentFrozenDicts:

    def test_freeze_different_components_gives_different_curves(self):
        """Freezing component 0 vs component 1 must produce different curves."""
        fitter = _make_mixed()
        p0_wrong = [(_LINE[0] + 5, _LINE[1] - 5), (_G1[0] + 5, _G1[1], _G1[2]), _G2]

        fitter.fit(p0=p0_wrong, frozen={0: [True, True]})
        curve_1 = fitter.get_fitted_curve().copy()

        fitter.fit(p0=p0_wrong, frozen={1: [True, True, True]})
        curve_2 = fitter.get_fitted_curve()

        assert not np.allclose(curve_1, curve_2), "freezing component 0 vs component 1 must yield different curves"


# ===========================================================================
# 3. All-frozen → free — the sharpest staleness check
# ===========================================================================


class TestAllFrozenThenFree:

    def test_free_curve_differs_from_fully_frozen_wrong_curve(self):
        """After an all-frozen wrong fit, a free fit must update the curve."""
        fitter = _make_mixed()
        p0_wrong = [(_LINE[0], _LINE[1] + 5.0), _G1, _G2]  # intercept wrong

        fitter.fit(p0=p0_wrong, frozen={0: [True, True], 1: [True, True, True], 2: [True, True, True]})
        curve_frozen = fitter.get_fitted_curve().copy()

        fitter.fit(p0=p0_wrong)  # free to correct itself
        curve_free = fitter.get_fitted_curve()

        assert not np.allclose(
            curve_frozen, curve_free, atol=0.1
        ), "free fit must move away from the all-frozen wrong curve"


# ===========================================================================
# 4. Curve reflects params, not p0  (highest-priority check)
# ===========================================================================


class TestCurveReflectsParams:

    def test_fitted_curve_equals_model_evaluated_at_params(self):
        """get_fitted_curve() must equal the model evaluated at fitter.params."""
        fitter = _make_mixed()
        fitter.fit(p0=[_LINE, _G1, _G2], frozen={1: [False, True, False]})

        actual = fitter.get_fitted_curve()
        expected = fitter._n_fitter(fitter.x_values, *fitter.params)

        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_curve_not_equal_to_model_at_p0_when_p0_is_wrong(self):
        """When p0 is wrong the fitted curve must differ from model(p0)."""
        fitter = _make_mixed()
        p0_wrong = [(_LINE[0] + 3, _LINE[1] - 3), _G1, _G2]

        fitter.fit(p0=p0_wrong)

        flat_p0 = np.array([v for tup in p0_wrong for v in tup])
        curve_at_p0 = fitter._n_fitter(fitter.x_values, *flat_p0)
        actual = fitter.get_fitted_curve()

        assert not np.allclose(actual, curve_at_p0, atol=0.1)

    def test_base_fitter_curve_equals_model_at_params(self):
        """Same params-vs-curve consistency check for a plain BaseFitter subclass."""
        x = np.linspace(-15, 15, 800)
        y = multi_gaussian(x, params=[_G1, _G2], noise_level=0.0)
        fitter = GaussianFitter(x, y)
        fitter.fit(p0=[_G1, _G2], frozen=[False, True, False])

        actual = fitter.get_fitted_curve()
        expected = fitter._n_fitter(fitter.x_values, *fitter.params)

        np.testing.assert_allclose(actual, expected, atol=1e-10)


# ===========================================================================
# 5. Individual component curves also update after re-fit
# ===========================================================================


class TestIndividualComponentCurves:

    def test_component_curves_change_after_refitting(self):
        """At least one component curve must change when frozen dict changes."""
        fitter = _make_mixed()
        p0_wrong = [(_LINE[0] + 5, _LINE[1] - 5), _G1, _G2]

        fitter.fit(p0=p0_wrong, frozen={0: [True, True]})
        components_1 = [c.copy() for c in _get_component_curves(fitter)]

        fitter.fit(p0=[_LINE, _G1, _G2])  # free — Line should move
        components_2 = _get_component_curves(fitter)

        assert any(
            not np.allclose(c1, c2) for c1, c2 in zip(components_1, components_2)
        ), "at least one component curve must differ after releasing frozen Line"

    def test_component_curves_consistent_with_flat_params(self):
        """Each component curve must equal the model evaluated at its param slice."""
        fitter = _make_mixed()
        fitter.fit(p0=[_LINE, _G1, _G2], frozen={2: [False, True, False]})

        x = fitter.x_values
        offset = 0
        for i, model in enumerate(fitter.model_list):
            n = fitter._instantiate_n_par(model)
            pars = list(fitter.params[offset : offset + n])
            cls = fitter._instantiate_class(model)
            expected = cls.fitter(x=x, params=pars)
            computed = _get_component_curves(fitter)[i]
            np.testing.assert_allclose(computed, expected, atol=1e-12, err_msg=f"component {i} curve mismatch")
            offset += n


# ===========================================================================
# 6. Ordering — get_fitted_curve() is eager, not lazy
# ===========================================================================


class TestCurveOrdering:

    def test_curve_same_whether_read_before_or_after_params(self):
        """get_fitted_curve() must give the same result regardless of read order."""
        fitter = _make_mixed()
        p0 = [_LINE, _G1, _G2]

        fitter.fit(p0=p0, frozen={0: [True, True]})

        # Read order A: params first, then curve
        params_a = fitter.params.copy()
        curve_a = fitter.get_fitted_curve().copy()

        # Read order B: curve first, then params — same fit, no re-fitting
        curve_b = fitter.get_fitted_curve().copy()
        params_b = fitter.params.copy()

        np.testing.assert_allclose(curve_a, curve_b, atol=1e-10)
        np.testing.assert_allclose(params_a, params_b, atol=1e-10)

    def test_repeated_curve_calls_are_idempotent(self):
        """Calling get_fitted_curve() twice without re-fitting must return equal arrays."""
        fitter = _make_mixed()
        fitter.fit(p0=[_LINE, _G1, _G2])

        curve_a = fitter.get_fitted_curve()
        curve_b = fitter.get_fitted_curve()

        np.testing.assert_allclose(curve_a, curve_b, atol=1e-15)
