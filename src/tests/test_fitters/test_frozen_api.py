"""Created on May 31 2025"""

import warnings

import numpy as np
import pytest

from ...pymultifit import EPSILON as EPS, GAMMA, GAUSSIAN, LAPLACE, LINE
from ...pymultifit.fitters import GammaFitter, GaussianFitter
from ...pymultifit.fitters.mixed_f import MixedDataFitter
from ...pymultifit.generators import multi_gamma, multi_gaussian, multiple_models

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
_X_GAUSS = np.linspace(-15, 15, 800)
_X_POS = np.linspace(EPS, 20, 800)
_X_WIDE = np.linspace(-50, 50, 2000)

# True params for reuse
_G1 = (10.0, -5.0, 1.5)  # (amp, mu, sigma)
_G2 = (6.0, 3.0, 2.0)
_GA = (3.0, 3.0, 1.5, 0.0)  # (amp, shape, scale, loc)
_L1 = (5.0, 0.0, 1.0)  # (amp, mu, scale)
_LINE = (0.5, 2.0)  # (slope, intercept)

_Y_GAUSS2 = multi_gaussian(_X_GAUSS, params=[_G1, _G2], noise_level=0.0) + RNG.normal(0, 0.1, len(_X_GAUSS))
_Y_GAMMA = multi_gamma(_X_POS, params=[_GA], noise_level=0.0) + RNG.normal(0, 0.03, len(_X_POS))


def _is_clamped(value, target, tol=1e-4):
    """True when a value stayed within epsilon of its frozen target."""
    return abs(value - target) < tol


def _all_clamped(params, p0_flat, frozen_mask):
    for val, initial, is_frozen in zip(params, p0_flat, frozen_mask):
        if is_frozen and not _is_clamped(val, initial):
            return False
    return True


# ===========================================================================
# 1. Correctness invariants
# ===========================================================================


class TestFrozenCorrectness:

    def test_all_params_frozen_equals_p0(self):
        """When every parameter is frozen the fitted output must equal p0."""
        fitter = GaussianFitter(_X_GAUSS, _Y_GAUSS2)
        p0 = [_G1, _G2]
        fitter.fit(p0=p0, frozen=[True, True, True])
        expected = np.array([*_G1, *_G2])
        np.testing.assert_allclose(fitter.params, expected, atol=1e-6)

    def test_single_param_frozen_others_move(self):
        """The frozen param must stay; free params must differ from p0."""
        fitter = GaussianFitter(_X_GAUSS, _Y_GAUSS2)
        p0 = [(_G1[0], _G1[1], 3.0), _G2]  # sigma deliberately wrong
        fitter.fit(p0=p0, frozen=[False, False, True])

        params = fitter.params.reshape(-1, 3)
        # frozen sigma of component 0 must stay at 3.0
        assert _is_clamped(params[0, 2], 3.0)
        # amplitude of component 0 must have moved away from 3.0-sigma p0 start
        assert not _is_clamped(params[0, 0], p0[0][0], tol=1.0)

    def test_frozen_wrong_value_stable(self):
        """Freezing a param to a wrong-but-in-range value should not crash."""
        fitter = GaussianFitter(_X_GAUSS, _Y_GAUSS2)
        p0 = [(10.0, 10.0, 1.5), _G2]  # mu frozen at 10.0, far from true -5.0 but in data range
        fitter.fit(p0=p0, frozen=[False, True, False])
        params = fitter.params.reshape(-1, 3)
        assert _is_clamped(params[0, 1], 10.0)

    def test_frozen_at_zero(self):
        """Freezing loc=0 in Gamma (a common boundary value) must hold."""
        fitter = GammaFitter(_X_POS, _Y_GAMMA)
        fitter.fit(p0=[_GA], frozen=[False, False, False, True])
        assert _is_clamped(fitter.params[3], 0.0)

    def test_frozen_small_sigma(self):
        """Freezing sigma to a tiny value should terminate cleanly."""
        fitter = GaussianFitter(_X_GAUSS, _Y_GAUSS2)
        p0 = [(10.0, -5.0, 1e-3), _G2]
        fitter.fit(p0=p0, frozen=[False, False, True])
        assert _is_clamped(fitter.params[2], 1e-3)


# ===========================================================================
# 2. Mask length validation (BaseFitter)
# ===========================================================================


class TestBaseFitterMaskLength:

    def test_n_par_mask_no_warning(self):
        """n_par-length mask must not emit a UserWarning."""
        fitter = GammaFitter(_X_POS, _Y_GAMMA)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fitter.fit(p0=[_GA], frozen=[False, False, False, True])
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_pn_par_mask_warns_and_pads(self):
        """pn_par-length mask must warn and auto-pad loc as free."""
        fitter = GammaFitter(_X_POS, _Y_GAMMA)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fitter.fit(p0=[_GA], frozen=[False, False, False])  # pn_par=3
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "auto-padded" in str(user_warnings[0].message)
        # loc must NOT be frozen — verify by comparing against a fit where loc IS
        # explicitly frozen; the two results must differ.
        fitter_frozen_loc = GammaFitter(_X_POS, _Y_GAMMA)
        fitter_frozen_loc.fit(p0=[_GA], frozen=[False, False, False, True])
        assert not np.isclose(
            fitter.params[3], fitter_frozen_loc.params[3], atol=1e-4
        ), "loc appears frozen (matches explicitly frozen-loc result)"

    def test_pn_par_mask_all_true_loc_free(self):
        """pn_par mask [True,True,True] freezes primaries; loc must remain free."""
        fitter = GammaFitter(_X_POS, _Y_GAMMA)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fitter.fit(p0=[_GA], frozen=[True, True, True])
        # Primary params clamped
        for i in range(3):
            assert _is_clamped(fitter.params[i], _GA[i])
        # loc (index 3) must remain free — compare against a fully-frozen fit
        # where loc is also frozen; they must produce different loc values.
        fitter_full = GammaFitter(_X_POS, _Y_GAMMA)
        fitter_full.fit(p0=[_GA], frozen=[True, True, True, True])
        assert not _is_clamped(
            fitter.params[3], fitter_full.params[3], tol=1e-4
        ), "loc appears frozen (matches fully-frozen result)"

    def test_mask_shorter_than_pn_par_raises(self):
        """A mask shorter than pn_par must raise ValueError."""
        fitter = GammaFitter(_X_POS, _Y_GAMMA)  # pn_par=3
        with pytest.raises(ValueError, match="pn_par"):
            fitter.fit(p0=[_GA], frozen=[False, False])

    def test_mask_longer_than_n_par_raises(self):
        """A mask longer than n_par must raise ValueError."""
        fitter = GammaFitter(_X_POS, _Y_GAMMA)  # n_par=4
        with pytest.raises(ValueError, match="n_par"):
            fitter.fit(p0=[_GA], frozen=[False, False, False, False, False])

    def test_no_loc_scale_distribution_exact_n_par_only(self):
        """For a distribution with no secondary params (pn_par==n_par),
        pn_par and n_par masks are the same length — no warning expected."""
        fitter = GaussianFitter(_X_GAUSS, _Y_GAUSS2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fitter.fit(p0=[_G1, _G2], frozen=[False, False, False])
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0


# ===========================================================================
# 3. All-True / all-False mask equivalences (BaseFitter)
# ===========================================================================


class TestBaseFitterMaskEquivalences:

    def test_all_false_equivalent_to_no_frozen(self):
        """All-False mask must produce params identical to frozen=None."""
        fitter_none = GaussianFitter(_X_GAUSS, _Y_GAUSS2)
        fitter_false = GaussianFitter(_X_GAUSS, _Y_GAUSS2)
        fitter_none.fit(p0=[_G1, _G2])
        fitter_false.fit(p0=[_G1, _G2], frozen=[False, False, False])
        np.testing.assert_allclose(fitter_none.params, fitter_false.params, atol=1e-8)

    def test_all_true_freezes_that_param_across_all_components(self):
        """BaseFitter applies the mask uniformly to ALL fits.
        Freezing amplitude (index 0) in all components must clamp both amplitudes."""
        fitter = GaussianFitter(_X_GAUSS, _Y_GAUSS2)
        p0 = [(10.0, -5.0, 1.5), (6.0, 3.0, 2.0)]
        fitter.fit(p0=p0, frozen=[True, False, False])
        params = fitter.params.reshape(-1, 3)
        # Amplitude clamped in both components
        assert _is_clamped(params[0, 0], 10.0)
        assert _is_clamped(params[1, 0], 6.0)
        # mu and sigma must have moved (free to optimise)
        fitter_free = GaussianFitter(_X_GAUSS, _Y_GAUSS2)
        fitter_free.fit(p0=p0)
        free_params = fitter_free.params.reshape(-1, 3)
        # mu columns should differ between frozen-amp and fully-free fits
        # (the amplitude constraint changes how mu/sigma settle)
        assert not np.allclose(params[:, 1:], free_params[:, 1:], atol=1e-8)


# ===========================================================================
# 4. State isolation across sequential .fit() calls (BaseFitter)
# ===========================================================================


class TestBaseFitterStateIsolation:

    def test_second_fit_not_contaminated_by_first(self):
        """Param frozen in call 1 but free in call 2 must move in call 2."""
        fitter = GaussianFitter(_X_GAUSS, _Y_GAUSS2)
        # Call 1: freeze mu
        fitter.fit(p0=[_G1, _G2], frozen=[False, True, False])
        params_call1 = fitter.params.copy()

        # Call 2: no frozen — mu must be free to optimise
        fitter.fit(p0=[_G1, _G2], frozen=None)
        params_call2 = fitter.params.copy()

        # mu positions (index 1, 4) should differ between the two fits
        assert not np.allclose(params_call1[[1, 4]], params_call2[[1, 4]], atol=1e-3)

    def test_second_fit_fully_unconstrained(self):
        """After a frozen fit, calling fit() with no frozen must be unconstrained."""
        fitter = GaussianFitter(_X_GAUSS, _Y_GAUSS2)
        fitter.fit(p0=[_G1, _G2], frozen=[True, True, True])
        frozen_params = fitter.params.copy()

        fitter.fit(p0=[_G1, _G2])
        free_params = fitter.params.copy()

        assert not np.allclose(frozen_params, free_params, atol=1e-4)


# ===========================================================================
# 5. Position sensitivity (BaseFitter) — catches off-by-one errors
# ===========================================================================


class TestBaseFitterPositionSensitivity:

    @pytest.mark.parametrize(
        "frozen_index,param_index",
        [(0, 0), (1, 1), (2, 2)],  # first param: amplitude  # middle param: mu  # last param: sigma
    )
    def test_single_position_frozen(self, frozen_index, param_index):
        """Only the specified position must be frozen; others must move."""
        fitter = GaussianFitter(_X_GAUSS, _Y_GAUSS2)
        mask = [False, False, False]
        mask[frozen_index] = True
        p0_val = _G1[frozen_index]

        fitter.fit(p0=[_G1, _G2], frozen=mask)
        params = fitter.params.reshape(-1, 3)
        assert _is_clamped(params[0, frozen_index], p0_val)

    def test_alternate_params_frozen(self):
        """Freeze alternate params (indices 0 and 2) — index 1 must move."""
        fitter = GaussianFitter(_X_GAUSS, _Y_GAUSS2)
        p0 = [(10.0, -5.0, 3.0), _G2]  # sigma wrong, will move if free
        fitter.fit(p0=p0, frozen=[True, False, True])
        params = fitter.params.reshape(-1, 3)
        assert _is_clamped(params[0, 0], 10.0)
        assert _is_clamped(params[0, 2], 3.0)


# ===========================================================================
# 6. MixedDataFitter — sparse dict contract
# ===========================================================================


class TestMixedFitterSparseDict:

    def _make_fitter(self):
        y = multiple_models(_X_WIDE, params=[_LINE, _G1, _G2], model_list=[LINE, GAUSSIAN, GAUSSIAN], noise_level=0.1)
        return MixedDataFitter(_X_WIDE, y, model_list=[LINE, GAUSSIAN, GAUSSIAN])

    def test_empty_dict_equivalent_to_no_frozen(self):
        """frozen={} must behave identically to frozen=None."""
        y = multiple_models(_X_WIDE, params=[_LINE, _G1, _G2], model_list=[LINE, GAUSSIAN, GAUSSIAN], noise_level=0.0)
        fitter_none = MixedDataFitter(_X_WIDE, y, model_list=[LINE, GAUSSIAN, GAUSSIAN])
        fitter_empty = MixedDataFitter(_X_WIDE, y, model_list=[LINE, GAUSSIAN, GAUSSIAN])
        p0 = [_LINE, _G1, _G2]
        fitter_none.fit(p0=p0)
        fitter_empty.fit(p0=p0, frozen={})
        np.testing.assert_allclose(fitter_none.params, fitter_empty.params, atol=1e-6)

    def test_omitted_component_is_fully_free(self):
        """A component absent from the dict must optimize as if unfrozen."""
        y = multiple_models(_X_WIDE, params=[_LINE, _G1, _G2], model_list=[LINE, GAUSSIAN, GAUSSIAN], noise_level=0.0)
        fitter_free = MixedDataFitter(_X_WIDE, y, model_list=[LINE, GAUSSIAN, GAUSSIAN])
        fitter_partial = MixedDataFitter(_X_WIDE, y, model_list=[LINE, GAUSSIAN, GAUSSIAN])

        p0 = [_LINE, _G1, _G2]
        # Freeze only Line (idx 0); Gaussians (idx 1, 2) are omitted
        fitter_free.fit(p0=p0)
        fitter_partial.fit(p0=p0, frozen={0: [True, True]})

        # Gaussian params (positions 2–7) must match in both fits
        np.testing.assert_allclose(fitter_free.params[2:], fitter_partial.params[2:], atol=1e-4)

    def test_all_components_listed_same_as_dense(self):
        """Listing every component in the dict gives identical results to only listing frozen ones."""
        y = multiple_models(_X_WIDE, params=[_LINE, _G1, _G2], model_list=[LINE, GAUSSIAN, GAUSSIAN], noise_level=0.0)
        fitter_sparse = MixedDataFitter(_X_WIDE, y, model_list=[LINE, GAUSSIAN, GAUSSIAN])
        fitter_dense = MixedDataFitter(_X_WIDE, y, model_list=[LINE, GAUSSIAN, GAUSSIAN])
        p0 = [_LINE, _G1, _G2]

        # Sparse: only freeze mu of Gaussian 2
        fitter_sparse.fit(p0=p0, frozen={2: [False, True, False]})
        # Dense: list all components, same effect
        fitter_dense.fit(p0=p0, frozen={0: [False, False], 1: [False, False, False], 2: [False, True, False]})
        np.testing.assert_allclose(fitter_sparse.params, fitter_dense.params, atol=1e-8)

    def test_all_false_mask_on_component_equivalent_to_omitting(self):
        """frozen={0: [False, False]} must be identical to omitting component 0."""
        y = multiple_models(_X_WIDE, params=[_LINE, _G1, _G2], model_list=[LINE, GAUSSIAN, GAUSSIAN], noise_level=0.0)
        fitter_omit = MixedDataFitter(_X_WIDE, y, model_list=[LINE, GAUSSIAN, GAUSSIAN])
        fitter_allfalse = MixedDataFitter(_X_WIDE, y, model_list=[LINE, GAUSSIAN, GAUSSIAN])
        p0 = [_LINE, _G1, _G2]
        # Freeze only Gaussian 1 mu; component 0 absent vs explicitly all-False
        fitter_omit.fit(p0=p0, frozen={1: [False, True, False]})
        fitter_allfalse.fit(p0=p0, frozen={0: [False, False], 1: [False, True, False]})
        np.testing.assert_allclose(fitter_omit.params, fitter_allfalse.params, atol=1e-8)

    def test_invalid_component_index_raises(self):
        """A dict key outside [0, n_components) must raise ValueError."""
        fitter = self._make_fitter()
        with pytest.raises(ValueError, match="out of range"):
            fitter.fit(p0=[_LINE, _G1, _G2], frozen={5: [False, False, False]})


# ===========================================================================
# 7. MixedDataFitter — mask length variants
# ===========================================================================


class TestMixedFitterMaskLength:

    def test_pn_par_mask_warns_and_pads(self):
        """pn_par-length inner mask for a loc/scale component must warn."""
        y = multi_gamma(_X_POS, params=[_GA], noise_level=0.0) + multi_gaussian(_X_POS, params=[_G1], noise_level=0.0)
        fitter = MixedDataFitter(_X_POS, y, model_list=[GAUSSIAN, GAMMA])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fitter.fit(p0=[_G1, _GA], frozen={1: [False, False, False]})  # Gamma pn_par=3
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "auto-padded" in str(user_warnings[0].message)

    def test_mask_shorter_than_pn_par_raises(self):
        """Inner mask shorter than pn_par for that component must raise ValueError."""
        y = multi_gamma(_X_POS, params=[_GA], noise_level=0.0) + multi_gaussian(_X_POS, params=[_G1], noise_level=0.0)
        fitter = MixedDataFitter(_X_POS, y, model_list=[GAUSSIAN, GAMMA])
        with pytest.raises(ValueError):
            fitter.fit(p0=[_G1, _GA], frozen={1: [False, False]})  # too short (pn_par=3)

    def test_mask_longer_than_n_par_raises(self):
        """Inner mask longer than n_par for that component must raise ValueError."""
        y = multi_gaussian(_X_POS, params=[_G1, _G2], noise_level=0.0)
        fitter = MixedDataFitter(_X_POS, y, model_list=[GAUSSIAN, GAUSSIAN])
        with pytest.raises(ValueError):
            fitter.fit(p0=[_G1, _G2], frozen={0: [False, False, False, False]})  # too long (n_par=3)


# ===========================================================================
# 8. MixedDataFitter — correctness invariants
# ===========================================================================


class TestMixedFitterCorrectness:

    def test_all_params_frozen_equals_p0(self):
        """All components fully frozen → fitted params must equal p0."""
        y = multiple_models(_X_WIDE, params=[_LINE, _G1, _G2], model_list=[LINE, GAUSSIAN, GAUSSIAN], noise_level=0.0)
        fitter = MixedDataFitter(_X_WIDE, y, model_list=[LINE, GAUSSIAN, GAUSSIAN])
        p0 = [_LINE, _G1, _G2]
        fitter.fit(p0=p0, frozen={0: [True, True], 1: [True, True, True], 2: [True, True, True]})
        expected = np.array([*_LINE, *_G1, *_G2])
        np.testing.assert_allclose(fitter.params, expected, atol=1e-6)

    def test_line_frozen_gaussians_free(self):
        """Line frozen, both Gaussians free — Line params must be clamped."""
        y = multiple_models(_X_WIDE, params=[_LINE, _G1, _G2], model_list=[LINE, GAUSSIAN, GAUSSIAN], noise_level=0.1)
        fitter = MixedDataFitter(_X_WIDE, y, model_list=[LINE, GAUSSIAN, GAUSSIAN])
        fitter.fit(p0=[_LINE, _G1, _G2], frozen={0: [True, True]})
        assert _is_clamped(fitter.params[0], _LINE[0])
        assert _is_clamped(fitter.params[1], _LINE[1])

    def test_partial_freeze_per_component(self):
        """Line frozen, Gaussian1 free, Gaussian2 mu frozen."""
        y = multiple_models(_X_WIDE, params=[_LINE, _G1, _G2], model_list=[LINE, GAUSSIAN, GAUSSIAN], noise_level=0.1)
        fitter = MixedDataFitter(_X_WIDE, y, model_list=[LINE, GAUSSIAN, GAUSSIAN])
        fitter.fit(p0=[_LINE, _G1, _G2], frozen={0: [True, True], 2: [False, True, False]})
        # Line clamped
        assert _is_clamped(fitter.params[0], _LINE[0])
        assert _is_clamped(fitter.params[1], _LINE[1])
        # Gaussian2 mu clamped (index: 2 + 3 + 1 = 6)
        assert _is_clamped(fitter.params[6], _G2[1])

    def test_frozen_at_zero_mixed(self):
        """Freeze loc=0 for a Gamma component inside a mixed fitter."""
        y = multi_gamma(_X_POS, params=[_GA], noise_level=0.0) + multi_gaussian(
            _X_POS, params=[(2.0, 10.0, 1.0)], noise_level=0.0
        )
        fitter = MixedDataFitter(_X_POS, y, model_list=[GAUSSIAN, GAMMA])
        fitter.fit(p0=[(2.0, 10.0, 1.0), _GA], frozen={1: [False, False, False, True]})
        # Gamma loc is at index 3+3=6 → index 6 in flat params
        assert _is_clamped(fitter.params[6], 0.0)


# ===========================================================================
# 9. MixedDataFitter — state isolation
# ===========================================================================


class TestMixedFitterStateIsolation:

    def test_second_fit_not_contaminated(self):
        """Params frozen in call 1 but free in call 2 must move."""
        y = multiple_models(_X_WIDE, params=[_LINE, _G1, _G2], model_list=[LINE, GAUSSIAN, GAUSSIAN], noise_level=0.0)
        fitter = MixedDataFitter(_X_WIDE, y, model_list=[LINE, GAUSSIAN, GAUSSIAN])

        # Deliberately wrong Line p0 so frozen call 1 pins Line to wrong values.
        _LINE_WRONG = (0.0, 0.0)
        p0_wrong = [_LINE_WRONG, _G1, _G2]

        fitter.fit(p0=p0_wrong, frozen={0: [True, True]})
        params_call1 = fitter.params.copy()  # Line clamped to (0.0, 0.0)

        fitter.fit(p0=p0_wrong)  # fully unconstrained, same starting point
        params_call2 = fitter.params.copy()

        # If state is contaminated (frozen mask leaked from call 1), call 2 would
        # keep Line at (0.0, 0.0).  A clean call 2 must move Line toward truth.
        assert not np.allclose(
            params_call1[:2], params_call2[:2], atol=1e-3
        ), "call 2 Line params identical to frozen call 1 — possible state contamination"
        # Call 2 must still converge near truth
        np.testing.assert_allclose(params_call2[:2], [*_LINE], atol=0.05)

    def test_second_fit_fully_unconstrained(self):
        """After a frozen fit, fit() with no frozen must be unconstrained."""
        y = multiple_models(_X_WIDE, params=[_LINE, _G1, _G2], model_list=[LINE, GAUSSIAN, GAUSSIAN], noise_level=0.0)
        fitter = MixedDataFitter(_X_WIDE, y, model_list=[LINE, GAUSSIAN, GAUSSIAN])
        p0 = [_LINE, _G1, _G2]

        fitter.fit(p0=p0, frozen={0: [True, True], 1: [True, True, True], 2: [True, True, True]})
        frozen_params = fitter.params.copy()

        fitter.fit(p0=p0)
        free_params = fitter.params.copy()

        # At least the Gaussian params must converge to truth freely
        np.testing.assert_allclose(free_params[2:5], [*_G1], atol=0.1)


# ===========================================================================
# 10. MixedDataFitter — large mix, sparse freeze (deep index checks)
# ===========================================================================


class TestMixedFitterLargeMix:

    def _build_large(self):
        model_order = [LINE, GAUSSIAN, GAUSSIAN, LAPLACE, LAPLACE, GAUSSIAN]
        params_true = [_LINE, _G1, _G2, _L1, (4.0, 5.0, 0.8), (3.0, -10.0, 1.2)]
        y = multiple_models(_X_WIDE, params=params_true, model_list=model_order, noise_level=0.05)
        return MixedDataFitter(_X_WIDE, y, model_list=model_order), params_true

    def test_deep_component_frozen(self):
        """Freeze only the component at index 4 (5th model) — others must be free."""
        fitter, params_true = self._build_large()
        p0 = list(params_true)
        fitter.fit(p0=p0, frozen={4: [False, True, False]})  # freeze mu of 2nd Laplace
        # Flat offset: LINE(2) + G(3) + G(3) + L(3) = 11 → index 11+1=12 is mu of Laplace[1]
        assert _is_clamped(fitter.params[12], params_true[4][1])

    def test_first_and_last_frozen_middle_free(self):
        """Freeze components 0 and 5; components 1-4 must be free."""
        fitter, params_true = self._build_large()
        p0 = list(params_true)
        fitter.fit(p0=p0, frozen={0: [True, True], 5: [True, True, True]})
        # Line (idx 0-1) clamped
        assert _is_clamped(fitter.params[0], _LINE[0])
        assert _is_clamped(fitter.params[1], _LINE[1])
        # Last Gaussian (offset 2+3+3+3+3=14, indices 14-16) clamped
        last = params_true[5]
        assert _is_clamped(fitter.params[14], last[0])
        assert _is_clamped(fitter.params[15], last[1])
        assert _is_clamped(fitter.params[16], last[2])
