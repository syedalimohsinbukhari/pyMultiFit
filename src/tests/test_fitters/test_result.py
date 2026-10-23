"""Created on Oct 06 2026

``FitResult`` / ``to_result()`` must agree with the fitters' own internals.
"""

import numpy as np
import pytest

from ...pymultifit import GAUSSIAN, LINE
from ...pymultifit.fitters import GaussianFitter, LineFitter
from ...pymultifit.fitters.mixed_f import MixedDataFitter
from ...pymultifit.generators import multi_gaussian, multiple_models

_G1 = (10.0, -5.0, 1.5)
_G2 = (6.0, 3.0, 2.0)
_G3 = (4.0, 12.0, 1.0)
_LINE = (0.5, 2.0)
_X = np.linspace(-30, 30, 800)


def _gaussian(n: int) -> GaussianFitter:
    params = [_G1, _G2, _G3][:n]
    fitter = GaussianFitter(_X, multi_gaussian(_X, params=params))
    fitter.fit(p0=params)
    return fitter


def _mixed(models, params) -> MixedDataFitter:
    fitter = MixedDataFitter(_X, multiple_models(_X, params=params, model_list=models), model_list=models)
    fitter.fit(p0=params)
    return fitter


@pytest.fixture(
    params=[
        pytest.param(lambda: _gaussian(1), id="base-1"),
        pytest.param(lambda: _gaussian(3), id="base-3"),
        pytest.param(lambda: _mixed([LINE, GAUSSIAN, GAUSSIAN], [_LINE, _G1, _G2]), id="mixed-line+2g"),
        pytest.param(lambda: _mixed([GAUSSIAN, LINE], [_G1, _LINE]), id="mixed-g+line"),
    ]
)
def fitter(request):
    return request.param()


def test_model_matches_n_fitter(fitter):
    result = fitter.to_result()
    np.testing.assert_allclose(result.model(), fitter._n_fitter(fitter.x_values, *fitter.params))
    np.testing.assert_allclose(result.model(), fitter.get_fitted_curve())


def test_model_with_explicit_x_and_params(fitter):
    result = fitter.to_result()
    x = np.linspace(-10, 10, 37)
    params = fitter.params * 1.1
    np.testing.assert_allclose(result.model(x, params), fitter._n_fitter(x, *params))


def test_residuals_match(fitter):
    np.testing.assert_allclose(fitter.to_result().residuals(), fitter.get_residuals())


def test_components_sum_to_model(fitter):
    result = fitter.to_result()
    total = sum(result.component_curve(i) for i in range(result.n_fits))
    np.testing.assert_allclose(total, result.model())


def test_structure_is_consistent(fitter):
    result = fitter.to_result()
    assert sum(c.n_par for c in result.components) == len(result.params) == len(result.param_labels)
    assert result.n_fits == fitter.n_fits
    np.testing.assert_allclose(result.errors, np.sqrt(np.diag(fitter.covariance)))


def test_param_labels():
    assert _gaussian(2).to_result().param_labels == ("p1", "p2", "p3", "p4", "p5", "p6")

    mixed = _mixed([LINE, GAUSSIAN], [_LINE, _G1]).to_result()
    assert mixed.param_labels == ("Line_1_p1", "Line_1_p2", "Gaussian_2_p1", "Gaussian_2_p2", "Gaussian_2_p3")
    assert [c.label for c in mixed.components] == ["Line", "Gaussian"]


def test_component_label_for_base_fitter():
    assert [c.label for c in _gaussian(2).to_result().components] == ["Gaussian", "Gaussian"]


def test_prefit_result_is_empty_but_usable():
    result = LineFitter(_X, 2 * _X + 1).to_result()
    assert not result.is_fitted
    assert result.params is None and result.components == ()
    assert len(result.x) == len(result.y) == len(_X)
    with pytest.raises(RuntimeError, match="Fit not performed"):
        result.model()
    with pytest.raises(RuntimeError, match="Fit not performed"):
        result.residuals()


def test_result_reflects_latest_fit_only():
    fitter = _gaussian(2)
    before = fitter.to_result()
    curve_before = fitter.get_fitted_curve().copy()
    fitter.fit(p0=[_G1, _G3])
    after = fitter.to_result()
    assert not np.allclose(before.params, after.params)
    np.testing.assert_allclose(before.model(), curve_before)  # an old snapshot is not mutated by a later fit
    np.testing.assert_allclose(after.model(), fitter.get_fitted_curve())
