"""Created on Oct 06 2026

The plotting and CI layers must not depend on the fitters, and the fitter plotting API must work before ``fit()``.
"""

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ...pymultifit.fitters import GaussianFitter
from ...pymultifit.fitters.mixed_f import MixedDataFitter

_SRC = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("module", ["pymultifit.plot", "pymultifit.ci", "pymultifit.result"])
def test_module_does_not_import_fitters(module):
    code = f"import sys, {module}; bad = [m for m in sys.modules if m.startswith('pymultifit.fitters')]; assert not bad, bad"
    proc = subprocess.run([sys.executable, "-c", code], cwd=_SRC, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_dry_run_before_fit():
    x = np.linspace(-5, 5, 50)
    for fitter in (GaussianFitter(x, np.exp(-(x**2))), MixedDataFitter(x, np.exp(-(x**2)), model_list=["gaussian"])):
        fitter.dry_run(is_scatter=True)
        plt.close("all")


def test_plotter_cache_is_invalidated_by_fit():
    x = np.linspace(-5, 5, 200)
    fitter = GaussianFitter(x, 3 * np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi))
    pre_fit = fitter.plotter
    assert not pre_fit.result.is_fitted

    fitter.fit(p0=[(3.0, 0.0, 1.0)])
    assert fitter.plotter is not pre_fit
    assert fitter.plotter.result.is_fitted
    assert fitter.plotter is fitter.plotter  # cached between fits


def test_plotting_before_fit_raises():
    x = np.linspace(-5, 5, 50)
    with pytest.raises(RuntimeError, match="Fit not performed"):
        GaussianFitter(x, np.exp(-(x**2))).plot_fit()
    plt.close("all")
