"""Created on Oct 06 2026

``_normal_quantiles`` replaced ``statsmodels``' ``ProbPlot(data, dist=norm, fit=True)``; the oracle holds the values
``ProbPlot`` produced for fixed inputs, so the Q-Q plots stay numerically unchanged.
"""

from pathlib import Path

import numpy as np
import pytest

from ...pymultifit.plot._plot_backend import _normal_quantiles

ORACLE = np.load(Path(__file__).parent / "data" / "qq_oracle.npz")
CASES = sorted({key.split("/")[0] for key in ORACLE.files})


@pytest.mark.parametrize("case", CASES)
def test_matches_statsmodels_probplot(case):
    theoretical, sample = _normal_quantiles(ORACLE[f"{case}/data"])
    np.testing.assert_allclose(theoretical, ORACLE[f"{case}/theoretical"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sample, ORACLE[f"{case}/sample"], rtol=1e-12, atol=1e-12)


def test_input_order_does_not_matter():
    data = np.random.default_rng(0).normal(size=30)
    np.testing.assert_allclose(_normal_quantiles(data)[1], _normal_quantiles(data[::-1])[1])
