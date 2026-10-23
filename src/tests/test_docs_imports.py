"""Every ``pymultifit`` import line shown in the docs and the README must work, and the public classes must be exported."""

import pathlib
import re

import pytest

import pymultifit.distributions as distributions
from pymultifit.distributions import generalized

ROOT = pathlib.Path(__file__).resolve().parents[2]
IMPORT_LINE = re.compile(r"^\s*(from\s+pymultifit[\w.]*\s+import\s+[^\n(\\]+|import\s+pymultifit[^\n]*)$", re.M)


def _snippets():
    files = list((ROOT / "docs" / "source").rglob("*.rst")) + [ROOT / "README.md", ROOT / "CONTRIBUTING.md"]
    return sorted(
        {
            (f.relative_to(ROOT).as_posix(), m.group(1).strip())
            for f in files
            if f.exists()
            for m in IMPORT_LINE.finditer(f.read_text())
        }
    )


SNIPPETS = _snippets()


@pytest.mark.skipif(not SNIPPETS, reason="the docs are not part of this checkout")
@pytest.mark.parametrize("path, statement", SNIPPETS)
def test_documented_import_works(path, statement):
    exec(statement, {})  # noqa: S102


@pytest.mark.parametrize("name", generalized.__all__)
def test_generalized_distributions_are_exported_from_distributions(name):
    assert getattr(distributions, name) is getattr(generalized, name)


def test_exceptions_are_importable_from_the_package_root():
    import pickle

    from pymultifit import AxesError, pyMultiFitErrors

    assert issubclass(AxesError, pyMultiFitErrors)
    assert isinstance(pickle.loads(pickle.dumps(AxesError("two axes"))), AxesError)
