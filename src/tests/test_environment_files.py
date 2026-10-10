"""Created on Oct 06 2026

The conda environment files are maintained by hand, so check that they stay in sync with ``pyproject.toml``.
"""

import re
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = pytest.importorskip("tomli")

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())


def _name(requirement: str) -> str:
    """Normalised distribution name of a requirement / conda spec (no extras, markers or version)."""
    base = re.split(r"[<>=!~;\[ ]", requirement.strip(), maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", base).lower()


def _environment(path: Path) -> tuple[set[str], str]:
    """Names listed in a conda environment file (conda and ``pip:`` sections) and its ``python`` spec."""
    names: set[str] = set()
    python = ""
    for raw in path.read_text().splitlines():
        line = raw.split(" #")[0].rstrip() if not raw.lstrip().startswith("#") else ""
        match = re.match(r"^\s*-\s+(\S.*)$", line)
        if not match or match.group(1).startswith("pip:"):
            continue
        item = match.group(1).strip()
        if _name(item) == "python":
            python = item
        else:
            names.add(_name(item))
    return names, python


RUNTIME = {_name(dep) for dep in PYPROJECT["project"]["dependencies"]}
DEV_GROUP = {_name(dep) for dep in PYPROJECT["dependency-groups"]["dev"]}
ENV, ENV_PYTHON = _environment(ROOT / "environment.yaml")
ENV_DEV, ENV_DEV_PYTHON = _environment(ROOT / "environment-dev.yaml")


def test_runtime_environment_has_all_runtime_dependencies():
    assert not RUNTIME - ENV, f"missing from environment.yaml: {sorted(RUNTIME - ENV)}"


def test_dev_environment_has_runtime_and_dev_dependencies():
    expected = RUNTIME | DEV_GROUP
    assert not expected - ENV_DEV, f"missing from environment-dev.yaml: {sorted(expected - ENV_DEV)}"


def test_runtime_environment_has_no_dev_only_packages():
    dev_only = (DEV_GROUP - RUNTIME) & ENV
    assert not dev_only, f"dev-only packages in environment.yaml: {sorted(dev_only)}"


@pytest.mark.parametrize("env", [ENV, ENV_DEV], ids=["environment", "environment-dev"])
def test_environment_does_not_list_the_project_itself(env):
    assert "pymultifit" not in env


@pytest.mark.parametrize("python", [ENV_PYTHON, ENV_DEV_PYTHON], ids=["environment", "environment-dev"])
def test_python_version_matches_requires_python(python):
    assert python == f"python{PYPROJECT['project']['requires-python']}"
