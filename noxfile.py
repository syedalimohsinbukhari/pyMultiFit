"""Created on May 26 12:41:00 2025"""

import nox

nox.options.default_venv_backend = "uv"
nox.options.reuse_venv = "yes"

# Python versions to test across (optional)
PYTHON_VERSIONS = ["3.9"]

# Define the locations to lint/test/format
CODE_LOCATIONS = ["src/pymultifit/", "src/tests/", "noxfile.py"]


@nox.session
def lint(session):
    """Lint the code using Ruff and Pylint."""
    session.install(".", "ruff", "pylint")
    session.run("ruff", "check", *CODE_LOCATIONS, "--no-fix")
    session.run("pylint", CODE_LOCATIONS[0], "--rcfile=./.pylintrc", "--fail-under=7")


@nox.session
def type_check(session):
    """Type-check the code using Pyright."""
    session.install(".", "mypy", "pip")
    session.run("mypy", "--install-types", "--non-interactive", CODE_LOCATIONS[0])


@nox.session
def tests(session):
    """Run unit tests with pytest."""
    session.install(".", "pytest", "Deprecated")
    session.run("pytest", "./src/tests/")  # adjust the path if needed
