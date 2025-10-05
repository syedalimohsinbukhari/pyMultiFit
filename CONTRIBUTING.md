# Contributing to `pyMultiFit`

Thank you for your interest in contributing to `pyMultiFit`!
We welcome all contributions — whether it’s bug fixes, new features, improving documentation, or adding examples.

---

## How to Contribute

### 1. Reporting Issues

* Use the [GitHub Issues](https://github.com/syedalimohsinbukhari/pyMultiFit/issues) page to report bugs or request features.
* Before opening a new issue, please check if it already exists.
* Provide as much detail as possible:

  * Steps to reproduce
  * Expected vs actual behavior
  * Error messages, logs, or screenshots

---

### 2. Development Setup

Clone the repo and set up your environment:

```bash
git clone https://github.com/syedalimohsinbukhari/pyMultiFit.git
cd pyMultiFit

# create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

# install dependencies
pip install -r requirements[dev].txt

# install the package in editable mode
pip install -e .
```

---

### 3. Making Changes

* Create a new branch for your changes:

```bash
git checkout -b feature/your-feature-name
```

* Follow our coding style:

  * Use **ruff** for formatting.
  * Use **nox** for automated testing and linting.
  * Add **docstrings** for all public functions and classes.
  * Keep commits heading small but add a good description.

---

### 4. Testing

We use **pytest** for testing.

```bash
pytest /src/test/
```

* Please write tests for new features and bug fixes.
* All tests must pass before submitting a pull request.

---

### 5. Submitting Pull Requests

* Push your branch to GitHub:

```bash
git push origin feature/your-feature-name
```

* Open a **Pull Request** (PR) against the `main` branch.
* In your PR description, include:

  * What the change does
  * Why it’s necessary
  * Links to related issues (if any)

---

### 6. Code of Conduct

By participating in this project, you agree to follow our [Code of Conduct](https://github.com/syedalimohsinbukhari/pyMultiFit/blob/main/CODE_OF_CONDUCT.md).
Be respectful, collaborative, and kind.

---

## Types of Contributions

* Bug reports / fixes
* New features
* Documentation improvements
* Unit tests and benchmarks
* Example notebooks / plots

---

## Need Help?

If you’re unsure about anything, feel free to open a **Draft PR** or start a discussion. We’re happy to guide you!
