Installation
============

This section guides you through the installation process for **pyMultiFit**.
Whether you're a user or a developer, follow the steps below to get started quickly.

**pyMultiFit** depends on a few core libraries to ensure smooth functionality:

- `numpy <https://numpy.org>`_
- `scipy <https://scipy.org>`_
- `matplotlib <https://matplotlib.org>`_
- `plotez <https://github.com/syedalimohsinbukhari/plotez>`_
- `statsmodels <https://www.statsmodels.org>`_
- `custom-inherit <https://github.com/rsokl/custom_inherit>`_
- `deprecation <https://github.com/briancurtin/deprecation>`_

Python 3.10 or newer is required.

-------------------------------

Installation for Users
-----------------------

If you are a **user** looking to install and use the library, follow these steps.

Using Pip with Virtual Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Create a virtual environment**:

   .. code-block:: bash

      python -m venv multifit-env

2. **Activate the virtual environment**:

   - On Linux/macOS:

     .. code-block:: bash

        source multifit-env/bin/activate

   - On Windows:

     .. code-block:: bash

        .\multifit-env\Scripts\activate

3. **Install the library**:

   .. code-block:: bash

      pip install pymultifit

4. **Verify the installation**:

   .. code-block:: bash

      python -c "import pymultifit; print('pyMultiFit installed successfully!')"

Using Conda
^^^^^^^^^^^

1. **Create a new Conda environment**:

   .. code-block:: bash

      conda create -n multifit python=3.10

2. **Activate the environment**:

   .. code-block:: bash

      conda activate multifit

3. **Install the library**:

   .. code-block:: bash

      pip install pymultifit

4. **Verify the installation**:

   .. code-block:: bash

      python -c "import pymultifit; print('pyMultiFit installed successfully!')"

--------------------------------

Installation for Developers
---------------------------

If you are a **developer** looking to contribute or set up the library for development purposes, follow these steps for a complete setup.

1. **Fork** the repository:
   Visit the `pyMultiFit repository <https://github.com/syedalimohsinbukhari/pyMultiFit>`_ and fork it to your GitHub account.

2. **Clone** the repository:

   .. code-block:: bash

      git clone https://github.com/<YOUR-USERNAME>/pyMultiFit.git

3. Alternatively, download the ZIP archive from the `main branch <https://codeload.github.com/syedalimohsinbukhari/pyMultiFit/zip/refs/heads/main>`_ and extract it.

4. Use **pip with a virtual environment** or **conda** to set up the development environment.

Using Pip with Virtual Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Create a virtual environment**:

   .. code-block:: bash

      python -m venv multifit-env

2. **Activate the virtual environment**:

   - On Linux/macOS:

     .. code-block:: bash

        source multifit-env/bin/activate

   - On Windows:

     .. code-block:: bash

        .\multifit-env\Scripts\activate

3. **Install dependencies**:

   Use the ``requirements-dev.txt`` file (an export of the locked runtime *and* development dependencies, which also installs ``pymultifit`` in editable mode) to install everything at once:

   .. code-block:: bash

      pip install -r requirements-dev.txt

   ``requirements.txt`` holds the runtime dependencies only.

   Alternatively, if you use `uv <https://docs.astral.sh/uv>`_, ``uv sync`` creates the environment from ``pyproject.toml`` and ``uv.lock``.

Using Conda
^^^^^^^^^^^

1. **Create a Conda environment**:

   Use the ``environment-dev.yaml`` file in the repository (runtime dependencies plus the development tools; ``environment.yaml`` holds the runtime dependencies only):

   .. code-block:: bash

      conda env create -f environment-dev.yaml

2. **Activate the Conda environment**:

   .. code-block:: bash

      conda activate pymultifit-dev

3. **Install the library** from the checkout in editable mode:

   .. code-block:: bash

      pip install -e .

**Next Steps**
Now that you have installed **pyMultiFit**, head over to the :doc:`tutorials` section to start exploring its features and capabilities.
