Installation
============

This section guides you through the installation process for **pyMultiFit**.
Whether you're a user or a developer, follow the steps below to get started quickly.

**pyMultiFit** depends on a few core libraries to ensure smooth functionality:

- `numpy<2.1.0 <https://numpy.org>`_
- `matplotlib <https://matplotlib.org>`_
- `scipy <https://scipy.org>`_
- `mpyez <https://github.com/syedalimohsinbukhari/mpyez>`_
- `custom-inherit <https://github.com/rsokl/custom_inherit>`_
- `Deprecated <https://github.com/laurent-laporte-pro/deprecated>`_

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

   Use the `requirements[dev].txt` file to completely install all dependencies at once:

   .. code-block:: bash

      pip install -r requirements[dev].txt

Using Conda
^^^^^^^^^^^

1. **Create a Conda environment**:

   Use the `environment.yml` file in the repository:

   .. code-block:: bash

      conda env create -f environment.yml

2. **Activate the Conda environment**:

   .. code-block:: bash

      conda activate multifit

**Next Steps**
Now that you have installed **pyMultiFit**, head over to the :doc:`tutorials` section to start exploring its features and capabilities.
