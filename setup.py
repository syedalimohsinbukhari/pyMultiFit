"""Setup py file"""

from setuptools import find_packages, setup

with open('README.md', 'r') as f:
    readme = f.read()

setup(name='pymultifit',
      version='0.1.3',
      packages=find_packages(where="src", exclude=["test"]),
      url='https://github.com/syedalimohsinbukhari/pyMultiFit',
      license='MIT',
      author='Syed Ali Mohsin Bukhari',
      author_email='syedali.b@outlook.com',
      description='A library to fit fit the data with multiple fitters.',
      long_description=readme,
      long_description_content_type="text/markdown",
      python_requires=">=3.9",
      install_requires=["setuptools", "numpy==1.26.4", "matplotlib", "scipy"],
      include_package_data=True,
      classifiers=["License :: OSI Approved :: MIT License",
                   "Programming Language :: Python :: 3.9",
                   "Programming Language :: Python :: 3.10",
                   "Programming Language :: Python :: 3.11"],
      )
