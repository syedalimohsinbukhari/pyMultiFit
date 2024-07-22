from setuptools import find_packages, setup

with open('README.md', 'r') as f:
    readme = f.read()

setup(
    name='pymultifit',
    version='0.1.0',
    packages=find_packages(where="src", exclude=["test", "examples"]),
    url='https://github.com/syedalimohsinbukhari/pyMultiFit',
    license='MIT',
    author='Syed Ali Mohsin Bukhari',
    author_email='syedali.b@outlook.com',
    description='',
    long_description=readme,
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    install_requires=["setuptools==70.3.0", "numpy==1.26.4",
                      "matplotlib==3.9.1", "scipy==1.13.1"],
    include_package_data=True,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11"],
)
