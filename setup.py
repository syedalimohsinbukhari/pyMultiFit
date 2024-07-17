from setuptools import find_packages
from setuptools import setup

with open('README.md', 'r') as f:
    readme = f.read()

setup(
        name='pymultifit',
        version='0.1.0',
        packages=find_packages(where="src"),
        url='https://github.com/syedalimohsinbukhari/pyMultiFit',
        license='MIT',
        author='Syed Ali Mohsin Bukhari',
        author_email='syedali.b@outlook.com',
        description='',
        long_description=readme,
        long_description_content_type="text/markdown",
        python_requires=">=3.10",
        install_requires=["setuptools~=73.3.0"],
        include_package_data=True,
        classifiers=[
                "License :: OSI Approved :: MIT License",
                "Programming Language :: Python :: 3.10",
                "Programming Language :: Python :: 3.11"],
        )
