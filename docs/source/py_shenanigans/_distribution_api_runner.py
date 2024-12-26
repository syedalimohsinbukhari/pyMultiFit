"""Created on Dec 22 18:04:13 2024"""

from docs.source.py_shenanigans._distribution_api_template import generate_distribution_rst_simple

dist_names = ['ArcSine', 'Beta', 'ChiSquare', 'Exponential', 'FoldedNormal', 'Gaussian', 'HalfNormal', 'Laplace',
              'LogNormal', 'SkewNormal', 'Uniform']

for dist in dist_names:
    module_name, rst_content = generate_distribution_rst_simple(dist)

    # Save to an .rst file
    with open(f"./../distributions/{module_name}_d.rst", "w") as file:
        file.write(rst_content)
