# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import os
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# General information about the project
year = datetime.date.today().year

project = 'pymultifit'
copyright = f'2024-{year}, Syed Ali Mohsin Bukhari'
author = 'Syed Ali Mohsin Bukhari'
release = 'v1.0.4'

sys.path.insert(0, os.path.abspath('./../../'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

## COPIED from PYLOPS
sys.path.insert(0, os.path.abspath("../../src"))
extensions = ["sphinx.ext.autodoc",
              "sphinx.ext.autosummary",
              "sphinx.ext.coverage",
              "sphinx.ext.mathjax",
              "sphinx.ext.doctest",
              "sphinx.ext.extlinks",
              "sphinx.ext.intersphinx",
              # "sphinx_gallery.gen_gallery", # have to work on gallery later
              "sphinx_copybutton",
              "numpydoc",
              "nbsphinx",
              "nbsphinx_link",
              "sphinx.ext.viewcode",
              # 'sphinx.ext.napoleon',
              "matplotlib.sphinxext.plot_directive"]

nbsphinx_execute = 'auto'
source_suffix = {'.rst': 'restructuredtext', '.md': 'restructuredtext'}
suppress_warnings = ["config.cache"]

intersphinx_mapping = {"python": ("https://docs.python.org/3/", None),
                       "numpy": ('https://numpy.org/devdocs', None),
                       'scipy': ('https://docs.scipy.org/doc/scipy/', None),
                       "pandas": ("http://pandas.pydata.org/pandas-docs/stable/", None),
                       "matplotlib": ("https://matplotlib.org/stable/", None),
                       "numpydoc": ('https://numpydoc.readthedocs.io/en/latest', None), }

autosummary_generate = True
autodoc_default_flags = ["members"]
autodoc_typehints = "signature"
add_module_names = False
html_show_sourcelink = False

numpydoc_show_class_members = True
numpydoc_show_inherited_class_members = True
numpydoc_class_members_toctree = False

# sphinx_gallery_conf = {
#     # path to your examples scripts
#     "examples_dirs": ["../../examples", ],
#     # path where to save gallery generated examples
#     "gallery_dirs": ["gallery", "examples"],
#     "filename_pattern": r"\.py",
#     # Remove the "Download all examples" button from the top level gallery
#     "download_all_examples": False,
#     # Sort gallery example by file name instead of number of lines (default)
#     "within_subsection_order": ExampleTitleSortKey,
#     # Modules for which function level galleries are created.
#     "doc_module": "pymultifit",
#     # Insert links to documentation of objects in the examples
#     "reference_url": {"pymultifit": None},
# }

# Always show the source code that generates a plot
plot_include_source = True
plot_formats = ["png"]

templates_path = ['_templates']
exclude_patterns = ["_build", "**.ipynb_checkpoints", "**.ipynb", "**.md5"]

# These enable substitutions using |variable| in the rst files
rst_epilog = """
.. |year| replace:: {year}
""".format(year=year)
html_static_path = ["_static"]
html_css_files = ['custom.css']
html_last_updated_fmt = "%b %d, %Y"
html_title = "pyMultiFit"
html_short_title = "pyMultiFit"
html_extra_path = []
pygments_style = "colorful"
add_function_parentheses = True
html_show_sphinx = True
html_show_copyright = True

# Theme config
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_toc_level": 3,
    "github_url": "https://github.com/syedalimohsinbukhari/pyMultiFit",
    "navbar_end": [
        "search-button",
        "theme-switcher",
        "navbar-icon-links"
    ],
}

html_context = {
    "menu_links_name": "Repository",
    "menu_links": [
        (
            '<i class="fa fa-github fa-fw"></i> Source Code',
            "https://github.com/syedalimohsinbukhari/pyMultiFit",
        ),
    ],
    # Custom variables to enable "Improve this page"" and "Download notebook"
    # links
    "doc_path": "docs/source",
    # "galleries": sphinx_gallery_conf["gallery_dirs"],
    # "gallery_dir": dict(
    #     zip(sphinx_gallery_conf["gallery_dirs"], sphinx_gallery_conf["examples_dirs"])
    # ),
    "github_project": "pyMultiFit",
    "github_repo": "pymultifit",
    "github_version": "doc",
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

def setup(app):
    app.add_css_file("style.css")
