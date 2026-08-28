"""Configuration file for the Sphinx documentation builder."""

import datetime
import os

from pymultifit.version import __version__

# General information about the project
year = datetime.date.today().year

project = "pymultifit"
copyright = f"2024-{year}, Syed Ali Mohsin Bukhari"
author = "Syed Ali Mohsin Bukhari"
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "nbsphinx",
    "nbsphinx_link",
    "sphinx.ext.viewcode",
    "matplotlib.sphinxext.plot_directive",
]

nbsphinx_execute = "auto"
source_suffix = {".rst": "restructuredtext", ".md": "restructuredtext"}
suppress_warnings = ["config.cache"]

# intersphinx_mapping = {
#     "python": ("https://docs.python.org/3/", None),
#     "numpy": ("https://numpy.org/doc/stable/", None),
#     "scipy": ("https://docs.scipy.org/doc/scipy/", None),
#     "pandas": ("http://pandas.pydata.org/pandas-docs/stable/", None),
#     "matplotlib": ("https://matplotlib.org/stable/", None),
#     "numpydoc": ("https://numpydoc.readthedocs.io/en/latest", None),
# }

# Add banner for review version
if os.environ.get("READTHEDOCS_VERSION") == "pyopensci-review":
    rst_prolog = """
    .. warning::

        This is a **review version** of the documentation for PyOpenSci submission.
    """

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {"members": True, "member-order": "groupwise", "undoc-members": True}
autodoc_type_aliases = {
    "ArrayLike": "~pymultifit.typing.ArrayLike",
    "NDArray": "~pymultifit.typing.NDArray",
    "Sequence": "~collections.abc.Sequence",
    "Params_": "~pymultifit.typing.Params_",
}
autodoc_typehints_format = "short"
autodoc_typehints = "description"

# Autosummary settings
autosummary_generate = True
add_module_names = False
html_show_sourcelink = False

plot_include_source = True
plot_formats = ["png"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints", "**.ipynb", "**.md5"]

# These enable substitutions using |variable| in the rst files
rst_epilog = """
.. |year| replace:: {year}
""".format(year=year)
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_last_updated_fmt = "%b %d, %Y"
html_title = f"pyMultiFit {__version__}"
html_short_title = "pyMultiFit"
html_extra_path = []
pygments_style = "sphinx"
add_function_parentheses = True
html_show_sphinx = True
html_show_copyright = True
show_version_warning_banner = True

# Theme config
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_toc_level": 3,
    "github_url": "https://github.com/syedalimohsinbukhari/pyMultiFit",
    "navbar_end": ["search-button", "theme-switcher", "navbar-icon-links"],
    "back_to_top_button": "True",
}

html_context = {
    "menu_links_name": "Repository",
    "menu_links": [
        ('<i class="fa fa-github fa-fw"></i> Source Code', "https://github.com/syedalimohsinbukhari/pyMultiFit")
    ],
    "doc_path": "docs/source",
    "github_project": "pyMultiFit",
    "github_repo": "pymultifit",
    "github_version": "doc",
}


def setup(app):
    app.add_css_file("style.css")
