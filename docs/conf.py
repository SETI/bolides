# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------
# Feel free to add to the author if you make substantial contributions to the
# docs!

project = 'bolides'
copyright = ''
author = 'Anthony Ozerov, Jeffrey Smith and the NASA ATAP team'

# The full version, including alpha/beta/rc tags
release = '0.6.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'numpydoc',
              'sphinx.ext.intersphinx',
              'nbsphinx',
              'sphinx.ext.viewcode']

autosummary_generate = True
numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

html_favicon = "favicon.svg"

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

html_context = {"default_mode": "light"}
html_theme_options = {
    "external_links": [{'url': 'https://bolides.seti.org', 'name': '🌠 webapp'}],
    "github_url": "https://github.com/jcsmithhere/bolides",
    "navbar_end": ["navbar-icon-links"],
    "footer_items": ["footer"]
}


html_sidebars = {
    "tutorials/*": [],
    "tutorials/*/*": [],
    "tutorials/*/*/*": [],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

default_role = 'py:obj'
intersphinx_mapping = {'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
                       'geopandas': ('https://geopandas.org/en/stable/', None),
                       'astropy': ('https://docs.astropy.org/en/latest/', None),
                       'shapely': ('https://shapely.readthedocs.io/en/stable/', None),
                       'cartopy': ('https://scitools.org.uk/cartopy/docs/latest/', None),
                       'plotly': ('https://plotly.com/python-api-reference/', None),
                       'lightkurve': ('http://docs.lightkurve.org', None),
                       'poliastro': ('https://docs.poliastro.space/en/stable/', None)}
