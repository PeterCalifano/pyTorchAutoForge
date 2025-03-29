# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os, sys
sys.path.insert(0, os.path.abspath('../pyTorchAutoForge/')) # Add project root to path

from importlib.metadata import version as get_version
release: str = get_version("pyTorchAutoForge")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyTorchAutoForge'
copyright = '2025, Pietro Califano'
author = 'Pietro Califano'
version: str = ".".join(release.split('.')[:3])  # Major.Minor.Patch

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx_rtd_theme']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    #'analytics_id': 'G-XXXXXXXXXX',  # Provided by Google in your dashboard
    #'analytics_anonymize_ip': False,
    'logo_only': False,
    'display_version': True,
    #'prev_next_buttons_location': 'bottom',
    #'style_external_links': False,
    #'vcs_pageview_mode': '',
    #'style_nav_header_background': '#2980B9',
    # Toc options
    #'collapse_navigation': True,
    #'sticky_navigation': True,
    #'navigation_depth': 4,
    #'includehidden': True,
    #'titles_only': False
}

# Add custom static files (such as style sheets)
html_static_path = ['_static']

# Add custom CSS


def setup(app):
    app.add_css_file('custom.css')


# -- Options for autodoc -----------------------------------------------------
napoleon_google_docstring = True  # Enable Google-style
napoleon_numpy_docstring = True   # Enable NumPy-style
napoleon_include_init_with_doc = False  # Don't include __init__ docstring
napoleon_use_param = True         # Use :param: for function params
napoleon_use_rtype = True         # Use :rtype: for return type
