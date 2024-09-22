# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os, sys
sys.path.insert(0, os.path.abspath('../pyTorchAutoForge/')) # Add project root to path

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyTorchAutoForge'
copyright = '2024, Pietro Califano'
author = 'Pietro Califano'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx_rtd_theme']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


# -- Options for autodoc -----------------------------------------------------
napoleon_google_docstring = True  # Enable Google-style
napoleon_numpy_docstring = True   # Enable NumPy-style
napoleon_include_init_with_doc = False  # Don't include __init__ docstring
napoleon_use_param = True         # Use :param: for function params
napoleon_use_rtype = True         # Use :rtype: for return type
