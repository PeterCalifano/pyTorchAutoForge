# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os, sys, runpy
sys.path.insert(0, os.path.abspath('../pyTorchAutoForge/')) # Add project root to path

# Determine the path to the _version.py file
version_file_path = os.path.join(os.path.dirname(__file__), '..', '_version.py')

# Execute the _version.py file and retrieve the __version__ variable
version_info = runpy.run_path(version_file_path)

# Only mock when on RTD
on_rtd = os.environ.get('READTHEDOCS') == 'True'

if on_rtd:

    from unittest.mock import MagicMock

    MOCK_MODULES = [
        'torchvision',
        'pycuda',  # if needed
        'pycuda.driver',
        'pycuda.autoinit',
        'numpy',
        'scipy',
        'pandas',  # Added for data manipulation
        'sklearn',  # Added for machine learning utilities
        'tensorflow',  # Added for alternative ML framework
        'cv2',  # Added for OpenCV
        'seaborn',  # Added for statistical data visualization
        'h5py',  # Added for HDF5 file handling
        'yaml',  # Added for YAML file parsing
    ]
    for mod_name in MOCK_MODULES:
        sys.modules[mod_name] = MagicMock()
    
    autodoc_mock_imports = MOCK_MODULES

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyTorchAutoForge'
copyright = '2025, Pietro Califano'
author = 'Pietro Califano'
email = 'petercalifano.gs@gmail.com'
version: str = version_info['__version__']  # Major.Minor.Patch

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    'sphinx_rtd_theme'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    #'analytics_id': 'G-XXXXXXXXXX',  # Provided by Google in your dashboard
    #'analytics_anonymize_ip': False,
    'logo_only': False,
    #'display_version': True,
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


# -- Options for autodoc extensions -----------------------------------------------------
napoleon_google_docstring = True  # Enable Google-style
napoleon_numpy_docstring = True   # Enable NumPy-style
napoleon_include_init_with_doc = False  # Don't include __init__ docstring
napoleon_use_param = True         # Use :param: for function params
napoleon_use_rtype = True         # Use :rtype: for return type

#autodoc_member_order = 'bysource'
autosummary_generate = True  # Generate .rst files for all modules

# Interspinx mapping for cross-referencing
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}


