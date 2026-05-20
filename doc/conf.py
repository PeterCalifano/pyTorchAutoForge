from __future__ import annotations

import os
from pathlib import Path
import sys
import logging

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


class _AutoapiPlaceholderWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Unknown type: placeholder" not in record.getMessage()


logging.getLogger("autoapi._mapper").addFilter(_AutoapiPlaceholderWarningFilter())

try:
    from autoapi._mapper import Mapper

    _AUTOAPI_CREATE_CLASS = Mapper.create_class

    def _CreateClassSkippingPlaceholders(self: Mapper, data: dict, options: object = None):
        if data.get("type") == "placeholder":
            return
        yield from _AUTOAPI_CREATE_CLASS(self, data, options=options)

    Mapper.create_class = _CreateClassSkippingPlaceholders
except Exception:
    pass

project = "pyTorchAutoForge"
author = "Pietro Califano"
copyright = "2026, Pietro Califano"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "autoapi.extension",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = [
    "_build",
    "_autoapi_templates",
    "_autoapi_templates/**",
    "Thumbs.db",
    ".DS_Store",
    "developments",
    "developments/**",
]
suppress_warnings = [
    "autoapi",
    "autoapi.python_import_resolution",
    "docutils",
    "toc.not_included",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

autoapi_type = "python"
autoapi_dirs = [str(REPO_ROOT / "pyTorchAutoForge")]
autoapi_root = "api/generated"
autoapi_template_dir = "_autoapi_templates"
autoapi_add_toctree_entry = False
autoapi_keep_files = False
autoapi_member_order = "bysource"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_ignore = [
    "*/.deprecated/*",
    "*/.experimental/*",
    "*/.experimental.py",
    "*/extra/*",
    "*/programs/*",
    "*/tensorboard/*",
    "*/model_building/factories/*",
    "*/utils/pytest_test.py",
    "*/utils/test_fixtures/*",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

html_theme = "pydata_sphinx_theme"
html_logo = "assets/ptaf_logo_small.jpg"
html_static_path = ["_static"]
html_title = "pyTorchAutoForge"
html_baseurl = os.environ.get("PTAF_DOC_BASE_URL", "")
doc_version = os.environ.get("PTAF_DOC_VERSION", "stable")
switcher_json_url = os.environ.get("PTAF_DOC_SWITCHER_JSON_URL", "_static/switcher.json")
html_theme_options = {
    "show_toc_level": 2,
    "navbar_align": "left",
    "navigation_depth": 3,
    "collapse_navigation": False,
    "switcher": {
        "json_url": switcher_json_url,
        "version_match": doc_version,
    },
    "navbar_end": [
        "theme-switcher",
        "version-switcher",
        "navbar-icon-links",
    ],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/PeterCalifano/pyTorchAutoForge",
            "icon": "fa-brands fa-github",
        },
    ],
}
