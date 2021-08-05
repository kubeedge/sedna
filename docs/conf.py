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
import shutil
import subprocess

import sphinx_rtd_theme

try:
    import m2r2
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "m2r2"])

try:
    import autoapi
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip",
                           "install", "sphinx-autoapi"])


_base_path = os.path.abspath('..')
sys.path.append(os.path.join(_base_path, "lib"))
sys.path.append(_base_path)

extra_paths = [
    os.path.join(_base_path, "examples"),
]
for p in extra_paths:
    dst = os.path.join(
        _base_path, "docs",
        os.path.basename(p)
    )
    if os.path.isfile(dst):
        os.remove(dst)
    elif os.path.isdir(dst):
        shutil.rmtree(dst)
    if os.path.isdir(p):
        shutil.copytree(p, dst)
    else:
        shutil.copy2(p, dst)


with open('../lib/sedna/VERSION', "r", encoding="utf-8") as fh:
    __version__ = fh.read().strip()

# -- Project information -----------------------------------------------------

project = 'Sedna'
copyright = '2020, Kubeedge'
author = 'Kubeedge'

version = __version__
release = __version__
# -- General configuration ---------------------------------------------------


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "m2r2",
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "autoapi.extension",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon"
]

autodoc_inherit_docstrings = False
autodoc_member_order = "bysource"
# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The master toctree document
master_doc = 'index'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

html_static_path = ['_static']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_last_updated_fmt = "%b %d, %Y"
html_theme_options = {
    'prev_next_buttons_location': 'both'
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

autoapi_type = "python"
autoapi_dirs = ["../lib/sedna"]
autoapi_options = ['members', 'undoc-members', 'show-inheritance', 'show-module-summary', 'special-members', 'imported-members']

extlinks = {
    "issue": ("https://github.com/kubeedge/sedna/issues/%s", "#"),
    "pr": ("https://github.com/kubeedge/sedna/pull/%s", "PR #"),
}



def setup(app):
    app.add_css_file('css/custom.css')
