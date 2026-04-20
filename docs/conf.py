# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.

import os
import sys


# -- Path setup --------------------------------------------------------------

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "selkit"
copyright = "2026 Jacob L. Steenwyk"
author = "Jacob L. Steenwyk <jlsteenwyk@gmail.com>"


# -- General configuration ---------------------------------------------------

extensions = ["sphinx.ext.githubpages"]

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = None


# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_theme_options = {
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}


# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = "selkitdoc"


# -- Options for LaTeX output ------------------------------------------------

latex_elements: dict = {}

latex_documents = [
    (
        master_doc,
        "selkit.tex",
        "selkit Documentation",
        "Jacob L. Steenwyk \\textless{}jlsteenwyk@gmail.com\\textgreater{}",
        "manual",
    ),
]


# -- Options for manual page output ------------------------------------------

man_pages = [(master_doc, "selkit", "selkit Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (
        master_doc,
        "selkit",
        "selkit Documentation",
        author,
        "selkit",
        "Python reimplementation of PAML selection-analysis workflows.",
        "Miscellaneous",
    ),
]


# -- Setup -------------------------------------------------------------------


def setup(app):
    app.add_css_file("custom.css")
