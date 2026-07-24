"""Sphinx configuration for Central Limit Theorem Simulation documentation."""

project = "Central Limit Theorem Simulation"
copyright = "2024, Your Name"
author = "Your Name"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_member_order = "bysource"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
}
