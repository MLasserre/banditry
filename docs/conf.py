import os
import sys
from datetime import datetime

project = "banditry"
author = "MLasserre"
year = datetime.now().year
copyright = f"{year}, {author}"

# Add src to sys.path so autodoc can find the package if needed
sys.path.insert(0, os.path.abspath("../src"))

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

myst_enable_extensions = ["colon_fence"]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "furo"
html_static_path = ["_static"]
