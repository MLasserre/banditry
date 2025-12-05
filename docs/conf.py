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
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

myst_enable_extensions = ["colon_fence", "dollarmath"]

# Execute notebooks as needed during doc build
nb_execution_mode = "auto"

exclude_patterns = [
    "_build",
    "**/_build/**",
    "jupyter_execute/*",
    "**/jupyter_execute/*",
    "**/.ipynb_checkpoints",
]

templates_path = ["_templates"]

html_theme = "furo"
html_static_path = ["_static"]
