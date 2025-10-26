"""Created on Jul 15 09:35:00 2025"""

import re
from pathlib import Path

# Config
REQUIREMENTS = Path("requirements.txt")
INSTALL_RST = Path("docs/source/installation.rst")

URL_OVERRIDES = {
    "python": "https://www.python.org/",
    "numpy": "https://numpy.org",
    "scipy": "https://scipy.org",
    "matplotlib": "https://matplotlib.org",
    "mpyez": "https://github.com/syedalimohsinbukhari/mpyez",
    "custom-inherit": "https://github.com/rsokl/custom_inherit",
    "deprecated": "https://github.com/laurent-laporte-pro/deprecated",
}


# Helper to extract name (basic)
def get_package_name(req_line):
    return re.split(r"[<>=]", req_line.strip())[0].lower()


# Parse and enrich
with REQUIREMENTS.open() as f:
    lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

formatted = []
for line in lines:
    name = get_package_name(line)
    url = URL_OVERRIDES.get(name, "")
    if url:
        formatted.append(f"- `{line} <{url}>`_")
    else:
        formatted.append(f"- ``{line}``")

block = "\n" + "\n".join(formatted) + "\n\n"

# Replace block between markers
content = INSTALL_RST.read_text()
updated = re.sub(
    r"(?s)(^\s*\.\. BEGIN REQUIREMENTS\s*$)(.*?)(^\s*\.\. END REQUIREMENTS\s*$)", block, content, flags=re.DOTALL
)
INSTALL_RST.write_text(updated)
