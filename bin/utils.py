"""Utility functions for working with marimo notebooks."""

import re
from pathlib import Path


def get_notebook_title(path: Path) -> str | None:
    """Return the first H1 Markdown heading in a marimo notebook, or None."""
    text = path.read_text(encoding="utf-8")
    for match in re.finditer(r'mo\.md\(r?f?"""(.*?)"""', text, re.DOTALL):
        for line in match.group(1).splitlines():
            if line.strip().startswith("# "):
                return line.strip()[2:].strip()
    return None
