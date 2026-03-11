#!/usr/bin/env python
"""Report marimo notebooks that are missing an H1 title."""

import sys
from pathlib import Path

from utils import get_notebook_title


def main():
    root = Path(__file__).parent.parent
    notebooks = sorted(root.glob("*/[0-9]*.py"))
    missing = [nb for nb in notebooks if get_notebook_title(nb) is None]
    if missing:
        for nb in missing:
            print(nb.relative_to(root))
        sys.exit(1)


if __name__ == "__main__":
    main()
