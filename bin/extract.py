#!/usr/bin/env python
"""Extract lesson metadata and notebook lists into a JSON file."""

import argparse
import json
import re
from pathlib import Path

import frontmatter


NOTEBOOK_PATTERN = re.compile(r"^\d{2}_.*\.py$")


def extract_lessons(root: Path) -> dict:
    lessons = {}
    for index_file in sorted(root.glob("*/index.md")):
        lesson_dir = index_file.parent
        post = frontmatter.load(index_file)
        notebooks = sorted(
            p.name
            for p in lesson_dir.glob("*.py")
            if NOTEBOOK_PATTERN.match(p.name)
        )
        lessons[lesson_dir.name] = {
            **post.metadata,
            "notebooks": notebooks,
        }
    return lessons


def main():
    parser = argparse.ArgumentParser(description="Extract lesson metadata to JSON")
    parser.add_argument("--root", required=True, help="Project root directory")
    parser.add_argument("--data", required=True, help="Output JSON file")
    args = parser.parse_args()

    root = Path(args.root)
    data = Path(args.data)
    data.parent.mkdir(parents=True, exist_ok=True)

    lessons = extract_lessons(root)
    data.write_text(json.dumps(lessons, indent=2))


if __name__ == "__main__":
    main()
