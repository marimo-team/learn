#!/usr/bin/env python
"""Check that marimo notebooks in the same lesson directory agree on package versions.

It is acceptable for different notebooks in a directory to specify different packages,
but if two or more notebooks specify the same package, their version constraints must
be identical.
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path


# Regex to extract the inline script metadata block (PEP 723)
SCRIPT_BLOCK_RE = re.compile(r"^# /// script\s*\n((?:#[^\n]*\n)*?)# ///", re.MULTILINE)
DEPENDENCY_LINE_RE = re.compile(r'^#\s+"([^"]+)",?\s*$')


def parse_script_header(text: str) -> list[str]:
    """Return the list of dependency strings from a PEP 723 script header, or []."""
    match = SCRIPT_BLOCK_RE.search(text)
    if not match:
        return []
    block = match.group(1)
    deps: list[str] = []
    in_deps = False
    for raw_line in block.splitlines():
        line = raw_line.lstrip("#").strip()
        if line.startswith("dependencies"):
            in_deps = True
            continue
        if in_deps:
            if line.startswith("]"):
                break
            # strip surrounding quotes and comma: e.g. '    "polars==1.0",' -> 'polars==1.0'
            stripped = line.strip().strip('"\'').rstrip(",").strip('"\'')
            if stripped:
                deps.append(stripped)
    return deps


def package_name(dep: str) -> str:
    """Extract the bare package name from a PEP 508 dependency string.

    Examples:
        "polars==1.22.0"  -> "polars"
        "pandas>=2.0,<3"  -> "pandas"
        "marimo"          -> "marimo"
    """
    return re.split(r"[><=!;\s\[]", dep, maxsplit=1)[0].lower()


def check_directory(lesson_dir: Path, only: set[str]) -> list[str]:
    """Return a list of error messages for version inconsistencies among *only* in lesson_dir."""
    # Map package name -> {version_spec: [notebook_path, ...]}
    seen: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    for nb in sorted(lesson_dir.glob("*.py")):
        if nb.name not in only:
            continue
        try:
            text = nb.read_text(encoding="utf-8")
        except IOError:
            continue
        if "marimo.App" not in text:
            continue
        for dep in parse_script_header(text):
            name = package_name(dep)
            seen[name][dep].append(nb.name)

    errors: list[str] = []
    for name, specs in sorted(seen.items()):
        if len(specs) > 1:
            errors.append(f"  Package '{name}' has conflicting specifications:")
            for spec, files in sorted(specs.items()):
                errors.append(f"    {spec!r} in: {', '.join(files)}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="+", metavar="NOTEBOOK",
                        help="notebook files to check (grouped by directory)")
    args = parser.parse_args()

    dir_filter: dict[Path, set[str]] = defaultdict(set)
    for nb_path in (Path(p) for p in args.notebooks):
        dir_filter[nb_path.parent].add(nb_path.name)

    total_errors = 0
    for lesson_dir, only in sorted(dir_filter.items()):
        errors = check_directory(lesson_dir, only=only)
        if errors:
            print(f"\n{lesson_dir}/")
            for msg in errors:
                print(msg)
            total_errors += len(errors)

    if total_errors:
        print(f"\nFound package version inconsistencies in {total_errors} package(s).")
        sys.exit(1)
    else:
        print("All package version specifications are consistent.")
        sys.exit(0)


if __name__ == "__main__":
    main()
