---
title: Contributing to This Site
---

## Introduction

- what we're trying to achieve
- what we're looking for
- guidance for educators
- where to find community
- licensing
- use of AI
- how we acknowledge contributions

## How to Contribute

- setting up environment
  - including a quick intro to useful `uv` commands
- WASM
  - what it is
  - package compatibility (discussed in more detail below)
- formatting and checking with `ruff`
- naming conventions
  - `dd_some_title.py` is included in index page
  - other Python files aren't (e.g., notebooks under development)
  - see note above about WASM file inclusion
- useful `make` targets
  - `make install`: install packages required *to build the site*
  - `make check`: run all quick checks
  - `make check_exec NOTEBOOKS="??_*.py"`: run a set of notebooks to check for runtime errors
  - `make check_packages NOTEBOOKS="??_*.py"`: check for inconsistent package versions across notebooks
  - `make build`: build website
  - `make clean`: clean up stray files

## Things to Know

- marimo skills
- underscore-prefixed variables
- returning `mo.markdown()` from code cell
- `mo.show_code()`
- marimo slides
- localizing files for WASM
  - the `public` directory and the `marimo_learn` package
  - examples of URLs
- WASM package compatibility issues (polars, numba)
- widgets (wigglystuff and https://anywidget.dev/en/community/#widgets-gallery)
