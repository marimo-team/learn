# Default target.
all : commands

## commands : show all commands.
commands :
	@grep -h -E '^##' ${MAKEFILE_LIST} | sed -e 's/## //g' | column -t -s ':'

## install: install minimal required packages into current environment.
install:
	uv pip install marimo jinja2 markdown

## build: build entire site.
build:
	rm -rf _site
	uv run scripts/build.py

## serve: run local web server without rebuilding.
serve:
	uv run python -m http.server --directory _site

## clean: clean up stray files.
clean:
	@find . -name '*~' -exec rm {} +
	@find . -name '.DS_Store' -exec rm {} +
