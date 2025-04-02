install:
	uv pip install marimo jinja2 markdown

build:
	rm -rf _site
	uv run scripts/build.py

serve:
	uv run python -m http.server --directory _site