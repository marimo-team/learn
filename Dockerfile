FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

COPY _server/main.py _server/main.py
COPY polars/ polars/
COPY duckdb/ duckdb/

RUN uv venv
RUN uv export --script _server/main.py | uv pip install -r -

ENV PORT=8000
EXPOSE 8000

CMD ["uv", "run", "_server/main.py"]
