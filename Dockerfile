FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Create a non-root user
RUN useradd -m appuser

# Copy application files
COPY _server/main.py _server/main.py
COPY polars/ polars/
COPY duckdb/ duckdb/

# Set proper ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Create virtual environment and install dependencies
RUN uv venv
RUN uv export --script _server/main.py | uv pip install -r -

ENV PORT=7860
EXPOSE 7860

CMD ["uv", "run", "_server/main.py"]
