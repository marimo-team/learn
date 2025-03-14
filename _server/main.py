# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi",
#     "marimo",
#     "starlette",
#     "python-dotenv",
#     "pydantic",
#     "polars",
#     "duckdb",
# ]
# ///

import logging
from pathlib import Path

import marimo
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

root_dir = Path(__file__).parent.parent

ROOTS = [
    root_dir / "polars",
    root_dir / "duckdb",
]


server = marimo.create_asgi_app(include_code=True)
app_names: list[str] = []

for root in ROOTS:
    for filename in root.iterdir():
        if filename.is_file() and filename.suffix == ".py":
            app_path = root.stem + "/" + filename.stem
            server = server.with_app(path=f"/{app_path}", root=str(filename))
            app_names.append(app_path)

# Create a FastAPI app
app = FastAPI()

logger.info(f"Mounting {len(app_names)} apps")
for app_name in app_names:
    logger.info(f"  /{app_name}")


@app.get("/")
async def home(request: Request):
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>marimo learn</title>
    </head>
    <body>
        <h1>Welcome to marimo learn!</h1>
        <p>This is a collection of interactive tutorials for learning data science libraries with marimo.</p>
        <p>Check out the <a href="https://github.com/marimo-team/learn">GitHub repository</a> for more information.</p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


app.mount("/", server.build())

# Run the server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
