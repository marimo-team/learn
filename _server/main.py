# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi",
#     "marimo",
#     "starlette",
#     "python-dotenv",
#     "pydantic",
#     "duckdb==1.3.2",
#     "altair==5.5.0",
#     "beautifulsoup4==4.13.3",
#     "httpx==0.28.1",
#     "marimo",
#     "nest-asyncio==1.6.0",
#     "numba==0.61.0",
#     "numpy==2.1.3",
#     "polars==1.24.0",
# ]
# ///

import logging
import os
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

# Get port from environment variable or use default
PORT = int(os.environ.get("PORT", 7860))

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

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
