# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "requests==2.32.3",
# ]
# ///

import marimo

__generated_with = "0.10.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import subprocess
    from pathlib import Path

    import requests

    FOLDER = "ModernPythonDataScienceHandbook"
    REPO = "jakevdp/PythonDataScienceHandbook"

    def download_notebooks():
        # Get list of notebooks from GitHub API
        api_url = f"https://api.github.com/repos/{REPO}/git/trees/master?recursive=1"
        response = requests.get(api_url)
        files = response.json()["tree"]

        # Filter for .ipynb files in notebooks directory
        notebook_files = [
            f["path"]
            for f in files
            if f["path"].endswith(".ipynb") and f["path"].startswith("notebooks/")
        ]

        # Create output folder if it doesn't exist
        os.makedirs(FOLDER, exist_ok=True)

        for notebook in notebook_files:
            # Download each notebook
            raw_url = f"https://raw.githubusercontent.com/{REPO}/master/{notebook}"
            response = requests.get(raw_url)

            # Save notebook locally
            notebook_name = Path(notebook).name
            temp_path = f"temp_{notebook_name}"

            with open(temp_path, "wb") as f:
                f.write(response.content)

            # Convert to marimo
            output_file = os.path.join(FOLDER, notebook_name.replace(".ipynb", ".py"))
            subprocess.run(["marimo", "convert", temp_path, "-o", output_file])

            # Cleanup temp file
            os.remove(temp_path)

    return FOLDER, Path, REPO, download_notebooks, os, requests, subprocess


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Download Notebooks")
    run_button
    return (run_button,)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(download_notebooks, mo, run_button):
    if mo.app_meta().mode == "script" or run_button.value:
        download_notebooks()
    return


if __name__ == "__main__":
    app.run()
