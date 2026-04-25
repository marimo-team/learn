"""Utilities for teaching with marimo notebooks."""

import marimo as mo
import sys
import urllib.error
import urllib.request


def localize_file(filename: str, url: str) -> str:
    """
    Download a file from the url, returning the local path.

    Args:
        filename: name for local copy of file
        url: URL of file to download

    Returns:
        local file path

    Raises:
        FileNotFoundError: if remote file not found
    """

    local_path = mo.notebook_dir() / filename
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, local_path)
    except urllib.error.URLError as e:
        raise FileNotFoundError(f"unable to get file from '{url}'") from e

    return str(local_path)
