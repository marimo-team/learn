"""Unit tests for utility functions."""

from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest
from marimo_learn import localize_file


class TestLocalizeFile:
    """Tests for localize_file()"""

    def test_returns_local_path(self):
        """Returns path under notebook_dir on successful download."""
        notebook_dir = Path("/fake/notebook")
        mock_mo = MagicMock()
        mock_mo.notebook_dir.return_value = notebook_dir

        with (
            patch("marimo_learn.utilities.mo", mock_mo),
            patch("marimo_learn.utilities.urllib.request.urlretrieve"),
            patch("pathlib.Path.mkdir"),
        ):
            result = localize_file("data/image.png", "https://example.com/image.png")

        assert result == str(notebook_dir / "data" / "image.png")

    def test_raises_on_network_error(self):
        """Raises FileNotFoundError on any URLError (HTTP or network)."""
        notebook_dir = Path("/fake/notebook")
        mock_mo = MagicMock()
        mock_mo.notebook_dir.return_value = notebook_dir

        with (
            patch("marimo_learn.utilities.mo", mock_mo),
            patch(
                "marimo_learn.utilities.urllib.request.urlretrieve",
                side_effect=URLError("not found"),
            ),
            patch("pathlib.Path.mkdir"),
        ):
            with pytest.raises(FileNotFoundError):
                localize_file("missing.png", "https://example.com/missing.png")

    def test_creates_parent_dirs(self):
        """Creates parent directories before downloading."""
        notebook_dir = Path("/fake/notebook")
        mock_mo = MagicMock()
        mock_mo.notebook_dir.return_value = notebook_dir

        mock_mkdir = MagicMock()
        with (
            patch("marimo_learn.utilities.mo", mock_mo),
            patch("marimo_learn.utilities.urllib.request.urlretrieve"),
            patch("pathlib.Path.mkdir", mock_mkdir),
        ):
            localize_file("sub/dir/file.csv", "https://example.com/file.csv")

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
