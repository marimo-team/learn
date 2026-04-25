"""Matching Widget for Marimo"""

from pathlib import Path
import traitlets

from .base import BaseWidget


class MatchingWidget(BaseWidget):
    """
    A matching question widget where students pair items from two columns using drag-and-drop.

    Attributes:
        question (str): The question text to display
        left (list): Items in the left column
        right (list): Items in the right column
        correct_matches (dict): Mapping of left column indices to right column indices
        value (dict): Current state with 'matches', 'correct', and 'score' keys
    """

    _esm = Path(__file__).parent / "static" / "matching.js"

    question = traitlets.Unicode("").tag(sync=True)
    left = traitlets.List(trait=traitlets.Unicode()).tag(sync=True)
    right = traitlets.List(trait=traitlets.Unicode()).tag(sync=True)
    correct_matches = traitlets.Dict().tag(sync=True)

    def __init__(
        self,
        question: str,
        left: list[str],
        right: list[str],
        correct_matches: dict,
        lang: str = "en",
        **kwargs,
    ):
        super().__init__(
            question=question,
            left=left,
            right=right,
            correct_matches=correct_matches,
            lang=lang,
            **kwargs,
        )
