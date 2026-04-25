"""Labeling Widget for Marimo"""

from pathlib import Path
import traitlets

from .base import BaseWidget


class LabelingWidget(BaseWidget):
    """
    A text labeling widget where students drag numbered labels to text lines.

    Attributes:
        question (str): The question text to display
        labels (list): List of label texts (shown on left)
        text_lines (list): List of text lines to be labeled (shown on right)
        correct_labels (dict): Mapping of line indices to lists of correct label indices
        value (dict): Current state with 'placed_labels', 'score', 'total', and 'correct' keys
    """

    _esm = Path(__file__).parent / "static" / "labeling.js"

    question = traitlets.Unicode("").tag(sync=True)
    labels = traitlets.List(trait=traitlets.Unicode()).tag(sync=True)
    text_lines = traitlets.List(trait=traitlets.Unicode()).tag(sync=True)
    correct_labels = traitlets.Dict().tag(sync=True)

    def __init__(
        self,
        question: str,
        labels: list[str],
        text_lines: list[str],
        correct_labels: dict,
        lang: str = "en",
        **kwargs,
    ):
        super().__init__(
            question=question,
            labels=labels,
            text_lines=text_lines,
            correct_labels=correct_labels,
            lang=lang,
            **kwargs,
        )
