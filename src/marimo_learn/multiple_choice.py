"""Multiple Choice Widget for Marimo"""

from pathlib import Path
import traitlets

from .base import BaseWidget


class MultipleChoiceWidget(BaseWidget):
    """
    A multiple choice question widget.

    Attributes:
        question (str): The question text to display
        options (list): List of answer options
        correct_answer (int): Index of the correct answer (0-based)
        explanation (str): Optional explanation text shown after answering
        value (dict): Current state with 'selected', 'correct', and 'answered' keys
    """

    _esm = Path(__file__).parent / "static" / "multiple-choice.js"

    question = traitlets.Unicode("").tag(sync=True)
    options = traitlets.List(trait=traitlets.Unicode()).tag(sync=True)
    correct_answer = traitlets.Int(0).tag(sync=True)
    explanation = traitlets.Unicode("").tag(sync=True)

    def __init__(
        self,
        question: str,
        options: list[str],
        correct_answer: int,
        explanation: str = "",
        lang: str = "en",
        **kwargs,
    ):
        super().__init__(
            question=question,
            options=options,
            correct_answer=correct_answer,
            explanation=explanation,
            lang=lang,
            **kwargs,
        )
