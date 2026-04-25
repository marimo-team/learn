"""Ordering Widget for Marimo"""

from pathlib import Path
import random
import traitlets

from .base import BaseWidget


class OrderingWidget(BaseWidget):
    """
    An ordering question widget where students arrange items in sequence using drag-and-drop.

    Attributes:
        question (str): The question text to display
        items (list): Items in the correct order
        shuffle (bool): Whether to shuffle items initially
        value (dict): Current state with 'order' and 'correct' keys
    """

    _esm = Path(__file__).parent / "static" / "ordering.js"

    question = traitlets.Unicode("").tag(sync=True)
    items = traitlets.List(trait=traitlets.Unicode()).tag(sync=True)
    current_order = traitlets.List(trait=traitlets.Unicode()).tag(sync=True)
    shuffle = traitlets.Bool(True).tag(sync=True)

    def __init__(
        self,
        question: str,
        items: list[str],
        shuffle: bool = True,
        lang: str = "en",
        **kwargs,
    ):
        current = items.copy()
        if shuffle:
            random.shuffle(current)
        super().__init__(
            question=question,
            items=items,
            shuffle=shuffle,
            current_order=current,
            lang=lang,
            **kwargs,
        )
