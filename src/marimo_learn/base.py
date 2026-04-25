"""Base widget class for marimo learn widgets."""

import anywidget
import traitlets


class BaseWidget(anywidget.AnyWidget):
    lang = traitlets.Unicode("en").tag(sync=True)
    value = traitlets.Dict(default_value=None, allow_none=True).tag(sync=True)
