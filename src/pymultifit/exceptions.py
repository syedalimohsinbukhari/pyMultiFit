"""Created on Jun 05 09:41:03 2026"""


class pyMultiFitErrors(Exception):
    """Base class of the exceptions raised by ``pymultifit`` itself."""

    __module__ = "pymultifit"
    pass


class AxesError(pyMultiFitErrors):
    """Raised when the axes handed to a plotting method do not fit what it draws, e.g. one axis for two panels."""

    __module__ = pyMultiFitErrors.__module__
    pass
