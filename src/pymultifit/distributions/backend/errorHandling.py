"""Created on Dec 09 02:28:10 2024"""


class DistributionError(Exception):
    """Base class for distribution-related errors."""
    pass


class NegativeAmplitudeError(DistributionError):
    """Raised when the amplitude is negative."""
    pass


class NegativeStandardDeviationError(DistributionError):
    """Raised when the standard deviation is negative."""
    pass
