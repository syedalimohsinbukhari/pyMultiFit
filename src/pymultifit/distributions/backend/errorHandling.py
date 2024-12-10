"""Created on Dec 09 02:28:10 2024"""

neg_message = "cannot be negative."


class DistributionError(Exception):
    """Base class for distribution-related errors."""
    pass


class NegativeAmplitudeError(DistributionError):
    """Raised when the amplitude is negative."""

    def __init__(self, message=f"Amplitude {neg_message}"):
        super().__init__(message)


class NegativeStandardDeviationError(DistributionError):
    """Raised when the standard deviation is negative."""

    def __init__(self, message=f"Standard deviation {neg_message}"):
        super().__init__(message)


class NegativeVarianceError(DistributionError):
    """Raised when the variance value is negative."""

    def __init__(self, message="Variance cannot be negative."):
        super().__init__(message)


class NegativeAlphaError(DistributionError):
    """Raised when the alpha parameter value is negative."""

    def __init__(self, message=r"The $\alpha$ parameter cannot be negative."):
        super().__init__(message)


class NegativeBetaError(DistributionError):
    """Raised when the beta parameter value is negative."""

    def __init__(self, message=r"The $\beta$ parameter cannot be negative."):
        super().__init__(message)


class DegreeOfFreedomError(DistributionError):
    """Raised when the degree of freedom is a float instead of int."""

    def __init__(self, message=r"The degree of freedom parameter can only be integer, k $\in N^+$"):
        super().__init__(message)


class NegativeShapeError(DistributionError):
    """Raised when the value of shape parameter is negative."""

    def __init__(self, message=f"Shape {neg_message}"):
        super().__init__(message)


class NegativeRateError(DistributionError):
    """Raised when the value of rate parameter is negative."""

    def __init__(self, message=f"Rate {neg_message}"):
        super().__init__(message)


class NegativeScaleError(DistributionError):
    """Raised when the value of scale parameter is negative."""

    def __init__(self, parameter='scale'):
        super().__init__(f"{parameter.capitalize()} {neg_message}")
