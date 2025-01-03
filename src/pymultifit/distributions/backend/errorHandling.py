"""Created on Dec 09 02:28:10 2024"""

neg_message = "cannot be negative."


class BaseDistributionError(Exception):
    """Base class for distribution-related errors."""
    pass


class DegreeOfFreedomError(BaseDistributionError):
    """Raised when the degree of freedom is a float instead of int."""

    def __init__(self):
        super().__init__(r"DOF can only be integer, N+")


class InvalidUniformParameters(BaseDistributionError):
    """Raised when the parameters of uniform distributions are not valid."""

    def __init__(self):
        super().__init__("High < Low, invalid parameter selection.")


class NegativeAlphaError(BaseDistributionError):
    """Raised when the alpha parameter value is negative."""

    def __init__(self):
        super().__init__(f"Alpha {neg_message}.")


class NegativeAmplitudeError(BaseDistributionError):
    """Raised when the amplitude is negative."""

    def __init__(self):
        super().__init__(f"Amplitude {neg_message}")


class NegativeBetaError(BaseDistributionError):
    """Raised when the beta parameter value is negative."""

    def __init__(self):
        super().__init__(f"Beta {neg_message}")


class NegativeRateError(BaseDistributionError):
    """Raised when the value of rate parameter is negative."""

    def __init__(self):
        super().__init__(f"Rate {neg_message}")


class NegativeScaleError(BaseDistributionError):
    """Raised when the value of scale parameter is negative."""

    def __init__(self, parameter='scale'):
        super().__init__(f"{parameter.capitalize()} {neg_message}")


class NegativeShapeError(BaseDistributionError):
    """Raised when the value of shape parameter is negative."""

    def __init__(self):
        super().__init__(f"Shape {neg_message}")


class NegativeStandardDeviationError(BaseDistributionError):
    """Raised when the standard deviation is negative."""

    def __init__(self):
        super().__init__(f"Standard deviation {neg_message}")


class NegativeVarianceError(BaseDistributionError):
    """Raised when the variance value is negative."""

    def __init__(self):
        super().__init__("Variance cannot be negative.")


class XOutOfRange(BaseDistributionError):
    """Raised when the x value is out of range for the distribution."""

    def __init__(self):
        super().__init__("X out of range.")
