"""Created on Aug 10 23:37:54 2024"""

from typing import Optional

import numpy as np

optFloat = Optional[float]

__all__ = ['line', 'linear', 'quadratic', 'cubic', 'nth_polynomial']


def line(x, slope: optFloat = 1., intercept: optFloat = 0.) -> np.ndarray:
    """
    Computes the y-values of a line given x-values, slope, and intercept.

    This function calculates the y-values for a linear equation of the form
    `y = slope * x + intercept` for the provided x-values.

    Parameters
    ----------
    x : array_like
        The input x-values for which the y-values are to be computed.
    slope : float
        The slope of the line.
    intercept : float
        The y-intercept of the line.

    Returns
    -------
    array_like
        The computed y-values of the line corresponding to the given x-values.
    """
    return slope * x + intercept


def linear(x: np.ndarray, a: optFloat = 1., b: optFloat = 1) -> np.ndarray:
    """
    Computes the y-values of a linear function given x-values.

    This function calculates the y-values for a linear equation of the form
    `y = a * x + b `.

    Parameters
    ----------
    x : array_like
        The input x-values for which the y-values are to be computed.
    a : float
        The coefficient of the linear term (x).
    b : float
        The constant term (y-intercept).

    Returns
    -------
    array_like
        The computed y-values of the quadratic function corresponding to the given x-values.
    """
    return line(x, a, b)


def quadratic(x, a: optFloat = 1., b: optFloat = 1., c: optFloat = 1.) -> np.ndarray:
    """
    Computes the y-values of a quadratic function given x-values.

    This function calculates the y-values for a quadratic equation of the form
    `y = a * x^2 + b * x + c`.

    Parameters
    ----------
    x : array_like
        The input x-values for which the y-values are to be computed.
    a : float
        The coefficient of the quadratic term (x^2).
    b : float
        The coefficient of the linear term (x).
    c : float
        The constant term (y-intercept).

    Returns
    -------
    array_like
        The computed y-values of the quadratic function corresponding to the given x-values.
    """
    return a * x**2 + b * x + c


def cubic(x: np.ndarray, a: optFloat = 1., b: optFloat = 1., c: optFloat = 1., d: optFloat = 1.) -> np.ndarray:
    """
    Computes the y-values of a cubic function given x-values.

    This function calculates the y-values for a quadratic equation of the form
    `y = a * x^3 + b * x^2 + c * x + d`.

    Parameters
    ----------
    x : array_like
        The input x-values for which the y-values are to be computed.
    a : float
        The coefficient of the cubic term (x^3).
    b : float
        The coefficient of the quadratic term (x^2).
    c : float
        The coefficient of the linear term (x).
    d : float
        The constant term (y-intercept).

    Returns
    -------
    array_like
        The computed y-values of the cubic function corresponding to the given x-values.
    """
    return a * x**3 + b * x**2 + c * x + d


def nth_polynomial(x: np.ndarray, coefficients: list[float]) -> np.ndarray:
    """
    Evaluate a polynomial at given points.

    Parameters
    ----------
    x : np.ndarray
        An array of values at which to evaluate the polynomial.
    coefficients : list of float
        Coefficients of the polynomial in descending order of degree.
        For example, [a, b, c] represents the polynomial ax^2 + bx + c.

    Returns
    -------
    np.ndarray
        The evaluated polynomial values at each point in `x`.
    """
    return np.polyval(coefficients, x)
