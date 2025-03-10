"""Created on Jan 13 11:45:38 2025"""

from typing import Optional

import numpy as np

optFloat = Optional[float]


def line(x: np.ndarray, slope: optFloat = 1., intercept: optFloat = 0.) -> np.ndarray:
    """
    Computes the y-values of a line given x-values, slope, and intercept.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.
    slope : float
        The slope of the line.
    intercept : float
        The y-intercept of the line.

    Returns
    -------
    np.ndarray
        Array of the same shape as :math:`x`, containing the evaluated values.
    """
    return slope * x + intercept


def linear(x: np.ndarray, a: optFloat = 1., b: optFloat = 1) -> np.ndarray:
    """
    Computes the y-values of a linear function given x-values.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.
    a : float
        The coefficient of the linear term (x).
    b : float
        The constant term (y-intercept).

    Returns
    -------
    np.ndarray
        Array of the same shape as :math:`x`, containing the evaluated values.
    """
    return line(x, slope=a, intercept=b)


def quadratic(x: np.ndarray, a: optFloat = 1., b: optFloat = 1., c: optFloat = 1.) -> np.ndarray:
    """
    Computes the y-values of a quadratic function given x-values.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.
    a : float
        The coefficient of the quadratic term (x^2).
    b : float
        The coefficient of the linear term (x).
    c : float
        The constant term (y-intercept).

    Returns
    -------
    np.ndarray
        Array of the same shape as :math:`x`, containing the evaluated values.
    """
    return a * x**2 + b * x + c


def cubic(x: np.ndarray, a: optFloat = 1., b: optFloat = 1., c: optFloat = 1., d: optFloat = 1.) -> np.ndarray:
    """
    Computes the y-values of a cubic function given x-values.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.
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
    np.ndarray
        Array of the same shape as :math:`x`, containing the evaluated values.
    """
    return a * x**3 + b * x**2 + c * x + d


def nth_polynomial(x: np.ndarray, coefficients: list[float]) -> np.ndarray:
    """
    Evaluate a polynomial at given points.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.
    coefficients : list of float
        Coefficients of the polynomial in descending order of degree.
        For example, [a, b, c] represents the polynomial ax^2 + bx + c.

    Returns
    -------
    np.ndarray
        Array of the same shape as :math:`x`, containing the evaluated values.
    """
    return np.polyval(p=coefficients, x=x)
