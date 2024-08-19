"""Created on Aug 03 17:13:21 2024"""


def integral_check(pdf_function, x_range: tuple) -> float:
    """
    Compute the integral of a given PDF function over a specified range.

    Parameters
    ----------
    pdf_function : function
        The PDF function to integrate.
    x_range : tuple
        The range (a, b) over which to integrate the PDF.

    Returns
    -------
    float
        The integral result of the PDF function over the specified range.
    """
    from scipy.integrate import quad
    integral = quad(lambda x: pdf_function(x), x_range[0], x_range[1])[0]
    return integral
