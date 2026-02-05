""" Mandelbrot Set 
Generator Author: [Your Name] 
Course: Numerical Scientific Computing 2026 
"""
import cmath
import time 

def f(x): 
    """
    Example function.

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    float
        Output value.
        new change
    """
    pass


def mandelbrot_point(c, max_iter):
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter
