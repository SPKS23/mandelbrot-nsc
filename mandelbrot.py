""" Mandelbrot Set 
Generator Author: [Your Name] 
Course: Numerical Scientific Computing 2026 
"""
import cmath
import time 
import numpy as np

x = np.linspace(-2, 1, 1000)
y = np.linspace(-1.5, 1.5, 1000)
complex_grid = np.array([[complex(xi, yi) for xi in x] for yi in y])
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

print(complex_grid)
print("osten9000")