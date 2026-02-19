""" Mandelbrot Set 
Generator Author: Søren Peter Krag Sørensen 
Course: Numerical Scientific Computing 2026 
"""
import cmath
import time 
import numpy as np
import matplotlib.pyplot as plt

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

def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    return (r1, r2, np.array([[mandelbrot_point(complex(r, i), max_iter) for r in r1] for i in r2]))

if __name__ == "__main__":
    start_time = time.time()
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
    width, height = 1024, 1024
    max_iter = 100
    r1, r2, mandelbrot_image = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)
    plt.imshow(mandelbrot_image, extent=(xmin, xmax, ymin, ymax))
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.title('Mandelbrot Set')
    plt.show()
    end_time = time.time()
    print(f"Tid for kode: {end_time - start_time} sekunder")