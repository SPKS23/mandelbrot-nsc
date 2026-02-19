""" Mandelbrot Set 
Generator Author: Søren Peter Krag Sørensen 
Course: Numerical Scientific Computing 2026 
"""
import cmath
import time 
import numpy as np
import matplotlib.pyplot as plt
import time, statistics 

def benchmark(func, *args, n_runs=5): 
    """Time func, return median of n_runs.""" 
    times = [] 
    for _ in range(n_runs): 
        t0 = time.perf_counter() 
        result = func(*args) 
        times.append(time.perf_counter()- t0) 
        median_t = statistics.median(times) 
    print(f"Median: {median_t:.4f}s " f"(min={min(times):.4f}, max={max(times):.4f})") 
    return median_t, result


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
    return (r1, r2, [[mandelbrot_point(complex(r, i), max_iter) for r in r1] for i in r2])


def mandelbrot_set_numpy(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)

    X , Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C)
    M = np.zeros(C.shape, dtype=int)
    for i in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask] ** 2 + C[mask]
        M[mask] += 1
    return x, y, M




if __name__ == "__main__":
    """New code for testing the mandelbrot set generation and plotting."""
    x, y, M = mandelbrot_set_numpy(-2.0, 1.0, -1.5, 1.5, 1024, 1024,100)
    

    """ Old code for testing the mandelbrot set generation and plotting. """
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
    width, height = 1024, 1024
    max_iter = 100
    r1, r2, mandelbrot_image = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)
    if np.allclose(mandelbrot_image, M):
        print("Results match!")
    # plt.imshow(mandelbrot_image, extent=(xmin, xmax, ymin, ymax))
    # plt.xlabel('Re')
    # plt.ylabel('Im')
    # plt.title('Mandelbrot Set')
    # plt.show()
    
# if __name__ == "__main__":
#     benchmark(mandelbrot_set, -2.0, 1.0, -1.5, 1.5, 1024, 1024,100)