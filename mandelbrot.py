"""Mandelbrot Set Generator.

Author: Søren Peter Krag Sørensen
Course: Numerical Scientific Computing 2026
"""

import statistics
import time
from typing import Callable, Any

import numpy as np


def benchmark(
    func: Callable[..., Any], *args: Any, n_runs: int = 5
) -> tuple[float, Any]:
    """Time a function across multiple runs and return median timing.

    Parameters
    ----------
    func : Callable
        Function to benchmark.
    *args : Any
        Positional arguments to pass to func.
    n_runs : int, optional
        Number of runs to execute (default: 5). Median of timings is reported.

    Returns
    -------
    tuple[float, Any]
        Tuple of (median_time_seconds, result_of_func).
    """
    times: list[float] = []
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - t0)

    median_t = statistics.median(times)
    print(
        f"Median: {median_t:.4f}s "
        f"(min={min(times):.4f}, max={max(times):.4f})"
    )
    return median_t, result


def mandelbrot_point(c: complex, max_iter: int) -> int:
    """Count iterations until escape for a single complex point.

    Iterates z_{n+1} = z_n^2 + c starting from z_0 = 0,
    returning the iteration count when |z| > 2 or max_iter.

    Parameters
    ----------
    c : complex
        Complex parameter for the iteration.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    int
        Iteration count at escape (|z| > 2) or max_iter if bounded.
    """
    z = 0j
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z * z + c
    return max_iter


def mandelbrot_set(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    width: int,
    height: int,
    max_iter: int = 100,
) -> np.ndarray:
    """Compute iteration count grid using scalar mandelbrot_point calls.

    Generates a 2D grid of complex parameters and computes escape iteration
    counts by calling mandelbrot_point for each pixel. Demonstrates nested
    function calls as reference implementation.

    Parameters
    ----------
    xmin : float
        Minimum real coordinate.
    xmax : float
        Maximum real coordinate.
    ymin : float
        Minimum imaginary coordinate.
    ymax : float
        Maximum imaginary coordinate.
    width : int
        Number of pixels in real direction.
    height : int
        Number of pixels in imaginary direction.
    max_iter : int, optional
        Maximum iterations per point (default: 100).

    Returns
    -------
    np.ndarray
        2D array of shape (height, width) with dtype int, containing
        iteration counts for each point. Computed via nested calls to
        mandelbrot_point.
    """
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)

    result = np.zeros((height, width), dtype=int)
    for j, yi in enumerate(y):
        for i, xi in enumerate(x):
            c = complex(xi, yi)
            result[j, i] = mandelbrot_point(c, max_iter)

    return result


def mandelbrot_set_numpy(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    width: int,
    height: int,
    max_iter: int = 100,
) -> np.ndarray:
    """Compute iteration count grid for Mandelbrot set using NumPy vectorization.

    Generates a 2D grid of complex parameters over [xmin, xmax] × [ymin, ymax]
    and computes escape iteration counts for each point using vectorized operations.

    Parameters
    ----------
    xmin : float
        Minimum real coordinate.
    xmax : float
        Maximum real coordinate.
    ymin : float
        Minimum imaginary coordinate.
    ymax : float
        Maximum imaginary coordinate.
    width : int
        Number of pixels in real direction.
    height : int
        Number of pixels in imaginary direction.
    max_iter : int, optional
        Maximum iterations per point (default: 100).

    Returns
    -------
    np.ndarray
        2D array of shape (height, width) with dtype int, containing
        iteration counts until escape for each point.
    """
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)

    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C)
    M = np.zeros(C.shape, dtype=int)

    for i in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask] ** 2 + C[mask]
        M[mask] += 1

    return M

if __name__ == "__main__":
    """New code for testing the mandelbrot set generation and plotting."""
    M = mandelbrot_set_numpy(-2.0, 1.0, -1.5, 1.5, 1024, 1024,100)
    
    benchmark(mandelbrot_set_numpy, -2.0, 1.0, -1.5, 1.5, 1024, 1024,100)
    benchmark(mandelbrot_set, -2.0, 1.0, -1.5, 1.5, 1024, 1024,100)
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