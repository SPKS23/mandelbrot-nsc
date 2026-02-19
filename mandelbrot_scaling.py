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
    return M


if __name__ == "__main__":
    """New code for testing the mandelbrot set generation and plotting."""
    for n in range(5):
        print(f"Run {256*2**n}:")
        benchmark(mandelbrot_set_numpy, -2.0, 1.0, -1.5, 1.5, 256*2**n, 256*2**n,100)