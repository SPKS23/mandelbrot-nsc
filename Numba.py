from numba import njit
import numpy as np
import time, statistics
from mandelbrot import mandelbrot_set, mandelbrot_set_numpy
import matplotlib.pyplot as plt

def bench(fn, *args, runs=5):
    fn(*args)  # warmup
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)

# Approach A — Hybrid
@njit
def mandelbrot_point_numba(c, max_iter=100):
    z = 0j
    for n in range(max_iter):
        if z.real*z.real + z.imag*z.imag > 4.0:
            return n
        z = z*z + c
    return max_iter

def mandelbrot_hybrid(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    # outer loops still in Python
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    result = np.zeros((height, width), dtype=np.int32)
    for i in range(height):
        for j in range(width):
            c = x[j] + 1j * y[i]
            result[i, j] = mandelbrot_point_numba(c, max_iter)
    return result


# Approach B — Fully compiled (recommended)
@njit
def mandelbrot_naive_numba(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    result = np.zeros((height, width), dtype=np.int32)
    for i in range(height):
        for j in range(width):
            c = x[j] + 1j * y[i]
            z = 0j
            n = 0
            while n < max_iter and \
                z.real*z.real+z.imag*z.imag <= 4.0:
                z = z*z + c
                n += 1
            result[i, j] = n
    return result

@njit
def mandelbrot_typed_numba(xmin, xmax, ymin, ymax, width, height, max_iter=100,dtype=np.float64):
    x = np.linspace(xmin, xmax, width).astype(dtype)
    y = np.linspace(ymin, ymax, height).astype(dtype)
    result = np.zeros((height, width), dtype=np.int32)
    for i in range(height):
        for j in range(width):
            c = x[j] + 1j * y[i]
            z = 0j
            n = 0
            while n < max_iter and \
                z.real*z.real+z.imag*z.imag <= 4.0:
                z = z*z + c
                n += 1
            result[i, j] = n
    return result


### Test of milestone 3 ###
# # Warm up (triggers JIT compilation -- exclude from timing)
# _ = mandelbrot_hybrid(-2, 1, -1.5, 1.5, 64, 64)
# _ = mandelbrot_naive_numba(-2, 1, -1.5, 1.5, 64, 64)

# t_hybrid = bench(mandelbrot_hybrid, -2, 1, -1.5, 1.5, 1024, 1024)
# t_full = bench(mandelbrot_naive_numba, -2, 1, -1.5, 1.5, 1024, 1024)
# t_set = bench(mandelbrot_set, -2, 1, -1.5, 1.5, 1024, 1024)
# t_numpy = bench(mandelbrot_set_numpy, -2, 1, -1.5, 1.5, 1024, 1024)



# print(f"Hybrid: {t_hybrid:.3f}s")
# print(f"Fully compiled: {t_full:.3f}s")
# print(f"Original Python: {t_set:.3f}s")
# print(f"NumPy: {t_numpy:.3f}s")
# print(f"Ratio: {t_hybrid/t_full:.1f}x")

for type  in [np.float64, np.float32]:
    mandelbrot_typed_numba(-2, 1, -1.5, 1.5, 1024, 1024, dtype=type)
    t0 = time.perf_counter()
    mandelbrot_typed_numba(-2, 1, -1.5, 1.5, 1024, 1024, dtype=type)
    print(f"Time for dtype={type.__name__}: {time.perf_counter() - t0:.3f}s")

# r16 = mandelbrot_typed_numba(-2, 1, -1.5, 1.5, 1024, 1024, dtype=np.float16)
# r32 = mandelbrot_typed_numba(-2, 1, -1.5, 1.5, 1024, 1024, dtype=np.float32)
# r64 = mandelbrot_typed_numba(-2, 1, -1.5, 1.5, 1024, 1024, dtype=np.float64)

# fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# for ax, result, title in zip(axes, [r16, r32, r64],
#                              ['float16', 'float32', 'float64 (ref)']):
#     ax.imshow(result, cmap='hot')
#     ax.set_title(title); ax.axis('off')
# plt.savefig('precision_comparison.png', dpi=150)

# print(f"Max diff float32 vs float64: {np.abs(r32-r64).max()}")
# print(f"Max diff float16 vs float64: {np.abs(r16-r64).max()}")
