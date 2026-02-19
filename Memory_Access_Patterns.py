import numpy as np
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

def col_sum(A):
    N = A.shape[0]
    for i in range(N):
        s = np.sum(A[:, i])

def row_sum(A):
    N = A.shape[0]
    for i in range(N):
        s = np.sum(A[i, :])

A = np.random.rand(10000, 10000)
print("Normal array:")
benchmark(col_sum, A)
benchmark(row_sum, A)
A_f = np.asfortranarray(A)
print("Fortran array:")
benchmark(col_sum, A_f)
benchmark(row_sum, A_f)