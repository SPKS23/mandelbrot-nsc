

if __name__ == '__main__':
    import time
    import statistics
    import numpy as np
    from multiprocessing import Pool
    from dask.distributed import Client
    from dask import delayed

    # ----------------------------
    # CONFIG
    # ----------------------------
    N, max_iter = 8192, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
    n_workers = 12

    # ----------------------------
    # JIT warm-up
    # ----------------------------
    mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)

    # ----------------------------
    # SERIAL BASELINE
    # ----------------------------
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_chunk(0, N, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)

    t_serial = statistics.median(times)
    print(f"Serial: {t_serial:.3f}s")

    # ----------------------------
    # MULTIPROCESSING SWEEP
    # ----------------------------
    tiny = [(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)]

    for mult in [1, 2, 4, 8, 16]:
        n_chunks = mult * n_workers

        with Pool(processes=n_workers) as pool:
            pool.map(_worker, tiny)  # warm-up

            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                mandelbrot_parallel(
                    N, X_MIN, X_MAX, Y_MIN, Y_MAX,
                    max_iter=max_iter,
                    n_workers=n_workers,
                    n_chunks=n_chunks,
                    pool=pool
                )
                times.append(time.perf_counter() - t0)

        t_par = statistics.median(times)
        lif = n_workers * t_par / t_serial - 1

        print(f"{n_chunks:4d} chunks  {t_par:.3f}s  {t_serial/t_par:.1f}x  LIF={lif:.2f}")

    # ----------------------------
    # DASK (FIXED DISTRIBUTED)
    # ----------------------------
    client = Client("tcp://10.92.1.228:8786")
    print(client)

    # safe warm-up on workers
    client.run(lambda: None)

    def mandelbrot_dask(N, x_min, x_max, y_min, y_max,
                        max_iter=100, n_chunks=32):

        chunk_size = max(1, N // n_chunks)
        tasks = []
        row = 0

        while row < N:
            row_end = min(row + chunk_size, N)
            tasks.append(delayed(mandelbrot_chunk)(
                row, row_end, N,
                x_min, x_max, y_min, y_max,
                max_iter
            ))
            row = row_end

        futures = client.compute(tasks)
        parts = client.gather(futures)

        return np.vstack(parts)

    # ----------------------------
    # DASK BENCHMARK
    # ----------------------------
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_chunks=32)
        times.append(time.perf_counter() - t0)

    print(f"Dask distributed (n_chunks=32): {statistics.median(times):.3f}s")

    client.close()