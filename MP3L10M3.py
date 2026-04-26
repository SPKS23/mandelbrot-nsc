import matplotlib.pyplot as plt

def main():
    results = {
        "Naïve": 2.92,
        "NumPy": 1.287,
        "Hybrid": 1.615,
        "Numba": 0.054,
        "Multiprocessing": 0.020,
        "GPU f32": 0.0013,
        "GPU f64": 0.0123,
    }

    names = list(results.keys())
    times = list(results.values())

    plt.figure()
    plt.bar(names, times)
    plt.yscale("log")  # log scale for large differences
    plt.ylabel("seconds (log scale)")
    plt.xticks(rotation=30, ha="right")
    plt.title("Benchmark (N = 1024)")
    plt.tight_layout()
    plt.savefig("benchmark_mp3.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()