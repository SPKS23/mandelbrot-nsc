"""
Minimal test suite for Mandelbrot escape_count.

3 core tests covering:
1. Analytically provable values
2. Parametrized cross-validation (naive vs NumPy)
3. Full grid agreement on small 32×32 region
"""

import numpy as np
import pytest


def escape_count_naive_scalar(c, max_iter):
    """Reference: single point. Direct iteration formula."""
    z = 0.0j
    for k in range(max_iter):
        z = z * z + c
        if abs(z) > 2.0:
            return k
    return max_iter


def escape_count_naive_array(C, max_iter):
    """Reference: grid of points. Oracle for correctness."""
    result = np.zeros(C.shape, dtype=np.int32)
    for idx in np.ndindex(C.shape):
        result[idx] = escape_count_naive_scalar(C[idx], max_iter)
    return result


def escape_count_numpy(C, max_iter):
    """Production: NumPy vectorized (from MP3L08M2)."""
    z = np.zeros_like(C, dtype=C.dtype)
    cnt = np.full(C.shape, max_iter, dtype=np.int32)
    esc = np.zeros(C.shape, dtype=bool)
    
    for k in range(max_iter):
        z[~esc] = z[~esc]**2 + C[~esc]
        newly = ~esc & (np.abs(z) > 2.0)
        cnt[newly] = k
        esc[newly] = True
    
    return cnt


# ============================================================================
# TEST 1: ANALYTICALLY PROVABLE VALUES
# ============================================================================

def test_analytically_provable():
    """
    Test points proven directly from definition.
    c=0: z stays 0 → never escapes → max_iter
    c=1: k=0: z=1, k=1: z=2, k=2: z=5 > 2 → escapes at k=2
    c=-1: periodic {-1, 0, -1, 0, ...} → max |z|=1 < 2 → max_iter
    """
    assert escape_count_naive_scalar(0 + 0j, 100) == 100
    assert escape_count_naive_scalar(1 + 0j, 100) == 2
    assert escape_count_naive_scalar(-1 + 0j, 100) == 100


# ============================================================================
# TEST 2: PARAMETRIZED CROSS-VALIDATION
# ============================================================================

@pytest.mark.parametrize(
    "c,max_iter",
    [
        (0 + 0j, 100),           # Interior
        (1 + 0j, 100),           # Exterior
        (-1 + 0j, 100),          # Periodic
        (-0.75 + 0j, 256),       # Main cardioid boundary
        (-0.125 + 0.649j, 256),  # Periodic bulb
        (10 + 10j, 50),          # Far exterior
    ],
)
def test_numpy_vs_naive_agreement(c, max_iter):
    """
    NumPy vectorized should match naive reference.
    Validates implementation correctness across diverse points.
    """
    C = np.array([[c]], dtype=np.complex128)
    result_naive = escape_count_naive_scalar(c, max_iter)
    result_numpy = escape_count_numpy(C, max_iter)[0, 0]
    assert result_numpy == result_naive


# ============================================================================
# TEST 3: 32×32 GRID CROSS-VALIDATION
# ============================================================================

def test_grid_exact_agreement():
    """
    Full grid: naive reference vs NumPy on 32×32 region.
    Ensures vectorized implementation is numerically identical.
    """
    x = np.linspace(-0.75, -0.74, 32)
    y = np.linspace(0.10, 0.11, 32)
    C = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)
    
    result_naive = escape_count_naive_array(C, max_iter=256)
    result_numpy = escape_count_numpy(C, max_iter=256)
    
    np.testing.assert_array_equal(result_numpy, result_naive)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

