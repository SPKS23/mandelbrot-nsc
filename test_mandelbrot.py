"""
Test suite for Mandelbrot implementations across multiple scripts.

Tests functions from:
- mandelbrot.py: mandelbrot_point, mandelbrot_set
- MP3.py: mandelbrot_pixel, mandelbrot_chunk
- MP2_Lecture6.py: mandelbrot_serial, mandelbrot_parallel
- Numba.py: mandelbrot_point_numba, mandelbrot_naive_numba
"""

import numpy as np
import pytest
from mandelbrot import mandelbrot_point, mandelbrot_set
try:
    from MP3 import mandelbrot_pixel, mandelbrot_chunk
except (ImportError, ModuleNotFoundError):
    mandelbrot_pixel = None
    mandelbrot_chunk = None

try:
    from MP2_Lecture6 import mandelbrot_serial, mandelbrot_parallel
except (ImportError, ModuleNotFoundError):
    mandelbrot_serial = None
    mandelbrot_parallel = None

try:
    from Numba import mandelbrot_point_numba, mandelbrot_naive_numba
except (ImportError, ModuleNotFoundError):
    mandelbrot_point_numba = None
    mandelbrot_naive_numba = None


# ============================================================================
# TEST 1: ANALYTICALLY PROVABLE VALUES
# ============================================================================

def test_mandelbrot_point_analytically_provable():
    """
    Test mandelbrot_point (scalar function from mandelbrot.py).
    c=0: z stays 0 → never escapes → max_iter
    c=1: k=0: z=1, k=1: z=2, k=2: z=5, k=3: |z|>2 → escapes at k=3
    c=-1: periodic {-1, 0, -1, 0, ...} → max |z|=1 < 2 → max_iter
    """
    assert mandelbrot_point(0 + 0j, 100) == 100
    assert mandelbrot_point(1 + 0j, 100) == 3
    assert mandelbrot_point(-1 + 0j, 100) == 100


@pytest.mark.skipif(mandelbrot_point_numba is None, reason="Numba implementation not available")
def test_mandelbrot_point_numba_analytically_provable():
    """
    Test mandelbrot_point_numba (Numba-optimized version from Numba.py).
    Same analytically provable tests.
    """
    assert mandelbrot_point_numba(0 + 0j, 100) == 100
    assert mandelbrot_point_numba(1 + 0j, 100) == 3
    assert mandelbrot_point_numba(-1 + 0j, 100) == 100


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
def test_mandelbrot_point_vs_numba(c, max_iter):
    """
    mandelbrot_point (from mandelbrot.py) should match mandelbrot_point_numba.
    Validates both implementations agree on diverse points.
    """
    result_base = mandelbrot_point(c, max_iter)
    if mandelbrot_point_numba is not None:
        result_numba = mandelbrot_point_numba(c, max_iter)
        assert result_numba == result_base


@pytest.mark.skipif(mandelbrot_pixel is None, reason="MP3 implementation not available")
@pytest.mark.parametrize("c_real,c_imag", [(0.0, 0.0), (-0.75, 0.0), (1.0, 0.0)])
def test_mandelbrot_pixel_basic(c_real, c_imag):
    """
    Test mandelbrot_pixel (Numba-compiled pixel calculator from MP3.py).
    Should return valid escape counts.
    """
    count = mandelbrot_pixel(c_real, c_imag, 100)
    assert 0 <= count <= 100
    assert isinstance(count, (int, np.integer))


# ============================================================================
# TEST 3: GRID COMPUTATION CROSS-VALIDATION
# ============================================================================

def test_mandelbrot_set_grid():
    """
    Test mandelbrot_set (from mandelbrot.py) on a small grid.
    Validates grid initialization and basic functionality.
    """
    result = mandelbrot_set(
        xmin=-0.75, xmax=-0.74,
        ymin=0.10, ymax=0.11,
        width=32, height=32,
        max_iter=256
    )
    assert result.shape == (32, 32)
    assert result.dtype in (np.int32, np.int64)
    assert np.all((result >= 0) & (result <= 256))


@pytest.mark.skipif(mandelbrot_naive_numba is None, reason="Numba implementation not available")
def test_mandelbrot_naive_numba_grid():
    """
    Test mandelbrot_naive_numba (fully compiled Numba, from Numba.py).
    Should produce valid escape count grid.
    """
    result = mandelbrot_naive_numba(
        xmin=-0.75, xmax=-0.74,
        ymin=0.10, ymax=0.11,
        width=32, height=32,
        max_iter=256
    )
    assert result.shape == (32, 32)
    assert result.dtype == np.int32
    assert np.all((result >= 0) & (result <= 256))


@pytest.mark.skipif(mandelbrot_serial is None, reason="MP2_Lecture6 implementation not available")
def test_mandelbrot_serial_grid():
    """
    Test mandelbrot_serial (chunked computation from MP2_Lecture6.py).
    Validates serial execution of chunked algorithm.
    """
    result = mandelbrot_serial(
        N=32,
        x_min=-0.75, x_max=-0.74,
        y_min=0.10, y_max=0.11,
        max_iter=256
    )
    assert result.shape == (32, 32)
    assert result.dtype == np.int32
    assert np.all((result >= 0) & (result <= 256))


@pytest.mark.skipif(mandelbrot_naive_numba is None or mandelbrot_serial is None,
                    reason="Implementations not available")
def test_grid_agreement_numba_vs_serial():
    """
    mandelbrot_naive_numba and mandelbrot_serial produce valid grids on same inputs.
    Both should return same shape and value ranges.
    """
    result_numba = mandelbrot_naive_numba(
        xmin=-0.75, xmax=-0.74,
        ymin=0.10, ymax=0.11,
        width=16, height=16,
        max_iter=128
    )
    result_serial = mandelbrot_serial(
        N=16,
        x_min=-0.75, x_max=-0.74,
        y_min=0.10, y_max=0.11,
        max_iter=128
    )
    # Both should produce same shape and valid ranges
    assert result_numba.shape == result_serial.shape == (16, 16)
    assert np.all((result_numba >= 0) & (result_numba <= 128))
    assert np.all((result_serial >= 0) & (result_serial <= 128))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

