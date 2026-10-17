import os
import struct

import numpy as np
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def _has_rasterio():
    try:
        import rasterio  # noqa: F401
        return True
    except ImportError:
        return False


HAS_RASTERIO = _has_rasterio()
requires_rasterio = pytest.mark.skipif(not HAS_RASTERIO, reason="rasterio is not installed")


def write_geosoft(path, grid, x0=0.0, y0=0.0, dx=1.0, dy=1.0, dtype='f4', sf=2,
                  kx=1, zbase=0.0, zmult=1.0, rot=0.0):
    """Write an uncompressed Geosoft GRD for tests (grid row 0 = south)."""
    grid = np.asarray(grid)
    if kx == 1:
        nv, ne = grid.shape
        data = grid
        de, dv = dx, dy
    else:
        ne, nv = grid.shape
        data = grid.T          # each vector is a column
        de, dv = dy, dx
    es = np.dtype(dtype).itemsize
    header = bytearray(512)
    struct.pack_into('<5i', header, 0, es, sf, ne, nv, kx)
    struct.pack_into('<7d', header, 20, de, dv, x0, y0, rot, zbase, zmult)
    with open(path, 'wb') as f:
        f.write(bytes(header))
        f.write(np.ascontiguousarray(data).astype('<' + dtype).tobytes())
    return str(path)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
