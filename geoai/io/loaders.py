"""
geoai.io.loaders
================
Readers for geophysical file formats:
  - Surfer GRD: ASCII (DSAA), Binary 6 (DSBB), Binary 7 (DSRB)
  - Geosoft / Oasis Montaj GRD (uncompressed + zlib-compressed)
  - ER Mapper ERS (.ers header + raw data file)
  - ESRI ASCII Grid (.asc)
  - netCDF-3 grids (.grd / .nc): COARDS / CF / GMT 4+ and the legacy GMT format,
    read natively with scipy
  - GeoTIFF / TIFF (rasterio; otherwise Pillow + GeoTIFF tags)
  - Other rasters readable by GDAL (if rasterio is installed; e.g. netCDF-4 / HDF5)
  - XYZ / CSV / DAT (regular lattice → direct grid; scattered points → interpolation),
    including the Geosoft / Oasis montaj XYZ dialect
  - Well / drillhole CSV

All grid readers return the same dictionary structure:
    grid : (ny, nx) float64 — row 0 = southernmost row, NaN = no data
    x, y : pixel-centre coordinate vectors in ascending order
    xmin, xmax, ymin, ymax, nx, ny, format (+ format-specific extra fields)
"""

import csv
import io
import os
import re
import struct
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


# Surfer blank value. Nodes equal to or greater than this value are blank.
SURFER_BLANK = 1.70141e38

# Geosoft dummy value for floating-point data. Values equal to or less than
# this value are blank.
GEOSOFT_DUMMY = -1.0e32

# Magnitude above which a nodata value is a sentinel that may be matched with a
# relative tolerance (see _nodata_mask)
_HUGE_SENTINEL = 1e30


# ─────────────────────────────────────────────────────────────────────────────
# COMMON HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_layer(grid, x, y, fmt: str = 'array', **extra) -> dict:
    """
    Builds a standard layer dictionary from a grid + coordinate vectors.

    - Puts the axes in ascending order (flipping the grid if needed).
    - Validates size consistency and strict monotonicity of the axes.
    - Converts ±inf values to NaN.
    """
    grid = np.array(grid, dtype=np.float64, copy=True)
    x = np.array(x, dtype=np.float64, copy=True).ravel()
    y = np.array(y, dtype=np.float64, copy=True).ravel()

    if grid.ndim != 2:
        raise ValueError(f"Grid must be 2-dimensional, got ndim: {grid.ndim}")
    ny, nx = grid.shape
    if len(x) != nx or len(y) != ny:
        raise ValueError(
            f"Coordinate/grid size mismatch: grid {ny}×{nx}, "
            f"len(x)={len(x)}, len(y)={len(y)}"
        )
    if nx < 2 or ny < 2:
        raise ValueError(f"Grid must be at least 2×2, got: {ny}×{nx}")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("Coordinate vectors must be finite.")

    if x[0] > x[-1]:
        x = x[::-1]
        grid = grid[:, ::-1]
    if y[0] > y[-1]:
        y = y[::-1]
        grid = grid[::-1, :]
    if not (np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)):
        raise ValueError("Coordinate vectors must be strictly increasing/decreasing.")

    grid[~np.isfinite(grid)] = np.nan

    layer = {
        'grid': grid,
        'x': x, 'y': y,
        'xmin': float(x[0]), 'xmax': float(x[-1]),
        'ymin': float(y[0]), 'ymax': float(y[-1]),
        'nx': int(nx), 'ny': int(ny),
        'format': fmt,
    }
    layer.update(extra)
    return layer


def _nodata_mask(values: np.ndarray, nodata) -> np.ndarray:
    """
    Boolean mask of the cells that equal a nodata value (a number or its text).

    The nodata value is cast to the cell type before comparing: values such as
    -1e32 or -99999.9 cannot be represented exactly in float32, so a float64
    comparison with the stored float32 cells would never match. Huge sentinel
    values (|nodata| ≥ 1e30, e.g. -3.40282e+38 printed with '%g') also match
    within a relative tolerance of 1e-5, because they are often written to text
    headers with too few digits to round-trip; no real data lives there.
    """
    values = np.asarray(values)
    none = np.zeros(values.shape, dtype=bool)
    try:
        nd = float(nodata)
    except (TypeError, ValueError):
        return none
    kind = values.dtype.kind
    if np.isnan(nd):
        return np.isnan(values) if kind == 'f' else none
    if kind == 'f':
        with np.errstate(over='ignore', invalid='ignore'):
            cast = np.array(nd).astype(values.dtype)
            mask = values == cast
            if np.isfinite(nd) and abs(nd) >= _HUGE_SENTINEL:
                mask |= np.abs(values.astype(np.float64) - nd) <= 1e-5 * abs(nd)
        return mask
    if kind in 'iub':
        if not np.isfinite(nd) or nd != np.round(nd):
            return none
        info = np.iinfo(values.dtype) if kind != 'b' else None
        if info is not None and not info.min <= nd <= info.max:
            return none
        return values == int(nd)
    return none


def _regularize_rotated(grid, east2d, north2d, fmt, **extra) -> dict:
    """
    Resamples a rotated grid onto an axis-parallel regular grid.

    The node positions (east2d, north2d) must be an affine function of the
    (row, column) indices, as for rotated Geosoft grids and rotated GDAL
    rasters. Each output node is mapped back to fractional (row, column)
    indices of the source grid and interpolated bilinearly from the four
    enclosing source nodes. An output node is NaN when it falls outside the
    rotated rectangle, or when one of the enclosing source nodes that carries
    weight is NaN, so blanked areas stay blank instead of being filled in.
    """
    grid = np.asarray(grid, dtype=np.float64)
    east2d = np.asarray(east2d, dtype=np.float64)
    north2d = np.asarray(north2d, dtype=np.float64)
    n_rows, n_cols = grid.shape
    if n_rows < 2 or n_cols < 2:
        raise ValueError(f"Rotated grid must be at least 2×2, got: {n_rows}×{n_cols}")
    if np.isfinite(grid).sum() < 3:
        raise ValueError("Not enough valid nodes in the rotated grid.")

    # Affine geometry: east = e0 + col*ec + row*er, north = n0 + col*nc + row*nr
    e0, n0 = float(east2d[0, 0]), float(north2d[0, 0])
    ec, nc = float(east2d[0, 1] - e0), float(north2d[0, 1] - n0)
    er, nr = float(east2d[1, 0] - e0), float(north2d[1, 0] - n0)
    det = ec * nr - er * nc
    if det == 0:
        raise ValueError("Rotated grid has a degenerate geometry.")

    # Original node spacing (distance between neighbouring nodes)
    step = float(min(np.hypot(ec, nc), np.hypot(er, nr)))

    xmin, xmax = float(east2d.min()), float(east2d.max())
    ymin, ymax = float(north2d.min()), float(north2d.max())
    nx = int(np.floor((xmax - xmin) / step + 1e-9)) + 1
    ny = int(np.floor((ymax - ymin) / step + 1e-9)) + 1
    max_nodes = 4000
    if nx > max_nodes or ny > max_nodes:
        raise ValueError(
            f"Rotated grid is too large to convert to a regular grid ({ny}×{nx})."
        )
    xi = xmin + np.arange(nx) * step
    yi = ymin + np.arange(ny) * step
    Xi, Yi = np.meshgrid(xi, yi)

    de, dn = Xi - e0, Yi - n0
    col = (nr * de - er * dn) / det
    row = (-nc * de + ec * dn) / det
    eps = 1e-6
    inside = ((col >= -eps) & (col <= n_cols - 1 + eps) &
              (row >= -eps) & (row <= n_rows - 1 + eps))
    c = np.clip(col, 0, n_cols - 1)
    r = np.clip(row, 0, n_rows - 1)
    c0 = np.minimum(np.floor(c).astype(np.int64), n_cols - 2)
    r0 = np.minimum(np.floor(r).astype(np.int64), n_rows - 2)
    fc, fr = c - c0, r - r0

    acc = np.zeros(Xi.shape)
    wsum = np.zeros(Xi.shape)
    blocked = ~inside
    for dr, dc, w in ((0, 0, (1 - fr) * (1 - fc)), (0, 1, (1 - fr) * fc),
                      (1, 0, fr * (1 - fc)), (1, 1, fr * fc)):
        v = grid[r0 + dr, c0 + dc]
        ok = np.isfinite(v)
        acc += np.where(ok, v, 0.0) * np.where(ok, w, 0.0)
        wsum += np.where(ok, w, 0.0)
        blocked |= ~ok & (w > 1e-9)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = acc / wsum
    out[blocked | (wsum <= 0)] = np.nan

    warnings.warn(
        f"Rotated grid ({extra.get('rotation', '?')}°) was resampled onto an "
        f"axis-parallel {ny}×{nx} grid by bilinear interpolation.",
        UserWarning,
    )
    return make_layer(out, xi, yi, fmt=fmt, **extra)


def _read_text_tokens(filepath: str) -> list:
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        return f.read().split()


# ─────────────────────────────────────────────────────────────────────────────
# FORMAT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

GEOSOFT_ELEMENT_SIZES = (1, 2, 4, 8, 1025, 1026, 1028, 1032)

# netCDF-3 classic and 64-bit offset files (readable by scipy.io.netcdf_file)
_NETCDF3_MAGICS = (b'CDF\x01', b'CDF\x02')
_HDF5_MAGIC = b'\x89HDF'

_EXT_MAP = {
    '.grd': 'auto_grd',      # may be Surfer / Geosoft / GMT-netCDF
    '.tif': 'geotiff',
    '.tiff': 'geotiff',
    '.ers': 'ermapper',
    '.asc': 'auto_text',     # ESRI ASCII Grid or XYZ
    '.xyz': 'xyz',
    '.csv': 'csv',
    '.txt': 'auto_text',
    '.dat': 'auto_text',
    '.nc': 'auto_nc',        # netCDF-3 (native) or netCDF-4 / HDF5 (GDAL)
    '.img': 'gdal',
    '.vrt': 'gdal',
    '.bil': 'gdal',
}


def detect_format(filepath: str) -> str:
    """
    Detects the format from the file extension and content signature (magic bytes).

    Returned values: 'surfer_ascii', 'surfer_binary', 'surfer7_binary',
    'geosoft_grd', 'ermapper', 'esri_ascii', 'netcdf', 'geotiff', 'gdal',
    'xyz', 'csv', 'unknown'.
    """
    ext = Path(filepath).suffix.lower()
    fmt = _EXT_MAP.get(ext, 'unknown')

    if fmt == 'auto_grd':
        return _detect_grd_type(filepath)
    if fmt == 'auto_nc':
        with open(filepath, 'rb') as f:
            magic = f.read(4)
        return 'netcdf' if magic in _NETCDF3_MAGICS else 'gdal'
    if fmt == 'auto_text':
        return _detect_text_type(filepath, default='xyz')
    if fmt == 'unknown' and _rasterio_available():
        return 'gdal'
    return fmt


def _detect_grd_type(filepath: str) -> str:
    """Determines the subtype of a .grd file from its header bytes."""
    with open(filepath, 'rb') as f:
        header = f.read(512)

    if header[:4] == b'DSAA':
        return 'surfer_ascii'
    if header[:4] == b'DSBB':
        return 'surfer_binary'
    if header[:4] == b'DSRB':
        return 'surfer7_binary'
    if header[:4] in _NETCDF3_MAGICS:
        return 'netcdf'      # GMT / COARDS / CF netCDF-3 grid
    if header[:3] == b'CDF' or header[:4] == _HDF5_MAGIC:
        return 'gdal'        # CDF-5 or netCDF-4 / HDF5
    if _looks_like_geosoft(header, os.path.getsize(filepath)):
        return 'geosoft_grd'
    if _rasterio_available():
        return 'gdal'
    return 'unknown'


def _looks_like_geosoft(header: bytes, file_size: int) -> bool:
    if len(header) < 512 or file_size <= 512:
        return False
    es, sf, ne, nv, kx = struct.unpack_from('<5i', header, 0)
    if es not in GEOSOFT_ELEMENT_SIZES or sf not in (0, 1, 2, 3):
        return False
    if ne <= 0 or nv <= 0 or kx not in (1, -1):
        return False
    if es < 1024 and file_size < 512 + ne * nv * es:
        return False
    return True


def _detect_text_type(filepath: str, default: str = 'xyz') -> str:
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        head = f.read(256).lstrip()
    if head.startswith('DSAA'):
        return 'surfer_ascii'
    if head.lower().startswith('ncols') or head.lower().startswith('nrows'):
        return 'esri_ascii'
    return default


def _rasterio_available() -> bool:
    try:
        import rasterio  # noqa: F401
        return True
    except ImportError:
        return False


def _normalize_crs(crs_text: str) -> str:
    """
    Returns a compact, comparable CRS string ('EPSG:32635' where possible).
    Without rasterio, or when the text cannot be parsed, it is returned unchanged.
    """
    if not crs_text:
        return 'unknown'
    try:
        from rasterio.crs import CRS
    except ImportError:
        return crs_text
    try:
        return CRS.from_user_input(crs_text).to_string()
    except Exception:
        return crs_text


# ─────────────────────────────────────────────────────────────────────────────
# SURFER GRD READERS
# ─────────────────────────────────────────────────────────────────────────────

def _surfer_blank_to_nan(grid: np.ndarray, blank: float = SURFER_BLANK) -> np.ndarray:
    # also catch blank values that were rounded to float32
    return np.where(grid >= blank * (1 - 1e-6), np.nan, grid)


def read_surfer_ascii_grd(filepath: str) -> dict:
    """Reads a Surfer ASCII GRD (Golden Software DSAA)."""
    tokens = _read_text_tokens(filepath)
    if not tokens or tokens[0] != 'DSAA':
        raise ValueError(f"Invalid Surfer ASCII GRD header: {filepath}")
    try:
        nx, ny = int(tokens[1]), int(tokens[2])
        xmin, xmax = float(tokens[3]), float(tokens[4])
        ymin, ymax = float(tokens[5]), float(tokens[6])
    except (IndexError, ValueError) as e:
        raise ValueError(f"Could not read Surfer ASCII GRD header: {e}")

    data = np.array(tokens[9:9 + nx * ny], dtype=np.float64)
    if data.size != nx * ny:
        raise ValueError(
            f"Data size mismatch: expected {nx*ny}, found {data.size}"
        )
    # Rows are ordered from Ymin to Ymax (south to north)
    grid = _surfer_blank_to_nan(data.reshape((ny, nx)))
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    return make_layer(grid, x, y, fmt='surfer_ascii')


def read_surfer_binary_grd(filepath: str) -> dict:
    """Reads a Surfer 6 Binary GRD (DSBB)."""
    with open(filepath, 'rb') as f:
        raw = f.read()
    if raw[:4] != b'DSBB':
        raise ValueError("Invalid Surfer Binary GRD")
    nx, ny = struct.unpack_from('<hh', raw, 4)
    xmin, xmax, ymin, ymax, _zmin, _zmax = struct.unpack_from('<6d', raw, 8)
    n_values = nx * ny
    expected = 56 + n_values * 4
    if len(raw) < expected:
        raise ValueError(
            f"Surfer Binary GRD too short: expected ≥{expected} bytes, found {len(raw)}"
        )
    data = np.frombuffer(raw, dtype='<f4', count=n_values, offset=56)
    grid = _surfer_blank_to_nan(data.reshape((ny, nx)).astype(np.float64))
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    return make_layer(grid, x, y, fmt='surfer_binary')


def read_surfer7_binary_grd(filepath: str) -> dict:
    """
    Reads a Surfer 7+ Binary GRD (DSRB).

    Layout (all fields little-endian):
      'DSRB' + long size + long version
      'GRID' + long size + nRow, nCol (long) +
             xLL, yLL, xSize, ySize, zMin, zMax, Rotation, BlankValue (double)
      'DATA' + long size + nRow*nCol double (rows from south to north)
      (optional 'FLTI' fault section — skipped)
    """
    with open(filepath, 'rb') as f:
        raw = f.read()
    if raw[:4] != b'DSRB':
        raise ValueError("Invalid Surfer 7 Binary GRD")

    header_size = struct.unpack_from('<i', raw, 4)[0]
    offset = 8 + header_size
    grid_info = None
    data = None

    while offset + 8 <= len(raw):
        tag = raw[offset:offset + 4]
        size = struct.unpack_from('<i', raw, offset + 4)[0]
        body = offset + 8
        if size < 0 or body + size > len(raw):
            raise ValueError(f"Invalid Surfer 7 section size: {tag!r} {size}")
        if tag == b'GRID':
            n_row, n_col = struct.unpack_from('<ii', raw, body)
            (xll, yll, xsize, ysize, _zmin, _zmax,
             rotation, blank) = struct.unpack_from('<8d', raw, body + 8)
            grid_info = dict(n_row=n_row, n_col=n_col, xll=xll, yll=yll,
                             xsize=xsize, ysize=ysize, rotation=rotation,
                             blank=blank)
        elif tag == b'DATA':
            if grid_info is None:
                raise ValueError("Surfer 7: DATA section appears before the GRID section.")
            n = grid_info['n_row'] * grid_info['n_col']
            if size < n * 8:
                raise ValueError("Surfer 7: DATA section is shorter than expected.")
            data = np.frombuffer(raw, dtype='<f8', count=n, offset=body)
            break
        offset = body + size

    if grid_info is None or data is None:
        raise ValueError("Surfer 7: GRID/DATA sections not found.")

    if grid_info['rotation'] != 0:
        warnings.warn(
            "Surfer 7 rotation field is non-zero; it was ignored because Surfer "
            "does not use this field.", UserWarning,
        )

    grid = data.reshape((grid_info['n_row'], grid_info['n_col'])).astype(np.float64)
    grid = _surfer_blank_to_nan(grid, blank=grid_info['blank'])
    x = grid_info['xll'] + np.arange(grid_info['n_col']) * grid_info['xsize']
    y = grid_info['yll'] + np.arange(grid_info['n_row']) * grid_info['ysize']
    return make_layer(grid, x, y, fmt='surfer7_binary')


# ─────────────────────────────────────────────────────────────────────────────
# GEOSOFT / OASIS MONTAJ GRD
# ─────────────────────────────────────────────────────────────────────────────

# Geosoft dummy (blank) values per data type
_GEOSOFT_DUMMIES = {
    'u1': 255, 'i1': -127,
    'u2': 65535, 'i2': -32767,
    'u4': 4294967295, 'i4': -2147483647,
}
_GEOSOFT_FLOAT_DUMMY = GEOSOFT_DUMMY


def _geosoft_dtype(es: int, sf: int) -> str:
    size = es - 1024 if es > 1024 else es
    table = {
        (1, 0): 'u1', (1, 1): 'i1',
        (2, 0): 'u2', (2, 1): 'i2',
        (4, 0): 'u4', (4, 1): 'i4', (4, 2): 'f4',
        (8, 0): 'u8', (8, 1): 'i8', (8, 2): 'f8',
    }
    if sf == 3:
        raise NotImplementedError("Geosoft colour grids are not supported.")
    if (size, sf) not in table:
        raise ValueError(f"Invalid Geosoft data type: ES={es}, SF={sf}")
    return table[(size, sf)]


def _geosoft_decompress(raw: bytes) -> bytes:
    """
    Decompresses compressed Geosoft grid blocks.
    After the header (512 bytes): signature (4), compression type (4), block count (4),
    vectors per block (4), block offsets (int64 × n, relative to start of file),
    block sizes (int32 × n). Each block begins with a 16-byte pre-header,
    followed by a zlib stream.
    """
    n_blocks = struct.unpack_from('<i', raw, 512 + 8)[0]
    if n_blocks <= 0 or n_blocks > 10_000_000:
        raise ValueError(f"Geosoft: invalid block count {n_blocks}")
    offsets = struct.unpack_from(f'<{n_blocks}q', raw, 512 + 16)
    sizes = struct.unpack_from(f'<{n_blocks}i', raw, 512 + 16 + 8 * n_blocks)
    chunks = []
    for off, size in zip(offsets, sizes):
        start, end = off + 16, off + size
        if end > len(raw) or start >= end:
            raise ValueError("Geosoft: compressed block extends beyond the end of the file.")
        try:
            chunks.append(zlib.decompress(raw[start:end]))
        except zlib.error as e:
            raise IOError(
                f"Could not decompress Geosoft compressed block ({e}). This compression "
                "type is not supported; save the grid uncompressed in Oasis Montaj."
            )
    return b''.join(chunks)


def read_geosoft_grd(filepath: str) -> dict:
    """
    Geosoft / Oasis Montaj GRD (format v2) reader.

    Header (512 bytes, little-endian):
      0  ES  int32  element size (1,2,4,8; >1024 → compressed)
      4  SF  int32  sign: 0 unsigned integer, 1 signed integer, 2 floating point
      8  NE  int32  elements per vector
      12 NV  int32  number of vectors
      16 KX  int32  1: vectors are rows, -1: vectors are columns
      20 DE, 28 DV, 36 X0, 44 Y0, 52 ROT, 60 ZBASE, 68 ZMULT (double)
    Value = stored / ZMULT + ZBASE. Dummy values become NaN.
    """
    with open(filepath, 'rb') as f:
        raw = f.read()
    if len(raw) < 512:
        raise ValueError("Geosoft GRD header is incomplete (shorter than 512 bytes).")

    es, sf, ne, nv, kx = struct.unpack_from('<5i', raw, 0)
    de, dv, x0, y0, rot, zbase, zmult = struct.unpack_from('<7d', raw, 20)

    if es not in GEOSOFT_ELEMENT_SIZES:
        raise ValueError(f"Geosoft: invalid element size ES={es}")
    if kx not in (1, -1):
        raise NotImplementedError(f"Geosoft: KX={kx} ordering is not supported.")
    if ne <= 0 or nv <= 0:
        raise ValueError(f"Geosoft: invalid dimensions NE={ne}, NV={nv}")
    if zmult == 0:
        raise ValueError("Geosoft: ZMULT cannot be zero.")

    dtype = _geosoft_dtype(es, sf)
    n = ne * nv
    payload = _geosoft_decompress(raw) if es > 1024 else raw[512:]
    itemsize = np.dtype(dtype).itemsize
    if len(payload) < n * itemsize:
        raise ValueError(
            f"Geosoft: data section too short (expected {n*itemsize}, found {len(payload)})"
        )
    stored = np.frombuffer(payload, dtype='<' + dtype, count=n).astype(np.float64)

    if dtype in ('f4', 'f8'):
        stored[stored <= _GEOSOFT_FLOAT_DUMMY] = np.nan
    elif dtype in _GEOSOFT_DUMMIES:
        stored[stored == _GEOSOFT_DUMMIES[dtype]] = np.nan
    values = stored / zmult + zbase

    if kx == 1:
        grid = values.reshape((nv, ne))           # row = vector
        dx, dy = de, dv
    else:
        grid = values.reshape((ne, nv), order='F')  # column = vector
        dx, dy = dv, de
    ny, nx = grid.shape

    if rot != 0:
        cols = np.arange(nx) * dx
        rows = np.arange(ny) * dy
        C, R = np.meshgrid(cols, rows)
        th = np.radians(rot)
        east = x0 + C * np.cos(th) - R * np.sin(th)
        north = y0 + C * np.sin(th) + R * np.cos(th)
        return _regularize_rotated(grid, east, north, fmt='geosoft_grd', rotation=rot)

    x = x0 + np.arange(nx) * dx
    y = y0 + np.arange(ny) * dy
    return make_layer(grid, x, y, fmt='geosoft_grd')


# ─────────────────────────────────────────────────────────────────────────────
# ER MAPPER ERS
# ─────────────────────────────────────────────────────────────────────────────

_ERS_CELLTYPES = {
    'unsigned8bitinteger': 'u1', 'signed8bitinteger': 'i1',
    'unsigned16bitinteger': 'u2', 'signed16bitinteger': 'i2',
    'unsigned32bitinteger': 'u4', 'signed32bitinteger': 'i4',
    'ieee4bytereal': 'f4', 'ieee8bytereal': 'f8',
}

# ER Mapper projection codes that carry no real coordinate reference system
_ERS_UNKNOWN_PROJECTIONS = {'', 'RAW', 'LOCAL'}


def _parse_ers_header(text: str) -> dict:
    """Parses an ERS header into a nested dictionary (keys are lower-cased)."""
    root = {}
    stack = [root]
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        m = re.match(r'^(\w+)\s+Begin$', s, re.IGNORECASE)
        if m:
            child = {}
            stack[-1][m.group(1).lower()] = child
            stack.append(child)
            continue
        if re.match(r'^(\w+)\s+End$', s, re.IGNORECASE):
            if len(stack) > 1:
                stack.pop()
            continue
        if '=' in s:
            key, val = s.split('=', 1)
            stack[-1][key.strip().lower()] = val.strip().strip('"')
    return root


def _dms_to_degrees(value: str) -> float:
    """Converts a 'DD:MM:SS.s' string to decimal degrees."""
    value = value.strip()
    sign = -1.0 if value.startswith('-') else 1.0
    parts = [abs(float(p)) for p in value.lstrip('+-').split(':')]
    deg = parts[0] + (parts[1] / 60 if len(parts) > 1 else 0) + \
        (parts[2] / 3600 if len(parts) > 2 else 0)
    return sign * deg


def _ers_crs(filepath: str, projection: str, datum: str) -> str:
    """
    Resolves the CRS of an ERS raster.

    ER Mapper projection codes ('NUTM35', 'TMAMG56', ...) are not CRS strings
    that PROJ understands. When rasterio is available, GDAL's ERS driver
    resolves them (e.g. NUTM35 + WGS84 → 'EPSG:32635'). Otherwise the common
    WGS84 UTM and geodetic codes are mapped directly and any other code is
    returned unchanged. RAW / LOCAL / empty projections give 'unknown'.
    """
    proj = projection.strip().upper()
    if proj in _ERS_UNKNOWN_PROJECTIONS:
        return 'unknown'
    if _rasterio_available():
        try:
            import rasterio
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with rasterio.open(filepath) as src:
                    crs = src.crs
            return crs.to_string() if crs else 'unknown'
        except Exception:
            pass
    if datum.strip().upper() == 'WGS84':
        m = re.match(r'^([NS])UTM(\d{1,2})$', proj)
        if m and 1 <= int(m.group(2)) <= 60:
            return f"EPSG:{(32600 if m.group(1) == 'N' else 32700) + int(m.group(2))}"
        if proj == 'GEODETIC':
            return 'EPSG:4326'
    return projection.strip()


def read_ermapper_ers(filepath: str, band: int = 1) -> dict:
    """
    Reads an ER Mapper ERS raster (.ers text header + raw binary data).

    Data file: the 'DataFile' key, otherwise the same name without extension.
    If 'HeaderOffset' is present, the data is stored inside the .ers file.
    Data layout is BIL (line by line, bands consecutive within each line).

    The returned 'crs' is a CRS string resolved from the ER Mapper projection
    (see _ers_crs) or 'unknown'; the raw ER Mapper codes are kept under
    'ers_projection' and 'ers_datum'.
    """
    with open(filepath, 'rb') as f:
        raw_header = f.read()
    text = raw_header.decode('latin-1')
    hdr = _parse_ers_header(text).get('datasetheader')
    if hdr is None:
        raise ValueError(f"Invalid ERS header (no DatasetHeader): {filepath}")
    ri = hdr.get('rasterinfo', {})

    try:
        n_lines = int(ri['nroflines'])
        n_cells = int(ri['nrofcellsperline'])
    except KeyError as e:
        raise ValueError(f"Missing field in ERS header: {e}")
    n_bands = int(ri.get('nrofbands', 1))
    if not 1 <= band <= n_bands:
        raise ValueError(f"ERS: band={band} is invalid (number of bands: {n_bands})")
    cell_type = ri.get('celltype', 'IEEE4ByteReal').lower()
    if cell_type not in _ERS_CELLTYPES:
        raise ValueError(f"ERS: unsupported CellType '{cell_type}'")
    endian = '>' if hdr.get('byteorder', 'LSBFirst').lower() == 'msbfirst' else '<'
    dtype = np.dtype(endian + _ERS_CELLTYPES[cell_type])

    # Data source
    if 'headeroffset' in hdr:
        data_bytes = raw_header[int(hdr['headeroffset']):]
    else:
        base = Path(filepath)
        data_file = hdr.get('datafile')
        candidates = [base.parent / data_file] if data_file else []
        candidates += [base.with_suffix(''), base.with_suffix('.dat')]
        data_path = next((p for p in candidates if p.exists() and p != base), None)
        if data_path is None:
            raise FileNotFoundError(
                f"ERS data file not found. Tried: {[str(c) for c in candidates]}"
            )
        with open(data_path, 'rb') as f:
            data_bytes = f.read()

    n_needed = n_lines * n_bands * n_cells
    if len(data_bytes) < n_needed * dtype.itemsize:
        raise ValueError("ERS data file is shorter than expected.")
    cube = np.frombuffer(data_bytes, dtype=dtype, count=n_needed)
    cube = cube.reshape((n_lines, n_bands, n_cells))
    cells = cube[:, band - 1, :].astype(dtype.newbyteorder('='))
    grid = cells.astype(np.float64)

    null = ri.get('nullcellvalue')
    if null is not None:
        # Compare in the cell type: e.g. -1e32 is not exactly representable in float32
        grid[_nodata_mask(cells, null)] = np.nan

    # Coordinates
    cs = hdr.get('coordinatespace', {})
    rotation = cs.get('rotation')
    if rotation is not None and abs(_dms_to_degrees(rotation)) > 1e-9:
        raise NotImplementedError(
            "Rotated ERS rasters are not supported; export without rotation."
        )
    cell = ri.get('cellinfo', {})
    dx = float(cell.get('xdimension', 1.0))
    dy = float(cell.get('ydimension', 1.0))
    reg = ri.get('registrationcoord', {})
    if 'eastings' in reg:
        e0, n0 = float(reg['eastings']), float(reg['northings'])
    elif 'metersx' in reg:
        e0, n0 = float(reg['metersx']), float(reg['metersy'])
    elif 'longitude' in reg:
        e0, n0 = _dms_to_degrees(reg['longitude']), _dms_to_degrees(reg['latitude'])
    else:
        warnings.warn("ERS: no RegistrationCoord, using pixel coordinates.",
                      UserWarning)
        e0, n0 = 0.0, float(n_lines) * dy
    cell_x = float(ri.get('registrationcellx', 0.0))
    cell_y = float(ri.get('registrationcelly', 0.0))
    left = e0 - cell_x * dx
    top = n0 + cell_y * dy

    x = left + (np.arange(n_cells) + 0.5) * dx
    y = top - (np.arange(n_lines) + 0.5) * dy     # row 0 = north → make_layer flips it
    proj = str(cs.get('projection', '') or '')
    datum = str(cs.get('datum', '') or '')
    return make_layer(grid, x, y, fmt='ermapper',
                      crs=_ers_crs(filepath, proj, datum),
                      ers_projection=proj, ers_datum=datum)


# ─────────────────────────────────────────────────────────────────────────────
# ESRI ASCII GRID
# ─────────────────────────────────────────────────────────────────────────────

def read_esri_ascii(filepath: str) -> dict:
    """Reads an ESRI ASCII Grid (.asc). The first data row is the northernmost."""
    tokens = _read_text_tokens(filepath)
    header = {}
    i = 0
    while i + 1 < len(tokens) and re.match(r'^[A-Za-z_]+$', tokens[i]):
        header[tokens[i].lower()] = tokens[i + 1]
        i += 2
    try:
        ncols, nrows = int(header['ncols']), int(header['nrows'])
    except KeyError:
        raise ValueError(f"ESRI ASCII header has no ncols/nrows: {filepath}")
    if 'cellsize' in header:
        dx = dy = float(header['cellsize'])
    elif 'dx' in header and 'dy' in header:
        dx, dy = float(header['dx']), float(header['dy'])
    else:
        raise ValueError("ESRI ASCII header has no cellsize (or dx/dy).")

    if 'xllcenter' in header:
        x0 = float(header['xllcenter'])
    elif 'xllcorner' in header:
        x0 = float(header['xllcorner']) + dx / 2
    else:
        raise ValueError("ESRI ASCII header has no xllcorner/xllcenter.")
    if 'yllcenter' in header:
        y0 = float(header['yllcenter'])
    elif 'yllcorner' in header:
        y0 = float(header['yllcorner']) + dy / 2
    else:
        raise ValueError("ESRI ASCII header has no yllcorner/yllcenter.")

    data = np.array(tokens[i:i + ncols * nrows], dtype=np.float64)
    if data.size != ncols * nrows:
        raise ValueError(
            f"ESRI ASCII data size mismatch: expected {ncols*nrows}, found {data.size}"
        )
    grid = data.reshape((nrows, ncols))[::-1, :]   # south to north
    if 'nodata_value' in header:
        grid = np.where(grid == float(header['nodata_value']), np.nan, grid)
    x = x0 + np.arange(ncols) * dx
    y = y0 + np.arange(nrows) * dy
    return make_layer(grid, x, y, fmt='esri_ascii')


# ─────────────────────────────────────────────────────────────────────────────
# NETCDF-3 GRIDS (COARDS / CF / GMT)
# ─────────────────────────────────────────────────────────────────────────────

# Preferred data variable names, in order (compared case-insensitively)
_NC_PREFERRED_VARIABLES = ('z', 'elevation', 'band1', 'data', 'value')
# Default netCDF fill value for float and double variables (NC_FILL_FLOAT/DOUBLE)
_NC_DEFAULT_FILL = 9.969209968386869e36
_NC_LONLAT_NAMES = {'lat', 'lon', 'latitude', 'longitude'}


class _NoGridVariableError(ValueError):
    """The netCDF file contains no variable that the native reader can use as a grid."""


def _nc_attr_text(obj, name: str):
    """Returns a text attribute of a scipy netCDF variable/file, or None."""
    val = (getattr(obj, '_attributes', None) or {}).get(name)
    if val is None:
        return None
    if isinstance(val, bytes):
        return val.decode('latin-1').strip('\x00 ').strip()
    if isinstance(val, str):
        return val.strip()
    arr = np.asarray(val)
    if arr.dtype.kind == 'S':
        return b''.join(arr.ravel().tolist()).decode('latin-1').strip('\x00 ').strip()
    if arr.dtype.kind == 'U':
        return ''.join(arr.ravel().tolist()).strip()
    return None


def _nc_attr_numbers(obj, name: str):
    """Returns a numeric attribute as a 1-D float64 array, or None."""
    val = (getattr(obj, '_attributes', None) or {}).get(name)
    if val is None or isinstance(val, (bytes, str)):
        return None
    arr = np.asarray(val).ravel()
    if arr.size == 0 or arr.dtype.kind not in 'biuf':
        return None
    return arr.astype(np.float64)


def _nc_is_numeric(var) -> bool:
    return np.asarray(var.data).dtype.kind in 'biuf'


def _nc_unpack(var, raw) -> np.ndarray:
    """
    Converts stored netCDF values to physical float64 values.

    _FillValue, missing_value (both in packed units), the default netCDF fill
    value of float variables without _FillValue, and NaN become NaN; then
    scale_factor and add_offset are applied.
    """
    raw = np.array(raw, copy=True)
    raw = raw.astype(raw.dtype.newbyteorder('='))
    if raw.dtype.kind == 'i' and (_nc_attr_text(var, '_Unsigned') or '').lower() == 'true':
        raw = raw.view(raw.dtype.str.replace('i', 'u'))
    mask = ~np.isfinite(raw) if raw.dtype.kind == 'f' else np.zeros(raw.shape, dtype=bool)
    fills = []
    fill = _nc_attr_numbers(var, '_FillValue')
    if fill is not None:
        fills.extend(fill.tolist())
    elif raw.dtype.kind == 'f':
        fills.append(_NC_DEFAULT_FILL)
    missing = _nc_attr_numbers(var, 'missing_value')
    if missing is not None:
        fills.extend(missing.tolist())
    for value in fills:
        mask |= _nodata_mask(raw, value)

    values = raw.astype(np.float64)
    scale = _nc_attr_numbers(var, 'scale_factor')
    offset = _nc_attr_numbers(var, 'add_offset')
    if scale is not None:
        values = values * scale[0]
    if offset is not None:
        values = values + offset[0]
    values[mask] = np.nan
    return values


def _nc_axis_kind(name: str, var):
    """Classifies a coordinate variable as 'x', 'y' or None from its metadata."""
    axis = (_nc_attr_text(var, 'axis') or '').upper()
    if axis in ('X', 'Y'):
        return axis.lower()
    std = (_nc_attr_text(var, 'standard_name') or '').lower()
    if std in ('longitude', 'projection_x_coordinate', 'grid_longitude'):
        return 'x'
    if std in ('latitude', 'projection_y_coordinate', 'grid_latitude'):
        return 'y'
    units = (_nc_attr_text(var, 'units') or '').lower().replace(' ', '')
    if units in ('degrees_east', 'degree_east', 'degrees_e', 'degree_e', 'degreese', 'degreee'):
        return 'x'
    if units in ('degrees_north', 'degree_north', 'degrees_n', 'degree_n', 'degreesn', 'degreen'):
        return 'y'
    lname = name.lower()
    if lname in ('x', 'lon', 'long', 'longitude', 'easting', 'east', 'xc'):
        return 'x'
    if lname in ('y', 'lat', 'latitude', 'northing', 'north', 'yc'):
        return 'y'
    return None


def _nc_is_geographic(name: str, var) -> bool:
    units = (_nc_attr_text(var, 'units') or '').lower()
    std = (_nc_attr_text(var, 'standard_name') or '').lower()
    return (units.startswith('degree') or std in ('longitude', 'latitude')
            or name.lower() in _NC_LONLAT_NAMES)


def _nc_coordinate(variables: dict, dim: str, size: int):
    """Returns the 1-D coordinate variable of a dimension, or None."""
    var = variables.get(dim)
    if var is None or tuple(var.dimensions) != (dim,) or not _nc_is_numeric(var):
        return None
    if np.asarray(var.data).size != size:
        return None
    return var


def _is_legacy_gmt(variables: dict) -> bool:
    need = ('x_range', 'y_range', 'spacing', 'dimension', 'z')
    return all(k in variables for k in need) and np.asarray(variables['z'].data).ndim == 1


def _read_gmt_legacy(variables: dict, band: int = 1) -> dict:
    """
    Reads the legacy GMT netCDF grid format (GMT ≤ 3: x_range, y_range, z_range,
    spacing, dimension and a 1-D z). Rows are stored from north to south. The
    z attribute node_offset=1 means pixel registration (node centres are half
    a cell inside x_range / y_range); 0 means gridline registration.
    """
    if band != 1:
        raise ValueError(f"band={band} is invalid for a legacy GMT grid (single band).")
    xr = np.asarray(variables['x_range'].data, dtype=np.float64).ravel()
    yr = np.asarray(variables['y_range'].data, dtype=np.float64).ravel()
    dim = np.asarray(variables['dimension'].data).ravel().astype(np.int64)
    if xr.size < 2 or yr.size < 2 or dim.size < 2:
        raise ValueError("Legacy GMT grid: x_range, y_range and dimension need two values.")
    nx, ny = int(dim[0]), int(dim[1])
    zvar = variables['z']
    values = _nc_unpack(zvar, zvar.data)
    if values.size != nx * ny:
        raise ValueError(
            f"Legacy GMT grid: z has {values.size} values, expected {nx}×{ny}={nx * ny}"
        )
    node_offset = _nc_attr_numbers(zvar, 'node_offset')
    pixel = node_offset is not None and int(node_offset[0]) == 1
    if pixel:
        dx = (xr[1] - xr[0]) / nx
        dy = (yr[1] - yr[0]) / ny
        x = xr[0] + (np.arange(nx) + 0.5) * dx
        y = yr[1] - (np.arange(ny) + 0.5) * dy
    else:
        x = np.linspace(xr[0], xr[1], nx)
        y = np.linspace(yr[1], yr[0], ny)
    grid = values.reshape((ny, nx))               # row 0 = north → make_layer flips it
    return make_layer(grid, x, y, fmt='netcdf', crs='unknown', variable='z',
                      registration='pixel' if pixel else 'gridline')


def _read_cf_grid(nc, variables: dict, variable=None, band: int = 1) -> dict:
    """Reads a COARDS / CF / GMT 4+ style grid variable with 1-D coordinate variables."""
    coord_names = {n for n, v in variables.items() if tuple(v.dimensions) == (n,)}
    auxiliary = set()
    for v in variables.values():
        coords = _nc_attr_text(v, 'coordinates')
        if coords:
            auxiliary.update(coords.split())
        gm = _nc_attr_text(v, 'grid_mapping')
        if gm:
            auxiliary.add(gm.split()[0].rstrip(':'))

    if variable is not None:
        name = variable if variable in variables else next(
            (n for n in variables if n.lower() == str(variable).lower()), None)
        if name is None:
            raise ValueError(
                f"netCDF variable '{variable}' not found. Available: {sorted(variables)}"
            )
        var = variables[name]
        if np.asarray(var.data).ndim < 2 or not _nc_is_numeric(var):
            raise ValueError(f"netCDF variable '{name}' is not a numeric grid "
                             f"(shape {np.asarray(var.data).shape}).")
    else:
        candidates = []
        for n, v in variables.items():
            shape = np.asarray(v.data).shape
            if (n in coord_names or n in auxiliary or n.lower() in _NC_LONLAT_NAMES
                    or len(shape) < 2 or any(s != 1 for s in shape[:-2])
                    or not _nc_is_numeric(v)):
                continue
            candidates.append(n)
        if not candidates:
            raise _NoGridVariableError(
                f"No 2-D numeric data variable found. Variables: {sorted(variables)}"
            )
        by_lower = {}
        for n in candidates:
            by_lower.setdefault(n.lower(), n)
        name = next((by_lower[p] for p in _NC_PREFERRED_VARIABLES if p in by_lower), None)
        if name is None:
            def _has_coords(n):
                v = variables[n]
                shape = np.asarray(v.data).shape
                return all(_nc_coordinate(variables, d, s) is not None
                           for d, s in zip(v.dimensions[-2:], shape[-2:]))
            name = next((n for n in candidates if _has_coords(n)), candidates[0])
        var = variables[name]

    data = np.asarray(var.data)
    planes = data.reshape((-1,) + data.shape[-2:])
    if not 1 <= band <= planes.shape[0]:
        raise ValueError(
            f"band={band} is invalid for netCDF variable '{name}' "
            f"(number of 2-D planes: {planes.shape[0]})"
        )
    grid = _nc_unpack(var, planes[band - 1])

    dim0, dim1 = var.dimensions[-2:]
    v0 = _nc_coordinate(variables, dim0, grid.shape[0])
    v1 = _nc_coordinate(variables, dim1, grid.shape[1])
    k0 = _nc_axis_kind(dim0, v0) if v0 is not None else None
    k1 = _nc_axis_kind(dim1, v1) if v1 is not None else None
    if (k0 == 'x' and k1 != 'x') or (k1 == 'y' and k0 != 'y'):
        grid = grid.T                          # stored as (x, y)
        (dim0, v0), (dim1, v1) = (dim1, v1), (dim0, v0)
    ydim, yvar, xdim, xvar = dim0, v0, dim1, v1
    ny, nx = grid.shape

    if xvar is None or yvar is None:
        warnings.warn(
            f"netCDF variable '{name}' has no coordinate variables for its dimensions "
            f"({ydim}, {xdim}); pixel coordinates are used.", UserWarning,
        )
        return make_layer(grid, np.arange(nx, dtype=float), np.arange(ny, dtype=float),
                          fmt='netcdf', crs='pixel', variable=name)

    x = _nc_unpack(xvar, xvar.data).ravel()
    y = _nc_unpack(yvar, yvar.data).ravel()

    crs = 'unknown'
    gm = _nc_attr_text(var, 'grid_mapping')
    if gm:
        gm_var = variables.get(gm.split()[0].rstrip(':'))
        if gm_var is not None:
            wkt = _nc_attr_text(gm_var, 'crs_wkt') or _nc_attr_text(gm_var, 'spatial_ref')
            if wkt:
                crs = _normalize_crs(wkt)
    if crs == 'unknown' and _nc_is_geographic(xdim, xvar) and _nc_is_geographic(ydim, yvar):
        crs = 'EPSG:4326'
    return make_layer(grid, x, y, fmt='netcdf', crs=crs, variable=name)


def read_netcdf_grid(filepath: str, variable: str = None, band: int = 1) -> dict:
    """
    Reads a netCDF-3 grid (classic 'CDF\\x01' or 64-bit offset 'CDF\\x02') with
    scipy.io.netcdf_file — no GDAL needed.

    Supported layouts
    -----------------
    COARDS / CF / GMT 4+: a 2-D data variable (preferred names z, elevation,
        Band1, data, value; otherwise the first 2-D numeric variable whose two
        dimensions have 1-D coordinate variables). The coordinate variables are
        the node coordinates (ascending or descending). scale_factor and
        add_offset are applied; _FillValue, missing_value and NaN become NaN.
        Variables with more than two dimensions can be named explicitly; `band`
        (1-based) then selects the 2-D plane along the leading dimension(s).
    Legacy GMT (x_range, y_range, z_range, spacing, dimension and a 1-D z):
        rows are stored north to south; z:node_offset=1 is pixel registration.

    Parameters
    ----------
    filepath : path of the .grd / .nc file
    variable : name of the data variable (None = automatic selection)
    band     : 1-based plane index for variables with more than two dimensions
    """
    with open(filepath, 'rb') as f:
        magic = f.read(4)
    if magic not in _NETCDF3_MAGICS:
        raise ValueError(
            f"{filepath} is not a netCDF-3 file (classic or 64-bit offset); "
            "netCDF-4 / HDF5 files are read through GDAL (pip install rasterio)."
        )
    from scipy.io import netcdf_file

    nc = netcdf_file(filepath, 'r', mmap=False)
    try:
        variables = dict(nc.variables)
        if _is_legacy_gmt(variables) and (variable is None or str(variable).lower() == 'z'):
            return _read_gmt_legacy(variables, band=band)
        return _read_cf_grid(nc, variables, variable=variable, band=band)
    finally:
        nc.close()


def _load_netcdf(filepath: str, variable=None, band: int = 1) -> dict:
    """Native netCDF-3 reader; falls back to GDAL if no usable grid variable is found."""
    try:
        return read_netcdf_grid(filepath, variable=variable, band=band)
    except _NoGridVariableError as e:
        if not _rasterio_available():
            raise
        warnings.warn(f"{filepath}: {e} Trying GDAL instead.", UserWarning)
        return read_raster_gdal(filepath, band=band)


def _gdal_source(filepath: str, variable=None) -> str:
    """GDAL dataset name: the netCDF subdataset of `variable` when one is given."""
    if variable is None:
        return filepath
    with open(filepath, 'rb') as f:
        magic = f.read(4)
    if magic[:3] == b'CDF' or magic == _HDF5_MAGIC:
        return f'NETCDF:"{filepath}":{variable}'
    warnings.warn(f"'variable' is ignored for this raster: {filepath}", UserWarning)
    return filepath


# ─────────────────────────────────────────────────────────────────────────────
# GEOTIFF / GDAL RASTER READERS
# ─────────────────────────────────────────────────────────────────────────────

def read_raster_gdal(filepath: str, band: int = 1, fmt: str = 'gdal') -> dict:
    """
    Reads any GDAL-readable raster with rasterio.

    The band's scale and offset (GDAL scale/offset metadata, e.g. netCDF
    scale_factor/add_offset) are applied, so packed integer rasters are
    returned in physical units. A dataset without georeferencing (identity
    transform and no CRS) is read in pixel coordinates, north-up (the first
    row of the file is the top), with a UserWarning and crs='pixel'.
    """
    import rasterio
    from rasterio.errors import NotGeoreferencedWarning

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', NotGeoreferencedWarning)
        with rasterio.open(filepath) as src:
            if not 1 <= band <= src.count:
                raise ValueError(f"band={band} is invalid (number of bands: {src.count})")
            arr = src.read(band, masked=True)
            grid = np.ma.filled(arr.astype(np.float64), np.nan)
            t = src.transform
            crs_obj = src.crs
            nodata = src.nodata
            scales = src.scales or ()
            offsets = src.offsets or ()
    scale = scales[band - 1] if len(scales) >= band and scales[band - 1] is not None else 1.0
    offset = offsets[band - 1] if len(offsets) >= band and offsets[band - 1] is not None else 0.0
    if scale != 1.0 or offset != 0.0:
        grid = grid * float(scale) + float(offset)
    crs = str(crs_obj) if crs_obj else 'unknown'
    ny, nx = grid.shape

    if tuple(t)[:6] == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0) and not crs_obj:
        warnings.warn(
            f"{filepath}: the raster has no georeferencing (identity transform and "
            "no CRS); pixel coordinates are used.", UserWarning,
        )
        x = np.arange(nx, dtype=float)
        y = np.arange(ny, dtype=float)[::-1]     # row 0 = top = largest y
        return make_layer(grid, x, y, fmt=fmt, crs='pixel', nodata=nodata)

    if t.b == 0 and t.d == 0:
        x = t.c + (np.arange(nx) + 0.5) * t.a
        y = t.f + (np.arange(ny) + 0.5) * t.e
        return make_layer(grid, x, y, fmt=fmt, crs=crs, nodata=nodata)

    cols, rows = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5)
    east = t.c + cols * t.a + rows * t.b
    north = t.f + cols * t.d + rows * t.e
    rotation = float(np.degrees(np.arctan2(t.d, t.a)))
    return _regularize_rotated(grid, east, north, fmt=fmt, crs=crs,
                               nodata=nodata, rotation=rotation)


def read_geotiff(filepath: str, band: int = 1) -> dict:
    """
    Reads a GeoTIFF band. Uses rasterio if available, otherwise Pillow + GeoTIFF
    tags (the Pillow fallback reads band 1 only).
    """
    try:
        import rasterio  # noqa: F401
    except ImportError:
        return _read_tiff_fallback(filepath, band=band)
    return read_raster_gdal(filepath, band=band, fmt='geotiff')


# GeoTIFF tag numbers
_TAG_PIXEL_SCALE = 33550
_TAG_TIEPOINT = 33922
_TAG_TRANSFORM = 34264
_TAG_GEOKEYS = 34735
_TAG_GDAL_NODATA = 42113
_GEOKEY_RASTER_TYPE = 1025   # 1 = PixelIsArea, 2 = PixelIsPoint


def _read_tiff_fallback(filepath: str, band: int = 1) -> dict:
    """
    Reads a TIFF when rasterio is unavailable (via Pillow). If GeoTIFF tags
    (ModelPixelScale + ModelTiepoint or ModelTransformation) are present,
    geographic coordinates are computed from them. Only band 1 can be read.
    """
    if band != 1:
        raise ValueError(
            f"band={band}: the Pillow TIFF fallback reads band 1 only. "
            "Install rasterio to read other bands: pip install rasterio"
        )
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "rasterio or Pillow is required to read GeoTIFF files.\n"
            "  pip install rasterio\n"
            "  or: pip install Pillow"
        )

    with Image.open(filepath) as img:
        tags = dict(img.tag_v2) if hasattr(img, 'tag_v2') else {}
        grid = np.array(img, dtype=np.float64)
    if grid.ndim == 3:
        grid = grid[:, :, 0]
    ny, nx = grid.shape

    nodata = tags.get(_TAG_GDAL_NODATA)
    if nodata is not None:
        try:
            nodata = float(str(nodata).strip('\x00 '))
            grid[grid == nodata] = np.nan
        except ValueError:
            nodata = None

    pixel_is_point = False
    geokeys = tags.get(_TAG_GEOKEYS)
    if geokeys:
        keys = list(geokeys)
        for k in range(4, len(keys) - 3, 4):
            if keys[k] == _GEOKEY_RASTER_TYPE and keys[k + 1] == 0:
                pixel_is_point = keys[k + 3] == 2

    half = 0.0 if pixel_is_point else 0.5
    scale = tags.get(_TAG_PIXEL_SCALE)
    tie = tags.get(_TAG_TIEPOINT)
    mtx = tags.get(_TAG_TRANSFORM)

    if scale is not None and tie is not None and len(tie) >= 6:
        sx, sy = float(scale[0]), float(scale[1])
        i0, j0, X0, Y0 = float(tie[0]), float(tie[1]), float(tie[3]), float(tie[4])
        x = X0 + (np.arange(nx) - i0 + half) * sx
        y = Y0 - (np.arange(ny) - j0 + half) * sy
        return make_layer(grid, x, y, fmt='tiff_pil', crs='unknown', nodata=nodata)

    if mtx is not None and len(mtx) >= 16:
        a, b, _, c, d, e = (float(v) for v in mtx[:6])
        if b == 0 and d == 0:
            x = c + (np.arange(nx) + half) * a
            y = float(mtx[7]) + (np.arange(ny) + half) * e
            return make_layer(grid, x, y, fmt='tiff_pil', crs='unknown', nodata=nodata)

    warnings.warn(
        "rasterio not found and the TIFF has no GeoTIFF coordinate tags; "
        "using pixel coordinates. pip install rasterio is recommended.",
        UserWarning,
    )
    x = np.arange(nx, dtype=float)
    y = np.arange(ny, dtype=float)[::-1]     # row 0 = top = largest y
    return make_layer(grid, x, y, fmt='tiff_pil', crs='pixel', nodata=nodata)


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _tokens(name: str) -> list:
    # The character class deliberately includes Turkish letters so that
    # Turkish column names are tokenized correctly.
    return [t for t in re.split(r'[^0-9a-zçğıöşü]+', str(name).lower()) if t]


def _is_id_like(series: pd.Series) -> bool:
    """True for integer, unique, strictly monotonic columns (station numbers, ids)."""
    v = pd.to_numeric(series, errors='coerce').dropna().to_numpy(dtype=np.float64)
    if v.size < 3 or not np.all(v == np.round(v)):
        return False
    d = np.diff(v)
    return bool(np.all(d > 0) or np.all(d < 0))


def _numeric_fallback(df: pd.DataFrame, fallback_index: int, exclude: tuple):
    """
    Chooses a numeric column when no candidate name matches.

    Preference order: the fallback_index-th numeric column (if it is still
    free), then the free numeric columns positioned after the columns already
    assigned to other roles, then the remaining free columns from the last one
    backwards. Integer id-like columns (unique, monotonic) are skipped when
    another choice exists. Returns (chosen, numeric_columns) or (None, ...).
    """
    numeric_all = df.select_dtypes(include=np.number).columns.tolist()
    numeric_free = [c for c in numeric_all if c not in exclude]
    if not numeric_free:
        return None, numeric_all
    position = {c: i for i, c in enumerate(df.columns)}
    last_used = max((position[c] for c in exclude if c in position), default=-1)
    idx = min(max(0, int(fallback_index)), len(numeric_all) - 1)

    order = []
    if numeric_all[idx] not in exclude:
        order.append(numeric_all[idx])
    order += [c for c in numeric_free if position[c] > last_used and c not in order]
    order += [c for c in reversed(numeric_free) if c not in order]
    chosen = next((c for c in order if not _is_id_like(df[c])), order[0])
    return chosen, numeric_all


def _find_column(
    df: pd.DataFrame,
    explicit: str,
    candidates: list,
    fallback_index: int = 0,
    allow_numeric_fallback: bool = True,
    exclude: tuple = (),
) -> str:
    """
    Finds the column name from the candidate list if it is not given explicitly.

    Search order (each stage is tried for all candidates):
      1. Exact match             ('x' == 'x')
      2. Token match             ('x' ∈ 'x_utm'.split('_'))
      3. Substring match         (candidates with ≥3 letters: 'res' ⊂ 'resistivity')
      4. Numeric column fallback (the fallback_index-th numeric column; if that
                                  one is taken, the next free numeric column after
                                  the assigned ones; id-like columns are avoided)
    Columns in `exclude` (already assigned to another role) are skipped.
    """
    if explicit is not None:
        key = str(explicit).strip().lower()
        if key in df.columns:
            return key
        raise KeyError(f"Column not found: '{explicit}'. Available: {list(df.columns)}")

    cols = [c for c in df.columns if c not in exclude]

    for cand in candidates:
        if cand in cols:
            return cand
    for cand in candidates:
        for col in cols:
            if cand in _tokens(col):
                return col
    for cand in candidates:
        if len(cand) < 3:
            continue
        for col in cols:
            if cand in col:
                return col

    if allow_numeric_fallback:
        chosen, numeric_all = _numeric_fallback(df, fallback_index, tuple(exclude))
        if chosen is not None:
            warnings.warn(
                f"Could not detect column, using numeric fallback: "
                f"'{chosen}' (numeric columns: {numeric_all[:5]})",
                UserWarning
            )
            return chosen

    raise KeyError(
        f"No suitable column found. Available columns: {list(df.columns)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEXT TABLE READER (CSV / XYZ / Geosoft XYZ)
# ─────────────────────────────────────────────────────────────────────────────

_WHITESPACE = r'\s+'
_DELIMITER_CANDIDATES = ('\t', ';', ',', '|')     # in order of preference
_SNIFF_LINES = 50
# Geosoft / Oasis montaj XYZ line records: 'Line 1010', 'Tie 9010', 'Trend 5'
_LINE_RECORD_ML_RE = re.compile(r'^[ \t]*(?:line|tie|trend)[ \t]+(\S+)[ \t]*$',
                                re.IGNORECASE | re.MULTILINE)
# Comment lines: first non-blank character '#' or '/' (Geosoft)
_COMMENT_LINE_RE = re.compile(r'^[ \t]*[#/][^\n]*$', re.MULTILINE)
_SLASH_COMMENT_RE = re.compile(r'^[ \t]*/([^\n]*)$', re.MULTILINE)
_FIRST_RECORD_RE = re.compile(r'^[ \t]*[^\s#/]', re.MULTILINE)
_BLANK_LINE_RE = re.compile(r'^[ \t\f\v]+$', re.MULTILINE)
_DATA_LINE_RE = re.compile(r'^[ \t]*\S[^\n]*', re.MULTILINE)
_DATA_START_RE = re.compile(r'^[ \t]*\S', re.MULTILINE)
_COMMA_DECIMAL_RE = re.compile(r'^[+-]?\d+,\d+(?:[eE][+-]?\d+)?$')
_DOT_DECIMAL_RE = re.compile(r'^[+-]?(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?$')
_QUOTED_RE = re.compile(r'"[^"]*"')
# Geosoft dummy token in XYZ exports
_GEOSOFT_DUMMY_TOKEN = '*'


def _split_fields(line: str, delimiter: str) -> list:
    """Splits one text line into fields (quote-aware for single-character delimiters)."""
    if delimiter == _WHITESPACE:
        return line.split()
    if len(delimiter) == 1:
        try:
            return next(csv.reader([line], delimiter=delimiter))
        except (csv.Error, StopIteration):
            return line.split(delimiter)
    return re.split(delimiter, line.strip())


def _sniff_delimiter_lines(lines: list) -> str:
    """
    Chooses the delimiter whose field count is the same (and > 1) on every
    sample line. Preference order: tab, ';', ',', '|' — ';' wins over ',' when
    both are consistent, as in European CSV files with decimal commas. If no
    candidate is consistent, the one with the most lines sharing its most
    common field count (> 1) is used; otherwise whitespace.
    """
    # Delimiters inside quoted fields do not count
    sample = [_QUOTED_RE.sub('""', ln) for ln in lines[:_SNIFF_LINES] if ln.strip()]
    if not sample:
        return _WHITESPACE
    best = None
    for rank, delim in enumerate(_DELIMITER_CANDIDATES):
        counts = [ln.count(delim) + 1 for ln in sample]
        if min(counts) > 1 and len(set(counts)) == 1:
            return delim
        values, freq = np.unique(counts, return_counts=True)
        multi = values > 1
        if multi.any():
            score = (int(freq[multi].max()), -rank)
            if best is None or score > best[0]:
                best = (score, delim)
    return best[1] if best is not None else _WHITESPACE


def _sniff_delimiter(filepath: str, nbytes: int = 4096) -> str:
    """Delimiter of a text table (comment lines starting with '#' or '/' are ignored)."""
    with open(filepath, 'r', encoding='utf-8-sig', errors='replace') as f:
        lines = [ln for ln in f.read(nbytes).splitlines()
                 if ln.strip() and ln.lstrip()[0] not in '#/']
    return _sniff_delimiter_lines(lines)


def _sniff_decimal(lines: list, delimiter: str) -> str:
    """',' if the numeric-looking fields use decimal commas (and none uses a dot), else '.'."""
    if delimiter == ',':
        return '.'
    n_comma = n_dot = 0
    for ln in lines[:_SNIFF_LINES]:
        for tok in _split_fields(ln, delimiter):
            t = tok.strip().strip('"')
            if _COMMA_DECIMAL_RE.match(t):
                n_comma += 1
            elif _DOT_DECIMAL_RE.match(t):
                n_dot += 1
    return ',' if n_comma > 0 and n_dot == 0 else '.'


def _is_number(token: str, decimal: str = '.') -> bool:
    if decimal != '.':
        token = token.replace(decimal, '.')
    try:
        float(token)
        return True
    except ValueError:
        return False


def _is_data_token(token: str, decimal: str = '.') -> bool:
    return token == _GEOSOFT_DUMMY_TOKEN or _is_number(token, decimal)


def _dedupe_names(names: list) -> list:
    seen = {}
    out = []
    for n in names:
        key = n
        if key in seen:
            seen[n] += 1
            key = f'{n}.{seen[n]}'
        else:
            seen[n] = 0
        out.append(key)
    return out


def _read_table(filepath: str, delimiter: str = None, decimal: str = None) -> pd.DataFrame:
    """
    Reads a CSV / XYZ table into a DataFrame with lower-case column names.

    - Only lines whose first non-blank character is '#' are comments ('#'
      inside a field, as in 'BH#1', is kept).
    - Headerless (fully numeric) files get the columns x, y, z, col3, ...
    - Delimiter: sniffed from the per-line field counts (see
      _sniff_delimiter_lines) unless given.
    - Decimal separator: ',' is detected automatically for non-comma
      delimiters when numeric fields use decimal commas, unless given.
    - Geosoft / Oasis montaj XYZ dialect: lines starting with '/' are comments;
      for a headerless body, the column names come from the last '/' comment
      line before the first data or line record whose token count equals the
      number of data fields. 'Line <id>', 'Tie <id>' and 'Trend <id>' records
      start a new line; the id is attached to the following rows as the string
      column 'line'. The Geosoft dummy '*' is read as missing.
    """
    try:
        with open(filepath, 'r', encoding='utf-8-sig', errors='replace') as f:
            text = f.read()
    except OSError as e:
        raise IOError(f"Could not read table ({filepath}): {e}")

    # Line classification is done with multi-line regular expressions (C speed),
    # since airborne XYZ exports can have millions of rows.
    if '\r' in text:
        text = text.replace('\r\n', '\n').replace('\r', '\n')
    first = _FIRST_RECORD_RE.search(text)
    head_end = first.start() if first else len(text)
    # '/' comment texts before the first data or line record (header candidates)
    comments = [c.strip() for c in _SLASH_COMMENT_RE.findall(text, 0, head_end)]
    text = _COMMENT_LINE_RE.sub('', text)
    text = _BLANK_LINE_RE.sub('', text)
    parts = _LINE_RECORD_ML_RE.split(text)       # [segment0, id1, segment1, id2, ...]
    line_ids = parts[1::2]
    n_line_records = len(line_ids)
    data_text = ''.join(parts[0::2]) if n_line_records else text

    rows = []                                    # first data lines (header included)
    for m in _DATA_LINE_RE.finditer(data_text):
        rows.append(m.group(0))
        if len(rows) >= _SNIFF_LINES:
            break
    if not rows:
        raise IOError(f"Could not read table ({filepath}): no data rows found.")
    row_line = []                                # line id of each data line
    if n_line_records:
        segments = parts[0::2]
        row_line = [None] * len(_DATA_START_RE.findall(segments[0]))
        for line_id, segment in zip(line_ids, segments[1:]):
            row_line.extend([line_id] * len(_DATA_START_RE.findall(segment)))

    if delimiter is None:
        delimiter = _sniff_delimiter_lines(rows)
    if decimal is None:
        decimal = _sniff_decimal(rows, delimiter)

    first_tokens = [t.strip().strip('"') for t in _split_fields(rows[0], delimiter)]
    first_tokens = [t for t in first_tokens if t]
    n_numeric = sum(_is_data_token(t, decimal) for t in first_tokens)
    names = None
    if first_tokens and n_numeric == len(first_tokens):
        headerless = True
    else:
        headerless = False
        # Geosoft dialect: a whitespace data row with some non-numeric fields
        # (e.g. dates) under a matching '/' header comment is still data.
        if comments and delimiter == _WHITESPACE and n_numeric * 2 >= len(first_tokens):
            headerless = any(len(c.split()) == len(first_tokens) for c in comments)

    if headerless:
        n_fields = len(_split_fields(rows[0], delimiter))
        for c in reversed(comments):
            tokens = c.split()
            if len(tokens) == n_fields:
                names = _dedupe_names([t.lower() for t in tokens])
                break
        if names is None:
            default_names = ['x', 'y', 'z']
            names = [default_names[i] if i < 3 else f'col{i}' for i in range(n_fields)]
        body_line = row_line
    else:
        body_line = row_line[1:]

    engine = 'c' if (delimiter == _WHITESPACE or len(delimiter) == 1) else 'python'
    read_kwargs = dict(sep=delimiter, engine=engine, skip_blank_lines=True,
                       na_values=[_GEOSOFT_DUMMY_TOKEN], decimal=decimal)
    if headerless:
        read_kwargs.update(header=None, names=names)
    else:
        read_kwargs['header'] = 0
    try:
        df = pd.read_csv(io.StringIO(data_text), **read_kwargs)
    except Exception as e:
        raise IOError(f"Could not read table ({filepath}): {e}")

    df.columns = [str(c).strip().lower() for c in df.columns]
    data_cols = list(df.columns)
    if n_line_records:
        if len(df) == len(body_line):
            line_col = 'line' if 'line' not in df.columns else 'line_id'
            df[line_col] = pd.Series(body_line, index=df.index, dtype=object)
        else:
            warnings.warn(
                f"{filepath}: line records could not be matched to the data rows; "
                "the line ids were not attached.", UserWarning,
            )
    df = df.dropna(how='all', subset=data_cols)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# XYZ / CSV READER (regular lattice or scattered points → grid)
# ─────────────────────────────────────────────────────────────────────────────

def _unique_sorted(values: np.ndarray, tol: float) -> np.ndarray:
    v = np.sort(values)
    keep = np.concatenate([[True], np.diff(v) > tol])
    return v[keep]


# Largest allowed distance of a lattice coordinate from its node, in cells
_LATTICE_TOL = 0.01


def _detect_lattice(x: np.ndarray, y: np.ndarray, max_nodes: int = 16_000_000):
    """
    Returns (x0, dx, nx, y0, dy, ny) if the points lie on a regular
    lattice (grid nodes); otherwise None.

    The node spacing is estimated from the full span of each axis divided by
    the number of whole steps, not from the smallest difference: coordinates
    rounded to the printed precision (e.g. a 100/3 m spacing written with two
    decimals) make the smallest difference slightly too small, and that error
    would otherwise accumulate along the axis.
    """
    n = len(x)
    if n < 4:
        return None
    span = max(np.ptp(x), np.ptp(y), 1e-12)
    tol = span * 1e-7

    def _axis(v):
        u = _unique_sorted(v, tol)
        if len(u) < 2 or len(u) > n // 2 + 1:
            return None
        d = np.diff(u)
        step = float(np.min(d))
        if step <= tol:
            return None
        total = float(u[-1] - u[0])
        for _ in range(4):
            k = np.maximum(np.round(d / step), 1.0)
            new_step = total / float(k.sum())
            converged = abs(new_step - step) <= 1e-12 * abs(step)
            step = new_step
            if converged:
                break
        ratio = d / step
        k = np.round(ratio)
        if np.any(k < 1) or np.max(np.abs(ratio - k)) > _LATTICE_TOL:
            return None
        count = int(k.sum()) + 1
        return float(u[0]), step, count

    ax = _axis(x)
    ay = _axis(y)
    if ax is None or ay is None:
        return None
    x0, dx, nx = ax
    y0, dy, ny = ay
    # The lattice must be at least 20% filled; otherwise (e.g. sparse scattered
    # samples with integer coordinates) interpolation is more appropriate.
    if nx * ny > max_nodes or nx * ny > 5 * n:
        return None
    fx = (x - x0) / dx
    fy = (y - y0) / dy
    if (np.max(np.abs(fx - np.round(fx))) > _LATTICE_TOL or
            np.max(np.abs(fy - np.round(fy))) > _LATTICE_TOL):
        return None
    return x0, dx, nx, y0, dy, ny


def _mask_dummies(values: np.ndarray) -> np.ndarray:
    """Surfer blanks (≥ 1.70141e38) and Geosoft dummies (≤ -1e32) → NaN."""
    values = np.array(values, dtype=np.float64, copy=True)
    with np.errstate(invalid='ignore'):
        bad = (values >= SURFER_BLANK * (1 - 1e-6)) | (values <= GEOSOFT_DUMMY * (1 - 1e-6))
    values[bad] = np.nan
    return values


def _nodata_list(nodata) -> list:
    if nodata is None:
        return []
    if isinstance(nodata, (str, bytes)) or np.ndim(nodata) == 0:
        return [float(nodata)]
    return [float(v) for v in np.ravel(nodata)]


def read_xyz(
    filepath: str,
    x_col: str = None,
    y_col: str = None,
    z_col: str = None,
    delimiter: str = None,
    target_nx: int = 200,
    target_ny: int = 200,
    interp_method: str = 'linear',
    smooth_sigma: float = 0.5,
    grid_method: str = 'auto',
    max_distance: float = None,
    nodata=None,
    decimal: str = None,
) -> dict:
    """
    Reads XYZ/CSV points and converts them to a grid.

    Parameters
    ----------
    filepath       : File path (CSV, whitespace XYZ or Geosoft / Oasis montaj XYZ)
    x_col, y_col, z_col : Column names (auto-detected if None)
    delimiter      : Delimiter (auto-detected if None)
    target_nx/ny   : Output grid size for scattered data
    interp_method  : 'linear', 'nearest', 'cubic'
    smooth_sigma   : Gaussian smoothing after interpolation (pixels; 0 = off)
    grid_method    : 'auto'  → direct gridding if the points lie on a regular
                                lattice, otherwise interpolation
                     'lattice' / 'interpolate' → force that method
    max_distance   : For scattered data, nodes farther than this distance
                     (in coordinate units) from the nearest measurement point
                     are set to NaN (None = off)
    nodata         : Extra value(s) of z to treat as missing (a number or a
                     sequence). Surfer blanks (≥ 1.70141e38), Geosoft dummies
                     (≤ -1e32) and '*' tokens are always missing.
    decimal        : Decimal separator ('.' or ','); None = auto-detect
                     (decimal commas in ';'-separated files)

    Missing nodes of a regular lattice stay in the lattice as NaN.
    """
    df = _read_table(filepath, delimiter, decimal=decimal)

    x_col = _find_column(
        df, x_col, ['x', 'easting', 'east', 'utm_e', 'lon', 'long', 'longitude', 'e'],
        fallback_index=0, allow_numeric_fallback=True,
    )
    y_col = _find_column(
        df, y_col, ['y', 'northing', 'north', 'utm_n', 'lat', 'latitude', 'n'],
        fallback_index=1, allow_numeric_fallback=True, exclude=(x_col,),
    )
    z_col = _find_column(
        df, z_col, ['z', 'value', 'val', 'data', 'field', 'mag', 'tmi', 'grav',
                    'res', 'ip', 'signal', 'depth'],
        fallback_index=2, allow_numeric_fallback=True, exclude=(x_col, y_col),
    )

    x = _mask_dummies(pd.to_numeric(df[x_col], errors='coerce').to_numpy(dtype=np.float64))
    y = _mask_dummies(pd.to_numeric(df[y_col], errors='coerce').to_numpy(dtype=np.float64))
    z = _mask_dummies(pd.to_numeric(df[z_col], errors='coerce').to_numpy(dtype=np.float64))
    for value in _nodata_list(nodata):
        if np.isfinite(value):
            z[np.isclose(z, value, rtol=1e-7, atol=0)] = np.nan

    has_xy = np.isfinite(x) & np.isfinite(y)
    valid = has_xy & np.isfinite(z)
    n_valid = int(valid.sum())

    if n_valid < 10:
        raise ValueError(f"Not enough data points: {n_valid} points")

    if grid_method not in ('auto', 'lattice', 'interpolate'):
        raise ValueError(f"Invalid grid_method: {grid_method}")

    # Points with coordinates but a missing value (blank nodes) still define the lattice
    lattice = None if grid_method == 'interpolate' else _detect_lattice(x[has_xy], y[has_xy])
    if grid_method == 'lattice' and lattice is None:
        raise ValueError("Points do not form a regular lattice (grid_method='lattice').")

    x, y, z = x[valid], y[valid], z[valid]

    if lattice is not None:
        x0, dx, nx, y0, dy, ny = lattice
        ix = np.round((x - x0) / dx).astype(np.int64)
        iy = np.round((y - y0) / dy).astype(np.int64)
        flat = iy * nx + ix
        sums = np.bincount(flat, weights=z, minlength=nx * ny)
        counts = np.bincount(flat, minlength=nx * ny)
        grid = np.full(nx * ny, np.nan)
        has = counts > 0
        grid[has] = sums[has] / counts[has]
        grid = grid.reshape(ny, nx)
        xi = x0 + np.arange(nx) * dx
        yi = y0 + np.arange(ny) * dy
        return make_layer(grid, xi, yi, fmt='xyz_lattice', n_points=int(len(x)),
                          columns=(x_col, y_col, z_col))

    # Average duplicate points that share the same coordinate
    pts = pd.DataFrame({'x': x, 'y': y, 'z': z}).groupby(['x', 'y'], as_index=False)['z'].mean()
    x, y, z = pts['x'].to_numpy(), pts['y'].to_numpy(), pts['z'].to_numpy()

    xi = np.linspace(x.min(), x.max(), int(target_nx))
    yi = np.linspace(y.min(), y.max(), int(target_ny))
    Xi, Yi = np.meshgrid(xi, yi)

    try:
        grid = griddata((x, y), z, (Xi, Yi), method=interp_method)
    except Exception as e:   # e.g. all points on a single line (QhullError)
        warnings.warn(
            f"'{interp_method}' interpolation failed ({type(e).__name__}); "
            "falling back to 'nearest'.", UserWarning,
        )
        grid = griddata((x, y), z, (Xi, Yi), method='nearest')

    # Fill nodes outside the convex hull with nearest
    nan_mask = np.isnan(grid)
    if nan_mask.any():
        grid_nearest = griddata((x, y), z, (Xi, Yi), method='nearest')
        grid[nan_mask] = grid_nearest[nan_mask]

    if smooth_sigma and smooth_sigma > 0:
        grid = gaussian_filter(grid, sigma=smooth_sigma)

    if max_distance is not None:
        from scipy.spatial import cKDTree
        dist, _ = cKDTree(np.column_stack([x, y])).query(
            np.column_stack([Xi.ravel(), Yi.ravel()]), k=1
        )
        grid[dist.reshape(grid.shape) > max_distance] = np.nan

    return make_layer(grid, xi, yi, fmt='xyz_interpolated', n_points=int(len(x)),
                      columns=(x_col, y_col, z_col))


# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSAL LOADER
# ─────────────────────────────────────────────────────────────────────────────

_XYZ_KWARGS = {'x_col', 'y_col', 'z_col', 'delimiter', 'target_nx', 'target_ny',
               'interp_method', 'smooth_sigma', 'grid_method', 'max_distance',
               'nodata', 'decimal'}


def load_grid(filepath: str, **kwargs) -> dict:
    """
    Reads any supported format automatically through a single function.

    Parameters
    ----------
    filepath : file path
    **kwargs : read_xyz parameters for XYZ/CSV; for raster formats,
               'band' (multi-band GeoTIFF/ERS/GDAL files, 1-based) and
               'variable' (netCDF data variable name).

    Returns
    -------
    dict: grid (2D array, row 0 = south), x, y vectors and metadata
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    fmt = detect_format(filepath)
    band = kwargs.pop('band', 1)
    variable = kwargs.pop('variable', None)
    if variable is not None and fmt not in ('netcdf', 'gdal'):
        warnings.warn(f"'variable' is ignored for format '{fmt}'.", UserWarning)

    if fmt in ('xyz', 'csv'):
        unknown = set(kwargs) - _XYZ_KWARGS
        if unknown:
            raise TypeError(f"Unknown parameter(s) for read_xyz: {sorted(unknown)}")
        result = read_xyz(filepath, **kwargs)
    else:
        if kwargs:
            warnings.warn(
                f"XYZ parameters ignored for format '{fmt}': {sorted(kwargs)}",
                UserWarning,
            )
        loaders = {
            'surfer_ascii': read_surfer_ascii_grd,
            'surfer_binary': read_surfer_binary_grd,
            'surfer7_binary': read_surfer7_binary_grd,
            'geosoft_grd': read_geosoft_grd,
            'esri_ascii': read_esri_ascii,
            'netcdf': lambda f: _load_netcdf(f, variable=variable, band=band),
            'geotiff': lambda f: read_geotiff(f, band=band),
            'ermapper': lambda f: read_ermapper_ers(f, band=band),
            'gdal': lambda f: read_raster_gdal(_gdal_source(f, variable), band=band),
        }
        if fmt not in loaders:
            raise ValueError(
                f"Unsupported format: '{fmt}' ({filepath})\n"
                f"Supported: {sorted(list(loaders) + ['xyz', 'csv'])}"
            )
        result = loaders[fmt](filepath)

    result['filepath'] = str(filepath)
    result['name'] = Path(filepath).stem

    n_nan = int(np.isnan(result['grid']).sum())
    pct = 100 * n_nan / result['grid'].size
    result['nan_percent'] = float(pct)
    if pct >= 100:
        raise ValueError(f"{filepath}: grid is entirely empty (NaN).")
    if pct > 50:
        warnings.warn(
            f"{filepath}: {pct:.1f}% missing data (NaN). "
            "Try grid interpolation or a different format.", UserWarning
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# WELL / DRILLHOLE LOADER
# ─────────────────────────────────────────────────────────────────────────────

# Accepted label strings; 'evet'/'var'/'pozitif' and 'hayir'/'yok'/'negatif'
# are Turkish aliases for yes/no and must be kept.
_POSITIVE_STRINGS = {'1', '1.0', 'true', 't', 'yes', 'y', 'pos', 'positive',
                     'evet', 'var', 'pozitif', '+'}
_NEGATIVE_STRINGS = {'0', '0.0', 'false', 'f', 'no', 'n', 'neg', 'negative',
                     'hayir', 'hayır', 'yok', 'negatif', '-'}

# Column names accepted as TARGET_TYPE. Only exact names (after normalising
# spaces / hyphens to '_') are used: generic names such as 'type' would also
# match drillhole columns like ROCK_TYPE, HOLE_TYPE or a TYPE column holding
# the drilling method (DD, RC). 'hedef_tipi' is the Turkish alias.
_TARGET_TYPE_COLUMNS = ('target_type', 'targettype', 'resource_type', 'commodity',
                        'deposit_type', 'hedef_tipi')


def parse_labels(series) -> pd.Series:
    """
    Converts well labels to a float Series (the index is preserved).

    Returns 1.0 for positive and 0.0 for negative labels, NaN for missing
    values and +inf for unrecognised values (e.g. 'maybe' or 2); callers
    should raise an error when the result contains inf. Accepted labels:
    0/1 (numbers, numeric text or booleans), 'yes/no', 'y/n', 'true/false',
    'pos/neg', 'positive/negative', '+/-' and the Turkish aliases
    'evet/hayir', 'var/yok', 'pozitif/negatif' (case-insensitive).
    """
    series = pd.Series(series) if not isinstance(series, pd.Series) else series
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        out = series.astype(float)
    else:
        def _map(v):
            if pd.isna(v):
                return np.nan
            if isinstance(v, (bool, np.bool_)):
                return float(v)
            s = str(v).strip().lower()
            if s in _POSITIVE_STRINGS:
                return 1.0
            if s in _NEGATIVE_STRINGS:
                return 0.0
            if s == '':
                return np.nan
            try:
                return float(s)
            except ValueError:
                return float('inf')        # invalid → triggers an error downstream
        out = series.map(_map).astype(float)
    bad = out.notna() & ~out.isin([0.0, 1.0])
    if bad.any():
        out = out.copy()
        out[bad] = np.inf
    return out


# Backward-compatible private alias
_parse_labels = parse_labels


def _normalize_column_name(name: str) -> str:
    return re.sub(r'[\s\-.]+', '_', str(name).strip().lower())


def _find_target_type_column(df: pd.DataFrame, exclude: tuple = ()):
    """
    Auto-detects the TARGET_TYPE column: exact names from _TARGET_TYPE_COLUMNS
    only. A candidate whose non-null values are all numeric (e.g. a code) is
    rejected. Returns the column name or None.
    """
    for cand in _TARGET_TYPE_COLUMNS:
        for col in df.columns:
            if col in exclude or _normalize_column_name(col) != cand:
                continue
            values = df[col].dropna()
            if len(values) and pd.to_numeric(
                    values.astype(str).str.strip(), errors='coerce').notna().all():
                continue
            return col
    return None


def load_well_data(
    filepath: str,
    x_col: str = None,
    y_col: str = None,
    label_col: str = None,
    target_type_col=None,
    delimiter: str = None,
    target_type: str = None,
    decimal: str = None,
) -> pd.DataFrame:
    """
    Loads well / drillhole data.

    Expected format (CSV, also ';'-separated with decimal commas, or Geosoft XYZ):
      X, Y, LABEL, TARGET_TYPE (optional), DEPTH, NOTES, WELL_ID, etc.

    LABEL: 1 = positive (ore/water/geothermal resource found), 0 = negative.
           'yes/no', 'true/false' and Turkish 'evet/hayir' are also accepted.
    target_type_col: TARGET_TYPE column name. None = auto-detect from the exact
           names target_type, targettype, resource_type, commodity, deposit_type
           and hedef_tipi (generic names such as TYPE or ROCK_TYPE are never
           auto-detected); False = no TARGET_TYPE column.
    target_type: if given and the file has a TARGET_TYPE column, only wells
                 of this type (or with an empty type) are returned.
    decimal: decimal separator (None = auto-detect).

    Returns
    -------
    pd.DataFrame: standard columns x, y, label (+ target_type, depth, notes, well_id)
    """
    df = _read_table(filepath, delimiter, decimal=decimal)
    n_raw = len(df)

    x_col = _find_column(
        df, x_col, ['x', 'easting', 'east', 'utm_e', 'lon', 'long', 'longitude'],
        fallback_index=0, allow_numeric_fallback=True
    )
    y_col = _find_column(
        df, y_col, ['y', 'northing', 'north', 'utm_n', 'lat', 'latitude'],
        fallback_index=1, allow_numeric_fallback=True, exclude=(x_col,)
    )
    used = [x_col, y_col]
    # Candidate lists below include Turkish column-name aliases
    # (e.g. 'hedef_tipi', 'etiket', 'sonuc', 'derinlik', 'aciklama', 'kuyu_id').
    if target_type_col is False:
        type_col = None
    elif target_type_col is not None:
        type_col = _find_column(df, target_type_col, [], allow_numeric_fallback=False)
    else:
        type_col = _find_target_type_column(df, exclude=tuple(used))
    if type_col is not None:
        used.append(type_col)
    # Columns named like a target type are never taken as the label
    type_like = tuple(c for c in df.columns if _normalize_column_name(c) in _TARGET_TYPE_COLUMNS)
    label_col = _find_column(
        df, label_col,
        ['label', 'result', 'positive', 'hit', 'found', 'mineralized',
         'success', 'class', 'outcome', 'etiket', 'sonuc', 'sonuç', 'target'],
        allow_numeric_fallback=False, exclude=tuple(used) + type_like,
    )
    used.append(label_col)

    out = pd.DataFrame({
        'x': pd.to_numeric(df[x_col], errors='coerce'),
        'y': pd.to_numeric(df[y_col], errors='coerce'),
        'label': parse_labels(df[label_col]),
    })
    if type_col is not None:
        out['target_type'] = df[type_col].astype('string').str.strip().str.lower()

    for dst_col, candidates in [
        ('depth', ['depth', 'derinlik', 'depth_m', 'total_depth', 'td']),
        ('notes', ['notes', 'note', 'comment', 'comments', 'aciklama', 'açıklama',
                   'description']),
        ('well_id', ['well_id', 'hole_id', 'holeid', 'bhid', 'kuyu_id', 'id',
                     'well', 'hole', 'name']),
    ]:
        try:
            found = _find_column(df, None, candidates, allow_numeric_fallback=False,
                                 exclude=tuple(used))
            out[dst_col] = df[found]
            used.append(found)
        except KeyError:
            pass

    out = out.dropna(subset=['x', 'y', 'label'])
    out = out[np.isfinite(out['x']) & np.isfinite(out['y'])]
    invalid_labels = out.loc[~out['label'].isin([0, 1]), 'label']
    if len(invalid_labels) > 0:
        bad_idx = invalid_labels.index[:5]
        preview = ", ".join(df.loc[bad_idx, label_col].astype(str).tolist())
        raise ValueError(
            "LABEL column must contain only 0 or 1 (or yes/no, true/false). "
            f"Invalid examples: {preview}"
        )
    out['label'] = out['label'].astype(int)

    n_dropped = n_raw - len(out)
    if n_dropped > 0:
        warnings.warn(
            f"{filepath}: {n_dropped} rows skipped due to missing/invalid X, Y "
            "or LABEL.", UserWarning,
        )

    if target_type is not None and 'target_type' in out.columns:
        wanted = str(target_type).strip().lower()
        types = out['target_type']
        keep = types.isna() | (types == '') | (types == wanted)
        keep = keep.fillna(True).astype(bool)
        n_other = int((~keep).sum())
        if n_other and not keep.any():
            found = sorted(set(types.dropna().astype(str)) - {''})[:5]
            warnings.warn(
                f"{filepath}: all {n_other} wells have a TARGET_TYPE other than "
                f"'{wanted}' (column '{type_col}', values: {found}); no wells remain. "
                "Pass target_type_col=False if this column is not a target type.",
                UserWarning,
            )
        out = out[keep]
        if n_other:
            print(f"  ℹ {n_other} wells filtered out for '{wanted}' "
                  f"due to a different TARGET_TYPE.")

    out = out.reset_index(drop=True)
    n_pos = int((out['label'] == 1).sum())
    n_neg = int((out['label'] == 0).sum())
    print(f"  ✓ Well data loaded: {len(out)} wells | "
          f"Positive: {n_pos} | Negative: {n_neg}")

    return out
