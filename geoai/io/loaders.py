"""
geoai.io.loaders
================
Readers for geophysical file formats:
  - Surfer GRD: ASCII (DSAA), Binary 6 (DSBB), Binary 7 (DSRB)
  - Geosoft / Oasis Montaj GRD (uncompressed + zlib-compressed)
  - ER Mapper ERS (.ers header + raw data file)
  - ESRI ASCII Grid (.asc)
  - GeoTIFF / TIFF (rasterio; otherwise Pillow + GeoTIFF tags)
  - Other rasters readable by GDAL (if rasterio is installed; e.g. GMT netCDF .grd)
  - XYZ / CSV / DAT (regular lattice → direct grid; scattered points → interpolation)
  - Well / drillhole CSV

All grid readers return the same dictionary structure:
    grid : (ny, nx) float64 — row 0 = southernmost row, NaN = no data
    x, y : pixel-centre coordinate vectors in ascending order
    xmin, xmax, ymin, ymax, nx, ny, format (+ format-specific extra fields)
"""

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


def _regularize_rotated(grid, east2d, north2d, fmt, **extra) -> dict:
    """
    Resamples a rotated grid onto an axis-parallel regular grid by
    linear interpolation. Corner regions that fall outside the rotated
    rectangle become NaN.
    """
    valid = np.isfinite(grid)
    if valid.sum() < 3:
        raise ValueError("Not enough valid nodes in the rotated grid.")
    pts_e = east2d[valid]
    pts_n = north2d[valid]

    # Original node spacing (distance between neighbouring nodes)
    d_col = np.hypot(east2d[0, 1] - east2d[0, 0], north2d[0, 1] - north2d[0, 0]) \
        if grid.shape[1] > 1 else 1.0
    d_row = np.hypot(east2d[1, 0] - east2d[0, 0], north2d[1, 0] - north2d[0, 0]) \
        if grid.shape[0] > 1 else 1.0
    step = float(min(d_col, d_row))

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
    out = griddata((pts_e, pts_n), grid[valid], (Xi, Yi), method='linear')

    warnings.warn(
        f"Rotated grid ({extra.get('rotation', '?')}°) was resampled onto an "
        f"axis-parallel {ny}×{nx} grid by linear interpolation.",
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
    '.nc': 'gdal',
    '.img': 'gdal',
    '.vrt': 'gdal',
    '.bil': 'gdal',
}


def detect_format(filepath: str) -> str:
    """
    Detects the format from the file extension and content signature (magic bytes).

    Returned values: 'surfer_ascii', 'surfer_binary', 'surfer7_binary',
    'geosoft_grd', 'ermapper', 'esri_ascii', 'geotiff', 'gdal', 'xyz',
    'csv', 'unknown'.
    """
    ext = Path(filepath).suffix.lower()
    fmt = _EXT_MAP.get(ext, 'unknown')

    if fmt == 'auto_grd':
        return _detect_grd_type(filepath)
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
    if header[:3] == b'CDF' or header[:4] == b'\x89HDF':
        return 'gdal'   # GMT / COARDS netCDF grid
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
_GEOSOFT_FLOAT_DUMMY = -1.0e32


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


def read_ermapper_ers(filepath: str, band: int = 1) -> dict:
    """
    Reads an ER Mapper ERS raster (.ers text header + raw binary data).

    Data file: the 'DataFile' key, otherwise the same name without extension.
    If 'HeaderOffset' is present, the data is stored inside the .ers file.
    Data layout is BIL (line by line, bands consecutive within each line).
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
    grid = cube[:, band - 1, :].astype(np.float64)

    null = ri.get('nullcellvalue')
    if null is not None:
        grid[grid == float(null)] = np.nan

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
    proj = cs.get('projection')
    return make_layer(grid, x, y, fmt='ermapper',
                      crs=proj if proj else 'unknown')


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
# GEOTIFF / GDAL RASTER READERS
# ─────────────────────────────────────────────────────────────────────────────

def read_raster_gdal(filepath: str, band: int = 1, fmt: str = 'gdal') -> dict:
    """Reads any GDAL-readable raster with rasterio."""
    import rasterio

    with rasterio.open(filepath) as src:
        if not 1 <= band <= src.count:
            raise ValueError(f"band={band} is invalid (number of bands: {src.count})")
        arr = src.read(band, masked=True)
        grid = np.ma.filled(arr.astype(np.float64), np.nan)
        t = src.transform
        crs = str(src.crs) if src.crs else 'unknown'
        nodata = src.nodata
    ny, nx = grid.shape

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


def read_geotiff(filepath: str) -> dict:
    """Reads a GeoTIFF. Uses rasterio if available, otherwise Pillow + GeoTIFF tags."""
    try:
        import rasterio  # noqa: F401
    except ImportError:
        return _read_tiff_fallback(filepath)
    return read_raster_gdal(filepath, fmt='geotiff')


# GeoTIFF tag numbers
_TAG_PIXEL_SCALE = 33550
_TAG_TIEPOINT = 33922
_TAG_TRANSFORM = 34264
_TAG_GEOKEYS = 34735
_TAG_GDAL_NODATA = 42113
_GEOKEY_RASTER_TYPE = 1025   # 1 = PixelIsArea, 2 = PixelIsPoint


def _read_tiff_fallback(filepath: str) -> dict:
    """
    Reads a TIFF when rasterio is unavailable (via Pillow). If GeoTIFF tags
    (ModelPixelScale + ModelTiepoint or ModelTransformation) are present,
    geographic coordinates are computed from them.
    """
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
      4. Numeric column fallback (the fallback_index-th numeric column)
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
        numeric_all = df.select_dtypes(include=np.number).columns.tolist()
        numeric_free = [c for c in numeric_all if c not in exclude]
        if numeric_free:
            idx = min(max(0, int(fallback_index)), len(numeric_all) - 1)
            chosen = numeric_all[idx] if numeric_all[idx] not in exclude else numeric_free[0]
            warnings.warn(
                f"Could not detect column, using numeric fallback: "
                f"'{chosen}' (numeric columns: {numeric_all[:5]})",
                UserWarning
            )
            return chosen

    raise KeyError(
        f"No suitable column found. Available columns: {list(df.columns)}"
    )


def _sniff_delimiter(filepath: str, nbytes: int = 4096) -> str:
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = [ln for ln in f.read(nbytes).splitlines()
                 if ln.strip() and not ln.lstrip().startswith('#')]
    sample = '\n'.join(lines[:20])
    if '\t' in sample:
        return '\t'
    if ',' in sample:
        return ','
    if ';' in sample:
        return ';'
    return r'\s+'


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _read_table(filepath: str, delimiter: str = None) -> pd.DataFrame:
    """Reads a CSV/XYZ table; also recognises headerless (fully numeric) files."""
    if delimiter is None:
        delimiter = _sniff_delimiter(filepath)

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        first = ''
        for line in f:
            if line.strip() and not line.lstrip().startswith('#'):
                first = line.strip()
                break
    split_re = r'\s+' if delimiter == r'\s+' else re.escape(delimiter)
    first_tokens = [t.strip().strip('"') for t in re.split(split_re, first) if t.strip()]
    headerless = bool(first_tokens) and all(_is_number(t) for t in first_tokens)

    try:
        df = pd.read_csv(
            filepath, sep=delimiter, engine='python', comment='#',
            skip_blank_lines=True, header=None if headerless else 'infer',
        )
    except Exception as e:
        raise IOError(f"Could not read table ({filepath}): {e}")

    if headerless:
        default_names = ['x', 'y', 'z']
        df.columns = [default_names[i] if i < 3 else f'col{i}' for i in range(df.shape[1])]
    df = df.dropna(how='all')
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# XYZ / CSV READER (regular lattice or scattered points → grid)
# ─────────────────────────────────────────────────────────────────────────────

def _unique_sorted(values: np.ndarray, tol: float) -> np.ndarray:
    v = np.sort(values)
    keep = np.concatenate([[True], np.diff(v) > tol])
    return v[keep]


def _detect_lattice(x: np.ndarray, y: np.ndarray, max_nodes: int = 16_000_000):
    """
    Returns (x0, dx, nx, y0, dy, ny) if the points lie on a regular
    lattice (grid nodes); otherwise None.
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
        ratio = d / step
        if np.max(np.abs(ratio - np.round(ratio))) > 1e-3:
            return None
        count = int(np.round((u[-1] - u[0]) / step)) + 1
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
    if np.max(np.abs(fx - np.round(fx))) > 1e-3 or np.max(np.abs(fy - np.round(fy))) > 1e-3:
        return None
    return x0, dx, nx, y0, dy, ny


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
) -> dict:
    """
    Reads XYZ/CSV points and converts them to a grid.

    Parameters
    ----------
    filepath       : File path
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
    """
    df = _read_table(filepath, delimiter)

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

    x = pd.to_numeric(df[x_col], errors='coerce').to_numpy(dtype=np.float64)
    y = pd.to_numeric(df[y_col], errors='coerce').to_numpy(dtype=np.float64)
    z = pd.to_numeric(df[z_col], errors='coerce').to_numpy(dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]

    if len(x) < 10:
        raise ValueError(f"Not enough data points: {len(x)} points")

    if grid_method not in ('auto', 'lattice', 'interpolate'):
        raise ValueError(f"Invalid grid_method: {grid_method}")

    lattice = None if grid_method == 'interpolate' else _detect_lattice(x, y)
    if grid_method == 'lattice' and lattice is None:
        raise ValueError("Points do not form a regular lattice (grid_method='lattice').")

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
               'interp_method', 'smooth_sigma', 'grid_method', 'max_distance'}


def load_grid(filepath: str, **kwargs) -> dict:
    """
    Reads any supported format automatically through a single function.

    Parameters
    ----------
    filepath : file path
    **kwargs : read_xyz parameters for XYZ/CSV; for raster formats,
               'band' (multi-band ERS/GDAL files).

    Returns
    -------
    dict: grid (2D array, row 0 = south), x, y vectors and metadata
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    fmt = detect_format(filepath)
    band = kwargs.pop('band', 1)

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
            'geotiff': read_geotiff,
            'ermapper': lambda f: read_ermapper_ers(f, band=band),
            'gdal': lambda f: read_raster_gdal(f, band=band),
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


def _parse_labels(series: pd.Series) -> pd.Series:
    """Converts labels to numbers; 'yes/no', 'true/false' and Turkish 'evet/hayir' are recognised."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    def _map(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        if s in _POSITIVE_STRINGS:
            return 1.0
        if s in _NEGATIVE_STRINGS:
            return 0.0
        try:
            return float(s)
        except ValueError:
            return np.nan if s == '' else float('inf')   # invalid → triggers an error downstream
    return series.map(_map).astype(float)


def load_well_data(
    filepath: str,
    x_col: str = None,
    y_col: str = None,
    label_col: str = None,
    target_type_col: str = None,
    delimiter: str = None,
    target_type: str = None,
) -> pd.DataFrame:
    """
    Loads well / drillhole data.

    Expected format (CSV):
      X, Y, LABEL, TARGET_TYPE (optional), DEPTH, NOTES, WELL_ID, etc.

    LABEL: 1 = positive (ore/water/geothermal resource found), 0 = negative.
           'yes/no', 'true/false' and Turkish 'evet/hayir' are also accepted.
    target_type: if given and the file has a TARGET_TYPE column, only wells
                 of this type (or with an empty type) are returned.

    Returns
    -------
    pd.DataFrame: standard columns x, y, label (+ target_type, depth, notes, well_id)
    """
    df = _read_table(filepath, delimiter)
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
    try:
        type_col = _find_column(
            df, target_type_col,
            ['target_type', 'targettype', 'resource_type', 'hedef_tipi', 'type', 'category'],
            allow_numeric_fallback=False, exclude=tuple(used),
        )
        used.append(type_col)
    except KeyError:
        if target_type_col is not None:
            raise
        type_col = None
    label_col = _find_column(
        df, label_col,
        ['label', 'result', 'positive', 'hit', 'found', 'mineralized',
         'success', 'class', 'outcome', 'etiket', 'sonuc', 'sonuç', 'target'],
        allow_numeric_fallback=False, exclude=tuple(used),
    )
    used.append(label_col)

    out = pd.DataFrame({
        'x': pd.to_numeric(df[x_col], errors='coerce'),
        'y': pd.to_numeric(df[y_col], errors='coerce'),
        'label': _parse_labels(df[label_col]),
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
        n_other = int((~keep).sum())
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
