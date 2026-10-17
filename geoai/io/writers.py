"""
geoai.io.writers
================
Grid yazıcıları — sonuç haritalarını (olasılık, belirsizlik, skor) GIS ve
jeofizik yazılımlarının doğrudan açabileceği formatlarda kaydeder.

  - Surfer ASCII GRD (DSAA)  : Surfer, Oasis Montaj, QGIS (GDAL GSAG)
  - Surfer 7 Binary GRD      : Surfer 7+, QGIS (GDAL GS7BG)
  - ESRI ASCII Grid (.asc)   : ArcGIS, QGIS (GDAL AAIGrid)
  - GeoTIFF (.tif)           : her GIS — rasterio gerektirir

Tüm yazıcılar loaders.py'deki konvansiyonu bekler:
grid (ny, nx), satır 0 = güney; x, y artan ve eşit aralıklı piksel merkezleri.
"""

import struct
from pathlib import Path

import numpy as np

from geoai.io.loaders import SURFER_BLANK


def _check_regular(grid, x, y):
    grid = np.asarray(grid, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if grid.ndim != 2 or grid.shape != (len(y), len(x)):
        raise ValueError(
            f"Grid/koordinat boyutu uyuşmuyor: grid {grid.shape}, "
            f"len(y)={len(y)}, len(x)={len(x)}"
        )
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Grid en az 2×2 olmalı.")
    dx = np.diff(x)
    dy = np.diff(y)
    if np.any(dx <= 0) or np.any(dy <= 0):
        raise ValueError("x ve y artan sırada olmalı (satır 0 = güney).")
    if not (np.allclose(dx, dx[0], rtol=1e-6, atol=0) and
            np.allclose(dy, dy[0], rtol=1e-6, atol=0)):
        raise ValueError("Grid düzenli (eşit aralıklı) olmalı.")
    return grid, x, y, float(dx.mean()), float(dy.mean())


def _prepare_path(filepath) -> Path:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_surfer_ascii_grd(grid, x, y, filepath) -> str:
    """Surfer ASCII GRD (DSAA) olarak kaydeder. NaN → Surfer blank."""
    grid, x, y, _, _ = _check_regular(grid, x, y)
    path = _prepare_path(filepath)
    ny, nx = grid.shape
    valid = np.isfinite(grid)
    zmin = float(grid[valid].min()) if valid.any() else 0.0
    zmax = float(grid[valid].max()) if valid.any() else 0.0
    out = np.where(valid, grid, SURFER_BLANK)

    with open(path, 'w', encoding='ascii') as f:
        f.write('DSAA\n')
        f.write(f'{nx} {ny}\n')
        f.write(f'{x[0]:.10g} {x[-1]:.10g}\n')
        f.write(f'{y[0]:.10g} {y[-1]:.10g}\n')
        f.write(f'{zmin:.10g} {zmax:.10g}\n')
        for row in out:
            f.write(' '.join(f'{v:.10g}' for v in row))
            f.write('\n')
    return str(path)


def save_surfer7_binary_grd(grid, x, y, filepath) -> str:
    """Surfer 7 Binary GRD (DSRB, version 2) olarak kaydeder."""
    grid, x, y, dx, dy = _check_regular(grid, x, y)
    path = _prepare_path(filepath)
    ny, nx = grid.shape
    valid = np.isfinite(grid)
    zmin = float(grid[valid].min()) if valid.any() else 0.0
    zmax = float(grid[valid].max()) if valid.any() else 0.0
    data = np.where(valid, grid, SURFER_BLANK).astype('<f8')

    with open(path, 'wb') as f:
        f.write(b'DSRB')
        f.write(struct.pack('<ii', 4, 2))                     # boyut, versiyon
        f.write(b'GRID')
        f.write(struct.pack('<i', 72))
        f.write(struct.pack('<ii', ny, nx))
        f.write(struct.pack('<8d', x[0], y[0], dx, dy, zmin, zmax, 0.0, SURFER_BLANK))
        f.write(b'DATA')
        f.write(struct.pack('<i', data.nbytes))
        f.write(data.tobytes())
    return str(path)


def save_esri_ascii(grid, x, y, filepath, nodata: float = -9999.0) -> str:
    """ESRI ASCII Grid (.asc) olarak kaydeder (kare hücre gerektirir)."""
    grid, x, y, dx, dy = _check_regular(grid, x, y)
    if not np.isclose(dx, dy, rtol=1e-6):
        raise ValueError(
            f"ESRI ASCII Grid kare hücre gerektirir (dx={dx}, dy={dy}). "
            "GeoTIFF veya Surfer GRD kullanın."
        )
    path = _prepare_path(filepath)
    ny, nx = grid.shape
    out = np.where(np.isfinite(grid), grid, nodata)[::-1, :]   # ilk satır = kuzey

    with open(path, 'w', encoding='ascii') as f:
        f.write(f'ncols {nx}\n')
        f.write(f'nrows {ny}\n')
        f.write(f'xllcorner {x[0] - dx / 2:.10g}\n')
        f.write(f'yllcorner {y[0] - dy / 2:.10g}\n')
        f.write(f'cellsize {dx:.10g}\n')
        f.write(f'NODATA_value {nodata:.10g}\n')
        for row in out:
            f.write(' '.join(f'{v:.10g}' for v in row))
            f.write('\n')
    return str(path)


def save_geotiff(grid, x, y, filepath, crs=None, nodata: float = np.nan) -> str:
    """
    GeoTIFF olarak kaydeder (float32, kuzey-yukarı). rasterio gerektirir.
    crs: 'EPSG:32635' gibi bir dize veya None.
    """
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        raise ImportError("GeoTIFF yazmak için rasterio gereklidir: pip install rasterio")

    grid, x, y, dx, dy = _check_regular(grid, x, y)
    path = _prepare_path(filepath)
    ny, nx = grid.shape
    transform = from_origin(x[0] - dx / 2, y[-1] + dy / 2, dx, dy)
    data = grid[::-1, :].astype(np.float32)                  # satır 0 = kuzey
    if crs in (None, '', 'unknown', 'pixel'):
        crs = None

    with rasterio.open(
        path, 'w', driver='GTiff', height=ny, width=nx, count=1,
        dtype='float32', crs=crs, transform=transform, nodata=nodata,
        compress='deflate',
    ) as dst:
        dst.write(data, 1)
    return str(path)


_WRITERS = {
    'grd': save_surfer_ascii_grd,
    'surfer': save_surfer_ascii_grd,
    'surfer7': save_surfer7_binary_grd,
    'asc': save_esri_ascii,
    'tif': save_geotiff,
    'tiff': save_geotiff,
}


def save_grid(grid, x, y, filepath, fmt: str = None, **kwargs) -> str:
    """
    Formatı uzantıdan (veya `fmt`) seçerek grid kaydeder.
    fmt: 'grd'/'surfer' (Surfer ASCII), 'surfer7', 'asc', 'tif'.
    """
    key = (fmt or Path(filepath).suffix.lstrip('.')).lower()
    if key not in _WRITERS:
        raise ValueError(f"Bilinmeyen çıkış formatı: '{key}'. Seçenekler: {sorted(_WRITERS)}")
    writer = _WRITERS[key]
    if writer is save_geotiff:
        return writer(grid, x, y, filepath, **kwargs)
    return writer(grid, x, y, filepath)
