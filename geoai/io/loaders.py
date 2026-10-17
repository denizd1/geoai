"""
geoai.io.loaders
================
Jeofizik dosya formatlarını okuma:
  - Surfer GRD: ASCII (DSAA), Binary 6 (DSBB), Binary 7 (DSRB)
  - Geosoft / Oasis Montaj GRD (sıkıştırılmamış + zlib sıkıştırılmış)
  - ER Mapper ERS (.ers başlık + ham veri dosyası)
  - ESRI ASCII Grid (.asc)
  - GeoTIFF / TIFF (rasterio; yoksa Pillow + GeoTIFF etiketleri)
  - GDAL'ın okuyabildiği diğer rasterlar (rasterio kuruluysa; örn. GMT netCDF .grd)
  - XYZ / CSV / DAT (düzenli kafes → doğrudan grid; dağınık nokta → interpolasyon)
  - Kuyu / sondaj CSV

Tüm grid okuyucuları aynı sözlük yapısını döndürür:
    grid : (ny, nx) float64 — satır 0 = en güney satır, NaN = veri yok
    x, y : artan sıralı piksel-merkezi koordinat vektörleri
    xmin, xmax, ymin, ymax, nx, ny, format (+ formata özel ek alanlar)
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


# Surfer'ın boş (blank) değeri. Bu değere eşit veya büyük düğümler boştur.
SURFER_BLANK = 1.70141e38


# ─────────────────────────────────────────────────────────────────────────────
# ORTAK YARDIMCILAR
# ─────────────────────────────────────────────────────────────────────────────

def make_layer(grid, x, y, fmt: str = 'array', **extra) -> dict:
    """
    Grid + koordinat vektörlerinden standart katman sözlüğü üretir.

    - Eksenleri artan sıraya getirir (gerekirse grid'i çevirir).
    - Boyut uyumunu ve eksenlerin kesin monotonluğunu doğrular.
    - ±inf değerleri NaN yapar.
    """
    grid = np.array(grid, dtype=np.float64, copy=True)
    x = np.array(x, dtype=np.float64, copy=True).ravel()
    y = np.array(y, dtype=np.float64, copy=True).ravel()

    if grid.ndim != 2:
        raise ValueError(f"Grid 2 boyutlu olmalı, gelen boyut: {grid.ndim}")
    ny, nx = grid.shape
    if len(x) != nx or len(y) != ny:
        raise ValueError(
            f"Koordinat/grid boyut uyuşmazlığı: grid {ny}×{nx}, "
            f"len(x)={len(x)}, len(y)={len(y)}"
        )
    if nx < 2 or ny < 2:
        raise ValueError(f"Grid en az 2×2 olmalı, gelen: {ny}×{nx}")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("Koordinat vektörleri sonlu (finite) olmalı.")

    if x[0] > x[-1]:
        x = x[::-1]
        grid = grid[:, ::-1]
    if y[0] > y[-1]:
        y = y[::-1]
        grid = grid[::-1, :]
    if not (np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)):
        raise ValueError("Koordinat vektörleri kesin artan/azalan olmalı.")

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
    Döndürülmüş (rotated) bir grid'i eksen-paralel düzenli grid'e
    lineer interpolasyonla yeniden örnekler. Döndürülmüş dikdörtgenin
    dışında kalan köşe bölgeleri NaN olur.
    """
    valid = np.isfinite(grid)
    if valid.sum() < 3:
        raise ValueError("Döndürülmüş grid'de yeterli geçerli düğüm yok.")
    pts_e = east2d[valid]
    pts_n = north2d[valid]

    # Orijinal düğüm aralığı (komşu düğümler arası mesafe)
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
            f"Döndürülmüş grid düzenli grid'e çevrilemeyecek kadar büyük ({ny}×{nx})."
        )
    xi = xmin + np.arange(nx) * step
    yi = ymin + np.arange(ny) * step
    Xi, Yi = np.meshgrid(xi, yi)
    out = griddata((pts_e, pts_n), grid[valid], (Xi, Yi), method='linear')

    warnings.warn(
        f"Döndürülmüş grid ({extra.get('rotation', '?')}°) eksen-paralel "
        f"{ny}×{nx} grid'e lineer interpolasyonla yeniden örneklendi.",
        UserWarning,
    )
    return make_layer(out, xi, yi, fmt=fmt, **extra)


def _read_text_tokens(filepath: str) -> list:
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        return f.read().split()


# ─────────────────────────────────────────────────────────────────────────────
# FORMAT TESPİTİ
# ─────────────────────────────────────────────────────────────────────────────

GEOSOFT_ELEMENT_SIZES = (1, 2, 4, 8, 1025, 1026, 1028, 1032)

_EXT_MAP = {
    '.grd': 'auto_grd',      # Surfer / Geosoft / GMT-netCDF olabilir
    '.tif': 'geotiff',
    '.tiff': 'geotiff',
    '.ers': 'ermapper',
    '.asc': 'auto_text',     # ESRI ASCII Grid veya XYZ
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
    Dosya uzantısı ve içerik imzasından (magic bytes) format tespit eder.

    Döndürülen değerler: 'surfer_ascii', 'surfer_binary', 'surfer7_binary',
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
    """.grd dosyasının alt tipini başlık baytlarından belirler."""
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
# SURFER GRD OKUYUCULARI
# ─────────────────────────────────────────────────────────────────────────────

def _surfer_blank_to_nan(grid: np.ndarray, blank: float = SURFER_BLANK) -> np.ndarray:
    # float32'ye yuvarlanmış blank değerlerini de yakala
    return np.where(grid >= blank * (1 - 1e-6), np.nan, grid)


def read_surfer_ascii_grd(filepath: str) -> dict:
    """Surfer ASCII GRD (Golden Software DSAA) okur."""
    tokens = _read_text_tokens(filepath)
    if not tokens or tokens[0] != 'DSAA':
        raise ValueError(f"Geçersiz Surfer ASCII GRD başlığı: {filepath}")
    try:
        nx, ny = int(tokens[1]), int(tokens[2])
        xmin, xmax = float(tokens[3]), float(tokens[4])
        ymin, ymax = float(tokens[5]), float(tokens[6])
    except (IndexError, ValueError) as e:
        raise ValueError(f"Surfer ASCII GRD başlığı okunamadı: {e}")

    data = np.array(tokens[9:9 + nx * ny], dtype=np.float64)
    if data.size != nx * ny:
        raise ValueError(
            f"Veri boyutu uyuşmuyor: beklenen {nx*ny}, bulunan {data.size}"
        )
    # Satırlar Ymin'den Ymax'a (güneyden kuzeye) sıralıdır
    grid = _surfer_blank_to_nan(data.reshape((ny, nx)))
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    return make_layer(grid, x, y, fmt='surfer_ascii')


def read_surfer_binary_grd(filepath: str) -> dict:
    """Surfer 6 Binary GRD (DSBB) okur."""
    with open(filepath, 'rb') as f:
        raw = f.read()
    if raw[:4] != b'DSBB':
        raise ValueError("Geçersiz Surfer Binary GRD")
    nx, ny = struct.unpack_from('<hh', raw, 4)
    xmin, xmax, ymin, ymax, _zmin, _zmax = struct.unpack_from('<6d', raw, 8)
    n_values = nx * ny
    expected = 56 + n_values * 4
    if len(raw) < expected:
        raise ValueError(
            f"Surfer Binary GRD kısa: beklenen ≥{expected} bayt, bulunan {len(raw)}"
        )
    data = np.frombuffer(raw, dtype='<f4', count=n_values, offset=56)
    grid = _surfer_blank_to_nan(data.reshape((ny, nx)).astype(np.float64))
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    return make_layer(grid, x, y, fmt='surfer_binary')


def read_surfer7_binary_grd(filepath: str) -> dict:
    """
    Surfer 7+ Binary GRD (DSRB) okur.

    Yapı (tüm alanlar little-endian):
      'DSRB' + long size + long version
      'GRID' + long size + nRow, nCol (long) +
             xLL, yLL, xSize, ySize, zMin, zMax, Rotation, BlankValue (double)
      'DATA' + long size + nRow*nCol double (satırlar güneyden kuzeye)
      (opsiyonel 'FLTI' fay bölümü — atlanır)
    """
    with open(filepath, 'rb') as f:
        raw = f.read()
    if raw[:4] != b'DSRB':
        raise ValueError("Geçersiz Surfer 7 Binary GRD")

    header_size = struct.unpack_from('<i', raw, 4)[0]
    offset = 8 + header_size
    grid_info = None
    data = None

    while offset + 8 <= len(raw):
        tag = raw[offset:offset + 4]
        size = struct.unpack_from('<i', raw, offset + 4)[0]
        body = offset + 8
        if size < 0 or body + size > len(raw):
            raise ValueError(f"Surfer 7 bölüm boyutu geçersiz: {tag!r} {size}")
        if tag == b'GRID':
            n_row, n_col = struct.unpack_from('<ii', raw, body)
            (xll, yll, xsize, ysize, _zmin, _zmax,
             rotation, blank) = struct.unpack_from('<8d', raw, body + 8)
            grid_info = dict(n_row=n_row, n_col=n_col, xll=xll, yll=yll,
                             xsize=xsize, ysize=ysize, rotation=rotation,
                             blank=blank)
        elif tag == b'DATA':
            if grid_info is None:
                raise ValueError("Surfer 7: DATA bölümü GRID bölümünden önce geldi.")
            n = grid_info['n_row'] * grid_info['n_col']
            if size < n * 8:
                raise ValueError("Surfer 7: DATA bölümü beklenenden kısa.")
            data = np.frombuffer(raw, dtype='<f8', count=n, offset=body)
            break
        offset = body + size

    if grid_info is None or data is None:
        raise ValueError("Surfer 7: GRID/DATA bölümleri bulunamadı.")

    if grid_info['rotation'] != 0:
        warnings.warn(
            "Surfer 7 rotasyon alanı sıfır değil; Surfer bu alanı kullanmadığı "
            "için yok sayıldı.", UserWarning,
        )

    grid = data.reshape((grid_info['n_row'], grid_info['n_col'])).astype(np.float64)
    grid = _surfer_blank_to_nan(grid, blank=grid_info['blank'])
    x = grid_info['xll'] + np.arange(grid_info['n_col']) * grid_info['xsize']
    y = grid_info['yll'] + np.arange(grid_info['n_row']) * grid_info['ysize']
    return make_layer(grid, x, y, fmt='surfer7_binary')


# ─────────────────────────────────────────────────────────────────────────────
# GEOSOFT / OASIS MONTAJ GRD
# ─────────────────────────────────────────────────────────────────────────────

# Veri tipi başına Geosoft dummy (boş) değerleri
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
        raise NotImplementedError("Geosoft renk (colour) grid'leri desteklenmiyor.")
    if (size, sf) not in table:
        raise ValueError(f"Geçersiz Geosoft veri tipi: ES={es}, SF={sf}")
    return table[(size, sf)]


def _geosoft_decompress(raw: bytes) -> bytes:
    """
    Sıkıştırılmış Geosoft grid bloklarını açar.
    Başlıktan (512 bayt) sonra: imza (4), sıkıştırma tipi (4), blok sayısı (4),
    blok başına vektör (4), blok ofsetleri (int64 × n, dosya başına göre),
    blok boyutları (int32 × n). Her blok 16 baytlık bir ön başlıkla başlar,
    ardından zlib akışı gelir.
    """
    n_blocks = struct.unpack_from('<i', raw, 512 + 8)[0]
    if n_blocks <= 0 or n_blocks > 10_000_000:
        raise ValueError(f"Geosoft: geçersiz blok sayısı {n_blocks}")
    offsets = struct.unpack_from(f'<{n_blocks}q', raw, 512 + 16)
    sizes = struct.unpack_from(f'<{n_blocks}i', raw, 512 + 16 + 8 * n_blocks)
    chunks = []
    for off, size in zip(offsets, sizes):
        start, end = off + 16, off + size
        if end > len(raw) or start >= end:
            raise ValueError("Geosoft: sıkıştırılmış blok dosya dışına taşıyor.")
        try:
            chunks.append(zlib.decompress(raw[start:end]))
        except zlib.error as e:
            raise IOError(
                f"Geosoft sıkıştırılmış blok açılamadı ({e}). Bu sıkıştırma "
                "türü desteklenmiyor; grid'i Oasis Montaj'da sıkıştırmasız kaydedin."
            )
    return b''.join(chunks)


def read_geosoft_grd(filepath: str) -> dict:
    """
    Geosoft / Oasis Montaj GRD (format v2) okuyucu.

    Başlık (512 bayt, little-endian):
      0  ES  int32  eleman boyutu (1,2,4,8; >1024 → sıkıştırılmış)
      4  SF  int32  işaret: 0 işaretsiz tamsayı, 1 işaretli, 2 kayan nokta
      8  NE  int32  vektör başına eleman
      12 NV  int32  vektör sayısı
      16 KX  int32  1: vektörler satır, -1: vektörler sütun
      20 DE, 28 DV, 36 X0, 44 Y0, 52 ROT, 60 ZBASE, 68 ZMULT (double)
    Değer = saklanan / ZMULT + ZBASE. Dummy değerler NaN olur.
    """
    with open(filepath, 'rb') as f:
        raw = f.read()
    if len(raw) < 512:
        raise ValueError("Geosoft GRD başlığı eksik (512 bayttan kısa).")

    es, sf, ne, nv, kx = struct.unpack_from('<5i', raw, 0)
    de, dv, x0, y0, rot, zbase, zmult = struct.unpack_from('<7d', raw, 20)

    if es not in GEOSOFT_ELEMENT_SIZES:
        raise ValueError(f"Geosoft: geçersiz eleman boyutu ES={es}")
    if kx not in (1, -1):
        raise NotImplementedError(f"Geosoft: KX={kx} sıralaması desteklenmiyor.")
    if ne <= 0 or nv <= 0:
        raise ValueError(f"Geosoft: geçersiz boyutlar NE={ne}, NV={nv}")
    if zmult == 0:
        raise ValueError("Geosoft: ZMULT sıfır olamaz.")

    dtype = _geosoft_dtype(es, sf)
    n = ne * nv
    payload = _geosoft_decompress(raw) if es > 1024 else raw[512:]
    itemsize = np.dtype(dtype).itemsize
    if len(payload) < n * itemsize:
        raise ValueError(
            f"Geosoft: veri bölümü kısa (beklenen {n*itemsize}, bulunan {len(payload)})"
        )
    stored = np.frombuffer(payload, dtype='<' + dtype, count=n).astype(np.float64)

    if dtype in ('f4', 'f8'):
        stored[stored <= _GEOSOFT_FLOAT_DUMMY] = np.nan
    elif dtype in _GEOSOFT_DUMMIES:
        stored[stored == _GEOSOFT_DUMMIES[dtype]] = np.nan
    values = stored / zmult + zbase

    if kx == 1:
        grid = values.reshape((nv, ne))           # satır = vektör
        dx, dy = de, dv
    else:
        grid = values.reshape((ne, nv), order='F')  # sütun = vektör
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
    """ERS başlığını iç içe sözlüğe çevirir (anahtarlar küçük harf)."""
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
    """'DD:MM:SS.s' biçimini ondalık dereceye çevirir."""
    value = value.strip()
    sign = -1.0 if value.startswith('-') else 1.0
    parts = [abs(float(p)) for p in value.lstrip('+-').split(':')]
    deg = parts[0] + (parts[1] / 60 if len(parts) > 1 else 0) + \
        (parts[2] / 3600 if len(parts) > 2 else 0)
    return sign * deg


def read_ermapper_ers(filepath: str, band: int = 1) -> dict:
    """
    ER Mapper ERS raster okur (.ers metin başlığı + ham ikili veri).

    Veri dosyası: 'DataFile' anahtarı, yoksa uzantısız aynı isim.
    'HeaderOffset' varsa veri .ers dosyasının içindedir.
    Veri düzeni BIL (satır satır, her satırda bantlar ardışık).
    """
    with open(filepath, 'rb') as f:
        raw_header = f.read()
    text = raw_header.decode('latin-1')
    hdr = _parse_ers_header(text).get('datasetheader')
    if hdr is None:
        raise ValueError(f"Geçersiz ERS başlığı (DatasetHeader yok): {filepath}")
    ri = hdr.get('rasterinfo', {})

    try:
        n_lines = int(ri['nroflines'])
        n_cells = int(ri['nrofcellsperline'])
    except KeyError as e:
        raise ValueError(f"ERS başlığında eksik alan: {e}")
    n_bands = int(ri.get('nrofbands', 1))
    if not 1 <= band <= n_bands:
        raise ValueError(f"ERS: band={band} geçersiz (bant sayısı {n_bands})")
    cell_type = ri.get('celltype', 'IEEE4ByteReal').lower()
    if cell_type not in _ERS_CELLTYPES:
        raise ValueError(f"ERS: desteklenmeyen CellType '{cell_type}'")
    endian = '>' if hdr.get('byteorder', 'LSBFirst').lower() == 'msbfirst' else '<'
    dtype = np.dtype(endian + _ERS_CELLTYPES[cell_type])

    # Veri kaynağı
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
                f"ERS veri dosyası bulunamadı. Denenen: {[str(c) for c in candidates]}"
            )
        with open(data_path, 'rb') as f:
            data_bytes = f.read()

    n_needed = n_lines * n_bands * n_cells
    if len(data_bytes) < n_needed * dtype.itemsize:
        raise ValueError("ERS veri dosyası beklenenden kısa.")
    cube = np.frombuffer(data_bytes, dtype=dtype, count=n_needed)
    cube = cube.reshape((n_lines, n_bands, n_cells))
    grid = cube[:, band - 1, :].astype(np.float64)

    null = ri.get('nullcellvalue')
    if null is not None:
        grid[grid == float(null)] = np.nan

    # Koordinatlar
    cs = hdr.get('coordinatespace', {})
    rotation = cs.get('rotation')
    if rotation is not None and abs(_dms_to_degrees(rotation)) > 1e-9:
        raise NotImplementedError(
            "Döndürülmüş ERS raster'ları desteklenmiyor; rotasyonsuz dışa aktarın."
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
        warnings.warn("ERS: RegistrationCoord yok, piksel koordinatları kullanılıyor.",
                      UserWarning)
        e0, n0 = 0.0, float(n_lines) * dy
    cell_x = float(ri.get('registrationcellx', 0.0))
    cell_y = float(ri.get('registrationcelly', 0.0))
    left = e0 - cell_x * dx
    top = n0 + cell_y * dy

    x = left + (np.arange(n_cells) + 0.5) * dx
    y = top - (np.arange(n_lines) + 0.5) * dy     # satır 0 = kuzey → make_layer çevirir
    proj = cs.get('projection')
    return make_layer(grid, x, y, fmt='ermapper',
                      crs=proj if proj else 'unknown')


# ─────────────────────────────────────────────────────────────────────────────
# ESRI ASCII GRID
# ─────────────────────────────────────────────────────────────────────────────

def read_esri_ascii(filepath: str) -> dict:
    """ESRI ASCII Grid (.asc) okur. İlk veri satırı kuzeydir."""
    tokens = _read_text_tokens(filepath)
    header = {}
    i = 0
    while i + 1 < len(tokens) and re.match(r'^[A-Za-z_]+$', tokens[i]):
        header[tokens[i].lower()] = tokens[i + 1]
        i += 2
    try:
        ncols, nrows = int(header['ncols']), int(header['nrows'])
    except KeyError:
        raise ValueError(f"ESRI ASCII başlığında ncols/nrows yok: {filepath}")
    if 'cellsize' in header:
        dx = dy = float(header['cellsize'])
    elif 'dx' in header and 'dy' in header:
        dx, dy = float(header['dx']), float(header['dy'])
    else:
        raise ValueError("ESRI ASCII başlığında cellsize (veya dx/dy) yok.")

    if 'xllcenter' in header:
        x0 = float(header['xllcenter'])
    elif 'xllcorner' in header:
        x0 = float(header['xllcorner']) + dx / 2
    else:
        raise ValueError("ESRI ASCII başlığında xllcorner/xllcenter yok.")
    if 'yllcenter' in header:
        y0 = float(header['yllcenter'])
    elif 'yllcorner' in header:
        y0 = float(header['yllcorner']) + dy / 2
    else:
        raise ValueError("ESRI ASCII başlığında yllcorner/yllcenter yok.")

    data = np.array(tokens[i:i + ncols * nrows], dtype=np.float64)
    if data.size != ncols * nrows:
        raise ValueError(
            f"ESRI ASCII veri boyutu uyuşmuyor: beklenen {ncols*nrows}, bulunan {data.size}"
        )
    grid = data.reshape((nrows, ncols))[::-1, :]   # güneyden kuzeye
    if 'nodata_value' in header:
        grid = np.where(grid == float(header['nodata_value']), np.nan, grid)
    x = x0 + np.arange(ncols) * dx
    y = y0 + np.arange(nrows) * dy
    return make_layer(grid, x, y, fmt='esri_ascii')


# ─────────────────────────────────────────────────────────────────────────────
# GEOTIFF / GDAL RASTER OKUYUCULARI
# ─────────────────────────────────────────────────────────────────────────────

def read_raster_gdal(filepath: str, band: int = 1, fmt: str = 'gdal') -> dict:
    """GDAL'ın okuyabildiği herhangi bir raster'ı rasterio ile okur."""
    import rasterio

    with rasterio.open(filepath) as src:
        if not 1 <= band <= src.count:
            raise ValueError(f"band={band} geçersiz (bant sayısı {src.count})")
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
    """GeoTIFF okur. rasterio varsa kullanır, yoksa Pillow + GeoTIFF etiketleri."""
    try:
        import rasterio  # noqa: F401
    except ImportError:
        return _read_tiff_fallback(filepath)
    return read_raster_gdal(filepath, fmt='geotiff')


# GeoTIFF etiket numaraları
_TAG_PIXEL_SCALE = 33550
_TAG_TIEPOINT = 33922
_TAG_TRANSFORM = 34264
_TAG_GEOKEYS = 34735
_TAG_GDAL_NODATA = 42113
_GEOKEY_RASTER_TYPE = 1025   # 1 = PixelIsArea, 2 = PixelIsPoint


def _read_tiff_fallback(filepath: str) -> dict:
    """
    rasterio yokken TIFF okuma (Pillow ile). GeoTIFF etiketleri
    (ModelPixelScale + ModelTiepoint veya ModelTransformation) varsa
    coğrafi koordinatlar bunlardan hesaplanır.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "GeoTIFF okumak için rasterio veya Pillow gereklidir.\n"
            "  pip install rasterio\n"
            "  veya: pip install Pillow"
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
        "rasterio bulunamadı ve TIFF'te GeoTIFF koordinat etiketi yok; "
        "piksel koordinatları kullanılıyor. pip install rasterio önerilir.",
        UserWarning,
    )
    x = np.arange(nx, dtype=float)
    y = np.arange(ny, dtype=float)[::-1]     # satır 0 = üst = en büyük y
    return make_layer(grid, x, y, fmt='tiff_pil', crs='pixel', nodata=nodata)


# ─────────────────────────────────────────────────────────────────────────────
# KOLON TESPİTİ
# ─────────────────────────────────────────────────────────────────────────────

def _tokens(name: str) -> list:
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
    Kolon adını açık verilmemişse aday listesinden bulur.

    Arama sırası (her aşama tüm adaylar için denenir):
      1. Tam eşleşme            ('x' == 'x')
      2. Parça (token) eşleşmesi ('x' ∈ 'x_utm'.split('_'))
      3. Alt dize eşleşmesi      (≥3 harfli adaylar: 'res' ⊂ 'resistivity')
      4. Sayısal kolon yedeği    (fallback_index'inci sayısal kolon)
    `exclude` içindeki kolonlar (başka role atanmış olanlar) atlanır.
    """
    if explicit is not None:
        key = str(explicit).strip().lower()
        if key in df.columns:
            return key
        raise KeyError(f"Kolon bulunamadı: '{explicit}'. Mevcut: {list(df.columns)}")

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
                f"Kolon tespit edilemedi, sayısal fallback kullanılıyor: "
                f"'{chosen}' (sayısal kolonlar: {numeric_all[:5]})",
                UserWarning
            )
            return chosen

    raise KeyError(
        f"Uygun kolon bulunamadı. Mevcut kolonlar: {list(df.columns)}"
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
    """CSV/XYZ tablosunu okur; başlıksız (tamamen sayısal) dosyaları da tanır."""
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
        raise IOError(f"Tablo okunamadı ({filepath}): {e}")

    if headerless:
        default_names = ['x', 'y', 'z']
        df.columns = [default_names[i] if i < 3 else f'col{i}' for i in range(df.shape[1])]
    df = df.dropna(how='all')
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# XYZ / CSV OKUYUCU (düzenli kafes veya dağınık nokta → grid)
# ─────────────────────────────────────────────────────────────────────────────

def _unique_sorted(values: np.ndarray, tol: float) -> np.ndarray:
    v = np.sort(values)
    keep = np.concatenate([[True], np.diff(v) > tol])
    return v[keep]


def _detect_lattice(x: np.ndarray, y: np.ndarray, max_nodes: int = 16_000_000):
    """
    Noktalar düzenli bir kafes (grid düğümleri) üzerindeyse
    (x0, dx, nx, y0, dy, ny) döndürür; değilse None.
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
    # Kafes en az %20 dolu olmalı; aksi halde (örn. tamsayı koordinatlı seyrek
    # dağınık örnekler) interpolasyon daha uygundur.
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
    XYZ/CSV noktalarını okur ve grid'e dönüştürür.

    Parametreler
    ------------
    filepath       : Dosya yolu
    x_col, y_col, z_col : Kolon isimleri (None ise otomatik tespit)
    delimiter      : Ayırıcı (None ise otomatik)
    target_nx/ny   : Dağınık veri için çıkış grid boyutu
    interp_method  : 'linear', 'nearest', 'cubic'
    smooth_sigma   : Interpolasyon sonrası Gaussian yumuşatma (piksel; 0 = kapalı)
    grid_method    : 'auto'  → noktalar düzenli kafes üzerindeyse doğrudan grid,
                                değilse interpolasyon
                     'lattice' / 'interpolate' → zorla
    max_distance   : Dağınık veride, en yakın ölçüm noktasına bu mesafeden
                     (koordinat birimi) uzak düğümler NaN yapılır (None = kapalı)
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
        raise ValueError(f"Yeterli veri noktası yok: {len(x)} nokta")

    if grid_method not in ('auto', 'lattice', 'interpolate'):
        raise ValueError(f"Geçersiz grid_method: {grid_method}")

    lattice = None if grid_method == 'interpolate' else _detect_lattice(x, y)
    if grid_method == 'lattice' and lattice is None:
        raise ValueError("Noktalar düzenli bir kafes oluşturmuyor (grid_method='lattice').")

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

    # Aynı koordinattaki tekrar eden noktaların ortalamasını al
    pts = pd.DataFrame({'x': x, 'y': y, 'z': z}).groupby(['x', 'y'], as_index=False)['z'].mean()
    x, y, z = pts['x'].to_numpy(), pts['y'].to_numpy(), pts['z'].to_numpy()

    xi = np.linspace(x.min(), x.max(), int(target_nx))
    yi = np.linspace(y.min(), y.max(), int(target_ny))
    Xi, Yi = np.meshgrid(xi, yi)

    try:
        grid = griddata((x, y), z, (Xi, Yi), method=interp_method)
    except Exception as e:   # örn. tüm noktalar tek doğru üzerinde (QhullError)
        warnings.warn(
            f"'{interp_method}' interpolasyonu başarısız ({type(e).__name__}); "
            "'nearest' kullanılıyor.", UserWarning,
        )
        grid = griddata((x, y), z, (Xi, Yi), method='nearest')

    # Konveks örtü dışını nearest ile doldur
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
# EVRENSEL YÜKLEYİCİ
# ─────────────────────────────────────────────────────────────────────────────

_XYZ_KWARGS = {'x_col', 'y_col', 'z_col', 'delimiter', 'target_nx', 'target_ny',
               'interp_method', 'smooth_sigma', 'grid_method', 'max_distance'}


def load_grid(filepath: str, **kwargs) -> dict:
    """
    Tek fonksiyonla tüm formatları otomatik okur.

    Parametreler
    ------------
    filepath : dosya yolu
    **kwargs : XYZ/CSV için read_xyz parametreleri; raster formatları için
               'band' (çok bantlı ERS/GDAL dosyaları).

    Döndürür
    --------
    dict: grid (2D array, satır 0 = güney), x, y vektörleri ve metadata
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dosya bulunamadı: {filepath}")

    fmt = detect_format(filepath)
    band = kwargs.pop('band', 1)

    if fmt in ('xyz', 'csv'):
        unknown = set(kwargs) - _XYZ_KWARGS
        if unknown:
            raise TypeError(f"read_xyz için bilinmeyen parametre(ler): {sorted(unknown)}")
        result = read_xyz(filepath, **kwargs)
    else:
        if kwargs:
            warnings.warn(
                f"'{fmt}' formatı için XYZ parametreleri yok sayıldı: {sorted(kwargs)}",
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
                f"Desteklenmeyen format: '{fmt}' ({filepath})\n"
                f"Desteklenen: {sorted(list(loaders) + ['xyz', 'csv'])}"
            )
        result = loaders[fmt](filepath)

    result['filepath'] = str(filepath)
    result['name'] = Path(filepath).stem

    n_nan = int(np.isnan(result['grid']).sum())
    pct = 100 * n_nan / result['grid'].size
    result['nan_percent'] = float(pct)
    if pct >= 100:
        raise ValueError(f"{filepath}: grid tamamen boş (NaN).")
    if pct > 50:
        warnings.warn(
            f"{filepath}: %{pct:.1f} boş veri (NaN). "
            "Grid interpolasyonu veya farklı format deneyin.", UserWarning
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# KUYU / SONDAJ YÜKLEYİCİ
# ─────────────────────────────────────────────────────────────────────────────

_POSITIVE_STRINGS = {'1', '1.0', 'true', 't', 'yes', 'y', 'pos', 'positive',
                     'evet', 'var', 'pozitif', '+'}
_NEGATIVE_STRINGS = {'0', '0.0', 'false', 'f', 'no', 'n', 'neg', 'negative',
                     'hayir', 'hayır', 'yok', 'negatif', '-'}


def _parse_labels(series: pd.Series) -> pd.Series:
    """Etiketleri sayıya çevirir; 'yes/no', 'true/false', 'evet/hayır' tanınır."""
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
            return np.nan if s == '' else float('inf')   # geçersiz → hata üretir
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
    Kuyu / sondaj verisi yükler.

    Beklenen format (CSV):
      X, Y, LABEL, TARGET_TYPE (opsiyonel), DEPTH, NOTES, WELL_ID vb.

    LABEL: 1 = pozitif (maden/su/jeotermal bulundu), 0 = negatif.
           'yes/no', 'true/false', 'evet/hayır' de kabul edilir.
    target_type: verilirse ve dosyada TARGET_TYPE kolonu varsa, sadece bu
                 tipteki (veya tipi boş olan) kuyular döndürülür.

    Döndürür
    --------
    pd.DataFrame: standart kolonlar x, y, label (+ target_type, depth, notes, well_id)
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
            "LABEL kolonu sadece 0 veya 1 (ya da yes/no, true/false) içermelidir. "
            f"Geçersiz örnekler: {preview}"
        )
    out['label'] = out['label'].astype(int)

    n_dropped = n_raw - len(out)
    if n_dropped > 0:
        warnings.warn(
            f"{filepath}: {n_dropped} satır eksik/geçersiz X, Y veya LABEL "
            "nedeniyle atlandı.", UserWarning,
        )

    if target_type is not None and 'target_type' in out.columns:
        wanted = str(target_type).strip().lower()
        types = out['target_type']
        keep = types.isna() | (types == '') | (types == wanted)
        n_other = int((~keep).sum())
        out = out[keep]
        if n_other:
            print(f"  ℹ {n_other} kuyu farklı TARGET_TYPE nedeniyle "
                  f"'{wanted}' için filtrelendi.")

    out = out.reset_index(drop=True)
    n_pos = int((out['label'] == 1).sum())
    n_neg = int((out['label'] == 0).sum())
    print(f"  ✓ Kuyu verisi yüklendi: {len(out)} kuyu | "
          f"Pozitif: {n_pos} | Negatif: {n_neg}")

    return out
