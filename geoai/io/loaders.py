"""
geoai.io.loaders
================
Jeofizik dosya formatlarını okuma:
  - Geosoft GRD (binary ve ASCII)
  - GeoTIFF / TIFF
  - Surfer GRD (ASCII Golden Software)
  - ERMapper ERS
  - XYZ / CSV (scattered points → grid)
  - Well/drillhole CSV
"""

import os
import struct
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import warnings


# ─────────────────────────────────────────────────────────────────────────────
# FORMAT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_format(filepath: str) -> str:
    """Dosya uzantısı ve magic byte'lardan format tespit eder."""
    path = Path(filepath)
    ext = path.suffix.lower()

    ext_map = {
        '.grd': 'auto_grd',    # Geosoft veya Surfer olabilir
        '.tif': 'geotiff',
        '.tiff': 'geotiff',
        '.ers': 'ermapper',
        '.xyz': 'xyz',
        '.csv': 'csv',
        '.txt': 'xyz',
        '.dat': 'xyz',
    }

    fmt = ext_map.get(ext, 'unknown')

    # GRD için magic byte kontrolü
    if fmt == 'auto_grd':
        fmt = _detect_grd_type(filepath)

    return fmt


def _detect_grd_type(filepath: str) -> str:
    """GRD dosyasının Geosoft mu Surfer mı olduğunu belirler."""
    with open(filepath, 'rb') as f:
        header = f.read(4)

    # Surfer 6 ASCII: "DSAA"
    if header[:4] == b'DSAA':
        return 'surfer_ascii'
    # Surfer 6 Binary: "DSBB"
    if header[:4] == b'DSBB':
        return 'surfer_binary'
    # Surfer 7 Binary: "DSRB"
    if header[:4] == b'DSRB':
        return 'surfer7_binary'
    # Geosoft GRD: binary, ilk 4 byte farklı
    return 'geosoft_grd'


# ─────────────────────────────────────────────────────────────────────────────
# GRD READERS
# ─────────────────────────────────────────────────────────────────────────────

def read_surfer_ascii_grd(filepath: str) -> dict:
    """Surfer ASCII GRD (Golden Software format DSAA) okur."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    if lines[0].strip() != 'DSAA':
        raise ValueError(f"Geçersiz Surfer ASCII GRD başlığı: {filepath}")

    nx, ny = map(int, lines[1].split())
    xmin, xmax = map(float, lines[2].split())
    ymin, ymax = map(float, lines[3].split())
    zmin, zmax = map(float, lines[4].split())

    # Veriyi oku
    data_str = ' '.join(lines[5:])
    data = np.array(list(map(float, data_str.split())))

    if len(data) != nx * ny:
        raise ValueError(
            f"Veri boyutu uyuşmuyor: beklenen {nx*ny}, bulunan {len(data)}"
        )

    grid = data.reshape((ny, nx))

    # Surfer'ın özel boş veri değeri
    nodata_val = 1.70141e+38
    grid = np.where(np.abs(grid) > nodata_val * 0.9, np.nan, grid)

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    return {
        'grid': grid,
        'x': x,
        'y': y,
        'xmin': xmin, 'xmax': xmax,
        'ymin': ymin, 'ymax': ymax,
        'nx': nx, 'ny': ny,
        'format': 'surfer_ascii',
    }


def read_surfer_binary_grd(filepath: str) -> dict:
    """Surfer Binary GRD (DSBB) okur."""
    with open(filepath, 'rb') as f:
        tag = f.read(4)
        if tag != b'DSBB':
            raise ValueError("Geçersiz Surfer Binary GRD")

        nx = struct.unpack('<H', f.read(2))[0]
        ny = struct.unpack('<H', f.read(2))[0]
        xmin = struct.unpack('<d', f.read(8))[0]
        xmax = struct.unpack('<d', f.read(8))[0]
        ymin = struct.unpack('<d', f.read(8))[0]
        ymax = struct.unpack('<d', f.read(8))[0]
        zmin = struct.unpack('<d', f.read(8))[0]
        zmax = struct.unpack('<d', f.read(8))[0]

        n_values = nx * ny
        data = np.frombuffer(f.read(n_values * 4), dtype=np.float32)

    grid = data.reshape((ny, nx)).astype(np.float64)
    nodata_val = 1.70141e+38
    grid = np.where(np.abs(grid) > nodata_val * 0.9, np.nan, grid)

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    return {
        'grid': grid,
        'x': x, 'y': y,
        'xmin': xmin, 'xmax': xmax,
        'ymin': ymin, 'ymax': ymax,
        'nx': nx, 'ny': ny,
        'format': 'surfer_binary',
    }


def read_surfer7_binary_grd(filepath: str) -> dict:
    """Surfer 7+ Binary GRD (DSRB) okur."""
    with open(filepath, 'rb') as f:
        tag = f.read(4)
        if tag != b'DSRB':
            raise ValueError("Geçersiz Surfer 7 Binary GRD")

        # Section header: Id + size
        section_id = struct.unpack('<I', f.read(4))[0]
        section_size = struct.unpack('<I', f.read(4))[0]

        # Grid header section (id=1)
        version = struct.unpack('<I', f.read(4))[0]
        nx = struct.unpack('<I', f.read(4))[0]
        ny = struct.unpack('<I', f.read(4))[0]
        xlo = struct.unpack('<d', f.read(8))[0]
        ylo = struct.unpack('<d', f.read(8))[0]
        xhi = struct.unpack('<d', f.read(8))[0]
        yhi = struct.unpack('<d', f.read(8))[0]
        zlo = struct.unpack('<d', f.read(8))[0]
        zhi = struct.unpack('<d', f.read(8))[0]
        rotation = struct.unpack('<d', f.read(8))[0]
        blank_val = struct.unpack('<d', f.read(8))[0]

        # Data section
        _ = f.read(4)  # section id
        _ = f.read(4)  # section size
        data = np.frombuffer(f.read(nx * ny * 8), dtype=np.float64)

    grid = data.reshape((ny, nx))
    grid = np.where(grid >= blank_val * 0.99, np.nan, grid)

    x = np.linspace(xlo, xhi, nx)
    y = np.linspace(ylo, yhi, ny)

    return {
        'grid': grid,
        'x': x, 'y': y,
        'xmin': xlo, 'xmax': xhi,
        'ymin': ylo, 'ymax': yhi,
        'nx': nx, 'ny': ny,
        'format': 'surfer7_binary',
    }


def read_geosoft_grd(filepath: str) -> dict:
    """
    Geosoft GRD binary format okuyucu.
    Geosoft formatı kapalı kaynaklıdır; bu implementasyon
    standart Geosoft binary yapısını temel alır.
    Tam destek için: pip install geosoft (Geosoft SDK)
    """
    try:
        # Geosoft Python SDK varsa kullan
        import geosoft.gxpy.grid as gxgrd
        grd = gxgrd.Grid.open(filepath)
        props = grd.properties()
        grid = grd.read_rows()
        x0, y0 = props['x0'], props['y0']
        dx, dy = props['dx'], props['dy']
        ny, nx = grid.shape
        x = np.arange(nx) * dx + x0
        y = np.arange(ny) * dy + y0
        return {
            'grid': grid, 'x': x, 'y': y,
            'xmin': x.min(), 'xmax': x.max(),
            'ymin': y.min(), 'ymax': y.max(),
            'nx': nx, 'ny': ny,
            'format': 'geosoft_grd',
        }
    except ImportError:
        pass

    # Geosoft SDK yoksa manuel binary okuma (basitleştirilmiş)
    warnings.warn(
        f"Geosoft SDK bulunamadı. {filepath} dosyasını Surfer GRD formatında "
        "kaydetmenizi öneririz. Şimdilik ham binary okuma deneniyor.",
        UserWarning
    )
    return _read_geosoft_binary_fallback(filepath)


def _read_geosoft_binary_fallback(filepath: str) -> dict:
    """Geosoft binary için fallback okuyucu (yaklaşık)."""
    with open(filepath, 'rb') as f:
        raw = f.read()

    # Geosoft header: 512 byte
    # nx, ny float32 olarak offset 4-12'de
    try:
        nx = struct.unpack_from('<i', raw, 4)[0]
        ny = struct.unpack_from('<i', raw, 8)[0]
        x0 = struct.unpack_from('<d', raw, 16)[0]
        y0 = struct.unpack_from('<d', raw, 24)[0]
        dx = struct.unpack_from('<d', raw, 32)[0]
        dy = struct.unpack_from('<d', raw, 40)[0]

        if nx <= 0 or ny <= 0 or nx > 100000 or ny > 100000:
            raise ValueError("Geçersiz boyutlar")

        data_start = 512
        data = np.frombuffer(raw[data_start:data_start + nx * ny * 4], dtype=np.float32)
        grid = data.reshape((ny, nx)).astype(np.float64)

        # Geosoft dummy değeri
        dummy = -1e32
        grid = np.where(grid < dummy * 0.5, np.nan, grid)

        x = np.arange(nx) * dx + x0
        y = np.arange(ny) * dy + y0

        return {
            'grid': grid, 'x': x, 'y': y,
            'xmin': x.min(), 'xmax': x.max(),
            'ymin': y.min(), 'ymax': y.max(),
            'nx': nx, 'ny': ny,
            'format': 'geosoft_grd_fallback',
        }
    except Exception as e:
        raise IOError(
            f"Geosoft GRD okunamadı: {e}\n"
            "Lütfen dosyayı Surfer ASCII (.grd DSAA) formatına dönüştürün."
        )


# ─────────────────────────────────────────────────────────────────────────────
# GEOTIFF READER
# ─────────────────────────────────────────────────────────────────────────────

def read_geotiff(filepath: str) -> dict:
    """GeoTIFF okur. rasterio varsa kullanır, yoksa struct ile dener."""
    try:
        import rasterio
        with rasterio.open(filepath) as src:
            grid = src.read(1).astype(np.float64)
            nodata = src.nodata
            if nodata is not None:
                grid = np.where(grid == nodata, np.nan, grid)

            transform = src.transform
            ny, nx = grid.shape
            # Koordinat vektörleri (piksel merkezleri)
            x = np.array([
                rasterio.transform.xy(transform, 0, j, offset='center')[0]
                for j in range(nx)
            ], dtype=np.float64)
            y = np.array([
                rasterio.transform.xy(transform, i, 0, offset='center')[1]
                for i in range(ny)
            ], dtype=np.float64)

            # Eksenleri artan sıraya getir (searchsorted/indexleme için)
            if nx > 1 and x[0] > x[-1]:
                x = x[::-1]
                grid = grid[:, ::-1]
            if ny > 1 and y[0] > y[-1]:
                y = y[::-1]
                grid = grid[::-1, :]

            crs_str = str(src.crs) if src.crs else 'unknown'

        return {
            'grid': grid,
            'x': x, 'y': y,
            'xmin': x.min(), 'xmax': x.max(),
            'ymin': y.min(), 'ymax': y.max(),
            'nx': nx, 'ny': ny,
            'crs': crs_str,
            'format': 'geotiff',
        }
    except ImportError:
        return _read_tiff_fallback(filepath)


def _read_tiff_fallback(filepath: str) -> dict:
    """rasterio yokken TIFF okuma (PIL/pillow ile)."""
    try:
        from PIL import Image
        img = Image.open(filepath)
        grid = np.array(img, dtype=np.float64)
        if grid.ndim == 3:
            grid = grid[:, :, 0]
        ny, nx = grid.shape
        x = np.arange(nx, dtype=float)
        y = np.arange(ny, dtype=float)
        warnings.warn(
            "rasterio bulunamadı, PIL ile TIFF okundu. "
            "Coğrafi koordinat bilgisi eksik olabilir. "
            "pip install rasterio önerilir.", UserWarning
        )
        return {
            'grid': grid, 'x': x, 'y': y,
            'xmin': 0, 'xmax': nx - 1,
            'ymin': 0, 'ymax': ny - 1,
            'nx': nx, 'ny': ny,
            'crs': 'pixel',
            'format': 'tiff_pil',
        }
    except ImportError:
        raise ImportError(
            "GeoTIFF okumak için rasterio veya Pillow gereklidir.\n"
            "  pip install rasterio\n"
            "  veya: pip install Pillow"
        )


# ─────────────────────────────────────────────────────────────────────────────
# XYZ / CSV READER (scattered → grid)
# ─────────────────────────────────────────────────────────────────────────────

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
) -> dict:
    """
    Dağınık XYZ noktalarını okur ve grid'e interpolasyon yapar.

    Parametreler
    ------------
    filepath       : Dosya yolu
    x_col, y_col, z_col : Kolon isimleri (None ise otomatik tespit)
    delimiter      : Ayırıcı (None ise otomatik)
    target_nx/ny   : Çıkış grid boyutu
    interp_method  : 'linear', 'nearest', 'cubic'
    smooth_sigma   : Grid üzerinde Gaussian yumuşatma
    """
    # Otomatik ayırıcı tespiti
    if delimiter is None:
        with open(filepath, 'r') as f:
            sample = f.read(1024)
        if '\t' in sample:
            delimiter = '\t'
        elif ',' in sample:
            delimiter = ','
        else:
            delimiter = r'\s+'

    try:
        df = pd.read_csv(filepath, sep=delimiter, engine='python',
                         comment='#', skip_blank_lines=True)
        df = df.dropna(how='all')
        df.columns = [str(c).strip().lower() for c in df.columns]
    except Exception as e:
        raise IOError(f"XYZ/CSV okunamadı: {e}")

    # Kolon tespiti
    x_col = _find_column(
        df, x_col, ['x', 'easting', 'lon', 'longitude', 'east', 'e'],
        fallback_index=0, allow_numeric_fallback=True
    )
    y_col = _find_column(
        df, y_col, ['y', 'northing', 'lat', 'latitude', 'north', 'n'],
        fallback_index=1, allow_numeric_fallback=True
    )
    z_col = _find_column(
        df, z_col, ['z', 'value', 'val', 'data', 'field', 'mag',
                    'grav', 'res', 'ip', 'depth', 'signal'],
        fallback_index=2, allow_numeric_fallback=True
    )

    x = pd.to_numeric(df[x_col], errors='coerce').values
    y = pd.to_numeric(df[y_col], errors='coerce').values
    z = pd.to_numeric(df[z_col], errors='coerce').values

    # NaN temizle
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[mask], y[mask], z[mask]

    if len(x) < 10:
        raise ValueError(f"Yeterli veri noktası yok: {len(x)} nokta")

    # Grid oluştur
    xi = np.linspace(x.min(), x.max(), target_nx)
    yi = np.linspace(y.min(), y.max(), target_ny)
    Xi, Yi = np.meshgrid(xi, yi)

    grid = griddata((x, y), z, (Xi, Yi), method=interp_method)

    # Kenar boşluklarını nearest ile doldur
    nan_mask = np.isnan(grid)
    if nan_mask.any():
        grid_nearest = griddata((x, y), z, (Xi, Yi), method='nearest')
        grid[nan_mask] = grid_nearest[nan_mask]

    # Hafif yumuşatma
    if smooth_sigma > 0:
        grid = gaussian_filter(grid, sigma=smooth_sigma)

    return {
        'grid': grid,
        'x': xi, 'y': yi,
        'xmin': xi.min(), 'xmax': xi.max(),
        'ymin': yi.min(), 'ymax': yi.max(),
        'nx': target_nx, 'ny': target_ny,
        'n_points': len(x),
        'format': 'xyz_interpolated',
    }


def _find_column(
    df: pd.DataFrame,
    explicit: str,
    candidates: list,
    fallback_index: int = 0,
    allow_numeric_fallback: bool = True,
) -> str:
    """Kolon adını açık verilmemişse aday listesinden bulur."""
    if explicit is not None:
        key = explicit.lower().strip()
        if key in df.columns:
            return key
        raise KeyError(f"Kolon bulunamadı: '{explicit}'. Mevcut: {list(df.columns)}")

    for cand in candidates:
        if cand in df.columns:
            return cand
        # Kısmi eşleşme
        for col in df.columns:
            if cand in col:
                return col

    if allow_numeric_fallback:
        # Son çare: sayısal kolon fallback (x->0, y->1, z->2)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            idx = min(max(0, int(fallback_index)), len(numeric_cols) - 1)
            warnings.warn(
                f"Kolon tespit edilemedi, sayısal fallback kullanılıyor: "
                f"'{numeric_cols[idx]}' (sayısal kolonlar: {numeric_cols[:5]})",
                UserWarning
            )
            return numeric_cols[idx]

    raise KeyError(
        f"Uygun kolon bulunamadı. Mevcut kolonlar: {list(df.columns)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSAL LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_grid(filepath: str, **kwargs) -> dict:
    """
    Tek fonksiyonla tüm formatları otomatik okur.

    Döndürür
    --------
    dict ile: grid (2D array), x, y vektörleri, metadata
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dosya bulunamadı: {filepath}")

    fmt = detect_format(filepath)

    loaders = {
        'surfer_ascii': read_surfer_ascii_grd,
        'surfer_binary': read_surfer_binary_grd,
        'surfer7_binary': read_surfer7_binary_grd,
        'geosoft_grd': read_geosoft_grd,
        'geotiff': read_geotiff,
        'xyz': lambda f: read_xyz(f, **kwargs),
        'csv': lambda f: read_xyz(f, **kwargs),
    }

    if fmt not in loaders:
        raise ValueError(
            f"Desteklenmeyen format: '{fmt}' ({filepath})\n"
            f"Desteklenen: {list(loaders.keys())}"
        )

    result = loaders[fmt](filepath)
    result['filepath'] = filepath
    result['name'] = Path(filepath).stem

    # NaN kontrolü ve raporu
    n_nan = np.isnan(result['grid']).sum()
    n_total = result['grid'].size
    if n_nan > 0:
        pct = 100 * n_nan / n_total
        if pct > 50:
            warnings.warn(
                f"{filepath}: %{pct:.1f} boş veri (NaN). "
                "Grid interpolasyonu veya farklı format deneyin.", UserWarning
            )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# WELL / DRILLHOLE LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_well_data(
    filepath: str,
    x_col: str = None,
    y_col: str = None,
    label_col: str = None,
    target_type_col: str = None,
    delimiter: str = None,
) -> pd.DataFrame:
    """
    Kuyu / sondaj verisi yükler.

    Beklenen format (CSV):
      X, Y, LABEL, TARGET_TYPE (opsiyonel), DEPTH, NOT vb.

    LABEL: 1 = pozitif (maden/su/jeotermal bulundu), 0 = negatif

    Döndürür
    --------
    pd.DataFrame: standartlaşmış kolon isimleriyle
    """
    if delimiter is None:
        with open(filepath, 'r') as f:
            sample = f.read(512)
        delimiter = '\t' if '\t' in sample else ',' if ',' in sample else r'\s+'

    df = pd.read_csv(filepath, sep=delimiter, engine='python',
                     comment='#', skip_blank_lines=True)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Kolon tespiti
    x_col = _find_column(
        df, x_col, ['x', 'easting', 'lon', 'east', 'longitude'],
        fallback_index=0, allow_numeric_fallback=True
    )
    y_col = _find_column(
        df, y_col, ['y', 'northing', 'lat', 'north', 'latitude'],
        fallback_index=1, allow_numeric_fallback=True
    )
    label_col = _find_column(df, label_col,
                              ['label', 'result', 'target', 'positive', 'hit',
                               'found', 'mineralized', 'success', 'class'],
                              allow_numeric_fallback=False)

    out = pd.DataFrame({
        'x': pd.to_numeric(df[x_col], errors='coerce'),
        'y': pd.to_numeric(df[y_col], errors='coerce'),
        'label': pd.to_numeric(df[label_col], errors='coerce'),
    })

    # Opsiyonel kolonlar
    for src_col, dst_col, candidates in [
        (target_type_col, 'target_type',
         ['target_type', 'type', 'category', 'resource_type', 'mineral']),
        (None, 'depth',
         ['depth', 'derinlik', 'depth_m', 'total_depth']),
        (None, 'notes',
         ['notes', 'note', 'comment', 'aciklama']),
    ]:
        try:
            found = _find_column(df, src_col, candidates, allow_numeric_fallback=False)
            out[dst_col] = df[found]
        except (KeyError, Exception):
            pass

    out = out.dropna(subset=['x', 'y', 'label'])
    invalid_labels = out.loc[~out['label'].isin([0, 1]), 'label']
    if len(invalid_labels) > 0:
        preview = ", ".join(invalid_labels.astype(str).head(5).tolist())
        raise ValueError(
            "LABEL kolonu sadece 0 veya 1 içermelidir. "
            f"Geçersiz örnekler: {preview}"
        )
    out['label'] = out['label'].astype(int)

    n_pos = (out['label'] == 1).sum()
    n_neg = (out['label'] == 0).sum()
    print(f"  ✓ Kuyu verisi yüklendi: {len(out)} kuyu | "
          f"Pozitif: {n_pos} | Negatif: {n_neg}")

    return out.reset_index(drop=True)
