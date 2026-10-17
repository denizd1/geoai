"""Grid reader/writer tests (cross-validated against GDAL and real Oasis Montaj files)."""
import os
import struct
import warnings

import numpy as np
import pytest

from conftest import DATA_DIR, HAS_RASTERIO, requires_rasterio, write_geosoft
from geoai.io import loaders as L
from geoai.io import writers as W

GEOSOFT_DIR = os.path.join(DATA_DIR, 'geosoft')


def _grid_xy(rng, ny=7, nx=9, with_nan=True):
    g = rng.normal(size=(ny, nx)) * 50 + 1000
    if with_nan:
        g[2, 3] = np.nan
        g[ny - 1, nx - 1] = np.nan
    x = 500000 + 25.0 * np.arange(nx)
    y = 4100000 + 40.0 * np.arange(ny)
    return g, x, y


def _read_gdal(path):
    import rasterio
    with rasterio.open(path) as s:
        arr = s.read(1, masked=True).astype(float).filled(np.nan)[::-1]
        t = s.transform
        x = t.c + (np.arange(s.width) + 0.5) * t.a
        y = (t.f + (np.arange(s.height) + 0.5) * t.e)[::-1]
    return arr, x, y


# ─── make_layer ─────────────────────────────────────────────────────────────

def test_make_layer_flips_descending_axes():
    g = np.arange(6, dtype=float).reshape(2, 3)
    lyr = L.make_layer(g, [3, 2, 1], [20, 10])
    assert lyr['x'].tolist() == [1, 2, 3]
    assert lyr['y'].tolist() == [10, 20]
    np.testing.assert_array_equal(lyr['grid'], g[::-1, ::-1])


def test_make_layer_validates_shape_and_monotonicity():
    with pytest.raises(ValueError):
        L.make_layer(np.zeros((2, 3)), [0, 1], [0, 1])
    with pytest.raises(ValueError):
        L.make_layer(np.zeros((2, 3)), [0, 2, 1], [0, 1])
    with pytest.raises(ValueError):
        L.make_layer(np.zeros((1, 3)), [0, 1, 2], [0])


def test_make_layer_converts_inf_to_nan():
    lyr = L.make_layer(np.array([[1.0, np.inf], [2.0, -np.inf]]), [0, 1], [0, 1])
    assert np.isnan(lyr['grid']).sum() == 2


# ─── Surfer ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('writer', [W.save_surfer_ascii_grd, W.save_surfer7_binary_grd])
def test_surfer_roundtrip_preserves_values_coords_and_blanks(tmp_path, rng, writer):
    g, x, y = _grid_xy(rng)
    path = writer(g, x, y, tmp_path / 'g.grd')
    lyr = L.load_grid(path)
    np.testing.assert_allclose(lyr['grid'], g, rtol=1e-9, equal_nan=True)
    np.testing.assert_allclose(lyr['x'], x)
    np.testing.assert_allclose(lyr['y'], y)


def test_surfer6_binary_reader_handwritten(tmp_path):
    ny, nx = 3, 4
    data = np.arange(12, dtype=np.float32).reshape(ny, nx)
    data[1, 2] = 1.70141e38
    path = tmp_path / 's6.grd'
    with open(path, 'wb') as f:
        f.write(b'DSBB' + struct.pack('<hh', nx, ny))
        f.write(struct.pack('<6d', 10, 40, 100, 120, 0, 11))
        f.write(data.tobytes())
    lyr = L.load_grid(str(path))
    assert lyr['format'] == 'surfer_binary'
    assert np.isnan(lyr['grid'][1, 2])
    assert lyr['grid'][0, 0] == 0 and lyr['grid'][2, 3] == 11
    np.testing.assert_allclose(lyr['x'], [10, 20, 30, 40])
    np.testing.assert_allclose(lyr['y'], [100, 110, 120])


def test_surfer7_reader_skips_unknown_sections(tmp_path):
    """Must read correctly even when extra sections such as FLTI precede DATA."""
    ny, nx = 2, 3
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype='<f8')
    path = tmp_path / 's7.grd'
    with open(path, 'wb') as f:
        f.write(b'DSRB' + struct.pack('<ii', 4, 2))
        f.write(b'GRID' + struct.pack('<i', 72) + struct.pack('<ii', ny, nx))
        f.write(struct.pack('<8d', 0, 0, 5, 7, 1, 6, 0, 1.70141e38))
        f.write(b'FLTI' + struct.pack('<i', 8) + b'\x00' * 8)
        f.write(b'DATA' + struct.pack('<i', data.nbytes) + data.tobytes())
    lyr = L.load_grid(str(path))
    np.testing.assert_array_equal(lyr['grid'], data)
    np.testing.assert_allclose(lyr['x'], [0, 5, 10])
    np.testing.assert_allclose(lyr['y'], [0, 7])


@requires_rasterio
@pytest.mark.parametrize('driver,ext,dtype', [
    ('GSAG', 'grd', 'float32'), ('GSBG', 'grd', 'float32'),
    ('GS7BG', 'grd', 'float64'), ('ERS', 'ers', 'float32'), ('AAIGrid', 'asc', 'float32'),
])
def test_readers_match_gdal_written_files(tmp_path, driver, ext, dtype):
    import rasterio
    from rasterio.transform import from_origin
    north_up = (np.arange(12, dtype=np.float64).reshape(3, 4) * 1.5) + 100
    path = tmp_path / f'f.{ext}'
    with rasterio.open(path, 'w', driver=driver, height=3, width=4, count=1, dtype=dtype,
                       transform=from_origin(1000.0, 2030.0, 10.0, 10.0)) as d:
        d.write(north_up.astype(dtype), 1)
    lyr = L.load_grid(str(path))
    np.testing.assert_allclose(lyr['grid'], north_up[::-1])
    np.testing.assert_allclose(lyr['x'], 1005 + 10 * np.arange(4))
    np.testing.assert_allclose(lyr['y'], 2005 + 10 * np.arange(3))


@requires_rasterio
@pytest.mark.parametrize('writer,ext', [
    (W.save_surfer_ascii_grd, 'grd'), (W.save_surfer7_binary_grd, 'grd'),
    (W.save_geotiff, 'tif'),
])
def test_writers_are_readable_by_gdal(tmp_path, rng, writer, ext):
    g, x, y = _grid_xy(rng)
    path = writer(g, x, y, tmp_path / f'w.{ext}')
    arr, gx, gy = _read_gdal(path)
    np.testing.assert_allclose(arr, g, rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(gx, x)
    np.testing.assert_allclose(gy, y)


# ─── ESRI ASCII ─────────────────────────────────────────────────────────────

def test_esri_ascii_roundtrip_and_gdal(tmp_path, rng):
    g, x, _ = _grid_xy(rng)
    y = 4100000 + 25.0 * np.arange(g.shape[0])
    path = W.save_esri_ascii(g, x, y, tmp_path / 'g.asc')
    lyr = L.load_grid(path)
    assert lyr['format'] == 'esri_ascii'
    np.testing.assert_allclose(lyr['grid'], g, equal_nan=True)
    np.testing.assert_allclose(lyr['x'], x)
    np.testing.assert_allclose(lyr['y'], y)
    if HAS_RASTERIO:
        arr, gx, gy = _read_gdal(path)
        np.testing.assert_allclose(arr, g, equal_nan=True)
        np.testing.assert_allclose(gy, y)


def test_esri_ascii_center_registration(tmp_path):
    path = tmp_path / 'c.txt'
    path.write_text("NCOLS 2\nNROWS 2\nXLLCENTER 10\nYLLCENTER 20\nCELLSIZE 5\n"
                    "NODATA_VALUE -1\n1 2\n-1 4\n")
    assert L.detect_format(str(path)) == 'esri_ascii'
    lyr = L.load_grid(str(path))
    np.testing.assert_allclose(lyr['x'], [10, 15])
    np.testing.assert_allclose(lyr['y'], [20, 25])
    assert np.isnan(lyr['grid'][0, 0])           # south row: "-1 4"
    assert lyr['grid'][1, 1] == 2


def test_esri_writer_rejects_non_square_cells(tmp_path, rng):
    g, x, y = _grid_xy(rng)
    with pytest.raises(ValueError):
        W.save_esri_ascii(g, x, y, tmp_path / 'g.asc')


def test_writers_reject_irregular_grid(tmp_path):
    with pytest.raises(ValueError):
        W.save_surfer_ascii_grd(np.zeros((2, 3)), [0, 1, 3], [0, 1], tmp_path / 'a.grd')


def test_save_grid_dispatches_by_extension(tmp_path, rng):
    g, x, y = _grid_xy(rng)
    p = W.save_grid(g, x, y, tmp_path / 'a.grd')
    assert L.detect_format(p) == 'surfer_ascii'
    p = W.save_grid(g, x, y, tmp_path / 'b.grd', fmt='surfer7')
    assert L.detect_format(p) == 'surfer7_binary'
    with pytest.raises(ValueError):
        W.save_grid(g, x, y, tmp_path / 'c.xyz')


# ─── Geosoft ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('name,atol', [
    ('om_float', 1e-8), ('om_short', 1e-3), ('om_long', 1e-8), ('om_double', 1e-8),
    ('om_byte', 0.15), ('om_order', 1e-8), ('om_compress', 1e-8),
])
def test_geosoft_real_oasis_files(name, atol):
    exp = np.load(os.path.join(GEOSOFT_DIR, 'om_expected.npz'))
    path = os.path.join(GEOSOFT_DIR, f'{name}.grd')
    assert L.detect_format(path) == 'geosoft_grd'
    lyr = L.load_grid(path)
    assert np.array_equal(np.isnan(lyr['grid']), np.isnan(exp['grid']))
    np.testing.assert_allclose(lyr['grid'], exp['grid'], atol=atol, equal_nan=True)
    np.testing.assert_allclose(lyr['x'], exp['easting'])
    np.testing.assert_allclose(lyr['y'], exp['northing'])


def _rotated_node_errors(lyr, g, rot_deg, x0=1.0, y0=-24.0):
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((lyr['y'], lyr['x']), lyr['grid'], bounds_error=False)
    th = np.radians(rot_deg)
    errs = []
    for r in range(2, g.shape[0] - 2):
        for c in range(2, g.shape[1] - 2):
            if not np.isfinite(g[r - 1:r + 2, c - 1:c + 2]).all():
                continue
            e = x0 + c * np.cos(th) - r * np.sin(th)
            n = y0 + c * np.sin(th) + r * np.cos(th)
            v = interp([[n, e]])[0]
            if np.isfinite(v):
                errs.append(abs(v - g[r, c]))
    return np.array(errs)


def test_geosoft_rotated_real_file_uses_correct_convention():
    """Real Oasis file (ROT=-30°): node values must be found at their rotated positions."""
    exp = np.load(os.path.join(GEOSOFT_DIR, 'om_expected.npz'))
    with pytest.warns(UserWarning, match='Rotated'):
        lyr = L.load_grid(os.path.join(GEOSOFT_DIR, 'om_rotate.grd'))
    g = exp['grid']
    right = _rotated_node_errors(lyr, g, -30.0)
    wrong = _rotated_node_errors(lyr, g, +30.0)
    assert len(right) > 200
    # The residual error is only the roughness error of the double interpolation (σ≈11.2)
    assert np.median(right) < 0.2 * np.nanstd(g)
    assert np.median(right) < 0.25 * np.median(wrong)


def test_geosoft_rotation_reproduces_linear_field_exactly(tmp_path):
    rot, x0, y0, d = 25.0, 500.0, 800.0, 10.0
    th = np.radians(rot)
    rows, cols = np.mgrid[0:20, 0:30]
    east = x0 + cols * d * np.cos(th) - rows * d * np.sin(th)
    north = y0 + cols * d * np.sin(th) + rows * d * np.cos(th)
    field = 0.3 * east - 0.7 * north + 11.0
    p = write_geosoft(tmp_path / 'rot.grd', field.astype(np.float64), x0=x0, y0=y0,
                      dx=d, dy=d, dtype='f8', rot=rot)
    with pytest.warns(UserWarning):
        lyr = L.load_grid(p)
    X, Y = np.meshgrid(lyr['x'], lyr['y'])
    ok = np.isfinite(lyr['grid'])
    assert ok.mean() > 0.4
    np.testing.assert_allclose(lyr['grid'][ok], (0.3 * X - 0.7 * Y + 11.0)[ok], atol=1e-6)


def test_geosoft_scaling_dummies_and_column_order(tmp_path):
    stored = np.array([[0, 10, -32767], [20, 30, 40]], dtype='<i2')
    p = write_geosoft(tmp_path / 'a.grd', stored, x0=100, y0=200, dx=2, dy=3,
                      dtype='i2', sf=1, zbase=5.0, zmult=10.0)
    lyr = L.load_grid(p)
    expected = stored / 10.0 + 5.0
    expected = expected.astype(float)
    expected[0, 2] = np.nan
    np.testing.assert_allclose(lyr['grid'], expected, equal_nan=True)
    np.testing.assert_allclose(lyr['x'], [100, 102, 104])
    np.testing.assert_allclose(lyr['y'], [200, 203])

    g = np.arange(12, dtype=np.float32).reshape(3, 4)
    p2 = write_geosoft(tmp_path / 'b.grd', g, x0=0, y0=0, dx=1, dy=2, kx=-1)
    lyr2 = L.load_grid(p2)
    np.testing.assert_allclose(lyr2['grid'], g)
    np.testing.assert_allclose(lyr2['y'], [0, 2, 4])


def test_geosoft_float_dummy_is_nan(tmp_path):
    g = np.array([[1.0, -1e32], [3.0, 4.0]], dtype=np.float32)
    lyr = L.load_grid(write_geosoft(tmp_path / 'd.grd', g))
    assert np.isnan(lyr['grid'][0, 1]) and lyr['grid'][1, 1] == 4


# ─── ER Mapper ERS ──────────────────────────────────────────────────────────

def _ers_header(nl, nc, celltype='IEEE4ByteReal', order='LSBFirst', bands=1,
                extra_raster='', reg='Eastings = 1000\n\t\t\tNorthings = 2030'):
    return (
        "DatasetHeader Begin\n\tVersion = \"6.0\"\n\tDataSetType = ERStorage\n"
        f"\tDataType = Raster\n\tByteOrder = {order}\n"
        "\tCoordinateSpace Begin\n\t\tDatum = \"WGS84\"\n\t\tProjection = \"SUTM35\"\n"
        "\t\tCoordinateType = EN\n\tCoordinateSpace End\n"
        f"\tRasterInfo Begin\n\t\tCellType = {celltype}\n\t\tNrOfLines = {nl}\n"
        f"\t\tNrOfCellsPerLine = {nc}\n\t\tNrOfBands = {bands}\n"
        "\t\tCellInfo Begin\n\t\t\tXdimension = 10\n\t\t\tYdimension = 20\n\t\tCellInfo End\n"
        f"\t\tRegistrationCoord Begin\n\t\t\t{reg}\n\t\tRegistrationCoord End\n"
        f"{extra_raster}\tRasterInfo End\nDatasetHeader End\n"
    )


def test_ers_reader_bil_bands_byteorder_and_registration(tmp_path):
    nl, nc, nb = 2, 3, 2
    band1 = np.array([[1, 2, 3], [4, 5, 6]], dtype='>i2')
    band2 = band1 * 10
    cube = np.stack([band1, band2], axis=1)          # (lines, bands, cells) = BIL
    (tmp_path / 'r').write_bytes(cube.astype('>i2').tobytes())
    hdr = _ers_header(nl, nc, 'Signed16BitInteger', 'MSBFirst', nb,
                      extra_raster="\t\tNullCellValue = 5\n"
                                   "\t\tRegistrationCellX = 0.5\n\t\tRegistrationCellY = 0.5\n")
    (tmp_path / 'r.ers').write_text(hdr)
    lyr = L.load_grid(str(tmp_path / 'r.ers'))
    # row 0 = south → last row of band1
    np.testing.assert_allclose(lyr['grid'], np.array([[4, np.nan, 6], [1, 2, 3]]), equal_nan=True)
    # registration at the cell centre (0.5): the first cell centre is exactly (1000, 2030)
    np.testing.assert_allclose(lyr['x'], [1000, 1010, 1020])
    np.testing.assert_allclose(lyr['y'], [2010, 2030])
    assert lyr['crs'] == 'SUTM35'
    lyr2 = L.load_grid(str(tmp_path / 'r.ers'), band=2)
    np.testing.assert_allclose(lyr2['grid'], band2[::-1])


def test_ers_missing_data_file_raises(tmp_path):
    (tmp_path / 'x.ers').write_text(_ers_header(2, 2))
    with pytest.raises(FileNotFoundError):
        L.load_grid(str(tmp_path / 'x.ers'))


# ─── GeoTIFF ────────────────────────────────────────────────────────────────

@requires_rasterio
def test_geotiff_reader_returns_pixel_center_coordinates(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    tif = tmp_path / 'mini.tif'
    with rasterio.open(tif, 'w', driver='GTiff', height=2, width=2, count=1,
                       dtype='float32', transform=from_origin(100.0, 200.0, 1.0, 1.0)) as d:
        d.write(np.array([[1, 2], [3, 4]], dtype=np.float32), 1)
    out = L.read_geotiff(str(tif))
    assert out['x'][0] == pytest.approx(100.5)
    assert out['y'][0] == pytest.approx(198.5)
    assert out['grid'][0, 0] == 3       # row 0 = south


@requires_rasterio
@pytest.mark.parametrize('area_or_point', ['Area', 'Point'])
def test_pillow_fallback_matches_rasterio(tmp_path, area_or_point):
    pytest.importorskip('PIL')
    import rasterio
    from rasterio.transform import from_origin
    p = tmp_path / f'{area_or_point}.tif'
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    data[0, 0] = -5
    with rasterio.open(p, 'w', driver='GTiff', height=3, width=4, count=1, dtype='float32',
                       transform=from_origin(1000.0, 2030.0, 10.0, 10.0), nodata=-5.0) as d:
        d.update_tags(AREA_OR_POINT=area_or_point)
        d.write(data, 1)
    a = L.read_raster_gdal(str(p))
    b = L._read_tiff_fallback(str(p))
    np.testing.assert_allclose(a['x'], b['x'])
    np.testing.assert_allclose(a['y'], b['y'])
    np.testing.assert_allclose(a['grid'], b['grid'], equal_nan=True)
    assert np.isnan(b['grid']).sum() == 1


def test_pillow_fallback_without_geotags_uses_pixel_coords(tmp_path):
    PIL = pytest.importorskip('PIL.Image')
    p = tmp_path / 'plain.tif'
    PIL.fromarray(np.arange(6, dtype=np.float32).reshape(2, 3)).save(p)
    with pytest.warns(UserWarning):
        lyr = L._read_tiff_fallback(str(p))
    assert lyr['crs'] == 'pixel'
    assert lyr['grid'][0, 0] == 3      # the bottom row is south


# ─── Format detection ───────────────────────────────────────────────────────

def test_detect_format_by_content(tmp_path):
    (tmp_path / 'a.dat').write_text("DSAA\n2 2\n0 1\n0 1\n0 3\n0 1\n2 3\n")
    assert L.detect_format(str(tmp_path / 'a.dat')) == 'surfer_ascii'
    (tmp_path / 'b.txt').write_text("x y z\n1 2 3\n")
    assert L.detect_format(str(tmp_path / 'b.txt')) == 'xyz'
    (tmp_path / 'c.grd').write_bytes(b'\x00' * 100)
    assert L.detect_format(str(tmp_path / 'c.grd')) in ('gdal', 'unknown')


# ─── XYZ / CSV ──────────────────────────────────────────────────────────────

def test_xyz_lattice_is_reconstructed_exactly(tmp_path):
    x = 300000 + 50.0 * np.arange(12)
    y = 4000000 + 75.0 * np.arange(9)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X / 300) + Y / 1e5
    keep = np.ones_like(Z, dtype=bool)
    keep[3:5, 4:7] = False                       # data gap
    keep[:, 11] = False                          # entirely empty last column
    import pandas as pd
    pd.DataFrame({'Easting': X[keep], 'Northing': Y[keep], 'TMI': Z[keep]}).to_csv(
        tmp_path / 'g.xyz', sep='\t', index=False, float_format='%.10g')
    lyr = L.load_grid(str(tmp_path / 'g.xyz'))
    assert lyr['format'] == 'xyz_lattice'
    assert lyr['grid'].shape == (9, 11)
    np.testing.assert_allclose(lyr['x'], x[:11])
    np.testing.assert_allclose(lyr['grid'][keep[:, :11]], Z[:, :11][keep[:, :11]], rtol=1e-8)
    assert np.isnan(lyr['grid'][3:5, 4:7]).all()


def test_xyz_scattered_linear_field_is_interpolated_exactly(tmp_path, rng):
    import pandas as pd
    px = rng.uniform(0, 1000, 400)
    py = rng.uniform(0, 1000, 400)
    pz = 2.0 * px - 3.0 * py + 7
    pd.DataFrame({'x': px, 'y': py, 'value': pz}).to_csv(tmp_path / 's.csv', index=False)
    lyr = L.load_grid(str(tmp_path / 's.csv'), target_nx=30, target_ny=25, smooth_sigma=0)
    assert lyr['format'] == 'xyz_interpolated'
    X, Y = np.meshgrid(lyr['x'], lyr['y'])
    inside = (X > 150) & (X < 850) & (Y > 150) & (Y < 850)
    np.testing.assert_allclose(lyr['grid'][inside], (2 * X - 3 * Y + 7)[inside], atol=1e-6)


def test_xyz_headerless_whitespace_file(tmp_path, rng):
    pts = rng.uniform(0, 100, (50, 3))
    np.savetxt(tmp_path / 'h.xyz', pts, fmt='%.6f')
    lyr = L.load_grid(str(tmp_path / 'h.xyz'), target_nx=10, target_ny=10)
    assert lyr['n_points'] == 50                 # the first row must not be mistaken for a header
    assert lyr['columns'] == ('x', 'y', 'z')


def test_xyz_column_detection_avoids_decoy_columns(tmp_path, rng):
    import pandas as pd
    n = 60
    df = pd.DataFrame({
        'station_index': np.arange(n),
        'UTM_Easting': rng.uniform(0, 100, n),
        'UTM_Northing': rng.uniform(0, 100, n),
        'Resistivity_ohmm': rng.uniform(1, 10, n),
    })
    df.to_csv(tmp_path / 'r.csv', index=False)
    lyr = L.load_grid(str(tmp_path / 'r.csv'), target_nx=8, target_ny=8)
    assert lyr['columns'] == ('utm_easting', 'utm_northing', 'resistivity_ohmm')


def test_xyz_duplicates_and_max_distance(tmp_path):
    import pandas as pd
    xs, ys = np.meshgrid(np.linspace(0, 10, 6), np.linspace(0, 10, 6))
    df = pd.DataFrame({'x': np.r_[xs.ravel() + 0.01, 0.01],
                       'y': np.r_[ys.ravel() + 0.02, 0.02],
                       'z': np.r_[np.ones(36), 3.0]})       # duplicate at (0,0) → mean 2
    df.loc[len(df)] = [100.0, 100.0, 1.0]                  # single distant point
    df.to_csv(tmp_path / 'd.csv', index=False)
    lyr = L.load_grid(str(tmp_path / 'd.csv'), target_nx=21, target_ny=21,
                      smooth_sigma=0, max_distance=6.0, grid_method='interpolate')
    assert lyr['grid'][0, 0] == pytest.approx(2.0)
    assert np.isnan(lyr['grid'][10, 10])        # (50, 50) is far from the data → NaN
    assert lyr['n_points'] == 37


def test_xyz_collinear_points_fall_back_to_nearest(tmp_path):
    import pandas as pd
    pd.DataFrame({'x': np.arange(20.0), 'y': np.arange(20.0), 'z': np.arange(20.0)}).to_csv(
        tmp_path / 'l.csv', index=False)
    with pytest.warns(UserWarning, match='nearest'):
        lyr = L.load_grid(str(tmp_path / 'l.csv'), target_nx=5, target_ny=5,
                          grid_method='interpolate')
    assert np.isfinite(lyr['grid']).all()


def test_load_grid_kwarg_validation(tmp_path, rng):
    g, x, y = _grid_xy(rng, with_nan=False)
    p = W.save_surfer_ascii_grd(g, x, y, tmp_path / 'g.grd')
    with pytest.warns(UserWarning, match='ignored'):
        L.load_grid(p, x_col='a')
    (tmp_path / 'p.csv').write_text("x,y,z\n" + "\n".join(f"{i},{i*2 % 7},{i}" for i in range(20)))
    with pytest.raises(TypeError):
        L.load_grid(str(tmp_path / 'p.csv'), bogus=1)


def test_load_grid_all_nan_raises(tmp_path):
    p = W.save_surfer_ascii_grd(np.full((2, 2), np.nan), [0, 1], [0, 1], tmp_path / 'n.grd')
    with pytest.raises(ValueError, match='entirely empty'):
        L.load_grid(p)


def test_load_grid_missing_file():
    with pytest.raises(FileNotFoundError):
        L.load_grid('/nonexistent/file.grd')


def test_find_column_exact_beats_partial():
    import pandas as pd
    df = pd.DataFrame({'station_index': [1], 'x': [2], 'y': [3]})
    assert L._find_column(df, None, ['x', 'easting']) == 'x'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # the numeric fallback must not pick a column already assigned to another role (exclude)
        assert L._find_column(df, None, ['easting'], fallback_index=0,
                              exclude=('station_index',)) == 'x'
    with pytest.raises(KeyError):
        L._find_column(df, 'missing', ['x'])
