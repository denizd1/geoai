"""
I/O regression and feature tests (second round): ERS nodata and CRS, target-type
detection, XYZ blanks / lattice / delimiters / Geosoft XYZ dialect, rotated-grid
holes, GDAL scale/offset and bands, native netCDF-3 reader and the Geosoft writer.
"""
import os
import struct
import warnings

import numpy as np
import pandas as pd
import pytest

from conftest import DATA_DIR, requires_rasterio, write_geosoft
from geoai.io import loaders as L
from geoai.io import writers as W

GEOSOFT_DIR = os.path.join(DATA_DIR, 'geosoft')


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding='utf-8')
    return str(p)


def _quiet_load(path, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return L.load_grid(path, **kw)


def _ers_header(nl, nc, celltype='IEEE4ByteReal', projection='NUTM35', datum='WGS84',
                extra_raster=''):
    return (
        "DatasetHeader Begin\n\tVersion = \"6.0\"\n\tDataSetType = ERStorage\n"
        "\tDataType = Raster\n\tByteOrder = LSBFirst\n"
        f"\tCoordinateSpace Begin\n\t\tDatum = \"{datum}\"\n\t\tProjection = \"{projection}\"\n"
        "\t\tCoordinateType = EN\n\tCoordinateSpace End\n"
        f"\tRasterInfo Begin\n\t\tCellType = {celltype}\n\t\tNrOfLines = {nl}\n"
        f"\t\tNrOfCellsPerLine = {nc}\n\t\tNrOfBands = 1\n"
        "\t\tCellInfo Begin\n\t\t\tXdimension = 10\n\t\t\tYdimension = 10\n\t\tCellInfo End\n"
        "\t\tRegistrationCoord Begin\n\t\t\tEastings = 500000\n\t\t\tNorthings = 4100000\n"
        "\t\tRegistrationCoord End\n"
        f"{extra_raster}\tRasterInfo End\nDatasetHeader End\n"
    )


def _write_ers(tmp_path, cells, name='r', **hdr):
    cells = np.asarray(cells)
    (tmp_path / name).write_bytes(cells.astype(cells.dtype.newbyteorder('<')).tobytes())
    path = tmp_path / f'{name}.ers'
    path.write_text(_ers_header(cells.shape[0], cells.shape[1], **hdr))
    return str(path)


# ─── ERS NullCellValue (io-01) ──────────────────────────────────────────────

@pytest.mark.parametrize('null_text,null_f32', [
    ('-1e+32', -1e32), ('-99999.9', -99999.9), ('-3.40282e+38', np.finfo(np.float32).min),
])
def test_ers_float32_null_value_is_masked(tmp_path, null_text, null_f32):
    cells = np.full((4, 5), 48000.0, np.float32)
    cells[1, 2] = np.float32(null_f32)
    p = _write_ers(tmp_path, cells,
                   extra_raster=f"\t\tNullCellValue = {null_text}\n")
    lyr = _quiet_load(p)
    assert int(np.isnan(lyr['grid']).sum()) == 1
    assert np.isnan(lyr['grid'][2, 2])                 # row 1 from the north = row 2 from the south
    assert np.nanmean(lyr['grid']) == pytest.approx(48000.0)


def test_ers_integer_null_value_is_masked(tmp_path):
    cells = np.array([[1, -32768, 3], [4, 5, 6]], dtype=np.int16)
    p = _write_ers(tmp_path, cells, celltype='Signed16BitInteger',
                   extra_raster="\t\tNullCellValue = -32768\n")
    lyr = _quiet_load(p)
    np.testing.assert_array_equal(np.isnan(lyr['grid']), [[False, False, False],
                                                          [False, True, False]])


@requires_rasterio
def test_ers_written_by_gdal_null_matches_gdal_mask(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    p = tmp_path / 'mag.ers'
    a = np.full((4, 5), 48000.0, np.float32)
    a[1, 2] = -1e32
    with rasterio.open(p, 'w', driver='ERS', width=5, height=4, count=1, dtype='float32',
                       transform=from_origin(500000, 4100000, 10, 10), nodata=-1e32) as dst:
        dst.write(a, 1)
    with rasterio.open(p) as s:
        gdal_mask = s.read(1, masked=True).mask[::-1]
    lyr = _quiet_load(str(p))
    np.testing.assert_array_equal(np.isnan(lyr['grid']), gdal_mask)
    assert np.nanmean(lyr['grid']) == pytest.approx(48000.0)


# ─── ERS CRS (io-02) ────────────────────────────────────────────────────────

@pytest.mark.parametrize('projection,datum,expected', [
    ('NUTM35', 'WGS84', 'EPSG:32635'), ('SUTM35', 'WGS84', 'EPSG:32735'),
    ('RAW', 'RAW', 'unknown'), ('LOCAL', 'WGS84', 'unknown'), ('', 'WGS84', 'unknown'),
])
def test_ers_crs_is_resolved_and_raw_code_kept(tmp_path, projection, datum, expected):
    p = _write_ers(tmp_path, np.ones((2, 3), np.float32), projection=projection, datum=datum)
    lyr = _quiet_load(p)
    assert lyr['crs'] == expected
    assert lyr['ers_projection'] == projection
    assert lyr['ers_datum'] == datum


@pytest.mark.parametrize('projection,datum,expected', [
    ('NUTM35', 'WGS84', 'EPSG:32635'), ('SUTM01', 'WGS84', 'EPSG:32701'),
    ('GEODETIC', 'WGS84', 'EPSG:4326'), ('RAW', 'RAW', 'unknown'),
    ('TMAMG56', 'GDA94', 'TMAMG56'),
])
def test_ers_crs_mapping_without_rasterio(tmp_path, monkeypatch, projection, datum, expected):
    monkeypatch.setattr(L, '_rasterio_available', lambda: False)
    p = _write_ers(tmp_path, np.ones((2, 3), np.float32), projection=projection, datum=datum)
    assert _quiet_load(p)['crs'] == expected


@requires_rasterio
def test_ers_crs_matches_geotiff_crs_and_can_be_exported(tmp_path):
    """An ERS and a GeoTIFF in the same projection report the same CRS string."""
    import rasterio
    from rasterio.transform import from_origin
    a = np.arange(20, dtype=np.float32).reshape(4, 5)
    paths = {}
    for driver, ext in (('ERS', 'ers'), ('GTiff', 'tif')):
        paths[ext] = tmp_path / f'g.{ext}'
        with rasterio.open(paths[ext], 'w', driver=driver, width=5, height=4, count=1,
                           dtype='float32', crs='EPSG:32635',
                           transform=from_origin(500000, 4100000, 10, 10)) as dst:
            dst.write(a, 1)
    ers = _quiet_load(str(paths['ers']))
    tif = _quiet_load(str(paths['tif']))
    assert ers['ers_projection'] == 'NUTM35'
    assert ers['crs'] == tif['crs'] == 'EPSG:32635'
    out = W.save_geotiff(ers['grid'], ers['x'], ers['y'], tmp_path / 'o.tif', crs=ers['crs'])
    with rasterio.open(out) as s:
        assert s.crs.to_epsg() == 32635


@requires_rasterio
def test_save_geotiff_with_unparseable_crs_warns_and_writes_without_crs(tmp_path):
    import rasterio
    g = np.arange(12, dtype=float).reshape(3, 4)
    x, y = 10.0 * np.arange(4), 10.0 * np.arange(3)
    with pytest.warns(UserWarning, match='not recognised'):
        p = W.save_geotiff(g, x, y, tmp_path / 'a.tif', crs='NUTM35')
    with rasterio.open(p) as s:
        assert s.crs is None
        np.testing.assert_allclose(s.read(1)[::-1], g)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        p2 = W.save_geotiff(g, x, y, tmp_path / 'b.tif', crs='EPSG:32635')
        p3 = W.save_geotiff(g, x, y, tmp_path / 'c.tif', crs='unknown')
    with rasterio.open(p2) as s2, rasterio.open(p3) as s3:
        assert s2.crs.to_epsg() == 32635 and s3.crs is None


# ─── TARGET_TYPE detection (io-03, orchestration-5) ─────────────────────────

@pytest.mark.parametrize('col,values', [
    ('ROCK_TYPE', ['granite', 'schist', 'granite', 'andesite']),
    ('HOLE_TYPE', ['RC', 'DD', 'RC', 'RAB']),
    ('SAMPLE_TYPE', ['core', 'chip', 'core', 'chip']),
    ('TYPE', ['DD', 'RC', 'DD', 'RC']),
    ('CATEGORY', ['a', 'b', 'a', 'b']),
])
def test_drillhole_type_columns_are_not_target_type(tmp_path, col, values):
    rows = ''.join(f'DH0{i},{500100 + 100 * i},{4100100 + 100 * i},{v},{i % 2}\n'
                   for i, v in enumerate(values))
    p = _write(tmp_path, 'holes.csv', f'HOLE_ID,EASTING,NORTHING,{col},LABEL\n' + rows)
    df = L.load_well_data(p, target_type='mineral')
    assert len(df) == 4
    assert 'target_type' not in df
    assert df['label'].tolist() == [0, 1, 0, 1]


@pytest.mark.parametrize('col', ['COMMODITY', 'DEPOSIT_TYPE', 'Target Type', 'Resource-Type',
                                 'TargetType', 'HEDEF_TIPI'])
def test_explicit_target_type_names_are_detected(tmp_path, col):
    p = _write(tmp_path, 'w.csv', f'X,Y,{col},LABEL\n1,1,Gold,1\n2,2,copper,0\n3,3,gold,0\n')
    df = L.load_well_data(p, target_type='gold')
    assert df['target_type'].tolist() == ['gold', 'gold']
    assert df['label'].tolist() == [1, 0]


def test_numeric_target_type_candidate_is_rejected(tmp_path):
    p = _write(tmp_path, 'w.csv', 'X,Y,COMMODITY,LABEL\n1,1,3,1\n2,2,7,0\n3,3,,1\n')
    df = L.load_well_data(p, target_type='mineral')
    assert len(df) == 3 and 'target_type' not in df


def test_target_type_col_explicit_and_disabled(tmp_path):
    p = _write(tmp_path, 'w.csv', 'X,Y,TYPE,TARGET_TYPE,LABEL\n'
                                  '1,1,DD,mineral,1\n2,2,RC,water,0\n3,3,DD,mineral,0\n')
    df = L.load_well_data(p, target_type_col='type')            # explicit: used as given
    assert df['target_type'].tolist() == ['dd', 'rc', 'dd']
    df = L.load_well_data(p, target_type_col=False, target_type='mineral')
    assert len(df) == 3 and 'target_type' not in df
    df = L.load_well_data(p, target_type='mineral')             # auto: TARGET_TYPE only
    assert df['target_type'].tolist() == ['mineral', 'mineral']


def test_warning_when_target_type_filter_removes_every_well(tmp_path):
    p = _write(tmp_path, 'w.csv', 'X,Y,TARGET_TYPE,LABEL\n1,1,water,1\n2,2,water,0\n')
    with pytest.warns(UserWarning, match='no wells remain'):
        df = L.load_well_data(p, target_type='mineral')
    assert len(df) == 0


# ─── parse_labels (public contract) ─────────────────────────────────────────

def test_parse_labels_public_contract():
    assert L._parse_labels is L.parse_labels
    s = pd.Series(['yes', 'No', ' 1 ', '0.0', None, '', 'evet', 'hayır', 'maybe', '2', '+', '-'],
                  index=list('abcdefghijkl'))
    out = L.parse_labels(s)
    assert out.dtype == np.float64
    assert out.index.tolist() == list('abcdefghijkl')
    expected = [1, 0, 1, 0, np.nan, np.nan, 1, 0, np.inf, np.inf, 1, 0]
    np.testing.assert_array_equal(out.to_numpy(), np.array(expected, dtype=float))
    np.testing.assert_array_equal(L.parse_labels(pd.Series([1, 0, 2, np.nan])).to_numpy(),
                                  [1.0, 0.0, np.inf, np.nan])
    np.testing.assert_array_equal(L.parse_labels(pd.Series([True, False])).to_numpy(), [1.0, 0.0])


# ─── XYZ blanks / dummies / nodata (io-04) ──────────────────────────────────

def test_xyz_surfer_blanks_become_nan_lattice_nodes(tmp_path):
    X, Y = np.meshgrid(500000 + 50 * np.arange(20), 4100000 + 50 * np.arange(15))
    Z = 48000 + 0.01 * (X - 500000)
    Z[:, :4] = 1.70141e38
    p = tmp_path / 'surfer_export.dat'
    with open(p, 'w') as f:
        for a, b, c in zip(X.ravel(), Y.ravel(), Z.ravel()):
            f.write(f'{a:.1f} {b:.1f} {c:.6g}\n')
    lyr = _quiet_load(str(p))
    assert lyr['format'] == 'xyz_lattice'
    assert lyr['grid'].shape == (15, 20)
    assert int(np.isnan(lyr['grid']).sum()) == 60
    assert lyr['nan_percent'] == pytest.approx(20.0)
    assert np.isnan(lyr['grid'][:, :4]).all()
    np.testing.assert_allclose(lyr['grid'][:, 4:], Z[:, 4:], rtol=1e-9)
    assert lyr['n_points'] == 240


def test_xyz_geosoft_dummy_star_and_user_nodata(tmp_path):
    rows = []
    for j in range(4):
        for i in range(5):
            z = 100.0 + i + 10 * j
            if (i, j) == (1, 1):
                z = '-1e32'
            elif (i, j) == (2, 2):
                z = '*'
            elif (i, j) == (3, 3):
                z = '-9999'
            elif (i, j) == (0, 3):
                z = '-8888'
            rows.append(f'{10 * i} {20 * j} {z}')
    p = _write(tmp_path, 'g.xyz', 'x y z\n' + '\n'.join(rows) + '\n')
    lyr = _quiet_load(p)
    nan_nodes = set(zip(*np.where(np.isnan(lyr['grid']))))
    assert nan_nodes == {(1, 1), (2, 2)}                    # (row j, col i)
    assert lyr['grid'][3, 3] == -9999
    lyr = _quiet_load(p, nodata=-9999)
    assert set(zip(*np.where(np.isnan(lyr['grid'])))) == {(1, 1), (2, 2), (3, 3)}
    lyr = _quiet_load(p, nodata=[-9999, -8888.0])
    assert set(zip(*np.where(np.isnan(lyr['grid'])))) == {(1, 1), (2, 2), (3, 3), (3, 0)}
    assert lyr['grid'][0, 0] == 100.0


# ─── Lattice detection with rounded coordinates (io-05) ─────────────────────

def test_lattice_with_rounded_non_representable_spacing(tmp_path):
    x1 = 500000 + np.arange(301) * (100 / 3)
    y1 = 4100000 + np.arange(151) * (100 / 3)
    X, Y = np.meshgrid(x1, y1)
    Z = np.zeros_like(X)
    Z[75, 150] = 1000.0
    p = tmp_path / 'grid33m.xyz'
    with open(p, 'w') as f:
        for a, b, c in zip(X.ravel(), Y.ravel(), Z.ravel()):
            f.write(f'{a:.2f} {b:.2f} {c:.3f}\n')
    for method in ('auto', 'lattice'):
        lyr = _quiet_load(str(p), grid_method=method)
        assert lyr['format'] == 'xyz_lattice'
        assert lyr['grid'].shape == (151, 301)
        assert np.nanmax(lyr['grid']) == 1000.0
        assert lyr['grid'][75, 150] == 1000.0
        np.testing.assert_allclose(lyr['x'], x1, atol=0.006)
        np.testing.assert_allclose(lyr['y'], y1, atol=0.006)


def test_lattice_geographic_30_arc_seconds_six_decimals(tmp_path):
    lon = 30.0 + np.arange(120) / 120.0
    lat = 38.0 + np.arange(90) / 120.0
    X, Y = np.meshgrid(lon, lat)
    Z = np.sin(X * 7) + np.cos(Y * 5)
    p = tmp_path / 'geo.csv'
    pd.DataFrame({'lon': X.ravel(), 'lat': Y.ravel(), 'value': Z.ravel()}).to_csv(
        p, index=False, float_format='%.6f')
    lyr = _quiet_load(str(p))
    assert lyr['format'] == 'xyz_lattice'
    assert lyr['grid'].shape == (90, 120)
    np.testing.assert_allclose(lyr['grid'], Z, atol=1e-6)


def test_jittered_points_are_not_a_lattice(tmp_path, rng):
    X, Y = np.meshgrid(50.0 * np.arange(20), 50.0 * np.arange(20))
    jx = X + rng.uniform(-5, 5, X.shape)
    jy = Y + rng.uniform(-5, 5, Y.shape)
    pd.DataFrame({'x': jx.ravel(), 'y': jy.ravel(), 'z': np.ones(400)}).to_csv(
        tmp_path / 'j.csv', index=False)
    assert L._detect_lattice(jx.ravel(), jy.ravel()) is None
    lyr = _quiet_load(str(tmp_path / 'j.csv'), target_nx=10, target_ny=10)
    assert lyr['format'] == 'xyz_interpolated'


# ─── '#' handling (io-06) ───────────────────────────────────────────────────

def test_hash_inside_fields_is_kept_and_comment_lines_dropped(tmp_path):
    p = _write(tmp_path, 'wells.csv',
               '# exported 2024-01-01\nX,Y,WELL_ID,LABEL\n500100,4100100,BH#1,1\n'
               '   # indented comment line\n500200,4100200,BH#2,0\n'
               '500300,4100300,BH-3,0\n500400,4100400,BH-4,1\n')
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        df = L.load_well_data(p)
    assert df['well_id'].tolist() == ['BH#1', 'BH#2', 'BH-3', 'BH-4']
    assert df['label'].tolist() == [1, 0, 0, 1]


def test_hash_in_xyz_attribute_column(tmp_path):
    rows = ''.join(f'Pad #{k},{10 * (k % 4)},{10 * (k // 4)},{k}\n' for k in range(16))
    p = _write(tmp_path, 'pads.csv', 'site,x,y,z\n' + rows)
    lyr = _quiet_load(p)
    assert lyr['format'] == 'xyz_lattice' and lyr['n_points'] == 16
    np.testing.assert_allclose(lyr['grid'], np.arange(16.0).reshape(4, 4))


# ─── GeoTIFF bands (io-07) ──────────────────────────────────────────────────

@requires_rasterio
def test_geotiff_band_selection(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    p = tmp_path / 'radiometrics_KThU.tif'
    bands = np.stack([np.full((4, 5), v, np.float32) for v in (1.5, 12.0, 3.0)])
    with rasterio.open(p, 'w', driver='GTiff', width=5, height=4, count=3, dtype='float32',
                       transform=from_origin(500000, 4100000, 10, 10), crs='EPSG:32635') as dst:
        dst.write(bands)
    for b, v in ((1, 1.5), (2, 12.0), (3, 3.0)):
        assert L.load_grid(str(p), band=b)['grid'][0, 0] == v
        assert L.read_geotiff(str(p), band=b)['grid'][0, 0] == v
    with pytest.raises(ValueError, match='band'):
        L.load_grid(str(p), band=99)


def test_pillow_fallback_rejects_other_bands(tmp_path):
    PIL = pytest.importorskip('PIL.Image')
    p = tmp_path / 'plain.tif'
    PIL.fromarray(np.arange(6, dtype=np.float32).reshape(2, 3)).save(p)
    with pytest.raises(ValueError, match='band'):
        L._read_tiff_fallback(str(p), band=2)
    with pytest.warns(UserWarning):
        assert L._read_tiff_fallback(str(p), band=1)['grid'][0, 0] == 3


def test_read_geotiff_without_rasterio_uses_fallback_band_check(tmp_path, monkeypatch):
    pytest.importorskip('PIL.Image')
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'rasterio' or name.startswith('rasterio.'):
            raise ImportError('rasterio hidden for this test')
        return real_import(name, *args, **kwargs)

    from PIL import Image
    p = tmp_path / 'plain.tif'
    Image.fromarray(np.arange(6, dtype=np.float32).reshape(2, 3)).save(p)
    monkeypatch.setattr(builtins, '__import__', fake_import)
    with pytest.raises(ValueError, match='band'):
        L.read_geotiff(str(p), band=2)


# ─── Numeric column fallback (io-08) ────────────────────────────────────────

def test_numeric_fallback_picks_data_column_not_station_id(tmp_path, rng):
    lines = ['station,easting,northing,k_pct']
    for i in range(200):
        lines.append(f'{1000 + i},{500000 + rng.uniform(0, 1000):.1f},'
                     f'{4100000 + rng.uniform(0, 1000):.1f},{rng.uniform(0.5, 3):.3f}')
    p = _write(tmp_path, 'radiometric.csv', '\n'.join(lines) + '\n')
    lyr = _quiet_load(p)
    assert lyr['columns'] == ('easting', 'northing', 'k_pct')
    assert 0.4 < np.nanmin(lyr['grid']) and np.nanmax(lyr['grid']) < 3.1


def test_numeric_fallback_skips_id_like_column():
    n = 30
    r = np.random.default_rng(3)
    df = pd.DataFrame({'easting': r.uniform(0, 10, n), 'northing': r.uniform(0, 10, n),
                       'station': np.arange(100, 100 + n), 'k': r.uniform(0, 1, n)})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert L._find_column(df, None, ['z'], fallback_index=2,
                              exclude=('easting', 'northing')) == 'k'
    # If every candidate is id-like, the positional choice is kept
    df2 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert L._find_column(df2, None, ['z'], fallback_index=2) == 'c'


# ─── Rotated grids keep their holes (io-09) ─────────────────────────────────

def test_rotated_grid_blank_block_stays_blank(tmp_path):
    ne, nv, d, rot = 40, 30, 25.0, 30.0
    data = np.full((nv, ne), 50.0, np.float32)
    data[10:20, 15:25] = -1.0e32
    p = write_geosoft(tmp_path / 'rotated.grd', data, x0=500000.0, y0=4100000.0,
                      dx=d, dy=d, rot=rot)
    with pytest.warns(UserWarning, match='Rotated'):
        lyr = L.load_grid(p)
    # Map every output node back to source (row, col)
    th = np.radians(rot)
    X, Y = np.meshgrid(lyr['x'], lyr['y'])
    de, dn = X - 500000.0, Y - 4100000.0
    col = (de * np.cos(th) + dn * np.sin(th)) / d
    row = (-de * np.sin(th) + dn * np.cos(th)) / d
    in_block = (row > 9) & (row < 20) & (col > 14) & (col < 25)
    assert in_block.sum() > 50
    assert np.isnan(lyr['grid'][in_block]).all()
    far = (row > 1) & (row < nv - 2) & (col > 1) & (col < ne - 2) & \
          ~((row > 8) & (row < 21) & (col > 13) & (col < 26))
    np.testing.assert_allclose(lyr['grid'][far], 50.0)
    np.testing.assert_allclose(lyr['grid'][np.isfinite(lyr['grid'])], 50.0)


# ─── GDAL scale/offset and missing georeferencing (io-10) ───────────────────

@requires_rasterio
def test_gdal_scale_and_offset_are_applied(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    p = tmp_path / 'packed_tmi.tif'
    codes = np.array([[4123, -32768], [0, -100]], dtype=np.int16)
    with rasterio.open(p, 'w', driver='GTiff', width=2, height=2, count=1, dtype='int16',
                       transform=from_origin(500000, 4100000, 10, 10), nodata=-32768) as dst:
        dst.write(codes, 1)
        dst.scales = (0.01,)
        dst.offsets = (48000.0,)
    lyr = L.load_grid(str(p))
    expected = (codes * 0.01 + 48000.0)[::-1].astype(float)
    expected[1, 1] = np.nan
    np.testing.assert_allclose(lyr['grid'], expected, equal_nan=True)


@requires_rasterio
def test_gdal_without_georeferencing_warns_and_uses_north_up_pixels(tmp_path):
    pytest.importorskip('PIL')
    import rasterio
    p = tmp_path / 'plain.tif'
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with rasterio.open(p, 'w', driver='GTiff', width=3, height=2, count=1,
                           dtype='float32') as dst:
            dst.write(data, 1)
    with pytest.warns(UserWarning, match='georeferencing'):
        a = L.read_raster_gdal(str(p))
    with pytest.warns(UserWarning):
        b = L._read_tiff_fallback(str(p))
    assert a['crs'] == 'pixel'
    np.testing.assert_allclose(a['grid'], b['grid'])
    np.testing.assert_allclose(a['x'], b['x'])
    np.testing.assert_allclose(a['y'], b['y'])
    assert a['grid'][0, 0] == 3                                 # bottom image row = south


# ─── Delimiters and decimal commas (io-11) ──────────────────────────────────

def test_semicolon_csv_with_decimal_commas(tmp_path):
    p = _write(tmp_path, 'excel_de.csv', 'X;Y;TMI\r\n' + ''.join(
        f'{500000 + 10 * i},5;{4100000 + 10 * j},25;{48000 + i + j},7\r\n'
        for j in range(5) for i in range(6)))
    for kw in ({}, {'delimiter': ';'}, {'delimiter': ';', 'decimal': ','}):
        lyr = _quiet_load(p, **kw)
        assert lyr['format'] == 'xyz_lattice' and lyr['grid'].shape == (5, 6)
        np.testing.assert_allclose(lyr['x'], 500000.5 + 10 * np.arange(6))
        np.testing.assert_allclose(lyr['y'], 4100000.25 + 10 * np.arange(5))
        np.testing.assert_allclose(lyr['grid'], 48000.7 + np.add.outer(np.arange(5), np.arange(6)))


def test_semicolon_wells_with_commas_in_notes(tmp_path):
    p = _write(tmp_path, 'wells.csv', 'X;Y;LABEL;NOTES\n500100;4100100;1;near road, north side\n'
                                      '500200;4100200;0;dry\n500300;4100300;1;ok\n')
    df = L.load_well_data(p)
    assert df['label'].tolist() == [1, 0, 1]
    assert df['notes'].tolist() == ['near road, north side', 'dry', 'ok']
    np.testing.assert_allclose(df['x'], [500100, 500200, 500300])


def test_tab_file_with_decimal_commas_and_explicit_decimal(tmp_path):
    body = ''.join(f'{i},5\t{j},25\t{i * j},5\n' for j in range(4) for i in range(4))
    p = _write(tmp_path, 'tab.txt', 'x\ty\tz\n' + body)
    lyr = _quiet_load(p)
    np.testing.assert_allclose(lyr['x'], 0.5 + np.arange(4))
    np.testing.assert_allclose(lyr['grid'], np.outer(np.arange(4), np.arange(4)) + 0.5)
    q = _write(tmp_path, 'dot.csv', 'x;y;z\n' + ''.join(
        f'{i}.5;{j}.25;{i + j}\n' for j in range(4) for i in range(4)))
    np.testing.assert_allclose(_quiet_load(q, decimal='.')['x'], 0.5 + np.arange(4))
    np.testing.assert_allclose(_quiet_load(q)['x'], 0.5 + np.arange(4))


@pytest.mark.parametrize('lines,expected', [
    (['X;Y;TMI', '500000,5;4100000,25;48000,7', '500010,5;4100000,25;48001,7'], ';'),
    (['x,y,z', '1,2,3', '4,5,6'], ','),
    (['x\ty\tz', '1\t2\t3'], '\t'),
    (['x y z', '1 2 3', '  4   5 6'], r'\s+'),
    (['x;y;label;notes', '1;2;1;a, b', '3;4;0;c'], ';'),
    (['a,b,c', '1,2,3', '4,5', '6,7,8'], ','),                   # ragged: majority wins
    (['x,y,"n;o;t;e"', '1,2,"a;b;c;d"'], ','),                   # quoted ';' ignored
])
def test_delimiter_sniffing_by_field_count(lines, expected):
    assert L._sniff_delimiter_lines(lines) == expected


# ─── Geosoft / Oasis montaj XYZ (io-12) ─────────────────────────────────────

def test_oasis_xyz_export_loads_as_lattice(tmp_path):
    rows = [(500000 + 10 * i, 4100000 + 10 * j, 48000 + i + j) for j in range(5) for i in range(6)]
    txt = ('/ ------------------------------------------\n/ XYZ EXPORT [03/25/2021]\n'
           '/ DATABASE  [mag.gdb]\n/ ------------------------------------------\n/\n'
           '/        X           Y        MAG\n/\nLine 10\n'
           + ''.join(f'  {a:.2f}  {b:.2f}  {c:.3f}\n' for a, b, c in rows[:15]) + 'Line 20\n'
           + ''.join(f'  {a:.2f}  {b:.2f}  {c:.3f}\n' for a, b, c in rows[15:]))
    p = _write(tmp_path, 'oasis_export.xyz', txt)
    for kw in ({}, {'x_col': 'x', 'y_col': 'y', 'z_col': 'mag'}):
        lyr = _quiet_load(p, **kw)
        assert lyr['format'] == 'xyz_lattice'
        assert lyr['columns'] == ('x', 'y', 'mag')
        assert lyr['grid'].shape == (5, 6)
        np.testing.assert_allclose(lyr['grid'], 48000 + np.add.outer(np.arange(5), np.arange(6)))


GEOSOFT_SPEC_EXAMPLE = """/ GeoAI synthetic airborne magnetic survey
/        X            Y          MAG     FID
Line 1010
  400000.00   4200000.00   48231.45      1
  400000.00   4200025.00   48230.91      2
Tie 9010
  400000.00   4200000.00          *      1
"""


def test_geosoft_xyz_table_columns_lines_and_dummies(tmp_path):
    df = L._read_table(_write(tmp_path, 'survey.xyz', GEOSOFT_SPEC_EXAMPLE))
    assert list(df.columns) == ['x', 'y', 'mag', 'fid', 'line']
    assert df['line'].tolist() == ['1010', '1010', '9010']
    np.testing.assert_allclose(df['mag'], [48231.45, 48230.91, np.nan])
    np.testing.assert_allclose(df['y'], [4200000, 4200025, 4200000])
    assert df['fid'].tolist() == [1, 2, 1]


def test_geosoft_xyz_case_insensitive_records_and_rows_before_first_line(tmp_path):
    txt = ('/ X Y Z\n1 2 3\nLINE L5\n4 5 6\ntrend T1\n7 8 9\n')
    df = L._read_table(_write(tmp_path, 'g.xyz', txt))
    assert df[['x', 'y', 'z']].to_numpy().tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert df['line'].isna().tolist() == [True, False, False]
    assert df['line'].tolist()[1:] == ['L5', 'T1']


def test_geosoft_header_uses_last_matching_comment(tmp_path):
    txt = ('/ survey of 03 2021\n/ EAST NORTH TMI\n/ note\nLine 1\n1 2 3\n4 5 6\n')
    df = L._read_table(_write(tmp_path, 'g.xyz', txt))
    assert list(df.columns) == ['east', 'north', 'tmi', 'line']
    # No comment with one token per field: default names
    df = L._read_table(_write(tmp_path, 'h.xyz', '/ title\n/ survey 2021\nLine 1\n1 2 3\n'))
    assert list(df.columns) == ['x', 'y', 'z', 'line']


def test_geosoft_xyz_well_file(tmp_path):
    txt = ('/ drillhole collars\n/  X  Y  LABEL  DEPTH\nLine 1\n'
           '500100 4100100 1 120\n500200 4100200 0 *\n')
    df = L.load_well_data(_write(tmp_path, 'collars.xyz', txt))
    assert df['label'].tolist() == [1, 0]
    np.testing.assert_allclose(df['depth'], [120, np.nan])


# ─── Native netCDF-3 reader ─────────────────────────────────────────────────

def _nc_write(path, build, version=1):
    from scipy.io import netcdf_file
    f = netcdf_file(str(path), 'w', version=version)
    try:
        build(f)
    finally:
        f.close()
    return str(path)


def _cf_builder(x, y, data, dims=('y', 'x'), name='z', dtype='f4', **attrs):
    def build(f):
        f.createDimension('x', len(x))
        f.createDimension('y', len(y))
        vx = f.createVariable('x', 'f8', ('x',))
        vx[:] = x
        vy = f.createVariable('y', 'f8', ('y',))
        vy[:] = y
        v = f.createVariable(name, dtype, dims)
        v[:] = data
        for k, val in attrs.items():
            setattr(v, k, val)
    return build


def test_netcdf_cf_packed_descending_y_and_fill_values(tmp_path):
    x = 1000.0 + 10 * np.arange(4)
    y = 2040.0 - 20 * np.arange(3)                          # descending (north first)
    codes = np.array([[0, 1, 2, 3], [4, -1, 6, 7], [8, 9, -2, 11]], dtype=np.int16)
    p = _nc_write(tmp_path / 'packed.grd',
                  _cf_builder(x, y, codes, dtype='i2', scale_factor=np.float32(0.5),
                              add_offset=np.float32(100.0), _FillValue=np.int16(-1),
                              missing_value=np.int16(-2)))
    assert L.detect_format(p) == 'netcdf'
    lyr = L.load_grid(p)
    assert lyr['format'] == 'netcdf' and lyr['variable'] == 'z'
    expected = (codes * 0.5 + 100.0).astype(float)
    expected[codes < 0] = np.nan
    np.testing.assert_allclose(lyr['grid'], expected[::-1], equal_nan=True)
    np.testing.assert_allclose(lyr['x'], x)
    np.testing.assert_allclose(lyr['y'], y[::-1])


def test_netcdf_float_nan_default_fill_and_transposed_dims(tmp_path):
    x = np.array([0.0, 5.0, 10.0])
    y = np.array([0.0, 2.0])
    g = np.array([[1.0, np.nan, 3.0], [4.0, 9.969209968386869e36, 6.0]], dtype=np.float32)
    p = _nc_write(tmp_path / 'xy.nc', _cf_builder(x, y, g.T, dims=('x', 'y')))
    assert L.detect_format(p) == 'netcdf'
    lyr = L.load_grid(p)
    expected = g.astype(float)
    expected[1, 1] = np.nan
    np.testing.assert_allclose(lyr['grid'], expected, equal_nan=True)
    np.testing.assert_allclose(lyr['x'], x)
    np.testing.assert_allclose(lyr['y'], y)


def test_netcdf_variable_selection_and_bands(tmp_path):
    x, y = np.arange(3.0), np.arange(2.0)

    def build(f):
        f.createDimension('x', 3)
        f.createDimension('y', 2)
        f.createDimension('time', 2)
        f.createVariable('x', 'f8', ('x',))[:] = x
        f.createVariable('y', 'f8', ('y',))[:] = y
        f.createVariable('quality', 'f4', ('y', 'x'))[:] = np.full((2, 3), 7.0)
        f.createVariable('Elevation', 'f4', ('y', 'x'))[:] = np.arange(6.0).reshape(2, 3)
        f.createVariable('series', 'f4', ('time', 'y', 'x'))[:] = \
            np.stack([np.zeros((2, 3)), np.ones((2, 3))])

    p = _nc_write(tmp_path / 'multi.nc', build, version=2)
    with open(p, 'rb') as fh:
        assert fh.read(4) == b'CDF\x02'
    lyr = L.load_grid(p)
    assert lyr['variable'] == 'Elevation'                    # preferred name
    np.testing.assert_allclose(lyr['grid'], np.arange(6.0).reshape(2, 3))
    assert L.load_grid(p, variable='quality')['grid'][0, 0] == 7.0
    assert L.read_netcdf_grid(p, variable='series', band=2)['grid'][0, 0] == 1.0
    assert L.load_grid(p, variable='series')['grid'][0, 0] == 0.0
    with pytest.raises(ValueError, match='band'):
        L.load_grid(p, variable='series', band=3)
    with pytest.raises(ValueError, match='not found'):
        L.load_grid(p, variable='nope')


def test_netcdf_first_variable_with_coordinates_and_lonlat_crs(tmp_path):
    def build(f):
        f.createDimension('lon', 3)
        f.createDimension('lat', 2)
        f.createDimension('row', 2)
        f.createDimension('col', 3)
        vlon = f.createVariable('lon', 'f8', ('lon',))
        vlon[:] = [30.0, 30.5, 31.0]
        vlon.units = 'degrees_east'
        vlat = f.createVariable('lat', 'f8', ('lat',))
        vlat[:] = [38.0, 38.5]
        vlat.units = 'degrees_north'
        f.createVariable('index_grid', 'f4', ('row', 'col'))[:] = np.zeros((2, 3))
        f.createVariable('gravity', 'f4', ('lat', 'lon'))[:] = [[1, 2, 3], [4, 5, 6]]

    p = _nc_write(tmp_path / 'geo.nc', build)
    lyr = L.load_grid(p)
    assert lyr['variable'] == 'gravity'
    assert lyr['crs'] == 'EPSG:4326'
    np.testing.assert_allclose(lyr['x'], [30.0, 30.5, 31.0])
    np.testing.assert_allclose(lyr['grid'], [[1, 2, 3], [4, 5, 6]])
    with pytest.warns(UserWarning, match='pixel coordinates'):
        lyr2 = L.load_grid(p, variable='index_grid')
    assert lyr2['crs'] == 'pixel'


@pytest.mark.parametrize('node_offset', [0, 1])
def test_netcdf_legacy_gmt_registration(tmp_path, node_offset):
    nx, ny = 4, 3
    north_first = np.arange(12, dtype=np.float32).reshape(ny, nx)
    north_first[0, 1] = np.nan

    def build(f):
        f.createDimension('side', 2)
        f.createDimension('xysize', nx * ny)
        f.createVariable('x_range', 'f8', ('side',))[:] = [0.0, 40.0]
        f.createVariable('y_range', 'f8', ('side',))[:] = [100.0, 130.0]
        f.createVariable('z_range', 'f8', ('side',))[:] = [0.0, 11.0]
        f.createVariable('spacing', 'f8', ('side',))[:] = \
            [10.0, 10.0] if node_offset else [40 / 3, 15.0]
        f.createVariable('dimension', 'i4', ('side',))[:] = [nx, ny]
        z = f.createVariable('z', 'f4', ('xysize',))
        z[:] = north_first.ravel()
        z.node_offset = np.int32(node_offset)

    p = _nc_write(tmp_path / 'legacy.grd', build)
    lyr = L.load_grid(p)
    assert lyr['format'] == 'netcdf'
    assert lyr['registration'] == ('pixel' if node_offset else 'gridline')
    np.testing.assert_allclose(lyr['grid'], north_first[::-1].astype(float), equal_nan=True)
    if node_offset:
        np.testing.assert_allclose(lyr['x'], [5, 15, 25, 35])
        np.testing.assert_allclose(lyr['y'], [105, 115, 125])
    else:
        np.testing.assert_allclose(lyr['x'], np.linspace(0, 40, 4))
        np.testing.assert_allclose(lyr['y'], [100, 115, 130])


@requires_rasterio
def test_netcdf_written_by_gdal_matches_geotiff(tmp_path):
    import rasterio
    from rasterio.shutil import copy as rio_copy
    from rasterio.transform import from_origin
    g = np.arange(12, dtype=np.float32).reshape(3, 4)
    g[1, 2] = -9999
    src = tmp_path / 'src.tif'
    with rasterio.open(src, 'w', driver='GTiff', width=4, height=3, count=1, dtype='float32',
                       crs='EPSG:32635', transform=from_origin(500000, 4100030, 10, 10),
                       nodata=-9999) as d:
        d.write(g, 1)
    nc = tmp_path / 'gdal.nc'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rio_copy(str(src), str(nc), driver='netCDF')
    assert L.detect_format(str(nc)) == 'netcdf'
    a, b = L.load_grid(str(nc)), L.load_grid(str(src))
    assert a['format'] == 'netcdf' and a['variable'] == 'Band1'
    assert a['crs'] == b['crs'] == 'EPSG:32635'
    np.testing.assert_allclose(a['x'], b['x'])
    np.testing.assert_allclose(a['y'], b['y'])
    np.testing.assert_allclose(a['grid'], b['grid'], equal_nan=True)


@requires_rasterio
def test_netcdf4_hdf5_goes_through_gdal_with_variable(tmp_path):
    import rasterio
    from rasterio.shutil import copy as rio_copy
    from rasterio.transform import from_origin
    src = tmp_path / 'src.tif'
    with rasterio.open(src, 'w', driver='GTiff', width=4, height=3, count=1, dtype='float32',
                       crs='EPSG:32635', transform=from_origin(500000, 4100030, 10, 10)) as d:
        d.write(np.arange(12, dtype=np.float32).reshape(3, 4), 1)
    nc4 = tmp_path / 'g4.nc'
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rio_copy(str(src), str(nc4), driver='netCDF', FORMAT='NC4')
    except Exception as e:      # GDAL built without netCDF-4 support
        pytest.skip(f'GDAL cannot write netCDF-4: {e}')
    assert L.detect_format(str(nc4)) == 'gdal'
    grd = tmp_path / 'g4.grd'
    grd.write_bytes(nc4.read_bytes())
    assert L.detect_format(str(grd)) == 'gdal'
    lyr = L.load_grid(str(nc4), variable='Band1')
    np.testing.assert_allclose(lyr['grid'][0], [8, 9, 10, 11])
    np.testing.assert_allclose(lyr['x'], 500005 + 10 * np.arange(4))
    with pytest.raises(ValueError, match='netCDF-3'):
        L.read_netcdf_grid(str(nc4))


def test_detect_format_netcdf_magic(tmp_path):
    for name, magic, expected in (('a.grd', b'CDF\x01', 'netcdf'), ('b.grd', b'CDF\x02', 'netcdf'),
                                  ('c.nc', b'CDF\x01', 'netcdf'), ('d.nc', b'\x89HDF\r\n\x1a\n', 'gdal'),
                                  ('e.grd', b'\x89HDF\r\n\x1a\n', 'gdal')):
        (tmp_path / name).write_bytes(magic + b'\x00' * 600)
        assert L.detect_format(str(tmp_path / name)) == expected


def test_variable_kwarg_ignored_for_other_formats(tmp_path):
    p = W.save_surfer_ascii_grd(np.ones((2, 2)), [0, 1], [0, 1], tmp_path / 's.grd')
    with pytest.warns(UserWarning, match="'variable' is ignored"):
        L.load_grid(p, variable='z')


# ─── Geosoft GRD writer ─────────────────────────────────────────────────────

def test_geosoft_writer_roundtrip(tmp_path, rng):
    g = rng.normal(size=(7, 9)) * 50 + 1000
    g[2, 3] = np.nan
    g[6, 8] = np.nan
    x = 500000 + 25.0 * np.arange(9)
    y = 4100000 + 40.0 * np.arange(7)
    p = W.save_geosoft_grd(g, x, y, tmp_path / 'g.grd')
    assert L.detect_format(p) == 'geosoft_grd'
    lyr = L.read_geosoft_grd(p)
    np.testing.assert_allclose(lyr['grid'], g.astype(np.float32), rtol=0, equal_nan=True)
    np.testing.assert_allclose(lyr['x'], x)
    np.testing.assert_allclose(lyr['y'], y)
    raw = open(p, 'rb').read()
    assert len(raw) == 512 + 4 * g.size
    assert struct.unpack_from('<5i', raw, 0) == (4, 2, 9, 7, 1)
    assert struct.unpack_from('<7d', raw, 20) == (25.0, 40.0, x[0], y[0], 0.0, 0.0, 1.0)
    valid = g[np.isfinite(g)].astype(np.float32).astype(float)
    nvpts, zmin, zmax, zmed, zmean = struct.unpack_from('<i4f', raw, 156)
    assert nvpts == valid.size
    assert (zmin, zmax) == (np.float32(valid.min()), np.float32(valid.max()))
    assert zmed == pytest.approx(np.median(valid), rel=1e-6)
    assert zmean == pytest.approx(valid.mean(), rel=1e-6)
    assert struct.unpack_from('<d', raw, 176)[0] == pytest.approx(valid.var(ddof=1), rel=1e-12)
    dummy = np.frombuffer(raw, '<f4', offset=512).reshape(7, 9)[2, 3]
    assert dummy == np.float32(-1e32)


def test_geosoft_writer_matches_oasis_montaj_layout():
    """The writer reproduces the header layout and data of a real Oasis montaj float grid."""
    import tempfile
    exp = np.load(os.path.join(GEOSOFT_DIR, 'om_expected.npz'))
    ref = open(os.path.join(GEOSOFT_DIR, 'om_float.grd'), 'rb').read()
    with tempfile.TemporaryDirectory() as td:
        p = W.save_geosoft_grd(exp['grid'], exp['easting'], exp['northing'],
                               os.path.join(td, 'om.grd'))
        out = open(p, 'rb').read()
    assert len(out) == len(ref)
    assert out[:76] == ref[:76]                      # ES SF NE NV KX DE DV X0 Y0 ROT ZBASE ZMULT
    assert struct.unpack_from('<5i', out, 0) == (4, 2, 50, 49, 1)
    assert out[512:] == ref[512:]                    # float32 data, dummies, row order
    assert struct.unpack_from('<i', out, 156) == struct.unpack_from('<i', ref, 156)
    o, r = struct.unpack_from('<4f', out, 160), struct.unpack_from('<4f', ref, 160)
    assert (o[0], o[1], o[3]) == (r[0], r[1], r[3])  # IZMIN, IZMAX, IZMEA (Oasis leaves IZMED 0)
    assert struct.unpack_from('<d', out, 176)[0] == pytest.approx(
        struct.unpack_from('<d', ref, 176)[0], rel=1e-12)


def test_save_grid_geosoft_format(tmp_path):
    g = np.arange(6, dtype=float).reshape(2, 3)
    p = W.save_grid(g, [0, 1, 2], [0, 5], tmp_path / 'a.grd', fmt='geosoft')
    assert L.detect_format(p) == 'geosoft_grd'
    np.testing.assert_allclose(L.load_grid(p)['grid'], g)
    # A plain '.grd' is still Surfer ASCII
    assert L.detect_format(W.save_grid(g, [0, 1, 2], [0, 5], tmp_path / 'b.grd')) == 'surfer_ascii'


def test_geosoft_writer_all_nan_grid(tmp_path):
    p = W.save_geosoft_grd(np.full((2, 2), np.nan), [0, 1], [0, 1], tmp_path / 'n.grd')
    raw = open(p, 'rb').read()
    assert struct.unpack_from('<i', raw, 156)[0] == 0
    assert np.isnan(L.read_geosoft_grd(p)['grid']).all()
