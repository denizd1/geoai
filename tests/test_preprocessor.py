"""Co-registration, NaN doldurma, katman tipi, feature ve normalizasyon testleri."""
import numpy as np
import pytest

from geoai.core.preprocessor import (
    LAYER_FILTER_PROFILES, CoRegistrar, GeophysicalFilters, RobustGridNormalizer,
    detect_layer_type, extract_features, feature_layer_type, fill_nans,
)


def _layer(x, y, grid):
    return {'x': np.asarray(x, float), 'y': np.asarray(y, float), 'grid': np.asarray(grid, float)}


# ─── fill_nans ──────────────────────────────────────────────────────────────

def test_fill_nans_nearest_and_linear():
    g = np.arange(25, dtype=float).reshape(5, 5)
    h = g.copy()
    h[2, 2] = np.nan
    np.testing.assert_allclose(fill_nans(h, 'linear')[2, 2], 12.0)
    assert fill_nans(h, 'nearest')[2, 2] in (7.0, 11.0, 13.0, 17.0)

    # Seyrek örneklenmiş düzlem → 'auto' lineer seçer ve düzlemi tam üretir
    Y, X = np.mgrid[0:30, 0:30]
    plane = 3.0 * X - 2.0 * Y + 1
    sparse = np.full(plane.shape, np.nan)
    sparse[::6, ::6] = plane[::6, ::6]
    out = fill_nans(sparse, 'auto')
    np.testing.assert_allclose(out[:25, :25], plane[:25, :25], atol=1e-9)


def test_fill_nans_no_op_cases():
    g = np.ones((3, 3))
    np.testing.assert_array_equal(fill_nans(g), g)
    allnan = np.full((2, 2), np.nan)
    assert np.isnan(fill_nans(allnan)).all()


# ─── CoRegistrar ────────────────────────────────────────────────────────────

def test_auto_resolution_uses_finest_cell_size():
    coarse = _layer(np.linspace(0, 1000, 11), np.linspace(0, 1000, 11), np.zeros((11, 11)))
    fine = _layer(np.linspace(0, 1000, 41), np.linspace(0, 1000, 21), np.zeros((21, 41)))
    cr = CoRegistrar().fit({'a': coarse, 'b': fine})
    assert (cr.ref_nx, cr.ref_ny) == (41, 21)
    assert cr.dx == pytest.approx(25.0) and cr.dy == pytest.approx(50.0)


def test_linear_field_is_resampled_exactly():
    x = np.linspace(0, 1000, 51)
    y = np.linspace(0, 800, 41)
    X, Y = np.meshgrid(x, y)
    lyr = _layer(x, y, 0.5 * X + 2 * Y)
    cr = CoRegistrar().fit({'a': lyr}, target_nx=37, target_ny=23)
    out = cr.transform_layer(lyr)
    RX, RY = np.meshgrid(cr.ref_x, cr.ref_y)
    np.testing.assert_allclose(out, 0.5 * RX + 2 * RY, rtol=1e-12, atol=1e-9)


def test_intersection_extent_and_union_masks():
    a = _layer(np.linspace(0, 100, 11), np.linspace(0, 100, 11), np.ones((11, 11)))
    b = _layer(np.linspace(50, 150, 11), np.linspace(0, 100, 11), 2 * np.ones((11, 11)))
    cr = CoRegistrar().fit({'a': a, 'b': b}, target_nx=11, target_ny=11,
                           common_extent='intersection')
    assert cr.ref_x[0] == 50 and cr.ref_x[-1] == 100

    cr = CoRegistrar().fit({'a': a, 'b': b}, target_nx=16, target_ny=11, common_extent='union')
    reg = cr.transform_all({'a': a, 'b': b})
    assert cr.ref_x[0] == 0 and cr.ref_x[-1] == 150
    # 'a' yalnızca x ≤ 100'ü kapsar: dışı NaN olmalı (eski sürüm kenarı tekrarlıyordu)
    assert np.isnan(reg['a'][:, cr.ref_x > 100]).all()
    assert np.isfinite(reg['a'][:, cr.ref_x <= 100]).all()
    assert cr.masks['a'][:, cr.ref_x > 100].all()
    assert not cr.masks['b'][:, cr.ref_x >= 50].any()


def test_nodata_holes_are_filled_but_masked():
    g = np.ones((11, 11))
    g[4:7, 4:7] = np.nan
    lyr = _layer(np.arange(11), np.arange(11), g)
    cr = CoRegistrar().fit({'a': lyr}, target_nx=11, target_ny=11)
    out, mask = cr.transform_layer(lyr, return_mask=True)
    assert np.isfinite(out).all()
    assert mask[4:7, 4:7].all() and mask.sum() == 9


def test_no_overlap_raises():
    a = _layer([0, 1], [0, 1], np.zeros((2, 2)))
    b = _layer([5, 6], [0, 1], np.zeros((2, 2)))
    with pytest.raises(ValueError, match='ortak alan'):
        CoRegistrar().fit({'a': a, 'b': b})


def test_max_pixels_cap_warns():
    a = _layer(np.linspace(0, 1, 3), np.linspace(0, 1, 3), np.zeros((3, 3)))
    with pytest.warns(UserWarning, match='kırpılıyor'):
        cr = CoRegistrar().fit({'a': a}, target_nx=5000, target_ny=10)
    assert cr.ref_nx == 2000


def test_transform_points_nearest_cell():
    a = _layer(np.linspace(0, 10, 11), np.linspace(0, 10, 11), np.zeros((11, 11)))
    cr = CoRegistrar().fit({'a': a}, target_nx=11, target_ny=11)
    rows, cols, inside = cr.transform_points([3.4, 3.6, 10.4, 10.6], [0.2, 9.9, 5, 5],
                                             return_mask=True)
    assert cols.tolist() == [3, 4, 10, 10]     # en yakın hücre (eski: searchsorted tavanı)
    assert rows.tolist() == [0, 10, 5, 5]
    assert inside.tolist() == [True, True, True, False]
    x, y = cr.index_to_coords(rows, cols)
    assert x.tolist() == [3, 4, 10, 10]


# ─── Katman tipi ────────────────────────────────────────────────────────────

@pytest.mark.parametrize('name,typ', [
    ('magnetic_tmi', 'magnetic'), ('mag_rtp', 'magnetic'), ('TMI', 'magnetic'),
    ('gravity_bouguer', 'gravity'), ('cba_2.67', 'gravity'),
    ('resistivity', 'resistivity'), ('dipole_dipole_res', 'resistivity'),
    ('ip_chargeability', 'ip'), ('chargeability_mrad', 'ip'),
    ('geochemistry_cu', 'geochemistry'), ('Cu_ppm', 'geochemistry'), ('soil_au', 'geochemistry'),
    ('em_conductivity', 'conductivity'), ('srtm_dem', 'topography'),
    ('radiometric_k', 'radiometric'), ('lithology', 'geology'), ('band7', 'default'),
    ('pipeline_route', 'default'), ('shipping', 'default'),
])
def test_detect_layer_type(name, typ):
    assert detect_layer_type(name) == typ


def test_feature_layer_type():
    assert feature_layer_type('mag_rtp__tilt') == 'magnetic'


# ─── Feature extraction ─────────────────────────────────────────────────────

def test_extract_features_profiles_shapes_and_types(rng):
    ny, nx = 40, 50
    layers = {'magnetic_tmi': rng.normal(size=(ny, nx)), 'geochemistry_cu': rng.normal(size=(ny, nx))}
    stack, names, maps = extract_features(layers, dx=100, dy=100, verbose=False)
    n_mag = len(LAYER_FILTER_PROFILES['magnetic']) + 2          # 'stats' → 3 feature
    n_geo = len(LAYER_FILTER_PROFILES['geochemistry']) + 2
    assert stack.shape == (ny, nx, n_mag + n_geo)
    assert stack.dtype == np.float32
    assert len(names) == len(set(names)) == stack.shape[-1]
    assert 'magnetic_tmi__tilt' in names and 'geochemistry_cu__log' in names
    assert np.isfinite(stack).all()
    tilt = maps['magnetic_tmi__tilt']
    assert tilt.min() >= -np.pi / 2 - 1e-6 and tilt.max() <= np.pi / 2 + 1e-6


def test_extract_features_profile_override_and_unknown_filter(rng):
    layers = {'x1': rng.normal(size=(10, 10))}
    _, names, _ = extract_features(layers, verbose=False, profiles={'x1': ['raw', 'log']})
    assert names == ['x1__raw', 'x1__log']
    with pytest.raises(ValueError):
        extract_features(layers, verbose=False, profiles={'x1': ['nope']})


def test_extract_features_fills_nan(rng):
    g = rng.normal(size=(20, 20))
    g[5:8, 5:8] = np.nan
    stack, _, _ = extract_features({'res': g}, verbose=False)
    assert np.isfinite(stack).all()


# ─── Normalizer ─────────────────────────────────────────────────────────────

def test_normalizer_median_iqr_and_clip(rng):
    X = rng.normal(loc=50, scale=10, size=(10000, 2)).astype(np.float32)
    X[0, 0] = 1e6                                    # uç değer
    Z = RobustGridNormalizer().fit_transform(X)
    assert abs(np.median(Z[:, 0])) < 0.02
    q25, q75 = np.percentile(Z[:, 1], [25, 75])
    assert q75 - q25 == pytest.approx(1.0, abs=0.03)
    assert Z.max() <= 5 and Z.min() >= -5


def test_normalizer_constant_and_degenerate_features():
    X = np.zeros((100, 2), dtype=np.float32)
    X[:95, 1] = 3.0
    X[95:, 1] = 10.0                                 # IQR = 0 ama varyans > 0
    Z = RobustGridNormalizer().fit_transform(X)
    assert np.isfinite(Z).all()
    assert np.all(Z[:, 0] == 0)
    assert Z[95:, 1].max() < 5                       # 1e-10 bölmesiyle ±5'e sıkışmamalı


def test_normalizer_preserves_nan_and_checks_shape(rng):
    X = rng.normal(size=(5, 5, 3)).astype(np.float32)
    X[0, 0, 0] = np.nan
    norm = RobustGridNormalizer().fit(X.reshape(-1, 3)[1:])
    Z = norm.transform(X)
    assert Z.shape == X.shape and np.isnan(Z[0, 0, 0])
    with pytest.raises(ValueError):
        norm.transform(np.zeros((4, 2), dtype=np.float32))


def test_stat_window_numerically_stable_for_large_offsets(rng):
    g = 48000.0 + rng.normal(size=(30, 30))          # TMI benzeri büyük taban değer
    m, s, r = GeophysicalFilters.stat_window(g, window=5)
    i, j = 15, 15
    win = g[i - 2:i + 3, j - 2:j + 3]
    assert m[i, j] == pytest.approx(win.mean(), rel=1e-12)
    assert s[i, j] == pytest.approx(win.std(), rel=1e-6)
    assert r[i, j] == pytest.approx(win.max() - win.min())
