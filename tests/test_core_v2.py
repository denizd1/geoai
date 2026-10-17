"""
Regression and feature tests for geoai.core.preprocessor (second review round):
plane removal in the Fourier filters, reduction to the pole, the soft-bounded
normalizer, layer type detection, the 'log' feature, float32 offsets,
geographic helpers and the co-registration coverage mask.
"""
import io
import contextlib
import warnings

import numpy as np
import pytest

from geoai.core.preprocessor import (
    CoRegistrar, GeophysicalFilters as F, RobustGridNormalizer, detect_layer_type,
    extract_features, looks_geographic, metric_cell_size,
)

# ─── Analytic models ────────────────────────────────────────────────────────

G = 6.674e-11
MASS = 1e12                       # kg
N = 201
DX = 100.0                        # m
x = (np.arange(N) - N // 2) * DX
X, Y = np.meshgrid(x, x)          # rows = ascending y (north), columns = x (east)
C = slice(50, 151)                # central region away from edge effects


def g_sphere(d, X=X, Y=Y):
    """Vertical gravity of a point mass at depth d below (0, 0) (mGal)."""
    return G * MASS * d / (X**2 + Y**2 + d**2) ** 1.5 * 1e5


def dgdz_sphere(d, X=X, Y=Y):
    """∂g/∂z of the point mass (z positive down), analytic."""
    r2 = X**2 + Y**2
    return G * MASS * (2 * d**2 - r2) / (r2 + d**2) ** 2.5 * 1e5


def _unit(inc_deg, dec_deg):
    """Unit vector (east, north, down) for an inclination / declination in degrees."""
    i, d = np.radians(inc_deg), np.radians(dec_deg)
    return np.array([np.cos(i) * np.sin(d), np.cos(i) * np.cos(d), np.sin(i)])


def dipole_tmi(X, Y, cx, cy, depth, inc, dec, m=1e10, mag_inc=None, mag_dec=None):
    """
    Total-field anomaly (nT) of a point dipole of moment m (A·m²) at depth
    `depth` below (cx, cy), measured on z = 0 along the field direction
    f_hat = (cosI sinD, cosI cosD, sinI) (z positive down):
        dT = 1e9 · 1e-7 · m · (3 (m̂·r̂)(f̂·r̂) − m̂·f̂) / r³,   r = (X−cx, Y−cy, −depth)
    For induced magnetisation (m̂ = f̂) this is 1e9·1e-7·m·(3 cos²(f,r) − 1) / r³.
    """
    f_hat = _unit(inc, dec)
    m_hat = f_hat if mag_inc is None else _unit(mag_inc, mag_dec)
    rx, ry, rz = X - cx, Y - cy, np.full(X.shape, -float(depth))
    r = np.sqrt(rx**2 + ry**2 + rz**2)
    cos_fr = (f_hat[0] * rx + f_hat[1] * ry + f_hat[2] * rz) / r
    cos_mr = (m_hat[0] * rx + m_hat[1] * ry + m_hat[2] * rz) / r
    return 1e9 * 1e-7 * m * (3 * cos_mr * cos_fr - m_hat @ f_hat) / r**3


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _layer(x, y, grid, **extra):
    d = {'x': np.asarray(x, float), 'y': np.asarray(y, float), 'grid': np.asarray(grid, float)}
    d.update(extra)
    return d


# ─── Fourier filters: first-order trend removal ─────────────────────────────

PLANE = 5.0 + 0.004 * X - 0.003 * Y          # regional gradient, 0.005 mGal/m


def test_vertical_derivative_of_plane_is_zero():
    # A plane is harmonic and has zero vertical derivative. Removing only the
    # mean (old code) gave max |FVD| ≈ 0.046 mGal/m, ~9× the plane's own gradient.
    for order in (1, 2):
        vd = F.vertical_derivative(PLANE, DX, DX, order=order)
        assert np.abs(vd).max() < 1e-12
    assert np.abs(F.vertical_derivative(PLANE, DX, DX, stabilize_height=150.0)).max() < 1e-12


def test_upward_continuation_and_rtp_leave_plane_unchanged():
    np.testing.assert_allclose(F.upward_continuation(PLANE, DX, DX, 500.0), PLANE, atol=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        rtp = F.reduce_to_pole(PLANE, DX, DX, 55.0, 20.0)
    np.testing.assert_allclose(rtp, PLANE, atol=1e-9)


def test_sphere_plus_plane_matches_analytic_fvd_tilt_and_upward():
    d = 700.0
    g = g_sphere(d) + PLANE
    ref = dgdz_sphere(d)

    fvd = F.first_vertical_derivative(g, DX, DX)
    assert np.abs(fvd - ref)[C, C].max() / np.abs(ref).max() < 2e-3   # old: 0.13
    assert np.abs(fvd - ref).max() / np.abs(ref).max() < 5e-3         # old: 1.18

    tilt_true = np.arctan2(ref, F.horizontal_gradient(g, DX, DX))
    err = np.degrees(np.abs(F.tilt_angle(g, DX, DX) - tilt_true))[C, C]
    assert np.median(err) < 0.5                                       # old: 15°
    assert err.max() < 3.0                                            # old: 91°

    up = F.upward_continuation(g, DX, DX, 300.0)
    up_ref = g_sphere(d + 300.0) + PLANE
    assert np.abs(up - up_ref)[C, C].max() / g_sphere(d).max() < 1e-3  # old: 0.11

    asa = F.analytic_signal(g, DX, DX)
    gx, gy = np.gradient(g, DX, axis=1), np.gradient(g, DX, axis=0)
    asa_ref = np.sqrt(gx**2 + gy**2 + ref**2)
    assert np.abs(asa - asa_ref)[C, C].max() / asa_ref.max() < 2e-3


# ─── Reduction to the pole ──────────────────────────────────────────────────

NM = 256
DXM = 50.0
xm = (np.arange(NM) - NM // 2) * DXM
XM, YM = np.meshgrid(xm, xm)
CM = slice(NM // 4, 3 * NM // 4)


def test_rtp_induced_dipole_equals_pole_dipole():
    tmi = dipole_tmi(XM, YM, 0.0, 0.0, 500.0, 55.0, 20.0)
    pole = dipole_tmi(XM, YM, 0.0, 0.0, 500.0, 90.0, 0.0)
    rtp = F.reduce_to_pole(tmi, DXM, DXM, inclination=55.0, declination=20.0)
    assert np.abs(rtp - pole)[CM, CM].max() < 0.01 * pole.max()
    # Peak moved over the source
    i, j = np.unravel_index(np.argmax(rtp), rtp.shape)
    assert (i, j) == (NM // 2, NM // 2)
    # Sign convention check: a wrong declination is far off
    wrong = F.reduce_to_pole(tmi, DXM, DXM, 55.0, -20.0)
    assert np.abs(wrong - pole)[CM, CM].max() > 0.1 * pole.max()


def test_rtp_offcentre_dipole_with_regional_and_anisotropic_cells():
    dy = 0.6 * DXM
    Xa, Ya = np.meshgrid(xm, (np.arange(NM) - NM // 2) * dy)
    regional = 48000.0 + 0.02 * Xa - 0.01 * Ya
    tmi = dipole_tmi(Xa, Ya, 1000.0, -700.0, 400.0, 55.0, 20.0) + regional
    pole = dipole_tmi(Xa, Ya, 1000.0, -700.0, 400.0, 90.0, 0.0)
    rtp = F.reduce_to_pole(tmi, DXM, dy, 55.0, 20.0) - regional
    assert np.abs(rtp - pole)[CM, CM].max() < 0.01 * pole.max()


def test_rtp_north_up_rows_with_negative_dy():
    tmi = dipole_tmi(XM, YM, 0.0, 0.0, 500.0, 55.0, 20.0)
    pole = dipole_tmi(XM, YM, 0.0, 0.0, 500.0, 90.0, 0.0)
    rtp = F.reduce_to_pole(tmi[::-1], DXM, -DXM, 55.0, 20.0)[::-1]
    assert np.abs(rtp - pole)[CM, CM].max() < 0.01 * pole.max()
    rtp = F.reduce_to_pole(tmi[:, ::-1], -DXM, DXM, 55.0, 20.0)[:, ::-1]   # columns east → west
    assert np.abs(rtp - pole)[CM, CM].max() < 0.01 * pole.max()


def test_rtp_remanent_magnetisation():
    tmi = dipole_tmi(XM, YM, 0.0, 0.0, 600.0, 60.0, 10.0, mag_inc=30.0, mag_dec=-40.0)
    pole = dipole_tmi(XM, YM, 0.0, 0.0, 600.0, 90.0, 0.0)
    rtp = F.reduce_to_pole(tmi, DXM, DXM, 60.0, 10.0, mag_inclination=30.0, mag_declination=-40.0)
    assert np.abs(rtp - pole)[CM, CM].max() < 0.01 * pole.max()
    # Assuming induced magnetisation instead leaves a clear residual
    induced = F.reduce_to_pole(tmi, DXM, DXM, 60.0, 10.0)
    assert np.abs(induced - pole)[CM, CM].max() > 0.05 * pole.max()


def test_rtp_at_pole_is_identity_and_low_inclination_warns():
    pole = dipole_tmi(XM, YM, 0.0, 0.0, 500.0, 90.0, 0.0)
    out = F.reduce_to_pole(pole, DXM, DXM, 90.0, 0.0)
    assert np.abs(out - pole)[CM, CM].max() < 1e-3 * pole.max()
    tmi = dipole_tmi(XM, YM, 0.0, 0.0, 500.0, 10.0, 0.0)
    with pytest.warns(UserWarning, match='low inclination'):
        out = F.reduce_to_pole(tmi, DXM, DXM, 10.0, 0.0)
    assert np.isfinite(out).all()
    with pytest.warns(UserWarning, match='low inclination'):
        F.reduce_to_pole(tmi, DXM, DXM, 60.0, 0.0, mag_inclination=5.0, mag_declination=0.0)
    with pytest.raises(ValueError):
        F.reduce_to_pole(tmi, DXM, DXM, 60.0, 0.0, epsilon=-1.0)
    bad = tmi.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        F.reduce_to_pole(bad, DXM, DXM, 60.0, 0.0)


# ─── RobustGridNormalizer: soft bound, no hard percentile clip ──────────────

def test_normalizer_keeps_sparse_indicator():
    ind = np.zeros((200, 200))
    ind[95:120, 95:120] = 1.0                    # 1.56 % ones (old code → all 0)
    z = RobustGridNormalizer().fit_transform(ind[..., None])[..., 0]
    vals = np.unique(z)
    assert len(vals) == 2
    assert vals[0] == 0.0 and 3.0 < vals[1] < 5.0
    # Known answer: IQR = 0 → std = sqrt(p(1−p)) → 5·tanh((1/std)/5)
    p = ind.mean()
    assert vals[1] == pytest.approx(5 * np.tanh(1 / np.sqrt(p * (1 - p)) / 5), rel=1e-6)


def test_normalizer_keeps_ordering_of_rare_anomalies(rng):
    bg = rng.normal(100.0, 10.0, 20000)
    anomalies = 100.0 + 10.0 * np.array([4.0, 6.0, 9.0, 13.0, 18.0, 25.0, 35.0])   # σ units
    col = np.concatenate([bg, anomalies])[:, None]
    z = RobustGridNormalizer().fit_transform(col)[-len(anomalies):, 0]
    # Old code clipped everything above the 98th percentile to a single value
    assert np.all(np.diff(z) > 0)
    assert z.max() < 5.0


def test_normalizer_soft_bound_properties(rng):
    X = rng.normal(loc=50, scale=10, size=(20000, 1))
    norm = RobustGridNormalizer().fit(X)
    _, _, median, scale = norm.params[0]
    q25, q75 = np.percentile(X, [25, 75])
    assert median == pytest.approx(np.median(X)) and scale == pytest.approx(q75 - q25)
    xs = np.linspace(-1000, 1000, 2001)[:, None]
    z = norm.transform(xs)[:, 0]
    assert z.dtype == np.float32
    assert np.all(np.abs(z) <= 5.0) and np.all(np.diff(z.astype(np.float64)) >= 0)
    lin = (xs[:, 0] - median) / scale
    small = np.abs(lin) < 0.5
    np.testing.assert_allclose(z[small], lin[small], rtol=5e-3, atol=1e-6)
    np.testing.assert_allclose(z, 5 * np.tanh(lin / 5), rtol=1e-6, atol=1e-6)
    # z_clip=None → plain robust z-score
    z_raw = RobustGridNormalizer(z_clip=None).fit(X).transform(xs)[:, 0]
    np.testing.assert_allclose(z_raw, lin, rtol=1e-6)
    with pytest.raises(ValueError):
        RobustGridNormalizer(z_clip=0)


def test_normalizer_computes_in_float64():
    # Old code cast to float32 before subtracting the median: steps of 1 mGal·1e-3
    # on a 979 812 mGal level collapsed to a handful of values.
    vals = 979812.0 + 1e-3 * np.arange(1000)
    z = RobustGridNormalizer().fit_transform(vals[:, None])[:, 0]
    assert len(np.unique(z)) == 1000
    np.testing.assert_allclose(np.median(z), 0.0, atol=2e-3)


def test_normalizer_scale_fallback_chain():
    X = np.zeros((10000, 3))
    X[:9500, 0] = 3.0
    X[9500:, 0] = 10.0                           # IQR = 0 → std
    X[:, 1] = 1e6
    X[0, 1] = 1e6 + 5e-5                         # std < 1e-12·|median| → max − min
    X[:, 2] = 7.0                                # constant → scale 1
    norm = RobustGridNormalizer().fit(X)
    assert norm.params[0][3] == pytest.approx(np.std(X[:, 0]))
    assert norm.params[1][3] == pytest.approx(5e-5, rel=1e-6)
    assert norm.params[2][3] == 1.0
    Z = norm.transform(X)
    assert np.all(Z[:, 2] == 0)
    assert Z[0, 1] == pytest.approx(5 * np.tanh(1 / 5), rel=1e-5)


def test_normalizer_clip_percentile_deprecated_but_accepted(rng):
    X = rng.normal(size=(500, 2))
    with pytest.warns(DeprecationWarning, match='clip_percentile'):
        norm = RobustGridNormalizer(clip_percentile=2.0)
    np.testing.assert_array_equal(norm.fit_transform(X), RobustGridNormalizer().fit_transform(X))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        RobustGridNormalizer()
        RobustGridNormalizer(z_clip=3.0)


def test_normalizer_preserves_nan_with_soft_bound():
    X = np.array([[np.nan], [1.0], [2.0], [3.0]] * 5)
    Z = RobustGridNormalizer().fit_transform(X)
    assert np.isnan(Z[::4]).all() and np.isfinite(Z[1::4]).all()


# ─── Layer type detection ───────────────────────────────────────────────────

@pytest.mark.parametrize('name,typ', [
    ('aem_electromagnetic', 'conductivity'), ('helicopter_electromagnetic', 'conductivity'),
    ('Electromagnetic_Conductivity', 'conductivity'), ('electromagnetic', 'conductivity'),
    ('airborneelectromagnetic', 'conductivity'), ('hem_ch3', 'conductivity'),
    ('tem_late_time', 'conductivity'), ('aem', 'conductivity'), ('em31', 'default'),
    ('em', 'conductivity'),
    ('cu_res', 'geochemistry'), ('radiometric_k_res', 'radiometric'),
    ('dem_res', 'topography'), ('dipole_dipole_res', 'resistivity'), ('res', 'resistivity'),
    ('aeromagnetic_res', 'magnetic'), ('mag_res', 'magnetic'), ('gravity_res', 'gravity'),
    ('groundwater_recharge', 'default'), ('discharge', 'default'),
    ('ip_chargeability', 'ip'), ('ipchargeability', 'ip'),
    ('aeromagnetic', 'magnetic'), ('magnetic_tmi', 'magnetic'),
    ('magnetotelluric_rho', 'resistivity'),
])
def test_detect_layer_type_collisions(name, typ):
    assert detect_layer_type(name) == typ


def test_em_layer_gets_conductivity_profile():
    rng = np.random.default_rng(0)
    layers = {'aem_electromagnetic': np.abs(rng.normal(0.05, 0.01, (40, 40)))}
    _, names, _ = extract_features(layers, dx=100, dy=100, verbose=False)
    filters = [n.split('__')[1] for n in names]
    assert 'log' in filters and 'tilt' not in filters and 'analytic_signal' not in filters


# ─── 'log' feature ──────────────────────────────────────────────────────────

def _skew(a):
    a = np.asarray(a, float).ravel()
    return float(np.mean((a - a.mean()) ** 3) / a.std() ** 3)


def test_log_feature_positive_data_is_log10_and_unit_independent():
    rng = np.random.default_rng(0)
    au_ppm = np.exp(rng.normal(np.log(0.005), 0.8, (150, 150)))   # log-normal soil Au
    maps = {}
    for unit, g in (('ppm', au_ppm), ('ppb', au_ppm * 1000)):
        _, _, m = extract_features({'geochemistry_au': g}, verbose=False)
        maps[unit] = m['geochemistry_au__log'].astype(np.float64)
    np.testing.assert_allclose(maps['ppm'], np.log10(au_ppm), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(maps['ppb'] - maps['ppm'], 3.0, atol=1e-5)
    # Old sign·log1p(|x|) left ppm data almost untransformed (skew 3.3)
    assert abs(_skew(maps['ppm'])) < 0.1
    assert abs(_skew(maps['ppb'])) < 0.1


def test_log_feature_signed_data_is_scale_invariant():
    g = np.array([[-4.0, -1.0, 0.0, 1.0], [2.0, 4.0, 8.0, 0.0]])
    _, _, m = extract_features({'x': g}, verbose=False, profiles={'x': ['log']})
    s = 3.0                                      # median(|x|) over the non-zero values
    np.testing.assert_allclose(m['x__log'], np.sign(g) * np.log1p(np.abs(g) / s), rtol=1e-6)
    _, _, m2 = extract_features({'x': g * 1e-3}, verbose=False, profiles={'x': ['log']})
    np.testing.assert_allclose(m2['x__log'], m['x__log'], rtol=1e-6)
    _, _, m0 = extract_features({'x': np.zeros((4, 4))}, verbose=False, profiles={'x': ['log']})
    assert np.all(m0['x__log'] == 0)


# ─── float32 offsets ────────────────────────────────────────────────────────

def test_level_features_keep_small_anomaly_on_large_offset():
    rng = np.random.default_rng(0)
    Yi, Xi = np.mgrid[0:120, 0:120]
    g = 979812.0 + 0.2 * np.exp(-((Xi - 60)**2 + (Yi - 60)**2) / (2 * 8**2)) + \
        rng.normal(0, 0.005, (120, 120))         # observed gravity, mGal
    _, names, m, offsets = extract_features(
        {'gravity_obs': g}, dx=10, dy=10, verbose=False, return_offsets=True)
    assert offsets == {'gravity_obs': pytest.approx(np.median(g))}
    for f in ('raw', 'smooth', 'upward', 'win_mean'):
        a = m[f'gravity_obs__{f}']
        assert a.dtype == np.float32
        assert len(np.unique(a)) > 10000         # old code: 3–4 distinct values
    restored = m['gravity_obs__raw'].astype(np.float64) + offsets['gravity_obs']
    np.testing.assert_allclose(restored, g, atol=1e-6, rtol=0)
    # Three-element return stays the default
    assert len(extract_features({'gravity_obs': g}, dx=10, dy=10, verbose=False)) == 3


# ─── extract_features: reduction to the pole ────────────────────────────────

def test_extract_features_adds_rtp_for_magnetic_layers():
    tmi = 48000.0 + dipole_tmi(XM, YM, 0.0, 0.0, 500.0, 55.0, 20.0)
    pole = dipole_tmi(XM, YM, 0.0, 0.0, 500.0, 90.0, 0.0)
    layers = {'magnetic_tmi': tmi, 'gravity_bouguer': g_sphere(500.0, XM, YM)}
    _, names0, _ = extract_features(layers, dx=DXM, dy=DXM, verbose=False)
    assert not any(n.endswith('__rtp') for n in names0)

    field = {'inclination': 55.0, 'declination': 20.0}
    stack, names, maps, offsets = extract_features(
        layers, dx=DXM, dy=DXM, verbose=False, magnetic_field=field, return_offsets=True)
    assert [n for n in names if n.endswith('__rtp')] == ['magnetic_tmi__rtp']
    assert set(names) - set(names0) == {'magnetic_tmi__rtp'}
    assert stack.shape[-1] == len(names0) + 1
    rtp = maps['magnetic_tmi__rtp'] + offsets['magnetic_tmi'] - 48000.0
    assert np.abs(rtp - pole)[CM, CM].max() < 0.01 * pole.max()


def test_extract_features_rtp_filter_validation():
    g = dipole_tmi(XM, YM, 0.0, 0.0, 500.0, 55.0, 20.0)
    with pytest.raises(ValueError, match='magnetic_field'):
        extract_features({'x': g}, verbose=False, profiles={'x': ['raw', 'rtp']})
    with pytest.raises(ValueError, match='magnetic_field'):
        extract_features({'mag': g}, verbose=False, magnetic_field={'inclination': 55.0})
    with pytest.raises(ValueError, match='magnetic_field'):
        extract_features({'mag': g}, verbose=False, magnetic_field=(55.0, 20.0))
    # Explicit profile with remanent magnetisation
    field = {'inclination': 55.0, 'declination': 20.0,
             'mag_inclination': 55.0, 'mag_declination': 20.0}
    _, names, maps = extract_features({'x': g}, dx=DXM, dy=DXM, verbose=False,
                                      profiles={'x': ['rtp']}, magnetic_field=field)
    assert names == ['x__rtp']
    expected = F.reduce_to_pole(g - np.median(g), DXM, DXM, 55.0, 20.0)
    np.testing.assert_allclose(maps['x__rtp'], expected, rtol=1e-5, atol=1e-3)
    # Explicit profiles are used verbatim (no automatic rtp)
    _, names, _ = extract_features({'mag': g}, verbose=False, profiles={'mag': ['raw']},
                                   magnetic_field=field)
    assert names == ['mag__raw']


# ─── Geographic helpers ─────────────────────────────────────────────────────

GEOG_WKT = ('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
            'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')
PROJ_WKT = ('PROJCS["WGS 84 / UTM zone 35N",' + GEOG_WKT +
            ',PROJECTION["Transverse_Mercator"],PARAMETER["central_meridian",27],UNIT["metre",1]]')


@pytest.mark.parametrize('crs,expected', [
    ('EPSG:4326', True), ('epsg:4326', True), ('+proj=longlat +datum=WGS84 +no_defs', True),
    (GEOG_WKT, True), ('urn:ogc:def:crs:EPSG::4326', True), (4326, True), ('GEODETIC', True),
    ('EPSG:32635', False), (PROJ_WKT, False), ('+proj=utm +zone=35 +datum=WGS84', False),
    ('NUTM35', False),
])
def test_looks_geographic_from_crs(crs, expected):
    lon = np.linspace(33.0, 33.2, 21)
    lat = np.linspace(39.0, 39.2, 21)
    utm_x = np.linspace(400000, 420000, 21)
    utm_y = np.linspace(4200000, 4220000, 21)
    # The CRS decides, whatever the coordinates look like
    assert looks_geographic(lon, lat, crs) is expected
    assert looks_geographic(utm_x, utm_y, crs) is expected


def test_looks_geographic_crs_object_and_heuristic():
    class FakeCRS:
        is_geographic = True
    assert looks_geographic(np.array([5e5, 6e5]), np.array([4e6, 4.1e6]), FakeCRS()) is True
    lon = np.linspace(33.0, 33.2, 21)
    lat = np.linspace(39.0, 39.2, 21)
    assert looks_geographic(lon, lat) is True
    assert looks_geographic(lon, lat, crs='unknown') is True        # unclassifiable → heuristic
    assert looks_geographic(np.linspace(4e5, 4.2e5, 5), np.linspace(4.2e6, 4.22e6, 5)) is False
    assert looks_geographic(np.linspace(-100, 100, 5), lat) is False  # span > 30
    assert looks_geographic(np.linspace(0, 10, 5), np.linspace(85, 95, 5)) is False   # |y| > 90
    assert looks_geographic(None, None) is False
    assert looks_geographic([np.nan], [np.nan]) is False


def test_metric_cell_size_known_values():
    dx_m, dy_m = metric_cell_size(1.0, 1.0, 0.0)
    assert dx_m == pytest.approx(np.pi / 180 * 6378137.0, rel=1e-12)      # 111 319.49 m
    assert dy_m == pytest.approx(111132.954 - 559.822 + 1.175, rel=1e-12)  # 110 574.31 m
    dx_m, dy_m = metric_cell_size(1.0, 1.0, 45.0)
    assert dx_m == pytest.approx(78846.8, abs=0.5)
    assert dy_m == pytest.approx(111131.78, abs=0.01)
    assert metric_cell_size(0.001, 0.002, -39.0) == pytest.approx(metric_cell_size(0.001, 0.002, 39.0))
    dxs, dys = metric_cell_size(1.0, 1.0, np.array([0.0, 60.0]))
    assert dxs.shape == (2,) and dxs[1] < dxs[0] / 1.9
    with pytest.raises(ValueError):
        metric_cell_size(1.0, 1.0, 91.0)


def test_geographic_grid_filters_are_isotropic_with_metric_cells():
    d = 800.0
    lat0 = 39.0
    lon = 33.0 + np.arange(201) * 0.001
    lat = lat0 + np.arange(201) * 0.001
    dx_m, dy_m = metric_cell_size(0.001, 0.001, lat0 + 0.1)
    Xm = (np.arange(201) - 100) * dx_m
    Ym = (np.arange(201) - 100) * dy_m
    XX, YY = np.meshgrid(Xm, Ym)
    g = G * MASS * d / (XX**2 + YY**2 + d**2) ** 1.5 * 1e5
    lyr = _layer(lon, lat, g, crs='EPSG:4326')
    cr = _quiet(CoRegistrar().fit, {'gravity': lyr})
    reg = _quiet(cr.transform_all, {'gravity': lyr})
    fdx, fdy = cr.filter_cell_size(crs='EPSG:4326')
    assert fdx == pytest.approx(dx_m, rel=1e-6) and fdy == pytest.approx(dy_m, rel=1e-6)
    assert cr.filter_cell_size(geographic=False) == (cr.dx, cr.dy)
    t = F.tilt_angle(reg['gravity'], fdx, fdy)
    row, col = t[100, 100:], t[100:, 100]
    zx = Xm[100 + np.argmin(np.abs(row))]
    zy = Ym[100 + np.argmin(np.abs(col))]
    r0 = np.sqrt(2) * d                      # tilt zero contour of a point mass
    assert abs(zx - r0) < 0.05 * r0 and abs(zy - r0) < 0.05 * r0   # degrees: 1384 / 995 m


# ─── CoRegistrar.coverage_mask ──────────────────────────────────────────────

def _fit_all(layers, **kw):
    cr = _quiet(CoRegistrar().fit, layers, **kw)
    _quiet(cr.transform_all, layers)
    return cr


def test_coverage_mask_dense_layer_with_hole():
    n = 100
    xs = np.arange(n) * 10.0
    g = np.ones((n, n))
    g[40:60, 40:60] = np.nan                   # 20-cell hole
    cr = _fit_all({'dense': _layer(xs, xs, g)})
    cov = cr.coverage_mask(max_gap_cells=3.0)
    assert cov.shape == (n, n) and cov.dtype == bool
    assert not cov[44:56, 44:56].any()         # hole interior (> 3 cells from data)
    assert cov[40:43, 40:60].all()             # within 3 cells of the hole edge
    outside_hole = np.ones((n, n), bool)
    outside_hole[40:60, 40:60] = False
    assert cov[outside_hole].all()
    assert cr.coverage_tolerance == {'dense': 3.0}
    assert cr.coverage_mask(max_gap_cells=10.0).all()   # a 20-cell hole is ≤ 10 cells from data


def test_coverage_mask_sparse_point_layer_is_covered_between_samples():
    n = 100
    xs = np.arange(n) * 10.0
    sparse = np.full((n, n), np.nan)
    sparse[::6, ::6] = 1.0                     # point samples every 6 cells
    cr = _fit_all({'sparse': _layer(xs, xs, sparse), 'dense': _layer(xs, xs, np.ones((n, n)))})
    cov = cr.coverage_mask()
    assert cov.mean() > 0.98
    assert cr.coverage_tolerance['sparse'] == pytest.approx(9.0)   # 1.5 × 6 cells
    # A large empty region inside a sparse layer is not covered
    sparse2 = sparse.copy()
    sparse2[:, 50:] = np.nan
    sparse2[::6, 96] = 1.0                     # keep the extent
    cr2 = _fit_all({'sparse': _layer(xs, xs, sparse2)})
    cov2 = cr2.coverage_mask()
    assert not cov2[:, 65:80].any() and cov2[:, :50].mean() > 0.98


def test_coverage_mask_coarse_samples_upsampled_to_blocks():
    n = 100
    xs = np.arange(n) * 10.0
    xc = np.arange(0, n * 10, 40.0)
    coarse = np.full((len(xc), len(xc)), np.nan)
    coarse[::2, ::2] = 1.0                     # samples every 80 m → 8 reference cells
    cr = _fit_all({'fine': _layer(xs, xs, np.ones((n, n))), 'coarse': _layer(xc, xc, coarse)})
    cov = cr.coverage_mask()
    assert cr.coverage_tolerance['coarse'] >= 1.5 * 6
    assert cov[:96, :96].mean() > 0.98


def test_coverage_mask_union_extent_strip_and_errors():
    a = _layer(np.linspace(0, 100, 11), np.linspace(0, 100, 11), np.ones((11, 11)))
    b = _layer(np.linspace(50, 150, 11), np.linspace(0, 100, 11), np.ones((11, 11)))
    cr = _quiet(CoRegistrar().fit, {'a': a, 'b': b}, target_nx=31, target_ny=11,
                common_extent='union')
    with pytest.raises(RuntimeError):
        cr.coverage_mask()
    _quiet(cr.transform_all, {'a': a, 'b': b})
    cov = cr.coverage_mask(max_gap_cells=3.0)
    assert not cov[:, cr.ref_x < 50].any()     # only 'b' is missing there …
    assert not cov[:, cr.ref_x > 100].any()    # … and only 'a' here
    assert cov[:, (cr.ref_x >= 50) & (cr.ref_x <= 100)].all()
    assert cr.outside_masks['a'][:, cr.ref_x > 100].all()
    assert not cr.outside_masks['a'][:, cr.ref_x <= 100].any()
    # Single-layer selection: 'a' alone covers x ≤ 100 only
    cov_a = cr.coverage_mask(layers=['a'])
    np.testing.assert_array_equal(cov_a, np.broadcast_to(cr.ref_x <= 100, cov_a.shape))
    with pytest.raises(KeyError):
        cr.coverage_mask(layers=['nope'])
    with pytest.raises(ValueError):
        cr.coverage_mask(max_gap_cells=-1)
    with pytest.raises(RuntimeError):
        CoRegistrar().coverage_mask()
